/**
 * @file
 *
 * Efficiently write a stream of data that is later constituted as an
 * @c stxxl::vector.
 */

#include <stxxl.h>
#include <vector>
#include <iterator>
#include <cstddef>
#include <string>
#include <boost/noncopyable.hpp>
#include <boost/iterator/iterator_categories.hpp>
#include <iterator>
#include "allocator.h"
#include "statistics.h"
#include "tr1_cstdint.h"

/**
 * Efficiently write a stream of data that is later constituted as an
 * @c stxxl::vector. Individual elements can be inserted using
 * @ref push_back, while ranges of known length can be more efficiently
 * inserted using @ref append. When all items have been appended, use
 * @ref move to create the vector.
 */
template<typename T, unsigned blockSize = STXXL_DEFAULT_BLOCK_SIZE(T)>
class VectorInserter : public boost::noncopyable
{
public:
    typedef T value_type;               ///< Type of elements stored in the vector
    typedef T &reference;               ///< Reference to element type
    typedef const T &const_reference;   ///< Const reference to element type

    /**
     * Return the current number of elements.
     */
    std::size_t size() const { return nElements; }

    /**
     * Append a single item to the stream.
     */
    void push_back(const T &value);

    /**
     * Append a range of items to the stream. This is optimized for the
     * case where the iterator type is a random access traversal iterator.
     *
     * @param first   First item to append.
     * @param last    One past the last item to append.
     */
    template<typename Iterator>
    void append(Iterator first, Iterator last);

    /**
     * Flushes any remaining data and transfers the data to the vector.
     *
     * @pre @a v is empty (see @c stxxl::vector::set_content).
     * @post @c this is empty.
     */
    template<typename V>
    void move(V &v);

    /**
     * Constructor. The internal memory requirement is @a bufferSize times the
     * block size, plus O(@a bufferSize) overheads.
     *
     * @param name           Name used for statistic tracking memory use
     * @param bufferSize     Number of buffers to hold in memory for overlapped I/O
     * @param batchSize      Number of buffers to write to disk at a time (see @c stxxl::buffered_writer)
     */
    explicit VectorInserter(
        const std::string &name,
        std::size_t bufferSize = 4, std::size_t batchSize = 1);

    ~VectorInserter();

private:
    typedef stxxl::typed_block<blockSize, T> block_type;
    typedef typename block_type::bid_type bid_type;
    typedef Statistics::Allocator<std::allocator<bid_type> > alloc_type;

    /// Number of blocks held in the writer
    const std::size_t bufferSize;
    /// Block manager singleton instance
    stxxl::block_manager * const blockManager;

    /// Allocator used to account for memory allocated to @ref writer
    alloc_type alloc;
    /**
     * Backing blocks that contain the data. Blocks are allocated and added to
     * this vector when they are passed to @ref writer for writing, but before
     * they are flushed to disk.
     */
    std::vector<bid_type, alloc_type> blocks;
    /// Manages the buffer and asynchronous writes
    stxxl::buffered_writer<block_type> writer;

    /// Current block returned from @ref writer (always non-NULL)
    block_type *curBlock;
    /**
     * Current offset within @ref curBlock for the next item. This is always
     * less than the block-size: when a block is full, @ref writeCurBlock must
     * be called.
     */
    std::size_t curOffset;
    /// Total number of elements that have been appended
    std::tr1::uint64_t nElements;

    /**
     * Send the current block to be written. This allocates a new block from
     * the block manager. It is only safe to call this when either the current
     * block is full, or when no more items will be added. Otherwise it will
     * leave a partially filled block.
     *
     * @post @ref curOffset == 0 and @ref curBlock points at a new buffer block.
     */
    void writeCurBlock();

    /// Generic implementation of @ref append
    template<typename Iterator, typename TraversalTag>
    void appendImpl(Iterator first, Iterator last, TraversalTag);

    /// Specialized implementation of @ref append for random access traversal iterators
    template<typename Iterator>
    void appendImpl(Iterator first, Iterator last, boost::random_access_traversal_tag);
};

template<typename T, unsigned blockSize>
VectorInserter<T, blockSize>::VectorInserter(
    const std::string &name, std::size_t bufferSize, std::size_t batchSize)
    :
    bufferSize(bufferSize),
    blockManager(stxxl::block_manager::get_instance()),
    alloc(Statistics::makeAllocator<alloc_type>(name)),
    blocks(alloc),
    writer(bufferSize, batchSize),
    curBlock(writer.get_free_block()),
    curOffset(0),
    nElements(0)
{
    // Record the memory used by the writer
    alloc.recordAllocate(bufferSize * sizeof(block_type));
}

template<typename T, unsigned blockSize>
VectorInserter<T, blockSize>::~VectorInserter()
{
    writer.flush(); // ensures the blocks aren't freed while being written
    blockManager->delete_blocks(blocks.begin(), blocks.end());
    alloc.recordDeallocate(bufferSize * sizeof(block_type));
}

template<typename T, unsigned blockSize>
void VectorInserter<T, blockSize>::writeCurBlock()
{
    bid_type bid;
    blockManager->new_block(stxxl::striping(), bid, blocks.size());
    curBlock = writer.write(curBlock, bid);
    curOffset = 0;
    blocks.push_back(bid);
}

template<typename T, unsigned blockSize>
void VectorInserter<T, blockSize>::push_back(const T &value)
{
    curBlock->elem[curOffset] = value;
    curOffset++;
    nElements++;
    if (curOffset == block_type::size)
        writeCurBlock();
}

template<typename T, unsigned blockSize>
template<typename Iterator>
void VectorInserter<T, blockSize>::append(Iterator first, Iterator last)
{
    appendImpl(first, last, typename boost::iterator_traversal<Iterator>::type());
}

template<typename T, unsigned blockSize>
template<typename Iterator, typename TraversalTag>
void VectorInserter<T, blockSize>::appendImpl(Iterator first, Iterator last, TraversalTag)
{
    for (Iterator i = first; i != last; ++i)
        push_back(*i);
}

template<typename T, unsigned blockSize>
template<typename Iterator>
void VectorInserter<T, blockSize>::appendImpl(Iterator first, Iterator last, boost::random_access_traversal_tag)
{
    while (first != last)
    {
        std::size_t len = last - first;
        if (block_type::size - curOffset < len)
            len = block_type::size - curOffset;
        // len is the number of elements to transfer to this block
        std::copy(first, first + len, curBlock->elem + curOffset);
        curOffset += len;
        nElements += len;
        first += len;
        if (curOffset == block_type::size)
            writeCurBlock();
    }
}

template<typename T, unsigned blockSize>
template<typename V>
void VectorInserter<T, blockSize>::move(V &v)
{
    if (curOffset > 0)
        writeCurBlock();  // write last partial block
    writer.flush();
    v.set_content(blocks.begin(), blocks.end(), nElements);
    blocks.clear();
    nElements = 0;
}
