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
#include <boost/noncopyable.hpp>
#include "allocator.h"
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
    typedef T value_type;
    typedef T &reference;
    typedef const T &const_reference;

    /**
     * Append a single item to the stream.
     */
    void push_back(const T &value);

    /**
     * Append a range of items to the stream. This is optimized for the
     * case where the iterator type is a random access iterator.
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

    explicit VectorInserter(std::size_t bufferSize = 4, std::size_t batchSize = 1);
    ~VectorInserter();

private:
    typedef stxxl::typed_block<blockSize, T> block_type;
    typedef typename block_type::bid_type bid_type;

    stxxl::block_manager *blockManager;

    Statistics::Container::vector<bid_type> blocks;
    stxxl::buffered_writer<block_type> writer;

    block_type *curBlock;
    std::size_t curOffset;
    std::size_t nElements;

    void writeCurBlock();
};

template<typename T, unsigned blockSize>
VectorInserter<T, blockSize>::VectorInserter(std::size_t bufferSize, std::size_t batchSize)
    : blockManager(stxxl::block_manager::get_instance()),
    blocks("mem.VectorInserter.blocks"),
    writer(bufferSize, batchSize),
    curBlock(writer.get_free_block()),
    curOffset(0),
    nElements(0)
{
    // TODO: account for memory used by writer
}

template<typename T, unsigned blockSize>
VectorInserter<T, blockSize>::~VectorInserter()
{
    writer.flush(); // ensures 
    blockManager->delete_blocks(blocks.begin(), blocks.end());
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
    // TODO: specialize for random access iterator
    for (Iterator i = first; i != last; ++i)
        append(*i);
}

template<typename T, unsigned blockSize>
template<typename V>
void VectorInserter<T, blockSize>::move(V &v)
{
    if (curOffset > 0)
        writeCurBlock();
    writer.flush();
    v.set_content(blocks.begin(), blocks.end(), nElements);
    blocks.clear();
}
