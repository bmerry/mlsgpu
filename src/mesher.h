/**
 * @file
 *
 * Data structures for storing the output of @ref Marching.
 *
 * The classes in this file are @ref MesherBase, an abstract base class, and
 * several concrete instantiations of it. They differ in terms of
 *  - the number of passes needed
 *  - the amount of temporary memory required.
 */

#ifndef MESHER_H
#define MESHER_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <CL/cl.h>

#include <string>
#include <vector>
#include <map>
#include <string>
#include <iosfwd>
#include <utility>
#include <boost/array.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/optional.hpp>
#include <boost/foreach.hpp>
#include "tr1_unordered_map.h"
#include "tr1_unordered_set.h"
#include <stxxl.h>
#include "marching.h"
#include "fast_ply.h"
#include "union_find.h"
#include "work_queue.h"
#include "statistics.h"
#include "allocator.h"

/**
 * Enumeration of the supported mesher types
 */
enum MesherType
{
    BIG_MESHER,
    STXXL_MESHER
};

/**
 * Unique ID for an output file chunk. It consists of a @em generation number,
 * which is increased monotonically, and a set of @em coordinates which are used
 * to name the file.
 *
 * Comparison of generation numbers does not necessarily correspond to
 * lexicographical ordering of coordinates, but there is a one-to-one
 * relationship that is preserved across passes.
 */
struct ChunkId
{
    typedef std::tr1::uint32_t gen_type;

    /// Monotonically increasing generation number
    gen_type gen;
    /**
     * Chunk coordinates. The chunks form a regular grid and the coordinates
     * give the position within the grid, starting from (0,0,0).
     */
    boost::array<Grid::size_type, 3> coords;

    /// Default constructor (does zero initialization)
    ChunkId() : gen(0)
    {
        for (unsigned int i = 0; i < 3; i++)
            coords[i] = 0;
    }

    /// Comparison by generation number
    bool operator<(const ChunkId &b) const
    {
        return gen < b.gen;
    }
};

/**
 * Wrapper around @ref MesherType for use with @ref Choice.
 */
class MesherTypeWrapper
{
public:
    typedef MesherType type;
    static std::map<std::string, MesherType> getNameMap();
};

/**
 * Data about a mesh passed in to a @ref MesherBase::InputFunctor. It contains
 * host mesh data that may still be being read asynchronously from a device,
 * together with the events that will signal data readiness.
 */
struct MesherWork
{
    HostKeyMesh mesh;              ///< Mesh data (may be empty)
    cl::Event verticesEvent;       ///< Signaled when vertices may be read
    cl::Event vertexKeysEvent;     ///< Signaled when vertex keys may be read
    cl::Event trianglesEvent;      ///< Signaled when triangles may be read
};

/**
 * Model of @ref MesherBase::Namer that always returns a fixed filename.
 */
class TrivialNamer
{
private:
    std::string name;

public:
    typedef std::string result_type;
    const std::string &operator()(const ChunkId &chunkId) const
    {
        (void) chunkId;
        return name;
    }

    TrivialNamer(const std::string &name) : name(name) {}
};

/**
 * Model of @ref MesherBase::Namer that adds the chunk ID into the name.
 *
 * The generated name is
 * <i>base</i><code>_</code><i>XXXX</i><code>_</code><i>YYYY</i><code>_</code><i>ZZZZ</i><code>.ply</code>,
 * where @a base is the base name given to the constructor and @a XXXX, @a YYYY
 * and @a ZZZZ are the coordinates.
 */
class ChunkNamer
{
private:
    std::string baseName;

public:
    typedef std::string result_type;
    std::string operator()(const ChunkId &chunkId) const;

    ChunkNamer(const std::string &baseName) : baseName(baseName) {}
};

/**
 * Abstract base class for output collectors for @ref Marching. This class
 * only captures the host side of the process. It needs to be wrapped in
 * using @ref deviceMesher or @ref MesherGroup to satisfy
 * the requirements for @ref Marching.
 *
 * The basic procedure for using one of these classes is:
 * -# Instantiate it.
 * -# Call @ref setPruneThreshold.
 * -# Call @ref numPasses to determine how many passes are required.
 * -# For each pass, call @ref functor to obtain a functor, then
 *    make as many calls to @ref Marching::generate as desired using this
 *    functor. Each call should set @a keyOffset so that vertex keys line up.
 *    Each pass must generate exactly the same geometry, but the blocks may
 *    be generated in different order within each chunk (chunks must be in
 *    order).
 * -# Call @ref write.
 *
 * @warning The functor is @em not required to be thread-safe. The caller must
 * serialize calls if necessary (@ref MesherGroup only uses one thread).
 */
class MesherBase
{
public:
    /**
     * Type returned by @ref functor. The argument is a mesh to be processed.
     * After the function returns the mesh is not used again, so it may be
     * modified as past of the implementation.
     */
    typedef boost::function<void(const ChunkId &chunkId, MesherWork &work)> InputFunctor;

    /**
     * Function object that generates a filename from a chunk ID.
     */
    typedef boost::function<std::string(const ChunkId &chunkId)> Namer;

    /**
     * Constructor. The mesher object retains a reference to @a writer and so it
     * must persist until the mesher is destroyed. The @a namer is copied and so
     * may be transient.
     *
     * The @a writer must not be open when the constructor is called, nor
     * should it be directly accessed when the mesher exists. The mesher will
     * open and close the writer once per output file.
     *
     * @param writer         Writer that will be used to emit output files.
     * @param namer          Callback function to assign names to output files.
     */
    MesherBase(FastPly::WriterBase &writer, const Namer &namer)
        : pruneThreshold(0.0), writer(writer), namer(namer) {}

    /// Virtual destructor to allow destruction via base class pointer
    virtual ~MesherBase() {}

    /// Number of passes required.
    virtual unsigned int numPasses() const = 0;

    /**
     * Sets the lower bound on component size. All components that are
     * smaller will be pruned from the output, if supported by the mesher
     * type. The default is not to prune anything.
     *
     * @param threshold The lower bound, specified as a fraction of the total
     * number of pre-pruning vertices.
     */
    void setPruneThreshold(double threshold) { pruneThreshold = threshold; }

    /// Retrieve the value set with @ref setPruneThreshold.
    double getPruneThreshold() const { return pruneThreshold; }

    /**
     * Retrieves a functor that will accept data in a specific pass.
     * Multi-pass classes may do finalization on a previous pass before
     * returning the functor, so this function should only be called for
     * pass @a pass once pass @a pass - 1 has completed. It must also
     * only be called once per pass.
     *
     * The functor might perform file I/O (depending on the subclass), in which
     * case it may throw any of the exceptions documented for @ref write.
     *
     * @pre @a pass is less than @ref numPasses().
     *
     * @warning The returned functor is @em not required to be thread-safe.
     */
    virtual InputFunctor functor(unsigned int pass) = 0;

    /**
     * Performs any final file I/O.
     *
     * @param progressStream  If non-NULL, a log stream for a progress meter.
     * @throw std::ios_base::failure on I/O failure (including failure to open the file).
     * @throw std::overflow_error if too many connected components were found.
     * @throw std::overflow_error if too many vertices were found in one output chunk.
     */
    virtual void write(std::ostream *progressStream = NULL) = 0;

protected:
    FastPly::WriterBase &getWriter() const { return writer; }
    std::string getOutputName(const ChunkId &id) const { return namer(id); }

private:
    /// Threshold set by @ref setPruneThreshold
    double pruneThreshold;

    FastPly::WriterBase &writer;   ///< Writer for output files
    const Namer namer;             ///< Output file namer
};

namespace detail
{

/**
 * An internal base class for @ref BigMesher and @ref StxxlMesher, implementing
 * algorithms common to both.
 *
 * Component identification is implemented with a two-level approach. Within each
 * block, a union-find is performed to identify local components. These
 * components are referred to as @em clumps. Each vertex is given a <em>clump
 * id</em>. During welding, external vertices are used to identify clumps that
 * form part of the same component, and this is recorded in a union-find
 * structure over the clumps. Clumps are represented in both the per-chunk data
 * and globally, but "clump IDs" refer to the global representation, over which
 * the union-find tree is built.
 *
 * Vertices in a block are reordered by clump, and within a clump the vertices are
 * first the internal ones, then the external ones. External vertices that already
 * appeared in a previous clump in the same chunk are elided.
 *
 * Triangles are also ordered by clump, and use clump-local indices. Where a vertex
 * has been elided, it is indexed by an enumeration over the external vertices
 * of the chunk, and this alternative encoding is signalled by flipping all
 * bits. This encoding is unambiguous provided that the total external vertices
 * in a chunk plus the total vertices in a clump do not exceed 2^32 (at which point
 * PLY would be useless anyway).
 *
 * External vertices are entered into a hash table that maps their keys to
 * their (global) chunk ID, and a chunk-local hash table that maps it to the
 * triangle index used to encode it.
 */
class KeyMapMesher : public MesherBase
{
protected:
    typedef std::tr1::int32_t clump_id;

    /**
     * Component within a single block. The root clump also tracks the number of
     * vertices and triangles in a component.
     */
    class Clump : public UnionFind::Node<clump_id>
    {
    public:
        /**
         * A tuple of vertices and triangles.
         */
        struct Counts
        {
            std::tr1::uint64_t vertices;
            std::tr1::uint64_t triangles;

            Counts &operator+=(const Counts &b)
            {
                vertices += b.vertices;
                triangles += b.triangles;
                return *this;
            }

            Counts() : vertices(0), triangles(0) {}
        };

        /**
         * @name
         * @{
         * Book-keeping counts of vertices and triangles. These counts are only valid
         * on root nodes.
         */
        Counts counts;                                                   ///< Global counts
        Statistics::Container::unordered_map<ChunkId::gen_type, Counts> chunkCounts;  ///< Per-chunk counts
        /** @} */

        void merge(Clump &b)
        {
            UnionFind::Node<cl_int>::merge(b);
            counts += b.counts;

            // Merge chunkCounts with b.chunkCounts
            typedef std::pair<unsigned int, Counts> item_type;
            BOOST_FOREACH(const item_type &item, b.chunkCounts)
            {
                chunkCounts[item.first] += item.second;
            }
            // No longer need this, since b is no longer a root node
            b.chunkCounts.clear();
        }

        /**
         * Constructor for a new clump in a chunk.
         * @param chunkGen           The chunk containing the clump.
         * @param numVertices        The number of vertices in the clump.
         *
         * @post
         * - <code>counts.vertices == numVertices</code>
         * - <code>counts.triangles == 0</code>
         * - <code>counts.chunkCounts[chunkGen].vertices == numVertices</code>
         * - <code>counts.chunkCounts[chunkGen].triangles == 0</code>
         * - <code>counts.chunkCounts.size() == 1</code>
         */
        Clump(ChunkId::gen_type chunkGen, std::tr1::uint64_t numVertices)
            : chunkCounts("mem.Clump::chunkCounts")
        {
            counts.vertices = numVertices;
            chunkCounts[chunkGen].vertices = numVertices;
        }
    };

    typedef Statistics::Container::unordered_map<cl_ulong, std::tr1::uint32_t> vertex_id_map_type;
    typedef Statistics::Container::unordered_map<cl_ulong, clump_id> clump_id_map_type;

    /// Maps external vertex keys to external indices for the current chunk
    vertex_id_map_type vertexIdMap;

    /// Maps external vertex keys to clumps
    clump_id_map_type clumpIdMap;

    /// All clumps, in a structure usable with @ref UnionFind.
    Statistics::Container::vector<Clump> clumps;

    /**
     * @name
     * @{
     * Temporary buffers.
     * These are stored in the object so that memory can be recycled if
     * possible, rather than thrashing the allocator.
     */
    Statistics::Container::vector<std::tr1::uint32_t> tmpIndexTable;
    Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > tmpNodes;
    Statistics::Container::vector<clump_id> tmpClumpId;
    /** @} */

    /**
     * Identifies components with a local set of triangles, and
     * returns a union-find tree for them.
     *
     * @param numVertices    Number of vertices indexed by @a triangles.
     *                       Also the size of the returned union-find tree.
     * @param triangles      The vertex indices of the triangles.
     * @param[out] nodes     A union-find tree over the vertices.
     */
    static void computeLocalComponents(
        std::size_t numVertices,
        const Statistics::Container::vector<boost::array<cl_uint, 3> > &triangles,
        Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes);

    /**
     * Creates clumps from local components.
     *
     * @param chunkGen       Chunk generation to which the local triangles belong.
     * @param nodes          A union-find tree over the vertices (see @ref computeLocalComponents).
     * @param triangles      The triangles used to determine connectivity.
     * @param[out] clumpId   The index into @ref clumps for each vertex.
     *
     * @post <code>clumpId.size() == nodes.size()</code>
     */
    void updateClumps(
        unsigned int chunkGen,
        const Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes,
        const Statistics::Container::vector<boost::array<cl_uint, 3> > &triangles,
        Statistics::Container::vector<clump_id> &clumpId);

    /**
     * Add external vertex keys to the key maps and computes an index rewrite table.
     * The index rewrite table maps local external indices for a block to their
     * almost final values (prior to component removal). It also does merging
     * of clumps into components.
     *
     * @param chunkGen        Chunk generation to which the local vertices belong.
     * @param vertexOffset    The final index for the first external vertex in the block.
     * @param keys            Keys of the external vertices.
     * @param clumpId         The clump IDs of all vertices, as computed by @ref computeLocalComponents.
     * @param[out] indexTable The index remapping table.
     *
     * @note Only the last <code>keys.size()</code> elements of @a clumpId are relevant.
     */
    void updateKeyMaps(
        ChunkId::gen_type chunkGen,
        std::tr1::uint32_t vertexOffset,
        const Statistics::Container::vector<cl_ulong> &keys,
        const Statistics::Container::vector<clump_id> &clumpId,
        Statistics::Container::vector<std::tr1::uint32_t> &indexTable);

    /**
     * Writes indices in place from being block-relative to the intermediate form (prior to
     * component removal).
     * @param priorVertices        First vertex in the block (internal or external).
     * @param indexTable           External index rewrite table computed by @ref updateKeyMaps.
     * @param[in,out] mesh         Triangles to rewrite (also uses the number of vertices).
     * @todo Only used in @ref StxxlMesher?
     */
    void rewriteTriangles(
        std::tr1::uint32_t priorVertices,
        const Statistics::Container::vector<std::tr1::uint32_t> &indexTable,
        HostKeyMesh &mesh) const;

    /**
     * @copydoc MesherBase::MesherBase
     */
    KeyMapMesher(FastPly::WriterBase &writer, const Namer &namer) :
        MesherBase(writer, namer),
        vertexIdMap("mem.KeyMapMesher::vertexIdMap"),
        clumpIdMap("mem.KeyMapMesher::clumpIdMap"),
        clumps("mem.KeyMapMesher::clumps"),
        tmpIndexTable("mem.KeyMapMesher::tmpIndexTable"),
        tmpNodes("mem.KeyMapMesher::tmpNodes"),
        tmpClumpId("mem.KeyMapMesher::tmpClumpId")
    {
    }
};

} // namespace detail

/**
 * Two-pass collector that can handle very large meshes by writing
 * the geometry to file as it is produced. It requires an out-of-order
 * writer, and requires the writer to be provided up front.
 *
 * The two passes are:
 * 1. Counting, and assigning key mappings to detect duplicate external
 *    vertices (both at the global level for component detection and at the
 *    chunk level for welding).
 * 2. Write the data.
 *
 * The external vertices are written out as they come in (immediately after the
 * internal vertices for the corresponding chunk), which avoids the need to
 * buffer them up until the end. The only unbounded memory is for the various
 * key maps.
 */
class BigMesher : public detail::KeyMapMesher
{
private:
    /**
     * Maps chunk generations to full chunk IDs. This is needed to generate
     * error messages during @ref prepareAdd.
     */
    Statistics::Container::unordered_map<ChunkId::gen_type, ChunkId> chunkIds;

    /**
     * @name
     * @{
     * Data used only during the second pass. These fields are initialized
     * by @ref prepareAdd and are undefined prior to that.
     */

    std::tr1::uint32_t nextVertex;   ///< Number of vertices written so far
    std::tr1::uint64_t nextTriangle; ///< Number of triangles written so far

    /// Minimum number of vertices to keep a component
    std::tr1::uint64_t pruneThresholdVertices;

    /// Keys of external vertices in retained chunks
    Statistics::Container::unordered_set<cl_ulong> retainedExternal;

    /// Number of vertices and triangles that will be produced for each chunk
    Statistics::Container::unordered_map<ChunkId::gen_type, Clump::Counts> chunkCounts;

    /**
     * @}
     */

    /// Temporary storage for local clump validity
    Statistics::Container::vector<bool> tmpClumpValid;

    /// Implementation of the first-pass functor
    void count(const ChunkId &chunkId, MesherWork &work);

    /// Chunk generation which is currently being written (empty if writer is not open)
    boost::optional<ChunkId::gen_type> curChunkGen;

    /// Preparation for the second pass after the first pass
    void prepareAdd();

    /// Implementation of the second-pass functor
    void add(const ChunkId &chunkId, MesherWork &work);

public:
    virtual unsigned int numPasses() const { return 2; }

    /**
     * @copydoc MesherBase::MesherBase
     */
    BigMesher(FastPly::WriterBase &writer, const Namer &namer);

    virtual InputFunctor functor(unsigned int pass);

    virtual void write(std::ostream *progressStream = NULL);
};

/**
 * Mesher class that uses the same algorithm as @ref BigMesher, but stores
 * the data in STXXL containers before concatenating them rather than
 * using multiple passes. It thus trades storage requirements against
 * performance, at least when @ref BigMesher is compute-bound.
 */
class StxxlMesher : public MesherBase
{
private:
    typedef std::tr1::int32_t clump_id;

    /// Type for storing vertex data out-of-core
    typedef stxxl::VECTOR_GENERATOR<boost::array<float, 3> >::result vertices_type;
    /// Type for storing intermediate triangle data out-of-core
    typedef stxxl::VECTOR_GENERATOR<boost::array<std::tr1::uint32_t, 3> >::result triangles_type;

    /**
     * Strict weak ordering for sorting a list of vertex indices based on their
     * chunk IDs. It is stable i.e. ties are broken by the IDs themselves.
     */
    class VertexCompare
    {
    private:
        const Statistics::Container::vector<clump_id> &clumpId;

    public:
        explicit VertexCompare(const Statistics::Container::vector<clump_id> &clumpId)
            : clumpId(clumpId) {}

        bool operator()(std::tr1::uint32_t a, std::tr1::uint32_t b) const
        {
            assert(a < clumpId.size());
            assert(b < clumpId.size());
            if (clumpId[a] != clumpId[b])
                return clumpId[a] < clumpId[b];
            else
                return a < b;
        }
    };

    /**
     * Strict weak ordering for sorting triangles by clump, determined
     * from a clumpId array indexed by the triangle indices.
     */
    class TriangleCompare
    {
    private:
        const Statistics::Container::vector<clump_id> &clumpId;

    public:
        explicit TriangleCompare(const Statistics::Container::vector<clump_id> &clumpId)
            : clumpId(clumpId) {}

        bool operator()(const boost::array<std::tr1::uint32_t, 3> &a,
                        const boost::array<std::tr1::uint32_t, 3> &b) const
        {
            assert(a[0] < clumpId.size());
            assert(b[0] < clumpId.size());
            return clumpId[a[0]] < clumpId[b[0]];
        }
    };

    /**
     * Data for a single chunk.
     */
    class Chunk
    {
    public:
        // Chunk-local clump data
        struct Clump
        {
            vertices_type::size_type firstVertex;
            std::tr1::uint32_t numInternalVertices;
            std::tr1::uint32_t numExternalVertices;
            triangles_type::size_type firstTriangle;
            std::tr1::uint32_t numTriangles;
            clump_id globalId;

            Clump(
                vertices_type::size_type firstVertex,
                std::tr1::uint32_t numInternalVertices,
                std::tr1::uint32_t numExternalVertices,
                triangles_type::size_type firstTriangle,
                std::tr1::uint32_t numTriangles,
                clump_id globalId)
                : firstVertex(firstVertex),
                numInternalVertices(numInternalVertices),
                numExternalVertices(numExternalVertices),
                firstTriangle(firstTriangle),
                numTriangles(numTriangles),
                globalId(globalId)
            {
            }
        };

        typedef Statistics::Container::unordered_map<cl_ulong, std::tr1::uint32_t> vertex_id_map_type;

        ChunkId chunkId;
        Statistics::Container::vector<Clump> clumps;
        vertex_id_map_type vertexIdMap;
        std::size_t numExternalVertices;

        explicit Chunk(const ChunkId chunkId = ChunkId())
            : chunkId(chunkId), clumps("mem.mesher.chunk.clumps"), vertexIdMap("mem.mesher.vertexIdMap"),
            numExternalVertices(0) {}
    };

    /**
     * Component within a single block. The root clump also tracks the number of
     * vertices and triangles in a component.
     */
    class Clump : public UnionFind::Node<clump_id>
    {
    public:
        /**
         * A tuple of vertices and triangles.
         */
        struct Counts
        {
            std::tr1::uint64_t vertices;
            std::tr1::uint64_t triangles;

            Counts &operator+=(const Counts &b)
            {
                vertices += b.vertices;
                triangles += b.triangles;
                return *this;
            }

            Counts() : vertices(0), triangles(0) {}
        };

        /**
         * @name
         * @{
         * Book-keeping counts of vertices and triangles. These counts are only valid
         * on root nodes.
         */
        Counts counts;  ///< Global counts
        /** @} */

        void merge(Clump &b)
        {
            UnionFind::Node<cl_int>::merge(b);
            counts += b.counts;
        }

        /**
         * Constructor for a new clump.
         * @param numVertices        The number of vertices in the clump.
         *
         * @post
         * - <code>counts.vertices == numVertices</code>
         * - <code>counts.triangles == 0</code>
         */
        Clump(std::tr1::uint64_t numVertices)
        {
            counts.vertices = numVertices;
        }
    };

    /**
     * @name
     * @{
     * Temporary buffers.
     * These are stored in the object so that memory can be recycled if
     * possible, rather than thrashing the allocator.
     */
    Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > tmpNodes;
    Statistics::Container::vector<clump_id> tmpClumpId;
    /** @} */

    vertices_type vertices;     ///< Buffer of all vertices seen so far
    triangles_type triangles;   ///< Buffer of all triangles seen so far
    Statistics::Container::unordered_map<ChunkId::gen_type, Chunk> chunks;  ///< All chunks seen so far
    Statistics::Container::vector<Clump> clumps; ///< All clumps seen so far

    typedef Statistics::Container::unordered_map<cl_ulong, clump_id> clump_id_map_type;
    /// Maps external vertex keys to clumps
    clump_id_map_type clumpIdMap;

    /**
     * Identifies components with a local set of triangles, and
     * returns a union-find tree for them.
     *
     * @param numVertices    Number of vertices indexed by @a triangles.
     *                       Also the size of the returned union-find tree.
     * @param triangles      The vertex indices of the triangles.
     * @param[out] nodes     A union-find tree over the vertices.
     */
    static void computeLocalComponents(
        std::size_t numVertices,
        const Statistics::Container::vector<boost::array<cl_uint, 3> > &triangles,
        Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes);

    void updateGlobalClumps(
        const Statistics::Container::vector<UnionFind::Node<std::tr1::int32_t> > &nodes,
        const Statistics::Container::vector<boost::array<cl_uint, 3> > &triangles,
        Statistics::Container::vector<clump_id> &clumpId);

    void updateClumpKeyMap(
        const Statistics::Container::vector<cl_ulong> &keys,
        const Statistics::Container::vector<clump_id> &clumpId);

    void updateLocalClumps(
        Chunk &chunk,
        const Statistics::Container::vector<clump_id> &globalClumpId,
        HostKeyMesh &mesh);


    /// Implementation of the functor
    void add(const ChunkId &chunkId, MesherWork &work);

    /// Function object that accepts incoming vertices and writes them to a writer.
    class VertexBuffer : public boost::noncopyable
    {
    public:
        typedef FastPly::WriterBase::size_type size_type;
    private:
        FastPly::WriterBase &writer;
        size_type nextVertex;
        std::vector<boost::array<float, 3> > buffer;
    public:
        typedef void result_type;

        VertexBuffer(FastPly::WriterBase &writer, size_type capacity);
        void operator()(const boost::array<float, 3> &vertex);
        void flush();
    };

    /// Function object that accepts incoming triangles and writes them to a writer.
    class TriangleBuffer : public boost::noncopyable
    {
    public:
        typedef FastPly::WriterBase::size_type size_type;
    private:
        FastPly::WriterBase &writer;
        size_type nextTriangle;
        std::vector<boost::array<std::tr1::uint32_t, 3> > buffer;
    public:
        typedef void result_type;

        TriangleBuffer(FastPly::WriterBase &writer, size_type capacity);
        void operator()(const boost::array<std::tr1::uint32_t, 3> &triangle);
        void flush();
    };

public:
    /**
     * @copydoc MesherBase::MesherBase
     */
    StxxlMesher(FastPly::WriterBase &writer, const Namer &namer)
        : MesherBase(writer, namer),
        tmpNodes("mem.StxxlMesher::tmpNodes"),
        tmpClumpId("mem.StxxlMesher::tmpClumpId"),
        chunks("mem.StxxlMesher::chunks"),
        clumps("mem.StxxlMesher::clumps"),
        clumpIdMap("mem.StxxlMesher::clumpIdMap") {}

    virtual unsigned int numPasses() const { return 1; }
    virtual InputFunctor functor(unsigned int pass);
    virtual void write(std::ostream *progressStream = NULL);
};

/**
 * Factory function to create a mesher of the specified type.
 *
 * @param writer, namer     Parameters to @ref MesherBase::MesherBase.
 * @param type              The type of mesher to create.
 */
MesherBase *createMesher(MesherType type, FastPly::WriterBase &writer, const MesherBase::Namer &namer);

/**
 * Creates an adapter between @ref MesherBase::InputFunctor and @ref Marching::OutputFunctor
 * that reads the mesh from the device to the host synchronously.
 *
 * @param in        The mesher functor which will receive the host copy of the mesh.
 * @param chunkId   Chunk ID to pass to @a in.
 */
Marching::OutputFunctor deviceMesher(const MesherBase::InputFunctor &in,
                                     const ChunkId &chunkId);

#endif /* !MESHER_H */
