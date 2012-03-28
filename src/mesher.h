/**
 * @file
 *
 * Data structures for storing the output of @ref Marching.
 *
 * The classes in this file are @ref MesherBase, an abstract base class, and
 * several concrete instantiations of it. They differ in terms of
 *  - the number of passes needed
 *  - whether they support welding of external vertices
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
#include <boost/foreach.hpp>
#include <tr1/unordered_map>
#include <tr1/unordered_set>
#include "marching.h"
#include "fast_ply.h"
#include "union_find.h"
#include "work_queue.h"

/**
 * Enumeration of the supported mesher types
 */
enum MesherType
{
    WELD_MESHER,
    BIG_MESHER,
    STXXL_MESHER
};

struct ChunkId
{
    unsigned int gen;
    boost::array<Grid::size_type, 3> coords;

    ChunkId() : gen(0)
    {
        for (unsigned int i = 0; i < 3; i++)
            coords[i] = 0;
    }

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

struct MesherWork
{
    HostKeyMesh mesh;
    cl::Event verticesEvent;
    cl::Event vertexKeysEvent;
    cl::Event trianglesEvent;
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
 * using @ref deviceMesher or @ref DeviceMesherAsync to satisfy
 * the requirements for @ref Marching.
 *
 * The basic procedure for using one of these classes is:
 * -# Instantiate it.
 * -# Call @ref setPruneThreshold.
 * -# Uses @ref numPasses to determine how many passes are required.
 * -# For each pass, call @ref functor to obtain a functor, then
 *    make as many calls to @ref Marching::generate as desired using this
 *    functor. Each call should set @a keyOffset so that vertex keys line up.
 *    Each pass must generate exactly the same geometry, but the chunks may
 *    be generated in different order.
 * -# Call @ref finalize.
 * -# If file output is desired, call @ref write.
 *
 * @warning The functor is @em not required to be thread-safe. The caller must
 * serialize calls if necessary.
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

    /// Constructor
    MesherBase() : pruneThreshold(0.0) {}

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
     * @pre @a pass is less than @ref numPasses().
     *
     * @warning The returned functor is @em not required to be thread-safe.
     */
    virtual InputFunctor functor(unsigned int pass) = 0;

    /**
     * Perform any final processing once the last pass has completed.
     *
     * @param progressStream If non-NULL and finalization does significant work,
     * a progress meter will be displayed.
     * @pre All passes have been executed.
     */
    virtual void finalize(std::ostream *progressStream = NULL) { (void) progressStream; }

    /**
     * Writes the data to file. The writer passed in must not yet have been opened
     * (this function will do that). The caller may optionally have set comments on it.
     *
     * @param writer          Writer to write to (unopened).
     * @param namer           Filename generator.
     * @param progressStream  If non-NULL, a log stream for a progress meter
     * @throw std::ios_base::failure on I/O failure (including failure to open the file).
     *
     * @pre @ref finalize() has been called.
     */
    virtual void write(FastPly::WriterBase &writer, const Namer &namer,
                       std::ostream *progressStream = NULL) const = 0;

private:
    /// Threshold set by @ref setPruneThreshold
    double pruneThreshold;
};

/**
 * Mesher that operates in one pass, and implements welding of external
 * vertices.
 *
 * Internal and external vertices are partitioned into separate arrays,
 * and external vertex keys are kept to allow final welding. In
 * intermediate stages, indices are encoded as either an index into
 * the internal vector, or as the bit-inverse (~) of an index into
 * the external vector. The external indices are rewritten during
 * @ref finalize().
 */
class WeldMesher : public MesherBase
{
private:
    /// Storage for internal vertices
    std::vector<boost::array<cl_float, 3> > internalVertices;
    /// Storage for external vertices
    std::vector<boost::array<cl_float, 3> > externalVertices;
    /// Storage for external vertex keys
    std::vector<cl_ulong> externalKeys;
    /// Storage for indices
    std::vector<boost::array<cl_uint, 3> > triangles;

    /// Implementation of the functor
    void add(const ChunkId &chunkId, MesherWork &work);

public:
    virtual unsigned int numPasses() const { return 1; }
    virtual InputFunctor functor(unsigned int pass);
    virtual void finalize(std::ostream *progressStream = NULL);

    virtual void write(FastPly::WriterBase &writer, const Namer &namer,
                       std::ostream *progressStream = NULL) const;
};

namespace detail
{

/**
 * An internal base class for @ref BigMesher and @ref StxxlMesher, implementing
 * algorithms common to both.
 *
 * External vertices are entered into a hash table that maps their keys to
 * their final indices in the output. When a new chunk of data comes in,
 * new external vertices are entered into the key map and known ones are
 * discarded. The indices are then rewritten using the key map.
 *
 * Component identification is implemented with a two-level approach. Within each
 * call to add(), a union-find is performed to identify local components. These
 * components are referred to as @em clumps. Each vertex is given a <em>clump
 * id</em>. During welding, external vertices are used to identify clumps that
 * form part of the same component, and this is recorded in a union-find
 * structure over the clumps.
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
        Counts counts;                                              ///< Global counts
        std::tr1::unordered_map<unsigned int, Counts> chunkCounts;  ///< Per-chunk counts
        /** @} */

        void merge(Clump &b)
        {
            UnionFind::Node<cl_int>::merge(b);
            counts += b.counts;

            // Merge chunkCounts with b.chunkCounts, destroying b.chunkCounts in
            // the process.
            typedef std::pair<unsigned int, Counts> item_type;
            BOOST_FOREACH(const item_type &item, b.chunkCounts)
            {
                chunkCounts[item.first] += item.second;
            }
            b.chunkCounts.clear();
        }

        Clump(unsigned int chunkGen, std::tr1::uint64_t numVertices)
        {
            counts.vertices = numVertices;
            chunkCounts[chunkGen].vertices = numVertices;
        }
    };

    typedef std::tr1::unordered_map<cl_ulong, cl_uint> vertex_id_map_type;
    typedef std::tr1::unordered_map<cl_ulong, clump_id> clump_id_map_type;

    /// Maps external vertex keys to external indices for the current chunk
    vertex_id_map_type vertexIdMap;

    /// Maps external vertex keys to clumps
    clump_id_map_type clumpIdMap;

    /// All clumps, in a structure usable with @ref UnionFind.
    std::vector<Clump> clumps;

    /**
     * @name
     * @{
     * Temporary buffers.
     * These are stored in the object so that memory can be recycled if
     * possible, rather than thrashing the allocator.
     */
    std::vector<cl_uint> tmpIndexTable;
    std::vector<UnionFind::Node<cl_int> > tmpNodes;
    std::vector<clump_id> tmpClumpId;
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
        const std::vector<boost::array<cl_uint, 3> > &triangles,
        std::vector<UnionFind::Node<cl_int> > &nodes);

    /**
     * Creates clumps from local components.
     *
     * @param chunkGen       Chunk generation to which the local triangles belong.
     * @param nodes          A union-find tree over the vertices (see @ref computeLocalComponents).
     * @param triangles      The triangles used to determine connectivity.
     * @param[out] clumpId   The index into @ref clumps for each vertex.
     *
     * @post <code>clumpId.size() == nodes.size()</code>
     *
     * @warning This function is not reentrant, because it appends to
     * @ref clumps.
     */
    void updateClumps(
        unsigned int chunkGen,
        const std::vector<UnionFind::Node<cl_int> > &nodes,
        const std::vector<boost::array<cl_uint, 3> > &triangles,
        std::vector<clump_id> &clumpId);

    /**
     * Add external vertex keys to the key maps and computes an index rewrite table.
     * The index rewrite table maps local external indices for a block to their
     * almost final values (prior to component removal). It also does merging
     * of clumps into components.
     *
     * @param chunkGen        Chunk generation to which the local vertices belong.
     * @param vertexOffset    The final index for the first external vertex in the block.
     * @param hKeys           Keys of the external vertices.
     * @param clumpId         The clump IDs of all vertices, as computed by @ref computeLocalComponents.
     * @param[out] indexTable The index remapping table.
     *
     * @note Only the last <code>hKeys.size()</code> elements of @a clumpId are relevant.
     */
    std::size_t updateKeyMaps(
        unsigned int chunkGen,
        cl_uint vertexOffset,
        const std::vector<cl_ulong> &hKeys,
        const std::vector<clump_id> &clumpId,
        std::vector<cl_uint> &indexTable);

    /**
     * Writes indices in place from being block-relative to the intermediate form (prior to
     * component removal).
     * @param priorVertices        First vertex in the block (internal or external).
     * @param indexTable           External index rewrite table computed by @ref updateKeyMap.
     * @param[in,out] mesh         Triangles to rewrite (also uses the number of vertices).
     */
    void rewriteTriangles(
        cl_uint priorVertices,
        const std::vector<cl_uint> &indexTable,
        HostKeyMesh &mesh) const;
};

} // namespace detail

/**
 * Two-pass collector that can handle very large meshes by writing
 * the geometry to file as it is produced. It requires an out-of-order
 * writer, and requires the writer to be provided up front.
 *
 * The two passes are:
 * 1. Counting, and assigning the key mapping to detect duplicate external
 *    vertices.
 * 2. Write the data.
 *
 * Unlike @ref WeldMesher, the external vertices are written out as they come in
 * (immediately after the internal vertices for the corresponding chunk), which
 * avoids the need to buffer them up until the end. The only unbounded memory is
 * for the key map.
 */
class BigMesher : public detail::KeyMapMesher
{
private:
    typedef FastPly::WriterBase::size_type size_type;

    FastPly::WriterBase &writer;
    const std::string filename;

    size_type nextVertex;   ///< Number of vertices written so far
    size_type nextTriangle; ///< Number of triangles written so far

    /**
     * Minimum number of vertices to keep a component. This is only valid
     * after @ref prepareAdd.
     */
    std::tr1::uint64_t pruneThresholdVertices;

    /// Keys of external vertices in retained chunks, computed by prepareAdd
    std::tr1::unordered_set<cl_ulong> retainedExternal;

    /// Temporary storage for local clump validity
    std::vector<bool> tmpClumpValid;

    /// Implementation of the first-pass functor
    void count(const ChunkId &chunkId, MesherWork &work);

    /// Preparation for the second pass after the first pass
    void prepareAdd();

    /// Implementation of the second-pass functor
    void add(const ChunkId &chunkId, MesherWork &work);

public:
    virtual unsigned int numPasses() const { return 2; }

    /**
     * Constructor. Unlike the in-core mesher types, the file information must
     * be passed to the constructor so that results can be streamed into it.
     *
     * The file will be created on the second pass.
     */
    BigMesher(FastPly::WriterBase &writer, const std::string &filename);

    virtual InputFunctor functor(unsigned int pass);

    /**
     * Completes writing. The parameters must have the same values given to the constructor.
     */
    virtual void write(FastPly::WriterBase &writer, const Namer &namer,
                       std::ostream *progressStream = NULL) const;
};

#include <stxxl.h>

/**
 * Mesher class that uses the same algorithm as @ref BigMesher, but stores
 * the data in STXXL containers before concatenating them rather than
 * using multiple passes. It thus trades storage requirements against
 * performance, at least when @ref BigMesher is compute-bound.
 */
class StxxlMesher : public detail::KeyMapMesher
{
private:
    typedef FastPly::WriterBase::size_type size_type;
    typedef stxxl::VECTOR_GENERATOR<std::pair<boost::array<float, 3>, clump_id> >::result vertices_type;
    typedef stxxl::VECTOR_GENERATOR<boost::array<cl_uint, 3> >::result triangles_type;

    struct Chunk
    {
        ChunkId chunkId;
        vertices_type::size_type firstVertex;
        triangles_type::size_type firstTriangle;
    };

    vertices_type vertices;
    triangles_type triangles;
    std::vector<Chunk> chunks;

    /// Implementation of the functor
    void add(const ChunkId &chunkId, MesherWork &work);

    /// Function object that accepts incoming vertices and writes them to a writer.
    class VertexBuffer : public boost::noncopyable
    {
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
    virtual unsigned int numPasses() const { return 1; }
    virtual InputFunctor functor(unsigned int pass);
    virtual void write(FastPly::WriterBase &writer, const Namer &namer,
                       std::ostream *progressStream = NULL) const;
};

/**
 * Factory function to create a mesher of the specified type.
 */
MesherBase *createMesher(MesherType type, FastPly::WriterBase &writer, const std::string &filename);

/**
 * Creates an adapter between @ref MesherBase::InputFunctor and @ref Marching::OutputFunctor
 * that reads the mesh from the device to the host synchronously.
 */
Marching::OutputFunctor deviceMesher(const MesherBase::InputFunctor &in);

#endif /* !MESHER_H */
