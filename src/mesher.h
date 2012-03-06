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
#include <tr1/unordered_map>
#include "marching.h"
#include "fast_ply.h"
#include "union_find.h"

/**
 * Enumeration of the supported mesher types
 */
enum MesherType
{
    SIMPLE_MESHER,
    WELD_MESHER,
    BIG_MESHER,
    STXXL_MESHER
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
 * Abstract base class for output collectors for @ref Marching.
 *
 * The basic procedure for using one of these classes is:
 * -# Instantiate it.
 * -# Call @ref setPruneThreshold.
 * -# Uses @ref numPasses to determine how many passes are required.
 * -# For each pass, call @ref outputFunctor to obtain a functor, then
 *    make as many calls to @ref Marching::generate as desired using this
 *    functor. Each call should set @a keyOffset so that vertex keys line up.
 *    Each pass must generate exactly the same geometry, but the chunks may
 *    be generated in different order. The functor may be called from
 *    multiple threads simultaneously.
 * -# Call @ref finalize.
 * -# If file output is desired, call @ref write.
 */
class MesherBase
{
protected:

    /**
     * Mutex used by subclasses to serialize their output functors.
     */
    boost::mutex mutex;

public:
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
     * Retrieves a functor to be passed to @ref Marching::generate in a
     * specific pass.
     * Multi-pass classes may do finalization on a previous pass before
     * returning the functor, so this function should only be called for
     * pass @a pass once pass @a pass - 1 has completed. It must also
     * only be called once per pass.
     *
     * @pre @a pass is less than @ref numPasses().
     */
    virtual Marching::OutputFunctor outputFunctor(unsigned int pass) = 0;

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
     * @param filename        Filename to write to.
     * @param progressStream  If non-NULL, a log stream for a progress meter
     * @throw std::ios_base::failure on I/O failure (including failure to open the file).
     *
     * @pre @ref finalize() has been called.
     */
    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
                       std::ostream *progressStream = NULL) const = 0;

private:
    /// Threshold set by @ref setPruneThreshold
    double pruneThreshold;
};

/**
 * Output collector for @ref Marching that does not do any welding of
 * external vertices. It simply collects all the vertices into one vector
 * and indices into another.
 */
class SimpleMesher : public MesherBase
{
private:
    /// Storage for vertices
    std::vector<boost::array<cl_float, 3> > vertices;

    /// Storage for indices.
    std::vector<boost::array<cl_uint, 3> > triangles;

    /// Function called by the functor.
    void add(const cl::CommandQueue &queue,
             const DeviceKeyMesh &mesh,
             const std::vector<cl::Event> *events,
             cl::Event *event);

public:
    virtual unsigned int numPasses() const { return 1; }
    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);

    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
                       std::ostream *progressStream = NULL) const;
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
    void add(const cl::CommandQueue &queue,
             const DeviceKeyMesh &mesh,
             const std::vector<cl::Event> *events,
             cl::Event *event);

public:
    virtual unsigned int numPasses() const { return 1; }
    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);
    virtual void finalize(std::ostream *progressStream = NULL);

    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
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
        explicit Clump(cl_uint vertices) : vertices(vertices), triangles(0) {}

        cl_uint vertices;               ///< Vertices in the component (valid for root clumps)
        std::tr1::uint64_t triangles;   ///< Triangles in the component (valid for root clumps)

        void merge(const Clump &b)
        {
            UnionFind::Node<cl_int>::merge(b);
            vertices += b.vertices;
            triangles += b.triangles;
        }
    };

    /// Data kept regarding each external vertex.
    struct ExternalVertexData
    {
        cl_uint vertexId;
        clump_id clumpId;

        ExternalVertexData(cl_uint vertexId, clump_id clumpId)
            : vertexId(vertexId), clumpId(clumpId) {}
    };

    typedef std::tr1::unordered_map<cl_ulong, ExternalVertexData> map_type;

    /// Maps external vertex keys to external indices
    map_type keyMap;

    /// All clumps, in a structure usable with @ref UnionFind.
    std::vector<Clump> clumps;

    /**
     * @name
     * @{
     * Temporary buffers.
     * These are stored in the object so that memory can be recycled if
     * possible, rather than thrashing the allocator.
     */
    HostKeyMesh tmpMesh;
    std::vector<cl_uint> tmpIndexTable;
    std::vector<UnionFind::Node<cl_int> > tmpNodes;
    std::vector<clump_id> tmpClumpId;
    /** @} */

    /**
     * Identifies clumps in the local set of triangles. Each new clump is
     * appended to @ref clumps.
     *
     * @param numVertices    The number of vertices indexed by @a triangles
     * @param triangles      Triangles, with indices in [0, @a numVertices)
     * @param[out] clumpId   The index into @ref clumps for each vertex
     *
     * @post <code>clumpId.size() == numVertices</code>
     *
     * @warning This function is not reentrant, because it uses the @c
     * tmpNodes vector for internal storage, and because it appends to
     * @ref clumps.
     */
    void computeLocalComponents(
        const HostMesh &mesh,
        std::vector<clump_id> &clumpId);

    /**
     * Add external vertex keys to the key map and computes an index rewrite table.
     * The index rewrite table maps local external indices for a block to their
     * final values. It also does merging of clumps into components.
     *
     * @param vertexOffset    The final index for the first external vertex in the block.
     * @param hKeys           Keys of the external vertices.
     * @param clumpId         The clump IDs of all vertices, as computed by @ref computeLocalComponents.
     * @param[out] indexTable The index remapping table.
     *
     * @note Only the last <code>hKeys.size()</code> elements of @a clumpId are relevant.
     */
    std::size_t updateKeyMap(
        cl_uint vertexOffset,
        const std::vector<cl_ulong> &hKeys,
        const std::vector<clump_id> &clumpId,
        std::vector<cl_uint> &indexTable);

    /**
     * Writes indices in place from being block-relative to the final form.
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
    size_type pruneThresholdVertices;

    typedef std::tr1::unordered_map<cl_ulong, clump_id> key_clump_type;
    /**
     * Clump information for external vertices. During the counting, it contains
     * a clump ID for each external vertex key (note that external vertices belong
     * to multiple clumps, but all in the same component). At the end of the
     * counting pass it is converted to boolean values, indicating whether each
     * external vertex belongs to a component that should be kept.
     */
    key_clump_type keyClump;

    /// Implementation of the first-pass functor
    void count(const cl::CommandQueue &queue,
               const DeviceKeyMesh &mesh,
               const std::vector<cl::Event> *events,
               cl::Event *event);

    /// Preparation for the second pass after the first pass
    void prepareAdd();

    /// Implementation of the second-pass functor
    void add(const cl::CommandQueue &queue,
             const DeviceKeyMesh &mesh,
             const std::vector<cl::Event> *events,
             cl::Event *event);

public:
    virtual unsigned int numPasses() const { return 2; }

    /**
     * Constructor. Unlike the in-core mesher types, the file information must
     * be passed to the constructor so that results can be streamed into it.
     *
     * The file will be created on the second pass.
     */
    BigMesher(FastPly::WriterBase &writer, const std::string &filename);

    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);

    /**
     * Completes writing. The parameters must have the same values given to the constructor.
     */
    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
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
    vertices_type vertices;
    triangles_type triangles;

    /// Implementation of the functor
    void add(const cl::CommandQueue &queue,
             const DeviceKeyMesh &mesh,
             const std::vector<cl::Event> *events,
             cl::Event *event);

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
    virtual Marching::OutputFunctor outputFunctor(unsigned int pass);
    virtual void write(FastPly::WriterBase &writer, const std::string &filename,
                       std::ostream *progressStream = NULL) const;
};

/**
 * Factory function to create a mesher of the specified type.
 */
MesherBase *createMesher(MesherType type, FastPly::WriterBase &writer, const std::string &filename);

#endif /* !MESHER_H */
