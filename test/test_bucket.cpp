/**
 * @file
 *
 * Test code for @ref bucket.cpp.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/bind.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/smart_ptr/shared_array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/foreach.hpp>
#include <boost/ref.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>
#include <utility>
#include <limits>
#include <sstream>
#include <cstring>
#include "../src/tr1_cstdint.h"
#include <boost/tr1/random.hpp>
#include "testutil.h"
#include "test_splat_set.h"
#include "../src/bucket.h"
#include "../src/bucket_internal.h"
#include "../src/splat_set.h"

using namespace Bucket;
using namespace Bucket::detail;

static bool gridsIntersect(const Grid &a, const Grid &b)
{
    for (int i = 0; i < 3; i++)
    {
        if (a.getExtent(i).second <= b.getExtent(i).first
            || b.getExtent(i).second <= a.getExtent(i).first)
            return false;
    }
    return true;
}

// clang doesn't find the overload except through argument dependent lookup,
// so it needs to be in the namespace of Node
namespace Bucket { namespace detail {

std::ostream &operator<<(std::ostream &o, const Node &node)
{
    return o << "Node("
        << node.getCoords()[0] << ", "
        << node.getCoords()[1] << ", "
        << node.getCoords()[2] << ", " << node.getLevel() << ")";
}

}}

/// Tests for @ref Bucket::detail::Node
class TestNode : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestNode);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testChild);
    CPPUNIT_TEST(testToCells);
    CPPUNIT_TEST(testToMicro);
    CPPUNIT_TEST(testSize);
    CPPUNIT_TEST_SUITE_END();
public:
    void testConstructor();            ///< Test constructor
    void testChild();                  ///< Test @c child
    void testToCells();                ///< Test @c toCells (both overloads)
    void testToMicro();                ///< Test @c toMicro (both overloads)
    void testSize();                   ///< Test @c size
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestNode, TestSet::perBuild());

void TestNode::testConstructor()
{
    Node n(1, 2, 3, 4);
    const boost::array<Node::size_type, 3> &coords = n.getCoords();
    unsigned int level = n.getLevel();
    CPPUNIT_ASSERT_EQUAL(Node::size_type(1), coords[0]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type(2), coords[1]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type(3), coords[2]);
    CPPUNIT_ASSERT_EQUAL(4U, level);

    Node n2(&coords[0], 4);
    CPPUNIT_ASSERT_EQUAL(n, n2);
}

void TestNode::testChild()
{
    Node parent(1, 2, 3, 4);
    CPPUNIT_ASSERT_EQUAL(Node(2, 4, 6, 3), parent.child(0));
    CPPUNIT_ASSERT_EQUAL(Node(3, 4, 6, 3), parent.child(1));
    CPPUNIT_ASSERT_EQUAL(Node(2, 5, 6, 3), parent.child(2));
    CPPUNIT_ASSERT_EQUAL(Node(3, 5, 6, 3), parent.child(3));
    CPPUNIT_ASSERT_EQUAL(Node(2, 4, 7, 3), parent.child(4));
    CPPUNIT_ASSERT_EQUAL(Node(3, 4, 7, 3), parent.child(5));
    CPPUNIT_ASSERT_EQUAL(Node(2, 5, 7, 3), parent.child(6));
    CPPUNIT_ASSERT_EQUAL(Node(3, 5, 7, 3), parent.child(7));

    CPPUNIT_ASSERT_THROW(Node(1, 2, 3, 0).child(0), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(Node(1, 2, 3, 1).child(8), std::invalid_argument);
}

void TestNode::testToCells()
{
    Node n(1, 2, 3, 2);
    Grid::size_type lower[3], upper[3];
    n.toCells(10, lower, upper);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type( 40), lower[0]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type( 80), lower[1]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(120), lower[2]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type( 80), upper[0]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(120), upper[1]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(160), upper[2]);

    const float ref[3] = {0.0f, 0.0f, 0.0f};
    Grid limit(ref, 3.0f, 1000, 1075, 1000, 1075, 1000, 2000);
    n.toCells(10, lower, upper, limit);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type( 40), lower[0]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type( 75), lower[1]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(120), lower[2]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type( 75), upper[0]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type( 75), upper[1]);
    CPPUNIT_ASSERT_EQUAL(Grid::size_type(160), upper[2]);
}

void TestNode::testToMicro()
{
    Node n(1, 2, 3, 2);
    Node::size_type lower[3], upper[3];
    n.toMicro(lower, upper);
    CPPUNIT_ASSERT_EQUAL(Node::size_type( 4), lower[0]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type( 8), lower[1]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type(12), lower[2]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type( 8), upper[0]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type(12), upper[1]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type(16), upper[2]);

    Node::size_type limit[3] = {7, 7, 200};
    n.toMicro(lower, upper, limit);
    CPPUNIT_ASSERT_EQUAL(Node::size_type( 4), lower[0]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type( 7), lower[1]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type(12), lower[2]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type( 7), upper[0]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type( 7), upper[1]);
    CPPUNIT_ASSERT_EQUAL(Node::size_type(16), upper[2]);
}

void TestNode::testSize()
{
    Node n(1, 2, 3, 4);
    CPPUNIT_ASSERT_EQUAL(Node::size_type(16), n.size());
}


/// Tests for @ref Bucket::detail::forEachNode.
class TestForEachNode : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestForEachNode);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testAsserts);
    CPPUNIT_TEST_SUITE_END();

private:
    std::vector<Node> nodes;
    bool nodeFunc(const Node &node);

public:
    void testSimple();          ///< Test normal usage
    void testAsserts();         ///< Test the assertions of preconditions
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestForEachNode, TestSet::perBuild());

bool TestForEachNode::nodeFunc(const Node &node)
{
    nodes.push_back(node);

    Node::size_type lower[3], upper[3];
    node.toMicro(lower, upper);
    return (lower[0] <= 2 && 2 < upper[0]
        && lower[1] <= 1 && 1 < upper[1]
        && lower[2] <= 4 && 4 < upper[2]);
}

void TestForEachNode::testSimple()
{
    const Node::size_type dims[3] = {4, 4, 6};
    forEachNode(dims, 4, boost::bind(&TestForEachNode::nodeFunc, this, _1));
    /* Note: the recursion order of forEachNode is not defined, so this
     * test is constraining the implementation. It should be changed
     * if necessary.
     */
    CPPUNIT_ASSERT_EQUAL(15, int(nodes.size()));
    CPPUNIT_ASSERT_EQUAL(Node(0, 0, 0, 3), nodes[0]);
    CPPUNIT_ASSERT_EQUAL(Node(0, 0, 0, 2), nodes[1]);
    CPPUNIT_ASSERT_EQUAL(Node(0, 0, 1, 2), nodes[2]);
    CPPUNIT_ASSERT_EQUAL(Node(0, 0, 2, 1), nodes[3]);
    CPPUNIT_ASSERT_EQUAL(Node(1, 0, 2, 1), nodes[4]);
    CPPUNIT_ASSERT_EQUAL(Node(2, 0, 4, 0), nodes[5]);
    CPPUNIT_ASSERT_EQUAL(Node(3, 0, 4, 0), nodes[6]);
    CPPUNIT_ASSERT_EQUAL(Node(2, 1, 4, 0), nodes[7]);
    CPPUNIT_ASSERT_EQUAL(Node(3, 1, 4, 0), nodes[8]);
    CPPUNIT_ASSERT_EQUAL(Node(2, 0, 5, 0), nodes[9]);
    CPPUNIT_ASSERT_EQUAL(Node(3, 0, 5, 0), nodes[10]);
    CPPUNIT_ASSERT_EQUAL(Node(2, 1, 5, 0), nodes[11]);
    CPPUNIT_ASSERT_EQUAL(Node(3, 1, 5, 0), nodes[12]);
    CPPUNIT_ASSERT_EQUAL(Node(0, 1, 2, 1), nodes[13]);
    CPPUNIT_ASSERT_EQUAL(Node(1, 1, 2, 1), nodes[14]);
}

// Not expected to ever be called - just to give a legal function pointer
static bool dummyNodeFunc(const Node &node)
{
    (void) node;
    return false;
}

void TestForEachNode::testAsserts()
{
    const Node::size_type dims[3] = {4, 4, 6};
    CPPUNIT_ASSERT_THROW(forEachNode(dims, 100, dummyNodeFunc), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(forEachNode(dims, 0, dummyNodeFunc), std::invalid_argument);
    CPPUNIT_ASSERT_THROW(forEachNode(dims, 3, dummyNodeFunc), std::invalid_argument);
}

/// Test for @ref Bucket::bucket.
class TestBucket : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestBucket);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testDensityError);
    CPPUNIT_TEST(testMultiLevel);
    CPPUNIT_TEST(testFlat);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST(testChunkCells);
    CPPUNIT_TEST_SUITE_ADD_CUSTOM_TESTS(addRandom);
    CPPUNIT_TEST_SUITE_END();

private:
    struct Block
    {
        Grid grid;
        SplatSet::splat_id numSplats;
        std::size_t numRanges;
        std::vector<SplatSet::splat_id> splatIds;
        std::vector<Splat> splats;
    };

    typedef SplatSet::FastBlobSet<SplatSet::VectorsSet, std::vector<SplatSet::BlobData> > Splats;
    Splats splats;

    void setupSimple();

    void validate(const Splats &splats, const Grid &fullGrid,
                  const std::vector<Block> &blocks,
                  std::size_t maxSplats, Grid::size_type maxCells, Grid::size_type chunkCells);

    template<typename T>
    static void bucketFunc(
        std::vector<Block> &blocks,
        const typename SplatSet::Traits<T>::subset_type &splats,
        const Grid &grid,
        const Recursion &recursionState);

    /// Adds random tests to the fixture
    static void addRandom(TestSuiteBuilderContextType &context);

public:
    void testSimple();            ///< Test basic usage
    void testDensityError();      ///< Test that @ref Bucket::DensityError is thrown correctly
    void testMultiLevel();        ///< Test recursion of @c bucketRecurse
    void testFlat();              ///< Top level already meets the requirements
    void testEmpty();             ///< Edge case with zero splats inside the grid
    void testChunkCells();        ///< Test non-zero @a chunkCells
    void testRandom(unsigned long seed); ///< Randomly-generated test case
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestBucket, TestSet::perBuild());

void TestBucket::addRandom(TestSuiteBuilderContextType &context)
{
    for (unsigned long seed = 0; seed < 30; seed++)
    {
        std::ostringstream name;
        name << "testRandom(" << seed << ")";
        context.addTest(new GenericTestCaller<TestBucket>(context.getTestNameFor(name.str()),
                                                          boost::bind(&TestBucket::testRandom, _1, seed),
                                                          context.makeFixture()));
    }
}

template<typename T>
void TestBucket::bucketFunc(
    std::vector<Block> &blocks,
    const typename SplatSet::Traits<T>::subset_type &splats,
    const Grid &grid,
    const Recursion &)
{
    (void) splats;
    blocks.push_back(Block());
    Block &block = blocks.back();
    block.numSplats = splats.numSplats();
    block.numRanges = splats.numRanges();
    block.grid = grid;
    boost::scoped_ptr<SplatSet::SplatStream> splatStream(splats.makeSplatStream());
    while (!splatStream->empty())
    {
        block.splatIds.push_back(splatStream->currentId());
        block.splats.push_back(**splatStream);
        ++*splatStream;
    }
}

void TestBucket::validate(
    const Splats &splats,
    const Grid &fullGrid,
    const std::vector<Block> &blocks,
    std::size_t maxSplats,
    Grid::size_type maxCells,
    Grid::size_type chunkCells)
{
    // TODO: also need to check that the blocks are no smaller than necessary

    /* To check that we haven't left out any part of a splat, we add up the
     * areas of the intersections with the blocks and check that it adds up to
     * the full bounding box of the splat.
     */
    std::map<SplatSet::splat_id, std::tr1::uint64_t> areas;

    /* First validate each individual block */
    BOOST_FOREACH(const Block &block, blocks)
    {
        CPPUNIT_ASSERT(block.numSplats <= maxSplats);
        CPPUNIT_ASSERT(block.grid.numCells(0) <= maxCells);
        CPPUNIT_ASSERT(block.grid.numCells(1) <= maxCells);
        CPPUNIT_ASSERT(block.grid.numCells(2) <= maxCells);
        CPPUNIT_ASSERT(block.numSplats > 0);
        CPPUNIT_ASSERT_EQUAL((int) block.numSplats, (int) block.splatIds.size());
        /* The grid must be a subgrid of the original */
        CPPUNIT_ASSERT_EQUAL(fullGrid.getSpacing(), block.grid.getSpacing());
        for (int i = 0; i < 3; i++)
        {
            CPPUNIT_ASSERT_EQUAL(fullGrid.getReference()[i], block.grid.getReference()[i]);
            std::pair<int, int> fullExtent = fullGrid.getExtent(i);
            std::pair<int, int> extent = block.grid.getExtent(i);
            CPPUNIT_ASSERT(fullExtent.first <= extent.first);
            CPPUNIT_ASSERT(fullExtent.second >= extent.second);
            // Check that chunking is respected
            if (chunkCells != 0)
            {
                CPPUNIT_ASSERT(divDown(extent.first - fullExtent.first, chunkCells)
                               == divDown(extent.second - fullExtent.first - 1, chunkCells));
            }
        }

        /* Checks that
         * - The splat count must be correct
         * - The splat IDs should be increasing
         * - The splats all intersect the block
         * - The number of ranges is no more than the number of contiguous
         *   ranges of splat IDs (it can be less because there may be
         *   discontinuities in the superset's splat IDs).
         *
         * At the same time, we accumulate the intersection area.
         */
        std::size_t maxRanges = 1;
        for (std::size_t i = 0; i < block.splatIds.size(); i++)
        {
            const SplatSet::splat_id cur = block.splatIds[i];
            if (i > 0)
            {
                const SplatSet::splat_id prev = block.splatIds[i - 1];
                CPPUNIT_ASSERT(prev < cur);
                if (cur != prev + 1)
                    maxRanges++;
            }

            const Splat &splat = block.splats[i];
            std::tr1::uint64_t area = 1.0;
            boost::array<Grid::difference_type, 3> lower, upper;
            SplatSet::detail::splatToBuckets(splat, block.grid, 1, lower, upper);
            for (int k = 0; k < 3; k++)
            {
                lower[k] = std::max(lower[k], 0);
                upper[k] = std::min(upper[k], Grid::difference_type(block.grid.numCells(k)) - 1);
                CPPUNIT_ASSERT(lower[k] <= upper[k]);
                area *= (upper[k] - lower[k] + 1);
            }
            areas[cur] += area;
        }
        CPPUNIT_ASSERT(block.numRanges <= maxRanges);
    }

    /* Check that the blocks do not overlap */
    for (std::size_t b1 = 0; b1 < blocks.size(); b1++)
        for (std::size_t b2 = b1 + 1; b2 < blocks.size(); b2++)
        {
            CPPUNIT_ASSERT(!gridsIntersect(blocks[b1].grid, blocks[b2].grid));
        }

    /* Check that each splat is fully covered */
    boost::scoped_ptr<SplatSet::SplatStream> splatStream(splats.makeSplatStream());
    while (!splatStream->empty())
    {
        Splat splat = **splatStream;

        boost::array<Grid::difference_type, 3> lower, upper;
        SplatSet::detail::splatToBuckets(splat, fullGrid, 1, lower, upper);
        std::tr1::uint64_t area = 1;
        for (unsigned int k = 0; k < 3; k++)
            if (lower[k] <= upper[k])
                area *= upper[k] - lower[k] + 1;
            else
                area = 0;
        CPPUNIT_ASSERT_EQUAL(area, areas[splatStream->currentId()]);
        ++*splatStream;
    }
}

void TestBucket::setupSimple()
{
    createSplats(splats);
    splats.computeBlobs(2.5f, 1);
}

void TestBucket::testSimple()
{
    setupSimple();

    // The grid is set up so that the origin is at (0, 0, 0)
    const float ref[3] = {-10.0f, 0.0f, 10.0f};
    Grid grid(ref, 2.5f, 4, 20, 0, 20, -4, 4);
    std::vector<Block> blocks;
    const int maxSplats = 5;
    const int maxCells = 8;
    const int maxSplit = 1000000;
    bucket(splats, grid, maxSplats, maxCells, 0, maxCells, maxSplit,
           boost::bind(&TestBucket::bucketFunc<Splats>, boost::ref(blocks), _1, _2, _3));
    validate(splats, grid, blocks, maxSplats, maxCells, 0);

    // 11 was found by inspecting the output and checking the
    // blocks by hand
    CPPUNIT_ASSERT_EQUAL(11, int(blocks.size()));
}

void TestBucket::testDensityError()
{
    setupSimple();

    const float ref[3] = {-10.0f, 0.0f, 10.0f};
    Grid grid(ref, 2.5f, 4, 20, 0, 20, -4, 4);
    std::vector<Block> blocks;
    const int maxSplats = 1;
    const int maxCells = 8;
    const int maxSplit = 1000000;
    CPPUNIT_ASSERT_THROW(
        bucket(splats, grid, maxSplats, maxCells, 0, maxCells, maxSplit,
           boost::bind(&TestBucket::bucketFunc<Splats>, boost::ref(blocks), _1, _2, _3)),
        DensityError);
}

void TestBucket::testFlat()
{
    setupSimple();

    const float ref[3] = {-10.0f, 0.0f, 10.0f};
    Grid grid(ref, 2.5f, 4, 20, 0, 20, -4, 4);
    std::vector<Block> blocks;
    const int maxSplats = 15;
    const int maxCells = 32;
    const int maxSplit = 1000000;
    bucket(splats, grid, maxSplats, maxCells, 0, maxCells, maxSplit,
           boost::bind(&TestBucket::bucketFunc<Splats>, boost::ref(blocks), _1, _2, _3));
    validate(splats, grid, blocks, maxSplats, maxCells, 0);

    CPPUNIT_ASSERT_EQUAL(1, int(blocks.size()));
}

void TestBucket::testEmpty()
{
    const float ref[3] = {-10.0f, 0.0f, 10.0f};
    Grid grid(ref, 2.5f, 4, 20, 0, 20, -4, 4);
    std::vector<Block> blocks;
    const int maxSplats = 5;
    const int maxCells = 8;
    const int maxSplit = 1000000;

    /* We can't use a FastBlobSet for this, because it requires at least one splat to create
     * the bounding box.
     */
    typedef SplatSet::VectorsSet Set;
    Set splatSet;
    bucket(splatSet, grid, maxSplats, maxCells, 0, maxCells, maxSplit,
           boost::bind(&TestBucket::bucketFunc<Set>, boost::ref(blocks), _1, _2, _3));
    CPPUNIT_ASSERT(blocks.empty());
}

void TestBucket::testMultiLevel()
{
    setupSimple();

    const float ref[3] = {-10.0f, 0.0f, 10.0f};
    Grid grid(ref, 2.5f, 4, 20, 0, 20, -4, 4);
    std::vector<Block> blocks;
    const int maxSplats = 5;
    const int maxCells = 8;
    const int maxSplit = 8;
    bucket(splats, grid, maxSplats, maxCells, 0, maxCells, maxSplit,
           boost::bind(&TestBucket::bucketFunc<Splats>, boost::ref(blocks), _1, _2, _3));
    validate(splats, grid, blocks, maxSplats, maxCells, 0);

    // 11 was found by inspecting the output and checking the
    // blocks by hand
    CPPUNIT_ASSERT_EQUAL(11, int(blocks.size()));
}

void TestBucket::testChunkCells()
{
    setupSimple();

    // The grid is set up so that the origin is at (0, 0, 0)
    const float ref[3] = {-10.0f, 0.0f, 10.0f};
    Grid grid(ref, 2.5f, 4, 20, 0, 20, -4, 4);
    std::vector<Block> blocks;
    const int maxSplats = 20;
    const int maxCells = 8;
    const int maxSplit = 1000000;
    const int chunkCells = 14;
    const int chunkCellsRounded = 16;
    bucket(splats, grid, maxSplats, INT_MAX, chunkCells, maxCells, maxSplit,
           boost::bind(&TestBucket::bucketFunc<Splats>, boost::ref(blocks), _1, _2, _3));
    validate(splats, grid, blocks, maxSplats, INT_MAX, chunkCellsRounded);
}

static int simpleRandomInt(std::tr1::mt19937 &engine, int min, int max)
{
    using std::tr1::mt19937;
    using std::tr1::uniform_int;
    using std::tr1::variate_generator;

    /* According to TR1, there has to be a conversion from the output type
     * of the engine to the input type of the distribution, so we can't
     * just use uniform_int<int> (and MSVC will do the wrong thing in this
     * case). We thus also have to manually bias to avoid negative numbers.
     */
    variate_generator<mt19937 &, uniform_int<mt19937::result_type> > gen(engine, uniform_int<mt19937::result_type>(0, max - min));
    return int(gen()) + min;
}

static float simpleRandomFloat(std::tr1::mt19937 &engine, float min, float max)
{
    using std::tr1::mt19937;
    using std::tr1::uniform_real;
    using std::tr1::variate_generator;

    variate_generator<mt19937 &, uniform_real<float> > gen(engine, uniform_real<float>(min, max));
    return gen();
}

void TestBucket::testRandom(unsigned long seed)
{
    using std::tr1::mt19937;
    using std::tr1::uniform_int;
    using std::tr1::uniform_real;
    using std::tr1::variate_generator;
    using std::tr1::bernoulli_distribution;

    mt19937 engine(seed);
    unsigned int numScans = simpleRandomInt(engine, 0, 20);
    unsigned int maxScan = simpleRandomInt(engine, 1, 2000); // maximum splats in one scan
    unsigned int maxSplit = simpleRandomInt(engine, 64, 1000);
    unsigned int maxCells = simpleRandomInt(engine, 40, 100);
    unsigned int chunkCells = simpleRandomInt(engine, 80, 513);
    if (bernoulli_distribution(0.5)(engine))
        chunkCells = 0;
    unsigned int maxSplats = simpleRandomInt(engine, 20, 10000);
    float minX = simpleRandomFloat(engine, -100.0f, 10.0f);
    float maxX = simpleRandomFloat(engine, 20.0f, 100.0f);
    float minY = simpleRandomFloat(engine, -100.0f, 1.0f);
    float maxY = simpleRandomFloat(engine, 20.0f, 100.0f);
    float minZ = simpleRandomFloat(engine, -100.0f, 1.0f);
    float maxZ = simpleRandomFloat(engine, 20.0f, 100.0f);
    float spacing = simpleRandomFloat(engine, 0.25f, 2.5f);
    float maxRadius = simpleRandomFloat(engine, 0.25f, 10.0f);

    variate_generator<mt19937 &, uniform_int<mt19937::result_type> > genNum(engine, uniform_int<mt19937::result_type>(0, maxScan));
    variate_generator<mt19937 &, uniform_real<float> > genX(engine, uniform_real<float>(minX, maxX));
    variate_generator<mt19937 &, uniform_real<float> > genY(engine, uniform_real<float>(minY, maxY));
    variate_generator<mt19937 &, uniform_real<float> > genZ(engine, uniform_real<float>(minZ, maxZ));
    variate_generator<mt19937 &, uniform_real<float> > genR(engine, uniform_real<float>(0.01f, maxRadius));

    splats.clear();
    for (unsigned int i = 0; i < numScans; i++)
    {
        splats.push_back(std::vector<Splat>());
        unsigned int num = genNum();
        for (unsigned int i = 0; i < num; i++)
        {
            Splat splat;
            splat.position[0] = genX();
            splat.position[1] = genY();
            splat.position[2] = genZ();
            splat.radius = genR();
            splat.quality = 0.0f;
            splat.normal[0] = splat.normal[1] = splat.normal[2] = 1.0f; // arbitrary
            splats.back().push_back(splat);
        }
    }

    try
    {
        splats.computeBlobs(spacing, maxCells);
    }
    catch (std::runtime_error &e)
    {
        // Harmless: caused if there are no splats
        return;
    }
    const Grid grid = splats.getBoundingGrid();
    std::vector<Block> blocks;
    try
    {
        bucket(splats, grid, maxSplats, maxCells, chunkCells, maxCells, maxSplit,
               boost::bind(&TestBucket::bucketFunc<Splats>, boost::ref(blocks), _1, _2, _3));
        validate(splats, grid, blocks, maxSplats, maxCells, 0);
    }
    catch (DensityError &e)
    {
        // Harmless: it just mean the random parameters were too dense.
    }
}
