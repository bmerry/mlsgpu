/**
 * @file
 *
 * Test code for @ref SplatSet::FastBlobSetMPI.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/iostreams/device/null.hpp>
#include <boost/iostreams/stream.hpp>
#include <memory>
#include <vector>
#include <string>
#include <mpi.h>
#include "../test_splat_set.h"
#include "../testutil.h"
#include "../../src/grid.h"
#include "../../src/splat.h"
#include "../../src/splat_set_mpi.h"

using namespace SplatSet;

/// Tests for @ref SplatSet::FastBlobSetMPI<SplatSet::FileSet>.
class TestFastFileSetMPI : public TestFastFileSet
{
    CPPUNIT_TEST_SUB_SUITE(TestFastFileSetMPI, TestFastFileSet);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST(testProgress);
    CPPUNIT_TEST_SUITE_END();

private:
    MPI_Comm comm;
    std::vector<std::string> store;

protected:
    virtual Set *setFactory(const std::vector<std::vector<Splat> > &splatData,
                            float spacing, Grid::size_type bucketSize);
public:
    virtual void setUp();
    virtual void tearDown();

    void testEmpty();            ///< Test error checking for an empty set
    void testProgress();         ///< Run with a progress stream (does not check output)
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestFastFileSetMPI, TestSet::perBuild());

void TestFastFileSetMPI::setUp()
{
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
}

void TestFastFileSetMPI::tearDown()
{
    MPI_Comm_free(&comm);
    MPI_Barrier(MPI_COMM_WORLD);
}

TestFastFileSetMPI::Set *TestFastFileSetMPI::setFactory(
    const std::vector<std::vector<Splat> > &splatData,
    float spacing, Grid::size_type bucketSize)
{
    if (splatData.empty())
        return NULL; // otherwise computeBlobs will throw
    std::auto_ptr<FastBlobSetMPI<FileSet> > set(new FastBlobSetMPI<FileSet>);
    TestFileSet::populate(*set, splatData, store);
    set->computeBlobs(comm, 0, spacing, bucketSize, NULL, false);
    return set.release();
}

void TestFastFileSetMPI::testEmpty()
{
    boost::scoped_ptr<FastBlobSetMPI<FileSet> > set(new FastBlobSetMPI<FileSet>());
    CPPUNIT_ASSERT_THROW(set->computeBlobs(comm, 0, 2.5f, 5, NULL, false), std::runtime_error);
}

void TestFastFileSetMPI::testProgress()
{
    boost::scoped_ptr<FastBlobSetMPI<FileSet> > set(new FastBlobSetMPI<FileSet>());
    TestFileSet::populate(*set, splatData, store);

    boost::iostreams::null_sink nullSink;
    boost::iostreams::stream<boost::iostreams::null_sink> nullStream(nullSink);
    set->computeBlobs(comm, 0, 2.5f, 5, &nullStream, false);
}
