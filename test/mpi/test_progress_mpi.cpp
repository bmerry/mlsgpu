/**
 * @file
 *
 * Tests for @ref ProgressMPI.
 */
#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/ref.hpp>
#include <string>
#include <mpi.h>
#include "../testutil.h"
#include "../../src/tr1_cstdint.h"
#include "../../src/progress_mpi.h"
#include "../../src/progress.h"

class TestProgressMPI : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestProgressMPI);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST_SUITE_END();

public:
    virtual void setUp();
    virtual void tearDown();

private:
    MPI_Comm comm;

    void testSimple();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestProgressMPI, TestSet::perBuild());

void TestProgressMPI::setUp()
{
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
}

void TestProgressMPI::tearDown()
{
    MPI_Comm_free(&comm);
    MPI_Barrier(MPI_COMM_WORLD);
}

void TestProgressMPI::testSimple()
{
    int rank, size;
    const ProgressMeter::size_type total = UINT64_C(100000000000);
    const int chunksPerProcess = 1024;
    const int chunksPerSync = 32;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int totalChunks = chunksPerProcess * size;

    boost::scoped_ptr<ProgressDisplay> parent;
    boost::scoped_ptr<boost::thread> thread;
    std::ostringstream output;
    if (rank == 0)
        parent.reset(new ProgressDisplay(total, output));

    ProgressMPI progress(parent.get(), total, MPI_COMM_WORLD, 0);
    if (rank == 0)
        thread.reset(new boost::thread(boost::ref(progress)));

    int firstChunk = rank * chunksPerProcess;
    for (int i = 0; i < chunksPerProcess; i++)
    {
        int chunk = i + firstChunk;
        ProgressMeter::size_type first = chunk * total / totalChunks;
        ProgressMeter::size_type last = (chunk + 1) * total / totalChunks;
        progress += last - first;
        if (i % chunksPerSync == 0)
            progress.sync();
    }
    progress.sync();

    if (rank == 0)
    {
        thread->join();
        CPPUNIT_ASSERT_EQUAL(std::string(
                "\n"
                "0%   10   20   30   40   50   60   70   80   90   100%\n"
                "|----|----|----|----|----|----|----|----|----|----|\n"
                "***************************************************\n"),
            output.str());
    }
}
