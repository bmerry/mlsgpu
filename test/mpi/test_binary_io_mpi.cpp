/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file
 *
 * Tests for @ref binary_io_mpi.h.
 */

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/system/error_code.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include "../../src/binary_io.h"
#include "../../src/binary_io_mpi.h"
#include "../../src/misc.h"
#include "../../src/serialize.h"
#include "../testutil.h"

static const BinaryIO::offset_type seekPos = 9876543210LL;

/**
 * Tests for @ref BinaryWriterMPI.
 */
class TestBinaryWriterMPI : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestBinaryWriterMPI);
    CPPUNIT_TEST(testResize);
    CPPUNIT_TEST(testWrite);
    CPPUNIT_TEST_SUITE_END();

public:
    virtual void setUp();
    virtual void tearDown();

private:
    MPI_Comm comm;
    boost::scoped_ptr<BinaryWriterMPI> writer;
    boost::filesystem::path testPath;

    void testResize();
    void testWrite();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestBinaryWriterMPI, TestSet::perBuild());

void TestBinaryWriterMPI::setUp()
{
    int rank;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
    {
        boost::filesystem::ofstream dummy;
        createTmpFile(testPath, dummy);
        dummy.close();
    }
    Serialize::broadcast(testPath, comm, 0);

    writer.reset(new BinaryWriterMPI(comm));
    writer->open(testPath);
}

void TestBinaryWriterMPI::tearDown()
{
    writer.reset();

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
    {
        boost::filesystem::remove(testPath);
    }

    MPI_Comm_free(&comm);
    MPI_Barrier(MPI_COMM_WORLD);
}

void TestBinaryWriterMPI::testResize()
{
    writer->resize(seekPos);
    writer->close();

    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
    {
        MLSGPU_ASSERT_EQUAL(boost::filesystem::file_size(testPath), seekPos);
    }
}

void TestBinaryWriterMPI::testWrite()
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    char buffer[64];
    for (std::size_t i = 0; i < sizeof(buffer); i++)
        buffer[i] = rank + i + 1;

    BinaryWriterMPI::offset_type offset;
    if (rank == 0)
        offset = seekPos;
    else
        offset = (rank - 1) * sizeof(buffer);

    writer->write(buffer, sizeof(buffer), offset);
    writer->close();
    MPI_Barrier(comm);

    if (rank == 0)
    {
        boost::scoped_ptr<BinaryReader> reader(createReader(SYSCALL_READER));
        reader->open(testPath);
        for (int r = 0; r < size; r++)
        {
            char expected[sizeof(buffer)];
            for (std::size_t i = 0; i < sizeof(buffer); i++)
                expected[i] = r + i + 1;

            if (r == 0)
                offset = seekPos;
            else
                offset = (r - 1) * sizeof(buffer);

            std::size_t c = reader->read(buffer, sizeof(buffer), offset);
            CPPUNIT_ASSERT_EQUAL(sizeof(buffer), c);
            for (std::size_t i = 0; i < sizeof(buffer); i++)
                CPPUNIT_ASSERT_EQUAL(int(expected[i]), int(buffer[i]));
        }
        reader->close();
    }
}
