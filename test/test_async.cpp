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
 * Tests for @ref AsyncWriter.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/tr1/random.hpp>
#include "testutil.h"
#include "../src/binary_io.h"
#include "../src/async_io.h"
#include "../src/misc.h"
#include "../src/tr1_cstdint.h"

class TestAsyncWriter : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestAsyncWriter);
    CPPUNIT_TEST(testStress);
    CPPUNIT_TEST_SUITE_END();

public:
    virtual void setUp();    ///< Obtain filename for temporary file
    virtual void tearDown(); ///< Wipe out the output file

private:
    boost::filesystem::path filename;

    void testStress();   ///< Make lots of writes to file, check that they arrive
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestAsyncWriter, TestSet::perNightly());

void TestAsyncWriter::setUp()
{
    boost::filesystem::ofstream dummy;
    createTmpFile(filename, dummy);
}

void TestAsyncWriter::tearDown()
{
    boost::filesystem::remove(filename);
}

void TestAsyncWriter::testStress()
{
    Timeplot::Worker tworker("test");
    typedef std::tr1::uint32_t value_type;
    const value_type size = 1100000000; // file will be larger than 4GiB
    const std::size_t bufferSize = 64 * 1024;

    boost::shared_ptr<BinaryWriter> writer(createWriter(SYSCALL_WRITER));
    writer->open(filename);
    writer->resize(BinaryWriter::offset_type(size) * sizeof(value_type));

    AsyncWriter async(4, bufferSize * sizeof(value_type));
    async.start();

    std::tr1::mt19937 engine;
    std::tr1::uniform_int<int> dist(1, bufferSize / 2);
    value_type pos = 0;
    while (pos < size)
    {
        value_type chunk = dist(engine);
        if (pos + chunk > size)
            chunk = size - pos;
        boost::shared_ptr<AsyncWriterItem> item = async.get(tworker, chunk * sizeof(value_type));
        value_type *ptr = reinterpret_cast<value_type *>(item->get());
        for (value_type i = 0; i < chunk; i++)
            ptr[i] = i + pos;
        async.push(tworker, item, writer,
                   chunk * sizeof(value_type),
                   BinaryWriter::offset_type(pos) * sizeof(value_type));
        pos += chunk;
    }
    writer.reset();
    async.stop();

    boost::scoped_ptr<BinaryReader> reader(createReader(SYSCALL_READER));
    reader->open(filename);
    value_type buffer[bufferSize];
    pos = 0;
    while (pos < size)
    {
        std::size_t read = reader->read(
            buffer,
            bufferSize * sizeof(value_type),
            BinaryWriter::offset_type(pos) * sizeof(value_type));
        CPPUNIT_ASSERT(read > 0 && read % sizeof(value_type) == 0);
        read /= sizeof(value_type);
        for (value_type i = 0; i < read; i++)
            CPPUNIT_ASSERT_EQUAL(i + pos, buffer[i]);
        pos += read;
    }
    CPPUNIT_ASSERT_EQUAL(size, pos);
    reader->close();
}
