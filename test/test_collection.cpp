/**
 * @file
 *
 * Test code for the implementation of the @ref Collections concept.
 */
#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/ptr_container/ptr_vector.hpp>
#include <memory>
#include <vector>
#include "testmain.h"
#include "../src/collection.h"

/// Tests for @ref CollectionStream
class TestCollectionStream : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestCollectionStream);
    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testSimple);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST_SUITE_END();
private:
    typedef StdVectorCollection<int> Collection;
public:
    void testConstructor();       ///< Test the default constructor
    void testSimple();            ///< Tests normal use
    void testEmpty();             ///< Tests an empty stream
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestCollectionStream, TestSet::perBuild());

void TestCollectionStream::testConstructor()
{
    CollectionStream<boost::ptr_vector<Collection>::const_iterator> stream;
    CPPUNIT_ASSERT(stream.empty());
}

void TestCollectionStream::testSimple()
{
    const unsigned int nFiles = 5;
    const unsigned int sizes[nFiles] = {0, 5, 1, 0, 3};

    std::vector<std::vector<int> > backing(nFiles);
    boost::ptr_vector<Collection> files;
    std::vector<int> expected;
    for (unsigned int i = 0; i < nFiles; i++)
    {
        backing.push_back(std::vector<int>());
        for (unsigned int j = 0; j < sizes[i]; j++)
        {
            backing[i].push_back(i * 100 + j);
            expected.push_back(i * 100 + j);
        }
        files.push_back(new Collection(backing[i]));
    }

    typedef boost::ptr_vector<Collection>::const_iterator iterator;
    CollectionStream<iterator> stream(files.begin(), files.end());
    std::vector<int> actual;
    while (!stream.empty())
    {
        actual.push_back(*stream);
        ++stream;
    }
    CPPUNIT_ASSERT_EQUAL(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); i++)
    {
        CPPUNIT_ASSERT_EQUAL(expected[i], actual[i]);
    }
}

void TestCollectionStream::testEmpty()
{
    std::vector<int> empty;
    boost::ptr_vector<Collection> files;

    files.push_back(new Collection(empty));
    CollectionStream<boost::ptr_vector<Collection>::const_iterator> stream;

    CPPUNIT_ASSERT(stream.empty());
}
