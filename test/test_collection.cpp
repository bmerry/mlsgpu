/**
 * @file
 *
 * Test code for the implementation of the @ref Collection concept.
 */
#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/smart_ptr/scoped_array.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <vector>
#include <utility>
#include "testmain.h"
#include "../src/collection.h"

/**
 * Tests for @ref VectorCollection.
 * This is a base class of tests that we specialize for specific vector classes.
 *
 * @param VectorType A vector-like class with integer values.
 */
template<typename Collection>
class TestVectorCollection : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestVectorCollection<Collection>);
    CPPUNIT_TEST(testRead);
    CPPUNIT_TEST(testForEach);
    CPPUNIT_TEST(testOverrun);
    CPPUNIT_TEST_SUITE_END();
protected:
    typedef typename Collection::size_type size_type;
    typename Collection::vector_type v;

private:
    typedef std::pair<size_type, int> item;
    void forEachFunc(std::vector<item> &out, size_type index, int value);
    static void dummyForEachFunc(size_type index, int value);

public:
    virtual void setUp();

    void testRead();          ///< Test the @ref Collection::read member
    void testForEach();       ///< Test the @ref Collection::forEach member
    void testOverrun();       ///< Test the error checking
};

template<typename Collection>
void TestVectorCollection<Collection>::forEachFunc(std::vector<item> &out, size_type index, int value)
{
    out.push_back(item(index, value));
}

template<typename Collection>
void TestVectorCollection<Collection>::dummyForEachFunc(size_type index, int value)
{
    (void) index;
    (void) value;
}

template<typename Collection>
void TestVectorCollection<Collection>::setUp()
{
    int size = 100000;
    v.reserve(size);
    for (int i = 0; i < size; i++)
        v.push_back(i * 10);
}

template<typename Collection>
void TestVectorCollection<Collection>::testRead()
{
    Collection c(v);

    boost::scoped_array<int> buffer(new int[v.size()]);
    std::fill(buffer.get(), buffer.get() + v.size(), -1);
    c.read(4, 9, buffer.get());
    for (int i = 4; i < 9; i++)
        CPPUNIT_ASSERT_EQUAL(i * 10, buffer[i - 4]);
    CPPUNIT_ASSERT_EQUAL(-1, buffer[5]);

    buffer[0] = -1;
    c.read(20, 20, buffer.get());
    CPPUNIT_ASSERT_EQUAL(-1, buffer[0]);

    c.read(0, v.size(), buffer.get());
    for (std::size_t i = 0; i < v.size(); i++)
        CPPUNIT_ASSERT_EQUAL(int(i * 10), buffer[i]);
}

template<typename Collection>
void TestVectorCollection<Collection>::testForEach()
{
    Collection c(v);
    std::vector<item> out;

    c.forEach(4, 9, boost::bind(&TestVectorCollection<Collection>::forEachFunc, this,
                                boost::ref(out), _1, _2));
    CPPUNIT_ASSERT_EQUAL(5, int(out.size()));
    for (int i = 0; i < 5; i++)
    {
        CPPUNIT_ASSERT_EQUAL(size_type(i + 4), out[i].first);
        CPPUNIT_ASSERT_EQUAL((i + 4) * 10, out[i].second);
    }
}

template<typename Collection>
void TestVectorCollection<Collection>::testOverrun()
{
    Collection c(v);

    CPPUNIT_ASSERT_THROW(c.read(1, 0, (int *) NULL), std::out_of_range);
    CPPUNIT_ASSERT_THROW(c.read(v.size() - 1, v.size() + 1, (int *) NULL), std::out_of_range);
    CPPUNIT_ASSERT_THROW(c.forEach(1, 0, dummyForEachFunc), std::out_of_range);
    CPPUNIT_ASSERT_THROW(c.forEach(1, v.size() + 1, dummyForEachFunc), std::out_of_range);
}

/// Tests for @ref StdVectorCollection
class TestStdVectorCollection : public TestVectorCollection<StdVectorCollection<int> >
{
    CPPUNIT_TEST_SUB_SUITE(TestStdVectorCollection, TestVectorCollection<StdVectorCollection<int> >);
    CPPUNIT_TEST_SUITE_END();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestStdVectorCollection, TestSet::perBuild());

#if HAVE_STXXL
/// Tests for @ref StxxlVectorCollection
class TestStxxlVectorCollection : public TestVectorCollection<StxxlVectorCollection<int> >
{
    CPPUNIT_TEST_SUB_SUITE(TestStxxlVectorCollection,
                           TestVectorCollection<StxxlVectorCollection<int> >);
    CPPUNIT_TEST_SUITE_END();
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestStxxlVectorCollection, TestSet::perBuild());
#endif

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
