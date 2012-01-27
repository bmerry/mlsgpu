/**
 * @file
 *
 * Tests for helper functionality defined in @ref splat.h.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <vector>
#include "testmain.h"
#include "../src/splat.h"

using namespace std;

/// Tests for @ref CompareSplatsMorton
class TestCompareSplatsMorton : public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestCompareSplatsMorton);
    CPPUNIT_TEST(testOrder);
    CPPUNIT_TEST(testMinMax);
    CPPUNIT_TEST_SUITE_END();
private:
    vector<Splat> splats;
public:
    virtual void setUp();

    void testOrder();         ///< Test that splats in @ref splats are in order.
    void testMinMax();        ///< Test that @c min_value and @c max_value interact correctly
};
CPPUNIT_TEST_SUITE_NAMED_REGISTRATION(TestCompareSplatsMorton, TestSet::perBuild());

static void addSplat(vector<Splat> &splats, float x, float y, float z, float r)
{
    Splat s;
    s.position[0] = x;
    s.position[1] = y;
    s.position[2] = z;
    s.radius = r;
    // Normal and quality are irrelevant - just init to avoid undefined data
    s.normal[0] = 1.0f;
    s.normal[1] = 0.0f;
    s.normal[2] = 0.0f;
    s.quality = 1.0f;
    splats.push_back(s);
}

void TestCompareSplatsMorton::setUp()
{
    addSplat(splats, 0.0f, 0.0f, 0.0f, 1.0f);

    // denorms
    const float dmin = std::numeric_limits<float>::denorm_min();
    addSplat(splats, 4 * dmin, 11 * dmin, 2 * dmin, 1.0f);
    addSplat(splats, 4 * dmin, 11 * dmin, 3 * dmin, 1.0f);
    addSplat(splats, 5 * dmin, 11 * dmin, 2 * dmin, 1.0f);
    addSplat(splats, 5 * dmin, 11 * dmin, 3 * dmin, 1.0f);
    addSplat(splats, 4 * dmin, 12 * dmin, 2 * dmin, 1.0f);

    addSplat(splats, 0.24f, 0.2f, 0.1f, 1.0f);
    addSplat(splats, 1.0f, 0.25f, 0.25f, 1.0f);
    addSplat(splats, 1.25f, 0.25f, 0.25f, 1.0f);
    addSplat(splats, 1.0f, 0.25f, 0.5f, 1.0f);
    addSplat(splats, 1.0f, 0.5f, 0.25f, 1.0f);
    addSplat(splats, 1.0f, 1.0f, 0.25f, 1.0f);

    // Other quadrants
    addSplat(splats, 0.1f, 0.1f, -0.1f, 1.0f);
    addSplat(splats, 0.1f, -0.1f, 0.1f, 1.0f);
    addSplat(splats, 0.1f, -0.1f, -0.1f, 1.0f);
    addSplat(splats, -0.1f, 0.1f, 0.1f, 1.0f);
    addSplat(splats, -0.1f, 0.1f, -0.1f, 1.0f);
    addSplat(splats, -0.1f, -0.1f, 0.1f, 1.0f);
    addSplat(splats, -0.1f, -0.1f, -0.1f, 1.0f);

    addSplat(splats, -2.0f, -256.0f, -8.0f, 1.0f);
    addSplat(splats, -2.0f, -256.0f, -9.0f, 1.0f);
    addSplat(splats, -2.0f, -257.0f, -8.0f, 1.0f);
    addSplat(splats, -2.0f, -257.0f, -9.0f, 1.0f);
    addSplat(splats, -3.0f, -256.0f, -8.0f, 1.0f);
    addSplat(splats, -3.0f, -256.0f, -9.0f, 1.0f);
    addSplat(splats, -3.0f, -257.0f, -9.0f, 1.0f);
    addSplat(splats, -2.0f, -256.0f, -10.0f, 1.0f);
    addSplat(splats, -2.0f, -256.0f, -11.0f, 1.0f);
}

void TestCompareSplatsMorton::testOrder()
{
    for (std::size_t i = 0; i < splats.size(); i++)
        for (std::size_t j = 0; j < splats.size(); j++)
        {
            CPPUNIT_ASSERT_EQUAL(i < j, CompareSplatsMorton()(splats[i], splats[j]));
        }
}

void TestCompareSplatsMorton::testMinMax()
{
    CompareSplatsMorton cmp;
    Splat min = cmp.min_value();
    Splat max = cmp.max_value();
    CPPUNIT_ASSERT(!cmp(min, min));
    CPPUNIT_ASSERT(!cmp(max, max));
    CPPUNIT_ASSERT(cmp(min, max));
    CPPUNIT_ASSERT(!cmp(max, min));

    for (std::size_t i = 0; i < splats.size(); i++)
    {
        CPPUNIT_ASSERT(cmp(min, splats[i]));
        CPPUNIT_ASSERT(!cmp(splats[i], min));
        CPPUNIT_ASSERT(cmp(splats[i], max));
        CPPUNIT_ASSERT(!cmp(max, splats[i]));
    }
}
