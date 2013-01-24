/**
 * @file
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include <climits>
#include <algorithm>
#include <limits>
#include <cmath>
#include <boost/tr1/cmath.hpp>
#include "splat.h"
#include "misc.h"

static const float BIG_MARKER = -1.0f;
static const float MEDIUM_MARKER = -2.0f;
static const float SMALL_MARKER = -3.0f;

/**
 * Determines the highest bit at which two values differ.
 *
 * @retval @c INT_MAX If the sign bits differ (+0 and -0 are thus different)
 * @retval @c INT_MIN If @a a and @a b are bit-wise identical
 * @retval exp If the raw exponent field in the floating-point encoding of the power of two
 *             at which they differ is @a exp
 */
static int compare1d(float a, float b)
{
    std::tr1::uint32_t aBits = floatToBits(a);
    std::tr1::uint32_t bBits = floatToBits(b);

    // Sign bit is the most major key
    std::tr1::uint32_t aSign = aBits >> 31;
    std::tr1::uint32_t bSign = bBits >> 31;
    if (aSign != bSign)
        return INT_MAX;

    std::tr1::int32_t aExp = (aBits >> 23) & 255;
    std::tr1::int32_t bExp = (bBits >> 23) & 255;
    if (aExp != bExp)
        return std::max(aExp, bExp);
    else
    {
        std::tr1::uint32_t mantissaDiff = aBits ^ bBits;
        if (mantissaDiff == 0)
            return INT_MIN;
        return aExp - (__builtin_clz(mantissaDiff) - 8);
    }
}

bool CompareSplatsMorton::operator()(const Splat &a, const Splat &b) const
{
    // We use special radii to indicate min_value and max_value, so we
    // need to check for them here.
    if (a.radius < 0.0f || b.radius < 0.0f)
    {
        float ar = a.radius < 0.0f ? a.radius : MEDIUM_MARKER;
        float br = b.radius < 0.0f ? b.radius : MEDIUM_MARKER;
        return ar < br;
    }

    int dim = 0;
    int bit = INT_MIN;
    for (int i = 0; i < 3; i++)
    {
        int cur = compare1d(a.position[i], b.position[i]);
        if (cur > bit)
        {
            bit = cur;
            dim = i;
        }
    }

    std::tr1::uint32_t aBits = floatToBits(a.position[dim]);
    std::tr1::uint32_t bBits = floatToBits(b.position[dim]);
    return aBits < bBits;
}

Splat CompareSplatsMorton::min_value() const
{
    Splat ans;
    ans.radius = SMALL_MARKER;
    return ans;
}

Splat CompareSplatsMorton::max_value() const
{
    Splat ans;
    ans.radius = BIG_MARKER;
    return ans;
}
