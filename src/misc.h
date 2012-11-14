/**
 * @file
 *
 * Miscellaneous small functions.
 */

#ifndef MLSGPU_MISC_H
#define MLSGPU_MISC_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <boost/numeric/conversion/converter.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <limits>
#include "tr1_cstdint.h"
#include <cstring>
#include "errors.h"

/**
 * Multiply @a a and @a b, clamping the result to the maximum value of the type
 * instead of overflowing.
 *
 * @pre @a a and @a b are non-negative.
 */
template<typename T>
static inline T mulSat(T a, T b)
{
    // Tests for >= 0 are written funny for stop GCC from warning when T is unsigned
    MLSGPU_ASSERT(a > 0 || a == 0, std::invalid_argument);
    MLSGPU_ASSERT(b > 0 || b == 0, std::invalid_argument);
    if (a == 0 || std::numeric_limits<T>::max() / a >= b)
        return a * b;
    else
        return std::numeric_limits<T>::max();
}

/**
 * Divide and round up (non-negative values only).
 *
 * @pre @a a &gt;= 0, @a b &gt; 0, and @a a + @a b - 1 does not overflow.
 */
template<typename S, typename T>
static inline S divUp(S a, T b)
{
    // a >= 0 is written funny to stop GCC warning when S is unsigned
    MLSGPU_ASSERT(a > 0 || a == 0, std::invalid_argument);
    MLSGPU_ASSERT(b > 0, std::invalid_argument);
    MLSGPU_ASSERT(a <= std::numeric_limits<S>::max() - S(b - 1), std::out_of_range);
    return (a + b - 1) / b;
}

/**
 * Round up to a multiple of a number.
 *
 * @pre @a a &gt;= 0, @a b &gt; 0, and @a a + @a b - 1 does not overflow.
 */
template<typename S, typename T>
static inline S roundUp(S a, T b)
{
    return divUp(a, b) * b;
}

/**
 * Implementation of @ref divDown for unsigned numerators.
 * Do not call this function directly.
 */
template<typename S, typename T>
static inline S divDown_(S a, T b, boost::true_type)
{
    MLSGPU_ASSERT(b > 0, std::invalid_argument);
    return a / b;
}

/**
 * Implementation of @ref divDown for signed numerators.
 * Do not call this function directly.
 */
template<typename S, typename T>
static inline S divDown_(S a, T b, boost::false_type)
{
    MLSGPU_ASSERT(b > 0, std::invalid_argument);
    MLSGPU_ASSERT(a >= -std::numeric_limits<S>::max(), std::out_of_range);
    if (a < 0)
        return -divUp(-a, b);
    else
        return a / b;
}

/**
 * Divide and round down, handling negative numerator.
 *
 * @pre @a b &gt 0, and @a -a is representable
 */
template<typename S, typename T>
static inline S divDown(S a, T b)
{
    return divDown_(a, b, typename boost::is_unsigned<S>::type());
}

/**
 * Obtain the raw bits from a floating-point number.
 */
static inline std::tr1::uint32_t floatToBits(float x)
{
    std::tr1::uint32_t ans;
    std::memcpy(&ans, &x, sizeof(ans));
    return ans;
}

#endif /* MLSGPU_MISC_H */
