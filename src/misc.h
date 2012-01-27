/**
 * @file
 *
 * Miscellaneous small functions.
 */

#ifndef MLSGPU_MISC_H
#define MLSGPU_MISC_H

#include <boost/numeric/conversion/converter.hpp>
#include <limits>
#include "errors.h"

typedef boost::numeric::converter<
    int,
    float,
    boost::numeric::conversion_traits<int, float>,
    boost::numeric::def_overflow_handler,
    boost::numeric::Ceil<float> > RoundUp;
typedef boost::numeric::converter<
    int,
    float,
    boost::numeric::conversion_traits<int, float>,
    boost::numeric::def_overflow_handler,
    boost::numeric::Floor<float> > RoundDown;

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
 * Divide and round up.
 *
 * @pre @a a &gt;= 0, @a b &gt; 0, and @a a + @a b - 1 does not overflow.
 */
template<typename S, typename T>
static inline S divUp(S a, T b)
{
    // a >= 0 is written funny to stop GCC warning when S is unsigned
    MLSGPU_ASSERT(a > 0 || a == 0, std::invalid_argument);
    MLSGPU_ASSERT(b > 0, std::invalid_argument);
    MLSGPU_ASSERT(a <= std::numeric_limits<S>::max() - (b - 1), std::overflow_error);
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

#endif /* MLSGPU_MISC_H */
