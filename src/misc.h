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
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
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
 * Performs precomputations on a 32-bit unsigned number to allow other 32-bit signed values to be
 * divided by it and rounded down.
 *
 * It relies on two's complement arithmetic, and in particular on
 * right-shifting a negative value with sign extension.
 */
class DownDivider
{
private:
    std::tr1::int32_t negAdd;  ///< Add 1 to inputs if they're less than this
    std::tr1::int32_t posAdd;  ///< Add 1 to inputs if they're greater than this
    std::tr1::int32_t inverse;  ///< Multiplier
    int shift;

public:
    typedef std::tr1::int32_t result_type;

    /**
     * Constructor.
     * @param d   Value to divide by
     * @pre @a d != 0
     */
    explicit DownDivider(std::tr1::uint32_t d);

    result_type operator()(std::tr1::int32_t x) const
    {
        std::tr1::int64_t xl = x; // avoids overflow when incrementing
        if (x < negAdd || x > posAdd)
            xl++;
        return (xl * inverse) >> shift;
    }

    std::tr1::int32_t getNegAdd() const { return negAdd; }
    std::tr1::int32_t getPosAdd() const { return posAdd; }
    std::tr1::int32_t getInverse() const { return inverse; }
    std::tr1::int32_t getShift() const { return shift; }
};

/**
 * Obtain the raw bits from a floating-point number.
 */
static inline std::tr1::uint32_t floatToBits(float x)
{
    std::tr1::uint32_t ans;
    std::memcpy(&ans, &x, sizeof(ans));
    return ans;
}

/**
 * Create and open a temporary file. If @ref setTmpFileDir has been called, that
 * directory is used, otherwise it uses the @c boost::filesystem default. The
 * file is opened for output in binary mode.
 *
 * @param[out] path      The path to the temporary file.
 * @param[out] out       The open temporary file.
 * @throw std::ios::failure if the file could not be opened (with boost error
 * info on the filename and errno)
 *
 * @see @ref setTmpFileDir
 */
void createTmpFile(boost::filesystem::path &path, boost::filesystem::ofstream &out);

/**
 * Set the directory to use for temporary files created by @ref createTmpFile.
 */
void setTmpFileDir(const boost::filesystem::path &tmpFileDir);

#endif /* MLSGPU_MISC_H */
