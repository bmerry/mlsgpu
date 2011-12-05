/**
 * @file
 *
 * Helper functions for robustly inputting numbers.
 */

#ifndef ASCII_IO_H
#define ASCII_IO_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <tr1/cstdint>
#include <tr1/cmath>
#include <limits>
#include <typeinfo>

/**
 * Convert a string to a number (format determined by template).
 *
 * This function is necessary because C++98 is ambiguous about how out-of-range
 * values are handled by stream extraction, and not all compilers set the
 * failbit on out-of-range values. This function has the same semantics as
 * @c boost::lexical_cast, except that out-of-range values are guaranteed to
 * throw, and that uint8_t and int8_t will be treated numerically rather than
 * as character types.
 *
 * Additionally, infinities and NaNs are rejected for the floating-point
 * versions, even if the implementation extracts them (it is somewhat
 * ambiguous in the C++ standard - see e.g.
 * http://gcc.gnu.org/bugzilla/show_bug.cgi?id=27904
 *
 * For implementation reasons, it is only supported for integer types excluding
 * @c uintmax_t and floating-point types.
 *
 * @param s  The string to convert
 * @return @a s converted to type @c T
 * @throw boost::bad_lexical_cast on conversion error.
 */
#if DOXYGEN_FAKE_CODE
template<typename T> T stringToNumber(const std::string &s);
#else

template<typename T>
typename boost::enable_if<boost::is_integral<T>,
    typename boost::disable_if<boost::is_same<T, std::tr1::uintmax_t>, T>::type>::type
stringToNumber(const std::string &s)
{
    BOOST_STATIC_ASSERT(sizeof(T) < sizeof(std::tr1::uintmax_t)
                        || (sizeof(T) == sizeof(std::tr1::uintmax_t)
                            && boost::is_signed<T>::value));

    intmax_t value = boost::lexical_cast<intmax_t>(s);
    if (value < static_cast<intmax_t>(std::numeric_limits<T>::min())
        || value > static_cast<intmax_t>(std::numeric_limits<T>::max()))
    {
        throw boost::bad_lexical_cast(typeid(std::string), typeid(T));
    }
    return static_cast<T>(value);
}

template<typename T>
typename boost::enable_if<boost::is_floating_point<T>, T>::type
stringToNumber(const std::string &s)
{
    T value = boost::lexical_cast<T>(s);
    if (!(std::tr1::isfinite)(value))
    {
        throw boost::bad_lexical_cast(typeid(std::string), typeid(T));
    }
    return value;
}

#endif /* !DOXYGEN_FAKE_CODE */

/**
 * Convert a number to a string representation.
 *
 * This function is expected to round-trip with stringToNumber, including for
 * floating-point types (although it is also implemented for @c uintmax_t).
 * It is largely a wrapper around @c boost::lexical_cast, except that
 * - @c int8_t and @c uint8_t will not be treated as characters
 * - non-finite floating-point values are rejected (since they do not round trip)
 *
 * @param v  The value to format
 * @return A string form of v
 * @throw boost::bad_lexical_cast if @a v is not finite.
 */
#if DOXYGEN_FAKE_CODE
template<typename T> std::string numberToString(T v);
#else

template<typename T>
typename boost::enable_if<boost::is_integral<T>,
    typename boost::disable_if<boost::is_same<signed char, T>,
    typename boost::disable_if<boost::is_same<unsigned char, T>,
    std::string>::type>::type>::type
numberToString(T v)
{
    return boost::lexical_cast<std::string>(v);
}

template<typename T>
typename boost::enable_if<boost::is_floating_point<T>, std::string>::type
numberToString(T v)
{
    if (!(std::tr1::isfinite)(v))
    {
        throw boost::bad_lexical_cast(typeid(T), typeid(std::string));
    }
    return boost::lexical_cast<std::string>(v);
}

template<typename T>
typename boost::enable_if<boost::is_same<signed char, T>, std::string>::type
numberToString(T v)
{
    return boost::lexical_cast<std::string>(static_cast<int>(v));
}

template<typename T>
typename boost::enable_if<boost::is_same<unsigned char, T>, std::string>::type
numberToString(T v)
{
    return boost::lexical_cast<std::string>(static_cast<unsigned int>(v));
}

#endif /* !DOXYGEN_FAKE_CODE */

#endif /* ASCII_IO_H */
