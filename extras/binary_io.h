/**
 * @file
 *
 * Helper functions for reading and writing binary files to/from streams.
 */

#ifndef BINARY_IO_H
#define BINARY_IO_H

#include "../src/tr1_cstdint.h"
#include <istream>
#include <ostream>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <boost/static_assert.hpp>
#include <limits>

// Forward declarations to keep clang happy
template<typename T, bool b>
static void readBinary(std::istream &in, T &out, const boost::integral_constant<bool, b> &endian);

template<typename T, bool b>
static void writeBinary(std::ostream &out, const T &in, const boost::integral_constant<bool, b> &endian);

namespace detail
{

/**
 * Reads an unsigned little-endian binary integer from a stream.
 */
template<typename T>
static void readBinaryImpl(std::istream &in, T &out, const boost::true_type &, const boost::false_type &, const boost::true_type &)
{
    T value = T();
    uint8_t bytes[sizeof(T)];
    in.read(reinterpret_cast<char *>(bytes), sizeof(T));
    for (int i = sizeof(T) - 1; i >= 0; i--)
    {
        value = (value << 8) | bytes[i];
    }
    out = value;
}

/**
 * Writes an unsigned little-endian binary integer to a stream.
 */
template<typename T>
static void writeBinaryImpl(std::ostream &out, T in, const boost::true_type &, const boost::false_type &, const boost::true_type &)
{
    uint8_t bytes[sizeof(T)];
    for (unsigned int i = 0; i < sizeof(T); i++)
    {
        bytes[i] = in & 0xFF;
        // The complex shift expression is to avoid undefined behavior when
        // shifting an 8-bit value by 8 bits
        in >>= (sizeof(in) == 1 ? 0 : 8);
    }
    out.write(reinterpret_cast<char *>(bytes), sizeof(T));
}

/**
 * Reads an unsigned big-endian binary integer from a stream.
 */
template<typename T>
static void readBinaryImpl(std::istream &in, T &out, const boost::false_type &, const boost::false_type &, const boost::true_type &)
{
    T value = T();
    uint8_t bytes[sizeof(T)];
    in.read(reinterpret_cast<char *>(bytes), sizeof(T));
    for (unsigned int i = 0; i < sizeof(T); i++)
    {
        value = (value << 8) | bytes[i];
    }
    out = value;
}

/**
 * Writes an unsigned big-endian binary integer to a stream.
 */
template<typename T>
static void writeBinaryImpl(std::ostream &out, T in, const boost::false_type &, const boost::false_type &, const boost::true_type &)
{
    uint8_t bytes[sizeof(T)];
    for (int i = sizeof(T) - 1; i >= 0; i--)
    {
        bytes[i] = in & 0xFF;
        // Complex shift expression is to avoid undefined behavior warnings
        // due to shifting an 8-bit value by 8 bits.
        in >>= (sizeof(T) == 1 ? 0 : 8);
    }
    out.write(reinterpret_cast<char *>(bytes), sizeof(T));
}

/**
 * Reads a signed binary integer from a stream.
 *
 * @note The implementation assumes two's complement.
 */
template<typename T, bool b>
static void readBinaryImpl(std::istream &in, T &out, const boost::integral_constant<bool, b> &endian, const boost::true_type &, const boost::true_type &)
{
    typename boost::make_unsigned<T>::type value;
    BOOST_STATIC_ASSERT(sizeof(value) == sizeof(T));
    readBinary(in, value, endian);
    out = static_cast<T>(value);
}

/**
 * Writes a signed binary integer to a stream.
 *
 * @note The implementation assumes two's complement.
 */
template<typename T, bool b>
static void writeBinaryImpl(std::ostream &out, T in, const boost::integral_constant<bool, b> &endian, const boost::true_type &, const boost::true_type &)
{
    const typename boost::make_unsigned<T>::type value = in;
    BOOST_STATIC_ASSERT(sizeof(value) == sizeof(T));
    writeBinary(out, value, endian);
}

/**
 * Reads a binary single-precision floating-point value from a stream.
 */
template<bool b>
static void readBinaryImpl(std::istream &in, float &out, const boost::integral_constant<bool, b> &endian, const boost::false_type &, const boost::false_type &)
{
    BOOST_STATIC_ASSERT(std::numeric_limits<float>::is_iec559);
    BOOST_STATIC_ASSERT(sizeof(float) == 4);
    std::tr1::uint32_t value;
    readBinary(in, value, endian);
    std::memcpy(&out, &value, sizeof(out));
}

/**
 * Writes a binary single-precision floating-point value to a stream.
 */
template<bool b>
static void writeBinaryImpl(std::ostream &out, float in, const boost::integral_constant<bool, b> &endian, const boost::false_type &, const boost::false_type &)
{
    BOOST_STATIC_ASSERT(std::numeric_limits<float>::is_iec559);
    BOOST_STATIC_ASSERT(sizeof(float) == 4);
    std::tr1::uint32_t value;
    std::memcpy(&value, &in, sizeof(in));
    writeBinary(out, value, endian);
}

/**
 * Reads a binary double-precision floating-point value from a stream.
 */
template<bool b>
static void readBinaryImpl(std::istream &in, double &out, const boost::integral_constant<bool, b> &endian, const boost::false_type &, const boost::false_type &)
{
    BOOST_STATIC_ASSERT(std::numeric_limits<double>::is_iec559);
    BOOST_STATIC_ASSERT(sizeof(double) == 8);
    std::tr1::uint64_t value;
    readBinary(in, value, endian);
    std::memcpy(&out, &value, sizeof(out));
}

/**
 * Writes a binary double-precision floating-point value to a stream.
 */
template<bool b>
static void writeBinaryImpl(std::ostream &out, double in, const boost::integral_constant<bool, b> &endian, const boost::false_type &, const boost::false_type &)
{
    BOOST_STATIC_ASSERT(std::numeric_limits<double>::is_iec559);
    BOOST_STATIC_ASSERT(sizeof(double) == 8);
    std::tr1::uint64_t value;
    std::memcpy(&value, &in, sizeof(in));
    writeBinary(out, value, endian);
}

} // namespace detail

/**
 * Reads a binary value from a stream, in either big or little endian.
 * The types supported are integral types, @c float and @c double.
 *
 * @param in      The stream to read from
 * @param out     The value to read.
 * @param endian  An instance of @c boost::true_type for little-endian or @c boost::false_type for big-endian.
 *
 * @note There is no specific error-checking code. Use standard stream
 * functions to check for I/O failure.
 */
template<typename T, bool b>
static void readBinary(std::istream &in, T &out, const boost::integral_constant<bool, b> &endian)
{
    detail::readBinaryImpl(in, out, endian, boost::is_signed<T>(), boost::is_integral<T>());
}

/**
 * writes a binary value to a stream, in either big or little endian.
 * The types supported are integral types, @c float and @c double.
 *
 * @param out     The stream to write to.
 * @param in      The value to write.
 * @param endian  An instance of @c boost::true_type for little-endian or @c boost::false_type for big-endian.
 *
 * @note There is no specific error-checking code. Use standard stream
 * functions to check for I/O failure.
 */
template<typename T, bool b>
static void writeBinary(std::ostream &out, const T &in, const boost::integral_constant<bool, b> &endian)
{
    detail::writeBinaryImpl(out, in, endian, boost::is_signed<T>(), boost::is_integral<T>());
}

#endif /* !BINARY_IO_H */
