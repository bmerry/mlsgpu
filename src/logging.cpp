/**
 * @file
 *
 * Information logging support.
 */

#include <ostream>
#include <iostream>
#include <cassert>
#include <boost/iostreams/device/null.hpp>
#include <boost/iostreams/stream.hpp>
#include "logging.h"

using namespace std;

namespace Log
{

namespace detail
{

LogArray::LogArray(Level minLevel) : minLevel(minLevel) {}

ostream &LogArray::operator[](Level level)
{
    static boost::iostreams::null_sink nullSink;
    static boost::iostreams::stream<boost::iostreams::null_sink> nullStream(nullSink);
    if (level >= minLevel)
        return cerr;
    else
        return nullStream;
}

void LogArray::setLevel(Level minLevel)
{
    this->minLevel = minLevel;
}

} // namespace detail

detail::LogArray log;

} // namespace Log
