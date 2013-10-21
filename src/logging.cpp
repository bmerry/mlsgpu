/*
 * mlsgpu: surface reconstruction from point clouds
 * Copyright (C) 2013  University of Cape Town
 *
 * This file is part of mlsgpu.
 *
 * mlsgpu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
