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

#ifndef MINIMLS_LOGGING_H
#define MINIMLS_LOGGING_H

#include <iosfwd>

namespace Log
{

enum Level
{
    debug,
    info,
    warn,
    error
};

namespace detail
{

class LogArray
{
private:
    Level minLevel;
public:
    explicit LogArray(Level minLevel = warn);
    void setLevel(Level minLevel);
    std::ostream &operator[](Level level);
};

} // namespace detail

extern detail::LogArray log;

} // namespace Log

#endif /* MINIMLS_LOGGING_H */
