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
 * Error and exception utilities.
 */

#ifndef MLSGPU_ERRORS_H
#define MLSGPU_ERRORS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cstddef>
#include <stdexcept>
#include <cstdlib>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/utility/enable_if.hpp>

#define MLSGPU_STRINGIZE(x) #x
#define MLSGPU_ASSERT_PASTE(file, line, msg) (file ":" MLSGPU_STRINGIZE(line) ": " msg)

#define MLSGPU_ASSERT_IMPL(cond, exception_type) \
    ((cond) ? ((void) 0) : (mlsgpuThrow<exception_type>(__FILE__, __LINE__, MLSGPU_ASSERT_PASTE(__FILE__, __LINE__, #cond))))

#define MLSGPU_UNUSED(x) (false ? (void) (x) : (void) 0)

template<typename ExceptionType>
static void mlsgpuThrow(const char *filename, int line, const char *msg,
                        typename boost::enable_if<boost::is_base_of<std::logic_error, ExceptionType> >::type *dummy = NULL)
    throw(ExceptionType)
{
    MLSGPU_UNUSED(dummy);
#if MLSGPU_ASSERT_ABORT
    std::cerr << filename << ':' << line << ": " << msg << endl;
    std::abort();
#else
    MLSGPU_UNUSED(filename);
    MLSGPU_UNUSED(line);
    throw ExceptionType(msg);
#endif
}

/**
 * A method was called on an object when it was not in a valid state to do so.
 */
class state_error : public std::logic_error
{
public:
    explicit state_error(const std::string &msg) : std::logic_error(msg) {}
};

/**
 * A command-line option is semantically invalid.
 */
class invalid_option : public std::runtime_error
{
public:
    explicit invalid_option(const std::string &msg) : std::runtime_error(msg) {}
};

#endif /* !MLSGPU_ERRORS_H */

/* This part is outside the include guard so that it can be redefined after resetting
 * NDEBUG.
 */

#undef MLSGPU_ASSERT
#if defined(NDEBUG)
# define MLSGPU_ASSERT(cond, except_type) (false ? MLSGPU_ASSERT_IMPL(cond, except_type) : (void) 0)
#else
# define MLSGPU_ASSERT(cond, except_type) MLSGPU_ASSERT_IMPL(cond, except_type)
#endif
