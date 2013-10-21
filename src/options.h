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
 * Utilities for command-line option processing.
 */

#ifndef OPTIONS_H
#define OPTIONS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <ostream>
#include <string>
#include <vector>
#include <map>
#include <cstddef>
#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include "tr1_cstdint.h"

/**
 * Wraps an enum so that it can be used with @c boost::program_options.
 *
 * The template class must provide the following:
 * - A typedef called @c type for the underlying enum type.
 * - A static method @c getNameMap() that returns an STL map from
 *   option names to values.
 *
 * This class has conversions to and from the enum type.
 */
template<typename EnumWrapper>
class Choice
{
public:
    typedef typename EnumWrapper::type type;

    Choice(type v) : value(v) {}
    Choice &operator =(type v) { value = v; return *this; }
    operator type() const { return value; }

private:
    type value;
};

/**
 * Wraps a 64-bit integer to allow it to be specified with a B/K/M/G suffix
 * to compactly write the value.
 */
class Capacity
{
public:
    Capacity() : value(0) {}
    Capacity(std::tr1::uint64_t value) : value(value) {}
    operator std::tr1::uint64_t() const { return value; }
private:
    std::tr1::uint64_t value;
};

/**
 * Output function for @ref Capacity. It pretty-prints the result
 * using a suitable multiplier suffix.
 */
std::ostream &operator<<(std::ostream &o, const Capacity &c);

/**
 * Output function for @ref Choice. This is mainly useful so that
 * the default value of such an option is pretty-printed.
 *
 * This implementation is not efficient, but it should not need to be.
 */
template<typename EnumWrapper>
std::ostream &operator<<(std::ostream &o, const Choice<EnumWrapper> &c)
{
    typedef typename EnumWrapper::type type;
    const std::map<std::string, type> nameMap = EnumWrapper::getNameMap();
    typename std::map<std::string, type>::const_iterator i;
    /* Search the name map for a reverse entry */
    for (i = nameMap.begin(); i != nameMap.end(); ++i)
    {
        if (i->second == (type) c)
        {
            o << i->first;
            return o;
        }
    }
    o << (type) c;
    return o;
}

/**
 * Validator for @c boost::program_options for the @ref Choice class.
 *
 * For reasons I don't fully understand, this does not work if it
 * is placed in the @c boost::program_options namespace. Argument-dependent
 * lookup seems to find it just fine in the root namespace.
 */
template<typename EnumWrapper>
void validate(boost::any &v, const std::vector<std::string> &values,
              Choice<EnumWrapper> *, int)
{
    using namespace boost::program_options;

    typedef typename EnumWrapper::type type;
    validators::check_first_occurrence(v);
    const std::string &s = validators::get_single_string(values);
    const std::map<std::string, type> &nameMap = EnumWrapper::getNameMap();
    typename std::map<std::string, type>::const_iterator pos = nameMap.find(s);
    if (pos == nameMap.end())
        throw validation_error(validation_error::invalid_option_value);
    else
        v = boost::any(Choice<EnumWrapper>(pos->second));
}

void validate(boost::any &v, const std::vector<std::string> &values, Capacity *, int);

#endif /* !OPTIONS_H */
