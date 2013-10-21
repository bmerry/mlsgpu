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
 * Wrapper header to include either <tt>&lt;tr1/cstdint</tt> or
 * <tt>&lt;cstdint&gt;</tt> depending on what the compiler provides.
 */

#ifndef TR1_CSTDINT_H
#define TR1_CSTDINT_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#if HAVE_TR1_CSTDINT
# include <tr1/cstdint>
#else
# include <cstdint>

/* It seems VC10 doesn't put these in the std::tr1 namespace, so
 * we hack in just the ones we actually use.
 */
#if _MSC_VER
namespace std
{
namespace tr1
{

using std::uint8_t;
using std::int8_t;
using std::uint16_t;
using std::int16_t;
using std::uint32_t;
using std::int32_t;
using std::uint64_t;
using std::int64_t;
using std::uintmax_t;
using std::intmax_t;
using std::uintptr_t;
using std::intptr_t;

}} // end namespace std::tr1
#endif

#endif

#endif /* TR1_CSTDINT_H */
