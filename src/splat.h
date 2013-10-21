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
 */

#ifndef SPLAT_H
#define SPLAT_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include <limits>
#include <cmath>
#include <boost/tr1/cmath.hpp>
#include <boost/numeric/conversion/cast.hpp>

/**
 * A point together with a normal and a radius of influence.
 */
struct Splat
{
    float position[3];
    float radius;
    float normal[3];
    float quality;

    /**
     * Checks whether all attributes are finite.
     */
    inline bool isFinite() const
    {
        return (std::tr1::isfinite)(position[0])
            && (std::tr1::isfinite)(position[1])
            && (std::tr1::isfinite)(position[2])
            && (std::tr1::isfinite)(radius)
            && (std::tr1::isfinite)(normal[0])
            && (std::tr1::isfinite)(normal[1])
            && (std::tr1::isfinite)(normal[2])
            && (std::tr1::isfinite)(quality);
    }
};

#endif /* !SPLAT_H */
