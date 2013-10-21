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
 * SSE implementation of @ref SplatSet::detail::SplatToBuckets.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include "splat_set_impl.h"

#if BLOBS_USE_SSE2

#include <xmmintrin.h>
#include <emmintrin.h>
#include <limits>
#include "tr1_cstdint.h"
#include "splat.h"
#include "misc.h"

namespace SplatSet
{
namespace detail
{

void SplatToBuckets::divide(
    __m128i in, boost::array<Grid::difference_type, 3> &out) const
{
    // The union is just to force alignment - we never use the vector member
    union
    {
        std::tr1::int32_t v[4];
        __m128i dummy;
    } u;

    __m128i lt = _mm_cmplt_epi32(in, negAdd);
    __m128i gt = _mm_cmpgt_epi32(in, posAdd);
    in = _mm_sub_epi32(in, lt);  // true is encoded as -1, so subtract to add 1
    in = _mm_sub_epi32(in, gt);

    // SSE2 doesn't have a signed 32x32 multiply (and even SSE4.1 doesn't have
    // a convenient one that won't require repacking afterwards), so we drop
    // to scalar at this point.

    _mm_store_si128((__m128i *) u.v, in);
    for (int i = 0; i < 3; i++)
    {
        // cvtps writes INT_MIN on overflow, although we may have added one to it
        if (u.v[i] <= std::numeric_limits<std::tr1::int32_t>::min() + 1)
            throw boost::numeric::bad_numeric_cast();
        std::tr1::int64_t prod = u.v[i] * inverse;
        out[i] = prod >> shift;
    }
}

void SplatToBuckets::operator()(
    const Splat &splat,
    boost::array<Grid::difference_type, 3> &lower,
    boost::array<Grid::difference_type, 3> &upper) const
{
    unsigned int csrOrig = _mm_getcsr();
    unsigned int csrDown = (csrOrig & ~_MM_ROUND_MASK) | _MM_ROUND_DOWN;

    __m128 position = _mm_loadu_ps(splat.position);
    __m128 radius = _mm_load1_ps(&splat.radius);
    __m128 loWorld = _mm_sub_ps(position, radius);
    __m128 hiWorld = _mm_add_ps(position, radius);
    loWorld = _mm_mul_ps(loWorld, invSpacing);
    hiWorld = _mm_mul_ps(hiWorld, invSpacing);
    __m128i loCell, hiCell;

    /* Ideally this would be written with intrinsics instead of inline asm,
     * but several compilers (MSVC and Clang) generate incorrect code because
     * they move the ldmxcsr instructions around relative to others.
     * Since compilers had to be white-listed anyway and GCC was the only
     * compiler we could whitelist (and even then, it may just be good luck,
     * since GCC explicitly does not support #pragma STDC FENV_ACCESS as of
     * 4.7), it seems both safer and no less portable to use GCC asm syntax.
     */
    asm(
        "\tldmxcsr %[csrDown]\n"
        "\tcvtps2dq %[loWorld], %[loCell]\n"
        "\tcvtps2dq %[hiWorld], %[hiCell]\n"
        "\tldmxcsr %[csrOrig]\n"
        : [loCell] "=&x" (loCell),
          [hiCell] "=x" (hiCell)
        : [loWorld] "x" (loWorld),
          [hiWorld] "x" (hiWorld),
          [csrDown] "m" (csrDown),
          [csrOrig] "m" (csrOrig)
    );

    divide(loCell, lower);
    divide(hiCell, upper);
}

SplatToBuckets::SplatToBuckets(float spacing, Grid::size_type bucketSize)
{
    float invSpacing1 = 1.0f / spacing;
    invSpacing = _mm_load1_ps(&invSpacing1);

    DownDivider divider(bucketSize);
    inverse = divider.getInverse();
    shift = divider.getShift();
    std::tr1::int32_t negAdd1 = divider.getNegAdd();
    std::tr1::int32_t posAdd1 = divider.getPosAdd();
    negAdd = _mm_set_epi32(negAdd1, negAdd1, negAdd1, negAdd1);
    posAdd = _mm_set_epi32(posAdd1, posAdd1, posAdd1, posAdd1);
}

} // namespace detail
} // namespace SplatSet

#endif // BLOBS_USE_SSE2
