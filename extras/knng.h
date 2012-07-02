#ifndef EXTRAS_KNNG_H
#define EXTRAS_KNNG_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <vector>
#include <utility>
#include "../src/allocator.h"
#include "../src/splat.h"
#include "../src/tr1_cstdint.h"

std::vector<std::vector<std::pair<float, std::tr1::uint32_t> > > knng(const Statistics::Container::vector<Splat> &splats, int K, float maxDistanceSquared);

#endif /* !EXTRAS_KNNG_H */
