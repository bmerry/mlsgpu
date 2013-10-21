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
 * Report information about the build.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <string>
#include "provenance.h"

#ifndef PROVENANCE_VERSION
# error "PROVENANCE_VERSION must be set in the build system"
#endif

#ifndef PROVENANCE_VARIANT
# error "PROVENANCE_VARIANT must be set in the build system"
#endif

std::string provenanceVersion()
{
    return PROVENANCE_VERSION;
}

std::string provenanceVariant()
{
    return PROVENANCE_VARIANT;
}
