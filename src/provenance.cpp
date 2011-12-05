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
