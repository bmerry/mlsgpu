/**
 * @file
 * Main program for running non-MPI unit tests.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include "testutil.h"

int main(int argc, const char **argv)
{
    return runTests(argc, argv, false);
}
