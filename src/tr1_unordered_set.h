/**
 * @file 
 *
 * Wrapper header to include either <tr1/unordered_set> or <unordered_set>
 * depending on what the compiler provides.
 */

#ifndef TR1_UNORDERED_SET_H
#define TR1_UNORDERED_SET_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#if HAVE_TR1_UNORDERED_SET
# include <tr1/unordered_set>
#else
# include <unordered_set>
#endif

#endif /* TR1_UNORDERED_SET_H */

