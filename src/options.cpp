/**
 * @file
 *
 * Utilities for command-line option processing.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <ostream>
#include <string>
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <limits>
#include "options.h"

static const std::tr1::uint64_t multG = 1024 * 1024 * 1024;
static const std::tr1::uint64_t multM = 1024 * 1024;
static const std::tr1::uint64_t multK = 1024;

void validate(boost::any &v, const std::vector<std::string> &values, Capacity *, int)
{
    using namespace boost::program_options;
    validators::check_first_occurrence(v);
    std::string s = validators::get_single_string(values);
    if (s.empty())
        throw validation_error(validation_error::invalid_option_value);

    std::tr1::uint64_t multiplier = 1;
    bool has_multiplier = true;
    switch (s[s.size() - 1])
    {
    case 'b':
    case 'B': multiplier = 1; break;
    case 'k':
    case 'K': multiplier = multK; break;
    case 'm':
    case 'M': multiplier = multM; break;
    case 'g':
    case 'G': multiplier = multG; break;
    default: has_multiplier = false; break;
    }
    if (has_multiplier)
        s.erase(s.size() - 1, 1);

    std::tr1::uint64_t n = boost::lexical_cast<std::tr1::uint64_t>(s);
    if (std::numeric_limits<std::tr1::uint64_t>::max() / multiplier < n)
        throw validation_error(validation_error::invalid_option_value);
    v = boost::any(Capacity(n * multiplier));
}

std::ostream &operator<<(std::ostream &o, const Capacity &c)
{
    const std::tr1::uint64_t value = std::tr1::uint64_t(c);

    if (value > 0 && value % multG == 0)
        return o << (value / multG) << 'G';
    else if (value > 0 && value % multM == 0)
        return o << (value / multM) << 'M';
    else if (value > 0 && value % multK == 0)
        return o << (value / multK) << 'K';
    else
        return o << value;
}
