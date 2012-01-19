/**
 * @file
 *
 * Miscellaneous helper functions.
 */

#ifndef MISC_H
#define MISC_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <ostream>
#include <string>
#include <vector>
#include <map>
#include <boost/any.hpp>
#include <boost/program_options.hpp>

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

#endif /* !MISC_H */
