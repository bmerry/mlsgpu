/**
 * @file
 *
 * A thread-safe progress meter, modelled on boost::progress_display.
 */

#ifndef PROGRESS_H
#define PROGRESS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <ostream>
#include <iostream>
#include <string>
#include <boost/thread/mutex.hpp>
#include <tr1/cstdint>
#include <boost/noncopyable.hpp>

/**
 * A thread-safe progress meter. It supports 64-bit progress values.
 */
class ProgressDisplay : public boost::noncopyable
{
public:
    typedef std::tr1::uintmax_t size_type;

    explicit ProgressDisplay(size_type total,
                             std::ostream &os = std::cout,
                             const std::string &s1 = "\n",
                             const std::string &s2 = "",
                             const std::string &s3 = "");

    void restart(size_type total);
    size_type operator++();
    size_type operator+=(std::tr1::uintmax_t increment);

    size_type count() const;
    size_type expected_count() const;

private:
    size_type current;
    unsigned int ticsShown;
    size_type nextTic;

    size_type total;
    size_type totalQ;
    size_type totalR;

    mutable boost::mutex mutex;
    std::ostream &os;
    const std::string s1, s2, s3;

    static const unsigned int totalTics = 51;

    void updateNextTic();
};

#endif /* !PROGRESS_H */
