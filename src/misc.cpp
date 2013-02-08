/**
 * @file
 *
 * Miscellaneous small functions.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#define __STDC_LIMIT_MACROS 1
#include "misc.h"
#include "errors.h"
#include <stdexcept>
#include <string>
#include <errno.h>
#include <climits>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/exception/all.hpp>

static boost::filesystem::path tmpFileDir;

DownDivider::DownDivider(std::tr1::uint32_t d)
{
    MLSGPU_ASSERT(d > 0, std::invalid_argument);
    shift = 0;
    std::tr1::int64_t k2 = 1; // 2^shift
    negAdd = INT32_MIN; // never matches
    posAdd = INT32_MAX; // never matches
    while (k2 / d <= INT32_MAX / 2)
    {
        shift++;
        k2 <<= 1;
    }
    if (k2 % d == 0)
    {
        // d is a power of 2
        inverse = 1;
        shift = 0;
        while (1U << shift != d)
            shift++;
    }
    else if (k2 % d <= d / 2)
    {
        inverse = k2 / d;
        posAdd = -1;
    }
    else
    {
        inverse = k2 / d + 1;
        negAdd = -1;
    }
}

void createTmpFile(boost::filesystem::path &path, boost::filesystem::ofstream &out)
{
    path = tmpFileDir;
    if (path.empty())
        path = boost::filesystem::temp_directory_path();
    boost::filesystem::path name = boost::filesystem::unique_path("mlsgpu-tmp-%%%%-%%%%-%%%%-%%%%");
    path /= name; // appends
    out.open(path);
    if (!out)
    {
        int e = errno;
        throw boost::enable_error_info(std::ios::failure("Could not open temporary file"))
            << boost::errinfo_file_name(path.string())
            << boost::errinfo_errno(e);
    }
}

void setTmpFileDir(const boost::filesystem::path &path)
{
    tmpFileDir = path;
}
