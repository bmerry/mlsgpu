/**
 * @file
 *
 * Miscellaneous small functions.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include "misc.h"
#include <stdexcept>
#include <string>
#include <errno.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/exception/all.hpp>

static boost::filesystem::path tmpFileDir;

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
