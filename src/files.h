/**
 * @file
 */

#ifndef FILES_H
#define FILES_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/noncopyable.hpp>
#include <streambuf>
#include <string>

/**
 * Metadata about an input file.
 */
struct InputFile : public boost::noncopyable
{
    bool useCin;
    std::streambuf *buffer;
    std::string filename;

    InputFile();
    InputFile(const std::string &filename);
    ~InputFile();
};

/**
 * Metadata about an output file.
 */
struct OutputFile : public boost::noncopyable
{
    bool useCout;
    std::streambuf *buffer;
    std::string filename;

    OutputFile();
    OutputFile(const std::string &filename);
    ~OutputFile();
};

#endif /* !FILES_H */
