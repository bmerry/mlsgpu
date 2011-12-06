/**
 * @file
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include "files.h"

using namespace std;

InputFile::InputFile() : useCin(true), buffer(cin.rdbuf()), filename("<stdin>") {}

InputFile::InputFile(const std::string &filename)
: useCin(false), buffer(NULL), filename(filename)
{
    auto_ptr<filebuf> fbuffer(new filebuf);
    fbuffer->open(filename.c_str(), ios::in | ios::binary);
    if (!fbuffer->is_open())
        throw ios::failure("Failed to open `" + filename + "' for reading");
    buffer = fbuffer.release();
}

InputFile::~InputFile()
{
    if (!useCin)
        delete buffer;
}

OutputFile::OutputFile() : useCout(true), buffer(cout.rdbuf()), filename("<stdout>") {}

OutputFile::OutputFile(const std::string &filename)
: useCout(false), buffer(NULL), filename(filename)
{
    auto_ptr<filebuf> fbuffer(new filebuf);
    fbuffer->open(filename.c_str(), ios::out | ios::binary);
    if (!fbuffer->is_open())
        throw ios::failure("Failed to open `" + filename + "' for writing");
    buffer = fbuffer.release();
}

OutputFile::~OutputFile()
{
    if (!useCout)
        delete buffer;
    else
        buffer->pubsync();
}


