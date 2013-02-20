/**
 * @file
 *
 * Binary file I/O classes, including some with thread-safe absolute positioning.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#if (HAVE_PREAD || HAVE_PWRITE) && !defined(_POSIX_C_SOURCE)
# define _POSIX_C_SOURCE 200809L
#endif
#include <cstddef>
#include <limits>
#include <string>
#include <stdexcept>
#include <fstream>
#include <cerrno>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/exception/all.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include "errors.h"
#include "binary_io.h"

#if SYSCALL_READER_POSIX
# include <fcntl.h>
# include <sys/types.h>
# include <sys/stat.h>
#endif

#if SYSCALL_READER_WIN32
# include <windows.h>
#endif

BinaryIO::~BinaryIO()
{
    if (!isOpen_)
        close();
}

void BinaryIO::open(const boost::filesystem::path &filename)
{
    MLSGPU_ASSERT(!isOpen_, state_error);

    std::string filenameStr = filename.string();
    try
    {
        openImpl(filename);
    }
    catch (boost::exception &e)
    {
        e << boost::errinfo_file_name(filenameStr);
        throw;
    }
    isOpen_ = true;
    filename_ = filenameStr;
}

void BinaryIO::close()
{
    MLSGPU_ASSERT(isOpen_, state_error);
    isOpen_ = false; // this is done first so that we don't try to close again on failure
    try
    {
        closeImpl();
    }
    catch (boost::exception &e)
    {
        e << boost::errinfo_file_name(filename_);
        throw;
    }
    filename_.clear();
}

bool BinaryIO::isOpen() const
{
    return isOpen_;
}

const std::string &BinaryIO::filename() const
{
    return filename_;
}

std::size_t BinaryReader::read(void *buf, std::size_t count, offset_type offset) const
{
    MLSGPU_ASSERT(isOpen(), state_error);
    try
    {
        return readImpl(buf, count, offset);
    }
    catch (boost::exception &e)
    {
        e << boost::errinfo_file_name(filename());
        throw;
    }
}

std::size_t BinaryWriter::write(const void *buf, std::size_t count, offset_type offset) const
{
    MLSGPU_ASSERT(isOpen(), state_error);
    try
    {
        return writeImpl(buf, count, offset);
    }
    catch (boost::exception &e)
    {
        e << boost::errinfo_file_name(filename());
        throw;
    }
}

namespace
{

class StreamReader : public BinaryReader
{
private:
    mutable boost::mutex mutex;
    mutable boost::filesystem::filebuf fb;

    virtual void openImpl(const boost::filesystem::path &path);
    virtual void closeImpl();
    virtual std::size_t readImpl(void *buf, std::size_t count, offset_type offset) const;
};

class StreamWriter : public BinaryWriter
{
private:
    mutable boost::mutex mutex;
    mutable boost::filesystem::filebuf fb;

    virtual void openImpl(const boost::filesystem::path &path);
    virtual void closeImpl();
    virtual std::size_t writeImpl(const void *buf, std::size_t count, offset_type offset) const;
};

void StreamReader::openImpl(const boost::filesystem::path &path)
{
    if (!fb.open(path, std::ios::in | std::ios::binary))
    {
        throw boost::enable_error_info(std::ios::failure("Open failed"))
            << boost::errinfo_errno(errno);
    }
}

void StreamWriter::openImpl(const boost::filesystem::path &path)
{
    if (!fb.open(path, std::ios::out | std::ios::binary | std::ios::trunc))
    {
        throw boost::enable_error_info(std::ios::failure("Open failed"))
            << boost::errinfo_errno(errno);
    }
}

void StreamReader::closeImpl()
{
    if (!fb.close())
        throw boost::enable_error_info(std::ios::failure("Close failed"))
            << boost::errinfo_errno(errno);
}

void StreamWriter::closeImpl()
{
    if (!fb.close())
        throw boost::enable_error_info(std::ios::failure("Close failed"))
            << boost::errinfo_errno(errno);
}

std::size_t StreamReader::readImpl(void *buf, std::size_t count, offset_type offset) const
{
    boost::unique_lock<boost::mutex> lock(mutex);
    std::streampos newpos = fb.pubseekpos(offset, std::ios_base::in);
    if (newpos == -1)
        throw boost::enable_error_info(std::ios::failure("Seek failed"));
    return fb.sgetn((char *) buf, count);
}

std::size_t StreamWriter::writeImpl(const void *buf, std::size_t count, offset_type offset) const
{
    boost::unique_lock<boost::mutex> lock(mutex);
    std::streampos newpos = fb.pubseekpos(offset, std::ios_base::out);
    if (newpos == -1)
        throw boost::enable_error_info(std::ios::failure("Seek failed"));
    return fb.sputn((const char *) buf, count);
};

class MmapReader : public BinaryReader
{
private:
    boost::iostreams::mapped_file_source mapping;

    virtual void openImpl(const boost::filesystem::path &path);
    virtual void closeImpl();
    virtual std::size_t readImpl(void *buf, std::size_t count, offset_type offset) const;
};

void MmapReader::openImpl(const boost::filesystem::path &path)
{
    mapping.open(path.string());
    if (!mapping.is_open())
    {
        throw boost::enable_error_info(std::ios::failure("Could not create mapping"))
            << boost::errinfo_errno(errno);
    }
}

void MmapReader::closeImpl()
{
    mapping.close();
}

std::size_t MmapReader::readImpl(void *buf, std::size_t count, offset_type offset) const
{
    if (offset >= mapping.size())
        return 0; // entire read is beyond end of file
    else if (count > mapping.size() - offset)
        count = mapping.size() - offset;  // clip at EOF

    std::memcpy(buf, mapping.data() + offset, count);
    return count;
}

class SyscallReader : public BinaryReader
{
private:
#if SYSCALL_READER_POSIX
    int fd;
#elif SYSCALL_READER_WIN32
    HANDLE fd;
#endif

    virtual void openImpl(const boost::filesystem::path &path);
    virtual void closeImpl();
    virtual std::size_t readImpl(void *buf, std::size_t count, offset_type offset) const;
};

class SyscallWriter : public BinaryWriter
{
private:
#if SYSCALL_READER_POSIX
    int fd;
#elif SYSCALL_READER_WIN32
    HANDLE fd;
#endif

    virtual void openImpl(const boost::filesystem::path &path);
    virtual void closeImpl();
    virtual std::size_t writeImpl(const void *buf, std::size_t count, offset_type offset) const;
};

#if SYSCALL_READER_POSIX

void SyscallReader::openImpl(const boost::filesystem::path &path)
{
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
    {
        throw boost::enable_error_info(std::ios::failure("Could not open file"))
            << boost::errinfo_errno(errno);
    }
}

void SyscallWriter::openImpl(const boost::filesystem::path &path)
{
    fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC);
    if (fd < 0)
    {
        throw boost::enable_error_info(std::ios::failure("Could not open file"))
            << boost::errinfo_errno(errno);
    }
}

void SyscallReader::closeImpl()
{
    if (::close(fd) != 0)
        throw boost::enable_error_info(std::ios::failure("Could not close file"))
            << boost::errinfo_errno(errno);
}

void SyscallWriter::closeImpl()
{
    if (::close(fd) != 0)
        throw boost::enable_error_info(std::ios::failure("Could not close file"))
            << boost::errinfo_errno(errno);
}

std::size_t SyscallReader::readImpl(void *buf, size_t count, offset_type offset) const
{
    size_t remain = count;
    while (remain > 0)
    {
        ssize_t bytes = ::pread(fd, buf, remain, offset);
        if (bytes < 0)
        {
            if (errno == EAGAIN || errno == EINTR)
                continue;
            throw boost::enable_error_info(std::ios::failure("read failed"))
                << boost::errinfo_errno(errno);
        }
        else if (bytes == 0)
        {
            return count - remain;
        }
        else
        {
            buf = (void *) ((char *) buf + bytes);
            offset += bytes;
            remain -= bytes;
        }
    }
}

std::size_t SyscallReader::writeImpl(void *buf, size_t count, offset_type offset) const
{
    size_t remain = count;
    while (remain > 0)
    {
        ssize_t bytes = ::pwrite(fd, buf, remain, offset);
        if (bytes < 0)
        {
            if (errno == EAGAIN || errno == EINTR)
                continue;
            throw boost::enable_error_info(std::ios::failure("write failed"))
                << boost::errinfo_errno(errno);
        }
        else if (bytes == 0)
        {
            throw boost::enable_error_info(std::ios::failure("pwrite did not write any bytes"));
        }
        else
        {
            buf = (const void *) ((const char *) buf + bytes);
            offset += bytes;
            remain -= bytes;
        }
    }
    return count;
}

#endif // SYSCALL_READER_POSIX

#if SYSCALL_READER_WIN32

void SyscallReader::openImpl(const boost::filesystem::path &path)
{
    fd = CreateFile(path.c_str(),
                    GENERIC_READ,
                    FILE_SHARE_READ,
                    NULL,
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL,
                    NULL);
    if (fd == INVALID_HANDLE_VALUE)
    {
        throw boost::enable_error_info(std::ios::failure("Could not open file"))
            << boost::errinfo_errno(GetLastError());
    }
}

void SyscallWriter::openImpl(const boost::filesystem::path &path)
{
    fd = CreateFile(path.c_str(),
                    GENERIC_WRITE,
                    0,
                    NULL,
                    CREATE_ALWAYS,
                    FILE_ATTRIBUTE_NORMAL,
                    NULL);
    if (fd == INVALID_HANDLE_VALUE)
    {
        throw boost::enable_error_info(std::ios::failure("Could not open file"))
            << boost::errinfo_errno(GetLastError());
    }
}

void SyscallReader::closeImpl()
{
    if (!CloseHandle(fd))
        throw boost::enable_error_info(std::ios::failure("Could not close file"))
            << boost::errinfo_errno(GetLastError());
}

void SyscallWriter::closeImpl()
{
    if (!CloseHandle(fd))
        throw boost::enable_error_info(std::ios::failure("Could not close file"))
            << boost::errinfo_errno(GetLastError());
}

std::size_t SyscallReader::readImpl(void *buf, size_t count, offset_type offset) const
{
    std::size_t remain = count;
    while (remain > 0)
    {
        OVERLAPPED req;
        DWORD bytes;
        std::size_t bytesToRead;

        std::memset(&req, 0, sizeof(req));
        req.Offset = offset & 0xFFFFFFFFu;
        req.OffsetHigh = offset >> 32;

        bytesToRead = std::min(remain, std::size_t(std::numeric_limits<DWORD>::max()));
        if (!ReadFile(fd, buf, bytesToRead, &bytes, &req))
        {
            throw boost::enable_error_info(std::ios::failure("read failed"))
                << boost::errinfo_errno(GetLastError());
        }
        else if (bytes == 0)
        {
            return count - remain;
        }
        else
        {
            buf = (void *) ((char *) buf + bytes);
            offset += bytes;
            remain -= bytes;
        }
    }
    return count;
}

std::size_t SyscallReader::write(const void *buf, size_t count, offset_type offset) const
{
    std::size_t remain = count;
    while (remain > 0)
    {
        OVERLAPPED req;
        DWORD bytes;
        std::size_t bytesToWrite;

        std::memset(&req, 0, sizeof(req));
        req.Offset = offset & 0xFFFFFFFFu;
        req.OffsetHigh = offset >> 32;

        bytesToWrite = std::min(remain, std::size_t(std::numeric_limits<DWORD>::max()));
        if (!WriteFile(fd, buf, bytesToWrite, &bytes, &req))
        {
            throw boost::enable_error_info(std::ios::failure("write failed"))
                << boost::errinfo_errno(GetLastError());
        }
        else if (bytes == 0)
        {
            throw boost::enable_error_info(std::ios::failure("WriteFile wrote zero bytes"));
        }
        else
        {
            buf = (const void *) ((const char *) buf + bytes);
            offset += bytes;
            remain -= bytes;
        }
    }
    return count;
}

#endif // SYSCALL_READER_WIN32

} // anonymous namespace

BinaryReader *createReader(ReaderType type)
{
    switch (type)
    {
    case MMAP_READER:    return new MmapReader;
    case STREAM_READER:  return new StreamReader;
    case SYSCALL_READER: return new SyscallReader;
    default:
        MLSGPU_ASSERT(false, std::invalid_argument);
        return NULL;
    }
}

BinaryWriter *createWriter(WriterType type)
{
    switch (type)
    {
    case STREAM_WRITER:  return new StreamWriter;
    case SYSCALL_WRITER: return new SyscallWriter;
    default:
        MLSGPU_ASSERT(false, std::invalid_argument);
        return NULL;
    }
}
