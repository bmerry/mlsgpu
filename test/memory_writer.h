/**
 * @file
 *
 * Writer class that stores results in memory for easy testing.
 */

#ifndef MEMORY_WRITER_H
#define MEMORY_WRITER_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <string>
#include <vector>
#include <boost/filesystem/path.hpp>
#include <boost/array.hpp>
#include "../src/tr1_cstdint.h"
#include "../src/tr1_unordered_map.h"
#include "../src/binary_io.h"
#include "../src/fast_ply.h"

/**
 * An implementation of the @ref BinaryWriter
 * interface that does not actually write to file, but merely saves
 * a copy of the data in memory. It is aimed specifically at testing.
 *
 * Since a writer can be opened and closed multiple times, the results
 * are stored in a map indexed by the provided filename.
 */
class MemoryWriter : public BinaryWriter
{
public:
    /// Constructor
    MemoryWriter();

    virtual void openImpl(const boost::filesystem::path &filename);
    virtual void closeImpl();
    virtual std::size_t writeImpl(const void *buffer, std::size_t count, offset_type offset) const;
    virtual void resizeImpl(offset_type size) const;

    /**
     * Returns a previously written output. It is legal to retrieve an output that is in
     * progress.
     *
     * @param filename               The filename provided when the writer was opened.
     * @throw std::invalid_argument  if no such output exists.
     */
    const std::string &getOutput(const std::string &filename) const;

private:
    /**
     * Output file currently being written. It is @c NULL when the writer is closed.
     */
    std::string *curOutput;

    /**
     * Outputs organised by filename.
     */
    std::tr1::unordered_map<std::string, std::string> outputs;
};

/**
 * Combined a @ref MemoryWriter with a @ref FastPly::Writer, with additional
 * helper routines to decode output.
 */
class MemoryWriterPly : public FastPly::Writer
{
public:
    MemoryWriterPly();

    /**
     * Quick-and-dirty extraction of the vertices and triangles from the PLY file.
     * It's a @em long way from being a full PLY parser. It handles only the sorts
     * of files produced by FastPly::Writer.
     *
     * @param text           File contents
     * @param[out] vertices  Vertices
     * @param[out] triangles Triangles
     */
    static void parse(
        const std::string &content,
        std::vector<boost::array<float, 3> > &vertices,
        std::vector<boost::array<std::tr1::uint32_t, 3> > &triangles);

    /**
     * Wraps @ref MemoryWriter::getOutput.
     */
    const std::string &getOutput(const std::string &filename);
};

#endif /* !MEMORY_WRITER_H */
