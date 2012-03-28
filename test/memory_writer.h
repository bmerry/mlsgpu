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

#include <utility>
#include <vector>
#include <string>
#include <tr1/cstdint>
#include <tr1/unordered_map>
#include <boost/array.hpp>
#include "../src/fast_ply.h"

/**
 * An implementation of the @ref FastPly::WriterBase
 * interface that does not actually write to file, but merely saves
 * a copy of the data in memory. It is aimed specifically at testing.
 *
 * Since a writer can be opened and closed multiple times, the results
 * are stored in a map indexed by the provided filename.
 */
class MemoryWriter : public FastPly::WriterBase
{
public:
    struct Output
    {
        std::vector<boost::array<float, 3> > vertices;
        std::vector<boost::array<std::tr1::uint32_t, 3> > triangles;
    };

    /// Constructor
    MemoryWriter();

    virtual void open(const std::string &filename);
    virtual std::pair<char *, size_type> open();
    virtual void close();
    virtual void writeVertices(size_type first, size_type count, const float *data);
    virtual void writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data);
    virtual bool supportsOutOfOrder() const;

    /**
     * Returns a previously written output. It is legal to retrieve an output that is in
     * progress.
     *
     * @param filename               The filename provided when the writer was opened.
     * @throw std::invalid_argument  if no such output exists.
     */
    const Output &getOutput(const std::string &filename) const;

private:
    /**
     * Output file currently being written. It is @c NULL when the writer is closed.
     */
    Output *curOutput;

    /**
     * Outputs organised by filename.
     */
    std::tr1::unordered_map<std::string, Output> outputs;
};

#endif /* !MEMORY_WRITER_H */
