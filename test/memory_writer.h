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
#include <boost/array.hpp>
#include "../src/fast_ply.h"

/**
 * An implementation of the @ref FastPly::WriterBase
 * interface that does not actually write to file, but merely saves
 * a copy of the data in memory. It is aimed specifically at testing.
 */
class MemoryWriter : public FastPly::WriterBase
{
public:
    /// Constructor
    MemoryWriter();

    virtual void open(const std::string &filename);
    virtual std::pair<char *, size_type> open();
    virtual void close();
    virtual void writeVertices(size_type first, size_type count, const float *data);
    virtual void writeTriangles(size_type first, size_type count, const std::tr1::uint32_t *data);
    virtual bool supportsOutOfOrder() const;

    const std::vector<boost::array<float, 3> > &getVertices() const { return vertices; }
    const std::vector<boost::array<std::tr1::uint32_t, 3> > &getTriangles() const { return triangles; }

private:
    std::vector<boost::array<float, 3> > vertices;
    std::vector<boost::array<std::tr1::uint32_t, 3> > triangles;
};

#endif /* !MEMORY_WRITER_H */
