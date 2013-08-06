/**
 * @file
 *
 * Concatenate several PLY files containing points.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <memory>
#include <iostream>
#include <limits>
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include "src/fast_ply.h"
#include "src/logging.h"
#include "src/tr1_cstdint.h"
#include "src/splat.h"
#include "src/splat_set.h"

struct OutSplat
{
    float position[3];
    float normal[3];
    float radius;
};

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);

    if (argc <= 1)
    {
        std::cerr << "Usage: plypntcat file1.ply [file2.ply ... ] > output.ply\n";
        return 1;
    }

    SplatSet::FileSet files;
    for (int i = 1; i < argc; i++)
    {
        std::string filename(argv[i]);
        std::auto_ptr<FastPly::Reader> reader(new FastPly::Reader(SYSCALL_READER, filename, 1.0f, std::numeric_limits<float>::infinity()));
        files.addFile(reader.get());
        reader.release();
    }

    const std::size_t bufferSize = 1 << 20;
    std::vector<Splat> buffer(bufferSize);
    std::vector<SplatSet::splat_id> ids(bufferSize);
    std::vector<OutSplat> outBuffer(bufferSize);

    // First count the actual number of splats
    std::auto_ptr<SplatSet::SplatStream> stream(files.makeSplatStream());
    SplatSet::splat_id numSplats = 0;
    std::size_t numRead;
    do
    {
        numRead = stream->read(&buffer[0], &ids[0], bufferSize);
        numSplats += numRead;
    } while (numRead == bufferSize);

    // Now write the splats
    std::cout <<
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex " << numSplats << "\n" <<
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float radius\n"
        "end_header\n";
    stream.reset(files.makeSplatStream());
    do
    {
        numRead = stream->read(&buffer[0], &ids[0], bufferSize);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (std::size_t i = 0; i < numRead; i++)
        {
            std::copy(buffer[i].position, buffer[i].position + 3, outBuffer[i].position);
            std::copy(buffer[i].normal, buffer[i].normal + 3, outBuffer[i].normal);
            outBuffer[i].radius = buffer[i].radius;
        }
        std::cout.write(reinterpret_cast<const char *>(&outBuffer[0]), numRead * sizeof(OutSplat));
    } while (numRead == bufferSize);
}
