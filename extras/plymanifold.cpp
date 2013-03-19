/**
 * @file
 *
 * Program to check whether a PLY file is valid and contains a manifold surface.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <boost/array.hpp>
#include "../test/manifold.h"
#include "../src/logging.h"
#include "ply.h"

using namespace std;

/// Builder for @ref PLY::Reader that reads vertices (just position)
class VertexBuilder
#ifdef DOXYGEN_FAKE_CODE
: public PLY::Builder
#endif
{
public:
    typedef boost::array<float, 3> Element;

    template<typename Iterator>
    void setProperty(const string &name, Iterator first, Iterator last)
    {
        (void) name;
        (void) first;
        (void) last;
    }

    template<typename T>
    void setProperty(const std::string &name, const T &value)
    {
        if (name == "x" || name == "y" || name == "z")
            current[name[0] - 'x'] = value;
    }

    Element create()
    {
        return current;
    }

    static void validateProperties(const PLY::PropertyTypeSet &properties)
    {
        const string fields[] = {"x", "y", "z"};
        PLY::PropertyTypeSet::index<PLY::Name>::type::const_iterator p;
        for (int i = 0; i < 3; i++)
        {
            p = properties.get<PLY::Name>().find(fields[i]);
            if (p == properties.get<PLY::Name>().end())
            {
                throw PLY::FormatError("Missing property " + fields[i]);
            }
            else if (p->isList)
                throw PLY::FormatError("Property " + fields[i] + " should not be a list");
        }
    }

private:
    Element current;
};

/// Builder for @ref PLY::Reader that reads triangles
class TriangleBuilder
#ifdef DOXYGEN_FAKE_CODE
: public PLY::Builder
#endif
{
public:
    typedef boost::array<std::tr1::uint32_t, 3> Element;

    template<typename Iterator>
    void setProperty(const string &name, Iterator first, Iterator last)
    {
        if (name == "vertex_indices")
        {
            if (distance(first, last) != 3)
                throw PLY::FormatError("Face does not contain 3 vertices");
            Iterator c = first;
            for (unsigned int i = 0; i < 3; i++, c++)
            {
                // Roundabout way to write c < 0 so that it doesn't warn for unsigned types
                if (!(*c == 0 || *c > 0))
                    throw PLY::FormatError("Negative or out-of-range index");
                std::tr1::uint32_t idx = *c;
                current[i] = idx;
            }
        }
    }

    template<typename T>
    void setProperty(const std::string &name, const T &value)
    {
        (void) name;
        (void) value;
    }

    Element create()
    {
        return current;
    }

    static void validateProperties(const PLY::PropertyTypeSet &properties)
    {
        PLY::PropertyTypeSet::index<PLY::Name>::type::const_iterator p;
        p = properties.get<PLY::Name>().find("vertex_indices");
        if (p == properties.get<PLY::Name>().end())
        {
            throw PLY::FormatError("Missing property vertex_indices");
        }
        else if (!p->isList)
            throw PLY::FormatError("Property vertex_indices should be a list");
        else if (p->valueType != PLY::INT8
                 && p->valueType != PLY::UINT8
                 && p->valueType != PLY::INT16
                 && p->valueType != PLY::UINT16
                 && p->valueType != PLY::INT32
                 && p->valueType != PLY::UINT32)
            throw PLY::FormatError("Property vertex_indices should have integral type");
    }

private:
    Element current;
};

/// Checks that vertex coordinates are all finite
template<typename Iterator>
static void validateVertices(Iterator first, Iterator last)
{
    std::tr1::uint64_t id = 0;
    for (Iterator i = first; i != last; ++i, ++id)
    {
        const VertexBuilder::Element &e = *i;
        for (int j = 0; j < 3; j++)
            if (!std::tr1::isfinite(e[j]))
            {
                cout << "Vertex " << id << " has non-finite value\n";
                exit(1);
            }
    }
}

int main(int argc, const char **argv)
{
    if (argc != 2)
    {
        cerr << "Usage: plymanifold file.ply\n";
        return 2;
    }

    const char *filename = argv[1];
    try
    {
        filebuf in;
        in.open(filename, ios::in);
        if (!in.is_open())
        {
            cerr << "Could not open " << filename << "\n";
            return 1;
        }
        PLY::Reader reader(&in);
        reader.addBuilder("vertex", VertexBuilder());
        reader.addBuilder("face", TriangleBuilder());
        reader.readHeader();
        PLY::ElementRangeReader<VertexBuilder> &vertexReader = reader.skipTo<VertexBuilder>("vertex");
        size_t numVertices = vertexReader.getNumber();
        validateVertices(vertexReader.begin(), vertexReader.end());
        PLY::ElementRangeReader<TriangleBuilder> &triangleReader = reader.skipTo<TriangleBuilder>("face");
        Manifold::Metadata metadata;
        string reason = Manifold::isManifold(numVertices, triangleReader.begin(), triangleReader.end(), &metadata);
        if (reason != "")
        {
            cout << "Mesh is not manifold: " << reason << "\n";
            return 1;
        }
        else
        {
            cout << "Mesh is manifold."
                << "\nVertices: " << metadata.numVertices
                << "\nTriangles: " << metadata.numTriangles
                << "\nComponents: " << metadata.numComponents
                << "\nBoundaries: " << metadata.numBoundaries << endl;
        }
    }
    catch (ios::failure &e)
    {
        cerr << "Failed to read " << filename << ": " << e.what() << "\n";
        exit(1);
    }
    catch (PLY::FormatError &e)
    {
        cerr << "Failed to read " << filename << ": " << e.what() << "\n";
        exit(1);
    }
    return 0;
}
