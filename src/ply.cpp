/**
 * @file
 *
 * Point cloud support for the Stanford PLY format.
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <vector>
#include <string>
#include <map>
#include <istream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/utility.hpp>
#include <boost/foreach.hpp>
#include <stdexcept>
#include "ply.h"
#include "ascii_io.h"
#include "binary_io.h"

using namespace std;

namespace PLY
{
namespace detail
{

/**
 * Wrapper around @c boost::lexical_cast that ensures that the result is
 * reversible. It can be used to validate user input, although it is
 * stricter than necessary (e.g. it will not accept leading zeros or
 * plus signs). It is probably not a good idea to use this for floating-point
 * values, since non-canonical representations are generally used.
 *
 * @param s The value to convert.
 */
template<typename T, typename S>
static T validateLexicalCast(S s)
{
    T t = boost::lexical_cast<T>(s);
    if (boost::lexical_cast<S>(t) != s)
        throw boost::bad_lexical_cast();
    return t;
}

/**
 * Splits a string on whitespace, using operator>>.
 *
 * @param line The string to split.
 * @return A vector of tokens, not containing whitespace.
 */
static vector<string> splitLine(const string &line)
{
    istringstream splitter(line);
    return vector<string>(istream_iterator<string>(splitter), istream_iterator<string>());
}

/**
 * Maps the label for a type in the PLY header to a type token.
 * The types int, uint and float are mapped to INT32, UINT32 and FLOAT32
 * respectively.
 *
 * @param t The name of the type from the PLY header.
 * @return The #FieldType value corresponding to @a t.
 * @throw #FormatError if @a t is not recognized.
 */
static FieldType parseType(const string &t) throw(FormatError)
{
    if (t == "int8" || t == "char") return INT8;
    else if (t == "uint8" || t == "uchar") return UINT8;
    else if (t == "int16") return INT16;
    else if (t == "uint16") return UINT16;
    else if (t == "int32" || t == "int") return INT32;
    else if (t == "uint32" || t == "uint") return UINT32;
    else if (t == "float32" || t == "float") return FLOAT32;
    else if (t == "float64") return FLOAT64;
    else throw FormatError("Unknown type `" + t + "'");
}

/**
 * Retrieve a line from the header, throwing a suitable exception on failure.
 *
 * @param in The input stream containing the header
 * @return A line from @a in
 * @throw FormatError on EOF
 * @throw std::ios::failure on other I/O error
 */
static string getHeaderLine(istream &in) throw(FormatError)
{
    try
    {
        string line;
        getline(in, line);
        return line;
    }
    catch (ios::failure &e)
    {
        if (in.eof())
            throw FormatError("End of file in PLY header");
        else
            throw;
    }
}

} // namespace detail

template<typename T> long ElementRangeReaderBase::PropertyAsLong::operator()()
{
    return long(reader.readField<T>());
}

void ElementRangeReaderBase::skip()
{
    assert(getNumber() == 0 || &*reader.currentReader == this);
    PropertyAsLong convertHelper(reader);
    FieldTypeFunction<PropertyAsLong> converter(convertHelper);
    while (&*reader.currentReader == this)
    {
        BOOST_FOREACH(const PropertyType &p, getProperties())
        {
            // TODO: handle lists
            if (p.isList)
            {
                long length = converter(p.lengthType);
                for (long i = 0; i < length; i++)
                    (void) converter(p.valueType);
            }
            else
                (void) converter(p.valueType);
        }
        reader.increment();
    }
}

void Reader::increment()
{
    assert(currentReader != end());
    assert(currentPos < currentReader->getNumber());
    ++currentPos;
    while (currentReader != end() && currentPos == currentReader->getNumber())
    {
        ++currentReader;
        currentPos = 0;
    }
}

ElementRangeReaderBase &Reader::skipToBase(const std::string &name)
{
    std::size_t currentIdx = currentReader - begin();

    for (size_t i = 0; i < readers.size(); ++i)
        if (readers[i].getName() == name)
        {
            if (i < currentIdx)
            {
                // This is legal only if everything from i to currentIdx
                // is empty.
                bool legal = (currentPos == 0);
                for (size_t j = i; j < currentIdx && legal; j++)
                    if (readers[j].getNumber())
                        legal = false;
                if (legal)
                    return readers[i];
                else
                    throw FormatError("Element `" + name + "' has already been read");
            }
            else
            {
                while (size_t(currentReader - begin()) < i)
                    currentReader->skip();
                if (currentPos > 0)
                {
                    throw FormatError("Element `" + name + "' has already been started");
                }
                return readers[i];
            }
        }

    // Should hit one of the returns or throw above if we matched
    throw FormatError("No element called `" + name + "'");
}

void Reader::addElement(const std::string &name, const std::tr1::uintmax_t number, const PropertyTypeSet &properties)
{
    // Check for duplicates (TODO: use a SequencedMapType to speed this up?)
    BOOST_FOREACH(const ElementRangeReaderBase &e, readers)
    {
        if (e.getName() == name)
            throw FormatError("Duplicate element name `" + name + "'");
    }

    boost::ptr_map<std::string, FactoryBase>::const_iterator factory = factories.find(name);
    if (factory == factories.end())
        readers.push_back(new ElementRangeReader<EmptyBuilder>(*this, name, number, properties, EmptyBuilder()));
    else
        readers.push_back((*factory->second)(*this, name, number, properties));
}

void Reader::readHeader()
{
    using namespace detail;

    string elementName = "";
    std::tr1::uintmax_t elementNumber = 0;
    PropertyTypeSet elementProperties;
    bool haveElement = false;

    in.exceptions(ios::failbit);

    string line = getHeaderLine(in);
    if (line != "ply")
        throw FormatError("PLY signature missing");

    format = FILE_FORMAT_ASCII;
    // read the header
    while (true)
    {
        vector<string> tokens;

        line = getHeaderLine(in);
        tokens = splitLine(line);
        if (tokens.empty())
            continue; // ignore blank lines
        if (tokens[0] == "end_header")
            break;
        else if (tokens[0] == "format")
        {
            if (tokens.size() != 3)
                throw FormatError("Malformed format line");

            if (tokens[1] == "ascii")
                format = FILE_FORMAT_ASCII;
            else if (tokens[1] == "binary_big_endian")
                format = FILE_FORMAT_BIG_ENDIAN;
            else if (tokens[1] == "binary_little_endian")
                format = FILE_FORMAT_LITTLE_ENDIAN;
            else
                throw FormatError("Unknown PLY format " + tokens[1]);

            if (tokens[2] != "1.0")
                throw FormatError("Unknown PLY version " + tokens[2]);
        }
        else if (tokens[0] == "element")
        {
            if (haveElement)
            {
                addElement(elementName, elementNumber, elementProperties);
                haveElement = false;
                elementProperties.clear();
            }

            if (tokens.size() != 3)
                throw FormatError("Malformed element line");
            elementName = tokens[1];
            try
            {
                elementNumber = validateLexicalCast<std::tr1::uintmax_t>(tokens[2]);
            }
            catch (boost::bad_lexical_cast &e)
            {
                throw FormatError("Malformed element line");
            }
            haveElement = true;
        }
        else if (tokens[0] == "property")
        {
            PropertyType p;
            if (tokens.size() < 3)
                throw FormatError("Malformed property line");
            if (tokens[1] == "list")
            {
                if (tokens.size() != 5)
                    throw FormatError("Malformed property line");
                p.isList = true;
                p.lengthType = parseType(tokens[2]);
                p.valueType = parseType(tokens[3]);
                if (p.lengthType == FLOAT32 || p.lengthType == FLOAT64)
                    throw FormatError("List cannot have floating-point count");
                p.name = tokens[4];
            }
            else
            {
                if (tokens.size() != 3)
                    throw FormatError("Malformed property line");
                p.isList = false;
                p.lengthType = INT32; // unused, just to avoid undefined values
                p.valueType = parseType(tokens[1]);
                p.name = tokens[2];
            }
            if (!haveElement)
                throw FormatError("Property `" + p.name + "' appears before any element declaration");
            if (!elementProperties.push_back(p).second)
                throw FormatError("Duplicate property `" + p.name + "'");
        }
        else if (tokens[0] == "comment" || tokens[0] == "obj_info")
        {
            /* Ignore comments */
        }
        else
        {
            throw FormatError("Unknown header token `" + tokens[0] + "'");
        }
    }
    if (haveElement)
    {
        addElement(elementName, elementNumber, elementProperties);
    }

    currentReader = begin();
    currentPos = 0;
    while (currentReader != end() && currentPos == currentReader->getNumber())
    {
        ++currentReader;
        currentPos = 0;
    }
}

/**************************************************************************/

namespace detail
{

ElementRangeWriterBase::ElementRangeWriterBase(std::tr1::uintmax_t number)
    : number(number)
{
}

std::tr1::uintmax_t ElementRangeWriterBase::getNumber() const
{
    return number;
}

} // namespace detail

template<typename T>
void Writer::writeField(T value, bool final)
{
    switch (format)
    {
    case FILE_FORMAT_ASCII:
        out << numberToString(value) << (final ? '\n' : ' ');
        break;
    case FILE_FORMAT_LITTLE_ENDIAN:
        writeBinary(out, value, boost::true_type());
        break;
    case FILE_FORMAT_BIG_ENDIAN:
        writeBinary(out, value, boost::false_type());
        break;
    }
}

template void Writer::writeField<std::tr1::uint8_t>(std::tr1::uint8_t value, bool final);
template void Writer::writeField<std::tr1::int8_t>(std::tr1::int8_t value, bool final);
template void Writer::writeField<std::tr1::uint16_t>(std::tr1::uint16_t value, bool final);
template void Writer::writeField<std::tr1::int16_t>(std::tr1::int16_t value, bool final);
template void Writer::writeField<std::tr1::uint32_t>(std::tr1::uint32_t value, bool final);
template void Writer::writeField<std::tr1::int32_t>(std::tr1::int32_t value, bool final);
template void Writer::writeField<float>(float value, bool final);
template void Writer::writeField<double>(double value, bool final);

Writer::Writer(FileFormat format, std::streambuf *buf)
    : format(format), out(buf)
{
}

const char * Writer::fieldTypeName(FieldType type)
{
    switch (type)
    {
    case INT8:    return "int8";
    case UINT8:   return "uint8";
    case INT16:   return "int16";
    case UINT16:  return "uint16";
    case INT32:   return "int32";
    case UINT32:  return "uint32";
    case FLOAT32: return "float32";
    case FLOAT64: return "float64";
    default: abort();
    }
}

void Writer::writeHeaderElement(const detail::ElementType &element)
{
    out << "element " << element.name << ' ' << element.number << '\n';
    BOOST_FOREACH(const PropertyType &p, element.properties)
    {
        out << "property ";
        if (p.isList)
        {
            out << "list " << fieldTypeName(p.lengthType) << ' ';
        }
        out << fieldTypeName(p.valueType) << ' ' << p.name << '\n';
    }
}

void Writer::writeHeader()
{
    out << "ply\n";
    switch (format)
    {
    case FILE_FORMAT_ASCII:
        out << "format ascii 1.0\n";
        break;
    case FILE_FORMAT_LITTLE_ENDIAN:
        out << "format binary_little_endian 1.0\n";
        break;
    case FILE_FORMAT_BIG_ENDIAN:
        out << "format binary_big_endian 1.0\n";
        break;
    default:
        abort();
    }

    BOOST_FOREACH(const std::string &comment, comments)
    {
        out << "comment " << comment << '\n';
    }
    BOOST_FOREACH(const detail::ElementType &e, elements)
    {
        writeHeaderElement(e);
    }
    out << "end_header\n";
}

void Writer::addElement(std::auto_ptr<detail::ElementRangeWriterBase> e)
{
    detail::ElementType et(e->getName(), e->getNumber());
    et.properties = e->getProperties();
    if (!elements.push_back(et).second)
        throw std::invalid_argument("duplicate element " + et.name);
    try
    {
        writers.push_back(e);
    }
    catch (...)
    {
        elements.pop_back();
        throw;
    }
}

void Writer::addComment(const std::string &comment)
{
    comments.push_back(comment);
}

void Writer::write()
{
    writeHeader();
    BOOST_FOREACH(const detail::ElementRangeWriterBase &w, writers)
    {
        w.writeAll(*this);
    }
}

} // namespace PLY
