/**
 * @file
 *
 * Point cloud support for the Stanford PLY format.
 *
 * The support includes both ASCII and binary variants, and templates
 * are used to allow arbitrary properties to be supported.
 */

#ifndef PLY_H
#define PLY_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <streambuf>
#include <iterator>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/iterator_concepts.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/foreach.hpp>
#include <boost/noncopyable.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <tr1/cstdint>
#include "binary_io.h"
#include "ascii_io.h"
#include "errors.h"

class TestPlyReader;

namespace PLY
{

/**
 * An exception that is thrown when an invalid PLY file is encountered.
 * This is used to signal all errors in a PLY file (including early end-of-file),
 * but excluding I/O errors (which are signaled with @c std::ios::failure).
 */
class FormatError : public std::runtime_error
{
public:
    FormatError(const std::string &msg) : std::runtime_error(msg) {}
};

/**
 * The encoding used for a PLY file.
 */
enum FileFormat
{
    FILE_FORMAT_ASCII,
    FILE_FORMAT_LITTLE_ENDIAN,
    FILE_FORMAT_BIG_ENDIAN
};

/**
 * The type of a field in a PLY file.
 */
enum FieldType
{
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    FLOAT32,
    FLOAT64
};

template<typename T>
struct FieldTypeTraits
{
};

#define DEFINE_FIELD_TYPE_TRAITS(type_, value_) \
    template<> struct FieldTypeTraits<type_> { static const FieldType value = value_; }
DEFINE_FIELD_TYPE_TRAITS(std::tr1::uint8_t, UINT8);
DEFINE_FIELD_TYPE_TRAITS(std::tr1::int8_t, INT8);
DEFINE_FIELD_TYPE_TRAITS(std::tr1::uint16_t, UINT16);
DEFINE_FIELD_TYPE_TRAITS(std::tr1::int16_t, INT16);
DEFINE_FIELD_TYPE_TRAITS(std::tr1::uint32_t, UINT32);
DEFINE_FIELD_TYPE_TRAITS(std::tr1::int32_t, INT32);
DEFINE_FIELD_TYPE_TRAITS(float, FLOAT32);
DEFINE_FIELD_TYPE_TRAITS(double, FLOAT64);
#undef DEFINE_FIELD_TYPE_TRAITS

/**
 * A wrapper class that adapts a templated function object to be dynamically
 * variadic depending on the type of a field.
 *
 * The underlying function object must be a true function object (not a
 * function pointer), with a @c result_type member and a templatized
 * @c operator() that accepts zero arguments. The appropriate specialization
 * will be called depending on the @c FieldType passed to this function
 * object.
 *
 * @todo move to PLY::internal namespace?
 */
template<typename F>
class FieldTypeFunction
{
private:
    F f;
public:
    typedef typename F::result_type result_type;

    /**
     * Constructor.
     * @param f The underlying templatized function object.
     */
    FieldTypeFunction(const F f = F()) : f(f) {}

    /**
     * Callback function.
     */
    result_type operator()(FieldType type)
    {
        switch (type)
        {
        case INT8:     return f.template operator()<std::tr1::int8_t>(); break;
        case UINT8:    return f.template operator()<std::tr1::uint8_t>(); break;
        case INT16:    return f.template operator()<std::tr1::int16_t>(); break;
        case UINT16:   return f.template operator()<std::tr1::uint16_t>(); break;
        case INT32:    return f.template operator()<std::tr1::int32_t>(); break;
        case UINT32:   return f.template operator()<std::tr1::uint32_t>(); break;
        case FLOAT32:  return f.template operator()<float>(); break;
        case FLOAT64:  return f.template operator()<double>(); break;
        default:       abort();
        }
    }
};

/**
 * Encapsulates the information on a @c property line of a PLY header.
 */
struct PropertyType
{
    std::string name;            ///< Property name
    bool isList;                 ///< Whether the property is a list
    /**
     * Type of the length of the list.
     * Undefined for non-list properties.
     */
    FieldType lengthType;
    FieldType valueType;         ///< Type of the data

    PropertyType() {}

    /**
     * Constructor for a list property.
     */
    PropertyType(const std::string &name, FieldType lengthType, FieldType valueType)
        : name(name), isList(true), lengthType(lengthType), valueType(valueType)
    {}

    PropertyType(const std::string &name, FieldType valueType)
        : name(name), isList(false), lengthType(), valueType(valueType)
    {}
};

/**
 * Dummy structure used to allow the name index of #PLY::internal::SequencedMapType to be accessed by name.
 */
struct Name {};

namespace internal
{

/**
 * Wrapper to define an ordered sequence that can be efficiently searched by
 * name.
 *
 * The typical use is
 * <code>typedef SequencedMapType<Value>::type ValueSet;</code>
 * The name index can be accessed by valueSet.get<Name>().
 *
 * @param Value The value type held in the sequence (must have a @c name member of type @c std::string)
 */
template<typename Value>
struct SequencedMapType
{
    typedef boost::multi_index::multi_index_container<
        Value,
        boost::multi_index::indexed_by<
            boost::multi_index::sequenced<>,
            boost::multi_index::hashed_unique<boost::multi_index::tag<Name>, boost::multi_index::member<Value, std::string, &Value::name> >
        > > type;
};

} // namespace internal

/**
 * An ordered sequence of named properties, searchable by name.
 */
typedef internal::SequencedMapType<PropertyType>::type PropertyTypeSet;

#if DOXYGEN_FAKE_CODE  // is never actually defined, except by Doxygen's preprocessor
/**
 * Traits class describing how to extract an element of a particular type from a
 * PLY file.  This template must be specialized for each class that will
 * actually be used.
 *
 * For each element in the file, the caller will construct one of these objects,
 * call @c setProperty for each property read, and finally call @c create
 * to extract the built element.
 *
 * This class does not actually exist - it is a concept that needs a concrete
 * model.
 */
class Builder
{
public:
    typedef void Element;          ///< The type of element built by this builder

    /**
     * Validates that the required properties of the element are
     * present and have appropriate types. It is recommended that unrecognized
     * properties are ignored rather than throwing an exception.
     *
     * @param properties Properties of the element from the PLY header
     * @throw PLY::FormatError if the properties are not suitable.
     */
    static void validateProperties(const PropertyTypeSet &properties);

    /**
     * Set the value of a non-list property.
     * @param name      Name of the property.
     * @param value     Value of the property.
     */
    template<typename T>
    void setProperty(const std::string &name, const T &value);

    /**
     * Set the value of a list property.
     * @param name      Name of the property.
     * @param first     Start of the list of values.
     * @param last      End of the list of values.
     */
    template<typename Iterator>
    void setProperty(const std::string &name, Iterator first, Iterator last);

    /**
     * Produce the element.
     */
    Element create();
};
#endif // DOXYGEN_FAKE_CODE

class EmptyBuilder
{
public:
    struct Element {};

    static void validateProperties(const PropertyTypeSet &properties)
    {
        (void) properties;
    }

    template<typename T>
    void setProperty(const std::string &name, const T &value)
    {
        (void) name;
        (void) value;
    }

    template<typename Iterator>
    void setProperty(const std::string &name, Iterator first, Iterator last)
    {
        (void) name;
        (void) first;
        (void) last;
    }

    Element create() { return Element(); }
};

class Reader;

/**
 * A base class representing a single element type in a PLY file.
 *
 * This class is not useful on its own to actually get the elements.
 * Instances will always be of the @c ElementRangeReader<Builder>
 * subclass, and the caller should make an appropriate @c dynamic_cast.
 */
class ElementRangeReaderBase : public boost::noncopyable
{
protected:
    Reader &reader;

private:
    std::string name;               ///< Name of this element in the file.
    std::tr1::uintmax_t number;     ///< Number of elements of this type in the file.
    PropertyTypeSet properties;     ///< Properties of the element, in the order given

    /**
     * Templatized function object that reads a field from the reader and
     * returns it cast to a long. It is used for fetching lengths for lists.
     */
    class PropertyAsLong
    {
    private:
        Reader &reader;
    public:
        typedef long result_type;

        explicit PropertyAsLong(Reader &reader) : reader(reader) {}
        template<typename T> long operator()();
    };

public:
    ElementRangeReaderBase(Reader &reader, const std::string &name, std::tr1::uintmax_t number, const PropertyTypeSet &properties)
        : reader(reader), name(name), number(number), properties(properties)
    {}
    virtual ~ElementRangeReaderBase()
    {}

    /// Number of elements of this type in the file.
    std::tr1::uintmax_t getNumber() const { return number; }

    /// Name of this element
    const std::string &getName() const { return name; }

    /// Properties of the element, in the order given
    const PropertyTypeSet &getProperties() const { return properties; }

    /**
     * Skip over remainder of this element.
     *
     * @pre The reader must be on this element, or this element must be
     * empty.
     */
    void skip();
};

template<typename Builder_> class ElementRangeReader;

/**
 * Structure to read the vertices from a PLY file.
 *
 * The interface is streaming, allowing for larger files than would otherwise
 * fit in memory. After constructing the reader, iterators are used to read out
 * the elements.
 *
 * The class supports different types of elements, using templatized builder
 * class to determine which fields are expected and how to produce vertices
 * from them.
 */
class Reader : public boost::noncopyable
{
    friend class ElementRangeReaderBase;
    template<typename> friend class ElementRangeReader;
    friend class ::TestPlyReader;
private:
    /// Stream used to wrap the underlying streambuf to parse the header
    std::istream in;
    /// File format determines by the header
    FileFormat format;

    class FactoryBase
    {
    public:
        FactoryBase() {}
        virtual ~FactoryBase() {}
        virtual ElementRangeReaderBase *operator()(Reader &reader, const std::string &name, std::tr1::uintmax_t number, const PropertyTypeSet &properties) const = 0;
    };

    template<typename Builder>
    class Factory : public FactoryBase
    {
    private:
        Builder templateBuilder;
    public:
        explicit Factory(const Builder &templateBuilder) : templateBuilder(templateBuilder) {}
        virtual ElementRangeReaderBase *operator()(Reader &reader, const std::string &name, std::tr1::uintmax_t number, const PropertyTypeSet &properties) const;
    };

    boost::ptr_map<std::string, FactoryBase> factories;
    boost::ptr_vector<ElementRangeReaderBase> readers;
    /// Element currently being read
    boost::ptr_vector<ElementRangeReaderBase>::iterator currentReader;
    /// Element element of @ref currentReader about to be read.
    std::tr1::uintmax_t currentPos;

    /// Advance currentReader/currentPos to the next position
    void increment();

    /**
     * Implementation of @c skipTo that returns a base type and does
     * not do type checking.
     */
    ElementRangeReaderBase &skipToBase(const std::string &name);

    /**
     * Extract a scalar field using the known file format.
     *
     * @return The field value.
     * @throw PLY::FormatError on end-of-file
     * @throw PLY::FormatError if an ASCII field is not formatted correctly or is out of range
     * @throw std::ios::failure on other I/O errors
     * @pre @a T must be one of
     * - @c float or @c double
     * - 8-, 16- or 32-bit signed or unsigned integer
     */
    template<typename T> T readField();

    /**
     * Instantiate an @ref ElementRangeReaderBase from an element description
     * found in the header.
     */
    void addElement(const std::string &name, const std::tr1::uintmax_t number, const PropertyTypeSet &properties);

public:
    typedef boost::ptr_vector<ElementRangeReaderBase>::iterator iterator;

    /**
     * Constructor.
     * @param sb  Underlying stream buffer to read.
     *
     * @note The stream buffer is not owned by this class. The caller must
     * maintain its lifetime.
     */
    explicit Reader(std::streambuf *sb) : in(sb) {}

    /**
     * Register a new property handler.
     *
     * @param name               Name of the property to match.
     * @param templateBuilder    Builder which will be duplicated to handle each instance of the property.
     *
     * @pre There is not already a registered builder for @a name.
     */
    template<typename Builder>
    void addBuilder(const std::string &name, const Builder &templateBuilder);

    /**
     * Read the PLY header.
     *
     * This must only be called after registering builders with @ref addBuilder.
     */
    void readHeader();

    /**
     * Skip all elements until the specified one, and return the element
     * range reader for it.
     * @throw FormatError   if the named element is not found
     * @throw FormatError   if the file pointer has moved into or past the named element
     * @throw std::bad_cast if there is a type mismatch
     */
    template<typename Builder>
    ElementRangeReader<Builder> &skipTo(const std::string &name);

    /**
     * @name Range of @ref ElementReaderBase objects to access the properties
     * @{
     */
    iterator begin() { return readers.begin(); }
    iterator end()   { return readers.end(); }
    /**
     * @}
     */
};

template<typename Builder_>
class ElementRangeReader : public ElementRangeReaderBase
{
public:
    typedef Builder_ Builder;
    typedef typename Builder::Element Element;

private:
    Builder templateBuilder;

    /**
     * Function object template that extracts a field from the file and
     * casts it to a different type.
     */
    template<typename T>
    class FieldCaster
    {
    public:
        typedef T result_type;
    private:
        Reader &reader;

    public:
        /**
         * Constructor.
         */
        FieldCaster(Reader &reader) : reader(reader) {}

        template<typename S>
        T operator()()
        {
            return boost::numeric_cast<T>(reader.template readField<S>());
        }
    };

    /**
     * Function object template that extracts a scalar property from the file
     * and passes it to a builder. It is expected to be used inside
     * #PLY::FieldTypeFunction.
     */
    class PropertySetter
    {
    public:
        typedef void result_type;
    private:
        Reader &reader;                      ///< Reader from which the field will be read
        Builder &builder;                    ///< Builder to which the property will be passed
        const std::string &name;             ///< Name of the property

    public:
        /**
         * Constructor.
         * @param reader  Reader from which the field will be read
         * @param builder Builder to which the property will be passed
         * @param name    Name of the property
         *
         * @note References to all the parameters are kept. They must thus
         * survive until after the function object is invoked.
         */
        PropertySetter(Reader &reader, Builder &builder, const std::string &name) : reader(reader), builder(builder), name(name) {}

        template<typename T>
        void operator()()
        {
            builder.template setProperty<T>(name, reader.template readField<T>());
        }
    };

    /**
     * Function object template that extracts a list property from the file
     * and passes it to a builder. It is expected to be used inside
     * #PLY::FieldTypeFunction.
     */
    class PropertyListSetter
    {
    public:
        typedef void result_type;
    private:
        Reader &reader;                      ///< Reader from which the field will be read
        Builder &builder;                    ///< Builder to which the property will be passed
        const std::string &name;             ///< Name of the property
        std::size_t length;                  ///< Number of list elements

    public:
        /**
         * Constructor.
         * @param reader  Reader from which the field will be read
         * @param builder Builder to which the property will be passed
         * @param name    Name of the property
         * @param length  Number of items in the list
         *
         * @note References to all the parameters are kept. They must thus
         * survive until after the function object is invoked.
         */
        PropertyListSetter(Reader &reader, Builder &builder, const std::string &name, std::size_t length)
            : reader(reader), builder(builder), name(name), length(length) {}

        template<typename T>
        void operator()()
        {
            std::vector<T> values;
            values.reserve(length);
            for (std::size_t i = 0; i < length; i++)
                values.push_back(reader.template readField<T>());
            builder.setProperty(name, values.begin(), values.end());
        }
    };

public:
    /**
     * Input iterator for iterating over the vertices.
     */
    class iterator : public boost::iterator_facade<iterator, Element, std::input_iterator_tag, Element, std::tr1::intmax_t>
    {
        friend class boost::iterator_core_access;
    private:
        ElementRangeReader<Builder> *owner;  ///< Owning reader
        std::tr1::uintmax_t pos;             ///< Number of vertices already read

        void validate() const
        {
            MINIMLS_ASSERT(owner != NULL, std::invalid_argument);
            MINIMLS_ASSERT(&*owner->reader.currentReader == owner, std::invalid_argument);
            MINIMLS_ASSERT(owner->reader.currentPos == pos, std::invalid_argument);
            MINIMLS_ASSERT(pos < owner->getNumber(), std::invalid_argument);
        }

        void increment()
        {
            assert(owner != NULL);
            assert(pos < owner->getNumber());
            ++pos;
        }

        bool equal(const iterator &j) const
        {
            return owner == j.owner && pos == j.pos;
        }

        Element dereference() const
        {
            validate();
            Builder builder(owner->templateBuilder);
            BOOST_FOREACH(const PropertyType &p, owner->getProperties())
            {
                if (p.isList)
                {
                    FieldTypeFunction<FieldCaster<std::size_t> > caster(FieldCaster<std::size_t>(owner->reader));
                    std::size_t length = caster(p.lengthType);
                    FieldTypeFunction<PropertyListSetter> setter(PropertyListSetter(owner->reader, builder, p.name, length));
                    setter(p.valueType);
                }
                else
                {
                    FieldTypeFunction<PropertySetter> setter(PropertySetter(owner->reader, builder, p.name));
                    setter(p.valueType);
                }
            }
            owner->reader.increment();
            return builder.create();
        }

    public:
        iterator() : owner(NULL), pos(0) {}
        iterator(ElementRangeReader<Builder> *owner, std::tr1::uintmax_t pos = 0) : owner(owner), pos(pos) {}
    };

    ElementRangeReader(Reader &reader, const std::string &name, std::tr1::uintmax_t number, const PropertyTypeSet &properties, const Builder &templateBuilder)
        : ElementRangeReaderBase(reader, name, number, properties), templateBuilder(templateBuilder)
    {
        templateBuilder.validateProperties(properties);
    }

    /**
     * @name Input iterator range for the elements
     * @{
     */
    iterator begin() { return iterator(this); }
    iterator end()   { return iterator(this, getNumber()); }
    /**
     * @}
     */
};

template<typename Builder>
ElementRangeReaderBase *Reader::Factory<Builder>::operator()(Reader &reader, const std::string &name, std::tr1::uintmax_t number, const PropertyTypeSet &properties) const
{
    return new ElementRangeReader<Builder>(reader, name, number, properties, templateBuilder);
}

template<typename T>
T Reader::readField()
{
    try
    {
        if (format == FILE_FORMAT_ASCII)
        {
            std::string token;
            in >> token;
            try
            {
                return stringToNumber<T>(token);
            }
            catch (boost::bad_lexical_cast &e)
            {
                throw FormatError(e.what());
            }
        }
        else
        {
            T value;
            if (format == FILE_FORMAT_LITTLE_ENDIAN)
                readBinary(in, value, boost::true_type());
            else
                readBinary(in, value, boost::false_type());
            return value;
        }
    }
    catch (std::ios::failure &e)
    {
        if (in.eof())
            throw FormatError("Unexpected end of file");
        else
            throw;
    }
}

template<typename Builder>
void Reader::addBuilder(const std::string &name, const Builder &templateBuilder)
{
    MINIMLS_ASSERT(!factories.count(name), std::invalid_argument);
    std::string nameCopy = name;
    factories.insert(nameCopy, new Factory<Builder>(templateBuilder));
}

template<typename Builder>
ElementRangeReader<Builder> &Reader::skipTo(const std::string &name)
{
    ElementRangeReaderBase &r = skipToBase(name);
    return dynamic_cast<ElementRangeReader<Builder> &>(r);
}

class Writer;

#if DOXYGEN_FAKE_CODE
/**
 * Concept for transferring data from an element to a PLY file.
 *
 * @note This class does not actually exist - it is here purely as
 * documentation of a concept.
 */
class Fetcher
{
public:
    /// The element type handled by this fetcher.
    typedef int Element;
    /// The name for the element in the PLY header.
    std::string getName() const;
    /// The names and types of properties for the PLY header.
    internal::PropertyTypeSet getProperties() const;
    /**
     * Write the properties for an element to the PLY file.
     *
     * This method must call @ref PLY::Writer::writeField for each property to be
     * written. The properties to be written must match the properties
     * returned by getProperties(), or the PLY file will be malformed
     * (particularly if it is binary).
     *
     * This method must not call @ref PLY::Writer::startElement or @ref PLY::Writer::endElement.
     * That will be done automatically.
     *
     * @param e        The element to write.
     * @param writer   The PLY file to write to.
     */
    void writeElement(const Element &e, Writer &writer) const;
};
#endif // DOXYGEN_FAKE_CODE

namespace internal
{

struct ElementType
{
    std::string name;               ///< Element name
    std::tr1::uintmax_t number;     ///< Number of elements of this type
    PropertyTypeSet properties;     ///< Properties of the element, in the orde

    ElementType() {}
    ElementType(const std::string &name, std::tr1::uintmax_t number) : name(name), number(number) {}
};

typedef SequencedMapType<ElementType>::type ElementTypeSet;

class ElementRangeWriterBase : public boost::noncopyable
{
private:
    std::tr1::uintmax_t number;
protected:
    ElementRangeWriterBase(std::tr1::uintmax_t number);
public:
    virtual ~ElementRangeWriterBase() {}

    std::tr1::uintmax_t getNumber() const;
    virtual std::string getName() const = 0;
    virtual PropertyTypeSet getProperties() const = 0;
    virtual void writeAll(Writer &writer) const = 0;
};

} // namespace internal

/**
 * Uses a user-provided fetcher object to handle processing of each
 * element, while handling the details of iteration.
 *
 * The @a Fetcher class must implement the typedefs and methods listed
 * in @ref Fetcher.
 */
template<typename Iterator, typename Fetcher>
class ElementRangeWriter : public internal::ElementRangeWriterBase
{
public:
    typedef Iterator const_iterator;
private:
    const_iterator first;
    const_iterator last;
    const Fetcher fetcher;

public:
    ElementRangeWriter(const_iterator first, const_iterator last,
                       const Fetcher &fetcher = Fetcher())
        : internal::ElementRangeWriterBase(std::distance(first, last)),
        first(first), last(last), fetcher(fetcher)
    {
        BOOST_CONCEPT_ASSERT((boost_concepts::ForwardTraversalConcept<Iterator>));
    }

    ElementRangeWriter(const_iterator first, const_iterator last,
                       std::tr1::uintmax_t number,
                       const Fetcher &fetcher = Fetcher())
        : internal::ElementRangeWriterBase(number),
        first(first), last(last), fetcher(fetcher)
    {
    }

    virtual PropertyTypeSet getProperties() const
    {
        return fetcher.getProperties();
    }

    virtual std::string getName() const
    {
        return fetcher.getName();
    }

    virtual void writeAll(Writer &writer) const;
};

/**
 * Class for writing PLY files.
 *
 * The typical flow is as follows:
 *  -# Create or obtain a @c streambuf to write to.
 *  -# Create a Writer to wrap the @c streambuf.
 *  -# Call @ref addElement for each element that is to be written. In most
 *     cases, @c makeElementRangeWriter can be used to construct a suitable
 *     argument.
 *  -# Call @ref write.
 *  -# Destroy the Writer.
 *  -# Close the stream.
 */
class Writer
{
    friend class internal::ElementRangeWriterBase;
private:
    FileFormat format;
    internal::ElementTypeSet elements;
    boost::ptr_vector<internal::ElementRangeWriterBase> writers;
    std::vector<std::string> comments;

    static const char *fieldTypeName(FieldType type);
    void writeHeaderElement(const internal::ElementType &element);

    std::ostream out;

    /**
     * Writes the PLY header to the stream.
     */
    void writeHeader();

public:
    /**
     * Constructor.
     *
     * @param format The PLY format to use.
     * @param buf    The buffer to which the PLY file is written.
     */
    Writer(FileFormat format, std::streambuf *buf);

    /**
     * Register an element to be written out by @ref write.
     */
    void addElement(std::auto_ptr<internal::ElementRangeWriterBase> e);

    /**
     * Register a comment to be placed in the PLY header by @ref write.
     */
    void addComment(const std::string &comment);

    /**
     * Write the header and all elements registered by @ref addElement.
     *
     * After this function is called, no further methods should be called.
     */
    void write();

    /**
     * Write a scalar field to the PLY file.
     *
     * This should only be called by element fetcher classes.
     */
    template<typename T>
    void writeField(T value);

    /**
     * Start a new element record in the body.
     *
     * This should only be called by element fetcher classes.
     */
    void startElement();

    /**
     * Terminate an element record in the body.
     *
     * This should only be called by element fetcher classes.
     */
    void endElement();

};

template<typename Iterator, typename Fetcher>
void ElementRangeWriter<Iterator, Fetcher>::writeAll(Writer &writer) const
{
    std::tr1::uintmax_t n = 0;
    for (const_iterator i = first; i != last; ++i)
    {
        const typename std::iterator_traits<const_iterator>::value_type &e = *i;
        writer.startElement();
        fetcher.writeElement(e, writer);
        writer.endElement();
        n++;
    }
    MINIMLS_ASSERT(n == getNumber(), std::length_error);
}

/**
 * Create an element range writer suitable for passing to @ref PLY::Writer::addElement.
 *
 * @param first      Start of range to elements.
 * @param last       One-past-the-end of range of elements.
 * @param fetcher    Object for interpreting the elements.
 *
 * @note This method requires two passes over the elements. If the iterator is
 * not a forward traversal iterator, the version of this method that takes
 * an explicit count must be used.
 */
template<typename Iterator, typename Fetcher>
static inline std::auto_ptr<internal::ElementRangeWriterBase> makeElementRangeWriter(
    Iterator first, Iterator last, const Fetcher &fetcher)
{
    BOOST_CONCEPT_ASSERT((boost_concepts::ForwardTraversalConcept<Iterator>));
    return std::auto_ptr<internal::ElementRangeWriterBase>(new ElementRangeWriter<Iterator, Fetcher>(first, last, fetcher));
}

/**
 * Create an element range writer suitable for passing to @ref PLY::Writer::addElement.
 *
 * @param first      Start of range to elements.
 * @param last       One-past-the-end of range of elements.
 * @param number     Number of items in the range.
 * @param fetcher    Object for interpreting the elements.
 */
template<typename Iterator, typename Fetcher>
static inline std::auto_ptr<internal::ElementRangeWriterBase> makeElementRangeWriter(
    Iterator first, Iterator last,
    std::tr1::uintmax_t number,
    const Fetcher &fetcher)
{
    return std::auto_ptr<internal::ElementRangeWriterBase>(new ElementRangeWriter<Iterator, Fetcher>(first, last, number, fetcher));
}

} // namespace PLY

#endif /* !PLY_H */
