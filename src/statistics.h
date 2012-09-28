/**
 * @file
 *
 * Classes for gathering and reporting statistics.
 *
 * Several types of statistics are supported:
 *  - Counters, which count the number of times an event occurs
 *  - Variables, which model a random variable and determine mean and standard deviation
 *  - Peaks, which measure the highest value of some variable (useful for e.g. memory allocation)
 *
 * It also provides utility classes for interacting with timers.
 */

#ifndef MLSGPU_STATISTICS_H
#define MLSGPU_STATISTICS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <string>
#include <ostream>
#include <iterator>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/noncopyable.hpp>
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <memory>
#include "timer.h"
#include "tr1_cstdint.h"

class TestCounter;
class TestVariable;
class TestPeak;

/**
 * Functions and classes for gathering statistics.
 */
namespace Statistics
{

/**
 * Object that holds accumulated data about a statistic. This is a virtual base
 * class that is subclassed to define different types of statistics. All subclasses
 * must implement serialization: this is required by the implementation of the
 * @ref clone method and by @ref Registry::merge.
 *
 * Serialization of statistics is @em not thread-safe. It should only be done in
 * situations where it is guaranteed that no other threads are accessing the
 * statistic.
 */
class Statistic : public boost::noncopyable
{
    friend std::ostream &operator<<(std::ostream &o, const Statistic &data);
    friend class boost::serialization::access;
private:
    std::string name;         ///< Name of the statistic

    Statistic() {} // for serialization

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int);

protected:
    mutable boost::mutex mutex;     ///< Mutex protecting access to the sample data

    /**
     * Implementation of <code>operator &lt;&lt;</code>. The caller takes care
     * of locking and of printing the name.
     */
    virtual void write(std::ostream &o) const = 0;

public:
    Statistic(const std::string &name);
    virtual ~Statistic();

    /**
     * Merge another set of samples into this one.
     *
     * @throw bad_cast if @a other does not have the same type as @c this.
     */
    virtual void merge(const Statistic &other) = 0;

    const std::string &getName() const;  ///< Returns the name of the statistic (thread-safe)

    /**
     * Create a clone of the statistic, with the same name, type and values.
     * The caller is responsible for deleting the new statistic when it is
     * no longer needed.
     */
    Statistic *clone() const;
};

/**
 * Write a single statistic to a stream, without a newline.
 *
 * This function is thread-safe, and provides an atomic view of the statistic
 * (i.e. the mean and standard deviation will be consistent).
 */
std::ostream &operator <<(std::ostream &o, const Statistic &stat);

/**
 * Statistic subclass that just counts a number of events.
 */
class Counter : public Statistic
{
    friend class ::TestCounter;
    friend class boost::serialization::access;
private:
    unsigned long long total;

    Counter() : Statistic("") {} // for serialization

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int);

protected:
    virtual void write(std::ostream &o) const;

public:
    Counter(const std::string &name);

    /// Increment the counter by a specified amount
    void add(unsigned long long incr = 1ULL);

    /// Return the total value of the counter
    unsigned long long getTotal() const;

    virtual void merge(const Statistic &other);
};

/**
 * Statistic subclass that computes mean and standard deviation.
 */
class Variable : public Statistic
{
    friend class ::TestVariable;
    friend class boost::serialization::access;
private:
    double sum;             ///< sum of samples
    double sum2;            ///< sum of squares of samples
    unsigned long long n;   ///< number of samples

    double getVarianceUnlocked() const;  ///< compute variance with the caller taking the lock

    Variable() : Statistic("") {} // for serialization

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int);

protected:
    virtual void write(std::ostream &o) const;

public:
    Variable(const std::string &name);

    /// Add a sample of the variable
    void add(double value);

    unsigned long long getNumSamples() const;   ///< Return the number of calls to @ref add
    /**
     * Return the mean.
     * @throw std::length_error if no samples have been added.
     */
    double getMean() const;
    /**
     * Return the sample standard deviation.
     * @throw std::length_error if less than two samples have been added.
     */
    double getStddev() const;                   ///< Return the sample standard deviation
    /**
     * Return the sample variance.
     * @throw std::length_error if less than two samples have been added.
     */
    double getVariance() const;                 ///< Return the sample variance

    virtual void merge(const Statistic &other);
};

/**
 * Statistic class that measures the maximum value a variable takes. In the initial
 * state, the current value and the maximum are default-initialized. It is operated
 * on using @c =, @c += and @c -=.
 */
class Peak : public Statistic
{
    friend class ::TestPeak;
    friend class boost::serialization::access;
public:
    typedef std::tr1::int64_t value_type;

private:
    value_type current;
    value_type peak;

    Peak() : Statistic("") {} // for serialization

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int);

protected:
    virtual void write(std::ostream &o) const;

    /**
     * Replaces the current value.
     *
     * @pre The caller must hold the lock.
     */
    void set(value_type x);

public:
    /**
     * Construct, setting a name and default-initializing the current and maximum values.
     */
    Peak(const std::string &name);

    /// Increment the current value by @a x.
    Peak &operator+=(value_type x);

    /// Decrement the current value by @a x.
    Peak &operator-=(value_type x);

    /// Set the current value to @a x.
    Peak &operator=(value_type x);

    /// Retrieve the current value.
    value_type get() const;

    /// Retrieves the highest value that has been set.
    value_type getMax() const;

    virtual void merge(const Statistic &other);
};

/**
 * @ref Timer subclass that reports elapsed time to a statistic
 * on destruction.
 */
class Timer : public ::Timer
{
private:
    Variable &stat;

public:
    /// Constructor that will record in a named statistic in the default registry
    explicit Timer(const std::string &name);
    /// Constructor that will record to a specific statistic
    explicit Timer(Variable &stat);

    /// Destructor that records the elapsed time
    ~Timer();
};

namespace detail
{
    /**
     * Iterator adaptor that converts iteration over a pair pointer
     * associative container into an iterator over its values.
     */
    template<typename Base>
    class pair_second_iterator : public boost::iterator_adaptor<
        pair_second_iterator<Base>,
        Base,
        typename boost::remove_pointer<typename std::iterator_traits<Base>::value_type::second_type>::type,
        boost::use_default,
        typename boost::remove_pointer<typename std::iterator_traits<Base>::value_type::second_type>::type &
        >
    {
    private:
        friend class boost::iterator_core_access;

        // The base class already defines this, but MSVC 10 doesn't seem to work with it
        typedef boost::iterator_adaptor<
            pair_second_iterator<Base>,
            Base,
            typename boost::remove_pointer<typename std::iterator_traits<Base>::value_type::second_type>::type,
            boost::use_default,
            typename boost::remove_pointer<typename std::iterator_traits<Base>::value_type::second_type>::type &
        > iterator_adaptor_;

        typename pair_second_iterator::iterator_adaptor_::reference dereference() const
        {
            return *this->base()->second;
        }

        /// dummy class to prevent non-default argument being used with boost::enable_if
        class enabler {};

    public:
        pair_second_iterator() {}
        pair_second_iterator(const Base &base) : pair_second_iterator::iterator_adaptor_(base) {}

        template<typename Other>
        pair_second_iterator(const pair_second_iterator<Other> &other,
            typename boost::enable_if<boost::is_convertible<Other, Base>, enabler>::type = enabler())
            : pair_second_iterator::iterator_adaptor_(other.base())
        {}
    };

} // namespace detail

/**
 * Holds a list of statistics that can be queried or dumped.
 *
 * This class is thread-safe.
 */
class Registry
{
    friend class boost::serialization::access;
private:
    boost::ptr_map<std::string, Statistic> statistics;
    mutable boost::mutex mutex;  ///< Mutex protecting access to the statistics map

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int);

public:
    typedef detail::pair_second_iterator<boost::ptr_map<std::string, Statistic>::iterator> iterator;
    typedef detail::pair_second_iterator<boost::ptr_map<std::string, Statistic>::const_iterator> const_iterator;

    Registry();
    ~Registry();

    /// Obtain a singleton statistics registry
    static Registry &getInstance();

    /**
     * Retrieves a named statistic, creating it if necessary.
     *
     * @throw bad_cast If there is already a statistic by that name which is of a different type.
     */
    template<typename T>
    T &getStatistic(const std::string &name);

    friend std::ostream &operator<<(std::ostream &o, const Registry &reg);

    /**
     * @name Iteration
     * @{
     *
     * These functions iterate over the statistics in the registry, in lexicographical
     * order by name.
     *
     * @warning An iteration using these functions is not thread-safe! Use them
     * only while no other thread is modifying the registry by adding new statistics.
     */
    iterator begin();                ///< First statistic
    iterator end();                  ///< One past the last statistic
    const_iterator begin() const;    ///< First statistic
    const_iterator end() const;      ///< One past the last statistic
    /**
     * @}
     */

    /**
     * Merge in samples from another registry. Statistics with the same
     * name are matched up. They must then have the same type, or else
     * @c bad_cast is thrown and the current registry is corrupted!
     *
     * This function is @em not thread-safe.
     */
    void merge(const Registry &other);
};

/**
 * Write a whole registry to a stream, with newlines.
 *
 * @note This does not provide an atomic view of the whole registry: if another
 * thread updates two statistics, it is possible that the stream output will
 * contain one update and not the other. However, the set of statistics (i.e.
 * just the names) will be atomic.
 *
 * This is intended for human-readable output. It is not a serialization format.
 */
std::ostream &operator <<(std::ostream &o, const Registry &reg);

template<typename T>
T &Registry::getStatistic(const std::string &name)
{
    boost::lock_guard<boost::mutex> _(mutex);
    boost::ptr_map<std::string, Statistic>::iterator pos = statistics.find(name);
    if (pos != statistics.end())
    {
        return dynamic_cast<T &>(*pos->second);
    }
    else
    {
        T *stat = new T(name);
        std::auto_ptr<Statistic> wrap(stat);
        statistics.insert(name, wrap);
        return *stat;
    }
}

/**
 * Retrieves a named statistic from the default registry.
 * This is shorthand for <code>Registry::getInstance().getStatistic<T>(name)</code>.
 */
template<typename T>
static inline T &getStatistic(const std::string &name)
{
    return Registry::getInstance().getStatistic<T>(name);
}

} // namespace Statistics

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Statistics::Statistic)
BOOST_CLASS_EXPORT_KEY(Statistics::Statistic)
BOOST_CLASS_EXPORT_KEY(Statistics::Counter)
BOOST_CLASS_EXPORT_KEY(Statistics::Variable)
BOOST_CLASS_EXPORT_KEY(Statistics::Peak)
BOOST_CLASS_EXPORT_KEY(Statistics::Registry)

#endif /* !MLSGPU_STATISTICS_H */
