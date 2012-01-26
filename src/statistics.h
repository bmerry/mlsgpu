/**
 * @file
 *
 * Classes for gathering and reporting statistics.
 *
 * Several types of statistics are supported:
 *  - Counters, which count the number of times an event occurs
 *  - Variables, which model a random variable and determine mean and standard deviation
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
#include "timer.h"

class TestStatistic;

namespace Statistics
{

/**
 * Object that holds accumulated data about a statistic. This is a virtual base
 * class that is subclassed to define different types of statistics.
 */
class Statistic : public boost::noncopyable
{
    friend std::ostream &operator<<(std::ostream &o, const Statistic &data);
private:
    const std::string name;         ///< Name of the statistic

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

    const std::string &getName() const;  ///< Returns the name of the statistic (thread-safe)
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
    friend class ::TestStatistic;
private:
    unsigned long long total;

protected:
    virtual void write(std::ostream &o) const;

public:
    Counter(const std::string &name);

    /// Increment the counter by a specified amount
    void add(unsigned long long incr = 1ULL);

    /// Return the total value of the counter
    unsigned long long getTotal() const;
};

/**
 * Statistic subclass that computes mean and standard deviation.
 */
class Variable : public Statistic
{
    friend class ::TestStatistic;
private:
    double sum;             ///< sum of samples
    double sum2;            ///< sum of squares of samples
    unsigned long long n;   ///< number of samples

    double getVarianceUnlocked() const;  ///< compute variance with the caller taking the lock

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
};

/**
 * Statistic class that measures the maximum value a variable takes.
 */
template<typename T>
class Peak : public Statistic
{
private:
    T current;
    T max;
    bool hasValue;

protected:
    virtual void write(std::ostream &o) const
    {
        if (hasValue)
            o << max;
        else
            o << "[no samples]";
    }

public:
    Peak(const std::string &name) : Statistic(name), current(), max(), hasValue(false) {}

    void add(T x)
    {
        if (!hasValue)
            throw std::length_error("No samples");
        set(current + x);
    }

    void set(T x)
    {
        current = x;
        if (!hasValue || current > max)
        {
            max = current;
            hasValue = true;
        }
    }

    T get() const
    {
        if (!hasValue)
            throw std::length_error("No samples");
        return current;
    }

    T getMax() const
    {
        if (!hasValue)
            throw std::length_error("No samples");
        return max;
    }
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

namespace internal
{
    /**
     * Iterator adaptor that converts iteration over a pair pointer
     * associative container into an iterator over its values.
     *
     * @todo Move into a separate file.
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

        typename pair_second_iterator<Base>::iterator_adaptor_::reference dereference() const
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

} // namespace internal

/**
 * Holds a list of statistics that can be queried or dumped.
 *
 * This class is thread-safe.
 */
class Registry
{
private:
    boost::ptr_map<std::string, Statistic> statistics;
    mutable boost::mutex mutex;  ///< Mutex protecting access to the statistics map

public:
    typedef internal::pair_second_iterator<boost::ptr_map<std::string, Statistic>::iterator> iterator;
    typedef internal::pair_second_iterator<boost::ptr_map<std::string, Statistic>::const_iterator> const_iterator;

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
};

/**
 * Write a whole registry to a stream, with newlines.
 *
 * @note This does not provide an atomic view of the whole registry: if another
 * thread updates two statistics, it is possible that the stream output will
 * contain one update and not the other. However, the set of statistics (i.e.
 * just the names) will be atomic.
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

#endif /* MLSGPU_STATISTICS_H */
