/**
 * @file
 *
 * Information logging support.
 */

#ifndef MINIMLS_LOGGING_H
#define MINIMLS_LOGGING_H

#include <iosfwd>

namespace Log
{

enum Level
{
    debug,
    info,
    warn,
    error
};

namespace internal
{

class LogArray
{
private:
    Level minLevel;
public:
    explicit LogArray(Level minLevel = warn);
    void setLevel(Level minLevel);
    std::ostream &operator[](Level level);
};

} // namespace internal

extern internal::LogArray log;

} // namespace Log

#endif /* MINIMLS_LOGGING_H */
