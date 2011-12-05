#ifndef MINIMLS_ERRORS_H
#define MINIMLS_ERRORS_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <exception>

#define MINIMLS_STRINGIZE(x) #x
#define MINIMLS_ASSERT_PASTE(file, line, msg) (file ":" MINIMLS_STRINGIZE(line) ": " msg)

#define MINIMLS_ASSERT_IMPL(cond, exception_type) \
    ((cond) ? ((void) 0) : (minimlsThrow<exception_type>(__FILE__, __LINE__, MINIMLS_ASSERT_PASTE(__FILE__, __LINE__, #cond))))

#define MINIMLS_UNUSED(x) (false ? (void) (x) : (void) 0)

template<typename ExceptionType>
static void minimlsThrow(const char *filename, int line, const char *msg) throw(ExceptionType)
{
#if MINIMLS_ASSERT_ABORT
    std::cerr << filename << ':' << line << ": " << msg << endl;
    abort();
#else
    MINIMLS_UNUSED(filename);
    MINIMLS_UNUSED(line);
    throw ExceptionType(msg);
#endif
}

#endif /* !MINIMLS_ERRORS_H */

/* This part is outside the include guard so that it can be redefined after resetting
 * NDEBUG.
 */

#undef MINIMLS_ASSERT
#if defined(NDEBUG)
# define MINIMLS_ASSERT(cond, except_type) (false ? MINIMLS_ASSERT_IMPL(cond, except_type) : (void) 0)
#else
# define MINIMLS_ASSERT(cond, except_type) MINIMLS_ASSERT_IMPL(cond, except_type)
#endif
