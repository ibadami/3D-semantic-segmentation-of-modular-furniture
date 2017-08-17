#include "libforest/error_handling.h"
#include "libforest/io.h"
#include <iomanip>
#include <iostream>

std::string libf::lastMessage;

#ifdef LIBF_ENABLE_ASSERT

#if defined(BOOST_DISABLE_ASSERTS)
#elif defined(BOOST_ENABLE_ASSERT_HANDLER)

namespace boost
{
    void assertion_failed(char const * expr,
                          char const * function, char const * file, long line)
    {
        // Throw an assertion exception
        throw libf::AssertionException(expr, function, file, line);
    }
}

#endif

#if defined(BOOST_DISABLE_ASSERTS) || defined(NDEBUG)
#elif defined(BOOST_ENABLE_ASSERT_HANDLER)

/**
 * This function is called if an assertion with some message fails. 
 */
void boost::assertion_failed_msg(char const * expr, char const * msg, char const * function, char const * file, long line)
{
    // Throw an assertion exception
    throw libf::AssertionException(expr, msg, function, file, line);
}

#endif

#endif

char const* libf::AssertionException::what() const throw()
{
    // Build the error message
    std::stringstream ss;
    ss << std::endl;
    ss << std::setw(16) << std::left << "Expression:" << expression << std::endl;
    ss << std::setw(16) << std::left << "Message:" << LIBF_COLOR_RED << message << LIBF_COLOR_RESET << std::endl;
    ss << std::setw(16) << std::left << "Function:" << function << std::endl;
    ss << std::setw(16) << std::left << "File:" << file << std::endl;
    ss << std::setw(16) << std::left << "Line:" << line << std::endl;
    
    // Save the error message 
    lastMessage = ss.str();
    return lastMessage.c_str();
}
