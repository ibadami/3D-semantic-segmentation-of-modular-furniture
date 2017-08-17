#ifndef LIBF_ERRORHANDLING_H
#define LIBF_ERRORHANDLING_H

/**
 * Setting this macro allows us to throw our own exception when an assertion 
 * fails
 */
#ifdef LIBF_ENABLE_ASSERT

#undef NDEBUG
#define BOOST_ENABLE_ASSERT_HANDLER

#else

#define BOOST_DISABLE_ASSERTS

#endif

#include <boost/assert.hpp>

#include <exception>
#include <string>
#include <sstream>

namespace libf {
    /**
     * We save the last error message in this variable. 
     * TODO: This has to be fixed. This solution is not thread safe. 
     */
    extern std::string lastMessage;
    
    /**
     * This exception is thrown if an assertion fails.
     */
    class AssertionException : public std::exception {
    public:
        /**
         * This constructor should be used if there is an error message. 
         * 
         * @param expression The expression of the assertion that failed
         * @param message The error message given by the developer
         * @param function The function where the assertion is located
         * @param file The file where the function is located
         * @param line The line number of the exception
         */
        AssertionException( char const * expression, 
                            char const * message, 
                            char const * function, 
                            char const * file, 
                            long line) : 
                            expression(expression), 
                            message(message), 
                            function(function), 
                            file(file), 
                            line(line) {}
        
        /**
         * This constructor should be used if there is no error message. 
         * 
         * @param expression The expression of the assertion that failed
         * @param function The function where the assertion is located
         * @param file The file where the function is located
         * @param line The line number of the exception
         */
        AssertionException( char const * expression, 
                            char const * function, 
                            char const * file, 
                            long line) : 
                            expression(expression), 
                            message("No error message"), 
                            function(function), 
                            file(file), 
                            line(line) {}
        
        /**
         * Returns the error message for this exception. 
         */
        char const* what() const throw();
        
    private:
        /**
         * The expression that failed
         */
        const char* expression;
        /**
         * The error message
         */
        const char* message;
        /**
         * The function where the assertion failed
         */
        const char* function;
        /**
         * The file where the assertion failed
         */
        const char* file;
        /**
         * The line where the assertion failed
         */
        long line;
    };
    
    /**
     * This exception is thrown when some IO operations fail.
     */
    class IOException : public std::exception {
    public:
        /**
         * This constructor should be used if there is an error message. 
         * 
         * @param expression The expression of the assertion that failed
         * @param message The error message given by the developer
         * @param function The function where the assertion is located
         * @param file The file where the function is located
         * @param line The line number of the exception
         */
        IOException( const char * message) : message(message) {}
        
        /**
         * Returns the error message for this exception. 
         */
        char const* what() const throw()
        {
            return message;
        }
        
    private:
        /**
         * The error message
         */
        const char* message;
    };
}

#endif 