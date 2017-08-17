#ifndef LIBF_IO_H
#define LIBF_IO_H

#include "util.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <Eigen/Dense>

/**
 * These are color codes that can be used with prinft
 */

#define LIBF_COLOR_NORMAL   "\x1B[0m"
#define LIBF_COLOR_RED      "\x1B[31m"
#define LIBF_COLOR_GREEN    "\x1B[32m"
#define LIBF_COLOR_YELLOW   "\x1B[33m"
#define LIBF_COLOR_BLUE     "\x1B[34m"
#define LIBF_COLOR_MAGENTA  "\x1B[35m"
#define LIBF_COLOR_CYAN     "\x1B[36m"
#define LIBF_COLOR_WHITE    "\x1B[37m"
#define LIBF_COLOR_RESET    "\033[0m"
#define LIBF_BG_COLOR_RED      "\x1B[41m"
#define LIBF_BG_COLOR_GREEN    "\x1B[42m"
#define LIBF_BG_COLOR_YELLOW   "\x1B[43m"
#define LIBF_BG_COLOR_BLUE     "\x1B[44m"
#define LIBF_BG_COLOR_MAGENTA  "\x1B[45m"
#define LIBF_BG_COLOR_CYAN     "\x1B[46m"
#define LIBF_BG_COLOR_WHITE    "\x1B[47m"

namespace libf {
    
    /**
     * Writes a binary value to a stream
     */
    template<typename T>
    void writeBinary(std::ostream& stream, const T& value)
    {
        stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    /**
     * Reads a binary value from a stream
     */
    template<typename T>
    void readBinary(std::istream& stream, T& value)
    {
        stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    /**
     * Writes a binary string to a stream
     */
    template <>
    inline void writeBinary(std::ostream & stream, const std::string & value)
    {
        // Write the length of the string
        writeBinary(stream, static_cast<int>(value.size()));
        // Write the individual characters
        for (size_t i = 0; i < value.size(); i++)
        {
            writeBinary(stream, value[i]);
        }
    }

    /**
     * Reads a binary string from a stream
     */
    template <>
    inline void readBinary(std::istream & stream, std::string & value)
    {
        // Read the length
        int length;
        readBinary(stream, length);
        value.resize(length);
        
        // Read the characters
        for (int i = 0; i < length; i++)
        {
            readBinary(stream, value[i]);
        }
    }
    
    /**
     * Writes an eigen vector to a stream
     */
    template <>
    inline void writeBinary(std::ostream & stream, const Eigen::MatrixXf & v)
    {
        // Write the size of the matrix
        writeBinary(stream, static_cast<int>(v.rows()));
        writeBinary(stream, static_cast<int>(v.cols()));
        // Write the individual entries
        for (int r = 0; r < v.rows(); r++)
        {
            for (int c = 0; c < v.cols(); c++)
            {
                writeBinary(stream, v(r,c));
            }
        }
    }

    /**
     * Reads a binary string from a stream
     */
    template <>
    inline void readBinary(std::istream & stream, Eigen::MatrixXf & v)
    {
        // Read the size of the matrix
        int rows, cols;
        readBinary(stream, rows);
        readBinary(stream, cols);
        v.resize(rows, cols);
        // Write the individual entries
        for (int r = 0; r < v.rows(); r++)
        {
            for (int c = 0; c < v.cols(); c++)
            {
                readBinary(stream, v(r,c));
            }
        }
    }
    
    /**
     * Writes a vector to a stream
     */
    template <class T>
    inline void writeBinary(std::ostream & stream, const std::vector<T> & v)
    {
        writeBinary(stream, static_cast<int>(v.size()));
        for (size_t i = 0; i < v.size(); i++)
        {
            writeBinary(stream, v[i]);
        }
    }

    /**
     * Reads a vector of N elements from a stream. 
     */
    template <class T>
    inline void readBinary(std::istream & stream, std::vector<T> & v)
    {
        int N;
        readBinary(stream, N);
        v.resize(N);
        for (int i = 0; i < N; i++)
        {
            readBinary(stream, v[i]);
        }
    }
    
    /**
     * Writes a pair to a stream
     */
    template <class T, class S>
    inline void writeBinary(std::ostream & stream, const std::pair<T, S> & p)
    {
        writeBinary(stream, p.first);
        writeBinary(stream, p.second);
    }

    /**
     * Reads a pair from a stream. 
     */
    template <class T, class S>
    inline void readBinary(std::istream & stream, std::pair<T, S> & p)
    {
        readBinary(stream, p.first);
        readBinary(stream, p.second);
    }
    
    /**
     * Reads anything from a binary file
     */
    template <class T>
    void read(const std::string & filename, T & o) throw(IOException)
    {
        // Open the file
        std::ifstream stream(filename, std::ios::binary);
        if (!stream.is_open())
        {
            throw IOException("Could not open file.");
        }
        o.read(stream);
        stream.close();
    }
    
    /**
     * Reads anything from a binary file
     */
    template <class T>
    void write(const std::string & filename, const T & o) throw(IOException)
    {
        // Open the file
        std::ofstream stream(filename, std::ios::binary);
        if (!stream.is_open())
        {
            throw IOException("Could not open file.");
        }
        o.write(stream);
        stream.close();
    }
}

#endif
