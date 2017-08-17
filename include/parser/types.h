#ifndef PARSER_TYPES_H
#define PARSER_TYPES_H

#include <opencv2/opencv.hpp>

/**
 * This makro helps you creating custom exception classes which accept error messages as constructor arguments. 
 * You can define a new exception class by: DEFINE_EXCEPTION(classname)
 * You can throw a new exception by: throw classname("Error message");
 */
#define DEFINE_EXCEPTION(classname)		\
        class classname : public std::exception {	\
        public:		\
                classname() { this->ptrMessage = 0; };	\
                classname(const char* _ptrMessage) : ptrMessage(_ptrMessage) {};	\
                classname(std::string str) {_msg = str; ptrMessage = _msg.c_str(); };	\
                classname(const char* _ptrMessage, int l) : ptrMessage(_ptrMessage) { };	\
        virtual ~classname() throw() {}; \
                virtual const char* what() const throw() { return (this->ptrMessage != 0 ? this->ptrMessage : "No Message"); }	\
        private: \
        std::string _msg; \
                const char* ptrMessage;		\
        }; 

/**
 * This file contains all important type definitions.
 */
namespace parser {
    /**
     * For faster compile time and cleaner code, we define the floating point
     * type here instead of using templates. 
     */
    typedef float floatT;
    /**
     * This is the default 2D vector type
     */
    typedef cv::Vec<floatT, 2> Vec2;
    /**
     * This is the default 3D vector type
     */
    typedef cv::Vec<floatT, 3> Vec3;
    
    /**
     * The float matrix type. This should be consistent with floatT.
     */
    #define CV_FLOAT_T CV_32FC1
    
    /**
     * This function allows us to print vectors to streams such as std::cout
     */
    /*
    template <class T, int V>
    inline std::ostream& operator<<(std::ostream& _os, const cv::Vec<T, V>& v)
    {
        _os << '[';
        for (int i = 0; i < V; i++)
        {
            _os << v[i];
            if (i < V-1)
            {
                _os << ',';
            }
        }
        _os << ']';
        return _os;
    }*/

    /**
     * Define the general parser exception
     */
    DEFINE_EXCEPTION(ParserException);
    
    /**
     * This is the base class for polygons. We will only use it for lines and
     * projected (four sided) rectangles. 
     */
    template <int V>
    class Polygon {
    public:
        /**
         * Destructor
         */
        virtual ~Polygon()  {}
        
        /**
         * Gives access to the vertices
         */
        const Vec2 & at(int i) const
        {
            if (0 <= i && i < V)
            {
                return vertices[i];
            }
            else
            {
                throw ParserException("Invalid vertex index.");
            }
        }
        
        /**
         * Gives access to the vertices
         */
        Vec2 & at(int i)
        {
            if (0 <= i && i < V)
            {
                return vertices[i];
            }
            else
            {
                throw ParserException("Invalid vertex index.");
            }
        }
        
        /**
         * Gives access to the vertices
         */
        const Vec2 & operator[](int i) const
        {
            return at(i);
        }
        
        /**
         * Gives access to the vertices
         */
        Vec2 & operator[](int i)
        {
            return at(i);
        }
        
        /**
         * Returns the vertices.
         */
        Vec2* getVertices()
        {
            return &vertices[0];
        }
        
        /**
         * Writes the polygon to a file
         */
        void write(cv::FileStorage &fs) const 
        {
            fs << "vertices" << "[:";
            for (int i = 0; i < V; i++)
            {
                fs << at(i);
            }
            fs << "]";
        }
        
        /**
         * Reads a polygon from a file
         */
        void read(const cv::FileStorage & fs)
        {
            for (int i = 0; i < V; i++)
            {
                floatT x;
                vertices[i][0] = (floatT) fs["vertices"][i][0];
                vertices[i][1] = (floatT) fs["vertices"][i][1];
            }
        }
        
        /**
         * Returns the maximum x coordinate
         */
        floatT maxX() const
        {
            floatT extremum = vertices[0][0];
            for (int v = 0; v < V; v++)
            {
                extremum = std::max(extremum, vertices[v][0]);
            }
            return extremum;
        }
        
        /**
         * Returns the minimum x coordinate
         */
        floatT minX() const
        {
            floatT extremum = vertices[0][0];
            for (int v = 0; v < V; v++)
            {
                extremum = std::min(extremum, vertices[v][0]);
            }
            return extremum;
        }
        /**
         * Returns the maximum y coordinate
         */
        floatT maxY() const
        {
            floatT extremum = vertices[0][1];
            for (int v = 0; v < V; v++)
            {
                extremum = std::max(extremum, vertices[v][1]);
            }
            return extremum;
        }
        
        /**
         * Returns the minimum y coordinate
         */
        floatT minY() const
        {
            floatT extremum = vertices[0][1];
            for (int v = 0; v < V; v++)
            {
                extremum = std::min(extremum, vertices[v][1]);
            }
            return extremum;
        }
        
        /**
         * Serializes the polygon to the stream
         */
        void serialize(std::ostream & stream) const
        {
            for (int v = 0; v < V; v++)
            {
                stream << vertices[v];
                if (v != V-1)
                {
                    stream << ',';
                }
            }
        }
        
        /**
         * The vertices
         */
        Vec2 vertices[V];
    };
    
    /**
     * This class represents a line segment. It is defined in terms of the two
     * end points in the R^2 plane. 
     */
    class LineSegment : public Polygon<2> {
    public:
        /**
         * Creates a new unit line segment
         */
        LineSegment() {}
        
        /**
         * Creates a new line segment from two end points
         */
        LineSegment(const Vec2 & p0, const Vec2 & p1)
        {
            at(0) = p0;
            at(1) = p1;
        }
        
        /**
         * Destructor
         */
        virtual ~LineSegment()  {}
        
        /**
         * Returns the euclidean length of the line segment
         */
        floatT getLength() const
        {
            return cv::norm(at(0), at(1), CV_L2);
        }
        
        /**
         * Returns the normal form of a line defined by two points
         * 
         */
        Vec3 getLineNormal() const;
        
        /**
         * Normalizes the segment in a way that the first point is always
         * closer to the origin than the second point.
         */
        void normalize()
        {
            if (cv::norm(at(0)) > cv::norm(at(1)))
            {
                std::swap(at(0), at(1));
            }
        }
    };
    
    inline std::ostream& operator<<(std::ostream& os, const LineSegment& line)
    {
        line.serialize(os);
        return os;
    }

    /**
     * This class represents a projected rectangle. It basically corresponds to
     * a four sided polygon. The rectangle will be normalized such that p0 is
     * the point closest to the origin and the other points follow in clockwise
     * order. 
     */
    class Rectangle : public Polygon<4> {
    public:
        /**
         * Default constructor
         */
        Rectangle() : Polygon<4>() {}
        
        /**
         * Construct a rectangle from four corner points
         */
        Rectangle(const Vec2 & p0, const Vec2 & p1, const Vec2 & p2, const Vec2 & p3) : Polygon<4>()
        {
            at(0) = p0;
            at(1) = p1;
            at(2) = p2;
            at(3) = p3;
        }
        
        /**
         * Returns true if the rectangle is oriented correctly. This means that
         * p0 is the point closest to the origin. 
         */
        bool isOrientedCorrectly() const
        {
            return  cv::norm(at(0)) <= cv::norm(at(1)) && 
                    cv::norm(at(0)) <= cv::norm(at(2)) && 
                    cv::norm(at(0)) <= cv::norm(at(3));
        }
        
        /**
         * Rotates the vertices of the rectangle. 
         */
        void rotate() 
        {
            std::swap(at(0), at(1));
            std::swap(at(0), at(2));
            std::swap(at(0), at(3));
        }
        
        /**
         * Normalizes the rectangle such that it has the following structure:
         * p0----p1
         * |     |
         * p3----p2
         */
        void normalize();
        
        /**
         * Returns the width of axis aligned rectangles.
         */
        floatT getWidth() const
        {
            return this->maxX() - this->minX();
        }
        
        /**
         * Returns the height of axis aligned rectangles.
         */
        floatT getHeight() const
        {
            return this->maxY() - this->minY();
        }
        
        /**
         * Returns the area of axis aligned rectangles.
         */
        floatT getArea() const
        {
            return getWidth() * getHeight();
        }
        
        Vec2 getCenter() const
        {
            Vec2 v;
            v[0] = at(0)[0] + getWidth()/2.0f;
            v[1] = at(0)[1] + getHeight()/2.0f;
            return v;
        }
    };
    
    inline std::ostream& operator<<(std::ostream& os, const Rectangle& r)
    {
        r.serialize(os);
        return os;
    }

}

#endif