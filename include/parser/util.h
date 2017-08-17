/**
 * This module contains necessary data structures such as lines and projected
 * rectangles as well as useful plot functions. 
 */

#ifndef PARSER_UTIL_H
#define PARSER_UTIL_H

#include <opencv2/opencv.hpp>
#include <functional>
#include <cstdlib>
#include <cmath>

#include "types.h"

#define IS_ONE(n, i) ((n >> i) & 0x00000001)

namespace parser {
    
    /**
     * This is a utility class for working with CV vectors. 
     */
    class VectorUtil {
    public:
        /**
         * Applies a homography to a single vector
         */
        static void applyHomography(const cv::Mat & homography, const Vec2 & x_in, Vec2 & x_out);
        
        /**
         * Calculates the two dimensional cross product
         */
        static floatT cross(const Vec2 & x, const Vec2 & y)
        {
            return x[0] * y[1] - x[1] * y[0];
        }
    };
    
    /**
     * This is a utility class for working with line segments. 
     */
    class LineSegmentUtil {
    public:
        /**
         * Applis a homography to a line segment
         */
        static void applyHomography(const cv::Mat & homography, const LineSegment & line_in, LineSegment & line_out);
        
        /**
         * Returns the distance from a single point to a line segment. 
         */
        static floatT calcPointDistance(const LineSegment & line, const Vec2 & point);
        
        /**
         * The direction of the hinge between the line segment (p2) and x. 
         */
        static floatT calcHingeDirection(const LineSegment & line, const Vec2 & point);
        
        /**
         * Precondition: x lies on the line l. Returns true if x is on the 
         * line segment. 
         */
        static bool isOnSegment(const LineSegment & line, const Vec2 & x);
        
        /**
         * Returns true if the two line segments intersect. 
         * 
         * The algorithm is due to a helpful stack overflow answer:
         * @see http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
         */
        static bool doIntersect(const LineSegment & line1, const LineSegment & line2);
        
        /**
         * Joins two line segments to produce one larger line segment.
         */
        static void join(const LineSegment & line_in1, const LineSegment & line_in2, LineSegment & line_out);
        
        /**
         * Calculates the distance between two line segments.
         */
        static floatT calcDistance(const LineSegment & line1, const LineSegment & line2);
    };

    /**
     * These are utilities for working with lines instead of line segments. 
     */
    class LineUtil {
    public:
        /**
         * Returns the distance from a single point to a line. Please note:
         * We do not consider line as a line segment here. 
         */
        static floatT calcPointDistance(const LineSegment & line, const Vec2 & point);
        
        /**
         * Returns the intersection point of two lines
         */
        static void calcIntersectionPoint(const LineSegment & line1, const LineSegment & line2, Vec2 & point);
    };
    
    /**
     * This class collects useful utility functions for working with rectangles.
     */
    class RectangleUtil 
    {
    public:
        /**
         * Applis a homography to a line segment
         */
        static void applyHomography(const cv::Mat & homography, const Rectangle & rect_in, Rectangle & rect_out);
        
        /**
         * Converts a rectangle to an openCV rectangle. Works only for axis
         * aligned rectangles.
         */
        static void convertToOpenCV(const Rectangle & rect, cv::Rect_<floatT> & out);
        
        /**
         * Converts a rectangle from open CV. Works only for axis aligned 
         * rectangles.
         */
        static void convertFromOpenCV(const cv::Rect_<floatT> & in, Rectangle & rect);
        
        /**
         * Calculates the union of two axis aligned rectangles.
         */
        static void calcUnion(const Rectangle & rect_in1, const Rectangle & rect_in2, Rectangle & rect_out);
        
        /**
         * Calculates the intersection of two axis aligned rectangles.
         */
        static void calcIntersection(const Rectangle & rect_in1, const Rectangle & rect_in2, Rectangle & rect_out);
        
        /**
         * Calculates the smallest rectangle that can encloses the other one. The
         * resulting rectangle will also have integer coordinates.
         */
        static void calcTightAxisAlignedFit(const Rectangle & rect_in, Rectangle & rect_out);
        
        /**
         * Creates the unit rectangle
         */
        static void getUnitRectangle(Rectangle & rect);
        
        /**
         * Returns true if a point is inside an axis aligned rectangle
         */
        static bool isInsideAxisAlignedRectangle(const Rectangle & rect, const Vec2 & p);
        
        /**
         * Calculates the intersection over union score
         */
        static float calcIOU(const Rectangle & r, const Rectangle & q);
        
        /**
         * Returns true if the two rectangles are arranged horizontally
         */
        static bool isHorizontalArrangement(const Rectangle & r, const Rectangle & p);
        
        /**
         * Returns true if the two rectangles are similar. 
         */
        static bool areSimilar(const Rectangle & r, const Rectangle & p, float threshold = 30);
    };
    
    /**
     * This class can be used to plot polygons.
     */
    class PlotUtil {
    public:
        /**
         * Plots a polygon without filling its inside. 
         */
        static void plotRectangle(cv::Mat & img, const Rectangle & polygon, const cv::Scalar & color, int thickness=1)
        {
            std::vector<cv::Point> poly;
            poly.push_back(cv::Point(polygon[0][0], polygon[0][1]));
            poly.push_back(cv::Point(polygon[1][0], polygon[1][1]));
            poly.push_back(cv::Point(polygon[2][0], polygon[2][1]));
            poly.push_back(cv::Point(polygon[3][0], polygon[3][1]));
            const cv::Point *pts = (const cv::Point*) cv::Mat(poly).data;
            int npts = cv::Mat(poly).rows;

            cv::polylines(img, &pts, &npts, 1, true, color, thickness);
        }
        
        /**
         * Plots a polygon with filling its inside. 
         */
        static void plotRectangleFill(cv::Mat & img, const Rectangle & polygon, const cv::Scalar & color)
        {
            std::vector<cv::Point> poly;
            poly.push_back(cv::Point(polygon[0][0], polygon[0][1]));
            poly.push_back(cv::Point(polygon[1][0], polygon[1][1]));
            poly.push_back(cv::Point(polygon[2][0], polygon[2][1]));
            poly.push_back(cv::Point(polygon[3][0], polygon[3][1]));
            const cv::Point *pts = (const cv::Point*) cv::Mat(poly).data;
            int npts = cv::Mat(poly).rows;

            cv::fillPoly(img, &pts, &npts, 1, color);
        }
        
        /**
         * Plots a line
         */
        static void plotLineSegment(cv::Mat & img, const LineSegment & line, const cv::Scalar & color)
        {
            cv::Point p1;
            cv::Point p2;
            
            p1.x = static_cast<int>(std::round(line[0][0]));
            p1.y = static_cast<int>(std::round(line[0][1]));
            p2.x = static_cast<int>(std::round(line[1][0]));
            p2.y = static_cast<int>(std::round(line[1][1]));
            
            cv::line(img, p1, p2, color, 1);
        }
    };
    
    /**
     * This class contains useful functions for displaying images. 
     */
    class Util {
    public:
        /**
         * Shows an image in a new window and waits for a key press.
         */
        static void imshow(const cv::Mat & img, const std::string & title);
        
        /**
         * Shows an image in a new window and waits for a key press.
         */
        static void imshow(const cv::Mat & img)
        {
            imshow(img, "Debug");
        }
        
        /**
         * Exhaustively checks all possible subsets calls 
         * the callback every time.
         */
        template <class T>
        static void exhaustiveSubsetSearch(const std::vector<T> & elements, std::function<void(const std::vector<T> &)> callback)
        {
            const int N = static_cast<int>(elements.size());
            assert(N <= 8*sizeof(int));
            // We use an integer counter in order to list all subsets
            const unsigned int numberOfSubsets = (1 << N);
            
            for (unsigned int subsetMask = 0; subsetMask < numberOfSubsets; subsetMask++)
            {
                int subsetSize = 0;
                for (int i = 0; i < static_cast<int>(sizeof(int)*8); i++)
                {
                    if (IS_ONE(subsetMask, i))
                    {
                        subsetSize++;
                    }
                }
                
                // Set up the subset
                std::vector<T> subset(subsetSize);
                int k = 0;
                for (int n = 0; n < N; n++)
                {
                    if (IS_ONE(subsetMask, n))
                    {
                        subset[k++] = elements[n];
                    }
                }
                
                callback(subset);
            }
        }
        
        /**
         * Returns true if the two vectors are disjoint
         */
        template <class T>
        static bool areDisjoint(const std::vector<T> & v1, const std::vector<T> & v2)
        {
            for (size_t n = 0; n < v1.size(); n++)
            {
                if (std::find(v2.begin(), v2.end(), v1[n]) != v2.end())
                {
                    return false;
                }
            }
            return true;
        }
        
        /**
         * Normalizes an array of floats using the zscore
         */
        template <int N>
        static void normalizeZScore(float numbers[])
        {
            // Compute the mean and variance
            float mean = 0; 
            float variance = 0;
            for (int n = 0; n < N; n++)
            {
                mean += numbers[n];
                variance += numbers[n]*numbers[n];
            }
            mean /= N;
            variance /= N;
            variance -= mean*mean;
            variance = std::sqrt(variance);
            
            for (int n = 0; n < N; n++)
            {
                numbers[n] = (numbers[n] - mean)/variance;
            }
        }
        
        /**
         * Computes the smallest distance to a multiple of the given fraction.
         */
        static float calcClosestModFractionDistance(float x, float fraction)
        {
            int k = static_cast<int>(std::round(1./fraction)); 
            int j = 0;
            for (int i = 1; i <= k; i++)
            {
                if (std::abs(x - i*fraction) < std::abs(x - j*fraction))
                {
                    j = i;
                }
            }
            
            return std::abs(x - j*fraction);
        }
        
        /**
         * Converts a gray value image to an unsigned char array.
         */
        template <typename T, typename S>
        static T* toArrayImg(const cv::Mat & img)
        {
            T* result = new T[img.rows * img.cols];
            
            for (int x = 0; x < img.rows; x++)
            {
                for (int y = 0; y < img.cols; y++)
                {
                    result[x*img.cols + y] = static_cast<T>(img.at<S>(x,y));
                }
            }
            return result;
        }
        
        /**
         * Converts an array to an opencv image.
         */
        template <typename T>
        static void toOpenCVImg(const T* image, int rows, int cols, cv::Mat & result)
        {
            if (image == 0) return;
            
            for (int x = 0; x < rows; x++)
            {
                for (int y = 0; y < cols; y++)
                {
                    result.at<T>(x,y) = image[x*cols + y];
                }
            }
        }

        /**
         * Splits an edge image into horizontal and vertical edges.
         */
        static void splitEdgeImage(const cv::Mat & edgeImage, cv::Mat & horizontalEdges, cv::Mat & verticalEdges);
    };
}


#endif