#ifndef PARSER_DETECTOR_H
#define PARSER_DETECTOR_H

/**
 * This file contains the line and rectangle detectors. Each of these detectors
 * is set up as a processing pipeline with several possible processing steps. 
 */

#include "types.h"
#include "util.h"
#include "opencvModels.h"

#include <opencv2/opencv.hpp>

namespace parser {
    /**
     * Detects lines that converge to a vanishing point of the region of 
     * interest.
     */
    class LineDetector {
    public:
        /**
         * The parameters of this class
         */
        class Model {
        public:
            /**
             * The default constructor
             */
            Model () :  epsilon(1.5),
                        minLength(30),
                        maxJoinDistance(18), 
                        maxOrthogonalJoinDistance(1.25),
                        hough (0.5, CV_PI/180, 20, 2, 13) {}
            
            /**
             * The precision of the lines
             */
            double epsilon;
            /**
             * The minimum length of a line segment
             */
            double minLength;
            /**
             * The maximum join distance of two line segments
             */
            double maxJoinDistance;
            /**
             * The maximum join precision of two line segments
             */
            double maxOrthogonalJoinDistance;
            /**
             * The parameters for the hough line transform
             */
            OpenCVModels::Hough hough;
        };
        
        /**
         * Default constructor
         */
        LineDetector() : model() {}
        
        /**
         * Detects the lines and stores them as line segments in the passed
         * array. 
         */
        void detectLines(   const cv::Mat & image, 
                            std::vector< LineSegment > & lineSegments, 
                            bool horizontal) const;
        
        /**
         * Performs line joining on the passed array
         */
        void joinLines( std::vector<LineSegment> & lineSegments, 
                        bool horizontal) const;
        
        /**
         * Plots the lines onto the image
         */
        void visualize( cv::Mat & image, 
                        const std::vector<LineSegment> & lineSegments, 
                        const cv::Scalar & color) const;
        
        /**
         * Adds the bounding box lines to the line set
         */
        void addBoundingBoxLines(   const cv::Mat & image,
                                    std::vector<LineSegment> & lineSegments, 
                                    bool horizontal) const;
        
        /**
         * Filters out lines that are too short
         */
        void filterShortLines(  const std::vector<LineSegment> & lineSegmentsIn, 
                                std::vector<LineSegment> & lineSegmentsOut) const;
        
        /**
         * The parameter model
         */
        Model model;
    };
    
    /**
     * Detects rectangles in an image
     */
    class RectangleDetector {
    public:
        /**
         * The parameter class for this detector
         */
        class Model {
        public:
            /**
             * The default constructor
             */
            Model() : numSamples(200000), maxScore(20), maxIOU(0.95) {}
            
            /**
             * The number of proposals to sample
             */
            int numSamples;
            /**
             * The maximum rectangle score
             */
            double maxScore;
            /**
             * The maximum intersection over union score. If a rectangle has
             * an IOU above the threshold, it's not added to the list of 
             * detected rectangles
             */
            double maxIOU;
        };
        
        /**
         * Default constructor
         */
        RectangleDetector() : model() {}
        
        /**
         * Detects rectangles in an image. The result is a list of rectangles
         * and their projections to the unit rectangle.
         */
        void detectRectangles(  const cv::Mat & image, 
                                const std::vector<LineSegment> & lineSegmentsH, 
                                const std::vector<LineSegment> & lineSegmentsV,
                                std::vector<Rectangle> & result, 
                                std::function<bool(const Rectangle &)> callbak) const;
        
        /**
         * Plots the rectangles onto the image
         */
        void visualize( cv::Mat & image, 
                        const std::vector<Rectangle> & rectangles, 
                        const cv::Scalar & color) const;
        
    private:
        /**
         * The model for the detector
         */
        Model model;
    };
}

#endif
