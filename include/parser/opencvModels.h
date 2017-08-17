#ifndef OPENCVMODLELS_H
#define OPENCVMODLELS_H

#include <opencv2/opencv.hpp>

#include "util.h"

namespace parser {
    
    /**
     * This class contains several model classes that store parameters for certain
     * open CV functions.
     */
    class OpenCVModels {
    public:
        /**
         * Parameters for the canny edge detector. 
         * @see http://docs.opencv.org/modules/imgproc/doc/feature_detection.html?highlight=canny#canny
         */
        class Canny {
        public:
            Canny() : 
                    threshold1(50), 
                    threshold2(200), 
                    aperture(3) {}
            
            Canny(double _threshold1, double _threshold2, int _aperture) : 
                    threshold1(_threshold1), 
                    threshold2(_threshold2), 
                    aperture(_aperture) {}

            double threshold1;
            double threshold2;
            int aperture;
        };
        
        /**
         * Parameters for the probabilistic hough transform that is used
         * in order to detect line segments
         * @see http://docs.opencv.org/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp
         */
        class Hough {
        public:
            Hough() : 
                    rho(0.25),
                    theta(CV_PI/360), 
                    threshold(55),
                    minLineLength(2), 
                    maxLineGap(25) {}
            
            Hough(  double _rho, 
                    double _theta, 
                    int _threshold, 
                    double _minLineLength, 
                    double _maxLineGap) : 
                    rho(_rho), 
                    theta(_theta), 
                    threshold(_threshold), 
                    minLineLength(_minLineLength), 
                    maxLineGap(_maxLineGap) {}

            /**
             * Discretization levels for the two line parameters
             */
            double rho;
            double theta;
            int threshold;
            double minLineLength;
            double maxLineGap;
        };
    };
    
}

#endif
