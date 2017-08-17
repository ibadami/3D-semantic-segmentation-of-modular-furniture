#ifndef PARSER_PREPROCESSING_H
#define PARSER_PREPROCESSING_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "libforest/libforest.h"

#include "types.h"
#include "util.h"

namespace parser {
    
    class Processing {
    public:
        /**
         * Takes as input a gray scale uchar image and outputs the l2 gradient
         * magnitude image.
         */
        static void computeGradientMagnitudeImage(const cv::Mat & in, cv::Mat & out, float threshold);
        
        /**
         * Takes as input a gray scale uchar image and outputs the l2 gradient
         * magnitude image.
         */
        static void computeGradientMagnitudeImageFloat(const cv::Mat & in, cv::Mat & out);
        
        /**
         * Normalizes a float image such that it integrates to 1
         */
        static void normalizeFloatImageLebesgue(cv::Mat & in);
        
        /**
         * Takes as input a gray scale uchar image (CV_8UC1) and outputs a two
         * channel floating point image. The first channel corresponds to the 
         * derivatives in x direction and the second channel corresponds to the
         * derivatives in y direction.
         */
        static void computeGradients(const cv::Mat & in, cv::Mat & out, float threshold);

        /**
         * Performs edge detection on a gray scale uchar image (CV_8UC1). The result
         * is again a gray scale uchar image. We use the canny edge detector for
         * this step. The parameters are determined using Otsu's method on the 
         * gradient magnitude image. 
         */
        static void computeCannyEdges(const cv::Mat & in, cv::Mat & out);

        /**
         * Computes the homography from the source to the target rectangle
         */
        static void computeHomography(const Rectangle & source, const Rectangle & target, cv::Mat & homography);

        /**
         * This processor computes a distance transform of a given image binary 
         * (0,255) image. The output image is a one channel floating point (CV_32FC1)
         * image. 
         */
        static void computeDistanceTransform(const cv::Mat & in, cv::Mat & out);

        /**
         * Converts an image to floating point. 
         */
        static void convertToFloat(const cv::Mat & in, cv::Mat & out);

        /**
         * Computes the feature vector for the rectangle filter from a rectangle and
         * its rectified image.
         */
        static void computeRectangleFilterFeatures(const cv::Mat & image, const Rectangle & r, libf::DataPoint & point);

        /**
         * Returns the value of a subpixel using linear interpolation
         */
        static cv::Vec3b getSubpixel(const cv::Mat & in, const Vec2 & point);

        /**
         * Returns the value of a subpixel using Gaussian interpolation
         */
        static float getSubpixelGaussian(const cv::Mat & in, const Vec2 & point, float bandwidth);

        /**
         * Warps the region defined by the input rectangle in the input image to 
         * the region defined by the output rectangle in the output image. The input
         * image must be of floating point type.
         */
        static void warpImage(const cv::Mat & in, const Rectangle & rectIn, cv::Mat & out, const Rectangle & rectOut);

        /**
         * Warps the region defined by the input rectangle in the input image to 
         * the region defined by the output rectangle in the output image using 
         * Gaussian averages. The input image must be of floating point type.
         */
        static void warpImageGaussian(const cv::Mat & in, const Rectangle & rectIn, cv::Mat & out, const Rectangle & rectOut, float bandwidth);

        /**
         * Computes the rectified region of interest.
         */
        static void computeRectifiedRegionOfInterest(const Rectangle & regionOfInterest, int size, Rectangle & result);

        /**
         * The processor approximately rectifies the region in the input image
         * that is defined by the rectangle. 
         */
        static void rectifyRegion(const cv::Mat & in, const Rectangle & region, int size, cv::Mat & out);
        
        /**
         * Computes the weighted Hausdorff distance between A and B on the 
         * given mask area.
         */
        static float computeHausdorffDistance(const cv::Mat & A, const cv::Mat & dtA, const cv::Mat & B, const cv::Mat dtB, const cv::Mat & mask);
        
        /**
         * Performs thinning of a binary image
         * Paper: http://www-prima.inrialpes.fr/perso/Tran/Draft/gateway.cfm.pdf
         * Implementation: http://opencv-users.1802565.n2.nabble.com/Morphological-thinning-operation-from-Z-Guo-and-R-W-Hall-quot-Parallel-Thinn-td4225544.html
         */
        static void thinBinary(const cv::Mat & inputarray, cv::Mat & outputarray);
        
        /**
         * Computes the largest rectangle packing for a given set of rectangles. 
         */
        static int computeRectanglePacking(const std::vector<Rectangle> & rectangles, std::vector<Rectangle> & packing);
        
        /**
         * Computes the smallest rectangle cover for a given set and a given set 
         * of rectangles.  
         */
        static int computeRectangleCover(float roiSize, const std::vector<Rectangle> & rectangles, std::vector<Rectangle> & packing);
        
        /**
         * Computes the smallest rectangle cover for a given set and a given set 
         * of rectangles.  
         */
        //static int computeRectangleCoverMCMC(float roiSize, const std::vector<Rectangle> & rectangles, std::vector<Rectangle> & packing);
        
        /**
         * Computes the largest rectangle packing for a given set of rectangles. 
         */
        static int computeRectanglePackingGreedy(const std::vector<Rectangle> & rectangles, std::vector<Rectangle> & packing);
        
        /**
         * Computes the largest rectangle packing for a given set of rectangles. 
         */
        //static int computeRectanglePackingMCMC(const std::vector<Rectangle> & rectangles, std::vector<Rectangle> & packing);
        
        /**
         * Adds 1px wide borders around an image
         */
        static void add1pxBorders(cv::Mat & images);
        
    private:
        static void ThinSubiteration1(const cv::Mat & pSrc, cv::Mat & pDst);
        static void ThinSubiteration2(const cv::Mat & pSrc, cv::Mat & pDst);
    };
}


#endif
