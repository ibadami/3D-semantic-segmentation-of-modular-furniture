#include <iostream>
#include <random>
#include <set>
#include <string>
#include <sstream>
#include <boost/filesystem.hpp>
#include <netdb.h>
#include <algorithm>
#include <regex>
#include <cmath>
#include <chrono>


#include "parser/parser.h"
#include "parser/processing.h"
#include "parser/util.h"
#include "parser/detector.h"
#include "parser/json.h"
#include "parser/kde.h"
#include "parser/Stopwatch.h"
#include "parser/jump_moves.h"
#include "parser/diffuse_moves.h"
#include "parser/rjmcmc_sa.h"
#include "libforest/libforest.h"
#include "gurobi_c++.h"
#include <boost/filesystem.hpp>
#include "parser/canny.h"
#include "pam.h"



#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/png_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
//#include <pcl/visualization/pcl_visualizer.h>

/*
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>

#include <pcl/console/parse.h>*/

using namespace parser;

/**
 * Set this to boosted forest if you want to switch
 */
typedef libf::RandomForest<libf::DecisionTree> Forest;
typedef libf::RandomForestLearner<libf::DecisionTreeLearner> ForestLearner;

float rectangleAcceptanceThreshold = 0.65;
float rectangleDetectionThreshold = 6;// High Recall (low precision, almost fixed F1 measure) with high value (but over segmentation)
const float pruningThreshold = 0.65;

////////////////////////////////////////////////////////////////////////////////
//// CabinetParser
////////////////////////////////////////////////////////////////////////////////

void CabinetParser::parse(  const cv::Mat & image,
                            const cv::Mat & imageDepth,
                            const Rectangle & regionOfInterest, 
                            std::vector<Part> & parts)
{
    // Compute the multi channel image. For this purpose, we need to warp the 
    // region of interest
    cv::Mat rectifiedMultiChannelImage;
    extractRectifiedMultiChannelImage(image, regionOfInterest, rectifiedMultiChannelImage);
    
    //Depth Image
    cv::Mat rectifiedMultiChannelImageDepth;
    extractRectifiedMultiChannelImage(imageDepth, regionOfInterest, rectifiedMultiChannelImageDepth);
    std::vector<cv::Mat> depthChannels;
    cv::split(rectifiedMultiChannelImageDepth, depthChannels);
    cv::Mat intensityDepth;
    depthChannels[EDGE_DETECTOR_CHANNEL_INTENSITY].convertTo(intensityDepth,CV_8UC1);
    // Apply the edge detector to multichannel image
    cv::Mat edgeImage;
    applyEdgeDetector(rectifiedMultiChannelImage, edgeImage, 0);

    // Apply the edge detector to multi channel depth image
    cv::Mat edgeImageDepth;
    applyEdgeDetector(rectifiedMultiChannelImageDepth, edgeImageDepth, 1);

#if 0
	cv::Mat edgeImageDebug;
	edgeImage.copyTo(edgeImageDebug);
	cv::imshow("Edge Image",edgeImageDebug);
	cv::waitKey();
#endif

#if INCLUDE_DEPTH
    // Stabilizing RGB edges with depth edges
    cv::bitwise_or(edgeImage, edgeImageDepth, edgeImage);
#endif

#if 0
	cv::Mat edgeImageDepthDebug;
	edgeImageDepth.copyTo(edgeImageDepthDebug);
	cv::imshow("Edge Image Depth",edgeImageDepthDebug);
	cv::waitKey();

    cv::Mat edgeImageDebugBoosted;
    edgeImage.copyTo(edgeImageDebugBoosted);
    cv::imshow("Edge Image Boosted",edgeImageDebugBoosted);
    cv::waitKey();

#endif

    // Get the canny edge image
    std::vector<cv::Mat> channels;
    cv::split(rectifiedMultiChannelImage, channels);
    cv::Mat cannyEdges;
    Processing::computeCannyEdges(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], cannyEdges);
    
#if 0
    // Get the canny edge depth image
    cv::Mat cannyEdgesDepth;
    cv::Mat GMDepth;
    depthChannels[EDGE_DETECTOR_CHANNEL_GM].convertTo(GMDepth,CV_8UC1);
    Processing::computeCannyEdges(GMDepth, cannyEdgesDepth);
#endif

#if 0
	cv::Mat cannyEdgeDebug;
	cannyEdges.copyTo(cannyEdgeDebug);
	cv::imshow("Canny Edge Image",cannyEdgeDebug);
	cv::waitKey();
#endif

#if 0
    cv::bitwise_or(cannyEdges, cannyEdgesDepth, cannyEdges);
    cv::Mat cannyEdgesFloat = cv::Mat::zeros(cannyEdges.rows, cannyEdges.cols, CV_32FC1);
    Processing::computeGradientMagnitudeImageFloat(cannyEdges, cannyEdgesFloat);
#endif

#if 0
	visualizeFloatImage(cannyEdgesFloat);
#endif

#if 0
    addWeighted(cannyEdges, 1.0, cannyEdgesDepth, 1.0, 0.0, cannyEdges);
    Processing::computeGradientMagnitudeImageFloat(cannyEdges, cannyEdgesFloat);
#endif

#if 0
	visualizeFloatImage(cannyEdgesFloat);
#endif

#if 0
    Processing::computeGradientMagnitudeImageFloat(cannyEdgesDepth, cannyEdgesFloat);
	visualizeFloatImage(cannyEdgesFloat);
#endif

#if 0
	cv::Mat cannyEdgeDebugBoosted;
	cannyEdges.copyTo(cannyEdgeDebugBoosted);
	cv::imshow("Canny Edge Image Boosted",cannyEdgeDebugBoosted);
	cv::waitKey();
	
	cv::Mat cannyEdgeDepthDebug;
	cannyEdgesDepth.copyTo(cannyEdgeDepthDebug);
	cv::imshow("Canny Edge Image Depth",cannyEdgeDepthDebug);
	cv::waitKey();
#endif
        
    // Detect rectangles
    std::vector<Rectangle> partHypotheses;
    detectRectangles(edgeImage, cannyEdges, partHypotheses);
    std::cout << std::setw(5) << partHypotheses.size() << " rectangles detected" << std::endl;

#if 0
    cv::Mat demo(rectifiedMultiChannelImage.rows, rectifiedMultiChannelImage.cols, CV_8UC3);
    demo = cv::Scalar(0);
    cv::imshow("Edge Image", edgeImage);

    for (size_t n = 0; n < partHypotheses.size(); n++)
    {
        demo = cv::Scalar(0);
        PlotUtil::plotRectangle(demo, partHypotheses[n], cv::Scalar(255,0,255));
        cv::imshow("test", demo);
        cv::waitKey();
    }    
    //exit(0);    
#endif



#if 0
    cv::Mat demo2(rectifiedMultiChannelImage.rows, rectifiedMultiChannelImage.cols, CV_8UC3);
    demo2 = cv::Scalar(0);
    cv::imshow("Edge Image", edgeImage);

    for (size_t n = 0; n < partHypotheses.size(); n++)
    {
        demo2 = cv::Scalar(0);
        PlotUtil::plotRectangle(demo2, partHypotheses[n], cv::Scalar(255,0,255));
        cv::imshow("test", demo2);
        cv::waitKey();
    }
    //exit(0);
#endif

    /**
     * Proposal Selection using RJMCMC
     */
    std::cout<<"Selecting from un-Pruned pool of rectangles"<<std::endl;
    selectParts(rectifiedMultiChannelImage, intensityDepth, edgeImage, partHypotheses, parts);

}

void CabinetParser::visualizeSegmentation(const cv::Mat & image, const Rectangle & ROI, const std::vector<Part> & parts, cv::Mat & display)
{
    // Collect the rectangles
    std::vector<Rectangle> rectifiedRects, unrectifiedRects;
    for (size_t p = 0; p < parts.size(); p++)
    {
        rectifiedRects.push_back(parts[p].rect);
    }
    
    unrectifyParts(ROI, rectifiedRects, unrectifiedRects);
    
    // Plot the parts (only the inner stuff)
    cv::Mat inner = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    for (size_t r = 0; r < unrectifiedRects.size(); r++)
    {
        switch (parts[r].label)
        {
            case 0:
                PlotUtil::plotRectangleFill(inner, unrectifiedRects[r], cv::Scalar(0,0,255));
                break;
            case 1:
                PlotUtil::plotRectangleFill(inner, unrectifiedRects[r], cv::Scalar(0,255,0));
                break;
            case 2:
                PlotUtil::plotRectangleFill(inner, unrectifiedRects[r], cv::Scalar(255,0,0));
                break;
        }
    }
    
    // Combine the images
    display = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    for (int x = 0; x < image.cols; x++)
    {
        for (int y = 0; y < image.rows; y++)
        {
            for (int i = 0; i < 3; i++)
            {
                if (inner.at<cv::Vec3b>(y,x)[0] != 0 || inner.at<cv::Vec3b>(y,x)[1] != 0 || inner.at<cv::Vec3b>(y,x)[2] != 0)
                {
                    display.at<cv::Vec3b>(y,x)[i] = 0.8*image.at<cv::Vec3b>(y,x)[i] + 0.2 * inner.at<cv::Vec3b>(y,x)[i];
                }
                else
                {
                    display.at<cv::Vec3b>(y,x)[i] = image.at<cv::Vec3b>(y,x)[i];
                }
            }
        }
    }
    
    // Plot the borders
    for (size_t r = 0; r < unrectifiedRects.size(); r++)
    {
        switch (parts[r].label)
        {
            case 0:
                PlotUtil::plotRectangle(display, unrectifiedRects[r], cv::Scalar(0,0,255), 2);
                break;
            case 1:
                PlotUtil::plotRectangle(display, unrectifiedRects[r], cv::Scalar(0,255,0), 2);
                break;
            case 2:
                PlotUtil::plotRectangle(display, unrectifiedRects[r], cv::Scalar(255,0,0), 2);
                break;
        }
    }
}

void CabinetParser::removeNonRectanglePixels(const std::vector<Rectangle> & rectangles, cv::Mat & edgeMap)
{
    // Plot all rectangles
    cv::Mat rectanglePlot = cv::Mat::zeros(edgeMap.rows, edgeMap.cols, CV_8UC1);
    for (size_t r = 0; r < rectangles.size(); r++)
    {
        PlotUtil::plotRectangle(rectanglePlot, rectangles[r], 255);
    }
    // Compute the distance transform in order to distinguish the pixels
    cv::Mat distanceTransform;
    Processing::computeDistanceTransform(rectanglePlot, distanceTransform);
    
    // Clear the edge map
    for (int x = 0; x < edgeMap.cols; x++)
    {
        for (int y = 0; y < edgeMap.rows; y++)
        {
            if (distanceTransform.at<float>(y,x) >= 3)
            {
                edgeMap.at<uchar>(y,x) = 0;
            }
        }
    }
}

void CabinetParser::removeNonRectanglePixels2(const std::vector<Rectangle> & rectangles, cv::Mat & edgeMap)
{
    // Plot all rectangles
    cv::Mat rectanglePlot = cv::Mat::zeros(edgeMap.rows, edgeMap.cols, CV_8UC1);
    for (size_t r = 0; r < rectangles.size(); r++)
    {
        PlotUtil::plotRectangle(rectanglePlot, rectangles[r], 255);
    }
    
    // Compute the distance transform in order to distinguish the pixels
    cv::Mat distanceTransform;
    Processing::computeDistanceTransform(rectanglePlot, distanceTransform);
    
    // Clear the edge map
    for (int x = 0; x < edgeMap.cols; x++)
    {
        for (int y = 0; y < edgeMap.rows; y++)
        {
            if (distanceTransform.at<float>(y,x) >= 4)
            {
                edgeMap.at<float>(y,x) = 0;
            }
        }
    }
}

void CabinetParser::removeNonRectanglePixels3(const std::vector<Rectangle> & rectangles, cv::Mat & edgeMap)
{
    // Plot all rectangles
    cv::Mat rectanglePlot = cv::Mat::zeros(edgeMap.rows, edgeMap.cols, CV_8UC1);
    for (size_t r = 0; r < rectangles.size(); r++)
    {
        PlotUtil::plotRectangle(rectanglePlot, rectangles[r], 255);
    }
    
    // Compute the distance transform in order to distinguish the pixels
    cv::Mat distanceTransform;
    Processing::computeDistanceTransform(rectanglePlot, distanceTransform);
    
    // Clear the edge map
    for (int x = 0; x < edgeMap.cols; x++)
    {
        for (int y = 0; y < edgeMap.rows; y++)
        {
            if (distanceTransform.at<float>(y,x) < 4)
            {
                edgeMap.at<float>(y,x) = 0;
            }
        }
    }
}


void CabinetParser::extractRectifiedMultiChannelImage(const cv::Mat & image, const Rectangle & region, cv::Mat & out)
{
    // First: Rectify the region of interest
    cv::Mat rectifiedRegionOfInterest;
    Processing::rectifyRegion(image, region, parameters.rectifiedROISize, rectifiedRegionOfInterest);
    
    // Convert the image to the to the Luv color space for the feature image
    // as well as to gray scale in order to compute the additional features
    cv::Mat rectifiedRegionOfInterestLuv;
    cv::Mat rectifiedRegionOfInterestGray;
    cv::cvtColor(rectifiedRegionOfInterest, rectifiedRegionOfInterestLuv, CV_BGR2Luv);
    cv::cvtColor(rectifiedRegionOfInterest, rectifiedRegionOfInterestGray, CV_BGR2GRAY);
    
    // Compute the gradients along the x and y directions
    cv::Mat gradients;
    Processing::computeGradients(rectifiedRegionOfInterestGray, gradients, 0.75f);
    
    // Compute the canny edge image 
    cv::Mat cannyEdges;
    Processing::computeCannyEdges(rectifiedRegionOfInterestGray, cannyEdges);
    // Plot the bounding box in the canny edge image
    // We do this in order for the distance transform to be less noisy
    for (int w = 0; w < cannyEdges.cols; w++)
    {
        cannyEdges.at<uchar>(0, w) = 255;
        cannyEdges.at<uchar>(cannyEdges.rows - 1, w) = 255;
    }
    for (int h = 0; h < cannyEdges.rows; h++)
    {
        cannyEdges.at<uchar>(h, 0) = 255;
        cannyEdges.at<uchar>(h, cannyEdges.cols - 1) = 255;
    }
    
    // Compute the distance transform of the canny edge image
    cv::Mat distanceTransform;
    Processing::computeDistanceTransform(cannyEdges, distanceTransform);
    
    // Compute the gradient magnitude image
    cv::Mat gradMag;
    Processing::computeGradientMagnitudeImage(rectifiedRegionOfInterestGray, gradMag, 0.75f);
    
    // Put together the final image consisting of a total of 7 channels
    out = cv::Mat::zeros(rectifiedRegionOfInterest.rows, rectifiedRegionOfInterest.cols, CV_32FC(EDGE_DETECTOR_CHANNELS));
    for (int w = 0; w < rectifiedRegionOfInterest.cols; w++)
    {
        for (int h = 0; h < rectifiedRegionOfInterest.rows; h++)
        {
#if 0
            // The color channels
            out.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_L] = rectifiedRegionOfInterestLuv.at<cv::Vec3b>(h,w)[0]/255.0f;
            out.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_U] = rectifiedRegionOfInterestLuv.at<cv::Vec3b>(h,w)[1]/255.0f;
            out.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_V] = rectifiedRegionOfInterestLuv.at<cv::Vec3b>(h,w)[2]/255.0f;
#endif
            // The gradient channels
            out.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_XDERIV] = gradients.at<cv::Vec2f>(h,w)[0];
            out.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_YDERIV] = gradients.at<cv::Vec2f>(h,w)[1];

            // The grayscale image
            out.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_INTENSITY] = rectifiedRegionOfInterestGray.at<uchar>(h,w);

            // The gradient magnitude image
            out.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_GM] = gradMag.at<uchar>(h,w);
#if 0
            // The edge channel
            out.at<EdgeDetectorVec>(h,w)[5] = cannyEdges.at<uchar>(h,w);
            // The distance transform channel
            out.at<EdgeDetectorVec>(h,w)[6] = distanceTransform.at<float>(h,w);
#endif
        }
    }
}

void CabinetParser::visualizeMultiChannelImage(const cv::Mat & image) const
{
    // Split the channels
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    for (size_t i = 0; i < channels.size(); i++)
    {
        Util::imshow(channels[i]);
    }
    return;
    // Visualize the distance transform
    float min = 1e30;
    float max = 0;
    for (int h = 0; h < channels[6].rows; h++)
    {
        for (int w = 0; w < channels[6].cols; w++)
        {
            min = std::min(min, channels[6].at<float>(h,w));
            max = std::max(max, channels[6].at<float>(h,w));
        }
    }
    
    cv::Mat demo(channels[6].rows, channels[6].cols, CV_8UC1);
    for (int h = 0; h < channels[6].rows; h++)
    {
        for (int w = 0; w < channels[6].cols; w++)
        {
            float val = (channels[6].at<float>(h,w) - min)/(max - min)*255;
            demo.at<uchar>(h,w) = static_cast<uchar>(round(val));
        }
    }
    
    Util::imshow(demo);
}

void CabinetParser::extractPatch(const cv::Mat & multiChannelImage, const cv::Vec2i & point, libf::DataPoint & dataPoint, int orientation)
{
    int counter = 0;

#if 0
    dataPoint(counter++) = point[0]/multiChannelImage.cols;
    dataPoint(counter++) = point[1]/multiChannelImage.rows;
#endif

    const EdgeDetectorVec & center = multiChannelImage.at<EdgeDetectorVec>(point[1],point[0]);
    EdgeDetectorVec mean;
    int meanCounter = 0;

    int dim1 = point[0];
    int dim2 = point[1];

    for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
    {
        mean[c] = 0;
    }

    for (int _x = dim1 - (PATCH_SIZE-1)/2; _x <= dim1 + (PATCH_SIZE-1)/2; _x++)
    {
        for (int _y = dim2 - (PATCH_SIZE-1)/2; _y <= dim2 + (PATCH_SIZE-1)/2; _y++)
        {
            int x = _x;
            int y = _y;

            if (!(x >= multiChannelImage.cols || x < 0 || y >= multiChannelImage.rows || y < 0))
            {
                const EdgeDetectorVec & v = multiChannelImage.at<EdgeDetectorVec>(y,x);
                for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
                {
                    mean[c] += v[c];
                    meanCounter++;
                }
            }
        }
    }
    for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
    {
        mean[c] /= meanCounter;
    }

    for (int _x = dim1 - (PATCH_SIZE-1)/2; _x <= dim1 + (PATCH_SIZE-1)/2; _x++)
    {
        for (int _y = dim2 - (PATCH_SIZE-1)/2; _y <= dim2 + (PATCH_SIZE-1)/2; _y++)
        {
            int x = _x;
            int y = _y;

            if (orientation == 1)
            {
                x = 2*dim1 - x;
                y = 2*dim2 - y;
            }
            else if (orientation == 2)
            {
                x = 2*dim1 - x;
                y = 2*dim2 - y;
            }
            else if (orientation == 3)
            {
                x = dim2 + _x - dim1;
                y = dim1 + _y - dim2;
                x = 2*dim2 - x;
                y = 2*dim1 - y;
            }

            // We don't use the canny channel and the distance transform channel
            // due to generalization issues
            if (x >= multiChannelImage.cols || x < 0 || y >= multiChannelImage.rows || y < 0)
            {
                for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
                {
                    dataPoint(counter++) = 0;
                }
            }
            else
            {
                const EdgeDetectorVec & v = multiChannelImage.at<EdgeDetectorVec>(y,x);

                if (x != point[0] && y != point[1])
                {
                    for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
                    {
                        if (std::isnan(v[c]))
                        {
                            throw std::exception();
                        }
                        dataPoint(counter++) = v[c] - center[c];
                    }
                }
                else
                {
                    for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
                    {
                        if (std::isnan(v[c]))
                        {
                            throw std::exception();
                        }
                        dataPoint(counter++) = v[c];
                    }
                }
            }
        }
    }
}

void CabinetParser::extractPatchFlipped(const cv::Mat & multiChannelImage, const cv::Vec2i & point, libf::DataPoint & dataPoint)
{
    int counter = 0;
#if 0
    dataPoint(counter++) = point[0]/multiChannelImage.cols;
    dataPoint(counter++) = point[1]/multiChannelImage.rows;
    const EdgeDetectorVec & center = multiChannelImage.at<EdgeDetectorVec>(point[1],point[0]);
#endif

    // Compute the mean pixel for this patch
    EdgeDetectorVec mean;
    for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
    {
        mean[c] = 0;
    }
    int countMean = 0;
    for (int x = point[0] - (PATCH_SIZE-1)/2; x <= point[0] + (PATCH_SIZE-1)/2; x++)
    {
        for (int y = point[1] - (PATCH_SIZE - 1) / 2; y <= point[1] + (PATCH_SIZE - 1) / 2; y++)
        {
            // We don't use the canny channel and the distance transform channel
            // due to generalization issues
            if (!(x >= multiChannelImage.cols || x < 0 || y >= multiChannelImage.rows || y < 0))
            {
                const EdgeDetectorVec & v = multiChannelImage.at<EdgeDetectorVec>(y,x);
                countMean++;
                for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
                {
                    mean[c] += v[c];
                }
            }
        }
    }

    for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
    {
        mean[c] /= countMean;
    }

    for (int y = point[1] - (PATCH_SIZE-1)/2; y <= point[1] + (PATCH_SIZE-1)/2; y++)
    {
        for (int x = point[0] - (PATCH_SIZE-1)/2; x <= point[0] + (PATCH_SIZE-1)/2; x++)
        {
            // We don't use the canny channel and the distance transform channel
            // due to generalization issues
            if (x >= multiChannelImage.cols || x < 0 || y >= multiChannelImage.rows || y < 0)
            {
                for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
                {
                    dataPoint(counter++) = 0;
                }
            }
            else
            {
                const EdgeDetectorVec & v = multiChannelImage.at<EdgeDetectorVec>(y,x);

                if (x != point[0] && y != point[1])
                {
                    for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
                    {
                        if (std::isnan(v[c]))
                        {
                            throw std::exception();
                        }
                        dataPoint(counter++) = v[c] - mean[c];
                    }
                }
                else
                {
                    for (int c = 0; c < EDGE_DETECTOR_CHANNELS; c++)
                    {
                        if (std::isnan(v[c]))
                        {
                            throw std::exception();
                        }
                        dataPoint(counter++) = v[c] - mean[c];
                    }
                }
            }
        }
    }
}

void CabinetParser::plotRectifiedEdgeMap(const cv::Mat& image, const Segmentation& segmentation, cv::Mat& edges)
{
    edges = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

    // Transform the annotated parts
    std::vector<Rectangle> transformedParts;
    CabinetParser::rectifyParts(segmentation.regionOfInterest, segmentation.parts, transformedParts);

    for(size_t k = 0; k < transformedParts.size(); k++)
    {
        PlotUtil::plotRectangle(edges, transformedParts[k], 255);
    }
}

void CabinetParser::plotRectifiedSegmentation(const cv::Mat & image, const Segmentation & segmentation, cv::Mat & segImg)
{
    segImg = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

    // Transform the annotated parts
    std::vector<Rectangle> transformedParts;
    CabinetParser::rectifyParts(segmentation.regionOfInterest, segmentation.parts, transformedParts);

    for(size_t k = 0; k < transformedParts.size(); k++)
    {
        PlotUtil::plotRectangleFill(segImg, transformedParts[k], cv::Scalar(static_cast<uchar>(k+1)));
    }
}

void CabinetParser::extractEdgeDetectorPatches(libf::DataStorage::ptr trainingSet, const std::vector< std::pair<cv::Mat, Segmentation> > & images)
{
    // We use random sampling in order to construct negative examples
    // For this purpose, we need a RNG
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    
    std::uniform_int_distribution<int> selectionDist(0,1);
    
    // Create a data set
    for (size_t i = 0; i < images.size(); i++)
    {
#if VERBOSE_MODE
        std::cout << "Processing image " << (i+1) << " out of " << images.size() << "(" << std::get<1>(images[i]).file << ")" << std::endl;
#endif
        
        // Extract the multichannel image
        cv::Mat multiChannelImage;
        extractRectifiedMultiChannelImage(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, multiChannelImage);
        
        // Plot the edge image
        cv::Mat edges;
        plotRectifiedEdgeMap(multiChannelImage, std::get<1>(images[i]), edges);
        
        cv::Mat cannyEdges, intensityImage;
        std::vector<cv::Mat> channels;
        cv::split(multiChannelImage, channels);
        channels[EDGE_DETECTOR_CHANNEL_INTENSITY].convertTo(intensityImage, CV_8UC1);
        Processing::computeCannyEdges(intensityImage, cannyEdges);
        
        // Compute a distance transform of the true edge map
        cv::Mat distanceTransform, cannyDistanceTransform;
        Processing::computeDistanceTransform(edges, distanceTransform);
        Processing::computeDistanceTransform(cannyEdges, cannyDistanceTransform);

        int counter = 0;
        // Extract the descriptor for each pixel on an edge and count the 
        // number of extracted descriptors
        for (int h = 0; h < edges.rows; h++)
        {
            for (int w = 0; w < edges.cols; w++)
            {
                // Only extract the descriptor if this pixel is on an edge
                //if (distanceTransform.at<float>(h,w) <= 1)
                if (edges.at<uchar>(h,w) != 0)
                {
                    // Extract the descriptor
                    cv::Vec2i pos;
                    pos[0] = w;
                    pos[1] = h;

                    {
                        libf::DataPoint point(PATCH_SIZE*PATCH_SIZE*EDGE_DETECTOR_CHANNELS);
                        extractPatch(multiChannelImage, pos, point, 0);
                        trainingSet->addDataPoint(point, 1);
                    }
                    {
                        libf::DataPoint point(PATCH_SIZE*PATCH_SIZE*EDGE_DETECTOR_CHANNELS);
                        extractPatch(multiChannelImage, pos, point, 1);
                        //trainingSet->addDataPoint(point, 1);
                    }

                    // Add the point to the data storage as positive example
                    counter++;
                }
                else if (cannyEdges.at<uchar>(h,w) != 0 && distanceTransform.at<float>(h,w) > 2)
//                else if (cannyDistanceTransform.at<float>(h,w) <= 1 && distanceTransform.at<float>(h,w) > 5)
                {
                    continue;
                    // Extract the descriptor
                    cv::Vec2i pos;
                    pos[0] = w;
                    pos[1] = h;

                    {
                        libf::DataPoint point(PATCH_SIZE*PATCH_SIZE*EDGE_DETECTOR_CHANNELS);
                        extractPatch(multiChannelImage, pos, point, 0);
                        trainingSet->addDataPoint(point, 0);
                    }
                    {
                        libf::DataPoint point(PATCH_SIZE*PATCH_SIZE*EDGE_DETECTOR_CHANNELS);
                        extractPatch(multiChannelImage, pos, point, 1);
                        //trainingSet->addDataPoint(point, 0);
                    }
                }
            }
        }

        // Set up a distribution over the horizontal and vertical pixels
        // of the image in order to sample negative examples
        std::uniform_int_distribution<int> horizontalDist(0, edges.cols - 1);
        std::uniform_int_distribution<int> verticalDist(0, edges.rows - 1);

        while (counter > 0)
        {
            // Sample a pixel and  check if it's on an edge
            const int w = horizontalDist(g);
            const int h = verticalDist(g);
            
            if (distanceTransform.at<float>(h,w) <= 2)
            {
                continue;
            }
            
            // This pixel is not on an edge
            // Extract the descriptor
            cv::Vec2i pos;
            pos[0] = w;
            pos[1] = h;

            {
                libf::DataPoint point(PATCH_SIZE*PATCH_SIZE*EDGE_DETECTOR_CHANNELS);
                extractPatch(multiChannelImage, pos, point, 0);
                trainingSet->addDataPoint(point, 0);
            }
            {
                libf::DataPoint point(PATCH_SIZE*PATCH_SIZE*EDGE_DETECTOR_CHANNELS);
                extractPatch(multiChannelImage, pos, point, 1);
                //trainingSet->addDataPoint(point, 0);
            }

            counter--;
        }
    }
}

void CabinetParser::trainEdgeDetector(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images)
{
#if VERBOSE_MODE
    std::cout << "Creating Data Set" << std::endl;
#endif

    std::vector< std::pair<cv::Mat, Segmentation > > imagesRGB;
    std::vector< std::pair<cv::Mat, Segmentation > > imagesD;
    
    for(int i=0; i<images.size(); i++)
    {
	imagesRGB.push_back(std::make_pair(std::get<0>(images[i]), std::get<1>(images[i])));
    imagesD.push_back(std::make_pair(std::get<2>(images[i]), std::get<1>(images[i])));
    }

    // This is the data set that the depth edge detector is trained upon
    std::cout<<"Started training edge model (Depth)"<<std::endl;
    libf::DataStorage::ptr trainingSetD = libf::DataStorage::Factory::create();
    extractEdgeDetectorPatches(trainingSetD, imagesD);
    
    ForestLearner forestLearnerD;
    forestLearnerD.getTreeLearner().setMinSplitExamples(15);
    forestLearnerD.getTreeLearner().setUseBootstrap(true);
    forestLearnerD.getTreeLearner().setNumBootstrapExamples(100000);
    
    forestLearnerD.setNumTrees(96);
    forestLearnerD.setNumThreads(8);
    
    libf::RandomForestLearner<libf::DecisionTreeLearner>::State stateD;
    libf::ConsoleGUI<libf::RandomForestLearner<libf::DecisionTreeLearner> > guiD(stateD,   	   libf::RandomForestLearner<libf::DecisionTreeLearner>::defaultGUI);
    
    auto forestD = forestLearnerD.learn(trainingSetD, stateD);    
   
    guiD.join();

    // Save the model
    libf::write("edge_model_depth.bin", *forestD);


    
    // This is the data set that the edge detector is trained upon
    std::cout<<"Started training edge model (RGB)"<<std::endl;
    libf::DataStorage::ptr trainingSet = libf::DataStorage::Factory::create();
    extractEdgeDetectorPatches(trainingSet, imagesRGB);
    
    
    // Now train a random forest on this data set
    
    ForestLearner forestLearner;
    forestLearner.getTreeLearner().setMinSplitExamples(15);
    forestLearner.getTreeLearner().setUseBootstrap(true);
    forestLearner.getTreeLearner().setNumBootstrapExamples(100000);
    
    forestLearner.setNumTrees(96);
    forestLearner.setNumThreads(8);
    
    libf::RandomForestLearner<libf::DecisionTreeLearner>::State state;
    libf::ConsoleGUI<libf::RandomForestLearner<libf::DecisionTreeLearner> > gui(state,   	libf::RandomForestLearner<libf::DecisionTreeLearner>::defaultGUI);
    
    auto forest = forestLearner.learn(trainingSet, state);    
   
    gui.join();

    // Save the model
    libf::write("edge_model.bin", *forest);
    return;

#if VERBOSE_MODE
    //libf::AccuracyTool accuracyTool;
    //accuracyTool.measureAndPrint(forest, trainingSet);
    libf::AccuracyTool accuracyToolD;
    accuracyToolD.measureAndPrint(forestD, trainingSetD);
    
    //libf::ConfusionMatrixTool confusionMatrixTool;
    //confusionMatrixTool.measureAndPrint(forest, trainingSet);
    libf::ConfusionMatrixTool confusionMatrixToolD;
    confusionMatrixToolD.measureAndPrint(forestD, trainingSetD);
#endif

#if 0
    // Select a subset of trees
    int N = forest->getSize();
    double* distances = new double[N];
    for (int i = 0; i < N; i++)
    {
        for (int j = i+1; j < N; j++)
        {
            int disagreements = 0;

            for (int n = 0; n < trainingSet->getSize(); n++)
            {
                if (forest->getTree(i)->classify(trainingSet->getDataPoint(n)) != forest->getTree(j)->classify(trainingSet->getDataPoint(n)))
                {
                    disagreements--;
                }
            }

            distances[i*N + j] = disagreements;
            distances[j*N + i] = disagreements;
        }
    }

    int K = 30;
    size_t* centers = new size_t[K];
    size_t* assignments = new size_t[N];
    double objective;
    pam(K, N, distances, centers, assignments, &objective);

    delete[] assignments;

    auto newforest = std::make_shared<libf::RandomForest<libf::DecisionTree> >();

    for (int i = 0; i < K; i++)
    {
        newforest->addTree(forest->getTree(centers[i]));
    }

    libf::write("edge_model.bin", *newforest);
#endif
}


int dmod(double x, double a)
{
    return std::round(x/a);
}

static void computeGradMag(const cv::Mat & img, cv::Mat & gradMag, cv::Mat & angles)
{
    const int boxSize = 5;
    const float sigma = 1;

    cv::Mat filter(2*boxSize + 1, 2*boxSize + 1, CV_32F);

    float normalize = 0;
    for (int x = -boxSize; x <= boxSize; x++)
    {
        for (int y = -boxSize; y <= boxSize; y++)
        {
            filter.at<float>(y+boxSize,x+boxSize) = -1.0/sigma * x * std::exp(-0.5*1/sigma*(x*x + y*y));
            normalize += std::abs(filter.at<float>(y+boxSize,x+boxSize));
        }
    }

    /// Generate grad_x and grad_y
    cv::Mat grad_x, grad_y;

    /// Gradient X
    cv::filter2D(img, grad_x, CV_32F, filter);

    /// Gradient Y
    cv::filter2D(img, grad_y, CV_32F, filter.t());

    // Compute gradient magnitude and gradient angle
    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++)
        {
            // Get the angle in [-pi, pi]
            double alpha = std::atan2(static_cast<double>(grad_y.at<float>(y,x)), static_cast<double>(grad_x.at<float>(y,x)));
            // Make sure the angle is in [0, pi]
            if (alpha < 0)
            {
                alpha += M_PI;
            }
            // Compute alpha mode pi/4
            angles.at<int>(y,x) = dmod(alpha, M_PI/4);
            // Compute gradient magnitude
            const float temp = grad_x.at<float>(y,x)*grad_x.at<float>(y,x) + grad_y.at<float>(y,x)*grad_y.at<float>(y,x);
            gradMag.at<float>(y,x) = temp;
        }
    }
}

/**
 * Filters the image and displays it 
 */
static void filterImage(const cv::Mat & votes, const cv::Mat & img, cv::Mat & edges)
{
    cv::Mat gradMag(img.rows, img.cols, CV_32F);
    cv::Mat angles(img.rows, img.cols, CV_32S);
    
    // Compute the gradient magnitude and angles
    computeGradMag(img, gradMag, angles);
    edges = cv::Scalar(0);
    
    // Perform non-maximum surpression
    for (int x = 0; x < gradMag.cols; x++)
    {
        for (int y = 0; y < gradMag.rows; y++)
        {
            // Check the magnitude along the gradient
            bool ok = true;
            int posX1 = 0, posX2 = 0, posY1 = 0, posY2 = 0;
            switch (angles.at<int>(y,x))
            {
                // 0: 0
                // 1: 45
                // 2: 90
                // 3: 135
                // 4: 180

                // 0°, 360°
                case 0:
                case 4:
                    posX1 = -1;
                    posY1 = 0;
                    posX2 = 1;
                    posY2 = 0;
                    break;

                // 45°, 225°
                case 1:
                    posX1 = 1;
                    posY1 = 1;
                    posX2 = -1;
                    posY2 = -1;
                    continue;
                    break;

                // 90°, 270°
                case 2:
                    posX1 = 0;
                    posY1 = 1;
                    posX2 = 0;
                    posY2 = -1;
                    break;

                // 135°, 315°
                case 3:
                    posX1 = -1;
                    posY1 = 1;
                    posX2 = 1;
                    posY2 = -1;
                    continue;
                    break;
            }

            // Perform non-maximum surpression
            if ((x + posX1) >= 0 && (x + posX1) < img.cols && (y + posY1) >= 0 && (y + posY1) < img.rows)
            {
                ok = ok && votes.at<float>(y,x) > votes.at<float>(y + posY1, x + posX1);
            }
            if ((x + posX2) >= 0 && (x + posX2) < img.cols && (y + posY2) >= 0 && (y + posY2) < img.rows)
            {
                ok = ok && votes.at<float>(y,x) > votes.at<float>(y + posY2, x + posX2);
            }
            // Set the edge pixels if they are above the threshold
            if (ok && votes.at<float>(y,x) >= 0.25)
            {
                edges.at<uchar>(y,x) = 255;
            }
        }
    }
    
#if 0
    // Perform non-maximum surpression
    for (int x = 1; x < gradMag.cols-1; x++)
    {
        for (int y = 1; y < gradMag.rows-1; y++)
        {
            bool ok = false;
            for (int _x = -1; _x <= 1; _x++)
            {
                for (int _y = -1; _y <= 1; _y++)
                {
                    if (!(_x == 0 && _y == 0) && edges.at<uchar>(y + _y,x + _x) == 255)
                    {
                        ok = true;
                    }
                }
            }
            if (!ok)
            {
                edges.at<uchar>(y,x) = 0;
            }
        }
    }
#endif

    cv::imshow("test", edges);
    cv::waitKey();
}

cv::Mat _votes, _image, _result;
int _radius;
int _lower;
int _upper;

void plot(int, void*)
{
    unsigned char* c_image = Util::toArrayImg<uchar, float>(_image);
    short int* c_votes = Util::toArrayImg<short int, short int>(_votes);
    unsigned char* c_edges = 0;

    // Apply canny
    canny(c_image, _votes.rows, _votes.cols, c_votes, _radius, _lower/1000., _upper/1000. , &c_edges, 0);

    Util::toOpenCVImg<uchar>(c_edges, _votes.rows, _votes.cols, _result);

    if (c_image != 0)
    {
        delete[] c_image;
    }

    if (c_votes != 0)
    {
        delete[] c_votes;
    }

    if (c_edges != 0)
    {
        delete[] c_edges;
    }

    cv::imshow("window", _result);
}

void histogram(const cv::Mat & votes, int bins, std::vector<int> & hist)
{
    // Determine the minimum and maximum values
    short min = 10000;
    short max = -1;

    for (int i = 0; i < votes.rows; i++)
    {
        for (int j = 0; j < votes.cols; j++)
        {
            if (votes.at<short>(i,j) == 0)
            {
                continue;
            }

            min = std::min(min, votes.at<short>(i,j));
            max = std::max(max, votes.at<short>(i,j));
        }
    }

    hist = std::vector<int>(bins, 0);

    float range = max - min;

    for (int i = 0; i < votes.rows; i++)
    {
        for (int j = 0; j < votes.cols; j++)
        {
            short value = votes.at<short>(i,j);
            if (value == 0)
            {
                continue;
            }
            int bin = (value - min)/range * bins;
            hist[bin]++;
        }
    }

    for (int i = 0; i < bins; i++)
    {
        std::cout << i/(static_cast<float>(bins)) << ": " << hist[i] << "\n";
    }
}

void CabinetParser::applyEdgeDetector(  const cv::Mat & multiChannelImage, 
                                        cv::Mat & edges, int depthFlag)
{
#if 1
    std::vector<cv::Mat> channels;
    cv::split(multiChannelImage, channels);
#endif
    // Initialize the output image
    edges = cv::Mat::zeros(multiChannelImage.rows, multiChannelImage.cols, CV_8UC1);
    
    Forest::ptr forest = std::make_shared<Forest>();
    if(depthFlag == 0)
	libf::read("edge_model.bin", *forest);
    else if(depthFlag == 1)
	libf::read("edge_model_depth.bin", *forest);
    else
	std::cout<<"Invalid depth flag"<<std::endl;
    

    cv::Mat votes = cv::Mat::zeros(multiChannelImage.rows, multiChannelImage.cols, CV_16S);

    // Iterate over the MC image and decide per pixel if it's an edge or
    // no edge
    //Stopwatch swatch;
    //swatch.set_mode(REAL_TIME);
    //swatch.start("My astounding algorithm");

    #pragma omp parallel for
    for (int w = 0; w < multiChannelImage.cols - 0; w++)
    {
        libf::DataPoint point(PATCH_SIZE*PATCH_SIZE*EDGE_DETECTOR_CHANNELS);
        cv::Vec2i pos;
        
        for (int h = 0; h < multiChannelImage.rows - 0; h++)
        {
            // Extract the image patch
            pos[0] = w;
            pos[1] = h;
            extractPatch(multiChannelImage, pos, point, 0);
#if 0
            votes.at<short>(h,w) += static_cast<short>(forest->getVotesFor1(point));
            if (votes.at<short>(h,w) < 7)
            {
                votes.at<short>(h,w) = 0;
            }
#else
            edges.at<uchar>(h,w) = static_cast<uchar>(255* forest->classify(point));
#endif
        }
    }
#if 1
    Processing::add1pxBorders(edges);
    return;
#endif
    //swatch.stop("My astounding algorithm");
    //swatch.report_all();

    // Convert the original image to an array
    unsigned char* c_image = Util::toArrayImg<uchar, float>(channels[EDGE_DETECTOR_CHANNEL_INTENSITY]);
    short int* c_votes = Util::toArrayImg<short int, short int>(votes);
    unsigned char* c_edges = 0;

    // Apply canny
    canny(c_image, votes.rows, votes.cols, c_votes, 3, 0.35, 0.65, &c_edges, 0);

    Util::toOpenCVImg<uchar>(c_edges, votes.rows, votes.cols, edges);

#if 0
    cv::namedWindow("window", 1);

    cv::createTrackbar( "radius", "window", &_radius, 100, plot);
    cv::createTrackbar( "lower", "window", &_lower, 1000, plot);
    cv::createTrackbar( "upper", "window", &_upper, 1000, plot);

    edges.copyTo(_result);
    votes.copyTo(_votes);
    channels[EDGE_DETECTOR_CHANNEL_INTENSITY].copyTo(_image);

    cv::imshow("window", _result);

    cv::waitKey();

    exit(0);
#endif

    if (c_image != 0)
    {
        delete[] c_image;
    }
        
    if (c_votes != 0)
    {
        delete[] c_votes;
    }
    
    if (c_edges != 0)
    {
        delete[] c_edges;
    }
    
    Processing::add1pxBorders(edges);
}

void CabinetParser::rectifyParts(  const Rectangle & regionOfInterest, 
                    const std::vector<Rectangle> & partsIn, 
                    std::vector<Rectangle> & partsOut)
{
    // Compute the homography to the rectified image
    Rectangle destination;
    // TODO: Replace 500 with a makro
    Processing::computeRectifiedRegionOfInterest(regionOfInterest, 500, destination);
    
    // Compute the corresponding homography
    cv::Mat homography;
    Processing::computeHomography(regionOfInterest, destination, homography);
    
    // Transform the parts
    for (size_t p = 0; p < partsIn.size(); p++)
    {
        Rectangle r;
        RectangleUtil::applyHomography(homography, partsIn[p], r);
        partsOut.push_back(r);
    }
}

void CabinetParser::unrectifyParts(  const Rectangle & regionOfInterest, 
                    const std::vector<Rectangle> & partsIn, 
                    std::vector<Rectangle> & partsOut)
{
    // Compute the homography to the rectified image
    Rectangle destination;
    // TODO: Replace 500 with a makro
    Processing::computeRectifiedRegionOfInterest(regionOfInterest, 500, destination);
    
    // Compute the corresponding homography
    cv::Mat homography;
    Processing::computeHomography(destination, regionOfInterest, homography);
    
    // Transform the parts
    for (size_t p = 0; p < partsIn.size(); p++)
    {
        Rectangle r;
        RectangleUtil::applyHomography(homography, partsIn[p], r);
        partsOut.push_back(r);
    }
}

void CabinetParser::detectLines(const cv::Mat & image, 
                                std::vector<LineSegment> & resultH, 
                                std::vector<LineSegment> & resultV)
{
    LineDetector detector;
    
    //std::vector<LineSegment> preliminaryH, preliminaryV;
    std::vector<LineSegment> tempResultH;
    std::vector<LineSegment> tempResultV;

    // Make
    do {
        tempResultH.clear();
        tempResultV.clear();
        // Detect the line segments in the binary image
        detector.detectLines(image, tempResultH, true);
        detector.detectLines(image, tempResultV, false);
        detector.model.minLength += 10;
    } while(tempResultH.size() + tempResultV.size() > 30);

    resultH = tempResultH;
    resultV = tempResultV;

    // Remove very short line segments
    //detector.filterShortLines(preliminaryH, resultH);
    //detector.filterShortLines(preliminaryV, resultV);
    
    // Join the line segments
    //detector.joinLines(resultH, true);
    //detector.joinLines(resultV, false);
    
    // Add the bounding box lines
    detector.addBoundingBoxLines(image, resultH, true);
    detector.addBoundingBoxLines(image, resultV, false);
#if 0
    cv::Mat demo(image.rows, image.cols, CV_8UC3);
    demo = cv::Scalar(0);
    detector.visualize(demo, resultH, cv::Scalar(255,0,255));
    detector.visualize(demo, resultV, cv::Scalar(255,255,0));
    Util::imshow(demo);
#endif 
}


void CabinetParser::detectRectangles(const cv::Mat & image, const cv::Mat & cannyEdges, std::vector<Rectangle> & result)
{

    //std::cout<<"Rectangle Acceptance Threshold: "<<rectangleAcceptanceThreshold<<std::endl;
    //std::cout<<"Rectangle Detection Threshold: "<<rectangleDetectionThreshold<<std::endl;

    // Detect line segments
    std::vector<LineSegment> lineSegmentsH, lineSegmentsV;
    detectLines(image, lineSegmentsH, lineSegmentsV);

#if 0
    Util::imshow(image);
    #pragma omp critical 
    {
        {
            Util::imshow(image);
            cv::Mat demo(image.rows, image.cols, CV_8UC3);
            LineDetector detector;
            demo = cv::Scalar(0);
            detector.visualize(demo, lineSegmentsH, cv::Scalar(255,0,255));
            detector.visualize(demo, lineSegmentsV, cv::Scalar(255,255,0));
            Util::imshow(demo);
        }
    }
#endif
    
    cv::Mat computedEdges;
    image.copyTo(computedEdges);
    cv::bitwise_or(computedEdges, cannyEdges, computedEdges);
    Processing::add1pxBorders(computedEdges);
    cv::Mat inputDist;
    
#if 0
    cv::Mat horizontalEdges, verticalEdges, cannyHorizontalEdges, cannyVerticalEdges, horizontalDistanceTransform, verticalDistanceTransform;
    Util::splitEdgeImage(computedEdges, horizontalEdges, verticalEdges);
    Util::splitEdgeImage(cannyEdges, cannyHorizontalEdges, cannyVerticalEdges);
    Processing::computeDistanceTransform(horizontalEdges, horizontalDistanceTransform);
    Processing::computeDistanceTransform(verticalEdges, verticalDistanceTransform);

    for (int i = 0; i < computedEdges.rows; i++)
    {
        for (int j = 0; j < computedEdges.cols; j++)
        {
            if (cannyHorizontalEdges.at<uchar>(i,j) > 0 && horizontalDistanceTransform.at<float>(i,j) <= 10)
            {
                computedEdges.at<uchar>(i,j) = 255;
            }
            if (cannyVerticalEdges.at<uchar>(i,j) > 0 && verticalDistanceTransform.at<float>(i,j) <= 10)
            {
                computedEdges.at<uchar>(i,j) = 255;
            }
        }
    }
#endif

    Processing::computeDistanceTransform(computedEdges, inputDist);

    RectangleDetector detector;
#if 0

#endif
#if 0
    cv::imshow("a", computedEdges);
    cv::imshow("a", horizontalEdges);
    cv::imshow("b", verticalEdges);
    cv::waitKey();

    cv::Mat horizontalDistanceTransform, verticalDistanceTransform;
    Processing::computeDistanceTransform(horizontalEdges, horizontalDistanceTransform);
    Processing::computeDistanceTransform(verticalEdges, verticalDistanceTransform);
#endif

    // Detect the initial set of rectangles
    detector.detectRectangles(image, lineSegmentsH, lineSegmentsV, result, [& inputDist](const Rectangle & r) -> bool
    {
        const float thresh = rectangleDetectionThreshold;
        
        const int y1 = static_cast<int>(std::round(r[0][1]));
        const int y2 = static_cast<int>(std::round(r[3][1]));

        // Trace the contours
        for (float _x = r[0][0]; _x <= r[1][0]; _x += 1)
        {
            const int x = static_cast<int>(std::round(_x));

            if (inputDist.at<float>(y1,x) > thresh || inputDist.at<float>(y2,x) > thresh)
            {
                return false;
            }
        }
        const int x1 = static_cast<int>(std::round(r[0][0]));
        const int x2 = static_cast<int>(std::round(r[1][0]));

        // Trace the contours
        for (float _y = r[1][1]; _y <= r[2][1]; _y += 1)
        {
            const int y = static_cast<int>(std::round(_y));

            if (inputDist.at<float>(y,x1) > thresh || inputDist.at<float>(y,x2) > thresh)
            {
                return false;
            }
        }
        
        return true;
    });
    
    // Add the region of interest box
    Rectangle r;
    r[0][0] = 0;
    r[0][1] = 0;
    r[1][0] = image.cols - 1;
    r[1][1] = 0;
    r[2][0] = image.cols - 1;
    r[2][1] = image.rows - 1;
    r[3][0] = 0;
    r[3][1] = image.rows - 1;
    result.push_back(r);
    
#if 0
    cv::Mat demo(image.rows, image.cols, CV_8UC3);
    demo = cv::Scalar(0);
    for (size_t i = 0; i < result.size(); i++)
    {
        demo = cv::Scalar(0);
        detector.visualize(demo, result, cv::Scalar(120,120,120));
        PlotUtil::plotRectangle(demo, result[i], cv::Scalar(0,255,0));
        cv::imshow("demo", demo);
        cv::waitKey();
    }
    cv::imshow("demo", demo);
    cv::waitKey();
#endif
}

void CabinetParser::loadImage(const std::string & directory, std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat > > & images)
{
    // Iterate over the directory and load the individual images together with
    // their annotation
    boost::filesystem::path p(directory);
    
    std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat> > trainingData;    
    
    if (boost::filesystem::exists(p))
    {
        // Iterate over all files in the directory
        boost::filesystem::directory_iterator end_iter;
        for(boost::filesystem::directory_iterator iter (p); iter != end_iter; ++iter)
        {
            // Only consider files that start with a number
            if (boost::filesystem::is_regular_file(iter->path()) && iter->path().extension() == ".JPG")
            {
                // Get the image ID
                const std::string filename = iter->path().filename().generic_string();
                const std::string imageID = filename.substr(0, filename.find('.'));
                const std::string imgFile = iter->path().generic_string();
                const std::string annotationFile = directory + imageID + "_annotation.json";
                const std::string imgDFile = directory + imageID + "_depth.png";

                
                // Load the image
                cv::Mat image = cv::imread(imgFile, 1);
                cv::Mat imageD = cv::imread(imgDFile, 1);

                
                if (!image.data)
                {
                    throw ParserException("Cannot read image.");
                }
                if (!imageD.data)
                {
                    std::cout<<imgDFile<<std::endl;
                    throw ParserException("Cannot read depth image.");
                }
                
                // Load the segmentation
                Segmentation segmentation;
                segmentation.readAnnotationFile(annotationFile);
                segmentation.id = imageID;
                
                images.push_back( std::make_tuple(image, segmentation, imageD));
            }
        }
    }
}

void CabinetParser::train(const std::string & directory)
{
    std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat> > trainingData;
    
    std::cout << "Load training data from " << directory << "\n";
    loadImage(directory, trainingData);
    
    std::cout << "Done loading training data\n";
    std::cout << trainingData.size() << " images loaded\n\n";
    
    train(trainingData);
}

void CabinetParser::train(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images)
{
    // Train the edge detector
    std::cout << "Train edge detector\n";
    std::cout << "===================\n";
    trainEdgeDetector(images);
    
    // Train the part priors
    std::cout << "Train part priors\n";
    std::cout << "===================\n";
    trainPartPrior(images);

    
    // Train the appearances
    std::cout << "Train appearance codebook\n";
    std::cout << "===================\n";
    trainAppearanceCodeBook(images);
    std::cout << "===================\n";
    std::cout << "DONE\n";
}

void CabinetParser::test(const std::string & directory)
{
    std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat> > trainingData;
    
    std::cout << "Load test data from " << directory << "\n";
    loadImage(directory, trainingData);
    
    std::cout << "Done loading test data\n";
    std::cout << trainingData.size() << " images loaded\n\n";
    
    test(trainingData);
}

void CabinetParser::test(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images)
{
    // Train the edge detector
    std::cout << "Test edge detector\n";
    std::cout << "===================\n";
    //evaluateEdgeDetector(images);

    std::cout << "Test rectangle detector\n";
    std::cout << "===================\n";
    //for (int i = 1; i <= 8; i++)
    {
        //rectangleDetectionThreshold = i;
        //evaluateRectangleDetector(images);
    }
    
    std::cout << "Test rectangle pruning\n";
    std::cout << "===================\n";
    //evaluateRectanglePruning(images);
    
    std::cout << "Test part classifier\n";
    std::cout << "===================\n";
    //testPartClassifier(images);
    
    std::cout << "Test appearance codebook\n";
    std::cout << "===================\n";
    //testAppearanceCodeBook(images);
    
    std::cout << "Test segmentation\n";
    std::cout << "===================\n";
    evaluateSegmentation(images);
    
    //for (float i = 0.74; i > 0.05f; i -= 0.05)
    {
        //rectangleAcceptanceThreshold = i;
        //rectangleAcceptanceThreshold = 0.8f;
        //computePrecisionRecallCurve(images);
    }
}

void CabinetParser::testEdgeDetector(const std::vector<std::tuple<cv::Mat, Segmentation, cv::Mat> >& images)
{
    
    // This is the data set that the edge detector is tested upon
    std::vector< std::pair<cv::Mat, Segmentation > > imagesRGB;
    //std::vector< std::pair<cv::Mat, Segmentation > >  imagesD;
    
    for(int i=0; i<images.size(); i++)
    {
        imagesRGB.push_back(std::make_pair(std::get<0>(images[i]), std::get<1>(images[i])));
        //imagesD.push_back(std::make_pair(std::get<2>(images[i]), std::get<1>(images[i])));
    }
    
    Forest::ptr forest = std::make_shared<Forest>();
    libf::read("edge_model.bin", *forest);
    
    libf::DataStorage::ptr testSet = libf::DataStorage::Factory::create();
    extractEdgeDetectorPatches(testSet, imagesRGB);
    
    libf::AccuracyTool accuracyTool;
    accuracyTool.measureAndPrint(forest, testSet);
    
    libf::ConfusionMatrixTool confusionMatrixTool;
    confusionMatrixTool.measureAndPrint(forest, testSet);
}


/**
 * Proposal Selection using rjMCMC
 */
void CabinetParser::selectParts(
    const cv::Mat & mcImage, 
    const cv::Mat & depthImage,    
    const cv::Mat & edgeImage,
    std::vector<Rectangle> & hypotheses,
    std::vector<Part> & result)
{
    // First, we create parts from the hypotheses rectangles.
    std::vector<Part> partHypotheses;
    
    // Compute the canny edge image
    std::vector<cv::Mat> channels;
    cv::split(mcImage, channels);

    // Compute canny edges
    cv::Mat cannyEdges, intensityImage;
    channels[EDGE_DETECTOR_CHANNEL_INTENSITY].convertTo(intensityImage, CV_8UC1);
    Processing::computeCannyEdges(intensityImage, cannyEdges);
        
    // Load the appearance codebook
    std::vector<Eigen::MatrixXf> codebooks(5);
    std::ifstream res("codebook.dat");
    for (int l = 0; l < 3; l++)
    {
        libf::readBinary(res, codebooks[l]);
        std::stringstream ss;
        ss << l << "_codebook.csv";
        std::ofstream o(ss.str());
        o << codebooks[l];
        o.close();
    }
    res.close();
    
    cv::Mat gradMag;
    Processing::computeGradientMagnitudeImageFloat(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], gradMag);
    cv::Mat gradMagDepth;
    Processing::computeGradientMagnitudeImageFloat(depthImage, gradMagDepth);
#if INCLUDE_DEPTH
    float linearBlendAlpha = 0.7;
    float linearBlendBeta = 1 - linearBlendAlpha;
    addWeighted(gradMag, linearBlendAlpha, gradMagDepth, linearBlendBeta, 0.0, gradMagDepth);
#endif

    cv::Mat depthImageFloat;
    depthImage.convertTo(depthImageFloat,CV_32FC1);

#if 0
	visualizeFloatImage(gradMag);
#endif

    // Compute the background detecting GM image
    cv::Mat bgGradMag;
    gradMag.copyTo(bgGradMag);
    //removeNonRectanglePixels2(hypotheses, bgGradMag);
    removeNonRectanglePixels3(hypotheses, gradMag);

    const double imageArea = edgeImage.rows*edgeImage.cols;
    int numClusters = OPTIMUM_RECTS;

#if SPLIT_MERGE_AUGMENT
    std::cout <<"Redundany Threshold: "<<CLUSTER_MAX_IOU<<std::endl;
    removeRedundantRects(hypotheses, CLUSTER_MAX_IOU);
    std::cout << std::setw(5) << hypotheses.size() << " possibly different rectangles after redundancy removal" << std::endl;
    augmentRectanglesMergeWH(hypotheses);
    removeRedundantRects(hypotheses, CLUSTER_MAX_IOU);
    std::cout<<"Effective no: of rectangles rectangles after merge augmenting part 1: "<<hypotheses.size()<<std::endl;
    augmentRectanglesMergeHW(hypotheses);
    removeRedundantRects(hypotheses, CLUSTER_MAX_IOU);
    std::cout<<"Effective no: of rectangles rectangles after merge augmenting part 2: "<<hypotheses.size()<<std::endl;
    augmentRectanglesSplitWH(gradMag, hypotheses, imageArea);
    removeRedundantRects(hypotheses, CLUSTER_MAX_IOU);
    std::cout<<"Effective no: of rectangles rectangles after split augmenting part 1: "<<hypotheses.size()<<std::endl;
    augmentRectanglesSplitHW(gradMag, hypotheses, imageArea);
    removeRedundantRects(hypotheses,CLUSTER_MAX_IOU);
    std::cout<<"Effective no: of rectangles rectangles after split augmenting part 2: "<<hypotheses.size()<<std::endl;

    std::vector<Rectangle> hypothesesDupli = hypotheses;
    removeRedundantRects(hypothesesDupli, 0.8f);// TO DO: Tuning
    numClusters = hypothesesDupli.size();
    std::cout <<"Number of rectangle clusters: "<<numClusters<<std::endl;
#endif


#if 0
    cv::Mat demo2(gradMag.rows, gradMag.cols, CV_8UC3);
    demo2 = cv::Scalar(0);
    cv::imshow("Edge Image", cannyEdges);

    for (size_t n = 0; n < hypotheses.size(); n++)
    {
        demo2 = cv::Scalar(0);
        PlotUtil::plotRectangle(demo2, hypotheses[n], cv::Scalar(255,0,255));
        cv::imshow("test", demo2);
        cv::waitKey();
    }
    //exit(0);
#endif


#if 0
    cv::imshow("Edge Map",gradMag);
    cv::waitKey();
#endif

    /**
     * Read the training parameters
     */
    float maxAspRatio, maxDWAspRatio, maxDHAspRatio, maxDepthGlobal, maxAngleRatio;
    int partCount[3];
    cv::FileStorage fsRead("trainParameters.yml", cv::FileStorage::READ);
    fsRead ["maxDepthTrain"] >> maxDepthGlobal;
    fsRead ["maxAngleRatio"] >> maxAngleRatio;
    fsRead ["maxAspRatioTrain"] >> maxAspRatio;
    fsRead ["maxDWAspRatioTrain"] >> maxDWAspRatio;
    fsRead ["maxDHAspRatioTrain"] >> maxDHAspRatio;
    fsRead ["class0Count"] >> partCount[0];
    fsRead ["class1Count"] >> partCount[1];
    fsRead ["class2Count"] >> partCount[2];
    fsRead.release();
    
#if 0
    std::cout<<"Maximum Depth Value (Global) from training : "<<maxDepthGlobal<<std::endl;
    std::cout<<"Maximum AngleRatio (Global) from training : "<<maxAngleRatio<<std::endl;
    std::cout<<"Maximum AspectRatio (Global) from training : "<<maxAspRatio<<std::endl;
    std::cout<<"Maximum Depth-Width AspectRatio (Global) from training : "<<maxDWAspRatio<<std::endl;
    std::cout<<"Maximum Depth-Height AspectRatio (Global) from training : "<<maxDHAspRatio<<std::endl;
    std::cout<<"Number of Doors : "<<partCount[0]<<std::endl;
    std::cout<<"Number of Drawers : "<<partCount[1]<<std::endl;
    std::cout<<"Number of Shelves : "<<partCount[2]<<std::endl;
#endif

#if 0
    for (int l = 0; l < 3; l++)
    {
        widthHeightDist.getDistribution(std::vector<int>({l})).normalize();
    }
#endif

    // Determine the prior probabilities
    float labelPrior[3];
    float normalizeLabels = 0;
    for (int l = 0; l < 3; l++)
    {
        labelPrior[l] = partCount[l];
        normalizeLabels += labelPrior[l];
    }
    for (int l = 0; l < 3; l++)
    {
        labelPrior[l] /= normalizeLabels;
    }
    
    float meanDepth[hypotheses.size()];

    cv::Mat depthImg =  cv::Scalar::all(255) - depthImage;
    cv::Mat rectifiedDepth;
    /**
     * Depth Rectification
     */
    rectifyDepthBilinear(depthImg,rectifiedDepth);

#if 0
    cv::imshow("RGB image",std::get<0>(images[i]));
    cv::waitKey();

    cv::imshow("Depth Image",depthImg);
    cv::imshow("Depth Image [rectified]",rectifiedDepth);
    cv::waitKey();
#endif

    float formFactorH = 0.0f;
    float angleRatio = 0.0f;

    //#pragma omp parallel for
    for (size_t h = 0; h < hypotheses.size(); h++)
    {
        // Extract the descriptor
        libf::DataPoint p, p2;
        extractDiscretizedAppearanceDataGM(gradMag, hypotheses[h], p, p2);

        std::vector<int> projProf;
        std::vector<int> projProfTyp;

        /**
         * Edge Projection Profile for Split Augmentation
         */
        extractEdgeProjectionProfile(gradMag, hypotheses[h], projProf, projProfTyp);

        // We have to normalize the reconstruction errors and prior probabilities 
        // in order to get decent numerical joint probabilities
        float reconstructionErrors[3];
        float priorProbabilities[3];

        /**
         * Mean depth of each IE
         */
        meanDepth[h] = extractMeanPartDepth(rectifiedDepth, hypotheses[h]);
        /**
         * Boundary Conditions
         */
        if(meanDepth[h]>maxDepthGlobal)
            maxDepthGlobal = meanDepth[h]; // precaution in case the depth under test is more than the ones alread seen

        if((hypotheses[h].getWidth()/edgeImage.cols)/(hypotheses[h].getHeight()/edgeImage.rows)>maxAspRatio)
            maxAspRatio = (hypotheses[h].getWidth()/edgeImage.cols)/(hypotheses[h].getHeight()/edgeImage.rows); // precaution in case the asp 	ratio under test is more than the ones alread seen
       
        if(meanDepth[h]/hypotheses[h].getWidth()>maxDWAspRatio)
            maxDWAspRatio = meanDepth[h]/hypotheses[h].getWidth();	

        if(meanDepth[h]/hypotheses[h].getHeight()>maxDHAspRatio)
            maxDHAspRatio = meanDepth[h]/hypotheses[h].getHeight();	
	
        formFactorH = 4*hypotheses[h].getWidth()*hypotheses[h].getHeight()/pow((hypotheses[h].getWidth() + hypotheses[h].getHeight()),2);
      
        angleRatio = std::atan2((double) hypotheses[h].getWidth(),(double) hypotheses[h].getHeight()) / std::atan2((double) hypotheses[h].getHeight(),(double) hypotheses[h].getWidth());
        
        if(angleRatio>maxAngleRatio)
            maxAngleRatio = angleRatio;	
	
        /**
         * probabilistic SVM for shape prior: Testing
         */
        CvSVM SVM;
        cv::Mat testDataSVM;
        /**
         * Feature Vector Formation
         */
        hconcat(hypotheses[h].getWidth()/edgeImage.cols, hypotheses[h].getHeight()/edgeImage.rows, testDataSVM);
        //hconcat(testDataSVM, (hypotheses[h].getWidth()/edgeImage.cols)/(hypotheses[h].getHeight()/edgeImage.rows)/maxAspRatio, testDataSVM);
        hconcat(testDataSVM, hypotheses[h].getWidth()/hypotheses[h].getHeight()/maxAspRatio, testDataSVM);
#if INCLUDE_DEPTH
        hconcat(testDataSVM, meanDepth[h]/hypotheses[h].getWidth()/maxDWAspRatio, testDataSVM);
        hconcat(testDataSVM, meanDepth[h]/hypotheses[h].getHeight()/maxDHAspRatio, testDataSVM);
        //hconcat(meanDepth[h]/hypotheses[h].getWidth()/maxDWAspRatio, meanDepth[h]/hypotheses[h].getHeight()/maxDHAspRatio, testDataSVM);
        hconcat(testDataSVM, meanDepth[h]/maxDepthGlobal, testDataSVM);// Normalised mean Depth
#endif
        hconcat(testDataSVM, angleRatio/maxAngleRatio, testDataSVM);// Normalised Angle Ratio between diagonals
        //hconcat(testDataSVM, formFactorH, testDataSVM);

        testDataSVM.convertTo(testDataSVM,CV_32FC1);

        // Determine the un-normalized probabilities
        for (int l = 0; l < 3; l++)
        {
            // Get the reconstruction error
            reconstructionErrors[l] = calcCodebookError(codebooks[l], p);
            
            // Get the prior probability (SVM)
            switch (l)
            {
                case 0 :
                    SVM.load("class0vsAllSVM.xml");
                    break;
                case 1 :
                    SVM.load("class1vsAllSVM.xml");
                    break;
                case 2 :
                    SVM.load("class2vsAllSVM.xml");
                    break;
                default :
                    std::cout<<"unknown label"<<std::endl;
            }

            float predicted = SVM.predict(testDataSVM, true);

            //Sigmoid of output
            priorProbabilities[l] = 1 - (1.0 / (1.0 + exp(-STEEPNESS*predicted)));

        }
        
        // Determine the normalized probabilities
        // For the reconstruction error, we first normalize the probabilities 
        // using the zscore. 
        //Util::normalizeZScore<3>(reconstructionErrors);
        
        /**
         * Rectangle Weights: Bayesian
         */

        float likelihoods[3];
        float normalizationFactor = 0.0f;
        for (int l = 0; l < 3; l++)
        {
            likelihoods[l] = -reconstructionErrors[l]/0.01f;
            likelihoods[l] += std::log(priorProbabilities[l]);
            likelihoods[l] += std::log(labelPrior[l]);
            normalizationFactor  += exp(likelihoods[l]);
        }

        float posterior[3];

        for (int l = 0; l < 3; l++)
        {
            posterior[l] = exp(likelihoods[l])/normalizationFactor;// normalised 0-1
        }




        for (int l = 0; l < 3; l++)
        {
#if 0
            std::cout << " ======== \n";
            {
                cv::Mat demo(edgeImage.rows, edgeImage.cols, CV_8UC3);
                cannyEdges.copyTo(demo);
                cv::cvtColor(demo, demo, CV_GRAY2BGR);

                std::cout << std::setw(25) << "Class=" << l << std::endl;
                std::cout << std::setw(25) << "appearance likelihood=" << exp(-reconstructionErrors[l]/0.01f) << std::endl;
                std::cout << std::setw(25) << "prior probability=" << priorProbabilities[l] << std::endl;
                std::cout << std::setw(25) << "label prior=" << labelPrior[l] << std::endl;
                std::cout << std::setw(25) << "Posterior (final weight)=" << posterior[l] << std::endl;
                
                switch (l)
                {
                    case 0:
                        PlotUtil::plotRectangle(demo, hypotheses[h], cv::Scalar(0,0,255), 2);
                        break;
                    case 1:
                        PlotUtil::plotRectangle(demo, hypotheses[h], cv::Scalar(0,255,0), 2);
                        break;
                    case 2:
                        PlotUtil::plotRectangle(demo, hypotheses[h], cv::Scalar(255,0,0), 2);
                        break;
                }
                cv::imshow("test", demo);
                cv::waitKey();
            }
#endif
            // Create the part (=Interaction Element = weighted and labelled rectangle)
            Part part;
            part.rect = hypotheses[h];
            part.label = l;
            part.meanDepth = meanDepth[h];
            part.likelihood = exp(-reconstructionErrors[l]/0.01f);
            part.shapePrior = priorProbabilities[l];
            part.posterior = posterior[l];
            part.projProf = projProf;
            part.projProfTyp = projProfTyp;
            
            #pragma omp critical
            {
                partHypotheses.push_back(part);
            }
        }
    }


#if 0
    // Sort the rectangles
    std::sort(partHypotheses.begin(), partHypotheses.end(), [](const Part & lhs, const Part & rhs) -> bool {
        return lhs.posterior > rhs.posterior;
    });

    for (size_t p = 0; p < partHypotheses.size(); p++)
    {
        partHypotheses[p].likelihood -= partHypotheses[0].likelihood;
    }
    for (size_t p = 0; p < partHypotheses.size(); p++)
    {
        partHypotheses[p].likelihood = log1p((double) partHypotheses[p].likelihood);
    }
#endif
    
#if 0
    {
        cv::Mat demo(edgeImage.rows, edgeImage.cols, CV_8UC3);
        for (size_t h = 0; h < partHypotheses.size(); h++)
        {
            cannyEdges.copyTo(demo);
            cv::cvtColor(demo, demo, CV_GRAY2BGR);

            std::cout << partHypotheses[h].posterior << "\n";
            switch (partHypotheses[h].label)
            {
                case 0:
                    PlotUtil::plotRectangle(demo, partHypotheses[h].rect, cv::Scalar(0,0,255), 2);
                    break;
                case 1:
                    PlotUtil::plotRectangle(demo, partHypotheses[h].rect, cv::Scalar(0,255,0), 2);
                    break;
                case 2:
                    PlotUtil::plotRectangle(demo, partHypotheses[h].rect, cv::Scalar(255,0,0), 2);
                    break;
            }
            cv::imshow("test", demo);
            cv::waitKey();
        }
    }
#endif

    // Rectangle proposal pool
    std::vector<Rectangle> proposals;
    for(size_t i = 0; i<partHypotheses.size(); i++)
    {
        proposals.push_back(partHypotheses[i].rect);
    }

    // One time computation of all matrices related to proposal rectangles
    Eigen::MatrixXi overlapPairs;
    Eigen::MatrixXi overlapPairs70;
    std::vector<float> areas;
    Eigen::MatrixXf overlapArea;
    Eigen::MatrixXi widthMergeable;
    Eigen::MatrixXi heightMergeable;
    computeProposalMatrices(proposals, imageArea, overlapPairs, overlapPairs70, areas, overlapArea, widthMergeable, heightMergeable);

    Eigen::SparseMatrix<int> widthMergeableSparse = widthMergeable.sparseView();
    std::cout<<"Possible Width Mergeable pairs : "<<widthMergeableSparse.nonZeros()<<" out of a max of : "<<( proposals.size()*proposals.size() - proposals.size() )/2<<std::endl;

    Eigen::SparseMatrix<int> heightMergeableSparse = heightMergeable.sparseView();
    std::cout<<"Possible Height Mergeable pairs : "<<heightMergeableSparse.nonZeros()<<" out of a max of : "<<( proposals.size()*proposals.size() - proposals.size() )/2<<std::endl;
    
    MCMCParserStateType bestState;
    //int initSize = std::max(1,rand() % 15);
    int initSize = 1;

    // Simulated annealing parameters
    SimulatedAnnealing sa(partHypotheses,
                          proposals,
                          gradMag,
                          areas,
                          overlapPairs,
                          overlapPairs70,
                          overlapArea,
                          cannyEdges);
    sa.setNumInnerLoops(NUM_INNER_LOOPS);
    sa.setMaxNoUpdateIterations(MAX_UPDATE_ITER);

    // Set up the cooling schedule
    GeometricCoolingSchedule schedule;
    schedule.setAlpha(ALPHA);
    schedule.setStartTemperature(MAX_TEMP);
    schedule.setEndTemperature(MIN_TEMP);
    sa.setCoolingSchedule(schedule);

    //sa.setProposals(proposals);

    // KEEP THE ORDER of MOVES
    // 0 RANDOM EXCHANGE
    // 1 BIRTH
    // 2 DEATH
    // 3 SPLIT
    // 4 MERGE
    // 5 LABEL DIFFUSE
    // 6 DD EXCHANGE
    // 7 UPDATE CENTER
    // 8 UPDATE WIDTH
    // 9 UPDATE HEIGHT


    std::vector<float> initialMoveProbs;

    //Set up the moves

    //Random Exchange move
    MCMCParserExchangeMove exchangeMove(partHypotheses);
    sa.addMove(&exchangeMove, INIT_PROB_EXCHANGE_RANDOM);
    initialMoveProbs.push_back(INIT_PROB_EXCHANGE_RANDOM);

    //Birth move
    MCMCParserBirthMove birthMove(partHypotheses, overlapPairs, numClusters);
    sa.addMove(&birthMove, INIT_PROB_BIRTH);
    initialMoveProbs.push_back(INIT_PROB_BIRTH);

    //Death move
    MCMCParserDeathMove deathMove(partHypotheses, overlapPairs, numClusters);
    sa.addMove(&deathMove, INIT_PROB_DEATH);
    initialMoveProbs.push_back(INIT_PROB_DEATH);

    //Split move
    MCMCParserSplitMove splitMove(proposals);
    sa.addMove(&splitMove, INIT_PROB_SPLIT);
    initialMoveProbs.push_back(INIT_PROB_SPLIT);

    //Merge move
    MCMCParserMergeMove mergeMove(proposals);
    sa.addMove(&mergeMove, INIT_PROB_MERGE);
    initialMoveProbs.push_back(INIT_PROB_MERGE);

#if 0
    MCMCParserSplitMove splitMove(static_cast<int>(partHypotheses.size()));
    sa.addMove(&splitMove, INIT_PROB_SPLIT);
    initialMoveProbs.push_back(INIT_PROB_SPLIT);

    MCMCParserMergeMove mergeMove(proposals, widthMergeable, heightMergeable);
    sa.addMove(&mergeMove, INIT_PROB_MERGE);
    initialMoveProbs.push_back(INIT_PROB_MERGE);
#endif

    //Switch(Label Diffuse) move
    MCMCParserLabelDiffuseMove labelDiffuseMove(partHypotheses);
    sa.addMove(&labelDiffuseMove, INIT_PROB_LABEL_DIFFUSE);
    initialMoveProbs.push_back(INIT_PROB_LABEL_DIFFUSE);

    //Data driven Exchange move
    MCMCParserDDExchangeMove exchangeDDMove(partHypotheses, overlapPairs70);// Needs unsorted rectangles
    sa.addMove(&exchangeDDMove, INIT_PROB_EXCHANGE_DATADRIVEN);
    initialMoveProbs.push_back(INIT_PROB_EXCHANGE_DATADRIVEN);

    //Update Center move
    MCMCParserUpdateCenterDiffuseMove updateCenterMove;
    sa.addMove(&updateCenterMove, INIT_PROB_UPDATE_CENTER);
    initialMoveProbs.push_back(INIT_PROB_UPDATE_CENTER);

    //Update Width move
    MCMCParserUpdateWidthDiffuseMove updateWidthMove;
    sa.addMove(&updateWidthMove, INIT_PROB_UPDATE_WIDTH);
    initialMoveProbs.push_back(INIT_PROB_UPDATE_WIDTH);

    //Update Height move
    MCMCParserUpdateHeightDiffuseMove updateHeightMove;
    sa.addMove(&updateHeightMove, INIT_PROB_UPDATE_HEIGHT);
    initialMoveProbs.push_back(INIT_PROB_UPDATE_HEIGHT);

    // Set up the energy function
    MCMCParserEnergy energyObj(partHypotheses, areas, overlapPairs, overlapArea, edgeImage);
    sa.setEnergyFunction(energyObj);

    //2phase architecture
#if DUMMY_MCMC_LOGIC
    // We prune the forest using simulated annealing

    //Phase 1 : Warmup
    SimulatedAnnealing saWarmup(partHypotheses,
                          proposals,
                          gradMag,
                          areas,
                          overlapPairs,
                          overlapPairs70,
                          overlapArea,
                          cannyEdges);


    saWarmup.setNumInnerLoops(NUM_INNER_LOOPS_WARMUP);
    saWarmup.setMaxNoUpdateIterations(MAX_UPDATE_ITER_WARMUP);

    // Set up the cooling schedule
    GeometricCoolingSchedule scheduleWarmup;
    scheduleWarmup.setAlpha(ALPHA_WARMUP);
    scheduleWarmup.setStartTemperature(MAX_TEMP_WARMUP);
    scheduleWarmup.setEndTemperature(MIN_TEMP_WARMUP);
    saWarmup.setCoolingSchedule(scheduleWarmup);

    std::vector<float> initialMoveProbsWarmup;

    //Set up the moves
    saWarmup.addMove(&exchangeMove, INIT_PROB_EXCHANGE_RANDOM);
    saWarmup.addMove(&birthMove, INIT_PROB_BIRTH);
    saWarmup.addMove(&deathMove, INIT_PROB_DEATH);
    // Energy Function
    saWarmup.setEnergyFunction(energyObj);

#endif

#if DUMMY_MCMC_LOGIC
    // We prune the forest using simulated annealing
    //Phase 2
    SimulatedAnnealing saMaster(partHypotheses,
                          proposals,
                          gradMag,
                          areas,
                          overlapPairs,
                          overlapPairs70,
                          overlapArea,
                          cannyEdges);

    saMaster.setNumInnerLoops(NUM_INNER_LOOPS_MASTER);
    saMaster.setMaxNoUpdateIterations(MAX_UPDATE_ITER_MASTER);

    // Set up the cooling schedule
    GeometricCoolingSchedule scheduleMaster;
    scheduleMaster.setAlpha(ALPHA_MASTER);
    scheduleMaster.setStartTemperature(MAX_TEMP_MASTER);
    scheduleMaster.setEndTemperature(MIN_TEMP_MASTER);
    saMaster.setCoolingSchedule(scheduleMaster);


    //Set up the moves
    saMaster.addMove(&exchangeMove, 1.0f);//Diffuse move only
    //Energy Function
    saMaster.setEnergyFunction(energyObj);

#endif


    // Initialization of Markov Chain
    std::cout<<std::endl<<std::endl<<"Intial State has chosen "<<initSize<<" rectangles from a total pool of "<<partHypotheses.size()<<" rectangles"<<std::endl;
    MCMCParserStateType state;
    int rectIdx = 0;
    float maxPosterior = 0.0f;
    //Find the rectangle with highest posterior
    for (int i = 0; i < partHypotheses.size(); i++)
    {
        maxPosterior = std::max(maxPosterior, partHypotheses[i].posterior);
        if(partHypotheses[i].posterior == maxPosterior)
        {
            rectIdx = i;
        }
    }
    for (int i = 0; i < initSize; i++)
    {

        //int rectIdx = rand() % partHypotheses.size();
        std::cout<<"The Max posterior is: "<<partHypotheses[rectIdx].posterior<<"   with rectangle index : "<<rectIdx<<std::endl;
        state.push_back(rectIdx);
    }


    float bestLabelEnergy;
    float bestWeightEnergy;
    float bestLayoutVarianceEnergy;
    float bestOverlapEnergy;
    float bestCoverEnergy;
    float bestStateSizeEnergy;

    //#pragma omp parallel for
    for (int iter = 1; iter <= 1; iter++)
    {

        // Set up the callback function
#if 0
        MCMCParserCallback callback(partHypotheses);
        sa.addCallback(&callback);
#endif

        std::cout<<" "<<std::endl;
#if DUMMY_MCMC_LOGIC
        float warmupBestEnergy = saWarmup.optimize(state);
        std::cout<<std::endl<<"Warmup initial state size "<<state.size()<<std::endl;
        float bestEnergy = saMaster.optimize(state);

#else
        float bestEnergy = sa.optimize(state);
#endif

        // Energy Function
        energyObj.energy(state, initialMoveProbs, areas, overlapPairs, overlapArea, partHypotheses);

        //#pragma omp critical
        {
#if 0
                std::cout << "labelEnergy = " << energyObj.lastLabelEnergy << ", labelPrior = " << energyObj.lastLabelPrior << ", layoutEnergy = " << energyObj.lastLayoutEnergy << "\n";
                std::cout << l << " - " << optEnergy << "\n";
                cv::Mat demo;
                edgeImage.copyTo(demo);
                cv::cvtColor(demo, demo, CV_GRAY2BGR);
                for (int i = 0; i < l; i++)
                {
                        switch (partHypotheses[state[i]].label)
                        {
                            case 0:
                                PlotUtil::plotRectangle(demo, partHypotheses[state[i]].rect, cv::Scalar(0,0,255), 2);
                                break;
                            case 1:
                                PlotUtil::plotRectangle(demo, partHypotheses[state[i]].rect, cv::Scalar(0,255,0), 2);
                                break;
                            case 2:
                                PlotUtil::plotRectangle(demo, partHypotheses[state[i]].rect, cv::Scalar(255,0,0), 2);
                                break;
                        }
                }
                cv::imshow("test", demo);
                cv::waitKey();
#endif 
                // Segmentation = Best State
                bestState = state;
                std::cout << std::endl<< std::endl;
                std::cout << "Finished with size" << std::setw(5) << bestState.size()<<" with Energy: "<<bestEnergy<<std::endl;
                std::cout << std::endl<< std::endl;
                bestLabelEnergy = energyObj.lastLabelEnergy;
                bestWeightEnergy = energyObj.lastWeightEnergy;
                bestLayoutVarianceEnergy = energyObj.lastLayoutVarianceEnergy;
                bestOverlapEnergy = energyObj.lastOverlapEnergy;
                bestCoverEnergy = energyObj.lastCoverEnergy;
                bestStateSizeEnergy = energyObj.lastStateSizeEnergy;
                std::cout << " best labelEnergy = " << energyObj.lastLabelEnergy <<std::endl<<
                             " best lastWeightEnergy = " << energyObj.lastWeightEnergy <<std::endl<<
                             " best lastLayoutVarianceEnergy = " << energyObj.lastLayoutVarianceEnergy <<std::endl<<
                             " best lastOverlapEnergy = " << energyObj.lastOverlapEnergy <<std::endl<<
                             " best lastCoverEnergy = " << energyObj.lastCoverEnergy <<std::endl<<
                             " best lastStateSizeEnergy = " << energyObj.lastStateSizeEnergy << "\n";
                std::cout << std::endl;


            }
        }


    result.resize(bestState.size());
    for (size_t h = 0; h < bestState.size(); h++)
    {
        std::cout << partHypotheses[bestState[h]].rect << " - > " << partHypotheses[bestState[h]].label << "\n";
        std::cout << "Mean part depth: "<<partHypotheses[bestState[h]].meanDepth <<std::endl;
        std::cout << "Posterior: "<<partHypotheses[bestState[h]].posterior <<std::endl;
        std::cout << "Likelihood: "<<partHypotheses[bestState[h]].likelihood <<std::endl;
        std::cout << "Shape Prior: "<<partHypotheses[bestState[h]].shapePrior <<std::endl;

        std::cout <<std::endl;
        result[h] = partHypotheses[bestState[h]];
    }


}

/*
 * One time Computation of all proposal matrices
*/
void CabinetParser::computeProposalMatrices(const std::vector<Rectangle> proposals, const float imageArea,
                Eigen::MatrixXi & overlapPairs, Eigen::MatrixXi & overlapPairs70,std::vector<float> & areas,
                Eigen::MatrixXf & overlapArea, Eigen::MatrixXi & widthMergeable, Eigen::MatrixXi & heightMergeable)
{
    //Initialize the matrices
    areas.resize(proposals.size());
    overlapPairs = Eigen::MatrixXi::Zero(static_cast<int>(proposals.size()),static_cast<int>(proposals.size()));
    overlapPairs70 = Eigen::MatrixXi::Zero(static_cast<int>(proposals.size()),static_cast<int>(proposals.size()));
    overlapArea = Eigen::MatrixXf::Zero(static_cast<int>(proposals.size()),static_cast<int>(proposals.size()));
    widthMergeable = Eigen::MatrixXi::Zero(static_cast<int>(proposals.size()),static_cast<int>(proposals.size()));
    heightMergeable = Eigen::MatrixXi::Zero(static_cast<int>(proposals.size()),static_cast<int>(proposals.size()));

    for (size_t n = 0; n < proposals.size(); n++)
    {

        //Area matrix
        areas[n] = proposals[n].getArea()/imageArea;
        // with overlap
        overlapPairs(static_cast<int>(n), static_cast<int>(n)) = 1;
        //with at least 70% overlap
        overlapPairs70(static_cast<int>(n), static_cast<int>(n)) = 1;
        // Extent of overlp
        overlapArea(static_cast<int>(n), static_cast<int>(n)) = 1.0f;

        for (size_t m = n+1; m < proposals.size(); m++)
        {
            // Compute the intersection rectangle
            Rectangle intersection;
            RectangleUtil::calcIntersection(proposals[n], proposals[m], intersection);
            const float intersectionScore = intersection.getArea()/(std::min(proposals[n].getArea(), proposals[m].getArea()));
            if (intersectionScore > MAX_OVERLAP)
            {
                overlapPairs(static_cast<int>(n),static_cast<int>(m)) = 1;
                overlapPairs(static_cast<int>(m),static_cast<int>(n)) = 1;
            }
            if (intersectionScore > 0.7f && intersectionScore < 1.0f)// Perfect matches also left out
            {
                overlapPairs70(static_cast<int>(n),static_cast<int>(m)) = 1;
                overlapPairs70(static_cast<int>(m),static_cast<int>(n)) = 1;
            }
            overlapArea(static_cast<int>(n),static_cast<int>(m)) = intersectionScore;//relative Area
            overlapArea(static_cast<int>(m),static_cast<int>(n)) = intersectionScore;


            //Mergeability Criteria: not all rectangle pairs are mergeable
#if 1
            //sharing width and center location with some allowance
            if(  std::abs(proposals[n].getWidth() - proposals[m].getWidth() ) < MERGE_ALLOWANCE && // width allowance
                    std::abs( proposals[n].getCenter()[0] - proposals[m].getCenter()[0] ) < MERGE_ALLOWANCE ) // center x co-ordinate shift allowance
            {
                if( ( proposals[n].getCenter()[1] - proposals[m].getCenter()[1] ) < 0 )//y co-ordinate check
                {
                    widthMergeable(static_cast<int>(n),static_cast<int>(m)) = 1;//n on top m on bottom
                }
                else
                {
                    widthMergeable(static_cast<int>(m),static_cast<int>(n)) = 1;
                }
            }

            //sharing height and center location with some allowance
            if( std::abs( proposals[n].getHeight() - proposals[m].getHeight() ) < MERGE_ALLOWANCE && // width allowance
                    std::abs( proposals[n].getCenter()[1] - proposals[m].getCenter()[1] ) < MERGE_ALLOWANCE ) // center y co-ordinate allowance
            {
                if( ( proposals[n].getCenter()[0] - proposals[m].getCenter()[0] ) < 0 )//x coordinate check
                {
                    heightMergeable(static_cast<int>(n),static_cast<int>(m)) = 1;//n on left m on right
                }
                else
                {
                    heightMergeable(static_cast<int>(m),static_cast<int>(n)) = 1;
                }
            }
#endif

        }
    }

}

/*
 * Shape Prior Training: SVM-based
*/

void CabinetParser::trainPartPrior(const std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat > > & images)
{

        // Extract the data
    libf::DataStorage::ptr dataStorage = libf::DataStorage::Factory::create();
    // extract the shape data
    extractRectangleData(dataStorage, images);

    libf::CSVDataWriter writer;
    writer.write("sizes.csv", dataStorage);
#if 0
    libf::CSVDataReader reader;
    reader.read("sizes.csv", dataStorage);
#endif
    // linear SVM + sigmoid
    probSVMPrior(dataStorage);
}

/*
 * Shape Prior Computation: prob SVM training
*/

void CabinetParser::probSVMPrior(libf::DataStorage::ptr dataStorage)
{
    // Compute the maximum depth and aspect ratio values in the dataset
    float maxDepthGlobal = 0.0f;
    float maxAspRatio = 0.0f;
    float maxDWAspRatio = 0.0f;
    float maxDHAspRatio = 0.0f;
    float maxAngleRatio = 0.0f;
    
    double atanWH = 0, atanHW = 0;//for angle ratio
    
    for (int l = 0; l < 3; l++)
    {
        // Get the data points of this class
        libf::DataStorage::ptr storageDepthNorm = dataStorage->select([l](const libf::DataPoint & x, int c) {
            return c == l;
        })->hardCopy();


        for (int n = 0; n < storageDepthNorm->getSize(); n++)
        {
            //Angle Ratio
            atanWH = std::atan2(double (storageDepthNorm->getDataPoint(n)(2)), double (storageDepthNorm->getDataPoint(n)(3)));
            atanHW = std::atan2(double (storageDepthNorm->getDataPoint(n)(3)), double (storageDepthNorm->getDataPoint(n)(2)));          maxDepthGlobal = std::max(maxDepthGlobal,storageDepthNorm->getDataPoint(n)(5));
            //Aspect ratios
            maxAspRatio = std::max(maxAspRatio,storageDepthNorm->getDataPoint(n)(0)/storageDepthNorm->getDataPoint(n)(1));
            maxDWAspRatio = std::max(maxDWAspRatio,storageDepthNorm->getDataPoint(n)(6));
            maxDHAspRatio = std::max(maxDHAspRatio,storageDepthNorm->getDataPoint(n)(7));
            maxAngleRatio = std::max(maxAngleRatio, float (atanWH/atanHW));
        }
    }
#if 1
    std::cout<<"Maximum Aspect Ratio (Train) = "<<maxAspRatio<<std::endl;
    std::cout<<"Maximum DW Aspect Ratio (Train) = "<<maxDWAspRatio<<std::endl;
    std::cout<<"Maximum DH Aspect Ratio (Train) = "<<maxDHAspRatio<<std::endl;
    std::cout<<"Maximum Depth  (Train) = "<<maxDepthGlobal<<std::endl;
    std::cout<<"Maximum Angle Ratio  (Train) = "<<maxAngleRatio<<std::endl;
#endif


    // Form feature vector for SVM training
    cv::Mat trainDataSVM;
    cv::Mat trainDataSVMhoriz;
    int partCount[3];

    for (int l = 0; l < 3; l++)
    {
        // Get the data points of this class
        libf::DataStorage::ptr rectData = dataStorage->select([l](const libf::DataPoint & x, int c) {
                return c == l;
        })->hardCopy();

        partCount[l] = rectData->getSize();
        std::cout<<" Number of parts in class "<<l<< " is "<<partCount[l]<<std::endl;

        float formFac = 0.0f;
        double atanWHRect = 0, atanHWRect = 0;

        for (int n = 0; n < rectData->getSize(); n++)
        {
            // Horizontal( Feature Vector )Concatenation
            formFac = 4*rectData->getDataPoint(n)(0)*rectData->getDataPoint(n)(1)/pow((rectData->getDataPoint(n)(0) + rectData->getDataPoint(n)(1)),2);
            atanWHRect = std::atan2(double (rectData->getDataPoint(n)(2)), double (rectData->getDataPoint(n)(3)));
            atanHWRect = std::atan2(double (rectData->getDataPoint(n)(3)), double (rectData->getDataPoint(n)(2)));	    
            hconcat(rectData->getDataPoint(n)(0),rectData->getDataPoint(n)(1),trainDataSVMhoriz);// relative width, height, already normalised
            //hconcat(trainDataSVMhoriz,rectData->getDataPoint(n)(0)/rectData->getDataPoint(n)(1)/maxAspRatio,trainDataSVMhoriz);// aspect ratio/ROI aspect ratio normalised
            hconcat(trainDataSVMhoriz,rectData->getDataPoint(n)(4)/maxAspRatio,trainDataSVMhoriz);// aspect ratio normalised
#if INCLUDE_DEPTH
            hconcat(trainDataSVMhoriz,rectData->getDataPoint(n)(6)/maxDWAspRatio,trainDataSVMhoriz);// depth to width aspect ratio normalised
            hconcat(trainDataSVMhoriz,rectData->getDataPoint(n)(7)/maxDHAspRatio,trainDataSVMhoriz);// depth to height aspect ratio normalised
            //hconcat(rectData->getDataPoint(n)(6)/maxDWAspRatio,rectData->getDataPoint(n)(7)/maxDHAspRatio,trainDataSVMhoriz);
            hconcat(trainDataSVMhoriz,rectData->getDataPoint(n)(5)/maxDepthGlobal,trainDataSVMhoriz);// mean depth normalised
#endif
            hconcat(trainDataSVMhoriz, float (atanWHRect/atanHWRect)/maxAngleRatio,trainDataSVMhoriz);// angle Ratio between diagonals normalised
            //hconcat(trainDataSVMhoriz,formFac,trainDataSVMhoriz);// form factor
            // Vertical Concatenation
            trainDataSVM.push_back(trainDataSVMhoriz);

        }

        std::cout<<"There are "<<trainDataSVM.rows<<" datapoints  with "<<trainDataSVM.cols<<" dimensions each."<<std::endl;

    }

    //Store the training parameters
    cv::FileStorage fsWrite("trainParameters.yml", cv::FileStorage::WRITE);
    fsWrite << "maxDepthTrain" << maxDepthGlobal;
    fsWrite << "maxAngleRatio" << maxAngleRatio;
    fsWrite << "maxAspRatioTrain" << maxAspRatio;
    fsWrite << "maxDWAspRatioTrain" << maxDWAspRatio;
    fsWrite << "maxDHAspRatioTrain" << maxDHAspRatio;
    fsWrite << "class0Count" << partCount[0];
    fsWrite << "class1Count" << partCount[1];
    fsWrite << "class2Count" << partCount[2];
    fsWrite.release();

    //Train Using SVM
    trainSVM(trainDataSVM, partCount);
    std::cout << "done\n";
}

/*
 * Shape Prior: Train the SVM
*/

void CabinetParser::trainSVM(cv::Mat & trainDataSVM, int partCount[3])
{
    //Define SVM params
    CvSVM SVM;
    CvSVMParams trainParamsSVM;
    trainParamsSVM.svm_type    = CvSVM::C_SVC;
    trainParamsSVM.kernel_type = CvSVM::LINEAR;//Linear SVM
    trainParamsSVM.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
    //Training
    cv::Mat trainLabelsSVM;
    trainDataSVM.convertTo(trainDataSVM,CV_32FC1);

    for (int l = 0; l < 3; l++)
    {
#if 0
        std::cout<<"Class "<<l<<" labels "<<trainLabelsSVM<<std::endl;
#endif
        /*
         *Generate models for each class: one vs others scheme
        */

        switch(l)
        {
                case(0)://Class: Door
                    vconcat(cv::Mat::ones(partCount[0],1,CV_32FC1), -1*cv::Mat::ones(partCount[1] + partCount[2],1,CV_32FC1), trainLabelsSVM);
                    SVM.train_auto(trainDataSVM, trainLabelsSVM, cv::Mat(), cv::Mat(), trainParamsSVM);
                    SVM.save("class0vsAllSVM.xml");
                    break;
                case(1)://class: Drawer
                    vconcat(-1*cv::Mat::ones(partCount[0],1,CV_32FC1), cv::Mat::ones(partCount[1],1,CV_32FC1), trainLabelsSVM);
                    vconcat(trainLabelsSVM, -1*cv::Mat::ones(partCount[2],1,CV_32FC1), trainLabelsSVM);
                    SVM.train_auto(trainDataSVM, trainLabelsSVM, cv::Mat(),cv::Mat(), trainParamsSVM);
                    SVM.save("class1vsAllSVM.xml");
                    break;
                case(2)://class: Shelf
                    vconcat(-1*cv::Mat::ones(partCount[0]+partCount[1],1,CV_32FC1), cv::Mat::ones(partCount[2],1,CV_32FC1), trainLabelsSVM);
                    SVM.train_auto(trainDataSVM, trainLabelsSVM, cv::Mat(),cv::Mat(), trainParamsSVM);
                    SVM.save("class2vsAllSVM.xml");
                    break;
                default:
                    std::cout<<" invalid label"<<std::endl;
        }
    }
}


void CabinetParser::evaluateProbSVMPrior(libf::DataStorage::ptr dataStorage)
{
    //TO DO
}




void CabinetParser::createGeneralEdgeDetectorSet(const std::string & inputDirectory, const std::string & imageOutputDirectory, const std::string & groundTruthDirectory)
{
    // Load all images from the input directory
    std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat> > images;
    loadImage(inputDirectory, images);
    
    // Create the output images
    for (size_t i = 0; i < images.size(); i++)
    {
#if VERBOSE_MODE
        std::cout << "Processing image " << (i+1) << " out of " << images.size() << std::endl;
#endif
        
        // Rectify the ROI
        cv::Mat rectifiedRegionOfInterest;
        Processing::rectifyRegion(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, 500, rectifiedRegionOfInterest);
        
        // Save the rectified image
        std::stringstream ss1;
        ss1 << imageOutputDirectory << std::get<1>(images[i]).id << ".png";
        cv::imwrite(ss1.str(), rectifiedRegionOfInterest);
        
        // Plot the edge map
        cv::Mat edges;
        plotRectifiedEdgeMap(rectifiedRegionOfInterest, std::get<1>(images[i]), edges);
        
        // Save the edge map
        std::stringstream ss2;
        ss2 << groundTruthDirectory << "edges/" << std::get<1>(images[i]).id << ".png";
        cv::imwrite(ss2.str(), edges);
        
        // Plot the segmentation
        cv::Mat segImg;
        plotRectifiedSegmentation(rectifiedRegionOfInterest, std::get<1>(images[i]), segImg);
        std::stringstream ss3;
        ss3 << groundTruthDirectory << "segmentations/" << std::get<1>(images[i]).id << ".png";
        cv::imwrite(ss3.str(), segImg);
    }
}

/*
 *Proposal Pool Augmentation: Redundancy Removal
*/

void CabinetParser::removeRedundantRects(std::vector<Rectangle> & hypotheses, const float maxIOU_thresh)
{
    //TO DO: upgrade to hash table based logic

    // Intersection over union score
    double iouFactor;
    size_t hypSize = hypotheses.size();
    size_t i=0;

    while(i<hypSize)
    {
        size_t j=i+1;
        while(j< hypSize)
        {
            Rectangle intersection, unionRect;
            RectangleUtil::calcIntersection(hypotheses[i], hypotheses[j], intersection);
            RectangleUtil::calcUnion(hypotheses[i], hypotheses[j], unionRect);
            // Intersection over union score
            iouFactor = (double) intersection.getArea()/unionRect.getArea();

            if(iouFactor > maxIOU_thresh)// could be reduced for faster over pipeline if update h/w/loc moves are perfect
            {
                if(hypotheses[j].getArea() > hypotheses[i].getArea())// keep the biggest
                {
                    Rectangle temp = hypotheses[i];
                    hypotheses[i] = hypotheses[j];
                    hypotheses[j] = temp;
                }

                hypotheses.erase(hypotheses.begin() + j);
                hypSize--;

                j--;//to compensate for the left shift in index after removal
                // if not one more iteration of the method will be required
            }

            j++;
        }

        i++;
    }
}

/*
 *Proposal Pool Augmentation: Merge: Widthwise then heightwise
*/

void CabinetParser::augmentRectanglesMergeWH(std::vector<Rectangle> & hypotheses)
{
    int limit = hypotheses.size();

    // Merge by width
    for(size_t i=0; i<limit; i++)
    {
        for(size_t j=i+1; j<limit; j++)
        {

            if(  std::abs(hypotheses[i].getHeight() - hypotheses[j].getHeight() ) < MERGE_ALLOWANCE
                 && // height criteria
                 (std::abs( hypotheses[i].getCenter()[0] - hypotheses[j].getCenter()[0] )
                  - hypotheses[i].getWidth()/2 - hypotheses[j].getWidth()/2)  < 2*MERGE_ALLOWANCE
                 && // center distance criteria
                std::abs( hypotheses[i].getCenter()[1] - hypotheses[j].getCenter()[1] ) < MERGE_ALLOWANCE
                 ) // center aligning criteria
            {
                Rectangle mergedRect;

                //x coordinates
                mergedRect[0][0] = std::min(hypotheses[i][0][0],hypotheses[j][0][0]);//TL x coord
                mergedRect[1][0] = std::max(hypotheses[i][1][0],hypotheses[j][1][0]);//TL x coord
                mergedRect[2][0] = std::max(hypotheses[i][2][0],hypotheses[j][2][0]);//BL x coord
                mergedRect[3][0] = std::min(hypotheses[i][3][0],hypotheses[j][3][0]);//BL x coord

                //y coordinates
                mergedRect[0][1] = std::min(hypotheses[i][0][1],hypotheses[j][0][1]);//TL x coord
                mergedRect[1][1] = std::min(hypotheses[i][1][1],hypotheses[j][1][1]);//TL x coord
                mergedRect[2][1] = std::max(hypotheses[i][2][1],hypotheses[j][2][1]);//BL x coord
                mergedRect[3][1] = std::max(hypotheses[i][3][1],hypotheses[j][3][1]);//BL x coord

                hypotheses.push_back(mergedRect);

            }
        }
    }

    //redundancy removal to prevent proposal pool size explosion
    removeRedundantRects(hypotheses, CLUSTER_MAX_IOU);

    // Merge by height
    limit = hypotheses.size();

    for(size_t i=0; i<limit; i++)
    {
        for(size_t j=i+1; j<limit; j++)
        {

            if(  std::abs(hypotheses[i].getWidth() - hypotheses[j].getWidth() ) < MERGE_ALLOWANCE
                 && // width criteria
                 (std::abs( hypotheses[i].getCenter()[1] - hypotheses[j].getCenter()[1] )
                  - hypotheses[i].getHeight()/2 - hypotheses[j].getHeight()/2)  < 2*MERGE_ALLOWANCE
                 && // center distance criteria
                std::abs( hypotheses[i].getCenter()[0] - hypotheses[j].getCenter()[0] ) < MERGE_ALLOWANCE
                 ) // center aligning criteria
            {
                Rectangle mergedRect;

                //x coordinates
                mergedRect[0][0] = std::min(hypotheses[i][0][0],hypotheses[j][0][0]);//TL x coord
                mergedRect[1][0] = std::max(hypotheses[i][1][0],hypotheses[j][1][0]);//BL x coord
                mergedRect[2][0] = std::max(hypotheses[i][2][0],hypotheses[j][2][0]);//TL x coord
                mergedRect[3][0] = std::min(hypotheses[i][3][0],hypotheses[j][3][0]);//BL x coord

                //y coordinates
                mergedRect[0][1] = std::min(hypotheses[i][0][1],hypotheses[j][0][1]);//TL x coord
                mergedRect[1][1] = std::min(hypotheses[i][1][1],hypotheses[j][1][1]);//BL x coord
                mergedRect[2][1] = std::max(hypotheses[i][2][1],hypotheses[j][2][1]);//TL x coord
                mergedRect[3][1] = std::max(hypotheses[i][3][1],hypotheses[j][3][1]);//BL x coord

                hypotheses.push_back(mergedRect);

            }
        }
    }

}


/*
 *Proposal Pool Augmentation: Merge: Heightwise then Widthwise
*/
void CabinetParser::augmentRectanglesMergeHW(std::vector<Rectangle> & hypotheses)
{
    int limit = hypotheses.size();

    // Merge by height
    for(size_t i=0; i<limit; i++)
    {
        for(size_t j=i+1; j<limit; j++)
        {
            if(  std::abs(hypotheses[i].getWidth() - hypotheses[j].getWidth() ) < MERGE_ALLOWANCE
                 && // width criteria
                 (std::abs( hypotheses[i].getCenter()[1] - hypotheses[j].getCenter()[1] )
                  - hypotheses[i].getHeight()/2 - hypotheses[j].getHeight()/2)  < 2*MERGE_ALLOWANCE
                 && // center distance criteria
                std::abs( hypotheses[i].getCenter()[0] - hypotheses[j].getCenter()[0] ) < MERGE_ALLOWANCE
                 ) // center aligning criteria
            {
                Rectangle mergedRect;

                //x coordinates
                mergedRect[0][0] = std::min(hypotheses[i][0][0],hypotheses[j][0][0]);//TL x coord
                mergedRect[1][0] = std::max(hypotheses[i][1][0],hypotheses[j][1][0]);//BL x coord
                mergedRect[2][0] = std::max(hypotheses[i][2][0],hypotheses[j][2][0]);//TL x coord
                mergedRect[3][0] = std::min(hypotheses[i][3][0],hypotheses[j][3][0]);//BL x coord

                //y coordinates
                mergedRect[0][1] = std::min(hypotheses[i][0][1],hypotheses[j][0][1]);//TL x coord
                mergedRect[1][1] = std::min(hypotheses[i][1][1],hypotheses[j][1][1]);//BL x coord
                mergedRect[2][1] = std::max(hypotheses[i][2][1],hypotheses[j][2][1]);//TL x coord
                mergedRect[3][1] = std::max(hypotheses[i][3][1],hypotheses[j][3][1]);//BL x coord

                hypotheses.push_back(mergedRect);

            }
        }
    }

    // rectangle redundancy removal
    removeRedundantRects(hypotheses, CLUSTER_MAX_IOU);

    limit = hypotheses.size();

    // Merge by width
    for(size_t i=0; i<limit; i++)
    {
        for(size_t j=i+1; j<limit; j++)
        {
            if(  std::abs(hypotheses[i].getHeight() - hypotheses[j].getHeight() ) < MERGE_ALLOWANCE
                 && // height criteria
                 (std::abs( hypotheses[i].getCenter()[0] - hypotheses[j].getCenter()[0] )
                  - hypotheses[i].getWidth()/2 - hypotheses[j].getWidth()/2)  < 2*MERGE_ALLOWANCE
                 && // center distance criteria
                std::abs( hypotheses[i].getCenter()[1] - hypotheses[j].getCenter()[1] ) < MERGE_ALLOWANCE
                 ) // center aligning criteria
            {
                Rectangle mergedRect;

                //x coordinates
                mergedRect[0][0] = std::min(hypotheses[i][0][0],hypotheses[j][0][0]);//TL x coord
                mergedRect[1][0] = std::max(hypotheses[i][1][0],hypotheses[j][1][0]);//TL x coord
                mergedRect[2][0] = std::max(hypotheses[i][2][0],hypotheses[j][2][0]);//BL x coord
                mergedRect[3][0] = std::min(hypotheses[i][3][0],hypotheses[j][3][0]);//BL x coord

                //y coordinates
                mergedRect[0][1] = std::min(hypotheses[i][0][1],hypotheses[j][0][1]);//TL x coord
                mergedRect[1][1] = std::min(hypotheses[i][1][1],hypotheses[j][1][1]);//TL x coord
                mergedRect[2][1] = std::max(hypotheses[i][2][1],hypotheses[j][2][1]);//BL x coord
                mergedRect[3][1] = std::max(hypotheses[i][3][1],hypotheses[j][3][1]);//BL x coord

                hypotheses.push_back(mergedRect);

            }
        }
    }

}




/*
 * Merge Augmentation: First widthwise merge then heightwise
*/
void CabinetParser::augmentRectanglesSplitWH(const cv::Mat gradMag, std::vector<Rectangle> & hypotheses, const double imageArea)
{
    //VERTICAL

    std::vector<int> vertProjProf;//Projection profile from edges
    double vertProj;
    std::vector<int> splitPointsV;

    for(int i = 0; i<hypotheses.size(); i++)
    {

        double relativeArea = (double) hypotheses[i].getArea()/imageArea;
        if(relativeArea < 0.20f)// split only the rectangles with a minimal size
            continue;

        // Form vertical edge projection profile
        vertProjProf.clear();
        splitPointsV.clear();

        //Vertical Projection profile formation from edges
        for(int w = 0; w< hypotheses[i].getWidth(); w++)
        {
            vertProj = 0;
            for(int h = 0; h< hypotheses[i].getHeight(); h++)
            {
                vertProj += static_cast<int>(gradMag.at<float>(hypotheses[i][0][1] + h, hypotheses[i][0][0] + w));
            }

            vertProj /= hypotheses[i].getHeight();
            vertProjProf.push_back(static_cast<int>(vertProj));
        }

        splitPointsV.push_back(static_cast<int>(0));
        int prevPoint = static_cast<int>(0);
        //Deciding the split location
        for(int j = 0; j<hypotheses[i].getWidth(); j++)// projected to the left
        {
            if(vertProjProf[j] > PROJ_PROF_THRESH  && (static_cast<int>(j) - prevPoint) > 0.10f*hypotheses[i].getWidth() )
            {
                //std::cout<<vertProjProf[j]<<std::endl;
                splitPointsV.push_back(static_cast<int>(j));
                prevPoint = static_cast<int>(j);
            }
        }

        splitPointsV.push_back(static_cast<int>(hypotheses[i].getWidth()));

        // Synthesize horizontal rectangles
        if(splitPointsV.size()>2)
        {

            for(int k = 0; k< (splitPointsV.size()-1) ; k++)
            {
                for(int j = k; j< (splitPointsV.size()-1) ; j++)
                {
                    Rectangle newRectV;
                    newRectV = hypotheses[i];
                    newRectV[0][0] = splitPointsV[k];
                    newRectV[3][0] = splitPointsV[k];

                    newRectV[1][0] = splitPointsV[j+1];
                    newRectV[2][0] = splitPointsV[j+1];

                    Rectangle intersection;
                    RectangleUtil::calcIntersection(newRectV, hypotheses[i], intersection);
                    float intersectionScore = intersection.getArea()/hypotheses[i].getArea();
                    if (intersectionScore > 0.30f && intersectionScore < 0.7f)
                    {
                        hypotheses.push_back(newRectV);
                    }
                }
            }
        }

    }
    //remove redundancy of proposal pool
    removeRedundantRects(hypotheses, CLUSTER_MAX_IOU);

    //HORIZONTAL
    std::vector<int> horizProjProf;
    double horizProj;
    std::vector<int> splitPoints;

    //Horizontal edge projection profile formation
    for(int i = 0; i<hypotheses.size(); i++)
    {
        double relativeArea = (double) hypotheses[i].getArea()/imageArea;
        if(relativeArea < 0.20f)// split only the rectangles with a minimal size
            continue;

        // Form Horizontal projection profile based on gradient
        horizProjProf.clear();
        splitPoints.clear();

        //std::cout<<"Rectangle number: "<<i<<std::endl;
        //cv::Mat edgeRect = cv::Mat::zeros(hypotheses[i].getHeight(), hypotheses[i].getWidth(), CV_8UC1);

        for(int h = 0; h< hypotheses[i].getHeight(); h++)
        {
            horizProj = 0;
            for(int w = 0; w< hypotheses[i].getWidth(); w++)
            {
                //edgeRect.at<uchar>(h,w) = static_cast<int>(gradMag.at<float>(hypotheses[i][0][1] + h, hypotheses[i][0][0] + w));
                horizProj += static_cast<int>(gradMag.at<float>(hypotheses[i][0][1] + h, hypotheses[i][0][0] + w));
            }

            horizProj /= hypotheses[i].getWidth();
            horizProjProf.push_back(static_cast<int>(horizProj));
        }

        //cv::imshow("Edge Part", edgeRect);
        //cv::waitKey();

        //Decide split locations from the horizontal edge projection Profile
        splitPoints.push_back(static_cast<int>(0));
        int prevPoint = static_cast<int>(0);
        for(int j = 0; j<hypotheses[i].getHeight(); j++)// projected to the left
        {
            if(horizProjProf[j] > PROJ_PROF_THRESH && (static_cast<int>(j) - prevPoint) > 0.10f*hypotheses[i].getHeight() )// TO DO:GENERALISE
            // Kind of non maximum suppression
            {
                splitPoints.push_back(static_cast<int>(j));
                prevPoint = static_cast<int>(j);
            }
        }

        splitPoints.push_back(static_cast<int>(hypotheses[i].getHeight()));


        //Synthesize new vertical rectangles
        if(splitPoints.size()>2)
        {

            for(int k = 0; k< (splitPoints.size()-1) ; k++)
            {
                for(int j = k; j< (splitPoints.size()-1) ; j++)
                {
                    Rectangle newRect;
                    newRect = hypotheses[i];
                    newRect[0][1] = splitPoints[k];
                    newRect[1][1] = splitPoints[k];

                    newRect[2][1] = splitPoints[j+1];
                    newRect[3][1] = splitPoints[j+1];


                    Rectangle intersection;
                    RectangleUtil::calcIntersection(newRect, hypotheses[i], intersection);
                    float intersectionScore = intersection.getArea()/hypotheses[i].getArea();
                    if (intersectionScore > 0.30f && intersectionScore < 0.70f)
                    {
                        hypotheses.push_back(newRect);
                    }
                }
            }
        }

    }

}

/*
 * Split Augmentation: First heightwise split then widthwise
*/

void CabinetParser::augmentRectanglesSplitHW(const cv::Mat gradMag, std::vector<Rectangle> & hypotheses, const double imageArea)
{

    std::vector<int> horizProjProf;
    double horizProj;
    std::vector<int> splitPoints;

    for(int i = 0; i<hypotheses.size(); i++)
    {

        double relativeArea = (double) hypotheses[i].getArea()/imageArea;
        if(relativeArea < 0.20f)// split only the rectangles with a minimal size
            continue;

        // Form Horizontal projection profile based on gradient
        horizProjProf.clear();
        splitPoints.clear();

        //std::cout<<"Rectangle number: "<<i<<std::endl;
        //cv::Mat edgeRect = cv::Mat::zeros(hypotheses[i].getHeight(), hypotheses[i].getWidth(), CV_8UC1);

        for(int h = 0; h< hypotheses[i].getHeight(); h++)
        {
            horizProj = 0;
            for(int w = 0; w< hypotheses[i].getWidth(); w++)
            {
                //edgeRect.at<uchar>(h,w) = static_cast<int>(gradMag.at<float>(hypotheses[i][0][1] + h, hypotheses[i][0][0] + w));
                horizProj += static_cast<int>(gradMag.at<float>(hypotheses[i][0][1] + h, hypotheses[i][0][0] + w));
            }

            horizProj /= hypotheses[i].getWidth();
            horizProjProf.push_back(static_cast<int>(horizProj));
        }

        //cv::imshow("Edge Part", edgeRect);
        //cv::waitKey();

        splitPoints.push_back(static_cast<int>(0));
        int prevPoint = static_cast<int>(0);
        for(int j = 0; j<hypotheses[i].getHeight(); j++)// projected to the left
        {
            if(horizProjProf[j] > PROJ_PROF_THRESH && (static_cast<int>(j) - prevPoint) > 0.10f*hypotheses[i].getHeight() )// TO DO:GENERALISE
            // Kind of non maximum suppression
            {
                //std::cout<<horizProjProf[j]<<std::endl;
                splitPoints.push_back(static_cast<int>(j));
                prevPoint = static_cast<int>(j);
            }
        }

        splitPoints.push_back(static_cast<int>(hypotheses[i].getHeight()));


        // creat new vertical rectangles
        if(splitPoints.size()>2)
        {

            for(int k = 0; k< (splitPoints.size()-1) ; k++)
            {
                for(int j = k; j< (splitPoints.size()-1) ; j++)
                {
                    Rectangle newRect;
                    newRect = hypotheses[i];
                    newRect[0][1] = splitPoints[k];
                    newRect[1][1] = splitPoints[k];

                    newRect[2][1] = splitPoints[j+1];
                    newRect[3][1] = splitPoints[j+1];


                    Rectangle intersection;
                    RectangleUtil::calcIntersection(newRect, hypotheses[i], intersection);
                    float intersectionScore = intersection.getArea()/hypotheses[i].getArea();
                    if (intersectionScore > 0.30f && intersectionScore < 0.70f)
                    {
                        hypotheses.push_back(newRect);
                    }
                }
            }
        }


    }

    removeRedundantRects(hypotheses, CLUSTER_MAX_IOU);

    //VERTICAL

    std::vector<int> vertProjProf;
    double vertProj;
    std::vector<int> splitPointsV;

    for(int i = 0; i<hypotheses.size(); i++)
    {

        double relativeArea = (double) hypotheses[i].getArea()/imageArea;
        if(relativeArea < 0.20f)// split only the rectangles with a minimal size
            continue;

        // Form vertical edge projection profile
        vertProjProf.clear();
        splitPointsV.clear();


        for(int w = 0; w< hypotheses[i].getWidth(); w++)
        {
            vertProj = 0;
            for(int h = 0; h< hypotheses[i].getHeight(); h++)
            {
                vertProj += static_cast<int>(gradMag.at<float>(hypotheses[i][0][1] + h, hypotheses[i][0][0] + w));
            }

            vertProj /= hypotheses[i].getHeight();
            vertProjProf.push_back(static_cast<int>(vertProj));
        }

        splitPointsV.push_back(static_cast<int>(0));
        int prevPoint = static_cast<int>(0);
        for(int j = 0; j<hypotheses[i].getWidth(); j++)// projected to the left
        {
            if(vertProjProf[j] > PROJ_PROF_THRESH  && (static_cast<int>(j) - prevPoint) > 0.10f*hypotheses[i].getWidth() )
            {
                //std::cout<<vertProjProf[j]<<std::endl;
                splitPointsV.push_back(static_cast<int>(j));
                prevPoint = static_cast<int>(j);
            }
        }

        splitPointsV.push_back(static_cast<int>(hypotheses[i].getWidth()));

        // Create new horizontal rectangles
        if(splitPointsV.size()>2)
        {

            for(int k = 0; k< (splitPointsV.size()-1) ; k++)
            {
                for(int j = k; j< (splitPointsV.size()-1) ; j++)
                {
                    Rectangle newRectV;
                    newRectV = hypotheses[i];
                    newRectV[0][0] = splitPointsV[k];
                    newRectV[3][0] = splitPointsV[k];

                    newRectV[1][0] = splitPointsV[j+1];
                    newRectV[2][0] = splitPointsV[j+1];


                    Rectangle intersection;
                    RectangleUtil::calcIntersection(newRectV, hypotheses[i], intersection);
                    float intersectionScore = intersection.getArea()/hypotheses[i].getArea();
                    if (intersectionScore > 0.30f && intersectionScore < 0.7f)// TO DO: could be tuned
                    {
                        hypotheses.push_back(newRectV);
                    }
                }
            }
        }

    }

}

/*
 * Extract the edge projection profile
*/

void CabinetParser::extractEdgeProjectionProfile(const cv::Mat& gradMag, const Rectangle& rect, std::vector<int> & edgeProjProfile, std::vector<int> & indexWH)
{
    //Horizontal projection profile
    std::vector<int> horizProjProf;
    int horizProj;
    for(int h = 0; h< rect.getHeight(); h++)
    {
        horizProj = 0;
        for(int w = 0; w< rect.getWidth(); w++)
        {
            horizProj += static_cast<int>(gradMag.at<float>(rect[0][1] + h, rect[0][0] + w));
        }
        horizProj /= rect.getWidth();
        horizProjProf.push_back(static_cast<int>(horizProj));
    }

    //Vertical projection profile
    int vertProj;
    std::vector<int> vertProjProf;
    for(int w = 0; w< rect.getWidth(); w++)
    {
        vertProj = 0;
        for(int h = 0; h< rect.getHeight(); h++)
        {
            vertProj += static_cast<int>(gradMag.at<float>(rect[0][1] + h, rect[0][0] + w));
        }
        vertProj /= rect.getHeight();
        vertProjProf.push_back(static_cast<int>(vertProj));
    }



    for(int i = 0; i<horizProjProf.size(); i++)// projected to the left
    {
        if(horizProjProf[i] > 128)
        {
            //edgeProjProfile.push_back(static_cast<int>(i));
            //indexWH.push_back(static_cast<int>(1));
        }
    }

    for(int i = 0; i<vertProjProf.size(); i++)// projected to the bottom
    {
        if(vertProjProf[i] > 128)
        {
            edgeProjProfile.push_back(static_cast<int>(i));
            indexWH.push_back(static_cast<int>(0));
        }
    }

    //cv::imshow("Edge Part", edgeRect);
    //cv::waitKey();
}

void CabinetParser::extractDiscretizedAppearanceDataGM(const cv::Mat& gradMag, const Rectangle& part, libf::DataPoint& p, libf::DataPoint& p2)
{
    const int size = 100;
    
    Rectangle unitSquare;
    unitSquare[0][0] = 0;
    unitSquare[0][1] = 0;
    unitSquare[1][0] = size-1;
    unitSquare[1][1] = 0;
    unitSquare[2][0] = size-1;
    unitSquare[2][1] = size-1;
    unitSquare[3][0] = 0;
    unitSquare[3][1] = size-1;
    Rectangle _r = part;
    
    cv::Mat warped;
    if (_r.getHeight() < 10 || _r.getWidth() < 10)
    {
        Processing::warpImageGaussian(gradMag, part, warped, unitSquare, 4);
    }
    else
    {
        Processing::warpImageGaussian(gradMag, part, warped, unitSquare, 4);
    /*        int offset = 0;
        _r[0][0] += offset;
        _r[0][1] += offset;
        _r[1][0] -= offset;
        _r[1][1] += offset;
        _r[2][0] -= offset;
        _r[2][1] -= offset;
        _r[3][0] += offset;
        _r[3][1] -= offset;
        Processing::warpImageGaussian(gradMag, , warped, unitSquare, 4);*/
    }
    //Processing::normalizeFloatImageLebesgue(warped);
    cv::Mat flipped;
    cv::flip(warped, flipped, 1);
    
    p.resize(size*size);
    p2.resize(size*size);
    for (int i = 0; i < size*size; i++)
    {
        p(i) = warped.reshape(0,1).at<float>(0,i);
        p2(i) = flipped.reshape(0,1).at<float>(0,i);
    }
}

void CabinetParser::extractDiscretizedAppearanceData(const cv::Mat & rectifiedEdgeImage, const Rectangle & part, libf::DataPoint & descriptor)
{
    // Compute a homography from the part rectangle to the unit rectangle
    cv::Mat homography;
    Rectangle unitRectangle;
    RectangleUtil::getUnitRectangle(unitRectangle);
    Processing::computeHomography(part, unitRectangle, homography);
    
    std::vector<Vec2> samples;
    int offset = 0;
    
    // Collect the points
    for (int x = part[0][0] + offset; x <= part[1][0] - offset; x++)
    {
        Vec2 p;
        p[0] = x;
        for (int y = part[0][1] + offset; y <= part[3][1] - offset; y++)
        {
            p[1] = y;
            
            if (RectangleUtil::isInsideAxisAlignedRectangle(part, p) && rectifiedEdgeImage.at<uchar>(y,x) != 0)
            {
                Vec2 _p;
                VectorUtil::applyHomography(homography, p, _p);
                assert(!std::isnan(_p[0]) && !std::isnan(_p[1]));
                samples.push_back(_p);
            }
        }
    }
    
    // Set up the kernel distribution
    KernelDistribution<float, float, 2>::Model model;
    model.estimate(samples);
    KernelDistribution<float, float, 2> kde(model);
    kde.normalizeMCMC();
    
    int latticeResolution = 50;
    
    descriptor.resize((latticeResolution+1)*(latticeResolution+1));
    
    // Discretize the KDE
    for (int _x = 0; _x <= latticeResolution; _x++)
    {
        for (int _y = 0; _y <= latticeResolution; _y++)
        {
            Vec2 p;
            p[0] = _x/static_cast<float>(latticeResolution);
            p[1] = _y/static_cast<float>(latticeResolution);
            
            descriptor(_x + (latticeResolution+1)*_y) = kde.eval(p);
        }
    }
}

void CabinetParser::extractDiscretizedAppearanceDistributions(libf::DataStorage::ptr trainingSet, const std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat > > & images)
{
    int counter = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < images.size(); i++)
    {
        #pragma omp critical
        {
#if VERBOSE_MODE
        std::cout << "Processing image " << ++counter << " out of " << images.size() << std::endl;
#endif
        }

        // Get the rectified canny edge image
        cv::Mat rectifiedMultiChannelImage;
        extractRectifiedMultiChannelImage(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImage);
        
#if INCLUDE_DEPTH
        cv::Mat rectifiedMultiChannelImageDepth;
        extractRectifiedMultiChannelImage(std::get<2>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImageDepth);
#endif

#if APPEARANCE_AUGMENT

        // Apply the edge detector
        cv::Mat edgeImage;
        applyEdgeDetector(rectifiedMultiChannelImage, edgeImage, 0);

#if INCLUDE_DEPTH
        // Apply the edge detector
        cv::Mat edgeImageDepth;
        applyEdgeDetector(rectifiedMultiChannelImageDepth, edgeImageDepth, 1);

        cv::bitwise_or(edgeImage, edgeImageDepth, edgeImage);
#endif

#endif

        // Get the gradient magnitude image
        std::vector<cv::Mat> channels;
        cv::split(rectifiedMultiChannelImage, channels);
        cv::Mat gradMag;
        //Processing::computeGradientMagnitudeImageFloat(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], gradMag);
#if 0
	visualizeFloatImage(gradMag);
#endif

        // Get the depth gradient magnitude image
#if INCLUDE_DEPTH
        std::vector<cv::Mat> channelsD;
        cv::split(rectifiedMultiChannelImageDepth, channelsD);
        //cv::Mat gradMagD;
        //Processing::computeGradientMagnitudeImageFloat(channelsD[EDGE_DETECTOR_CHANNEL_INTENSITY], gradMagD);

#if 0
	visualizeFloatImage(gradMagD);
#endif

	//cv::bitwise_or(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], channelsD[EDGE_DETECTOR_CHANNEL_INTENSITY], channels[EDGE_DETECTOR_CHANNEL_INTENSITY]);
	//addWeighted(channelsD[EDGE_DETECTOR_CHANNEL_INTENSITY], 1.0, channels[EDGE_DETECTOR_CHANNEL_INTENSITY], 1.0, 0.0, channels[EDGE_DETECTOR_CHANNEL_INTENSITY]);
#endif

 	Processing::computeGradientMagnitudeImageFloat(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], gradMag);

#if 0
	visualizeFloatImage(gradMag);
#endif

#if APPEARANCE_AUGMENT

        // Get the canny edge image
        cv::Mat cannyEdges;
        Processing::computeCannyEdges(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], cannyEdges);
        
        // Detect rectangles
        std::vector<Rectangle> partHypotheses;
        detectRectangles(edgeImage, cannyEdges, partHypotheses);
#endif
        
        //cv::Mat rectifiedMultiChannelImage;
        //extractRectifiedMultiChannelImage(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImage);
        //std::vector<cv::Mat> channels;
        //cv::split(rectifiedMultiChannelImage, channels);
        //cv::Mat gradMag;
        //Processing::computeGradientMagnitudeImageFloat(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], gradMag);
        
        
        std::vector<Rectangle> rectified;
        rectifyParts(std::get<1>(images[i]).regionOfInterest, std::get<1>(images[i]).parts, rectified);
        removeNonRectanglePixels3(rectified, gradMag);

        for (size_t l = 0; l < rectified.size(); l++)
        {
            libf::DataPoint p1, p2;
            extractDiscretizedAppearanceDataGM(gradMag, rectified[l], p1, p2);
            #pragma omp critical
            {
                trainingSet->addDataPoint(p1, std::get<1>(images[i]).labels[l]);
                trainingSet->addDataPoint(p2, std::get<1>(images[i]).labels[l]);
            }
        }

#if APPEARANCE_AUGMENT

        // Determine the label for each rectangle
        for (size_t r = 0; r < partHypotheses.size(); r++)
        {
            float maxIOU = 0;
            int bestLabel = 0;
            
            // Check against the ground truth44
            for (size_t l = 0; l < rectified.size(); l++)
            {
                const Rectangle & gtRect = rectified[l];
                
                // Compute the IOU
                Rectangle intersectionRect, unionRect;
                RectangleUtil::calcIntersection(gtRect, partHypotheses[r], intersectionRect);
                RectangleUtil::calcUnion(gtRect, partHypotheses[r], unionRect);

                const float score = intersectionRect.getArea()/unionRect.getArea();

                if (score > maxIOU)
                {
                    maxIOU = score;
                    bestLabel = std::get<1>(images[i]).labels[l];
                }
            }
            #pragma omp critical
            {
                if (maxIOU > 0.85)
                {
                    libf::DataPoint p1, p2;
                    extractDiscretizedAppearanceDataGM(gradMag, partHypotheses[r], p1, p2);

                    trainingSet->addDataPoint(p1, bestLabel);
                    trainingSet->addDataPoint(p2, bestLabel);
                }
            }
	
        }
#endif

    }

}

void CabinetParser::determineLatentVariables(const Eigen::MatrixXf& codebook, const Eigen::MatrixXf& data, Eigen::MatrixXf& pis)
{
    const int K = codebook.cols();
    const int N = data.cols();
    
    Eigen::MatrixXf G = codebook.adjoint()*codebook;
    
    for (int n = 0; n < N; n++)
    {
        Eigen::VectorXf a = -data.col(n).adjoint()*codebook;

        try {
            GRBEnv env = GRBEnv();
            GRBModel model = GRBModel(env);
            // Create one variable for each part
            auto lowerBounds = new double[K];
            auto upperBounds = new double[K];
            auto types = new char[K];
            for (int k = 0; k < K; k++)
            {
                lowerBounds[k] = 0.0;
                upperBounds[k] = 1.0;
                types[k] = GRB_CONTINUOUS;
            }
            GRBVar* variables = model.addVars(lowerBounds, upperBounds, 0, types, 0, K);
            
            // Integrate new variables
            model.update();

            GRBQuadExpr obj = 0.0;
            // Add the linear terms
            for (int k = 0; k < K; k++)
            {
                obj += a(k)*variables[k];
            }
            // Add the quadratic terms
            for (int k = 0; k < K; k++)
            {
                for (int l = 0; l < K; l++)
                {
                    obj += 0.5*variables[k]*variables[l]*G(k,l);
                }
            }
            model.setObjective(obj, GRB_MINIMIZE);

            // Add the constraint
            GRBLinExpr con = 0.0;
            for (int k = 0; k < K; k++)
            {
                con += variables[k];
            }
            model.addConstr(con, GRB_LESS_EQUAL, 1.0);

            model.getEnv().set(GRB_IntParam_Threads, 2);
            model.getEnv().set(GRB_IntParam_OutputFlag, 0);

            model.optimize();

            for (int k = 0;  k < K; k++)
            {
                pis(k,n) = variables[k].get(GRB_DoubleAttr_X);
            }
            delete[] variables;
            delete[] lowerBounds;
            delete[] upperBounds;
            delete[] types;
        }
        catch(GRBException e) {
            std::cout << "Error code = " << e.getErrorCode() << std::endl;
            std::cout << e.getMessage() << std::endl;
            exit(1);
        } catch(...) {
            std::cout << "Exception during optimization" << std::endl;
            exit(1);
        }    
    }
}


void CabinetParser::learnCodebook(libf::AbstractDataStorage::ptr _storage, Eigen::MatrixXf& bestCodebook, int K)
{
    const int D = _storage->getDimensionality();
    const int N = std::min(numPartsCBUpperBound,_storage->getSize());
    std::vector<bool> res;
    libf::AbstractDataStorage::ptr storage = _storage->bootstrap(N, res);
    // Convert the data storage to a matrix
    Eigen::MatrixXf data(D, N);
    for (int n = 0; n < N; n++)
    {
        float normalize = storage->getDataPoint(n).lpNorm<1>();
        if (normalize < 1e-10)
        {
            normalize = 1;
        }
        data.col(n) = storage->getDataPoint(n);
    }
    
    // Fix some parameters for the learning process
    // How many runs with different random initializations shall be performed
    const int runs = 3;
    // The number of codebook entries
    //int K = 8;
    // How many iterations the inner loop shall perform
    const int M = 100;
    
    std::random_device rd;
    // Set up a list of indices in order to sample without replacement
    std::vector<int> indices(N);
    for (int n = 0; n < N; n++)
    {
        indices[n] = n;
    }
    float bestError = -1;
    
    std::vector<int> components(K);
    for (int k = 0; k < K; k++)
    {
        components[k] = k;
    }
    
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> d(0, 1e-5);
    
    for (int r = 0; r < runs; r++)
    {
        std::cout << "Start run " << (r+1) << "/" << runs << "\n";
        float lastError = -1;
        
        // Set up a new random codebook
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(rd()));
        Eigen::MatrixXf codebook(D, K);
        for (int k = 0; k < K; k++)
        {
            codebook.col(k) = data.col(indices[k]);
        }
        
        Eigen::MatrixXf pis(K, N);
        for (int k = 0; k < K; k++)
        {
            for (int n = 0; n < N; n++)
            {
                pis(k,n) = 1.0/K;
            }
        }
        
        // Optimize iteratively
        for (int m = 0; m < M; m++)
        {
            std::cout << "-- Iteration " << (m+1) << "/" << M << "\n";
            // Determine the latent variables
            determineLatentVariables(codebook, data, pis);
            
            // Update the codebook
#if 0
            for (int l = 0; l < 3; l++)
            {
                std::shuffle(components.begin(), components.end(), std::default_random_engine(rd()));
                for (int _k = 0; _k < K; _k++)
                {
                    int k = components[_k];
                    Eigen::MatrixXf pis_copy = pis;
                    pis_copy.row(k) *= 0;
                    const float normalize = pis.row(k)*pis.row(k).adjoint();
                    
                    codebook.col(k) = (data*pis.row(k).adjoint() - (codebook*pis_copy)*pis.row(k).adjoint())/(normalize + lambda);
                }
            }
#else
            {
                try {
                    GRBEnv env = GRBEnv();
                    GRBModel model = GRBModel(env);
                    // Create one variable for each part
                    GRBVar* variables = model.addVars(K*D, GRB_CONTINUOUS);

                    // Integrate new variables
                    model.update();

                    // Set objective: maximize the covered area and minimize the overlap 
                    // by using a high penalty on the overlap

                    GRBQuadExpr obj = 0.0;
                    // Add the linear terms
                    for (int n = 0; n < N; n++)
                    {
                        for (int k = 0; k < K; k++)
                        {
                            for (int d = 0; d < D; d++)
                            {
                                obj += -pis(k,n)*data(d,n)*variables[k*D+d];
                            }
                        }
                    }
                    
                    // Add the quadratic terms
                    for (int n = 0; n < N; n++)
                    {
                        for (int k = 0; k < K; k++)
                        {
                            for (int j = k; j < K; j++)
                            {
                                for (int d = 0; d < D; d++)
                                {
                                    double factor = 1;
                                    if (j == k)
                                    {
                                        factor = 2;
                                    }
                                    obj += factor*pis(k,n)*pis(j,n)*variables[k*D+d]*variables[j*D+d];
                                }
                            }
                        }
                    }
                    
                    model.setObjective(obj, GRB_MINIMIZE);

                    // Add the constraint
                    for (int k = 0; k < K; k++)
                    {
                        for (int d = 0; d < D; d++)
                        {
                            GRBLinExpr con2 = variables[k*D+d];
                            model.addConstr(con2, GRB_GREATER_EQUAL, 0);
                        }
                    }
                    

                    model.getEnv().set(GRB_IntParam_Threads, 4);
                    model.getEnv().set(GRB_IntParam_OutputFlag, 1);

                    // Optimize model
                    model.optimize();

                    for (int k = 0; k < K; k++)
                    {
                        for (int d = 0; d < D; d++)
                        {
                            codebook(d,k) = variables[k*D+d].get(GRB_DoubleAttr_X);
                        }
                    }
                    
                    delete[] variables;
                }
                catch(GRBException e) {
                    std::cout << "Error code = " << e.getErrorCode() << std::endl;
                    std::cout << e.getMessage() << std::endl;
                    exit(1);
                } catch(...) {
                    std::cout << "Exception during optimization" << std::endl;
                    exit(1);
                }
            }
#endif
            
            float error = (data - codebook*pis).norm();
            std::cout << "-- Error: " << (error) << "/" << (bestError) << "\n";
            if (error < bestError || bestError < 0)
            {
                bestError = error;
                bestCodebook = codebook;
            }

            if (lastError > 0 && (std::abs(error - lastError) < 1 || (error > lastError + 1.5)))
            {
                break;
            }
            lastError = error;
        }
    }
}

float CabinetParser::calcCodebookError(const Eigen::MatrixXf& codebook, const Eigen::VectorXf& x)
{
    const int K = codebook.cols();
    const int D = codebook.rows();
    
    // Determine the pis
    Eigen::MatrixXf pi(K, 1);
    determineLatentVariables(codebook, x, pi);
    
    const float temp = (x - codebook*pi).lpNorm<2>();
    return std::sqrt(temp*temp/D);
}


void CabinetParser::trainAppearanceCodeBook(const std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat > > & images)
{
    // Get the training set
    libf::DataStorage::ptr trainingSet = libf::DataStorage::Factory::create();
    extractDiscretizedAppearanceDistributions(trainingSet, images);
    
    libf::CSVDataWriter writer;
    writer.write("appearance_01.csv", trainingSet);
    
    // Save the training set, so we can analyze it in MATLAB
    //writer.write("appearances4.csv", trainingSet);
    //libf::CSVDataReader reader;
    //reader.read("appearance_01.csv", trainingSet);
    
    std::ofstream result("codebook.dat");
    if (!result.is_open())
    {
        std::cout << "oh crap\n";
        exit(4);
    }
    
    std::vector<int> codebooklengths({4, 4, 2, 8, 16});
    for (int l = 0; l < 3; l++)
    {
        // Get all points of this class
        libf::DataStorage::ptr storage = trainingSet->select([l](const libf::DataPoint & x, int label) {
            return label == l;
        })->hardCopy();
        
        std::cout << storage->getSize() << "\n";
        Eigen::MatrixXf codebook;
        learnCodebook(storage, codebook, codebooklengths[l]);
        
        libf::writeBinary(result, codebook);
    }
    result.close();
}

void CabinetParser::testAppearanceCodeBook(const std::vector<std::tuple<cv::Mat,  Segmentation, cv::Mat> >& images)
{
    // This is the data set that the edge detector is tested upon
    libf::DataStorage::ptr testSet = libf::DataStorage::Factory::create();
    extractDiscretizedAppearanceDistributions(testSet, images);
    
    libf::CSVDataWriter writer;
    writer.write("test_appearances.csv", testSet);
    
    std::vector<Eigen::MatrixXf> codebooks(5);
    std::ifstream res("codebook.dat");
    for (int l = 0; l < 5; l++)
    {
        libf::readBinary(res, codebooks[l]);
    }
    res.close();
    
    float misclas = 0;
    
    for (int n = 0; n < testSet->getSize(); n++)
    {
        const Eigen::VectorXf & x = testSet->getDataPoint(n);
        
        float bestError = -1;
        int bestLabel = 0;
        for (int l = 0; l < 4; l++)
        {
            float error = calcCodebookError(codebooks[l], x);
            if (error < bestError || bestError < 0)
            {
                bestError = error;
                bestLabel = l;
            }
        }
        
        if (bestLabel != testSet->getClassLabel(n))
        {
            misclas += 1;
        }
        
        std::cout << "TRUE: " << testSet->getClassLabel(n) << ", PREDICTED: " << bestLabel << ", ERROR: " << misclas/(n+1) << "\n";
    }
}


static void merge(const std::vector<Rectangle> & nodes, Rectangle & mergedRectangle)
{
    // Find the minimum/maximum x/y coordinates
    float minX = 1e10;
    float minY = 1e10;
    float maxX = -1e10;
    float maxY = -1e10;
    
    for (size_t i = 0; i < nodes.size(); i++)
    {
        minX = std::min(nodes[i].minX(), minX);
        minY = std::min(nodes[i].minY(), minY);
        maxX = std::max(nodes[i].maxX(), maxX);
        maxY = std::max(nodes[i].maxY(), maxY);
    }
    
    mergedRectangle[0][0] = minX;
    mergedRectangle[0][1] = minY;
    
    mergedRectangle[1][0] = maxX;
    mergedRectangle[1][1] = minY;
    
    mergedRectangle[2][0] = maxX;
    mergedRectangle[2][1] = maxY;
    
    mergedRectangle[3][0] = minX;
    mergedRectangle[3][1] = maxY;
}

static bool isValidMerge(const std::vector<Rectangle> & nodes, const std::vector<Rectangle> & collection)
{
    // Compute the merged rectangle
    Rectangle merged;
    merge(nodes, merged);
    
    // Check if the merge rectangle overlaps significantly with any other rectangle
    // in the collection
    for (size_t n = 0; n < collection.size(); n++)
    {
        // Compute the relative intersection area
        Rectangle intersection;
        RectangleUtil::calcIntersection(collection[n], merged, intersection);
        
        const float relativeIntersectionArea = intersection.getArea()/collection[n].getArea();
        
        if (relativeIntersectionArea > 0.15)
        {
            return false;
        }
    }
    
    return true;
}

/*void CabinetParser::pruneRectangles(float roiSize, const std::vector<Rectangle>& rectangles, int& packingNumber, int& coverNumber, std::vector<Rectangle> & preselection)
{
    // Compute a packing
    std::vector<Rectangle> packing;
    packingNumber = Processing::computeRectanglePackingMCMC(rectangles, packing);

#if 0
    cv::Mat demo(500,500, CV_8UC3);
    for (size_t i = 0; i < packing.size(); i++)
    {
        demo = cv::Scalar(0);
        for (size_t j = 0; j < packing.size(); j++)
        {
            PlotUtil::plotRectangle(demo, packing[j], cv::Scalar(0,255,0));
        }
        PlotUtil::plotRectangle(demo, packing[i], cv::Scalar(0,255,0), 5);
        cv::imshow("demo", demo);
        cv::waitKey();
    }
#endif
#if 0
    cv::Mat demo(500, 500, CV_8UC3);
    demo = cv::Scalar(0);
    for (size_t n = 0; n < packing.size(); n++)
    {
        PlotUtil::plotRectangle(demo, packing[n], cv::Scalar(255,0,255));
        cv::imshow("test", demo);
        cv::waitKey();
    }
#endif
    
#if 0
    // Prune the set
    for (size_t r = 0; r < rectangles.size(); r++)
    {
        float maxIOU = 0;
        for (size_t p = 0; p < packing.size(); p++)
        {
            maxIOU = std::max(maxIOU, RectangleUtil::calcIOU(rectangles[r], packing[p]));
        }
        
        if (maxIOU >= 0.65)
        {
            preselection.push_back(rectangles[r]);
        }
    }
#endif
#if 0
    // Prune the set
    for (size_t r = 0; r < rectangles.size(); r++)
    {
        bool isSimilarToPackingRect = false;
        for (size_t p = 0; p < packing.size(); p++)
        {
            if (parser::RectangleUtil::areSimilar(rectangles[r], packing[p], 45))
            {
                isSimilarToPackingRect = true;
                break;
            }
        }
        
        if (isSimilarToPackingRect)
        {
            preselection.push_back(rectangles[r]);
        }
    }
#endif
#if 0
    for (size_t r = 0; r < rectangles.size(); r++)
    {
        // Find all rectangles from the packing within this rectangle
        std::vector<size_t> intersectionRectangles;
        for (size_t p = 0; p < packing.size(); p++)
        {
            Rectangle intersection;
            RectangleUtil::calcIntersection(rectangles[r], packing[p], intersection);
            float intersectionArea = intersection.getArea();
            
            if (intersectionArea/packing[p].getArea() > 0.75)
            {
                intersectionRectangles.push_back(p);
            }
        }

        bool modularSubrectangles = false;

        for (size_t i = 0; i < intersectionRectangles.size(); i++)
        {
            for (size_t j = 0; j < packing.size(); j++)
            {
                if (intersectionRectangles[i] == j) continue;
                
                std::vector<Rectangle> nodes;
                nodes.push_back(packing[intersectionRectangles[i]]);
                nodes.push_back(packing[j]);
                
                std::vector<Rectangle> check;
                for (size_t k = 0; k < packing.size(); k++)
                {
                    if (k != intersectionRectangles[i] && k != j)
                    {
                        check.push_back(packing[k]);
                    }
                }
                
                if (isValidMerge(nodes, check) && RectangleUtil::areSimilar(packing[intersectionRectangles[i]], packing[j]))
                {
                    modularSubrectangles = true;
                    break;
                }
            }
        }
        
        if (!modularSubrectangles)
        {
            preselection.push_back(rectangles[r]);
        }
    }
    
    for (size_t p = 0;  p < packing.size(); p++)
    {
        preselection.push_back(packing[p]);
    }
    
#endif
#if 1
    
    for (size_t p = 0; p < packing.size(); p++)
    {
        bool isModular = false;
        
        // Check if this is a modular rectangle
        for (size_t r = 0; r < packing.size(); r++)
        {
            if (r == p) continue;

            std::vector<Rectangle> nodes;
            nodes.push_back(packing[r]);
            nodes.push_back(packing[p]);

            std::vector<Rectangle> check;
            for (size_t k = 0; k < packing.size(); k++)
            {
                if (k != p && k != r)
                {
                    check.push_back(packing[k]);
                }
            }

            if (isValidMerge(nodes, check) && RectangleUtil::areSimilar(packing[r], packing[p]))
            {
                isModular = true;
                break;
            }
        }
        
        if (!isModular)
        {
            for (size_t r = 0; r < rectangles.size(); r++)
            {
                float iou = RectangleUtil::calcIOU(rectangles[r], packing[p]);

                if (iou >= 0.45)
                {
                    preselection.push_back(rectangles[r]);
                }
            }
        }
        else
        {
            for (size_t r = 0; r < rectangles.size(); r++)
            {
                float iou = RectangleUtil::calcIOU(rectangles[r], packing[p]);

                if (iou >= 0.8)
                {
                    preselection.push_back(rectangles[r]);
                }
            }
        }
    }
    
    for (size_t p = 0;  p < packing.size(); p++)
    {
        preselection.push_back(packing[p]);
    }
    
#endif
#if 0

    // Compute the parse graph
    // Set up the terminal nodes
    std::vector<ParseTreeNode*> nodes(packing.size());
    for (size_t n = 0; n < packing.size(); n++)
    {
        nodes[n] = new ParseTreeNode();
        nodes[n]->rect = packing[n];
        nodes[n]->part = static_cast<int>(n);
    }
    
    ParserEnergy parserEnergy;
    parserEnergy.similarityThreshold = 50;
    ParseTreeNode* tree;
    bool ok = true;
    try {
        tree = parserEnergy.parse(nodes);            
    } catch(...) {
        ok = false;
    }

    if (ok)
    {
        std::vector<bool> modular(packing.size(), false);

        // Traverse the tree 
        std::vector<ParseTreeNode*> queue;
        queue.push_back(tree);
        while (queue.size() > 0)
        {
            ParseTreeNode* node = queue.back();
            queue.pop_back();

            // Is this a terminal node?
            if (node->isTerminal())
            {
                continue;
            }

            for (size_t i = 0; i < node->children.size(); i++)
            {
                if (node->children[i]->isTerminal())
                {
                    modular[node->children[i]->part] = node->mesh;
                }
            }

            for (size_t i = 0; i < node->children.size(); i++)
            {
                queue.push_back(node->children[i]);
            }
        }
        //delete tree;

        for (size_t p = 0; p < packing.size(); p++)
        {
            if (!modular[p])
            {
                for (size_t r = 0; r < rectangles.size(); r++)
                {
                    float iou = RectangleUtil::calcIOU(rectangles[r], packing[p]);

                    if (iou >= 0.45)
                    {
                        preselection.push_back(rectangles[r]);
                    }
                }
            }
            else
            {
                for (size_t r = 0; r < rectangles.size(); r++)
                {
                    float iou = RectangleUtil::calcIOU(rectangles[r], packing[p]);

                    if (iou >= 0.8)
                    {
                        preselection.push_back(rectangles[r]);
                    }
                }
            }
        }

        for (size_t p = 0;  p < packing.size(); p++)
        {
            preselection.push_back(packing[p]);
        }
    }
    else
    {
        // Prune the set
        for (size_t r = 0; r < rectangles.size(); r++)
        {
            float maxIOU = 0;
            for (size_t p = 0; p < packing.size(); p++)
            {
                maxIOU = std::max(maxIOU, RectangleUtil::calcIOU(rectangles[r], packing[p]));
            }

            if (maxIOU >= 0.45)
            {
                preselection.push_back(rectangles[r]);
            }
        }
    }

#endif
    // Compute the cover number on the preselection
    std::vector<Rectangle> cover;
    coverNumber = 0;
    coverNumber = Processing::computeRectangleCoverMCMC(roiSize, preselection, cover);
    coverNumber = std::max(1, coverNumber);
    packingNumber = std::max(packingNumber, coverNumber);
}
*/
void CabinetParser::createGroundtruthEdgeMap(const cv::Mat& image, const Segmentation& segmentation, cv::Mat & edges)
{
    edges = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    std::vector<Rectangle> rectified;
    rectifyParts(segmentation.regionOfInterest, segmentation.parts, rectified);
    
    for (size_t r = 0; r < rectified.size(); r++)
    {
        PlotUtil::plotRectangle(edges, rectified[r], 255);
    }
}

void CabinetParser::evaluateEdgeDetector(const std::vector<std::tuple<cv::Mat,  Segmentation, cv::Mat> >& images)
{
    for (int o = 0; o < 2; o++)
    {
        std::cout << std::endl;
        if (o == 0)
        {
            std::cout << "RF" << std::endl;
            std::cout << "==" << std::endl;
        }
        else
        {
            std::cout << "CANNY" << std::endl;
            std::cout << "=====" << std::endl;
        }
        
        int evNumberGTPixels = 0;
        int evNumberGTFoundPixels = 0;
        int evNumberDTPixels = 0;
        int evNumberDTTruePixels = 0;

        // Keep track on the
        // - precision: How many of the ground truth edges have been found
        // - recall: How many of the found edges are also in the ground truth
        int counter = 0;
        for (size_t i = 0; i < images.size(); i++)
        {
    #if VERBOSE_MODE
            std::cout << "Processing image " << ++counter << " out of " << images.size() << std::endl;
    #endif
            cv::Mat rectifiedMultiChannelImage;
            extractRectifiedMultiChannelImage(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImage);

            cv::Mat rectifiedMultiChannelImageDepth;
            extractRectifiedMultiChannelImage(std::get<2>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImageDepth);

            // Get the GT edges
            cv::Mat groundTruth;
            createGroundtruthEdgeMap(rectifiedMultiChannelImage, std::get<1>(images[i]), groundTruth);

            // Detect edges
            cv::Mat detected;
            cv::Mat detectedDepth;

            // Apply the edge detector
            if (o == 0)
            {
                applyEdgeDetector(rectifiedMultiChannelImage, detected, 0);
#if INCLUDE_DEPTH
                applyEdgeDetector(rectifiedMultiChannelImageDepth, detectedDepth, 1);
                cv::bitwise_or(detected, detectedDepth, detected);
#endif
            }
            else
            {
                cv::Mat intensityImage;
                std::vector<cv::Mat> channels;
                cv::split(rectifiedMultiChannelImage, channels);
                channels[EDGE_DETECTOR_CHANNEL_INTENSITY].convertTo(intensityImage, CV_8UC1);
                Processing::computeCannyEdges(intensityImage, detected);
            }

            // Compute the distance transforms to do the evaluation
            cv::Mat gtDist, dtDist;
            Processing::computeDistanceTransform(groundTruth, gtDist);
            Processing::computeDistanceTransform(detected, dtDist);

            // Evaluate
            for (int x = 0; x < groundTruth.cols; x++)
            {
                for (int y = 0; y < groundTruth.rows; y++)
                {
                    // Recall
                    if (groundTruth.at<uchar>(y,x) != 0)
                    {
                        evNumberGTPixels++;
                        if (dtDist.at<float>(y,x) <= 2)
                        {
                            evNumberGTFoundPixels++;
                        }
                    }

                    // Precision
                    if (detected.at<uchar>(y,x) != 0)
                    {
                        evNumberDTPixels++;
                        if (gtDist.at<float>(y,x) <= 2)
                        {
                            evNumberDTTruePixels++;
                        }
                    }
                }
            }
        }

        float recall = evNumberGTFoundPixels/static_cast<float>(evNumberGTPixels);
        float precision = evNumberDTTruePixels/static_cast<float>(evNumberDTPixels);
        float f1 = 2* precision * recall/(precision + recall);
        std::cout << "RESULT\n";
        std::cout << "Precision: " << precision << "\n";
        std::cout << "Recall:    " << recall << "\n";
        std::cout << "F1:        " << f1 << "\n";
        std::cout << "Ratio:     " << evNumberDTPixels/static_cast<float>(evNumberGTPixels) << "\n";
    }
}

void CabinetParser::evaluateRectangleDetector(const std::vector<std::tuple<cv::Mat,  Segmentation, cv::Mat> >& images)
{
    for (int o = 0; o < 1; o++)// Canny part commented out
    {
        std::cout << std::endl;
        if (o == 0)
        {
            std::cout << "RF" << std::endl;
            std::cout << "==" << std::endl;
        }
        else
        {
            std::cout << "CANNY" << std::endl;
            std::cout << "=====" << std::endl;
        }
        
        std::cout << "rectangleDetectionThreshold =  " << rectangleDetectionThreshold << "\n";
        std::cout << "rectangleAcceptanceThreshold = " << rectangleAcceptanceThreshold << "\n";
        std::cout.flush();
        int evNumberGTRectangles = 0;
        int evNumberGTFoundRectangles = 0;
        int evNumberDTRectangles = 0;
        int evNumberDTTrueRectangles = 0;
        int counter = 0;
        #pragma omp parallel for
        for (size_t i = 0; i < images.size(); i++)
        {
            #pragma omp critical 
            {
                //std::cout << "Processing image " << ++counter << " out of " << images.size() << std::endl;
                std::cout << '.';
                std::cout.flush();
            }
            cv::Mat rectifiedMultiChannelImage;
            extractRectifiedMultiChannelImage(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImage);
	    
            cv::Mat rectifiedMultiChannelImageDepth;
            extractRectifiedMultiChannelImage(std::get<2>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImageDepth);

            // Detect edges
            cv::Mat detected;
            cv::Mat detectedDepth;

            // Apply the edge detector
            applyEdgeDetector(rectifiedMultiChannelImage, detected, 0);
#if INCLUDE_DEPTH
            applyEdgeDetector(rectifiedMultiChannelImageDepth, detectedDepth, 1);
            cv::bitwise_or(detected, detectedDepth, detected);
#endif

            // Get the canny edge image
            std::vector<cv::Mat> channels;
            cv::split(rectifiedMultiChannelImage, channels);
            cv::Mat cannyEdges;
            Processing::computeCannyEdges(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], cannyEdges);

            // Detect rectangles
            std::vector<Rectangle> partHypotheses, unrectifiedPartHypotheses;
            if (o == 0)
            {
                detectRectangles(detected, cannyEdges, partHypotheses);
            }
            else
            {
                detectRectangles(cannyEdges, cannyEdges, partHypotheses);
            }

    #if SPLIT_MERGE_AUGMENT
            std::vector<cv::Mat> depthChannels;
            cv::split(rectifiedMultiChannelImageDepth, depthChannels);
            cv::Mat depthImage;
            depthChannels[EDGE_DETECTOR_CHANNEL_INTENSITY].convertTo(depthImage,CV_8UC1);

            cv::Mat gradMag;
            Processing::computeGradientMagnitudeImageFloat(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], gradMag);
            cv::Mat gradMagDepth;
            Processing::computeGradientMagnitudeImageFloat(depthImage, gradMagDepth);
#if INCLUDE_DEPTH
            float linearBlendAlpha = 0.7; // TO DO: tune this, now just 30% depth gradient
            float linearBlendBeta = 1 - linearBlendAlpha;
            addWeighted(gradMag, linearBlendAlpha, gradMagDepth, linearBlendBeta, 0.0, gradMagDepth);
#endif
            removeNonRectanglePixels3(partHypotheses, gradMag);
            double imageArea = detected.rows*detected.cols;
            removeRedundantRects(partHypotheses, CLUSTER_MAX_IOU);
            augmentRectanglesMergeWH(partHypotheses);
            removeRedundantRects(partHypotheses, CLUSTER_MAX_IOU);
            augmentRectanglesMergeHW(partHypotheses);
            removeRedundantRects(partHypotheses, CLUSTER_MAX_IOU);
            augmentRectanglesSplitWH(gradMag, partHypotheses, imageArea);
            removeRedundantRects(partHypotheses, CLUSTER_MAX_IOU);
            augmentRectanglesSplitHW(gradMag, partHypotheses, imageArea);
            removeRedundantRects(partHypotheses,CLUSTER_MAX_IOU);
            removeRedundantRects(partHypotheses, 0.8f);// TO DO: Tuning
    #endif

    #if 0
            unrectifyParts(std::get<1>(images[i]).regionOfInterest, partHypotheses, unrectifiedPartHypotheses);
            cv::Mat computedEdges;
            detected.copyTo(computedEdges);
            cv::bitwise_or(computedEdges, cannyEdges, computedEdges);
            Processing::add1pxBorders(computedEdges);
    #endif

            // Rectify the GT rectangles
            std::vector<Rectangle> rectified;
            rectifyParts(std::get<1>(images[i]).regionOfInterest, std::get<1>(images[i]).parts, rectified);

            int localevNumberGTFoundRectangles = 0;
            int localevNumberDTTrueRectangles = 0;

            // Evaluate
            for (size_t r = 0; r < rectified.size(); r++)
            {
                float maxIOU = 0;
                Rectangle bestMatch;
                for (size_t p = 0; p < partHypotheses.size(); p++)
                {
                    float iou = 0;
                    iou = RectangleUtil::calcIOU(rectified[r], partHypotheses[p]);
                    if (iou > maxIOU)
                    {
                        maxIOU = iou;
    #if 0
                        bestMatch = unrectifiedPartHypotheses[p];
    #endif
                    }
                }

                if (maxIOU >= rectangleAcceptanceThreshold)
                {
                    localevNumberGTFoundRectangles++;
                }
                else
                {
    #if 0
    #pragma omp critical 
                    {

                    std::cout << maxIOU << "\n";
                    cv::Mat demo;
                    std::get<0>(images[i]).copyTo(demo);
                    PlotUtil::plotRectangle(demo, std::get<1>(images[i]).parts[r], cv::Scalar(0,0,255), 1);
                    PlotUtil::plotRectangle(demo, bestMatch, cv::Scalar(0,255,0), 2);
                    cv::imshow("detected", detected);
                    cv::imshow("cannyEdges", cannyEdges);
                    cv::imshow("test", demo);
                    cv::imshow("img", std::get<0>(images[i]));
                    cv::waitKey();
                    }
    #endif
                }
            }
            for (size_t p = 0; p < partHypotheses.size(); p++)
            {
                float maxIOU = 0;
                for (size_t r = 0; r < rectified.size(); r++)
                {
                    maxIOU = std::max(maxIOU, RectangleUtil::calcIOU(rectified[r], partHypotheses[p]));
                }

                if (maxIOU >= rectangleAcceptanceThreshold)
                {
                    localevNumberDTTrueRectangles++;
                }
            }
            #pragma omp critical
            {
                evNumberDTRectangles += static_cast<int>(partHypotheses.size());
                evNumberGTRectangles += static_cast<int>(rectified.size());
                evNumberGTFoundRectangles += localevNumberGTFoundRectangles;
                evNumberDTTrueRectangles += localevNumberDTTrueRectangles;
            }
        }
        float recall = evNumberGTFoundRectangles/static_cast<float>(evNumberGTRectangles);
        float precision = evNumberDTTrueRectangles/static_cast<float>(evNumberDTRectangles);
        float f1 = 2* precision * recall/(precision + recall);
        std::cout << "\nRESULT\n";
        std::cout << "Precision: " << precision << "\n";
        std::cout << "Recall:    " << recall << "\n";
        std::cout << "F1:        " << f1 << "\n";
    }
}
/*
void CabinetParser::evaluateRectanglePruning(const std::vector<std::tuple<cv::Mat,  Segmentation, cv::Mat> >& images)
{
    std::cout << pruningThreshold << "\n";
    int evNumberGTRectangles = 0;
    int evNumberGTFoundRectangles = 0;
    int evNumberDTRectangles = 0;
    int evNumberDTTrueRectangles = 0;
    float evTightness = 0;
    float evErrors = 0;
    
    int counter = 0;
    std::vector<int> lowerBounds;
    std::vector<int> trueNumbers;
    std::vector<int> upperBounds;
    #pragma omp parallel for
    for (size_t i = 0; i < images.size(); i++)
    {
#if VERBOSE_MODE
        //std::cout << "Processing image " << ++counter << " out of " << images.size() << std::endl;
        //std::cout << std::get<1>(images[i]).file << "\n";
        //std::cout.flush();
        #pragma omp critical
        {
            //std::cout << '.';
            //std::cout.flush();
        }
#endif
        cv::Mat rectifiedMultiChannelImage;
        extractRectifiedMultiChannelImage(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImage);
        
        cv::Mat rectifiedMultiChannelImageDepth;
        extractRectifiedMultiChannelImage(std::get<2>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImageDepth);
        // Detect edges
        cv::Mat detected;
        cv::Mat detectedDepth;

        // Apply the edge detector
        applyEdgeDetector(rectifiedMultiChannelImage, detected, 0);
        // Apply the edge detector on depth
#if INCLUDE_DEPTH
        applyEdgeDetector(rectifiedMultiChannelImageDepth, detectedDepth, 1);
#endif

        cv::bitwise_or(detected, detectedDepth, detected);

        // Get the canny edge image
        std::vector<cv::Mat> channels;
        cv::split(rectifiedMultiChannelImage, channels);
        cv::Mat cannyEdges;
        Processing::computeCannyEdges(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], cannyEdges);
        
        // Detect rectangles
        std::vector<Rectangle> partHypotheses;
        detectRectangles(detected, cannyEdges, partHypotheses);

        #pragma omp critical
        {
            std::cout << std::get<1>(images[i]).file << ": " << partHypotheses.size() << " rectangles detected" << std::endl;
        }
        // Rectify the GT rectangles
        std::vector<Rectangle> rectified;
        rectifyParts(std::get<1>(images[i]).regionOfInterest, std::get<1>(images[i]).parts, rectified);
        
        // Check if all parts have been found
        bool allPartsFound = true;
        for (size_t r = 0; r < rectified.size(); r++)
        {
            float maxIOU = 0;
            for (size_t p = 0; p < partHypotheses.size(); p++)
            {
                maxIOU = std::max(maxIOU, RectangleUtil::calcIOU(partHypotheses[p], rectified[r]));
            }
            if (maxIOU < rectangleAcceptanceThreshold)
            {
                allPartsFound = false;
                break;
            }
        }
        
        std::vector<Rectangle> selection;
        int packingNumber, coverNumber;
        pruneRectangles((rectifiedMultiChannelImage.rows+1)*(rectifiedMultiChannelImage.cols+1), partHypotheses, packingNumber, coverNumber, selection);

        #pragma omp critical
        {
            std::cout << std::get<1>(images[i]).file << ": " << selection.size() << " after pruning" << std::endl;
        }

        #pragma omp critical
        {
            if (allPartsFound)
            {
                lowerBounds.push_back(coverNumber);
                upperBounds.push_back(packingNumber);
                trueNumbers.push_back(static_cast<int>(rectified.size()));
            }

            if (allPartsFound)
            {
                if (coverNumber > static_cast<int>(rectified.size()) || packingNumber < static_cast<int>(rectified.size()))
                {
                    evErrors += 1;
                }
                else
                {
                    evTightness += packingNumber - coverNumber;
                }
            }
        }
        
        // Evaluate
        for (size_t r = 0; r < rectified.size(); r++)
        {
            float maxIOU = 0;
            for (size_t p = 0; p < selection.size(); p++)
            {
                maxIOU = std::max(maxIOU, RectangleUtil::calcIOU(rectified[r], selection[p]));
            }
            
            #pragma omp critical
            {
                evNumberGTRectangles++;
            }
            
            if (maxIOU >= rectangleAcceptanceThreshold)
            {
                #pragma omp critical
                {
                    evNumberGTFoundRectangles++;
                }
            }
            else
            {
#if 0
                cv::Mat demo;
                std::get<0>(images[i]).copyTo(demo);
                PlotUtil::plotRectangle(demo, std::get<1>(images[i]).parts[r], cv::Scalar(0,0,255), 2);
                cv::imshow("test", demo);
                cv::waitKey();
#endif
            }
        }
        
        for (size_t p = 0; p < selection.size(); p++)
        {
            #pragma omp critical
            {
                evNumberDTRectangles++;
            }
            float maxIOU = 0;
            for (size_t r = 0; r < rectified.size(); r++)
            {
                maxIOU = std::max(maxIOU, RectangleUtil::calcIOU(rectified[r], selection[p]));
            }
            
            if (maxIOU >= rectangleAcceptanceThreshold)
            {
                #pragma omp critical
                {
                    evNumberDTTrueRectangles++;
                }
            }
        }
    }
    float recall = evNumberGTFoundRectangles/static_cast<float>(evNumberGTRectangles);
    float precision = evNumberDTTrueRectangles/static_cast<float>(evNumberDTRectangles);
    float f1 = 2* precision * recall/(precision + recall);
    evTightness /= (static_cast<int>(lowerBounds.size()));
    evErrors /= static_cast<int>(lowerBounds.size());
    std::cout << "\n\nRESULT\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall:    " << recall << "\n";
    std::cout << "Error:     " << evErrors << "\n";
    std::cout << "Tightness: " << evTightness << "\n";
    std::cout << "F1:        " << f1 << "\n";
    std::cout << "BOUNDS\n";
    for(size_t i = 0; i < lowerBounds.size(); i++)
    {
        std::cout << lowerBounds[i] << ',' << trueNumbers[i] << ',' << upperBounds[i] << "\n";
    }
}
*/

/*
* Extract the shape data of each rectangle
*/
void CabinetParser::extractRectangleData(libf::DataStorage::ptr dataStorage, const std::vector<std::tuple<cv::Mat,  Segmentation, cv::Mat> >& images)
{
    int counter = 0;
    //#pragma omp parallel for
    for (size_t i = 0; i < images.size(); i++)
    {
        //#pragma omp critical
        {
#if VERBOSE_MODE
        std::cout << "Processing image " << ++counter << " out of " << images.size() << std::endl;
#endif
        }
        
        // Get the rectified canny edge image
        cv::Mat rectifiedMultiChannelImage;
        extractRectifiedMultiChannelImage(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImage);
        
        // Get the rectified canny depth edge image
        cv::Mat rectifiedMultiChannelImageDepth;
        extractRectifiedMultiChannelImage(std::get<2>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImageDepth);
        std::vector<cv::Mat> depthChannels;
        cv::split(rectifiedMultiChannelImageDepth, depthChannels);
        cv::Mat depthIntensity;
        depthChannels[EDGE_DETECTOR_CHANNEL_INTENSITY].convertTo(depthIntensity,CV_8UC1);
        cv::Mat rectifiedDepth;
        //Invert
        depthIntensity = cv::Scalar::all(255) - depthIntensity;
        //Rectify the depth
        rectifyDepthBilinear(depthIntensity, rectifiedDepth);
        //cv::Mat rectifiedRGB;
        //rectifyDepth(std::get<0>(images[i]), std::get<2>(images[i]), rectifiedRGB);
#if 0
        cv::imshow("RGB image",std::get<0>(images[i]));
        cv::waitKey();

        cv::imshow("Rectified RGB image",rectifiedRGB);
        cv::waitKey();
        exit(0);
	
#endif

#if 0
        cv::imshow("Depth Image",depthIntensity);
        cv::imshow("RGB image",std::get<0>(images[i]));
        //cv::imshow("Depth Image [rectified]",rectifiedDepth);
        cv::waitKey();
#endif
        
        std::vector<Rectangle> rectified;
        rectifyParts(std::get<1>(images[i]).regionOfInterest, std::get<1>(images[i]).parts, rectified);
        
        float meanDepth;
        for (size_t l = 0; l < rectified.size(); l++)
        {
            meanDepth = 2.0f + extractMeanPartDepth(rectifiedDepth, rectified[l]);
            std::cout<<" Mean depth of part "<<l<<" is "<<meanDepth<<std::endl;

            libf::DataPoint p(9);
            p(0) = rectified[l].getWidth()/rectifiedMultiChannelImage.cols;
            p(1) = rectified[l].getHeight()/rectifiedMultiChannelImage.rows;
            //p(2) = rectified[l].getCenter()[0]/rectifiedMultiChannelImage.rows;
            //p(3) = rectified[l].getCenter()[1]/rectifiedMultiChannelImage.cols;
            p(2) = rectified[l].getWidth();
            p(3) = rectified[l].getHeight();
            p(4) = rectified[l].getWidth()/rectified[l].getHeight();
            p(5) = meanDepth;//mean depth
            p(6) = meanDepth/rectified[l].getWidth();//depth width aspect ratio
            p(7) = meanDepth/rectified[l].getHeight();//depth height aspect ratio            
            p(8) = 4*rectified[l].getWidth()*rectified[l].getHeight()/pow((rectified[l].getWidth() + rectified[l].getHeight()),2);//form factor
            //#pragma omp critical 
            {
                dataStorage->addDataPoint(p, std::get<1>(images[i]).labels[l]);
            }
        }
    }
}


void CabinetParser::rectifyDepth(const cv::Mat & unRectifiedRGB, const cv::Mat & unRectifiedDepth, cv::Mat & rectifiedRGB)
{
	rectifiedRGB = cv::Mat::zeros(unRectifiedRGB.rows, unRectifiedRGB.cols, CV_32FC1);
	

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> ("cloud_1_xtion.pcd", *cloud) == -1) //* load the file
  	{
        PCL_ERROR ("Couldn't read file cloud_0.pcd \n");
    }

	std::cout << "Loaded the cloud with height "<<cloud->height<<" and width "<<cloud->width<<" \n\n";
	
	std::vector<int> mapping;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);

	// Filter object.
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
 	pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> filter;
	filter.setInputCloud(cloud);
	// Every point must have 10 neighbors within 15cm, or it will be removed.
	filter.setRadiusSearch(0.15);
	filter.setMinNeighborsInRadius(10); 
	filter.filter(*filteredCloud); 
	
    //---------------------------------------
    // Computing normals
    //---------------------------------------
	std::cerr << "Computing normals for: " << filteredCloud->width * filteredCloud->height << " data points " << std::endl;	
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  	//ne.setInputCloud (cloud);
  	ne.setInputCloud (filteredCloud);	
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr cloudTree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  	ne.setSearchMethod (cloudTree);
  	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  	ne.setRadiusSearch (0.1);// default 0.05 or 0.1
  	ne.compute (*normals);
  	std::cout << "Normals computed.\n\n";

	//---------------------------------------
	// Clustering the normals
  	//---------------------------------------
	
	std::cout << "Clustering by kmeans.\n\n";
	// feature formation
	cv::Mat normalsFeatVec = cv::Mat::zeros(normals->points.size()/2, 3, CV_32F);
	for (size_t currentPoint = 0; currentPoint < normals->points.size() ; currentPoint+=2)
	{
		float a = normals->points[currentPoint].normal[0];		
		float b = normals->points[currentPoint].normal[1];		
		float c = normals->points[currentPoint].normal[2];

		float r = sqrt(a*a + b*b + c*c);
		
		normalsFeatVec.at<float>(currentPoint/2,0) = acos(a/r);
		normalsFeatVec.at<float>(currentPoint/2,1) = acos(b/r);
		normalsFeatVec.at<float>(currentPoint/2,2) = acos(c/r);
	
	}
	
	int clusterCount = 8;
	std::cout <<normalsFeatVec.rows<<" normals each with "<<normalsFeatVec.cols<<" features to be clustered into "<<clusterCount<<"  groups\n\n";
	int attempts = 3;	
  	cv::Mat labels;  
  	cv::Mat centers;

    //std::cout <<normalsFeatVec;
    cv::kmeans(normalsFeatVec, clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01), attempts, cv::KMEANS_PP_CENTERS, centers);

    std::cout << "Clustering Done. "<<std::endl<<centers.rows<<" "<<centers.cols<<" \n\n";
	std::cout << "Kmeans converged-"<<std::endl;
	std::cout <<normalsFeatVec.rows<<" normals each with "<<normalsFeatVec.cols<<" features were clustered into "<<clusterCount<<" == "<<centers.rows<<" groups\n\n";

    std::cout <<centers<<std::endl;

	int count0 = 0, count1 = 0, count2 = 0, count3 = 0, count4 = 0, count5 = 0, count6 = 0, count7 = 0;

	for (int i=0; i<labels.rows; i++)
	{
		switch (labels.at<int>(i,0))
                        {
                            case 0:
                                count0 += 1;
                                break;
                            case 1:
                                count1 += 1;
                                break;
                            case 2:
                                count2 += 1;
                                break;
			    case 3:
                                count3 += 1;
                                break;
                            case 4:
                                count4 += 1;
                                break;
                            case 5:
                                count5 += 1;
                                break;
			    case 6:
                                count6 += 1;
                                break;
                            case 7:
                                count7 += 1;
                                break;
                         }
	}

#if 0		
	std::cout<<"cluster 0 has "<<count0<<" elements"<<std::endl;
	std::cout<<"cluster 1 has "<<count1<<" elements"<<std::endl;
	std::cout<<"cluster 2 has "<<count2<<" elements"<<std::endl;
	std::cout<<"cluster 3 has "<<count3<<" elements"<<std::endl;
	std::cout<<"cluster 4 has "<<count4<<" elements"<<std::endl;
	std::cout<<"cluster 5 has "<<count5<<" elements"<<std::endl;
	std::cout<<"cluster 6 has "<<count6<<" elements"<<std::endl;
	std::cout<<"cluster 7 has "<<count7<<" elements"<<std::endl;
#endif


	std::vector<int> test = {count0, count1, count2, count3, count4, count5, count6, count7};
	std::priority_queue<std::pair<int, int>> q;
  	for (int i = 0; i < test.size(); ++i)
	{
    		q.push(std::pair<int, int>(test[i], i));
  	}
  	

    //---------------------------------------
    // Compute 3 dominant directions
    //---------------------------------------

    int k = 3; // find top 3
	Mat33f main3normals;

  	for (int i = 0; i < k; ++i)
	{
    		int ki = q.top().second;
    		std::cout << "index[" << i << "] = " << ki << std::endl;
            main3normals.col[i] = Vec3f(cos(centers.at<float>(ki,0)), cos(centers.at<float>(ki,1)), cos(centers.at<float>(ki,2)));
    		q.pop();
  	}

    //---------------------------------------
    // Gram Schmidt Orthonormalization
    //---------------------------------------

	Eigen::Matrix3f transform_MGS;
    //---------------------------------------
    // Computing coordinate system
    //---------------------------------------

    computeCordSys(transform_MGS, main3normals);
    //---------------------------------------
    // Cancel the transformation
    //---------------------------------------

	transformPCL(transform_MGS);
	
}

void CabinetParser::computeCordSys(Eigen::Matrix3f &transform_MGS, Mat33f &main3normals)
{
	//---------------------------------------
	// Gram Schmidt Orthonormalization
  	//---------------------------------------	
	 
    Mat33f MGS;
    modified_gram_schmidt(MGS, main3normals);
    print_mat("MGS", MGS);

	transform_MGS (0,0) = MGS.col[0].v[0];
	transform_MGS (1,0) = MGS.col[0].v[1];
	transform_MGS (2,0) = MGS.col[0].v[2];
	transform_MGS (0,1) = MGS.col[1].v[0];
	transform_MGS (1,1) = MGS.col[1].v[1];
	transform_MGS (2,1) = MGS.col[1].v[2];
	transform_MGS (0,2) = MGS.col[2].v[0];
	transform_MGS (1,2) = MGS.col[2].v[1];
	transform_MGS (2,2) = MGS.col[2].v[2];

	std::cout<<"the transform matrix"<<std::endl<<transform_MGS<<std::endl;
}

//---------------------------------------
// Gram Schmidt orthonormalization
//---------------------------------------

void CabinetParser::modified_gram_schmidt(Mat33f &out, const Mat33f &in)
{
    //normalize
    out.col[0] = normalize(in.col[0]);
    out.col[1] = normalize(in.col[1] - dot(in.col[1], out.col[0])*out.col[0]);

    out.col[2] = in.col[2] - dot(in.col[2], out.col[0])*out.col[0];
    // note the second dot product is computed from the partial result!
    out.col[2] -= dot(out.col[2], out.col[1])*out.col[1];
    out.col[2] = normalize(out.col[2]);
}

//---------------------------------------
// Cancel the transformation
//---------------------------------------

void CabinetParser::transformPCL(Eigen::Matrix3f &transform_GM)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PCDReader readerP;
  readerP.read ("cloud_1_xtion.pcd", *input_cloud);
  std::cout << "Loaded "
            << input_cloud->width * input_cloud->height
            << " data points from cloud_1_xtion.pcd with the following fields: "
            << std::endl;

  //---------------------------------------
  // Rotation Matrix Formation
  //---------------------------------------

  float thetaX = 0*M_PI; // The angle of rotation in radians
  float thetaY = -1*M_PI/2; // The angle of rotation in radians
  float thetaZ = 1*M_PI; // The angle of rotation in radians
  
  Eigen::Matrix3f transform_RY = Eigen::Matrix3f::Identity();

  transform_RY (0,0) = cos (thetaY);
  transform_RY (0,2) = -sin(thetaY);
  transform_RY (2,0) = sin (thetaY);
  transform_RY (2,2) = cos (thetaY);

  Eigen::Matrix3f transform_RZ = Eigen::Matrix3f::Identity();

  transform_RZ (0,0) = cos (thetaZ);
  transform_RZ (0,1) = -sin(thetaZ);
  transform_RZ (1,0) = sin (thetaZ);
  transform_RZ (1,1) = cos (thetaZ);

  Eigen::Matrix3f transform_RX = Eigen::Matrix3f::Identity();

  transform_RX (1,1) = cos (thetaX);
  transform_RX (1,2) = -sin(thetaX);
  transform_RX (2,1) = sin (thetaX);
  transform_RX (2,2) = cos (thetaX);
  
  Eigen::Matrix3f transform_R = transform_RZ*transform_RY*transform_RX*transform_GM.transpose();
  
  //---------------------------------------
  // Transform matrix
  //---------------------------------------

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  
  //Rotation
  transform (0,0) = transform_R (0,0);
  transform (0,1) = transform_R (0,1);
  transform (0,2) = transform_R (0,2);
  transform (1,0) = transform_R (1,0);
  transform (1,1) = transform_R (1,1);
  transform (1,2) = transform_R (1,2);
  transform (2,0) = transform_R (2,0);
  transform (2,1) = transform_R (2,1);
  transform (2,2) = transform_R (2,2);

  //Translation
  transform (0,3) = -1.0;//t_x
  transform (1,3) = -0.5;//t_y
  transform (2,3) = 0.5;//t_z
  
  //---------------------------------------
  // Transform the cloud
  //---------------------------------------

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::transformPointCloud(*input_cloud, *transformed_cloud, transform);
  pcl::PCDWriter writerT;
  writerT.write ("cloud_1_xtion_transformed.pcd", *transformed_cloud);  
  std::cout << "Transformation complete "<< std::endl;
  
}


//---------------------------------------
// Bilinear Interpolation of Depth values of all pixels from the 4 corner points
//---------------------------------------

void CabinetParser::rectifyDepthBilinear(const cv::Mat & unRectifiedImage, cv::Mat & rectifiedImage)
{
    
    float dTL = unRectifiedImage.at<uchar>(0 + JITTERFACTOR, 0 + JITTERFACTOR);
    float dTR = unRectifiedImage.at<uchar>(0 + JITTERFACTOR, unRectifiedImage.cols-1 - JITTERFACTOR);
    float dBR = unRectifiedImage.at<uchar>(unRectifiedImage.rows-1 - JITTERFACTOR, unRectifiedImage.cols-1 - JITTERFACTOR);
    float dBL = unRectifiedImage.at<uchar>(unRectifiedImage.rows-1 - JITTERFACTOR, 0  + JITTERFACTOR);

    int dWidth = unRectifiedImage.cols - 2*JITTERFACTOR;
    int dHeight = unRectifiedImage.rows - 2*JITTERFACTOR;

    rectifiedImage = cv::Mat::zeros(unRectifiedImage.rows, unRectifiedImage.cols, CV_8UC1);

    float eachPixelDepthFactor,eachPixelDepth;

    cv::Mat correctionImage = cv::Mat::zeros(unRectifiedImage.rows, unRectifiedImage.cols, CV_8UC1);

    for(int h = JITTERFACTOR; h < (unRectifiedImage.rows - JITTERFACTOR); h++)
    {
        for(int w = JITTERFACTOR; w < (unRectifiedImage.cols - JITTERFACTOR); w++)
        {
            eachPixelDepthFactor = (dWidth - w)*(dHeight - h)*dTL + w*(dHeight-h)*dTR + w*h*dBR + (dWidth-w)*h*dBL;//bilinear interpolation of depth
            eachPixelDepth = (unRectifiedImage.at<uchar>(h, w) - (eachPixelDepthFactor/dHeight/dWidth));
            if(eachPixelDepth >= 0.0f)
                rectifiedImage.at<uchar>(h, w) = eachPixelDepth;
            else
                rectifiedImage.at<uchar>(h, w) = 0;

            //correctionImage.at<uchar>(h,w) = (eachPixelDepthFactor/dHeight/dWidth);
        }

    }

    //cv::imshow("Depth Correction Image",correctionImage);
    //cv::waitKey();
}


//---------------------------------------
// Extract mean depth of each part (rectangle)
//---------------------------------------

float CabinetParser::extractMeanPartDepth(const cv::Mat & rectifiedDepth, const Rectangle rectifiedRect)
{
    int minHeight = rectifiedRect.getCenter()[1] - rectifiedRect.getHeight()/2;
    int maxHeight = rectifiedRect.getCenter()[1] + rectifiedRect.getHeight()/2;
    int minWidth  = rectifiedRect.getCenter()[0] - rectifiedRect.getWidth()/2;
    int maxWidth  = rectifiedRect.getCenter()[0] + rectifiedRect.getWidth()/2;

    cv::Mat rectifiedPartDepth = cv::Mat::zeros((maxHeight-minHeight), (maxWidth-minWidth), CV_8UC1);

#if 0
    std::cout<<"minHeight = "<<minHeight<<" maxHeight = "<<maxHeight<<" minWidth = "<<minWidth<<" maxWidth = "<<maxWidth<<std::endl;
#endif

    float sumDepth = 0.0f, meanDepth;
    //float maxDepth = 0.0f;
    cv::Mat rectifiedDepthFloat;
    rectifiedDepth.convertTo(rectifiedDepthFloat,CV_32FC1);

    for(int h = minHeight; h < maxHeight; h++)
    {
        for(int w = minWidth; w < maxWidth; w++)
        {
            rectifiedPartDepth.at<uchar>(h-minHeight,w-minWidth) = rectifiedDepthFloat.at<float>(h,w);
            //maxDepth  = std::max(maxDepth,rectifiedDepthFloat.at<float>(h,w));
            sumDepth += rectifiedDepthFloat.at<float>(h,w);
#if 0
            rectifiedPartDepth.at<uchar>(h-minHeight,w-minWidth) = rectifiedMultiChannelImageDepth.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_INTENSITY];
            maxDepth  = std::max(maxDepth,rectifiedMultiChannelImageDepth.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_INTENSITY]);
            sumDepth += rectifiedMultiChannelImageDepth.at<EdgeDetectorVec>(h,w)[EDGE_DETECTOR_CHANNEL_INTENSITY];
#endif
        }

    }

    //sumDepth -= maxDepth*(maxHeight-minHeight)*(maxWidth-minWidth);
    meanDepth = sumDepth/(maxHeight-minHeight);
    meanDepth /= (maxWidth-minWidth);

#if 0
        std::cout<<"meanDepth = "<<std::abs(meanDepth)<<std::endl;

        cv::Mat demo;
        rectifiedPartDepth.copyTo(demo);
        cv::imshow("test", demo);
        cv::waitKey();
#endif

    return meanDepth;

}



void CabinetParser::evaluateSegmentation(const std::vector<std::tuple<cv::Mat, Segmentation, cv::Mat> >& images)
{
    // Output the results into a file
    
    // The total accuracy of detected rectangles
    float precision = 0;
    int precisionN = 0;
    float recall = 0;
    int recallN = 0;
    // The total label accuracy
    float labelAccuracy = 0;
    int labelAccuracyN = 0;
    // The segmentation accuracy per class
    float accuracyDoor = 0;
    int accuracyDoorN = 0;
    float accuracyDrawer = 0;
    int accuracyDrawerN = 0;
    float accuracyShelf = 0;
    int accuracyShelfN = 0;
    // The label accuracy per class
    float labelAccuracyDoor = 0;
    int labelAccuracyDoorN = 0;
    float labelAccuracyDrawer = 0;
    int labelAccuracyDrawerN = 0;
    float labelAccuracyShelf = 0;
    int labelAccuracyShelfN = 0;
    
    Eigen::MatrixXf confusionMatrix = Eigen::MatrixXf::Zero(3,3);
    
    // Go over all images and compute the segmentation
    //#pragma omp parallel for
    for (size_t i = 0; i < images.size(); i++)
    {
        #pragma omp critical
        {
            std::cout << std::get<1>(images[i]).file << "\n";
        }
        // Rectify the parts in order to compute matches
        std::vector<Rectangle> rectifiedParts;
        Rectangle modifiedROI = std::get<1>(images[i]).regionOfInterest;

    #if ROI_CORRUPTION
        std::uniform_int_distribution<int> corruptDistTLX(static_cast<int>(-1*CORRUPT_PIXELS), static_cast<int>(1*CORRUPT_PIXELS));
        std::uniform_int_distribution<int> corruptDistTLY(static_cast<int>(-1*CORRUPT_PIXELS), static_cast<int>(1*CORRUPT_PIXELS));
        std::uniform_int_distribution<int> corruptDistTRX(static_cast<int>(-1*CORRUPT_PIXELS), static_cast<int>(1*CORRUPT_PIXELS));
        std::uniform_int_distribution<int> corruptDistTRY(static_cast<int>(-1*CORRUPT_PIXELS), static_cast<int>(1*CORRUPT_PIXELS));
        std::uniform_int_distribution<int> corruptDistBLX(static_cast<int>(-1*CORRUPT_PIXELS), static_cast<int>(1*CORRUPT_PIXELS));
        std::uniform_int_distribution<int> corruptDistBLY(static_cast<int>(-1*CORRUPT_PIXELS), static_cast<int>(1*CORRUPT_PIXELS));
        std::uniform_int_distribution<int> corruptDistBRX(static_cast<int>(-1*CORRUPT_PIXELS), static_cast<int>(1*CORRUPT_PIXELS));
        std::uniform_int_distribution<int> corruptDistBRY(static_cast<int>(-1*CORRUPT_PIXELS), static_cast<int>(1*CORRUPT_PIXELS));

        std::cout<<std::endl<<static_cast<int>(corruptDistTLX(g))<<" "
                   <<static_cast<int>(corruptDistTLY(g))<<" "
                     <<static_cast<int>(corruptDistTRX(g))<<" "
                       <<static_cast<int>(corruptDistTRY(g))<<" "
                         <<static_cast<int>(corruptDistBLX(g))<<" "
                           <<static_cast<int>(corruptDistBLY(g))<<" "
                             <<static_cast<int>(corruptDistBRX(g))<<" "
                               <<static_cast<int>(corruptDistBRY(g))<<" "<<std::endl;

        modifiedROI[0][0] += static_cast<int>(corruptDistTLX(g));
        modifiedROI[0][1] += static_cast<int>(corruptDistTLY(g));
        modifiedROI[1][0] += static_cast<int>(corruptDistTRX(g));
        modifiedROI[1][1] += static_cast<int>(corruptDistTRY(g));
        modifiedROI[2][0] += static_cast<int>(corruptDistBLX(g));
        modifiedROI[2][1] += static_cast<int>(corruptDistBLY(g));
        modifiedROI[3][0] += static_cast<int>(corruptDistBRX(g));
        modifiedROI[3][1] += static_cast<int>(corruptDistBRY(g));
    #endif

        rectifyParts(modifiedROI, std::get<1>(images[i]).parts, rectifiedParts);
        
        // Segment the image
        std::vector<Part> segmentation;
        parse(std::get<0>(images[i]), std::get<2>(images[i]), modifiedROI, segmentation);
        
        // Save an image
        cv::Mat visualization = cv::Mat::zeros(std::get<0>(images[i]).rows, std::get<0>(images[i]).cols, CV_8UC3);
        visualizeSegmentation(std::get<0>(images[i]), modifiedROI, segmentation, visualization);
        std::stringstream ss, ss2;
        ss << "results/" << std::get<1>(images[i]).file << ".png";
        ss2 << "results/" << std::get<1>(images[i]).file << ".txt";
        std::cout << ss.str() << "\n";
        cv::imwrite(ss.str(), visualization);
        
        std::ofstream res(ss2.str());
        for (size_t p = 0; p < segmentation.size(); p++)
        {
            for (int l = 0; l < 4; l++)
            {
                res << segmentation[p].rect[l] << ' ';
            }
            res << "\nlabel: " << segmentation[p].label << "\n";
	    res << "meanDepth: " << segmentation[p].meanDepth << "\n";
	    res << "shapePrior: " << segmentation[p].shapePrior << "\n\n";
        res << "likelihood: " << segmentation[p].likelihood << "\n\n";
        res << "posterior: " << segmentation[p].posterior << "\n\n";
        }
        res.close();
        
        #pragma omp critical
        {
            // Compute the recall
            for (size_t r = 0; r < rectifiedParts.size(); r++)
            {
                recallN++;
                labelAccuracyN++;
                // Does the label match?
                switch (std::get<1>(images[i]).labels[r])
                {
                    case 0:
                        accuracyDoorN++;
                        break;
                    case 1:
                        accuracyDrawerN++;
                        break;
                    case 2:
                        accuracyShelfN++;
                        break;
                }

                float maxIOU = 0;
                size_t maxS = 0;
                bool found = false;
                for (size_t s = 0; s < segmentation.size(); s++)
                {
                    const float iou = RectangleUtil::calcIOU(segmentation[s].rect, rectifiedParts[r]);
                    if (iou > maxIOU)
                    {
                        maxIOU = iou;
                        maxS = s;
                    }
                }

                std::cout << maxIOU << "\n";

                found = maxIOU >= rectangleAcceptanceThreshold;

                // Do we have a match?
                if (found)
                {
                    // Yes, we have
                    recall += 1;
                    switch (std::get<1>(images[i]).labels[r])
                    {
                        case 0:
                            labelAccuracyDoorN++;
                            accuracyDoor += 1;
                            break;
                        case 1:
                            labelAccuracyDrawerN++;
                            accuracyDrawer += 1;
                            break;
                        case 2:
                            labelAccuracyShelfN++;
                            accuracyShelf += 1;
                            break;
                    }

                    confusionMatrix(std::get<1>(images[i]).labels[r], segmentation[maxS].label) += 1;
                    if (segmentation[maxS].label == std::get<1>(images[i]).labels[r])
                    {
                        labelAccuracy += 1;
                        switch (std::get<1>(images[i]).labels[r])
                        {
                            case 0:
                                labelAccuracyDoor += 1;
                                break;
                            case 1:
                                labelAccuracyDrawer += 1;
                                break;
                            case 2:
                                labelAccuracyShelf += 1;
                                break;
                        }
                    }
                }
            }
        }
        
        // Compute the precision
        for (size_t s = 0; s < segmentation.size(); s++)
        {
            precisionN++;
            
            float maxIOU = 0;
            for (size_t r = 0; r < rectifiedParts.size(); r++)
            {
                const float iou = RectangleUtil::calcIOU(segmentation[s].rect, rectifiedParts[r]);
                if (iou > maxIOU)
                {
                    maxIOU = iou;
                }
            }
            
            if (maxIOU >= rectangleAcceptanceThreshold)
            {
                precision += 1;
            }
        }
    }
    
    std::ofstream results("results.txt");
    results << "STRUCTURAL INFERENCE PERFORMANCE\n";
    results << std::setw(25) << "precision: " << std::setw(10) << (precision/precisionN) << "\n";
    results << std::setw(25) << "recall: " << std::setw(10) << (recall/recallN) << "\n";
    results << std::setw(25) << "F1: " << std::setw(10) << (2*recall*precision/recallN/precisionN/(precision/precisionN + recall/recallN)) << "\n";
    results << "\n";
    results << "PERFORMANCE PER CLASS\n";
    results << std::setw(25) << "accuracy (door): " << std::setw(10) << (accuracyDoor/accuracyDoorN) << "\n";
    results << std::setw(25) << "accuracy (drawer): " << std::setw(10) << (accuracyDrawer/accuracyDrawerN) << "\n";
    results << std::setw(25) << "accuracy (shelf): " << std::setw(10) << (accuracyShelf/accuracyShelfN) << "\n";
    results << "\n";
    results << "LABEL PERFORMANCE\n";
    results << std::setw(25) << "accuracy: " << std::setw(10) << (labelAccuracy/labelAccuracyN) << "\n";
    results << "\n";
    results << "PERFORMANCE PER CLASS\n";
    results << std::setw(25) << "accuracy (door): " << std::setw(10) << (labelAccuracyDoor/labelAccuracyDoorN) << "\n";
    results << std::setw(25) << "accuracy (drawer): " << std::setw(10) << (labelAccuracyDrawer/labelAccuracyDrawerN) << "\n";
    results << std::setw(25) << "accuracy (shelf): " << std::setw(10) << (labelAccuracyShelf/labelAccuracyShelfN) << "\n";
    results << "\n";
    results << "CONFUSION MATRIX\n";
    results << confusionMatrix << "\n";
    results.close();
}

void CabinetParser::exportEdgeDistributionFromGT(const cv::Mat& image, const Segmentation& segmentation)
{
    // We have to rectify the image
    // Get the rectified canny edge image
    cv::Mat rectifiedMultiChannelImage;
    extractRectifiedMultiChannelImage(image, segmentation.regionOfInterest, rectifiedMultiChannelImage);

    // Get the canny edge image
    std::vector<cv::Mat> channels;
    cv::split(rectifiedMultiChannelImage, channels);
    cv::Mat intensityImage;
    channels[EDGE_DETECTOR_CHANNEL_INTENSITY].convertTo(intensityImage, CV_8UC1);
    cv::Mat cannyEdgeImage;
    cv::GaussianBlur(intensityImage,intensityImage, cv::Size(5,5), 1.5);
    Processing::computeCannyEdges(intensityImage, cannyEdgeImage);
    Processing::add1pxBorders(cannyEdgeImage);

    // Rectify the parts
    std::vector<Rectangle> rectified;
    rectifyParts(segmentation.regionOfInterest, segmentation.parts, rectified);

    // We highlight the rectangle in the image and output the corresponding
    // distribution vector
    for (size_t p = 0; p < rectified.size(); p++)
    {
        std::cout << p << "\n";
        const Rectangle & r = rectified[p];
        libf::DataPoint p1;
        extractDiscretizedAppearanceData(cannyEdgeImage, r, p1);
        
        // Visualize the result
        std::vector<Part> partForVisualization(1);
        partForVisualization[0].label = segmentation.labels[p];
        partForVisualization[0].rect = r;
        cv::Mat visualization;
        visualizeSegmentation(image, segmentation.regionOfInterest, partForVisualization, visualization);
        
        // Save the visualization
        std::stringstream ss;
        ss << "visualizations/" << p << ".png";
        cv::imwrite(ss.str(), visualization);
        
        // Save the distribution
        std::stringstream ss2;
        ss2 << "visualizations/" << p << "_data.txt";
        std::ofstream out(ss2.str());
        out << p1 << "\n";
        out.close();
    }
}

void CabinetParser::exportRectifiedImages(const std::string& directory)
{
    std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat> > trainingData;
    
#if VERBOSE_MODE
    std::cout << "Load training data from " << directory << "\n";
#endif
    loadImage(directory, trainingData);
    
#if VERBOSE_MODE
    std::cout << "Done loading training data\n";
    std::cout << trainingData.size() << " images loaded\n\n";
#endif
    
    exportRectifiedImages(trainingData);
}

void CabinetParser::exportRectifiedImages(const std::vector<std::tuple<cv::Mat, Segmentation, cv::Mat> >& images)
{
    int counter = 0;
    
    int numElements = 0;
    for (size_t i = 0; i < images.size(); i++)
    {
        numElements += static_cast<int>(std::get<1>(images[i]).parts.size());
    }
    
    cv::Mat patches = cv::Mat::zeros(0, 100*100, CV_32F);
    cv::Mat labels(numElements, 1, CV_32S);
    
    int globCounter = 0;
    
    #pragma omp parallel for
    for (size_t i = 0; i < images.size(); i++)
    {
        #pragma omp critical
        {
#if VERBOSE_MODE
        std::cout << "Processing image " << ++counter << " out of " << images.size() << std::endl;
#endif
        }
        
        // Get the rectified canny image
        cv::Mat rectifiedMultiChannelImage;
        extractRectifiedMultiChannelImage(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImage);
        std::vector<cv::Mat> channels;
        cv::split(rectifiedMultiChannelImage, channels);
        
        // Compute the floating point gradient magnitude image
        cv::Mat gradMag;
        Processing::computeGradientMagnitudeImageFloat(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], gradMag);
        
        // Rectify the parts
        std::vector<Rectangle> rectified;
        rectifyParts(std::get<1>(images[i]).regionOfInterest, std::get<1>(images[i]).parts, rectified);
        
        Rectangle unitSquare;
        unitSquare[0][0] = 0;
        unitSquare[0][1] = 0;
        unitSquare[1][0] = 99;
        unitSquare[1][1] = 0;
        unitSquare[2][0] = 99;
        unitSquare[2][1] = 99;
        unitSquare[3][0] = 0;
        unitSquare[3][1] = 99;
        
        for (size_t r = 0; r < rectified.size(); r++)
        {
            // Warp the content of the GM image to a standard size
            cv::Mat warped;
            Rectangle _r = rectified[r];
            if (_r.getHeight() < 10 || _r.getWidth() < 10)
            {
                Processing::warpImageGaussian(gradMag, rectified[r], warped, unitSquare, 2);
            }
            else
            {
                int offset = 3;
                _r[0][0] += offset;
                _r[0][1] += offset;
                _r[1][0] -= offset;
                _r[1][1] += offset;
                _r[2][0] -= offset;
                _r[2][1] -= offset;
                _r[3][0] += offset;
                _r[3][1] -= offset;
                Processing::warpImageGaussian(gradMag, _r, warped, unitSquare, 2);
            }
            
            Processing::normalizeFloatImageLebesgue(warped);
            #pragma omp critical
            {
                patches.push_back(warped.reshape(0,1).row(0));
                labels.at<int>(globCounter, 0) = std::get<1>(images[i]).labels[r];
                globCounter++;
            }
            
#if 0
            // Save the result depending on the label
            std::stringstream ss;
            ss << "export2/";
            switch (std::get<1>(images[i]).labels[r])
            {
                case 0:
                    ss << "door_" << (doorCounter++) << ".txt";
                    break;
                case 1:
                    ss << "drawer_" << (drawerCounter++) << ".txt";
                    break;
                case 2:
                    ss << "shelf_" << (shelfCounter++) << ".txt";
                    break;
            }
            std::ofstream out(ss.str());
            out << warped << "\n";
            out.close();
#endif
        }
    }
    std::ofstream out1("patches.m");
    out1 << "M = " << patches.t() << ";\n";
    out1 << "L = " << labels << ";\n";
    out1.close();
}

void CabinetParser::visualizeFloatImage(const cv::Mat& img) const
{
    // Determine the highest and the lowest value
    float min = 1e10;
    float max = -1e10;
    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++)
        {
            min = std::min(min, img.at<float>(y,x));
            max = std::max(max, img.at<float>(y,x));
        }
    }
    std::cout << "min = " << min << ", max = " << max << "\n";
    
    // Create the result image
    cv::Mat show(img.rows, img.cols, CV_8UC1);
    
    for (int x = 0; x < img.cols; x++)
    {
        for (int y = 0; y < img.rows; y++)
        {
            show.at<uchar>(y,x) = static_cast<uchar>(std::round( (img.at<float>(y,x) - min)/(max - min) * 255));
        }
    }
    
    cv::imshow("float", show);
    cv::waitKey();
}

void CabinetParser::visualizeAppearanceDistribution(const libf::DataPoint& p, cv::Mat& img)
{
    int size = static_cast<int>(p.rows());
    int isize = static_cast<int>(std::sqrt(size));
    img = cv::Mat::zeros(size, 1, CV_32F);
    for (int i = 0; i < size; i++)
    {
        img.at<float>(i,0) = p(i);
    }
    img = img.reshape(0, isize);
}

Vec2 parseBracket(const std::string & s)
{
    Vec2 res;
    size_t i = 1;
    {
        std::stringstream ss;
        while (s[i] != ',')
        {
            ss << s[i];
            i++;
        }
        res[0] = atof(ss.str().c_str());
    }
    i++;
    {
        std::stringstream ss;
        while (s[i] != ']')
        {
            ss << s[i];
            i++;
        }
        res[1] = atof(ss.str().c_str());
    }
    return res;
}
void CabinetParser::computePrecisionRecallCurve(const std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat > > & images)
{
    // Output the results into a file
    
    // The total accuracy of detected rectangles
    float precision = 0;
    int precisionN = 0;
    float recall = 0;
    int recallN = 0;
    // The total label accuracy
    float labelAccuracy = 0;
    int labelAccuracyN = 0;
    // The segmentation accuracy per class
    float accuracyDoor = 0;
    int accuracyDoorN = 0;
    float accuracyDrawer = 0;
    int accuracyDrawerN = 0;
    float accuracyShelf = 0;
    int accuracyShelfN = 0;
    // The label accuracy per class
    float labelAccuracyDoor = 0;
    int labelAccuracyDoorN = 0;
    float labelAccuracyDrawer = 0;
    int labelAccuracyDrawerN = 0;
    float labelAccuracyShelf = 0;
    int labelAccuracyShelfN = 0;
    
    Eigen::MatrixXf confusionMatrix = Eigen::MatrixXf::Zero(3,3);
    
    std::cout << rectangleAcceptanceThreshold << "\n";
    // Go over all images and compute the segmentation
    //#pragma omp parallel for
    for (size_t i = 0; i < images.size(); i++)
    {
        #pragma omp critical
        {
            //std::cout << std::get<1>(images[i]).file << "\n";
        }
        
        std::stringstream ss2;
        ss2 << "results/" << std::get<1>(images[i]).file << ".txt";
        
        std::vector<Rectangle> rectifiedParts;
        rectifyParts(std::get<1>(images[i]).regionOfInterest, std::get<1>(images[i]).parts, rectifiedParts);
        
        // Load the computed segmentation
        std::ifstream ifs(ss2.str());
        
        // Parse the thing
        std::string word;
        std::vector<Part> segmentation;
        std::vector<Rectangle> rrr;
        int j = 0;
        Part part;
        while(ifs >> word)
        {
            if (j < 4)
            {
                auto res = parseBracket(word);
                part.rect[j][0] = res[0];
                part.rect[j][1] = res[1];
                j++;
            }
            else
            {
                part.rect.normalize();
                part.label = atoi(word.c_str());
                segmentation.push_back(part);
                rrr.push_back(part.rect);
                j = 0;
            }
        } 
        
        std::vector<Rectangle> unrectifiedParts;
        unrectifyParts(std::get<1>(images[i]).regionOfInterest, rrr, unrectifiedParts);
        
        #pragma omp critical
        {
            // Compute the recall
            for (size_t r = 0; r < rectifiedParts.size(); r++)
            {
                recallN++;
                labelAccuracyN++;
                // Does the label match?
                switch (std::get<1>(images[i]).labels[r])
                {
                    case 0:
                        labelAccuracyDoorN++;
                        accuracyDoorN++;
                        break;
                    case 1:
                        labelAccuracyDrawerN++;
                        accuracyDrawerN++;
                        break;
                    case 2:
                        labelAccuracyShelfN++;
                        accuracyShelfN++;
                        break;
                }

                float maxIOU = 0;
                size_t maxS = 0;
                bool found = false;
                for (size_t s = 0; s < segmentation.size(); s++)
                {
                    const float iou = RectangleUtil::calcIOU(segmentation[s].rect, rectifiedParts[r]);
                    if (iou > maxIOU)
                    {
                        maxIOU = iou;
                        maxS = s;
                        found = true;
                    }
                }

                // Do we have a match?
                if (maxIOU > rectangleAcceptanceThreshold)
                {
                    // Yes, we have
                    recall += 1;
                    switch (std::get<1>(images[i]).labels[r])
                    {
                        case 0:
                            accuracyDoor += 1;
                            break;
                        case 1:
                            accuracyDrawer += 1;
                            break;
                        case 2:
                            accuracyShelf += 1;
                            break;
                    }
                }

                if (found)
                {
                    confusionMatrix(std::get<1>(images[i]).labels[r], segmentation[maxS].label) += 1;
                    if (segmentation[maxS].label == std::get<1>(images[i]).labels[r])
                    {
                        labelAccuracy += 1;
                        switch (std::get<1>(images[i]).labels[r])
                        {
                            case 0:
                                labelAccuracyDoor += 1;
                                break;
                            case 1:
                                labelAccuracyDrawer += 1;
                                break;
                            case 2:
                                labelAccuracyShelf += 1;
                                break;
                        }
                    }
                }
            }
            
            // Compute the precision
            for (size_t s = 0; s < segmentation.size(); s++)
            {
                precisionN++;

                float maxIOU = 0;
                for (size_t r = 0; r < rectifiedParts.size(); r++)
                {
                    const float iou = RectangleUtil::calcIOU(segmentation[s].rect, rectifiedParts[r]);
                    if (iou > maxIOU)
                    {
                        maxIOU = iou;
                    }
                }

                if (maxIOU > rectangleAcceptanceThreshold)
                {
                    precision += 1;
                }
                else
                {
#if 0
                    std::cout << maxIOU << "\n";
                    cv::Mat demo;
                    std::get<0>(images[i]).copyTo(demo);
                    PlotUtil::plotRectangle(demo, unrectifiedParts[s], cv::Scalar(0,0,255));
                    cv::imshow("demp", demo);
                    cv::waitKey();
#endif
                }
            }
        }
    }
    std::cout << precisionN - precision << "\n";
    std::cout << "STRUCTURAL INFERENCE PERFORMANCE\n";
    std::cout << std::setw(25) << "precision: " << std::setw(10) << (precision/precisionN) << "\n";
    std::cout << std::setw(25) << "recall: " << std::setw(10) << (recall/recallN) << "\n";
    std::cout << "\n";
    std::cout << "PERFORMANCE PER CLASS\n";
    std::cout << std::setw(25) << "accuracy (door): " << std::setw(10) << (accuracyDoor/accuracyDoorN) << "\n";
    std::cout << std::setw(25) << "accuracy (drawer): " << std::setw(10) << (accuracyDrawer/accuracyDrawerN) << "\n";
    std::cout << std::setw(25) << "accuracy (shelf): " << std::setw(10) << (accuracyShelf/accuracyShelfN) << "\n";
    std::cout << "\n";
}

void saveImageToFile(const std::string & file, const cv::Mat & image)
{
    std::ofstream s(file);
    
    for (int x = 0; x < image.rows; x++)
    {
        for (int y = 0; y < image.cols; y++)
        {
            s << image.at<float>(x,y) << ",";
        }
    }
    s << image.rows << "," << image.cols;
    
    s.close();
}
/*
void CabinetParser::extractPartAppearances(const std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat > > & images)
{
    int counter = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < images.size(); i++)
    {
        #pragma omp critical
        {
#if VERBOSE_MODE
        std::cout << "Processing image " << ++counter << " out of " << images.size() << std::endl;
#endif
        }
        // Get the rectified canny edge image
        cv::Mat rectifiedMultiChannelImage;
        extractRectifiedMultiChannelImage(std::get<0>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImage);

        // Get the rectified canny edge image
        cv::Mat rectifiedMultiChannelImageDepth;
        extractRectifiedMultiChannelImage(std::get<2>(images[i]), std::get<1>(images[i]).regionOfInterest, rectifiedMultiChannelImageDepth);

        // Apply the edge detector
        cv::Mat edgeImage;
        applyEdgeDetector(rectifiedMultiChannelImage, edgeImage, 0);

#if INCLUDE_DEPTH
        cv::Mat edgeImageDepth;
        applyEdgeDetector(rectifiedMultiChannelImageDepth, edgeImageDepth, 1);
	
        cv::bitwise_or(edgeImage, edgeImageDepth, edgeImage);
#endif
        // Get the canny edge image
        std::vector<cv::Mat> channels;
        cv::split(rectifiedMultiChannelImage, channels);
        
        cv::Mat cannyEdges;
        Processing::computeCannyEdges(channels[EDGE_DETECTOR_CHANNEL_INTENSITY], cannyEdges);
        
        // Detect rectangles
        std::vector<Rectangle> partHypotheses;
        detectRectangles(edgeImage, cannyEdges, partHypotheses);

        std::vector<Rectangle> selection;
        int packingNumber, coverNumber;
        pruneRectangles((rectifiedMultiChannelImage.rows+1)*(rectifiedMultiChannelImage.cols+1), partHypotheses, packingNumber, coverNumber, selection);
    
        std::vector<Rectangle> rectified;
        rectifyParts(std::get<1>(images[i]).regionOfInterest, std::get<1>(images[i]).parts, rectified);
        
        // Get the two different gradient magnitude images
        cv::Mat gradientMagnitude;
        channels[EDGE_DETECTOR_CHANNEL_GM].copyTo(gradientMagnitude);
        
        cv::Mat foregroundGM, backgroundGM;
        gradientMagnitude.copyTo(foregroundGM);
        gradientMagnitude.copyTo(backgroundGM);
        removeNonRectanglePixels2(selection, backgroundGM);
        removeNonRectanglePixels3(selection, foregroundGM);
        
        
        // Determine the label for each rectangle
        for (size_t r = 0; r < selection.size(); r++)
        {
            float maxIOU = 0;
            int bestLabel = 0;
            
            // Check against the ground truth44
            for (size_t l = 0; l < rectified.size(); l++)
            {
                const Rectangle & gtRect = rectified[l];
                
                // Compute the IOU
                Rectangle intersectionRect, unionRect;
                RectangleUtil::calcIntersection(gtRect, selection[r], intersectionRect);
                RectangleUtil::calcUnion(gtRect, selection[r], unionRect);
                
                const float score = intersectionRect.getArea()/unionRect.getArea();
                
                if (score > maxIOU)
                {
                    bestLabel = std::get<1>(images[i]).labels[l];
                    maxIOU = score;
                }
            }
            
            // Extract the part image from the entire image
            int width = std::round(selection[r].getWidth());
            int height = std::round(selection[r].getHeight());
            cv::Mat extracted(height, width, CV_32F);
            
            if (maxIOU < 0.80)
            {
                // This is a background rectangle
                for (int w = 0; w < width; w++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        extracted.at<float>(h,w) = backgroundGM.at<float>(h+static_cast<int>(selection[r][0][1]), w+static_cast<int>(selection[r][0][0]));
                    }
                }
                
                // Save it
                std::stringstream ss;
                ss << "data/background/" << counter << ".csv";
                saveImageToFile(ss.str(), extracted);
            }
            else
            {
                // This is a foreground rectangle
                for (int w = 0; w < width; w++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        extracted.at<float>(h,w) = foregroundGM.at<float>(h+static_cast<int>(selection[r][0][1]), w+static_cast<int>(selection[r][0][0]));
                    }
                }
                
                // Save it
                std::stringstream ss;
                ss << "data/";
                switch (bestLabel)
                {
                    case 0:
                        ss << "door";
                        break;
                        
                    case 1:
                        ss << "drawer";
                        break;
                        
                    default:
                    case 2:
                        ss << "shelf";
                        break;
                }
                ss << "/" << counter << ".csv";
                saveImageToFile(ss.str(), extracted);
            }
            
            #pragma omp critical
            {
                counter++;
            }
#if 0
            // Convert the images to u char
            cv::Mat demo, demoE;
            extracted.convertTo(demoE, CV_8UC1);
            channels[EDGE_DETECTOR_CHANNEL_INTENSITY].convertTo(demo, CV_8UC1);
            
            PlotUtil::plotRectangle(demo, partHypotheses[r], 0, 2);
            cv::imshow("demoE", demoE);
            cv::imshow("demo", demo);
            
            cv::waitKey();
            
#endif
        }
    }

}
*/
        

////////////////////////////////////////////////////////////////////////////////
//// Segmentation
////////////////////////////////////////////////////////////////////////////////

static inline void loadRectangleFromJSON(const rapidjson::Value& _points, Rectangle & rectangle)
{
    rectangle[0][0] = _points[(rapidjson::SizeType) 0][(rapidjson::SizeType) 0].GetDouble();
    rectangle[0][1] = _points[(rapidjson::SizeType) 0][(rapidjson::SizeType) 1].GetDouble();
    rectangle[1][0] = _points[(rapidjson::SizeType) 1][(rapidjson::SizeType) 0].GetDouble();
    rectangle[1][1] = _points[(rapidjson::SizeType) 1][(rapidjson::SizeType) 1].GetDouble();
    rectangle[2][0] = _points[(rapidjson::SizeType) 2][(rapidjson::SizeType) 0].GetDouble();
    rectangle[2][1] = _points[(rapidjson::SizeType) 2][(rapidjson::SizeType) 1].GetDouble();
    rectangle[3][0] = _points[(rapidjson::SizeType) 3][(rapidjson::SizeType) 0].GetDouble();
    rectangle[3][1] = _points[(rapidjson::SizeType) 3][(rapidjson::SizeType) 1].GetDouble();

    // Normalize the rectangle (See comments on the function)
    rectangle.normalize();
}

void Segmentation::readAnnotationFile(const std::string& filename)
{
    file = boost::filesystem::basename(filename);
    // Load file file
    JSON::ptr annotation = JSON::Factory::loadFromFile(filename);
    
    // Load the region of interest
    loadRectangleFromJSON((*annotation)["regionOfInterest"], regionOfInterest);

    // Load the parts
    const rapidjson::Value& parts = (*annotation)["parts"];
    for (rapidjson::Value::ConstValueIterator partItr = parts.Begin(); partItr != parts.End(); ++partItr)
    {
        // Do not process background items
        const std::string label = (*partItr)["label"].GetString();
        if (label == "Background")
        {
            continue;
        }
        
        // Convert the label to int
        int intLabel = 0;
        if (label == "Door")
        {
            intLabel = 0;
        }
        else if (label == "Drawer")
        {
            intLabel = 1;
        }
        else if (label == "Shelf")
        {
            intLabel = 2;
        }
        else
        {
            throw ParserException("Unknown part label.");
        }
        
        labels.push_back(intLabel);
        
        // Load the bounding box of the functional part
        Rectangle partRect;
        loadRectangleFromJSON((*partItr)["points"], partRect);
        this->parts.push_back(partRect);
    }
}

void Segmentation::readAuxiliaryFile(const std::string& filename)
{
    // Load file file
    JSON::ptr annotation = JSON::Factory::loadFromFile(filename);
    
    // Load the region of interest
    loadRectangleFromJSON((*annotation)["regionOfInterest"], regionOfInterest);
}


/*
void CabinetParser::rectifyDepth(const cv::Mat & unRectifiedRGB, const cv::Mat & unRectifiedDepth, cv::Mat & rectifiedRGB)
{
	rectifiedRGB = cv::Mat::zeros(unRectifiedRGB.rows, unRectifiedRGB.cols, CV_32FC1);
	cv::Mat depth_imagetmp, depth_image;
	//unRectifiedDepth.convertTo(depth_image,CV_32FC1);
	//unRectifiedDepth.copyTo(depth_image);
	depth_imagetmp = cv::imread("apple_depth.png", 1);
	depth_imagetmp.convertTo(depth_image,CV_32F);
	std::cerr << "Depth Image Mat type " <<depth_image.channels()<< std::endl;
	
	if(!depth_image.data)
	{ 
        std::cerr << "No depth data!!!" << std::endl; 
        exit(EXIT_FAILURE); 
	} 

	const float dc1  = -0.0030711016; 
    	const float dc2  = 3.3309495161; 
    	const float fx_d = 525.0f; 
    	const float fy_d = 525.0f; 
    	const float px_d = 320.0f; 
    	const float py_d = 240.0f;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	std::cerr << "Setting up point cloud" << std::endl;
	
  	// Fill in the cloud data
  	cloud->width    = depth_image.cols;
  	cloud->height   = depth_image.rows;
  	cloud->is_dense = false;
  	cloud->points.resize (cloud->width * cloud->height);
	int centerX     = (cloud->width >> 1); 
	int centerY     = (cloud->height >> 1); 
	int depth_idx   = 0; 

	std::cerr << centerX<<"x"<<centerY<<std::endl;

  	for(int v=-centerY; v < centerY; ++v) 
	{
        	for(int u = -centerX; u < centerX; ++u, ++depth_idx)
		{
			pcl::PointXYZ& pt = cloud->points[depth_idx];
			
			float Z = depth_image.at<float>(v,u);
			if(Z>0.0f && Z<255.0f)
				//pt.z = Z* 0.001f;
				pt.z = 1.0f / (depth_image.at<float>(v,u)*dc1+dc2);	
			else
				//pt.z = static_cast<float> (nanf);// TO DO
				//pt.z = Z* 0.001f;
				pt.z = 1.0f / (depth_image.at<float>(v,u)*dc1+dc2);
			
			pt.x = static_cast<float>(u) *pt.z / fx_d;
    			pt.y = static_cast<float>(v) *pt.z / fy_d;			
    			
  		}
	}

	//cloud->sensor_origin_.setZero (); 
  	//cloud->sensor_orientation_.w () = 0.0f;
  	//cloud->sensor_orientation_.x () = 1.0f; 
  	//cloud->sensor_orientation_.y () = 0.0f; 
  	//cloud->sensor_orientation_.z () = 0.0f;

	std::vector<int> mapping;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);// Note: Will make the cloud unorganized for ever
  
 	pcl::PCDWriter writerFullCloud;
  	writerFullCloud.write<pcl::PointXYZ> ("cloud_0.pcd", *cloud, false);

	std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height << " data points." << std::endl;
	std::cerr << "Saved " << cloud->points.size () << " data points to cloud_0.pcd." << std::endl;

	

	// Downsampling the pcl data

	pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2), cloud_filtered_blob (new pcl::PCLPointCloud2);
  	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

  	// Fill in the cloud data
  	pcl::PCDReader reader;
  	//reader.read ("cloud_0.pcd", *cloud_blob);
	reader.read ("cloud_0.pcd", *cloud_blob);// TO DO

  	// Create the filtering object: downsample the dataset using a leaf size of 1cm
  	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  	sor.setInputCloud (cloud_blob);
 	sor.setLeafSize (0.01f, 0.01f, 0.01f);
  	sor.filter (*cloud_filtered_blob);

  	// Convert to the templated PointCloud
  	pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);

  	std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;

  	// Write the downsampled version to disk
  	pcl::PCDWriter writer;
  	writer.write<pcl::PointXYZ> ("cloud_0_downsampled.pcd", *cloud_filtered, false);
	std::cerr << "Downsampled to : " << cloud_filtered->width * cloud_filtered->height << " data points and saved to cloud_0_downsampled.pcd" << std::endl;
	
	

	// -----Compute surface normals -----
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr testCloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> ("cloud_0_xtion.pcd", *testCloud_ptr) == -1) //* load the file
  	{
    		PCL_ERROR ("Couldn't read file cloud_0.pcd \n");
    	}
	std::cout << "Loaded the cloud with height "<<testCloud_ptr->height<<" and width "<<testCloud_ptr->width<<" \n\n";
	std::cout << "Saving cloud to png.\n\n";
	pcl::io::savePNGFile("output_xtion.png", *testCloud_ptr,"rgb");
	std::cout << "cloud saved.\n\n";	
	
  	std::cerr << "Computing normals for: " << testCloud_ptr->width * testCloud_ptr->height << " data points " << std::endl;
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
  	//ne.setInputCloud (cloud_filtered);
  	ne.setInputCloud (testCloud_ptr);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr cloudTree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  	ne.setSearchMethod (cloudTree);
  	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  	ne.setRadiusSearch (0.1);// default 0.05 or 0.1
  	ne.compute (*cloud_normals);
  	std::cout << "Normals computed.\n\n";

	std::cout << "Clustering by kmeans.\n\n";

	//KMEANS clustering of normals
	// feature formation
	cv::Mat normalsFeatVecHoriz;
	cv::Mat normalsFeatVec;
	for (size_t currentPoint = 0; currentPoint < cloud_normals->points.size() ; currentPoint+=1)// TO DO
	{
		hconcat(cloud_normals->points[currentPoint].normal[0],cloud_normals->points[currentPoint].normal[1],normalsFeatVecHoriz);
		hconcat(normalsFeatVecHoriz,cloud_normals->points[currentPoint].normal[2],normalsFeatVecHoriz);
		
		normalsFeatVec.push_back(normalsFeatVecHoriz);
	}
	normalsFeatVec.convertTo(normalsFeatVec,CV_32F);  
	int clusterCount = 2;
	int attempts = 5; 	
    	
	
  	cv::Mat labels;  
  	cv::Mat centers;
  	cv::kmeans(normalsFeatVec, clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 100, 0.01), attempts, cv::KMEANS_PP_CENTERS, centers );

	std::cout << "Clustering Done. "<<std::endl<<centers.rows<<" "<<centers.cols<<" \n\n";
	std::cout << "Kmeans converged-"<<std::endl;
	std::cout <<normalsFeatVec.rows<<" normals each with "<<normalsFeatVec.cols<<" features were clustered into "<<clusterCount<<" =="<<centers.cols<<" groups\n\n";
	// Gram Schmidt Orthonormalization
	Mat33f A;
    	float e = 1e-4f;

	A.col[0] = Vec3f(1.0f, -1.0f, 1.0f);
    	A.col[1] = Vec3f(1.0f, 0.0f, 1.0f);
    	A.col[2] = Vec3f(1.0f, 1.0f, 2.0f);

    	print_mat("A", A);

    	Mat33f CGS;
    	classic_gram_schmidt(CGS, A);
    	print_mat("CGS", CGS);

    	Mat33f MGS;
    	modified_gram_schmidt(MGS, A);
    	print_mat("MGS", MGS);

	transformPCL();


}
*/

