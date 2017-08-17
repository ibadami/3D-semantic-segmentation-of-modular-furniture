#include <cmath>

#include "parser/detector.h"
#include "parser/processing.h"
#include "libforest/libforest.h"
#include <set>
#include <chrono>

using namespace parser;


////////////////////////////////////////////////////////////////////////////////
//// LineDetector
////////////////////////////////////////////////////////////////////////////////


void LineDetector::detectLines( const cv::Mat & image, 
                                std::vector< LineSegment > & lineSegments, 
                                bool horizontal) const
{
    // Detect all lines segments using probabilistic hough transform
#if 0
    // Probabilistic Hough transform that generates line segments
    std::vector<cv::Vec4i> detectedLines;
    cv::HoughLinesP(image,
            detectedLines,
            model.hough.rho,
            model.hough.theta,
            model.hough.threshold,
            model.hough.minLineLength,
            model.hough.maxLineGap);
#else
    // These are the bounding box lines. We need these in order to create line
    // segments from the detected lines
    std::vector<LineSegment> boundingBoxes(4);
    boundingBoxes[0][0][0] = 0;
    boundingBoxes[0][0][1] = 0;
    boundingBoxes[0][1][0] = 10;
    boundingBoxes[0][1][1] = 0;
    
    boundingBoxes[1][0][0] = 0;
    boundingBoxes[1][0][1] = image.rows - 1;
    boundingBoxes[1][1][0] = 10;
    boundingBoxes[1][1][1] = image.rows - 1;
    
    boundingBoxes[2][0][0] = 0;
    boundingBoxes[2][0][1] = 0;
    boundingBoxes[2][1][0] = 0;
    boundingBoxes[2][1][1] = 10;
    
    boundingBoxes[3][0][0] = image.cols - 1;
    boundingBoxes[3][0][1] = 0;
    boundingBoxes[3][1][0] = image.cols - 1;
    boundingBoxes[3][1][1] = 10;
    
    // Standard Hough Transform that generates lines
    cv::vector<cv::Vec2f> lines;
    cv::HoughLines(image, lines, model.hough.rho, model.hough.theta, model.minLength);

    cv::Mat distanceTransform;
    Processing::computeDistanceTransform(image, distanceTransform);

    // Create line segments from these lines
    std::vector<cv::Vec4i> detectedLines;
    for (size_t i = 0; i < lines.size(); i++)
    {
        const float rho = lines[i][0];
        const float theta = lines[i][1];
        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a*rho, y0 = b*rho;
        
        // First: Create a line object from the 
        LineSegment preliminary;
        // We just need two points on the line
        preliminary[0][0] = cvRound(x0 + 1000*(-b));
        preliminary[0][1] = cvRound(y0 + 1000*(a));
        preliminary[1][0] = cvRound(x0 - 1000*(-b));
        preliminary[1][1] = cvRound(y0 - 1000*(a));

        // Now compute the intersection points with the bounding box lines
        std::vector<Vec2> intersectionPoints;
        float eps = 1e-2;
        for (size_t j = 0; j < 4; j++)
        {
            Vec2 v;
            LineUtil::calcIntersectionPoint(boundingBoxes[j], preliminary, v);
            // Check if the point is within the region of interest
            if (-eps <= v[0] && v[0] <= image.cols - 1 + eps &&
                -eps <= v[1] && v[1] <= image.rows - 1 + eps)
            {
                intersectionPoints.push_back(v);
            }
        }
        
        if (intersectionPoints.size() != 2)
        {
            continue;
        }
        
        cv::Vec4i r;
        r[0] = intersectionPoints[0][0];
        r[1] = intersectionPoints[0][1];
        r[2] = intersectionPoints[1][0];
        r[3] = intersectionPoints[1][1];

        detectedLines.push_back(r);
    }
#endif
    
    // Filter out all lines that are not axis aligned
    for( std::vector<cv::Vec2i>::size_type i = 0; i < detectedLines.size(); i++ )
    {
        cv::Vec4i lineEndPoints = detectedLines[i];
        
        // Create a line segment from the detection
        LineSegment lineSegment;
        lineSegment[0][0] = lineEndPoints[0];
        lineSegment[0][1] = lineEndPoints[1];
        lineSegment[1][0] = lineEndPoints[2];
        lineSegment[1][1] = lineEndPoints[3];
        lineSegment.normalize();
        
        float deviation = 0;
        
        // Check if this line can be added
        if (horizontal)
        {
            // Compute the deviation in y direction
            deviation = std::abs(lineSegment[0][1] - lineSegment[1][1]); 
        }
        else
        {
            // Compute the deviation in x direction
            deviation = std::abs(lineSegment[0][0] - lineSegment[1][0]); 
        }
        
        // The line segments must not deviate more than epsilon
        if (deviation >= model.epsilon)
        {
            continue;
        }
        
#if 1
        // Check if there is a line that is just 1px away from this one
        bool conflict = false;
        for (size_t j = 0; j < lineSegments.size(); j++)
        {
            if ((cv::norm(lineSegments[j][0], lineSegment[0]) <= 1.5 && cv::norm(lineSegments[j][1], lineSegment[1]) <= 1.5) ||
                (cv::norm(lineSegments[j][1], lineSegment[0]) <= 1.5 && cv::norm(lineSegments[j][0], lineSegment[1]) <= 1.5))
            {
                conflict = true;
                break;
            }
        }
        if(conflict)
        {
            continue;
        }
#endif

        // This is a legit line
        lineSegments.push_back(lineSegment);
    }
#if 0
    std::cout << lineSegments.size() << "\n";
    
    cv::Mat demo = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    this->visualize(demo, lineSegments, cv::Scalar(255,255,0));
    cv::imshow("edges", image);
    cv::imshow("lines", demo);
    cv::waitKey();
    cv::destroyAllWindows();
#endif
}


void LineDetector::joinLines(   std::vector<LineSegment> & lineSegments, 
                                bool horizontal) const
{
    cv::Mat demo(500, 500, CV_8UC3);
    demo = cv::Scalar(0);
    
    // For each line, check if there is another line that can be joined
    for (size_t l = 0; l < lineSegments.size(); l++)
    {
        const LineSegment currentSegment = lineSegments[l];

        // Collect possible joins
        size_t best = lineSegments.size();
        LineSegment bestJoin;
        float bestScore = 1e20;

        // Search for another line segment that can be used to join the two
        // segments
        for (size_t m = 0; m < lineSegments.size(); m++)
        {
            // Do not perform self-joins
            if (m == l) continue;
            
            const LineSegment matchCandidate = lineSegments[m];
            
            // Check if the end points are sufficiently close to each other
            float deviation, orthogonalDeviation = 0;
            int x = 0;
            int y = 1;
            if (!horizontal)
            {
                std::swap(x,y);
            }
            
            
            // Compute the deviation in y direction
            orthogonalDeviation = std::max(
                std::max(
                    std::abs(currentSegment[0][y] - matchCandidate[0][y]), 
                    std::abs(currentSegment[1][y] - matchCandidate[0][y])), 
                std::max(
                    std::abs(currentSegment[0][y] - matchCandidate[1][y]), 
                    std::abs(currentSegment[1][y] - matchCandidate[1][y])) 
                );
            
            // Compute the orthogonal deviation
            // Do the line segments overlap in x direction?
            if ((currentSegment[0][x] <= matchCandidate[0][x] && matchCandidate[0][x] <= currentSegment[1][x]) || 
                (currentSegment[1][x] <= matchCandidate[0][x] && matchCandidate[0][x] <= currentSegment[0][x]) || 
                (currentSegment[0][x] <= matchCandidate[1][x] && matchCandidate[1][x] <= currentSegment[1][x]) || 
                (currentSegment[1][x] <= matchCandidate[1][x] && matchCandidate[1][x] <= currentSegment[0][x]))
            {
                // They do
                deviation = 0;
            }
            else
            {
                // Nope, the don't
                // Compute the minimum distance
                deviation = std::min(
                    std::min(
                        std::abs(currentSegment[0][x] - matchCandidate[0][x]), 
                        std::abs(currentSegment[1][x] - matchCandidate[0][x])), 
                    std::min(
                        std::abs(currentSegment[0][x] - matchCandidate[1][x]), 
                        std::abs(currentSegment[1][x] - matchCandidate[1][x])) 
                    );
            }
            
            float totalScore = deviation + orthogonalDeviation;
            /*
            demo = cv::Scalar(0);
            PlotUtil::plotLineSegment(demo, currentSegment, cv::Scalar(255,255,0));
            PlotUtil::plotLineSegment(demo, matchCandidate, cv::Scalar(0,255,255));
            cv::imshow("test", demo);
            std::cout << "d = " << deviation << ", o = " << orthogonalDeviation << "\n";
            cv::waitKey();
            */
            if (deviation < model.maxJoinDistance && orthogonalDeviation < model.maxOrthogonalJoinDistance && totalScore < bestScore)
            {
                // Generate the hypothesis
                LineSegmentUtil::join(currentSegment, matchCandidate, bestJoin);
                bestScore = totalScore;
                best = m;
            }
        }

        // Did we find a good join candidate?
        if (best < lineSegments.size())
        {
            // Remove the two joined segments
            if (best > l)
            {
                lineSegments.erase(lineSegments.begin() + best);
                lineSegments.erase(lineSegments.begin() + l);
            }
            else
            {
                lineSegments.erase(lineSegments.begin() + l);
                lineSegments.erase(lineSegments.begin() + best);
            }
            
            // Add the new segment
            lineSegments.push_back(bestJoin);
            
            // Restart the process
            joinLines(lineSegments, horizontal);
            return;
        }
    }
}


void LineDetector::visualize(   cv::Mat & image, 
                                const std::vector<LineSegment> & lineSegments, 
                                const cv::Scalar & color) const
{
    for (size_t i = 0; i < lineSegments.size(); i++)
    {
        PlotUtil::plotLineSegment(image, lineSegments[i], color);
    }
}

void LineDetector::addBoundingBoxLines( const cv::Mat & image,
                                        std::vector<LineSegment> & lineSegments, 
                                        bool horizontal) const
{
    LineSegment left, right;
    
    if (horizontal)
    {
        left[0][0] = 0;
        left[0][1] = 0;
        left[1][0] = image.cols - 1;
        left[1][1] = 0;
        
        right[0][0] = 0;
        right[0][1] = image.rows - 1;
        right[1][0] = image.cols - 1;
        right[1][1] = image.rows - 1;
    }
    else
    {
        left[0][0] = 0;
        left[0][1] = 0;
        left[1][0] = 0;
        left[1][1] = image.rows - 1;
        
        right[0][0] = image.cols - 1;
        right[0][1] = 0;
        right[1][0] = image.cols - 1;
        right[1][1] = image.rows - 1;
    }
    
    lineSegments.push_back(left);
    lineSegments.push_back(right);
}

void LineDetector::filterShortLines(const std::vector<LineSegment> & lineSegmentsIn, 
                                    std::vector<LineSegment> & lineSegmentsOut) const
{
    for (size_t i = 0; i < lineSegmentsIn.size(); i++)
    {
        if (lineSegmentsIn[i].getLength() >= model.minLength)
        {
            lineSegmentsOut.push_back(lineSegmentsIn[i]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//// RectangleDetector
////////////////////////////////////////////////////////////////////////////////

static inline void sampleTwoDistinct(std::mt19937 & g, std::uniform_int_distribution<int> & d, int & v1, int & v2)
{
    v1 = d(g);
    do {
        v2 = d(g);
    } while (v1 == v2);
    
    if (v2 < v1)
    {
        std::swap(v1, v2);
    }
}

void RectangleDetector::detectRectangles(   const cv::Mat & image, 
                                            const std::vector<LineSegment> & lineSegmentsH, 
                                            const std::vector<LineSegment> & lineSegmentsV,
                                            std::vector<Rectangle> & result, 
                                            std::function<bool(const Rectangle &)> callback) const
{
    // Set up probability distributions over the two sets of line segments
    std::uniform_int_distribution<int> distH(0, static_cast<int>(lineSegmentsH.size()) - 1);
    std::uniform_int_distribution<int> distV(0, static_cast<int>(lineSegmentsV.size()) - 1);
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    
    // This is the set of lines that we have already sampled
    std::set<std::string> sampled;

    //std::cout<<"Number of samples used for rectangle detection: "<<model.numSamples<<std::endl;
    
    // Start sampling
    #pragma omp parallel for
    for (int i = 0; i < model.numSamples; i++)
    {
        // Sample two distinct values
        int h1, h2, v1, v2;
        sampleTwoDistinct(g, distH, h1, h2);
        sampleTwoDistinct(g, distV, v1, v2);
        
        // Create the hash value and check if we have already sampled this pair
        std::stringstream hash;
        hash << h1 << '.' << h2 << '-' << v1 << '.' << v2;
        
        bool con = true;
        #pragma omp critical
        {
            if (sampled.find(hash.str()) != sampled.end())
            {
                // We already encountered this combination of lines
                con = false;
            }
            else
            {
                // Add the hash value to the set
                sampled.insert(hash.str());
            }
        }
        if (!con) continue;
        
        // Get the line segments
        const LineSegment & lh1 = lineSegmentsH[h1];
        const LineSegment & lh2 = lineSegmentsH[h2];
        const LineSegment & lv1 = lineSegmentsV[v1];
        const LineSegment & lv2 = lineSegmentsV[v2];
        
        // Ignore obviously wrong rectangles
        if (std::abs(lh1[0][1] - lh2[0][1]) < 5 || std::abs(lv1[0][0] - lv2[0][0]) < 5)
        {
            continue;
        }
        
        // The line joining might have broken some segments. Simply reject those.
        if (    std::abs(lh1[0][1] - lh1[1][1]) > 3 || 
                std::abs(lh2[0][1] - lh2[1][1]) > 3 || 
                std::abs(lv1[0][0] - lv1[1][0]) > 3 || 
                std::abs(lv2[0][0] - lv2[1][0]) > 3)
        {
            continue;
        }
        
        // Compute the rectangle score. This is the maximum distance between 
        // any pair of lines
        float score = std::max(
            std::max(LineSegmentUtil::calcDistance(lh1, lv1), LineSegmentUtil::calcDistance(lh1, lv2)),
            std::max(LineSegmentUtil::calcDistance(lh2, lv1), LineSegmentUtil::calcDistance(lh2, lv2))
        );
        
        // Only create the rectangle if the score is small enough
        if (score > model.maxScore)
        {
            continue;
        }
        
        // Create the rectangle
        Rectangle r;
        LineUtil::calcIntersectionPoint(lh1, lv1, r[0]);
        LineUtil::calcIntersectionPoint(lh1, lv2, r[1]);
        LineUtil::calcIntersectionPoint(lh2, lv1, r[2]);
        LineUtil::calcIntersectionPoint(lh2, lv2, r[3]);
        r.normalize();
        
        // Rectify the rectangle
        Rectangle r_;
        r_[0][0] = 0.5*(r[0][0] + r[3][0]);
        r_[3][0] = 0.5*(r[0][0] + r[3][0]);
        r_[0][1] = 0.5*(r[0][1] + r[1][1]);
        r_[1][1] = 0.5*(r[0][1] + r[1][1]);
        r_[1][0] = 0.5*(r[1][0] + r[2][0]);
        r_[2][0] = 0.5*(r[1][0] + r[2][0]);
        r_[2][1] = 0.5*(r[2][1] + r[3][1]);
        r_[3][1] = 0.5*(r[2][1] + r[3][1]);
        r = r_;
        r.normalize();
        
        // We enforce some minimum width/height in order to avoid numerical
        // problems later on in the pipeline
        if (r.getWidth() < 32 || r.getHeight() < 32)
        {
            continue;
        }
        
        // Don't add the rectangle if it's IOU with some other rectangle is too
        // high
        float maxIOU = 0;
        #pragma omp critical
        {
            for (size_t m = 0; m < result.size(); m++)
            {
                Rectangle intersectionRectangle;
                Rectangle unionRectangle;
                RectangleUtil::calcIntersection(r, result[m], intersectionRectangle);
                RectangleUtil::calcUnion(r, result[m], unionRectangle);
                const float iou = intersectionRectangle.getArea()/unionRectangle.getArea();
                maxIOU = std::max(maxIOU, iou);
            }
        }
        
        if (maxIOU > model.maxIOU)
        {
            continue;
        }
        
        if (std::isnan(r[0][0]) || std::isnan(r[1][0]) || std::isnan(r[2][0]) || std::isnan(r[3][0]))
        {
            continue;
        }
        
        if (callback(r))
        {
            #pragma omp critical
            {
                // Add the rectangle
                result.push_back(r);
            }
        }
    }
}

void RectangleDetector::visualize(cv::Mat& image, const std::vector<Rectangle>& rectangles, const cv::Scalar& color) const
{
    for (size_t r = 0; r < rectangles.size(); r++)
    {
        PlotUtil::plotRectangle(image, rectangles[r], color);
    }
}
