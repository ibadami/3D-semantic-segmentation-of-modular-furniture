#include "parser/util.h"

#include <algorithm>

using namespace parser;

////////////////////////////////////////////////////////////////////////////////
/// VectorUtil
////////////////////////////////////////////////////////////////////////////////

void VectorUtil::applyHomography(const cv::Mat & homography, const cv::Vec<floatT, 2> & x_in, cv::Vec<floatT, 2> & x_out)
{
    // Compute the normalization constant (Hx)_3
    const double normalize =    x_in[0] * homography.at<double>(2,0) + 
                                x_in[1] * homography.at<double>(2,1) + 
                                homography.at<double>(2,2);
    // Compute the first entry (Hx)_1
    const floatT x1 =   (   x_in[0] * homography.at<double>(0,0) + 
                            x_in[1] * homography.at<double>(0,1) + 
                            homography.at<double>(0,2)
                        ) / normalize;
    // Compute the second entry (Hx)_2
    const floatT x2 =   (   x_in[0] * homography.at<double>(1,0) + 
                            x_in[1] * homography.at<double>(1,1) + 
                            homography.at<double>(1,2)
                        ) / normalize;
    // Assign the results to the output vector
    x_out[0] = x1;
    x_out[1] = x2;
}

////////////////////////////////////////////////////////////////////////////////
/// LineSegment
////////////////////////////////////////////////////////////////////////////////

Vec3 LineSegment::getLineNormal() const
{
    Vec3 normal(at(0)[1] - at(1)[1], -(at(0)[0] - at(1)[0]), 0);
    floatT length = cv::norm(normal);

    normal[2] = -(normal[0] * at(0)[0] + normal[1] * at(0)[1]);

    // Normalize the equation
    normal[0] = normal[0]/length;
    normal[1] = normal[1]/length;
    normal[2] = normal[2]/length;

    return normal;
}

////////////////////////////////////////////////////////////////////////////////
/// LineSegment
////////////////////////////////////////////////////////////////////////////////

floatT LineUtil::calcPointDistance(const LineSegment & line, const Vec2 & point)
{
    // Get the normal form of the line
    Vec3 normal = line.getLineNormal();
    // Calculate the distance
    return std::abs(normal[0]*point[0] + normal[1]*point[1] + normal[2]);
}

void LineUtil::calcIntersectionPoint(const LineSegment & line1, const LineSegment & line2, Vec2 & point)
{
    floatT denominator = (line1[0][0] - line1[1][0])*
                    (line2[0][1] - line2[1][1]) 
                    - 
                    (line1[0][1] - line1[1][1])*
                    (line2[0][0] - line2[1][0]);
    point[0] =  (
                    (line1[0][0]*line1[1][1] - line1[0][1] * line1[1][0]) * 
                    (line2[0][0] - line2[1][0]) 
                    - 
                    (line1[0][0] - line1[1][0])*
                    (line2[0][0] * line2[1][1] - line2[0][1] * line2[1][0])
                )/denominator;

    point[1] =  (
                    (line1[0][0]*line1[1][1] - line1[0][1] * line1[1][0]) * 
                    (line2[0][1] - line2[1][1]) 
                    - 
                    (line1[0][1] - line1[1][1])*
                    (line2[0][0] * line2[1][1] - line2[0][1] * line2[1][0])
                )/denominator;
}

////////////////////////////////////////////////////////////////////////////////
/// LineSegmentUtil
////////////////////////////////////////////////////////////////////////////////

void LineSegmentUtil::applyHomography(const cv::Mat & homography, const LineSegment & line_in, LineSegment & line_out)
{
    // Apply the homography to each point individually
    VectorUtil::applyHomography(homography, line_in[0], line_out[0]);
    VectorUtil::applyHomography(homography, line_in[1], line_out[1]);
}

floatT LineSegmentUtil::calcPointDistance(const LineSegment & line, const Vec2 & point)
{
    // Get the projection point
    Vec2 projectionPoint = line[1] -line[0];
    // Project the point onto the line that is defined by the line segment
    floatT score =  projectionPoint.dot(point - line[0])/static_cast<double>(projectionPoint.dot(projectionPoint));

    // Check the length of the projection
    if (0 <= score && score <= 1)
    {
        // The projection lies in between p1 and p2. Thus the distance
        // is the distance from the point to the line defined by
        // p1 and p2
        return LineUtil::calcPointDistance(line, point);
    }
    else
    {
        // The projection of the point is not in-between p1 and p2. Thus
        // the distance to the line segment is the minimum distance to
        // any of the end points
        return std::min(
            cv::norm(line[0], point), 
            cv::norm(line[1], point)
        );
    }
}

floatT LineSegmentUtil::calcHingeDirection(const LineSegment & line, const Vec2 & point)
{
    Vec2 y1 = line[1] - line[0];
    Vec2 y2 = point - line[1];
    return VectorUtil::cross(y1, y2);
}

bool LineSegmentUtil::isOnSegment(const LineSegment & line, const Vec2 & x)
{
    Vec2 topRight(std::max(line[0][0], line[1][0]), std::max(line[0][1], line[1][1]));
    Vec2 bottomLeft(std::min(line[0][0], line[1][0]), std::min(line[0][1], line[1][1]));

    return x[0] <= topRight[0] && x[1] <= topRight[1] && bottomLeft[0] <= x[0] && bottomLeft[1] <= x[1];
}


bool LineSegmentUtil::doIntersect(const LineSegment & line1, const LineSegment & line2)
{
    floatT d1 = calcHingeDirection(line1, line2[0]);
    floatT d2 = calcHingeDirection(line1, line2[1]);
    
    if (std::abs(d1) <= 1e-10 && isOnSegment(line1, line2[0]))
    {
        return true;
    }
    else if (std::abs(d2) <= 1e-10 && isOnSegment(line1, line2[1]))
    {
        return true;
    }
    else if ((d1 >= 0 && d2 >= 0) || (d1 <= 0 && d2 <= 0))
    {
        return false;
    }
    else
    {
        floatT d3 = calcHingeDirection(line2, line1[0]);
        floatT d4 = calcHingeDirection(line2, line1[1]);
        
        if (std::abs(d3) <= 1e-8 && isOnSegment(line2, line1[0]))
        {
            return true;
        }
        else if (std::abs(d4) <= 1e-8 && isOnSegment(line2, line1[1]))
        {
            return true;
        }
        else if ( (d3 <= 0 && d4 <= 0) || (d3 >= 0 && d4 >= 0))
        {
            return false;
        }
        else
        {
            return true;
        }
    }
}

void LineSegmentUtil::join(const LineSegment & line_in1, const LineSegment & line_in2, LineSegment & line_out)
{
    // Calculate the distances between each two pair of points
    // Put all points into one array
    Vec2 points[4] = {
        line_in1[0], 
        line_in1[1], 
        line_in2[0], 
        line_in2[1], 
    };

    // Determine the maximum distance
    int i_max = 0;
    int j_max = 0;
    floatT maxDist = 0;
    for (int i = 0; i < 4; i++)
    {
        for (int j = i+1; j < 4; j++)
        {
            const double currentDist = cv::norm(points[i], points[j]);
            if (currentDist > maxDist)
            {
                i_max = i;
                j_max = j;
                maxDist = currentDist;
            }
        }
    }

    // Create the line
    line_out[0] = points[i_max];
    line_out[1] = points[j_max];
}

floatT LineSegmentUtil::calcDistance(const LineSegment & line1, const LineSegment & line2)
{
    // If the line segments intersect, their distance is 0
    if (doIntersect(line1, line2))
    {
        return 0;
    }
    else
    {
        // The distance is the minimum distance between the end points
        // and the other line
        return std::min(
            std::min(
                calcPointDistance(line1, line2[0]), 
                calcPointDistance(line1, line2[1])
            ),
            std::min(
                calcPointDistance(line2, line1[0]),
                calcPointDistance(line2, line1[1])
            )
        );
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Rectangle
////////////////////////////////////////////////////////////////////////////////

void Rectangle::normalize()
{
    // Determine the top left point
    int topLeft = 0;
    floatT s = cv::norm(at(topLeft));
    for (int i = 1; i < 4; i++)
    {
        const floatT s2 = cv::norm(at(i));
        if (s2 < s)
        {
            s = s2;
            topLeft = i;
        }
    }
    
    std::swap(at(0), at(topLeft));
    
    // Sort the points by their angle 
    std::sort(getVertices() + 1, getVertices() + 4, [this](const Vec2 & lhs, const Vec2 & rhs) -> bool {
        Vec2 lhsD = lhs - this->at(0);
        Vec2 rhsD = rhs - this->at(0);
        return std::atan2(lhsD[0], lhsD[1]) > std::atan2(rhsD[0], rhsD[1]);
    });
}

////////////////////////////////////////////////////////////////////////////////
/// RectangleUtil
////////////////////////////////////////////////////////////////////////////////

void RectangleUtil::convertToOpenCV(const Rectangle & rect, cv::Rect_<floatT> & out)
{
    out.x = rect[0][0];
    out.y = rect[0][1];
    out.width = rect.getWidth();
    out.height = rect.getHeight();
}

void RectangleUtil::convertFromOpenCV(const cv::Rect_<floatT> & in, Rectangle & rect)
{
    rect[0][0] = in.x;
    rect[0][1] = in.y;
    rect[1] = rect[0];
    rect[1][0] += in.width;
    rect[2] = rect[1];
    rect[2][1] += in.height;
    rect[3] = rect[0];
    rect[3][1] += in.height;
}

void RectangleUtil::calcUnion(const Rectangle & rect_in1, const Rectangle & rect_in2, Rectangle & rect_out)
{
    float x1 = std::min(rect_in1.minX(), rect_in2.minX());
    float x2 = std::max(rect_in1.maxX(), rect_in2.maxX());
    float y1 = std::min(rect_in1.minY(), rect_in2.minY());
    float y2 = std::max(rect_in1.maxY(), rect_in2.maxY());
    
    rect_out[0][0] = x1;
    rect_out[0][1] = y1;
    rect_out[1][0] = x2;
    rect_out[1][1] = y1;
    rect_out[2][0] = x2;
    rect_out[2][1] = y2;
    rect_out[3][0] = x1;
    rect_out[3][1] = y2;
    rect_out.normalize();
}

void RectangleUtil::calcIntersection(const Rectangle & rect_in1, const Rectangle & rect_in2, Rectangle & rect_out)
{
    float x1 = std::max(rect_in1.minX(), rect_in2.minX());
    float x2 = std::min(rect_in1.maxX(), rect_in2.maxX());
    float y1 = std::max(rect_in1.minY(), rect_in2.minY());
    float y2 = std::min(rect_in1.maxY(), rect_in2.maxY());
    
    if (x1 >= x2 || y1 >= y2)
    {
        x1 = x2 = y1 = y2 = 0;
    }
    
    rect_out[0][0] = x1;
    rect_out[0][1] = y1;
    rect_out[1][0] = x2;
    rect_out[1][1] = y1;
    rect_out[2][0] = x2;
    rect_out[2][1] = y2;
    rect_out[3][0] = x1;
    rect_out[3][1] = y2;
}

void RectangleUtil::calcTightAxisAlignedFit(const Rectangle & rect_in, Rectangle & rect_out)
{
    // Determine the smallest/largest x/y values
    floatT minX = std::floor(std::min(
            std::min(rect_in[0][0], rect_in[1][0]), 
            std::min(rect_in[2][0], rect_in[3][0])
    ));
    floatT minY = std::floor(std::min(
            std::min(rect_in[0][1], rect_in[1][1]), 
            std::min(rect_in[2][1], rect_in[3][1])
    ));
    floatT maxX = std::ceil(std::max(
            std::max(rect_in[0][0], rect_in[1][0]), 
            std::max(rect_in[2][0], rect_in[3][0])
    ));
    floatT maxY = std::ceil(std::max(
            std::max(rect_in[0][1], rect_in[1][1]), 
            std::max(rect_in[2][1], rect_in[3][1])
    ));
    
    rect_out[0][0] = minX;
    rect_out[0][1] = minY;
    rect_out[1][0] = maxX;
    rect_out[1][1] = minY;
    rect_out[2][0] = maxX;
    rect_out[2][1] = maxY;
    rect_out[3][0] = minX;
    rect_out[3][1] = maxY;
}

void RectangleUtil::applyHomography(const cv::Mat & homography, const Rectangle & rect_in, Rectangle & rect_out)
{
    VectorUtil::applyHomography(homography, rect_in[0], rect_out[0]);
    VectorUtil::applyHomography(homography, rect_in[1], rect_out[1]);
    VectorUtil::applyHomography(homography, rect_in[2], rect_out[2]);
    VectorUtil::applyHomography(homography, rect_in[3], rect_out[3]);
}

void RectangleUtil::getUnitRectangle(Rectangle & rect)
{
    rect[0][0] = 0;
    rect[0][1] = 0;
    rect[1][0] = 1;
    rect[1][1] = 0;
    rect[2][0] = 1;
    rect[2][1] = 1;
    rect[3][0] = 0;
    rect[3][1] = 1;
}

bool RectangleUtil::isInsideAxisAlignedRectangle(const Rectangle & rect, const Vec2 & p)
{
    return rect[0][0] <= p[0] && p[0] <= rect[1][0] && rect[0][1] <= p[1] && p[1] <= rect[3][1];
}

float RectangleUtil::calcIOU(const Rectangle& r, const Rectangle& q)
{
    Rectangle i, u;
    RectangleUtil::calcIntersection(r, q, i);
    RectangleUtil::calcUnion(r,q,u);
    return i.getArea()/u.getArea();
}

bool RectangleUtil::isHorizontalArrangement(const Rectangle& r, const Rectangle& p)
{
    // Check which edge is shared the most
    const float minX = std::max(r[0][0], p[0][0]);
    const float maxX = std::min(r[1][0], p[1][0]);
    const float minY = std::max(r[0][1], p[0][1]);
    const float maxY = std::min(r[3][1], p[3][1]);
    
    float shareX = std::max(0.0f, maxX - minX);
    float shareY = std::max(0.0f, maxY - minY);
    
    return shareX > shareY;
}

bool RectangleUtil::areSimilar(const Rectangle& r, const Rectangle& p, float threshold)
{
    const float widthDiff = std::abs(r.getWidth() - p.getWidth());
    const float heightDiff = std::abs(r.getHeight() - p.getHeight());
    
    return std::max(widthDiff, heightDiff) <= threshold;
    //return (widthDiff + heightDiff) <= threshold;
}


////////////////////////////////////////////////////////////////////////////////
/// Util
////////////////////////////////////////////////////////////////////////////////

void Util::imshow(const cv::Mat & img, const std::string & title)
{
    // Create the window
    cv::namedWindow(title, 1);
    // Show the image
    cv::imshow(title, img);
    // Move the window to the top left corner of the screen to prevent
    // Ubuntu from trying to arrange the windows
    cvMoveWindow(title.c_str(), 0,0);
    // Wait for the user to press any key
    cv::waitKey();
    // Destroy the created window
    cv::destroyWindow(title);
}

void Util::splitEdgeImage(
        const cv::Mat & edgeImage,
        cv::Mat & horizontalEdges,
        cv::Mat & verticalEdges)
{
    horizontalEdges = cv::Mat::zeros(edgeImage.rows, edgeImage.cols, CV_8UC1);
    verticalEdges = cv::Mat::zeros(edgeImage.rows, edgeImage.cols, CV_8UC1);

    for (int i = 0; i < edgeImage.rows; i++)
    {
        for (int j = 0; j < edgeImage.cols; j++)
        {
            if (edgeImage.at<uchar>(i,j) != 0)
            {
                bool isHorizontal = false;
                bool isVertical = false;
                if (i > 0)
                {
                    isVertical = isVertical || edgeImage.at<uchar>(i-1,j) > 0;
                }
                if (i < edgeImage.rows - 1)
                {
                    isVertical = isVertical || edgeImage.at<uchar>(i+1,j) > 0;
                }
                if (j > 0)
                {
                    isHorizontal = isHorizontal || edgeImage.at<uchar>(i,j-1) > 0;
                }
                if (j < edgeImage.cols - 1)
                {
                    isHorizontal = isHorizontal || edgeImage.at<uchar>(i,j+1) > 0;
                }

                if (isHorizontal)
                {
                    horizontalEdges.at<uchar>(i,j) = 255;
                }
                if (isVertical)
                {
                    verticalEdges.at<uchar>(i,j) = 255;
                }
            }
        }
    }
}
