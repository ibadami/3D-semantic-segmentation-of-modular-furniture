
#include "parser/util.h"
#include "gtest/gtest.h"

using namespace parser;

/**
 * Tests if the homography mapping works in general
 */
TEST(VectorUtil, applyHomography_general)
{
    // Input vector
    Vec2 in(1,1);
    // Output vector
    Vec2 out(7,4);
    // The homography
    cv::Mat H(3,3,CV_64F);
    
    H.at<double>(0,0) = 0.316227766016838;
    H.at<double>(0,1) = 0.421637021355785;
    H.at<double>(0,2) = 0;
    H.at<double>(1,0) = 0;
    H.at<double>(1,1) = 0.421637021355784;
    H.at<double>(1,2) = 0;
    H.at<double>(2,0) = -0.316227766016838;
    H.at<double>(2,1) = -0.210818510677891;
    H.at<double>(2,2) = 0.632455532033676;
    
    // Compute the result
    Vec2 res;
    VectorUtil::applyHomography(H, in, res);
    
    ASSERT_NEAR(out[0], res[0], 1e-6);
    ASSERT_NEAR(out[1], res[1], 1e-6);
}

/**
 * Tests if the inverse homography mapping works
 */
TEST(VectorUtil, applyHomography_inv)
{
    // Output vector
    Vec2 out(7,4);
    // The homography
    cv::Mat H(3,3,CV_64F);
    
    H.at<double>(0,0) = 0.316227766016838;
    H.at<double>(0,1) = 0.421637021355785;
    H.at<double>(0,2) = 0;
    H.at<double>(1,0) = 0;
    H.at<double>(1,1) = 0.421637021355784;
    H.at<double>(1,2) = 0;
    H.at<double>(2,0) = -0.316227766016838;
    H.at<double>(2,1) = -0.210818510677891;
    H.at<double>(2,2) = 0.632455532033676;
    
    // Compute the result
    Vec2 res;
    VectorUtil::applyHomography(H, out, res);
    VectorUtil::applyHomography(H.inv(), res, res);
    
    ASSERT_NEAR(out[0], res[0], 1e-6);
    ASSERT_NEAR(out[1], res[1], 1e-6);
}

/**
 * Tests if the 2D cross product works
 */
TEST(VectorUtil, cross)
{
    // Two vector we'd like to cross
    Vec2 x(1,2);
    Vec2 y(3,4);
    
    floatT res = VectorUtil::cross(x,y);
    ASSERT_FLOAT_EQ(-2, res);
}

/**
 * Tests if the general Polygon setter/getter work
 */
TEST(Polygon, bracketOperator)
{
    Polygon<10> polygon;
    
    // Fill the polygon
    for (int i = 0; i < 10; i++)
    {
        polygon[i][0] = i;
        polygon[i][1] = 2*i;
    }
    
    // Check if this worked
    for (int i = 0; i < 10; i++)
    {
        ASSERT_FLOAT_EQ(polygon[i][0], i);
        ASSERT_FLOAT_EQ(polygon[i][1], i*2);
    }
}

/**
 * Tests if the general Polygon setter/getter work
 */
TEST(Polygon, at)
{
    Polygon<10> polygon;
    
    // Fill the polygon
    for (int i = 0; i < 10; i++)
    {
        polygon.at(i)[0] = i;
        polygon.at(i)[1] = 2*i;
    }
    
    // Check if this worked
    for (int i = 0; i < 10; i++)
    {
        ASSERT_FLOAT_EQ(polygon.at(i)[0], i);
        ASSERT_FLOAT_EQ(polygon.at(i)[1], i*2);
    }
}

/**
 * Tests if the line segment length is computed correctly
 */
TEST(LineSegment, getLength)
{
    // Set up a line segment of length 5
    LineSegment l(Vec2(6, 2), Vec2(2, -1));
    
    ASSERT_FLOAT_EQ(l.getLength(), 5);
}

/**
 * Tests if the line normal is computed correctly.
 */
TEST(LineSegment, getLineNormal)
{
    // Set up a line segment
    LineSegment l(Vec2(1,1), Vec2(0,2));
    
    Vec3 n = l.getLineNormal();
    
    ASSERT_NEAR(n[0], -1./std::sqrt(2), 1e-6);
    ASSERT_NEAR(n[1], -1./std::sqrt(2), 1e-6);
    ASSERT_NEAR(n[2], std::sqrt(2), 1e-6);
}

/**
 * Tests if the distance between a point and a line is calculated correctly.
 */
TEST(LineUtil, calcPointDistance)
{
    // Set up a line segment
    LineSegment l(Vec2(1,1), Vec2(0,2));
    // Set up a point
    Vec2 x(2,2);
    
    floatT res = LineUtil::calcPointDistance(l, x);
    
    ASSERT_NEAR(res, std::sqrt(2), 1e-6);
}

/**
 * Tests if the intersection point between two lines is calculated correctly.
 */
TEST(LineUtil, calcIntersectionPoint)
{
    // Set up two line segments
    LineSegment l1(Vec2(1,1), Vec2(0,2));
    LineSegment l2(Vec2(0,0), Vec2(-1,-1));
    
    // This should be the intersection point
    Vec2 out(1,1);
    
    Vec2 res;
    LineUtil::calcIntersectionPoint(l1, l2, res);
    
    ASSERT_NEAR(res[0], out[0], 1e-6);
    ASSERT_NEAR(res[1], out[1], 1e-6);
}

/**
 * Tests if a homography is applied correctly to a line segment
 */
TEST(LineSegmentUtil, applyHomography)
{
    // Set up a line segment
    LineSegment l1(Vec2(1,1), Vec2(0,2));
    
    // Set up the homography
    cv::Mat H(3,3,CV_64F);
    
    H.at<double>(0,0) = 0.377964473009227;
    H.at<double>(0,1) = 0;
    H.at<double>(0,2) = 0;
    H.at<double>(1,0) = 0;
    H.at<double>(1,1) = 0.377964473009228;
    H.at<double>(1,2) = 0;
    H.at<double>(2,0) = -0.377964473009227;
    H.at<double>(2,1) = 0;
    H.at<double>(2,2) = 0.755928946018455;
    
    // The resulting line segment
    LineSegment out(Vec2(1,1), Vec2(0,1));
    LineSegment res;
    
    LineSegmentUtil::applyHomography(H, l1, res);
    
    ASSERT_NEAR(res[0][0], out[0][0], 1e-6);
    ASSERT_NEAR(res[0][1], out[0][1], 1e-6);
    ASSERT_NEAR(res[1][0], out[1][0], 1e-6);
    ASSERT_NEAR(res[1][1], out[1][1], 1e-6);
}

/**
 * Tests if the distance between a point and a line segment is calculated correctly
 */
TEST(LineSegmentUtil, calcPointDistance_NoEndPoint)
{
    // Set up a line segment
    LineSegment l(Vec2(2,0), Vec2(0,2));
    Vec2 x(2,2);
    
    ASSERT_NEAR(LineSegmentUtil::calcPointDistance(l, x), std::sqrt(2), 1e-6);
}

TEST(LineSegmentUtil, calcPointDistance_endPoint1)
{
    // Set up a line segment
    LineSegment l(Vec2(2,0), Vec2(0,2));
    Vec2 x(3,0);
    
    ASSERT_NEAR(LineSegmentUtil::calcPointDistance(l, x), 1, 1e-6);
}

TEST(LineSegmentUtil, calcPointDistance_endPoint2)
{
    // Set up a line segment
    LineSegment l(Vec2(2,0), Vec2(0,2));
    Vec2 x(-1,3);
    
    ASSERT_NEAR(LineSegmentUtil::calcPointDistance(l, x), std::sqrt(2), 1e-6);
}

/**
 * Checks if two line segments intersect.
 */
TEST(LineSegmentUtil, doIntersect_parallel)
{
    // Set up parallel line segments
    LineSegment l1(Vec2(-1,2), Vec2(2,-1));
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    
    ASSERT_FALSE(LineSegmentUtil::doIntersect(l1, l2));
    ASSERT_FALSE(LineSegmentUtil::doIntersect(l2, l1));
}

TEST(LineSegmentUtil, doIntersect_colinear)
{
    // Set up colinear line segments
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    LineSegment l1(Vec2(3,-1), Vec2(4,-2));
    
    ASSERT_FALSE(LineSegmentUtil::doIntersect(l1, l2));
    ASSERT_FALSE(LineSegmentUtil::doIntersect(l2, l1));
}

TEST(LineSegmentUtil, doIntersect_nonCross)
{
    // Set up non crossing line segments
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    LineSegment l1(Vec2(1.1,1.1), Vec2(2,2));
    
    ASSERT_FALSE(LineSegmentUtil::doIntersect(l1, l2));
    ASSERT_FALSE(LineSegmentUtil::doIntersect(l2, l1));
}

TEST(LineSegmentUtil, doIntersect_nonCross2)
{
    // Set up non crossing line segments
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    LineSegment l1(Vec2(0,0), Vec2(0.9,0.9));
    
    ASSERT_FALSE(LineSegmentUtil::doIntersect(l1, l2));
    ASSERT_FALSE(LineSegmentUtil::doIntersect(l2, l1));
}

TEST(LineSegmentUtil, doIntersect_cross)
{
    // Set up intersecting line segments
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    LineSegment l1(Vec2(0,0), Vec2(2,2));
    
    ASSERT_TRUE(LineSegmentUtil::doIntersect(l1, l2));
    ASSERT_TRUE(LineSegmentUtil::doIntersect(l2, l1));
}

/**
 * Tests if two line segments are joined correctly.
 */
TEST(LineSegmentUtil, join)
{
    // Set up colinear line segments
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    LineSegment l1(Vec2(3,-1), Vec2(4,-2));
    LineSegment out(Vec2(0,2), Vec2(4,-2));
    
    LineSegment res;
    LineSegmentUtil::join(l1, l2, res);
    
    ASSERT_NEAR(res[0][0], out[1][0], 1e-6);
    ASSERT_NEAR(res[0][1], out[1][1], 1e-6);
    ASSERT_NEAR(res[1][0], out[0][0], 1e-6);
    ASSERT_NEAR(res[1][1], out[0][1], 1e-6);
}

/**
 * Checks if the distance between two line segments is calculated correctly
 */
TEST(LineSegmentUtil, calcDistance_parallel)
{
    // Set up parallel line segments
    LineSegment l1(Vec2(-1,2), Vec2(2,-1));
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l1, l2), 1/std::sqrt(2), 1e-6);
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l2, l1), 1/std::sqrt(2), 1e-6);
}

TEST(LineSegmentUtil, calcDistance_colinear)
{
    // Set up colinear line segments
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    LineSegment l1(Vec2(3,-1), Vec2(4,-2));
    
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l1, l2), std::sqrt(2), 1e-6);
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l2, l1), std::sqrt(2), 1e-6);
}

TEST(LineSegmentUtil, calcDistance_nonCross)
{
    // Set up non crossing line segments
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    LineSegment l1(Vec2(1.1,1.1), Vec2(2,2));
    
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l1, l2), std::sqrt(0.02), 1e-6);
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l2, l1), std::sqrt(0.02), 1e-6);
}

TEST(LineSegmentUtil, calcDistance_nonCross2)
{
    // Set up non crossing line segments
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    LineSegment l1(Vec2(0,0), Vec2(0.9,0.9));
    
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l1, l2), std::sqrt(0.02), 1e-6);
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l2, l1), std::sqrt(0.02), 1e-6);
}

TEST(LineSegmentUtil, calcDistance_cross)
{
    // Set up intersecting line segments
    LineSegment l2(Vec2(2,0), Vec2(0,2));
    LineSegment l1(Vec2(0,0), Vec2(2,2));
    
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l1, l2), 0, 1e-6);
    ASSERT_NEAR(LineSegmentUtil::calcDistance(l2, l1), 0, 1e-6);
}

/**
 * Tests if a rectangle recognizes if it's oriented incorrectly
 */
TEST(Rectangle, isOrientedCorrectly_bad1)
{
    Rectangle r(Vec2(1,1), Vec2(0,1), Vec2(1,0), Vec2(0,0));
    
    ASSERT_FALSE(r.isOrientedCorrectly());
}

TEST(Rectangle, isOrientedCorrectly_bad2)
{
    Rectangle r(Vec2(0,1), Vec2(1,1), Vec2(1,0), Vec2(0,0));
    
    ASSERT_FALSE(r.isOrientedCorrectly());
}

TEST(Rectangle, isOrientedCorrectly_bad3)
{
    Rectangle r(Vec2(1,0), Vec2(0,1), Vec2(1,1), Vec2(0,0));
    
    ASSERT_FALSE(r.isOrientedCorrectly());
}

TEST(Rectangle, isOrientedCorrectly_good)
{
    Rectangle r(Vec2(0,0), Vec2(1,0), Vec2(0,1), Vec2(1,1));
    
    ASSERT_TRUE(r.isOrientedCorrectly());
}

/**
 * Tests if a rectangle rotates correctly.
 */
TEST(Rectangle, rotate)
{
    Rectangle r(Vec2(0,0), Vec2(1,0), Vec2(0,1), Vec2(1,1));
    Rectangle r2 = r;
    
    r2.rotate();
    ASSERT_LE(cv::norm(r2[1], r[0]), 1e-10);
    ASSERT_LE(cv::norm(r2[2], r[1]), 1e-10);
    ASSERT_LE(cv::norm(r2[3], r[2]), 1e-10);
    ASSERT_LE(cv::norm(r2[0], r[3]), 1e-10);
}

/**
 * Tests if a rectangle is normalized correctly
 */
TEST(Rectangle, normalize)
{
    Rectangle r(Vec2(1,0), Vec2(0,0), Vec2(0,1), Vec2(1,1));
    Rectangle r2 = r;
    
    r2.normalize();
    ASSERT_LE(cv::norm(r2[0], r[1]), 1e-10);
    ASSERT_LE(cv::norm(r2[1], r[0]), 1e-10);
    ASSERT_LE(cv::norm(r2[2], r[3]), 1e-10);
    ASSERT_LE(cv::norm(r2[3], r[2]), 1e-10);
}

/**
 * Tests if the width is calculated correctly.
 */
TEST(Rectangle, getWidth)
{
    Rectangle r(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    
    ASSERT_FLOAT_EQ(r.getWidth(), 2);
}

/**
 * Tests if the height is calculated correctly.
 */
TEST(Rectangle, getHeight)
{
    Rectangle r(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    
    ASSERT_FLOAT_EQ(r.getHeight(), 2);
}

/**
 * Tests if the area is calculated correctly.
 */
TEST(Rectangle, getArea)
{
    Rectangle r(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    
    ASSERT_FLOAT_EQ(r.getArea(), 4);
}

/**
 * Checks if a rectangle is converted correctly to an openCV rectangle.
 */
TEST(RectangleUtil, convertToOpenCV)
{
    Rectangle r(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    cv::Rect_<floatT> res;
    RectangleUtil::convertToOpenCV(r, res);
    
    ASSERT_FLOAT_EQ(r.getWidth(), res.width);
    ASSERT_FLOAT_EQ(r.getHeight(), res.height);
    ASSERT_FLOAT_EQ(r[0][0], res.x);
    ASSERT_FLOAT_EQ(r[0][1], res.y);
}

/**
 * Checks if a rectangle is converted correctly from an openCV rectangle.
 */
TEST(RectangleUtil, convertFromOpenCV)
{
    cv::Rect_<floatT> r;
    r.x = 0;
    r.y = 0;
    r.width = 2;
    r.height = 2;
    Rectangle res;
    RectangleUtil::convertFromOpenCV(r, res);
    
    ASSERT_FLOAT_EQ(res.getWidth(), r.width);
    ASSERT_FLOAT_EQ(res.getHeight(), r.height);
    ASSERT_FLOAT_EQ(res[0][0], r.x);
    ASSERT_FLOAT_EQ(res[0][1], r.y);
}

/**
 * Checks if the union of two rectangles is calculated correctly
 */
TEST(RectangleUtil, calcUnion_contained)
{
    Rectangle r1(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    Rectangle r2(Vec2(0,0), Vec2(1,0), Vec2(1,1), Vec2(0,1));
    
    Rectangle res;
    RectangleUtil::calcUnion(r1, r2, res);
    
    ASSERT_LE(cv::norm(r1[0], res[0]), 1e-10);
    ASSERT_LE(cv::norm(r1[1], res[1]), 1e-10);
    ASSERT_LE(cv::norm(r1[2], res[2]), 1e-10);
    ASSERT_LE(cv::norm(r1[3], res[3]), 1e-10);
}

TEST(RectangleUtil, calcUnion_contained2)
{
    Rectangle r1(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    Rectangle r2(Vec2(0,0), Vec2(1,0), Vec2(1,1), Vec2(0,1));
    
    Rectangle res;
    RectangleUtil::calcUnion(r2, r1, res);
    
    ASSERT_LE(cv::norm(r1[0], res[0]), 1e-10);
    ASSERT_LE(cv::norm(r1[1], res[1]), 1e-10);
    ASSERT_LE(cv::norm(r1[2], res[2]), 1e-10);
    ASSERT_LE(cv::norm(r1[3], res[3]), 1e-10);
}

TEST(RectangleUtil, calcUnion_notContained)
{
    Rectangle r1(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    Rectangle r2(Vec2(0,0), Vec2(3,0), Vec2(3,1), Vec2(0,1));
    Rectangle out(Vec2(0,0), Vec2(3,0), Vec2(3,2), Vec2(0,2));

    Rectangle res;
    RectangleUtil::calcUnion(r2, r1, res);
    
    ASSERT_LE(cv::norm(out[0], res[0]), 1e-10);
    ASSERT_LE(cv::norm(out[1], res[1]), 1e-10);
    ASSERT_LE(cv::norm(out[2], res[2]), 1e-10);
    ASSERT_LE(cv::norm(out[3], res[3]), 1e-10);
}

TEST(RectangleUtil, calcUnion_notContained2)
{
    Rectangle r1(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    Rectangle r2(Vec2(0,0), Vec2(3,0), Vec2(3,1), Vec2(0,1));
    Rectangle out(Vec2(0,0), Vec2(3,0), Vec2(3,2), Vec2(0,2));

    Rectangle res;
    RectangleUtil::calcUnion(r1, r2, res);
    
    ASSERT_LE(cv::norm(out[0], res[0]), 1e-10);
    ASSERT_LE(cv::norm(out[1], res[1]), 1e-10);
    ASSERT_LE(cv::norm(out[2], res[2]), 1e-10);
    ASSERT_LE(cv::norm(out[3], res[3]), 1e-10);
}

/**
 * Checks if the intersection is computed correctly.
 */
TEST(RectangleUtil, calcIntersection_contained)
{
    Rectangle r1(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    Rectangle r2(Vec2(0,0), Vec2(1,0), Vec2(1,1), Vec2(0,1));
    
    Rectangle res;
    RectangleUtil::calcIntersection(r1, r2, res);
    
    ASSERT_LE(cv::norm(r2[0], res[0]), 1e-10);
    ASSERT_LE(cv::norm(r2[1], res[1]), 1e-10);
    ASSERT_LE(cv::norm(r2[2], res[2]), 1e-10);
    ASSERT_LE(cv::norm(r2[3], res[3]), 1e-10);
}

TEST(RectangleUtil, calcIntersection_contained2)
{
    Rectangle r1(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    Rectangle r2(Vec2(0,0), Vec2(1,0), Vec2(1,1), Vec2(0,1));
    
    Rectangle res;
    RectangleUtil::calcIntersection(r2, r1, res);
    
    ASSERT_LE(cv::norm(r2[0], res[0]), 1e-10);
    ASSERT_LE(cv::norm(r2[1], res[1]), 1e-10);
    ASSERT_LE(cv::norm(r2[2], res[2]), 1e-10);
    ASSERT_LE(cv::norm(r2[3], res[3]), 1e-10);
}

TEST(RectangleUtil, calcIntersection_notContained)
{
    Rectangle r1(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    Rectangle r2(Vec2(0,0), Vec2(3,0), Vec2(3,1), Vec2(0,1));
    Rectangle out(Vec2(0,0), Vec2(2,0), Vec2(2,1), Vec2(0,1));

    Rectangle res;
    RectangleUtil::calcIntersection(r2, r1, res);
    
    ASSERT_LE(cv::norm(out[0], res[0]), 1e-10);
    ASSERT_LE(cv::norm(out[1], res[1]), 1e-10);
    ASSERT_LE(cv::norm(out[2], res[2]), 1e-10);
    ASSERT_LE(cv::norm(out[3], res[3]), 1e-10);
}

TEST(RectangleUtil, calcIntersection_notContained2)
{
    Rectangle r1(Vec2(0,0), Vec2(2,0), Vec2(2,2), Vec2(0,2));
    Rectangle r2(Vec2(0,0), Vec2(3,0), Vec2(3,1), Vec2(0,1));
    Rectangle out(Vec2(0,0), Vec2(2,0), Vec2(2,1), Vec2(0,1));

    Rectangle res;
    RectangleUtil::calcIntersection(r1, r2, res);
    
    ASSERT_LE(cv::norm(out[0], res[0]), 1e-10);
    ASSERT_LE(cv::norm(out[1], res[1]), 1e-10);
    ASSERT_LE(cv::norm(out[2], res[2]), 1e-10);
    ASSERT_LE(cv::norm(out[3], res[3]), 1e-10);
}


