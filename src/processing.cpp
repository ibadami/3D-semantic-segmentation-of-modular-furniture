#include <fstream>
#include <bitset>
#include <random>
#include "parser/processing.h"
#include "parser/util.h"
#include "parser/rjmcmc_sa.h"
#include "gurobi_c++.h"
//#include "parser/coverEnergy.h"
#include <Eigen/Dense>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace parser;

void Processing::computeGradients(const cv::Mat& in, cv::Mat& out, float threshold)
{
    // Check if the image has the right type
    if (in.type() != CV_8UC1)
    {
        throw ParserException("Invalid image type.");
    }

    cv::Mat blurred;
    cv::GaussianBlur(in, blurred, cv::Size(3,3), 0,0, cv::BORDER_DEFAULT);

    cv::Mat gradX, gradY;

    /// Gradient X
    cv::Sobel( blurred, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT );

    /// Gradient Y
    cv::Sobel( blurred, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT );

    out = cv::Mat::zeros(in.rows, in.cols, CV_32FC2);

    // Write the gradient magnitude into the gradient image
    for (int y = 0; y < in.rows; y++)
    {
        for (int x = 0; x < in.cols; x++)
        {
            const float dX = gradX.at<short>(y,x);
            const float dY = gradY.at<short>(y,x);
            out.at<cv::Vec2f>(y,x)[0] = dX;
            out.at<cv::Vec2f>(y,x)[1] = dY;
        }
    }
}

void Processing::computeGradientMagnitudeImage(const cv::Mat & _in, cv::Mat & out, float threshold)
{
    cv::Mat in;
    _in.convertTo(in, CV_8UC1);
    
    // Compute the gradients
    cv::Mat gradients;
    computeGradients(in, gradients, threshold);
    
    // Set up the output image
    out = cv::Mat::zeros(in.rows, in.cols, CV_8UC1);
    
    float maxGradMag = std::sqrt(2) * 255;
    
    for (int w = 0; w < in.cols; w++)
    {
        for (int h = 0; h < in.rows; h++)
        {
            const float temp1 = gradients.at<cv::Vec2f>(h,w)[0];
            const float temp2 = gradients.at<cv::Vec2f>(h,w)[1];
            const float gradMag = std::sqrt(temp1*temp1+temp2*temp2);
            out.at<uchar>(h,w) = std::min(static_cast<uchar>(255), static_cast<uchar>(std::round(gradMag / maxGradMag * 255)));
        }
    }
}

void Processing::computeGradientMagnitudeImageFloat(const cv::Mat & in, cv::Mat & out)
{
    // Compute the gradients
    cv::Mat gradients;
    {
        cv::Mat blurred;
        cv::GaussianBlur(in, blurred, cv::Size(3,3), 0,0, cv::BORDER_DEFAULT);

        cv::Mat gradX, gradY;

        /// Gradient X
        cv::Sobel( blurred, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT );

        /// Gradient Y
        cv::Sobel( blurred, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT );

        gradients = cv::Mat::zeros(in.rows, in.cols, CV_32FC2);

        // Write the gradient magnitude into the gradient image
        for (int y = 0; y < in.rows; y++)
        {
            for (int x = 0; x < in.cols; x++)
            {
                const float dX = gradX.at<short>(y,x);
                const float dY = gradY.at<short>(y,x);
                gradients.at<cv::Vec2f>(y,x)[0] = std::abs(dX);
                gradients.at<cv::Vec2f>(y,x)[1] = std::abs(dY);
            }
        }
    }
    
    // Set up the output image
    out = cv::Mat::zeros(in.rows, in.cols, CV_32FC1);
    
    for (int w = 0; w < in.cols; w++)
    {
        for (int h = 0; h < in.rows; h++)
        {
            const float temp1 = gradients.at<cv::Vec2f>(h,w)[0];
            const float temp2 = gradients.at<cv::Vec2f>(h,w)[1];
            const float gradMag = std::sqrt(temp1*temp1+temp2*temp2);
            out.at<float>(h,w) = gradMag;
        }
    }
}

void Processing::normalizeFloatImageLebesgue(cv::Mat & in)
{
    // Compute the total mass
    float mass = 0;
    for (int x = 0; x < in.cols; x++)
    {
        for (int y = 0; y < in.rows; y++)
        {
            mass += in.at<float>(y,x);
        }
    }
    mass /= in.cols * in.rows;
    if (mass >= 1e-1)
    {
        for (int x = 0; x < in.cols; x++)
        {
            for (int y = 0; y < in.rows; y++)
            {
                in.at<float>(y,x) /= mass;
            }
        }
    }
}

void Processing::computeCannyEdges(const cv::Mat & _in, cv::Mat & out)
{
    cv::Mat in;
    _in.convertTo(in, CV_8UC1);
    
    // Some "magic numbers"
    const float thresholdRatio = 0.75;

    // Get the gradient magnitude image in order to estimate the 
    // threshold values
    cv::Mat gradMag;
    computeGradientMagnitudeImage(in, gradMag, 1.0f);

    double high_thres = cv::threshold( gradMag, gradMag, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU );

    high_thres = std::min(200., high_thres);
    
    cv::GaussianBlur(in, in, cv::Size(5,5), 1);
    // Perform canny edge detection
    cv::Canny(  in, 
                out, 
                high_thres * thresholdRatio,
                high_thres*1.3,
                3,
                true);
}

void Processing::computeDistanceTransform(const cv::Mat & in, cv::Mat & out)
{
    cv::Mat invert;
    invert = cv::Scalar::all(255) - in;
    cv::distanceTransform(      invert, 
                                out, 
                                CV_DIST_C, 
                                CV_DIST_MASK_PRECISE);
}

void Processing::computeHomography(const Rectangle & rect, const Rectangle & target, cv::Mat & homography)
{
    cv::Mat hom = cv::Mat::zeros(3,3,CV_32FC1);
    
    // Set up the source/destination matrices
    cv::Mat source(4,1,CV_32FC2);
    cv::Mat dest(4,1,CV_32FC2);

    source.at<cv::Vec2f>(0,0) = rect[0];
    source.at<cv::Vec2f>(1,0) = rect[1];
    source.at<cv::Vec2f>(2,0) = rect[2];
    source.at<cv::Vec2f>(3,0) = rect[3];

    dest.at<cv::Vec2f>(0,0) = target[0];
    dest.at<cv::Vec2f>(1,0) = target[1];
    dest.at<cv::Vec2f>(2,0) = target[2];
    dest.at<cv::Vec2f>(3,0) = target[3];
    
    hom = cv::findHomography(source, dest, 0, 4);
    hom.copyTo(homography);
}

void Processing::computeRectifiedRegionOfInterest(const Rectangle & region, int size, Rectangle & result)
{
    float width = region.maxX() - region.minX();
    float height = region.maxY() - region.minY();

    // Set up the destination rectangle
    float alpha;
    if (height > width)
    {
        alpha = size/height;
    }
    else
    {
        alpha = size/width;
    }
    height *= alpha;
    width *= alpha;

    int iHeight = static_cast<int>(std::round(height));
    int iWidth = static_cast<int>(std::round(width));

    result[0][0] = 0;
    result[0][1] = 0;
    result[1][0] = iWidth - 1;
    result[1][1] = 0;
    result[2][0] = iWidth - 1;
    result[2][1] = iHeight - 1;
    result[3][0] = 0;
    result[3][1] = iHeight - 1;
}

void Processing::warpImage(const cv::Mat & in, const Rectangle & rectIn, cv::Mat & out, const Rectangle & rectOut)
{
    // Compute the transformation
    cv::Mat homography;
    computeHomography(rectOut, rectIn, homography);

    // Set up the output image
    out = cv::Mat::zeros(rectOut.maxY() + 1, rectOut.maxX() + 1, CV_8UC3);

    // Plot the mask
    cv::Mat mask = cv::Mat::zeros(rectOut.maxY() + 1, rectOut.maxX() + 1, CV_8UC1);
    PlotUtil::plotRectangleFill(mask, rectOut, 255);
    
    // Warp each pixel
    for (int w = 0; w < out.cols; w++)
    {
        for (int h = 0; h < out.rows; h++)
        {
            // Only warp the pixel if it's inside the rectangle
            Vec2 point;
            point[0] = w;
            point[1] = h;

            if (mask.at<uchar>(h,w) != 0)
            {
                // Warp the point
                Vec2 warpedPoint;
                VectorUtil::applyHomography(homography, point, warpedPoint);
                
                // Extract the image point at the warped location
                out.at<cv::Vec3b>(h,w) = getSubpixel(in, warpedPoint);
            }
        }
    }
}

void Processing::warpImageGaussian(const cv::Mat & in, const Rectangle & rectIn, cv::Mat & out, const Rectangle & rectOut, float bandwidth)
{
    // Compute the transformation
    cv::Mat homography;
    computeHomography(rectOut, rectIn, homography);

    // Set up the output image
    out = cv::Mat::zeros(rectOut.maxY() + 1, rectOut.maxX() + 1, CV_32F);

    // Plot the mask
    cv::Mat mask = cv::Mat::zeros(rectOut.maxY() + 1, rectOut.maxX() + 1, CV_8UC1);
    PlotUtil::plotRectangleFill(mask, rectOut, 255);
    
    // Warp each pixel
    for (int w = 0; w < out.cols; w++)
    {
        for (int h = 0; h < out.rows; h++)
        {
            // Only warp the pixel if it's inside the rectangle
            Vec2 point;
            point[0] = w;
            point[1] = h;

            if (mask.at<uchar>(h,w) != 0)
            {
                // Warp the point
                Vec2 warpedPoint;
                VectorUtil::applyHomography(homography, point, warpedPoint);
                
                // Extract the image point at the warped location
                out.at<float>(h,w) = getSubpixelGaussian(in, warpedPoint, bandwidth);
            }
        }
    }
}

float Processing::getSubpixelGaussian(const cv::Mat & in, const Vec2 & point, float bandwidth)
{
    // Average over all pixels
    float result = 0.0f;
    for (int x = std::max(0, static_cast<int>(point[0] - 5)); x <=  std::min(in.cols - 1, static_cast<int>(point[0] + 5)); x++)
    {
        for (int y = std::max(0, static_cast<int>(point[1] - 5)); y <=  std::min(in.rows - 1, static_cast<int>(point[1] + 5)); y++)
        {
            Vec2 p; p[0] = x; p[1] = y;
            const float distance = cv::norm(point, p);
            
            result += in.at<float>(y,x) * std::exp(- distance*distance/2/bandwidth/bandwidth);
        }
    }
    return result / in.cols / in.rows;
}

void Processing::rectifyRegion(const cv::Mat & in, const Rectangle & region, int size, cv::Mat & out)
{
    // Find the size of the output image
    Rectangle destination;
    computeRectifiedRegionOfInterest(region, size, destination);

    // Warp the image
    warpImage(in, region, out, destination);
}

cv::Vec3b Processing::getSubpixel(const cv::Mat & image, const Vec2 & point)
{
    // Check if the point lies within the image
    //assert (point[0] >= 0 && point[1] >= 0 && point[0] < in.cols || point[1] >= in.rows);

    cv::Vec3b res;
    
    const int x = (int)point[1];
    const int y = (int)point[0];
    const float dx = point[1] - x;
    const float dy = point[0] - y;
    const int xp1 = std::min(x+1, image.rows-1);
    const int yp1 = std::min(y+1, image.cols-1);
    
    for (int channel = 0; channel < 3; channel++)
    {

        float y1 = image.at<cv::Vec3b>(x, y)[channel] + dy *(image.at<cv::Vec3b>(x, yp1)[channel] - image.at<cv::Vec3b>(x, y)[channel]);
        float y2 = image.at<cv::Vec3b>(xp1, y)[channel] + dy *(image.at<cv::Vec3b>(xp1, yp1)[channel] - image.at<cv::Vec3b>(xp1, y)[channel]);

        // Interpolate the x coordinate
        res[channel] = static_cast<uchar>(y1 + dx*(y2-y1));
    }
    return res;
}


void Processing::computeRectangleFilterFeatures(const cv::Mat & image, const Rectangle & r, libf::DataPoint & point)
{
    if (point.rows() != 4)
    {
        point.resize(4);
    }
    
    point(0) = r.getWidth();
    point(1) = r.getHeight();
    point(2) = r.getWidth()/static_cast<float>(image.cols);
    point(3) = r.getHeight()/static_cast<float>(image.rows);
}

void Processing::thinBinary(const cv::Mat & inputarray, cv::Mat & outputarray)
{
    bool bDone = false;
    int rows = inputarray.rows;
    int cols = inputarray.cols;

    inputarray.copyTo(outputarray);

    /// pad source
    cv::Mat p_enlarged_src = cv::Mat(rows + 2, cols + 2, CV_8UC1);
    for(int i = 0; i < (rows+2); i++) {
        p_enlarged_src.at<uchar>(i, 0) = 0;
        p_enlarged_src.at<uchar>( i, cols+1) = 0;
    }
    for(int j = 0; j < (cols+2); j++) {
            p_enlarged_src.at<uchar>(0, j) = 0;
            p_enlarged_src.at<uchar>(rows+1, j) = 0;
    }
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            p_enlarged_src.at<uchar>( i+1, j+1) = inputarray.at<uchar>(i,j);
        }
    }

    /// start to thin
    cv::Mat p_thinMat1 = cv::Mat::zeros(rows + 2, cols + 2, CV_8UC1);
    cv::Mat p_thinMat2 = cv::Mat::zeros(rows + 2, cols + 2, CV_8UC1);
    cv::Mat p_cmp = cv::Mat::zeros(rows + 2, cols + 2, CV_8UC1);

    while (bDone != true) {
            /// sub-iteration 1
            ThinSubiteration1(p_enlarged_src, p_thinMat1);
            /// sub-iteration 2
            ThinSubiteration2(p_thinMat1, p_thinMat2);
            /// compare
            cv::compare(p_enlarged_src, p_thinMat2, p_cmp, CV_CMP_EQ);
            /// check
            int num_non_zero = cv::countNonZero(p_cmp);
            if(num_non_zero == (rows + 2) * (cols + 2)) {
                    bDone = true;
            }
            /// copy
            p_thinMat2.copyTo(p_enlarged_src);
    }
    // copy result
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            outputarray.at<uchar>( i, j) = p_enlarged_src.at<uchar>( i+1, j+1);
        }
    }
}        

void Processing::ThinSubiteration1(const cv::Mat & pSrc, cv::Mat & pDst)
{
    int rows = pSrc.rows;
    int cols = pSrc.cols;
    pSrc.copyTo(pDst);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(pSrc.at<uchar>(i, j)  != 0) {
                /// get 8 neighbors
                /// calculate C(p)
                int neighbor0 = (int) pSrc.at<uchar>( i-1, j-1) != 0;
                int neighbor1 = (int) pSrc.at<uchar>( i-1, j) != 0;
                int neighbor2 = (int) pSrc.at<uchar>( i-1, j+1) != 0;
                int neighbor3 = (int) pSrc.at<uchar>( i, j+1) != 0;
                int neighbor4 = (int) pSrc.at<uchar>( i+1, j+1) != 0;
                int neighbor5 = (int) pSrc.at<uchar>( i+1, j) != 0;
                int neighbor6 = (int) pSrc.at<uchar>( i+1, j-1) != 0;
                int neighbor7 = (int) pSrc.at<uchar>( i, j-1) != 0;
                int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                        int(~neighbor3 & ( neighbor4 | neighbor5)) +
                        int(~neighbor5 & ( neighbor6 | neighbor7)) +
                        int(~neighbor7 & ( neighbor0 | neighbor1));
                if(C == 1) {
                    /// calculate N
                    int N1 = int(neighbor0 | neighbor1) +
                            int(neighbor2 | neighbor3) +
                            int(neighbor4 | neighbor5) +
                            int(neighbor6 | neighbor7);
                    int N2 = int(neighbor1 | neighbor2) +
                            int(neighbor3 | neighbor4) +
                            int(neighbor5 | neighbor6) +
                            int(neighbor7 | neighbor0);
                    int N = std::min(N1,N2);
                    if ((N == 2) || (N == 3)) {
                        /// calculate criteria 3
                        int c3 = ( neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
                        if(c3 == 0) {
                                pDst.at<uchar>( i, j) = 0;
                        }
                    }
                }
            }
        }
    }
}

void Processing::ThinSubiteration2(const cv::Mat & pSrc, cv::Mat & pDst) 
{
    int rows = pSrc.rows;
    int cols = pSrc.cols;
    pSrc.copyTo( pDst);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if (pSrc.at<uchar>( i, j) != 0) {
                    /// get 8 neighbors
                    /// calculate C(p)
                int neighbor0 = (int) pSrc.at<uchar>( i-1, j-1) != 0;
                int neighbor1 = (int) pSrc.at<uchar>( i-1, j) != 0;
                int neighbor2 = (int) pSrc.at<uchar>( i-1, j+1) != 0;
                int neighbor3 = (int) pSrc.at<uchar>( i, j+1) != 0;
                int neighbor4 = (int) pSrc.at<uchar>( i+1, j+1) != 0;
                int neighbor5 = (int) pSrc.at<uchar>( i+1, j) != 0;
                int neighbor6 = (int) pSrc.at<uchar>( i+1, j-1) != 0;
                int neighbor7 = (int) pSrc.at<uchar>( i, j-1) != 0;
                int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                        int(~neighbor3 & ( neighbor4 | neighbor5)) +
                        int(~neighbor5 & ( neighbor6 | neighbor7)) +
                        int(~neighbor7 & ( neighbor0 | neighbor1));
                if(C == 1) {
                    /// calculate N
                    int N1 = int(neighbor0 | neighbor1) +
                            int(neighbor2 | neighbor3) +
                            int(neighbor4 | neighbor5) +
                            int(neighbor6 | neighbor7);
                    int N2 = int(neighbor1 | neighbor2) +
                            int(neighbor3 | neighbor4) +
                            int(neighbor5 | neighbor6) +
                            int(neighbor7 | neighbor0);
                    int N = std::min(N1,N2);
                    if((N == 2) || (N == 3)) {
                            int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
                            if(E == 0) {
                                    pDst.at<uchar>(i, j) = 0;
                            }
                    }
                }
            }
        }
    }
}

float Processing::computeHausdorffDistance( const cv::Mat & A, 
                                            const cv::Mat & dtA, 
                                            const cv::Mat & B, 
                                            const cv::Mat dtB, 
                                            const cv::Mat & mask)
{
    assert(A.rows == dtA.rows);
    assert(A.cols == dtA.cols);
    assert(B.rows == dtB.rows);
    assert(B.cols == dtB.cols);
    assert(A.rows == B.rows);
    assert(mask.rows == A.rows);
    assert(mask.cols == A.cols);
    
    int R_A = 0;
    int R_B = 0;
    float c_AB = 0;
    float c_BA = 0;
    
    for (int w = 0; w < A.cols; w++)
    {
        for (int h = 0; h < A.rows; h++)
        {
            if (mask.at<uchar>(h,w) != 0)
            {
                if (A.at<uchar>(h,w) != 0)
                {
                    R_A++;
                    c_AB = std::max(c_AB, dtB.at<float>(h,w));
                }
                if (B.at<uchar>(h,w) != 0)
                {
                    R_B++;
                    c_BA = std::max(c_BA, dtA.at<float>(h,w));
                }
            }
        }
    }
    return std::max(c_AB, c_BA);
    return (R_A*c_AB + R_B*c_BA)/R_A/R_B;
}

class CustomCallback : public GRBCallback {
public:
    void callback () {
      try {
        if (where == GRB_CB_POLLING) {
          // Ignore polling callback
        } else if (where == GRB_CB_PRESOLVE) {
        } else if (where == GRB_CB_SIMPLEX) {
        } else if (where == GRB_CB_MIP) {
          // General MIP callback
          double objbst = getDoubleInfo(GRB_CB_MIP_OBJBST);
          double time = getDoubleInfo(GRB_CB_RUNTIME);
          if (objbst > 0.9 || time > 25.0)
          {
            abort();
          }
        }
      } catch (GRBException e) {
        std::cout << "Error number: " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
      } catch (...) {
        std::cout << "Error during callback" << std::endl;
      }
    }
};

int Processing::computeRectanglePacking(const std::vector<Rectangle> & rectangles, std::vector<Rectangle> & packing)
{
    // Set up the matrix
    const int N = static_cast<int>(rectangles.size());
    
    // A captures which rectangles overlap
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(N,N);
    // B captures the overlap between rectangles
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(N,N);
    // C is the combined optimization matrix
    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(N,N);
    // a captures the area of each rectangle
    Eigen::VectorXf a = Eigen::VectorXf::Zero(N);

    float maxArea = 0;
    for (int n = 0; n < N; n++)
    {
        const Rectangle & r = rectangles[n];
        maxArea = std::max(maxArea, rectangles[n].getArea());
        a(n) = r.getArea();

        for (int m = n+1; m < N; m++)
        {
            const Rectangle & p = rectangles[m];
            Rectangle i,u;
            RectangleUtil::calcIntersection(p,r,i);
            
            const float iou = i.getArea()/std::min(r.getArea(), p.getArea());
            B(n,m) = i.getArea();
            if (iou > 0.3)
            {
                A(n,m) = 1;
                A(m,n) = 1;
            }
        }
    }
    C = 10*A;
    a /= maxArea;

    int packingNumber = 1;
    std::vector<int> packingState;
    while (true)
    {
        float cover = 0;
        if (packingNumber > N)
        {
            packingNumber--;
            break;
        }
        
        // Prepare the optimization problem
        try {
            GRBEnv env = GRBEnv();
            GRBModel model = GRBModel(env);
            // Create one variable for each part
            GRBVar* variables = model.addVars(N, GRB_BINARY);
            
            // Integrate new variables
            model.update();
            
            // Set objective: maximize the covered area and minimize the overlap 
            // by using a high penalty on the overlap

            GRBQuadExpr obj = 0.0;
            // Add the linear terms
            for (int n = 0; n < N; n++)
            {
                obj += a(n)*variables[n];
            }

            for (int n = 0; n < N; n++)
            {
                for (int m = 0; m < N; m++)
                {
                    if (n == m) continue;
                    
                    if (std::abs(C(n,m)) > 1e-4)
                    {
                        obj += -variables[n]*variables[m]*C(n,m);
                    }
                }
            }
            model.setObjective(obj, GRB_MAXIMIZE);

            // Add the constraint
            GRBLinExpr con = 0.0;
            for (int n = 0; n < N; n++)
            {
                con += variables[n];
            }
            model.addConstr(con, GRB_EQUAL, packingNumber);

            model.getEnv().set(GRB_IntParam_Threads, 4);
            model.getEnv().set(GRB_IntParam_OutputFlag, 0);
            CustomCallback cb;
            model.setCallback(&cb);

            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_int_distribution<int> d(0,1000000);

            // Optimize model
            int counter = 0;
            cover = 0;
            while (cover < 0.9f && counter < 5)
            {
                model.reset();
                model.getEnv().set(GRB_IntParam_Seed, d(g));
                model.optimize();
                cover = model.getObjective().getValue();
                counter++;
            }

            if (cover >= 0.9f)
            {
                packingState.erase(packingState.begin(), packingState.end());

                for (int n = 0; n < N; n++)
                {
                    if (variables[n].get(GRB_DoubleAttr_X) > 0.5)
                    {
                        packingState.push_back(n);
                    }
                }
                packingNumber++;
            }
            else
            {
                packingNumber--;
                break;
            }
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
    for (size_t n = 0; n < packingState.size(); n++)
    {
        packing.push_back(rectangles[packingState[n]]);
    }
    return packingNumber;
}

int Processing::computeRectangleCover(float roiSize, const std::vector<Rectangle> & rectangles, std::vector<Rectangle> & packing)
{
    // Set up the matrix
    const int N = static_cast<int>(rectangles.size());
    
    // A captures which rectangles overlap
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(N,N);
    // B captures the overlap between rectangles
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(N,N);
    // C is the combined optimization matrix
    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(N,N);
    // a captures the area of each rectangle
    Eigen::VectorXf a = Eigen::VectorXf::Zero(N);

    for (int n = 0; n < N; n++)
    {
        const Rectangle & r = rectangles[n];
        a(n) = r.getArea()/roiSize;

        for (int m = n+1; m < N; m++)
        {
            const Rectangle & p = rectangles[m];
            Rectangle i,u;
            RectangleUtil::calcIntersection(p,r,i);
            
            const float iou = i.getArea()/std::min(r.getArea(), p.getArea());
            B(n,m) = i.getArea()/roiSize;
            if (iou > 0.2)
            {
                A(n,m) = 1;
                A(m,n) = 1;
            }
        }
    }
    C = B;
    int packingNumber = 1;
    std::vector<int> packingState;
    while (true)
    {
        float cover = 0;

        // Prepare the optimization problem
        try {
            GRBEnv env = GRBEnv();
            GRBModel model = GRBModel(env);
            // Create one variable for each part
            GRBVar* variables = model.addVars(N, GRB_BINARY);
            
            // Integrate new variables
            model.update();
            
            // Set objective: maximize the covered area and minimize the overlap 
            // by using a high penalty on the overlap

            GRBQuadExpr obj = 0.0;
            // Add the linear terms
            for (int n = 0; n < N; n++)
            {
                obj += a(n)*variables[n];
            }

            for (int n = 0; n < N; n++)
            {
                for (int m = 0; m < N; m++)
                {
                    if (std::abs(C(n,m)) > 1e-4)
                    {
                        obj += -variables[n]*variables[m]*C(n,m);
                    }
                }
            }
            model.setObjective(obj, GRB_MAXIMIZE);

            // Add the constraint
            GRBLinExpr con = 0.0;
            for (int n = 0; n < N; n++)
            {
                con += variables[n];
            }
            model.addConstr(con, GRB_EQUAL, packingNumber);

            model.getEnv().set(GRB_IntParam_Threads, 4);
            model.getEnv().set(GRB_IntParam_OutputFlag, 0);
            CustomCallback cb;
            model.setCallback(&cb);

            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_int_distribution<int> d(0,1000000);

            // Optimize model
            model.getEnv().set(GRB_IntParam_Seed, d(g));
            model.optimize();
            cover = model.getObjective().getValue();
            
            if (cover < 0.9f && packingNumber < N)
            {
                packingState.erase(packingState.begin(), packingState.end());

                for (int n = 0; n < N; n++)
                {
                    if (variables[n].get(GRB_DoubleAttr_X) > 0.5)
                    {
                        packingState.push_back(n);
                    }
                }
                packingNumber++;
            }
            else
            {
                break;
            }
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
    for (size_t n = 0; n < packingState.size(); n++)
    {
        packing.push_back(rectangles[packingState[n]]);
    }
    return packingNumber;
}

void Processing::add1pxBorders(cv::Mat& image)
{
    assert(image.type() == CV_8UC1);
    
    for (int x = 0; x < image.cols; x++)
    {
        image.at<uchar>(0,x) = 255;
        image.at<uchar>(image.rows - 1,x) = 255;
    }
    for (int y = 0; y < image.rows; y++)
    {
        image.at<uchar>(y, 0) = 255;
        image.at<uchar>(y, image.cols - 1) = 255;
    }
}

int Processing::computeRectanglePackingGreedy(const std::vector<Rectangle> & rectangles, std::vector<Rectangle> & packing)
{
    // Set up the matrix
    const int N = static_cast<int>(rectangles.size());
    
    // A captures which rectangles overlap
    Eigen::MatrixXf overlap = Eigen::MatrixXf::Zero(N,N);
    // a captures the area of each rectangle
    Eigen::VectorXf area = Eigen::VectorXf::Zero(N);
    // A captures which rectangles overlap
    Eigen::MatrixXf overlapArea = Eigen::MatrixXf::Zero(N,N);

    float maxArea = 0;
    for (int n = 0; n < N; n++)
    {
        const Rectangle & r = rectangles[n];
        maxArea = std::max(maxArea, rectangles[n].getArea());
        area(n) = r.getArea();

        overlapArea(n,n) = r.getArea();
        
        for (int m = n+1; m < N; m++)
        {
            const Rectangle & p = rectangles[m];
            Rectangle i,u;
            RectangleUtil::calcIntersection(p,r,i);
            
            overlapArea(n,m) = i.getArea();
            overlapArea(m,n) = overlapArea(n,m);
            
            const float iou = i.getArea()/std::min(r.getArea(), p.getArea());
            if (iou > 0.3)
            {
                overlap(n,m) = 1;
                overlap(m,n) = 1;
            }
        }
    }
    
    area /= maxArea;

    // Set up the state space
    std::vector<bool> state(N, true);
    
    // Minimize the overlap
    for (int n = 0; n < N; n++)
    {
        for (int m = n+1; m < N; m++)
        {
            // Skip this rectangle, if it's no longer in the state
            if (!state[m]) continue;

            // If the two rectangles overlap, then remove the larger one
            if (overlap(n,m))
            {
                if (area[n] > area[m])
                {
                    state[n] = false;
                }
                else
                {
                    state[m] = false;
                }
            }
        }
    }
    
    // Compute the covered area and the packing
    float coveredArea = 0;
    for (int n = 0; n < N; n++)
    {
        if (state[n])
        {
            coveredArea += area[n];
            packing.push_back(rectangles[n]);
        }
    }
    
    std::cout << coveredArea << "\n";
    
    return static_cast<int>(packing.size());
}

/*int Processing::computeRectanglePackingMCMC(const std::vector<Rectangle> & rectangles, std::vector<Rectangle> & packing)
{
    // Set up the matrix
    const int N = static_cast<int>(rectangles.size());
    
    // A captures which rectangles overlap
    Eigen::MatrixXi overlap = Eigen::MatrixXi::Zero(N,N);
    // A captures which rectangles overlap
    Eigen::MatrixXf overlapArea = Eigen::MatrixXf::Zero(N,N);
    // a captures the area of each rectangle
    Eigen::VectorXf area = Eigen::VectorXf::Zero(N);

    float maxArea = 0;
    for (int n = 0; n < N; n++)
    {
        const Rectangle & r = rectangles[n];
        maxArea = std::max(maxArea, rectangles[n].getArea());
        area(n) = r.getArea();

        overlap(n,n) = 1;
        
        overlapArea(n,n) = r.getArea();
        
        for (int m = n+1; m < N; m++)
        {
            const Rectangle & p = rectangles[m];
            Rectangle i,u;
            RectangleUtil::calcIntersection(p,r,i);
            
            overlapArea(n,m) = i.getArea();
            overlapArea(m,n) = overlapArea(n,m);
            
            const float iou = i.getArea()/std::min(r.getArea(), p.getArea());
            if (iou > 0.15)
            {
                overlap(n,m) = 1;
                overlap(m,n) = 1;
            }
        }
    }
    
    area /= maxArea;
    overlapArea /= maxArea;

    // Set up the MCMC optimizer
    // We prune the forest using simulated annealing
    SimulatedAnnealing<CoverState, CoverEnergy, GeometricCoolingSchedule> sa;
    sa.setNumInnerLoops(1000);
    sa.setMaxNoUpdateIterations(1000000);

    // Set up the cooling schedule
    GeometricCoolingSchedule schedule;
    schedule.setAlpha(0.995f);
    schedule.setStartTemperature(200);
    schedule.setEndTemperature(1e-4);
    sa.setCoolingSchedule(schedule);

    // Set up the moves
    CoverExchangeMove exchangeMove(N);
    sa.addMove(&exchangeMove, 1.0f);

    // Set up the energy function
    CoverEnergy energy(overlap, area, overlapArea);
    sa.setEnergyFunction(energy);

    // Set up the callback function
    CoverCallback callback;
    sa.addCallback(&callback);

    // Find some initial state
    int l = 1;
    CoverState packingState;
    do {
        CoverState state;
        for (int i = 0; i < l; i++)
        {
            state.push_back(i);
        }

        float energy = sa.optimize(state);
        
        if (energy > 0.1 || l == 25)
        {
            l--;
            break;
        }
        else
        {
            packingState = state;
            l++;
        }
    } while(true);
    
    for (size_t i = 0; i < packingState.size(); i++)
    {
        packing.push_back(rectangles[packingState[i]]);
    }
    
    return static_cast<int>(packing.size());
}

int Processing::computeRectangleCoverMCMC(float roiSize, const std::vector<Rectangle>& rectangles, std::vector<Rectangle>& packing)
{
    // Set up the matrix
    const int N = static_cast<int>(rectangles.size());
    
    // A captures which rectangles overlap
    Eigen::MatrixXi overlap = Eigen::MatrixXi::Zero(N,N);
    // A captures which rectangles overlap
    Eigen::MatrixXf overlapArea = Eigen::MatrixXf::Zero(N,N);
    // a captures the area of each rectangle
    Eigen::VectorXf area = Eigen::VectorXf::Zero(N);

    for (int n = 0; n < N; n++)
    {
        const Rectangle & r = rectangles[n];
        area(n) = r.getArea();

        overlap(n,n) = 1;
        
        overlapArea(n,n) = r.getArea();
        
        for (int m = n+1; m < N; m++)
        {
            const Rectangle & p = rectangles[m];
            Rectangle i,u;
            RectangleUtil::calcIntersection(p,r,i);
            
            overlapArea(n,m) = i.getArea();
            overlapArea(m,n) = overlapArea(n,m);
            
            const float iou = i.getArea()/std::min(r.getArea(), p.getArea());
            if (iou > 0.15)
            {
                overlap(n,m) = 1;
                overlap(m,n) = 1;
            }
        }
    }
    
    area /= roiSize;
    overlapArea /= roiSize;

    // Set up the MCMC optimizer
    // We prune the forest using simulated annealing
    SimulatedAnnealing<CoverState, CoverEnergy, GeometricCoolingSchedule> sa;
    sa.setNumInnerLoops(1000);
    sa.setMaxNoUpdateIterations(1000000);

    // Set up the cooling schedule
    GeometricCoolingSchedule schedule;
    schedule.setAlpha(0.995f);
    schedule.setStartTemperature(200);
    schedule.setEndTemperature(1e-4);
    sa.setCoolingSchedule(schedule);

    // Set up the moves
    CoverExchangeMove exchangeMove(N);
    sa.addMove(&exchangeMove, 1.0f);

    // Set up the energy function
    CoverEnergy energy(overlap, area, overlapArea);
    sa.setEnergyFunction(energy);

    // Set up the callback function
    CoverCallback callback;
    sa.addCallback(&callback);

    // Find some initial state
    int l = 1;
    CoverState packingState;
    do {
        CoverState state;
        for (int i = 0; i < l; i++)
        {
            state.push_back(i);
        }

        float energy = sa.optimize(state);
        
        if (energy < 0.1)
        {
            packingState = state;
            break;
        }
        else
        {
            l++;
        }
    } while(true);
    
    for (size_t i = 0; i < packingState.size(); i++)
    {
        packing.push_back(rectangles[packingState[i]]);
    }
    
    return static_cast<int>(packing.size());
}
*/
