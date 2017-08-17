#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace libf;

Eigen::Matrix2f genCovar(float v0, float v1, float theta)
{
    Eigen::Matrix2f rot = Eigen::Rotation2Df(theta).matrix();
    return rot * Eigen::DiagonalMatrix<float, 2, 2>(v0, v1) * rot.transpose();
}

float randFloat(float min, float max)
{
    return min + static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * (max - min);
}

cv::Mat visualizeSamples(int H, int W, AbstractDataStorage::ptr storage)
{
    cv::Mat image(H, W, CV_8UC1, cv::Scalar(255));

    for (int n = 0; n < storage->getSize(); n++)
    {
        const DataPoint & x = storage->getDataPoint(n);

        int i = std::floor(x(0));
        int j = std::floor(x(1));

        if (i >= 0 && i < H && j >= 0 && j < W)
        {
            image.at<unsigned char>(i, j) = 0;
        }
    }

    return image;
}

cv::Mat visualizeClusters(int H, int W, AbstractDataStorage::ptr storage, 
        const std::vector<int> & labels, int K)
{
    cv::Mat image(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    std::vector<cv::Vec3b> colors(K, cv::Vec3b(0, 0, 0));
    
    for (int n = 0; n < storage->getSize(); n++)
    {
        const DataPoint & x = storage->getDataPoint(n);
        const int cluster = labels[n];
        
        while (colors[cluster][0] == 0 && colors[cluster][1] == 0
                && colors[cluster][2] == 0)
        {
            colors[cluster][0] = std::rand()%256;
            colors[cluster][1] = std::rand()%256;
            colors[cluster][2] = std::rand()%256;
        }
        
        int i = std::floor(x(0));
        int j = std::floor(x(1));

        if (i >= 0 && i < H && j >= 0 && j < W)
        {
            image.at<cv::Vec3b>(i, j) = colors[cluster];
        }
    }

    return image;
}

cv::Mat visualizeClusters(int H, int W, AbstractDataStorage::ptr storage, 
        const cv::Mat & labels, int K)
{
    cv::Mat image(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    std::vector<cv::Vec3b> colors(K, cv::Vec3b(0, 0, 0));
    
    for (int n = 0; n < storage->getSize(); n++)
    {
        const DataPoint & x = storage->getDataPoint(n);
        const int cluster = labels.at<int>(n, 0);
        
        while (colors[cluster][0] == 0 && colors[cluster][1] == 0
                && colors[cluster][2] == 0)
        {
            colors[cluster][0] = std::rand()%256;
            colors[cluster][1] = std::rand()%256;
            colors[cluster][2] = std::rand()%256;
        }
        
        int i = std::floor(x(0));
        int j = std::floor(x(1));

        if (i >= 0 && i < H && j >= 0 && j < W)
        {
            image.at<cv::Vec3b>(i, j) = colors[cluster];
        }
    }

    return image;
}

float calculateError(AbstractDataStorage::ptr storage, const cv::Mat & centers, 
        const cv::Mat & labels, int K)
{
    float error = 0;
    
    for (int n = 0; n < storage->getSize(); n++)
    {
        float distance = 0;
        DataPoint x = storage->getDataPoint(n);
        
        for (int d = 0; d < 2; d++)
        {
            distance += (centers.at<float>(labels.at<int>(n, 0), d) - x(d))
                    * (centers.at<float>(labels.at<int>(n, 0), d) - x(d));
        }
        
        error += distance;
    }
    
    return error;
}

/**
 * Example of kernel density estimation on 2D mixture of Gaussians.
 */
int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("num-components", boost::program_options::value<int>()->default_value(5), "number of Gaussian components")
        ("num-samples", boost::program_options::value<int>()->default_value(1000),"number of samples for training")
        ("seed", boost::program_options::value<int>()->default_value(std::time(0)), "seed used for std::srand");

    boost::program_options::positional_options_description positionals;
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end())
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    // Number of components.
    const int K = parameters["num-components"].as<int>();
    const int H = 400;
    const int W = 400;
    const int T = 100;
    const int M = 25;
    
    // New seed.
    std::srand(parameters["seed"].as<int>());
    
    std::vector<Gaussian> gaussians;
    std::vector<float> weights(K);
    float weights_sum = 0;
    
    for (int m = 0; m < K; m++)
    {
//        do
//        {
//            weights[m] = randFloat(0.1, 1);
//        }
//        while (weights[m] == 0);
        weights[m] = 1./K;
        weights_sum +=weights[m];
        
        float v0 = randFloat(25, H/4);
        float v1 = randFloat(25, W/4);
        float theta = randFloat(0, M_PI);
        
        Eigen::Matrix2f covariance = genCovar(v0, 2*v1, theta);
        
        Eigen::Vector2f mean(2);
        mean(0) = randFloat(H/4, 3*(H/4));
        mean(1) = randFloat(W/4, 3*(W/4));
        
        Gaussian gaussian;
        gaussian.setMean(mean);
        gaussian.setCovariance(covariance);
        
        gaussians.push_back(gaussian);
    }
    
    for (int m = 0; m < K; m++)
    {
        weights[m] /= weights_sum;
    }
    
    // Generate samples.
    const int N = parameters["num-samples"].as<int>();
    DataStorage::ptr storage = DataStorage::Factory::create();
    cv::Mat data(N, 2, CV_32FC1, cv::Scalar(0));
    
    for (int n = 0; n < N; n++)
    {
        int m = std::rand() % K;
        
        DataPoint x;
        gaussians[m].sample(x);
        
        storage->addDataPoint(x);
        
        // Also add the point to cv::Mat:
        data.at<float>(n, 0) = x(0);
        data.at<float>(n, 1) = x(1);
    }
    
    cv::Mat image_samples = visualizeSamples(H, W, storage);
    cv::imwrite("samples.png", image_samples);
    
    KMeans::ptr kmeans = std::make_shared<KMeans>();
    kmeans->setCenterInitMethod(KMeans::CENTERS_PP);
    kmeans->setNumTries(M);
    kmeans->setNumClusters(K);
    kmeans->setNumIterations(T);
    
    std::vector<int> labels;
    DataStorage::ptr centers = DataStorage::Factory::create();
    
    boost::timer timer;
    float error = kmeans->cluster(storage, centers, labels);
    float time = timer.elapsed();
    
    cv::Mat image_clusters = visualizeClusters(H, W, storage, labels, K);
    cv::imwrite("clusters.png", image_clusters);
    
    cv::Mat openCVLabels;
    cv::Mat openCVCenters;
    cv::TermCriteria criteria {cv::TermCriteria::COUNT, T, 0.0};
    
    timer.restart();
    cv::kmeans(data, K, openCVLabels, criteria, M, cv::KMEANS_PP_CENTERS, openCVCenters);
    float openCVTime = timer.elapsed();
    
    cv::Mat image_opencv = visualizeClusters(H, W, storage, openCVLabels, K);
    cv::imwrite("opencv_clusters.png", image_opencv);
    
    float openCVError = calculateError(storage, openCVCenters, openCVLabels, K);
    
    std::cout << "Error: " << error << " (" << time << ")." << std::endl;
    std::cout << "OpenCV Error: " << openCVError << " (" << openCVTime << ")." << std::endl;
    
    return 0;
}
