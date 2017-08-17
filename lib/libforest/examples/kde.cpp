#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
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

cv::Mat visualizeGaussians(int H, int W, std::vector<Gaussian> gaussians, std::vector<float> weights)
{
    assert(weights.size() == gaussians.size());
    const int M = weights.size();

    cv::Mat image(H, W, CV_32FC1, cv::Scalar(0));
    float p_max = 0;

    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            DataPoint x(2);
            x(0) = i;
            x(1) = j;

            float p_x = 0;

            for (int m = 0; m < M; m++)
            {
                p_x += weights[m]*gaussians[m].evaluate(x);
            }

            if (p_x > p_max)
            {
                p_max = p_x;
            }

            image.at<float>(i, j) = p_x;
        }
    }

    assert(p_max > 0);

    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            image.at<float>(i, j) = image.at<float>(i, j)/p_max * 255;
        }
    }

    return image;
}

cv::Mat visualizeKDE(int H, int W, KernelDensityEstimator::ptr tree)
{    
    cv::Mat image(H, W, CV_32FC1, cv::Scalar(0));
    float p_max = 0;

    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            DataPoint x(2);
            x(0) = i;
            x(1) = j;

            float p_x = tree->estimate(x);

            if (p_x > p_max)
            {
                p_max = p_x;
            }

            image.at<float>(i, j) = p_x;
        }
    }

    assert(p_max > 0);

    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            image.at<float>(i, j) = image.at<float>(i, j)/p_max * 255;
        }
    }

    return image;
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

cv::Mat visualizeKDE(int H, int W, KernelDensityEstimator* kde)
{
    cv::Mat image(H, W, CV_32FC1, cv::Scalar(0));
    float p_max = 0;
    
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            DataPoint x(2);
            x(0) = i;
            x(1) = j;

            float p_x = kde->estimate(x);

            if (p_x > p_max)
            {
                p_max = p_x;
            }
            
            if (i >= 0 && i < H && j >= 0 && j < W)
            {
                image.at<float>(i, j) = p_x;
            }
        }
    }
    
    assert(p_max > 0);

    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            image.at<float>(i, j) = image.at<float>(i, j)/p_max * 255;
        }
    }

    return image;
}
/**
 * Example of kernel density estimation on 2D mixture of Gaussians.
 */
int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("num-components", boost::program_options::value<int>()->default_value(1), "number of Gaussian components")
        ("num-samples", boost::program_options::value<int>()->default_value(1000),"number of samples for training")
        ("bandwidth-selection-method", boost::program_options::value<int>()->default_value(0), "bandwidth selection method")
        ("kernel", boost::program_options::value<int>()->default_value(0), "kernel")
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
    const int M = parameters["num-components"].as<int>();
    const int H = 200;
    const int W = 200;
    
    // New seed.
    std::srand(parameters["seed"].as<int>());
    
    std::vector<Gaussian> gaussians;
    std::vector<float> weights(M);
    float weights_sum = 0;
    
    for (int m = 0; m < M; m++)
    {
//        do
//        {
//            weights[m] = randFloat(0.1, 1);
//        }
//        while (weights[m] == 0);
        weights[m] = 1./M;
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
    
    for (int m = 0; m < M; m++)
    {
        weights[m] /= weights_sum;
    }
    
    cv::Mat image = visualizeGaussians(H, W, gaussians, weights);
    cv::imwrite("gaussians.png", image);
    
    // Generate samples.
    const int N = parameters["num-samples"].as<int>();
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    for (int n = 0; n < N; n++)
    {
        int m = std::rand() % M;
        DataPoint x;
        gaussians[m].sample(x);
        storage->addDataPoint(x);
    }
    
    cv::Mat image_samples = visualizeSamples(H, W, storage);
    cv::imwrite("samples.png", image_samples);    
    
    int bandWidthSelectionMethod = KernelDensityEstimator::BANDWIDTH_RULE_OF_THUMB;
    switch(parameters["bandwidth-selection-method"].as<int>())
    {
        case KernelDensityEstimator::BANDWIDTH_RULE_OF_THUMB:
        case KernelDensityEstimator::BANDWIDTH_RULE_OF_THUMB_INTERQUARTILE:
        case KernelDensityEstimator::BANDWIDTH_MAXIMAL_SMOOTHING_PRINCIPLE:
            bandWidthSelectionMethod = parameters["bandwidth-selection-method"].as<int>();
            break;
        default:
            std::cout << "Invalid bandwidth selection method." << std::endl;
            return 1;
    }
    
    MultivariateKernel* kernel;
    switch (parameters["kernel"].as<int>())
    {
        case 0:
            kernel = new MultivariateGaussianKernel();
            break;
        case 1:
            kernel = new MultivariateEpanechnikovKernel();
            break;
        case 2:
            kernel = new ProductKernel(new GaussianKernel());
            break;
        case 3:
            kernel = new ProductKernel(new EpanechnikovKernel());
            break;
        case 4:
            kernel = new ProductKernel(new BiweightKernel());
            break;
        case 5:
            kernel = new ProductKernel(new TriweightKernel());
            break;
        default:
            std::cout << "Invalid kernel." << std::endl;
            return 1;
    }
    
    KernelDensityEstimator::ptr kde = std::make_shared<KernelDensityEstimator>(storage, kernel);
    kde->selectBandwidth(bandWidthSelectionMethod);
    
    GaussianKullbackLeiblerTool klTool;
    klTool.measureAndPrint(kde, gaussians, weights, 10*N);
    
//    GaussianSquaredErrorTool seTool;
//    seTool.measureAndPrint(&kde, gaussians, weights, 10*N);
    
    cv::Mat image_kde = visualizeKDE(H, W, kde);
    cv::imwrite("kde.png", image_kde);
    
    return 0;
}
