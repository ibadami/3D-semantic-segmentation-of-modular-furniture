#ifndef LIBF_TOOLS_H
#define LIBF_TOOLS_H

#include <vector>
#include <boost/filesystem.hpp>
#include <memory>
#include <opencv2/opencv.hpp>

#include "data.h"
#include "classifier.h"
#include "learning.h"

/**
 * This file contains some function that can evaluate the performance of a
 * learned classifier. 
 */

namespace libf {
    class Estimator;
    class Gaussian;
    
    /**
     * Computes the accuracy on the data set.
     */
    class AccuracyTool {
    public:
        /**
         * Returns the accuracy
         */
        float measure(const AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage) const;
        
        /**
         * Prints the accuracy
         */
        void print(float accuracy) const;
        
        /**
         * Prints and measures the accuracy. 
         */
        void measureAndPrint(const AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage) const;
    };
    
    /**
     * Computes the confusion matrix on the data set.
     */
    class ConfusionMatrixTool {
    public:
        /**
         * Returns the accuracy
         */
        void measure(AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage, std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints the accuracy
         */
        void print(const std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints and measures the accuracy. 
         */
        void measureAndPrint(AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage) const;
    };
    
    /**
     * Measures the correlation between the the trees of an ensemble by using
     * the hamming distance on their results. 
     */
    class CorrelationTool {
    public:
        /**
         * Returns the correlation
         */
        void measure(typename RandomForest<AbstractClassifier>::ptr classifier, AbstractDataStorage::ptr storage, std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints the correlation
         */
        void print(const std::vector< std::vector<float> > & result) const;
        
        /**
         * Prints and measures the correlation. 
         */
        void measureAndPrint(typename RandomForest<AbstractClassifier>::ptr classifier, AbstractDataStorage::ptr storage) const;
    };
    
    /**
     * Reports the variable importance computed during training.
     */
#if 0
    // TODO: Update importance calculation
    class VariableImportanceTool {
    public:
        /**
         * Returns the variable importance (simple wrapper around 
         * getImportance).
         */
        template <class S, class T>
        void measure(RandomForestLearner<S, T> * learner, std::vector<float> & result) const
        {
            std::vector<float> importance = learner->getImportance();
            result = std::vector<float>(importance.begin(), importance.end());
        }
        
        /**
         * Prints the variable importance
         */
        void print(const std::vector<float> & result) const;
        
        /**
         * Retrieves (measures) and prints the variable importance
         */
        template <class S, class T>
        void measureAndPrint(RandomForestLearner<S, T>* learner) const
        {
            print(learner->getImportance());
        }
    };
    
    /**
     * Backprojects the variable importance onto a square image with the given
     * given width/height.
     */
    class PixelImportanceTool : public VariableImportanceTool {
    public:
        
        /**
         * Retrieves variable importance and stores an image visualizing variable
         * importance where the image has size rows x rows.
         */
        template <class S, class T>
        void measureAndSave(RandomForestLearner<S,T> learner, boost::filesystem::path file, int rows) const
        {
            const std::vector<float> result = learner->getImportance();
            const int F = static_cast<int>(result.size());

            cv::Mat image(rows, rows, CV_8UC3, cv::Scalar(255, 255, 255));

            float max = 0;
            for (int i = 0; i < F; i++)
            {
                if (result[i] > max)
                {
                    max = result[i];
                }
            }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < rows; j++)
                {
                    if (result[j + rows*i] > 0) 
                    {
                        image.at<cv::Vec3b>(i, j) = cv::Vec3b(0, (unsigned char) (result[j + rows*i]/max*255), 255);
                    }
                }
            }

            cv::imwrite(file.string(), image);
        }
    };
#endif 
    
    
    /*
     * Used to assess the quality of a density estimation given the true
     * Gaussian mixture density.
     */
    class GaussianKullbackLeiblerTool {
    public:
        
        /**
         * Measures the density accuracy using the Kulback-Leibler divergence on a discretegrid.
         */
        float measure(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
        /**
         * Print the Kulback-Leibler divergence.
         */
        void print(float kl);
        
        /**
         * Measure and print the accuracy in terms of the Kulback-Leibler divergence.
         */
        void measureAndPrint(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
    };
    
    class GaussianSquaredErrorTool {
    public:
        
        /**
         * Measures the density accuracy in terms for squared error.
         */
        float measure(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
        /**
         * Print the squared error.
         */
        void print(float se);
        
        /**
         * Measure and print the accuracy in terms of the squared error
         */
        void measureAndPrint(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
                const std::vector<float> & weights, int N);
        
    };
}

#endif