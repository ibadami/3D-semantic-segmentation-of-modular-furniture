#include "libforest/classifier_tools.h"
#include "libforest/data.h"
#include "libforest/classifier.h"
#include "libforest/estimators.h"
#include "libforest/io.h"
#include "libforest/util.h"

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <opencv2/opencv.hpp>

using namespace libf;

/**
 * Returns the color code for a number between 0 and 1. 
 * if x < t1: Green
 * if t1 <= x <= t2: Yellow
 * if x > t2: red
 */
const char* colorCodeLowToHigh(const float x, const float t1, const float t2)
{
    if (x < t1)
    {
        return LIBF_COLOR_NORMAL;
    }
    else if (t1 <= x && x <= t2)
    {
        return LIBF_COLOR_YELLOW;
    }
    else
    {
        return LIBF_COLOR_RED;
    }
}

/**
 * Returns the color code for a number between 0 and 1. 
 * if x < t1: Red
 * if t1 <= x <= t2: Yellow
 * if x > t2: Green
 */
const char* colorCodeHighToLow(const float x, const float t1, const float t2)
{
    if (x < t1)
    {
        return LIBF_COLOR_RED;
    }
    else if (t1 <= x && x <= t2)
    {
        return LIBF_COLOR_YELLOW;
    }
    else
    {
        return LIBF_COLOR_GREEN;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// AccuracyTool
////////////////////////////////////////////////////////////////////////////////

float AccuracyTool::measure(AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage) const
{
    // Classify all points
    std::vector<int> res;
    classifier->classify(storage, res);

    // Count the misclassified points
    int error = 0;
    for (int i = 0; i < storage->getSize(); i++)
    {
        if (res[i] != storage->getClassLabel(i))
        {
            error++;
        }
    }

    return 1.0f - error/static_cast<float>(storage->getSize());
}

void AccuracyTool::print(float accuracy) const
{
    printf("Accuracy: %2.2f%% (Error: %2.2f%%)\n", accuracy*100, (1-accuracy)*100);
}

void AccuracyTool::measureAndPrint(AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage) const
{
    float accuracy = measure(classifier, storage);
    print(accuracy);
}

////////////////////////////////////////////////////////////////////////////////
/// ConfusionMatrixTool
////////////////////////////////////////////////////////////////////////////////

void ConfusionMatrixTool::measure(AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage, std::vector<std::vector<float> >& result) const
{
    const int C = storage->getClasscount();
    
    // Reset the result
    result.resize(C);
    // Keep track on the number of elements per class
    std::vector<int> classCounts(C);
    
    // Initialize the result
    for (int c = 0; c < C; c++)
    {
        std::vector<float> row(C);
        for (int cc = 0; cc < C; cc++)
        {
            row[cc] = 0;
        }
        result[c] = row;
        classCounts[c] = 0;
    }
    
    // Classify each data point
    std::vector<int> res;
    classifier->classify(storage, res);
    
    // Compute the matrix
    for (int n = 0; n < storage->getSize(); n++)
    {
        const int trueClass = storage->getClassLabel(n);
        const int predictedClass = res[n];
        
        result[trueClass][predictedClass] += 1;
        classCounts[trueClass] += 1;
    }
    
    // Normalize the matrix
    for (int c = 0; c < C; c++)
    {
        for (int cc = 0; cc < C; cc++)
        {
            if (classCounts[c] > 0)
            {
                result[c][cc] /= classCounts[c];
            }
            else
            {
                result[c][cc] = 0;
            }
        }
    }
}

void ConfusionMatrixTool::print(const std::vector<std::vector<float> >& result) const
{
    const int C = static_cast<int>(result.size());
    
    // Print the header
    printf("        |");
    for (int c = 0; c < C; c++)
    {
        printf(" %6d |", c);
    }
    printf("\n");
    for (int c = 0; c < C+1; c++)
    {
        printf("--------|");
    }
    printf("\n");
    for (int c = 0; c < C; c++)
    {
        printf(" %6d |", c);
        for (int cc = 0; cc < C; cc++)
        {
            const char* code;
            if (cc == c)
            {
                code = colorCodeHighToLow(result[c][cc], 1-1.0f/C, 1-1.0f/C/2.0f);
            }
            else
            {
                code = colorCodeLowToHigh(result[c][cc], 1.0f/C/C, 1.0/C/2.0);
            }
            printf(" %s%5.2f%%%s |", code, result[c][cc] * 100, LIBF_COLOR_RESET);
        }
        printf("\n");
        for (int c = 0; c < C+1; c++)
        {
            printf("--------|");
        }
        printf("\n");
    }
}

void ConfusionMatrixTool::measureAndPrint(AbstractClassifier::ptr classifier, AbstractDataStorage::ptr storage) const
{
    std::vector< std::vector<float> > result;
    measure(classifier, storage, result);
    print(result);
}

////////////////////////////////////////////////////////////////////////////////
/// CorrelationTool
////////////////////////////////////////////////////////////////////////////////

void CorrelationTool::measure(typename RandomForest<AbstractClassifier>::ptr forest, AbstractDataStorage::ptr storage, std::vector<std::vector<float> >& result) const
{
    const int T = forest->getSize();
    
    // Reset the result
    result.resize(T);
    // Keep track on the number of elements per class
    std::vector<int> classCounts(T);
    
    // Initialize the result
    for (int t = 0; t < T; t++)
    {
        std::vector<float> row(T);
        for (int tt = 0; tt < T; tt++)
        {
            row[tt] = 0;
        }
        result[t] = row;
        classCounts[t] = 0;
    }
    
    // Compute the classification result for each tree
    std::vector< std::vector<int> > classificationResults(T);
    for (int t = 0; t < T; t++)
    {
        forest->getTree(t)->classify(storage, classificationResults[t]);
    }
    
    // Compute the matrix
    for (int t = 0; t < T; t++)
    {
        for (int tt = t; tt < T; tt++)
        {
            // Compute the normalized hamming distance
            float normalizedDistance = Util::hammingDist(classificationResults[t], classificationResults[tt])/static_cast<float>(storage->getSize());
            result[t][tt] = 1-normalizedDistance;
            result[tt][t] = 1-normalizedDistance;
        }
    }
}

void CorrelationTool::print(const std::vector<std::vector<float> >& result) const
{
    const int C = static_cast<int>(result.size());
    
    // Print the header
    printf("         |");
    for (int c = 0; c < C; c++)
    {
        printf(" %7d |", c);
    }
    printf("\n");
    for (int c = 0; c < C+1; c++)
    {
        printf("---------|");
    }
    printf("\n");
    for (int c = 0; c < C; c++)
    {
        printf(" %7d |", c);
        for (int cc = 0; cc < C; cc++)
        {
            printf(" %6.2f%% |", result[c][cc] * 100);
        }
        printf("\n");
        for (int c = 0; c < C+1; c++)
        {
            printf("---------|");
        }
        printf("\n");
    }
}

void CorrelationTool::measureAndPrint(typename RandomForest<AbstractClassifier>::ptr classifier, AbstractDataStorage::ptr storage) const
{
    std::vector< std::vector<float> > result;
    measure(classifier, storage, result);
    print(result);
}

////////////////////////////////////////////////////////////////////////////////
/// VariableImaportanceTool
////////////////////////////////////////////////////////////////////////////////

#if 0
// TODO: Update importance calculation
void VariableImportanceTool::print(const std::vector<float> & result) const
{
    const int F = static_cast<int>(result.size());
    
    float max = 0;
    for (int f = 0; f < F; f++)
    {
        if (result[f] > max)
        {
            max = result[f];
        }
    }
    
    for (int f = 0; f < F; ++f)
    {
        if (result[f] > 0)
        {
            printf(" %6d | %s%2.6f%s \n", f, colorCodeHighToLow(result[f], 0.1*max, 0.5*max), result[f], LIBF_COLOR_RESET);
        }
        else
        {
            printf(" %6d | %2.6f \n", f, result[f]);
        }
    }
}

#endif 
////////////////////////////////////////////////////////////////////////////////
/// GaussianKullbackLeiblerTool
////////////////////////////////////////////////////////////////////////////////

float GaussianKullbackLeiblerTool::measure(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
        const std::vector<float> & weights, int N)
{
    assert(weights.size() == gaussians.size());
    assert(gaussians.size() > 0);
    
    const int M = weights.size();
    
    float kl = 0;
    for (int n = 0; n < N; n++)
    {
        int m = std::rand()%M;
        DataPoint x;
        gaussians[m].sample(x);

        float p_x = 0;
        float p_x_hat = estimator->estimate(x);
        
        for (m = 0; m < M; m++)
        {
            p_x += weights[m]*gaussians[m].evaluate(x);
        }
        
        if (p_x > 0)
        {
            kl += fastlog2(p_x_hat/p_x);
        }
    }
    
    return kl/N;
}

void GaussianKullbackLeiblerTool::print(float kl)
{
    printf("Divergence: %2.2f\n", kl);
}

void GaussianKullbackLeiblerTool::measureAndPrint(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
        const std::vector<float> & weights, int N)
{
    float kl = measure(estimator, gaussians, weights, N);
    print(kl);
}

////////////////////////////////////////////////////////////////////////////////
/// GaussianSquaredErrorTool
////////////////////////////////////////////////////////////////////////////////

float GaussianSquaredErrorTool::measure(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
        const std::vector<float> & weights, int N)
{
    assert(weights.size() == gaussians.size());
    assert(gaussians.size() > 0);
    
    const int M = weights.size();
    
    float se = 0;
    for (int n = 0; n < N; n++)
    {
        int m = std::rand()%M;
        DataPoint x;
        gaussians[m].sample(x);

        float p_x = 0;
        float p_x_hat = estimator->estimate(x);
        
        for (m = 0; m < M; m++)
        {
            p_x += weights[m]*gaussians[m].evaluate(x);
        }
        
        se += (p_x - p_x_hat)*(p_x - p_x_hat);
    }
    
    return se/N;
}

void GaussianSquaredErrorTool::print(float se)
{
    printf("Error: %2.6f\n", se);
}

void GaussianSquaredErrorTool::measureAndPrint(std::shared_ptr<Estimator> estimator, std::vector<Gaussian> & gaussians,
        const std::vector<float> & weights, int N)
{
    float se = measure(estimator, gaussians, weights, N);
    print(se);
}