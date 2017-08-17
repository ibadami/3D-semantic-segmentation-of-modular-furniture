#include "libforest/data_tools.h"
#include "libforest/io.h"

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// KMeans
////////////////////////////////////////////////////////////////////////////////

inline float computeDistance(const DataPoint & x, const DataPoint & y)
{
    const int D = x.rows();
    
    float difference = 0;
    float distance = 0;
    
    // Checking whether this is more efficient than using Eigen.
    for (int d = 0; d < D; d++)
    {
        // Saves a single difference!
        difference = x(d) - y(d);
        
        distance += difference*difference;
    }
    
    return distance;
}

DataStorage::ptr KMeans::cluster(DataStorage::ptr storage, std::vector<int> & labels)
{
    BOOST_ASSERT_MSG(storage->getSize() > 0, "Cannot run k-means on an empty data storage.");
    
    const int N = storage->getSize();
    const int D = storage->getDimensionality();
    const int K = numClusters;
    const int T = numIterations;
    const int M = numTries;
    const float epsilon = minDrift;
    
    // To identify the best clustering.
    float error = 1e35;
    
    DataStorage::ptr centers = DataStorage::Factory::create();
    DataStorage::ptr _centers = DataStorage::Factory::create();
    DataStorage::ptr oldCenters = DataStorage::Factory::create();
    
    // Initialize temporal center storages.
    for (int k = 0; k < K; k++)
    {
        centers->addDataPoint(storage->getDataPoint(k));
        oldCenters->addDataPoint(storage->getDataPoint(k));
    }
    
    // To decide which initialization leads to the best result.
    float bestError = 1e35;
    
    // Used to stop iterations when changes are minimal.
    bool stop = false;
    
    // Coutners to update the centers.
    std::vector<int> counters(K, 0);
    
    labels.resize(N);
    
    for (int m = 0; m < M; m++)
    {
        stop = false;
        
        for (int t = 0; t < T; t++)
        {
            if (t == 0)
            {
                switch(centerInitMethod)
                {
                    case CENTERS_RANDOM:
                        initCentersRandom(storage, centers);
                        break;
                    case CENTERS_PP:
                        initCentersPP(storage, centers);
                        break;
                }

                BOOST_ASSERT_MSG(centers->getSize() == K, "Could not initialize the correct number of centers.");
            }
            else
            {
                // Update the centers
                for (int k = 0; k < K; k++)
                {
                    oldCenters->getDataPoint(k) = centers->getDataPoint(k);
                    centers->getDataPoint(k) = DataPoint::Zero(D);
                    counters[k] = 0;
                }

                for (int n = 0; n < N; n++)
                {
                    centers->getDataPoint(storage->getClassLabel(n)) += storage->getDataPoint(n);
                    counters[storage->getClassLabel(n)]++;
                }

                for (int k = 0; k < K; k++)
                {
                    if (counters[k] != 0) continue;

                    int max_k = 0;
                    for (int k1 = 1; k1 < K; k1++)
                    {
                        if (counters[max_k] < counters[k1])
                        {
                            max_k = k1;
                        }
                    }

                    // Find the point furthest away
                    DataPoint oldCenter = centers->getDataPoint(max_k);
                    oldCenter /= counters[max_k];
                    
                    int max_n = 0;
                    float maxDist = 0;

                    for (int n = 0; n < N; n++)
                    {
                        if (storage->getClassLabel(n) != max_k)
                        {
                            continue;
                        }

                        const float dist = computeDistance(storage->getDataPoint(n), oldCenter);
                        if (dist > maxDist)
                        {
                            maxDist = dist;
                            max_n = n;
                        }
                    }

                    counters[max_k]--;
                    counters[k]++;
                    centers->getDataPoint(max_k) -= storage->getDataPoint(max_n);
                    storage->getClassLabel(max_n) = k;
                    centers->getDataPoint(k) = storage->getDataPoint(max_n);
                }

                // To check how much centers changed.
                float maxDrift = 0;

                // Normalize the centers
                for (int k = 0; k < K; k++)
                {
                    centers->getDataPoint(k) /= counters[k];
                    maxDrift = std::max(maxDrift, computeDistance(centers->getDataPoint(k), oldCenters->getDataPoint(k)));
                }

                // Stop if changes are minimal:
                if (maxDrift < epsilon)
                {
                    stop = true;
                }
            }
            
            // Compute the distance matrix
            error = 0;
            
            for (int n = 0; n < N; n++)
            {
                float minDistance = 1e35;
                int bestLabel = -1;
                const DataPoint & x = storage->getDataPoint(n);
                
                for (int k = 0; k < K; k++)
                {
                    const float distance = computeDistance(centers->getDataPoint(k), x);
                    
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        bestLabel = k;
                    }
                }
                
                BOOST_ASSERT(bestLabel >= 0);
                
                // Update the error.
                error += minDistance;
                
                // Set best label.
                storage->getClassLabel(n) = bestLabel;
            }
            
            BOOST_ASSERT(error >= 0);
            
            if (stop)
            {
                break;
            }
        }
        
        if (error < bestError)
        {
            _centers = centers->hardCopy();
            bestError = error;
            for (int n = 0; n < N; n++)
            {
                labels[n] = storage->getClassLabel(n);
            }
        }
    }
    
    return _centers;
}

void KMeans::initCentersPP(AbstractDataStorage::ptr storage, 
        DataStorage::ptr centers)
{
    const int N = storage->getSize();
    const int K = numClusters;  
    
    // K-means++ initialization.
    for (int k = 0; k < K; k++)
    {
        // The probability dsitribution over all points we draw from.
        Eigen::VectorXf probability(N);

        if (k == 0)
        {
            // If this is the first center, then probability is unifrom.
            for (int n = 0; n < N; n++)
            {
                probability(n) = 1;
            }
        }
        else
        {
            // Compute distance to nearest center for each data point.
            for (int n = 0; n < N; n++)
            {
                float minDistance = 1e35;
                for (int c = 0; c < k; c++)
                {

                    DataPoint difference = storage->getDataPoint(n) 
                            - centers->getDataPoint(c);
                    
                    float distance = difference.transpose()*difference;
                    
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                    }
                }
                
                probability(n) = minDistance;
            }
        }
        
        // Normalize by the sum of all distances.
        probability /= probability.sum();

        // Compute the cumulative sum of all probabilities in order
        // to draw from the distribution.
        std::vector<float> probabilityCumSum(N, 0);
        for (int n = 0; n < N; n++)
        {
            if (n == 0)
            {
                probabilityCumSum[n] = probability(n);
            }
            else
            {
                probabilityCumSum[n] = probability(n) + probabilityCumSum[n - 1];
            }
        }

        // Choose random number in [0,1];
        float r = std::rand()/static_cast<float>(RAND_MAX);

        // Find the corresponding data point index.
        int n = 0;
        while (r > probabilityCumSum[n]) {
            n++;
        }

        // We overstepped the drawn point by one.
        n = std::max(0, n - 1);
        DataPoint center(storage->getDataPoint(n)); // Copy the center!
        centers->getDataPoint(k) = center;
    }
}

void KMeans::initCentersRandom(AbstractDataStorage::ptr storage, 
        DataStorage::ptr centers)
{
    const int N = storage->getSize();
    const int K = numClusters;
    
    for (int k = 0; k < K; k++)
    {
        // Centers are chosen uniformly at random.
        int n = std::rand()%N;
        
        DataPoint center(storage->getDataPoint(n)); // Copy the center!
        centers->getDataPoint(k) = center;
    }
}


////////////////////////////////////////////////////////////////////////////////
/// ClassStatisticsTool
////////////////////////////////////////////////////////////////////////////////

void ClassStatisticsTool::measure(AbstractDataStorage::ptr storage, std::vector<float> & result) const
{
    // Count the points
    result.resize(storage->getClasscount() + 1, 0.0f);
    
    for (int n = 0; n < storage->getSize(); n++)
    {
        if (storage->getClassLabel(n) != LIBF_NO_LABEL)
        {
            result[storage->getClassLabel(n)] += 1.0f;
        }
        else
        {
            // This data point has no label
            result[storage->getClasscount()] += 1.0f;
        }
    }
    
    // Normalize the distribution
    for (size_t c = 0; c < result.size(); c++)
    {
        result[c] /= storage->getSize();
    }
}

void ClassStatisticsTool::print(const std::vector<float> & result) const
{
    for (size_t c = 0; c < result.size(); c++)
    {
        printf("Class %3d: %4f%%\n", static_cast<int>(c), result[c]*100);
    }
}

void ClassStatisticsTool::measureAndPrint(AbstractDataStorage::ptr storage) const
{
    std::vector<float> result;
    measure(storage, result);
    print(result);
}

////////////////////////////////////////////////////////////////////////////////
/// ZScoreNormalizer
////////////////////////////////////////////////////////////////////////////////

void ZScoreNormalizer::learn(AbstractDataStorage::ptr storage)
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    // Reset the model
    mean = DataPoint::Zero(D);
    stdev = DataPoint::Zero(D);
    
    // Compute the mean of the data set
    for (int n = 0; n < N; n++)
    {
        mean += storage->getDataPoint(n);
    }
    
    // Normalize the mean
    mean /= N;
    
    // Compute the standard deviation
    for (int n = 0; n < N; n++)
    {
        const DataPoint temp = storage->getDataPoint(n) - mean;
        stdev += temp.cwiseProduct(temp);
    }
    
    // Normalize
    stdev /= N;
    
    // Compute the square root
    for (int d = 0; d < D; d++)
    {
        stdev(d) = std::sqrt(stdev(d));
        // Perform minor regularization
        if (stdev(d) == 0)
        {
            stdev(d) = 1;
        }
    }
}

void ZScoreNormalizer::apply(DataStorage::ptr storage) const
{
    const int N = storage->getSize();
    
    BOOST_ASSERT_MSG(storage->getDimensionality() == mean.rows(), "Mismatch between the learned model and the given data storage.");
    
    for (int n = 0; n < N; n++)
    {
        storage->getDataPoint(n) = (storage->getDataPoint(n) - mean).cwiseProduct(stdev.cwiseInverse());
    }
}

void ZScoreNormalizer::read(std::istream& stream)
{
    readBinary(stream, mean);
    readBinary(stream, stdev);
}

void ZScoreNormalizer::write(std::ostream& stream) const
{
    writeBinary(stream, mean);
    writeBinary(stream, stdev);
}


////////////////////////////////////////////////////////////////////////////////
/// PCA
////////////////////////////////////////////////////////////////////////////////

void PCA::learn(AbstractDataStorage::ptr storage)
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    // Set up the data matrix
    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(D, N);
    
    // Compute the mean of the data points
    mean = DataPoint::Zero(D);
    for (int n = 0; n < N; n++)
    {
        mean += storage->getDataPoint(n);
    }
    mean /= N;
    
    for (int n = 0; n < N; n++)
    {
        X.col(n) = storage->getDataPoint(n) - mean;
    }
    
    // Compute the singular value decomposition
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeFullU);
    
    V = svd.matrixU();
}

void PCA::apply(DataStorage::ptr storage, int M) const
{
    BOOST_ASSERT_MSG(1 <= M && M <= V.rows(), "Invalid number of projection dimensions.");
    
    const int N = storage->getSize();
    
    for (int n = 0; n < N; n++)
    {
        const DataPoint temp = storage->getDataPoint(n) - mean;
        storage->getDataPoint(n).resize(M, 1);
        storage->getDataPoint(n) = V.topRows(M)*temp;
    }
}

void PCA::read(std::istream& stream)
{
    readBinary(stream, V);
    readBinary(stream, mean);
}

void PCA::write(std::ostream& stream) const
{
    writeBinary(stream, V);
    writeBinary(stream, mean);
}