#ifndef LIBF_UTIL_H
#define LIBF_UTIL_H

#include "libforest/data.h"
#include "libforest/error_handling.h"
#include "libforest/fastlog.h"
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>

/**
 * This is the buffer size for the arrays in the graph structures
 */
#define LIBF_GRAPH_BUFFER_SIZE 10000

/**
 * Quickly computes the entropy of a single bin of a histogram.
 */
#define LIBF_ENTROPY(p) (-(p)*fastlog2(p))

namespace libf {
    // TODO: Remove this crap
    class Exception {
    public:
        Exception(const std::string & s) {}
    };
    
    /**
     * This class can be used as a base class for arbitrary objects
     */
    class Object {};
    
    /**
     * This class contains several use functions that are somewhat unrelated. 
     */
    class Util {
    public:
        /**
         * Creates a random permutation of [N]. 
         * TODO: Write unit test
         * 
         * @param N The number of points
         * @param sigma The permutation
         */
        static void generateRandomPermutation(int N, std::vector<int> & sigma);
        
        /**
         * Returns true if the given parameter is a valid permutation and returns
         * false otherwise. If sigma is of length N, then we check if sigma is in
         * S_N. The permutation must be given as a graph (n, sigma(n)). 
         * 
         * @param sigma The permutation. 
         * @return true if sigma is a valid permutation. 
         */
        static bool isValidPermutation(const std::vector<int> & sigma) throw()
        {
            // Initialize a list of flags to remember which images have several
            // points in the domain
            std::vector<bool> check(sigma.size(), false);
            
            for (size_t i = 0; i < sigma.size(); i++)
            {
                // If the given image is not in [N], then this cannot be a valid
                // permutation
                if (sigma[i] < 0 || sigma[i] >= static_cast<int>(sigma.size()))
                {
                    return false;
                }
                
                // Check if this image has already a partner
                if (check[sigma[i]])
                {
                    return false;
                }
                
                // Remember that we already used this image
                check[sigma[i]] = true;
            }
            
            return true;
        }
        
        /**
         * Applies a permutation to the vector in and saves it in the vector out. 
         * the permutation must be given as a mapping of the form n -> p(n). Thus
         * the n-th entry of the permutation vector must be the image under the
         * permutation. 
         * 
         * You must not set in = out. This would break the algorithm. 
         * 
         * The algorithm does not check if you pass a valid permutation. This
         * would create too much overhead.
         * 
         * @param permutation The permutation of the vector entries as a list of images
         * @param in The vector to permute
         * @param out The permuted vector
         */
        template <class T>
        static void permute(const std::vector<int> & permutation, const std::vector<T> & in, std::vector<T> & out) throw (AssertionException)
        {
            BOOST_ASSERT_MSG(permutation.size() == in.size(), "The permutation has invalid length.");
            BOOST_ASSERT_MSG(&in != &out, "The input vector must not be the same as the output vector.");
            BOOST_ASSERT_MSG(isValidPermutation(permutation), "The given vector does not encode a valid permutation.");

            // Make the output array of the correct size. 
            out.resize(in.size());

            // Copy the elements
            for (size_t i = 0; i < permutation.size(); i++)
            {
                out[permutation[i]] = in[i];
            }
        }
        
        /**
         * Computes the Hamming distance between two vectors of equal size. The
         * Hamming Distance is defined as the number of unequal entries of 
         * the two vectors. If the vectors are of unequal length, the distance
         * treats the "missing" entries as no match. 
         * 
         * @param v1 The first vector
         * @param v2 The second vector
         * @return The Hamming distance of v1 and v2
         */
        template <class T>
        static size_t hammingDist(const std::vector<T> & v1, const std::vector<T> & v2)
        {
            size_t result = 0;
            
            // Take care of vectors of unequal lengths
            if (v1.size() > v2.size())
            {
                result = v1.size() - v2.size();
            }
            else if (v2.size() > v1.size())
            {
                result = v2.size() - v1.size();
            }
            
            // Compute the actual distance
            for (size_t i = 0; i < std::min(v1.size(), v2.size()); i++)
            {
                if (v1[i] != v2[i])
                {
                    result++;
                }
            }
            
            return result;
        }
    
        /**
         * Non recursive calculation of the factorial k!.
         * TODO: Write unit tests
         */
        static int factorial(int k)
        {
            assert(k >= 0);
            
            if (k == 0)
            {
                return 1;
            }
            else
            {
                int factorial = 1;
                for (int i = 2; i < k + 1; i++)
                {
                    factorial *= i;
                }

                return factorial;
            }
        }
        
        /**
         * Non-recursive calculation of the double factorial for odd numbers k:
         *  
         * k!! = 1*3*5*...*k
         * TODO: Write unit tests
         */
        static int doubleFactorial(int k)
        {
            assert(k%2 == 1);
            
            int factorial = 1;
            for (int i = 3; i < k + 1; i += 2)
            {
                factorial *= i;
            }
            
            return factorial;
        }
        
        /**
         * Dumps a vector to standard out. The elements of the vector must be
         * accepted by std::cout <<. This function is only for debug purposes. 
         * 
         * @param v The vector to dump
         */
        template <class T>
        void dumpVector(const std::vector<T> & v)
        {
            for (size_t i = 0; i < v.size(); i++)
            {
                std::cout << i << ": " << v[i] << std::endl;
            }

            std::cout.flush();
        }
        
        /**
         * Returns the index of an array. If there are multiple maxima in an 
         * array the smallest of those indices is returned. If the array is
         * empty, the result is 0. 
         * 
         * TODO: Write unit test
         * 
         * @param v The array of comparable objects
         * @return The maximizing index
         */
        template <class T>
        static size_t argMax(const std::vector<T> & v)
        {
            // If thee array is empty, there is not much to do. 
            if (v.size() == 0) return 0;
            
            // Determine the maximizing index
            T arg = v[0];
            size_t index = 0;
            for (size_t i = 1; i < v.size(); i++)
            {
                if (v[i] > arg)
                {
                    arg = v[i];
                    index = i;
                }
            }
            
            return index;
        }
    };

    /**
     * This class collects functions that are useful for generating ncurses based
     * shell GUIs. 
     */
    class GUIUtil {
    public:
        /**
         * Shows a progress bar. 
         * 
         * @param p The current percentage that is completed (in [0,1])
         */
        static void printProgressBar(float p);
    };
    
    /**
     * A histogram over the class labels. We use this for training.
     */
    class EfficientEntropyHistogram {
    public:
        /**
         * Default constructor. Initializes the histogram with 0 bins.
         */
        EfficientEntropyHistogram() : 
                bins(0),
                histogram(0),
                mass(0),
                entropies(0),
                totalEntropy(0) {}

        /**
         * Construct a entropy histogram of the given size. All bins are 
         * initialized with 0. 
         * 
         * @param bins The number of bins
         */
        EfficientEntropyHistogram(int bins) : 
            bins(0),
            histogram(0),
            mass(0),
            entropies(0),
            totalEntropy(0)
        {
            // Allocate the histogram
            resize(bins);
        }

        /**
         * Copy constructor.
         * 
         * @param other The histogram to copy
         */
        EfficientEntropyHistogram(const EfficientEntropyHistogram & other) : 
                bins(0),
                histogram(0),
                mass(0),
                entropies(0),
                totalEntropy(0) 
        {
            resize(other.bins);
            for (int i = 0; i < bins; i++)
            {
                histogram[i] = other.histogram[i];
                entropies[i] = other.entropies[i];
            }
            totalEntropy = other.totalEntropy;
            mass = other.mass;
        }

        /**
         * Assignment operator
         * 
         * @param other The object on the right side of the assignment operator
         */
        EfficientEntropyHistogram & operator= (const EfficientEntropyHistogram &other)
        {
            // Prevent self assignment
            if (this != &other)
            {
                resize(other.bins);
                for (int i = 0; i < bins; i++)
                {
                    histogram[i] = other.histogram[i];
                    entropies[i] = other.entropies[i];
                }
                totalEntropy = other.totalEntropy;
                mass = other.mass;
            }
            
            return *this;
        }

        /**
         * Destructor
         */
        ~EfficientEntropyHistogram()
        {
            // Only delete the arrays if they have been allocated
            if (histogram != 0)
            {
                delete[] histogram;
            }
            if (entropies != 0)
            {
                delete[] entropies;
            }
        }
        
        /**
         * Sets all entries in the histogram to 0. 
         */
        void reset()
        {
            for (int i = 0; i < bins; i++)
            {
                histogram[i] = 0;
                entropies[i] = 0;
            }
            totalEntropy = 0;
            mass = 0;
        }
        
        /**
         * Resizes the histogram to a certain size and initializes all bins
         * with 0 even if the size did not change. 
         * 
         * @param newBins The new number of bins
         */
        void resize(int newBins)
        {
            BOOST_ASSERT_MSG(newBins >= 0, "Bin count must be non-negative.");
            
            // Release the current histogram
            if (newBins != bins)
            {
                if (histogram != 0)
                {
                    delete[] histogram;
                    histogram = 0;
                }
                if (entropies != 0)
                {
                    delete[] entropies;
                    entropies = 0;
                }
            }

            bins = newBins;
            
            // Only allocate a new histogram, if there is more than one class
            if (newBins > 0)
            {
                histogram = new int[bins];
                entropies = new float[bins];
                
                reset();
            }
        }

        /**
         * Returns the size of the histogram (= class count)
         * 
         * @return The number of bins of the histogram
         */
        int getSize() const
        {
            return bins; 
        }

        /**
         * Get the histogram value for class i.
         * 
         * @return The value in bin i.
         */
        int at(const int i) const
        {
            BOOST_ASSERT_MSG(i >= 0 && i < bins, "Bin index out of range.");
            return histogram[i];
        }
        
        /**
         * Adds one instance of class i while updating entropy information.
         * 
         * @param i The bin to which a single point shall be added.
         */
        void addOne(const int i)
        {
            BOOST_ASSERT_MSG(i >= 0 && i < bins, "Invalid bin bin index.");

            totalEntropy += LIBF_ENTROPY(mass);
            mass += 1;
            totalEntropy -= LIBF_ENTROPY(mass);
            histogram[i]++;
            totalEntropy -= entropies[i];
            entropies[i] = LIBF_ENTROPY(histogram[i]); 
            totalEntropy += entropies[i];
        }
        
        /**
         * Remove one instance of class i while updating the entropy information.
         * 
         * @param i The bin from which a single point shall be removed.
         */
        void subOne(const int i)
        {
            BOOST_ASSERT_MSG(i >= 0 && i < bins, "Invalid bin bin index.");
            BOOST_ASSERT_MSG(at(i) > 0, "Bin is already empty.");

            totalEntropy += LIBF_ENTROPY(mass);
            mass -= 1;
            totalEntropy -= LIBF_ENTROPY(mass);

            histogram[i]--;
            totalEntropy -= entropies[i];
            if (histogram[i] < 1)
            {
                entropies[i] = 0;
            }
            else
            {
                entropies[i] = LIBF_ENTROPY(histogram[i]); 
                totalEntropy += entropies[i];
            }
        }
        
        /**
         * Returns the total mass of the histogram.
         * 
         * @return The total mass of the histogram
         */
        float getMass() const
        {
            return mass;
        }

        /**
         * Calculates the entropy of the histogram. This only works if the 
         * function initEntropies has been called before. 
         * 
         * @return The calculated entropy
         */
        float getEntropy() const
        {
            return totalEntropy;
        }

        /**
         * Returns true if the histogram has at most a single non-empty bin. 
         * 
         * @return true if the histogram is pure. 
         */
        bool isPure() const
        {
            bool nonPure = false;
            for (int i = 0; i < bins; i++)
            {
                if (histogram[i] > 0)
                {
                    if (nonPure)
                    {
                        return false;
                    }
                    else
                    {
                        nonPure = true; 
                    }
                }
            }
            
            return true;
        }

    private:
        /**
         * The number of classes in this histogram
         */
        int bins;

        /**
         * The actual histogram
         */
        int* histogram;

        /**
         * The integral over the entire histogram
         */
        float mass;

        /**
         * The entropies for the single bins
         */
        float* entropies;

        /**
         * The total entropy
         */
        float totalEntropy;
    };
    
    /**
     * Represents the Gaussian at each leaf and allows to update mean and covariance
     * efficiently as well as compute the determinant of the covariance matrix
     * for learning.
     */
    class EfficientCovarianceMatrix {
    public:
        /**
         * Creates an empty covaraince matrix.
         */
        EfficientCovarianceMatrix() : 
                dimensions(0),
                mass(0),
                cachedTrueCovariance(false),
                cachedDeterminant(false),
                covarianceDeterminant(0),
                cachedTrueMean(false) {};
                
        /**
         * Creates a _classes x _classes covariance matrix.
         */
        EfficientCovarianceMatrix(int _dimensions) : 
                dimensions(_dimensions),
                mass(0),
                covariance(_dimensions, _dimensions),
                mean(_dimensions),
                cachedTrueCovariance(false),
                cachedDeterminant(false),
                trueCovariance(_dimensions, _dimensions),
                covarianceDeterminant(0),
                cachedTrueMean(false), 
                trueMean(dimensions) {};
                
        /**
         * Destructor.
         */
        virtual ~EfficientCovarianceMatrix() {};
        
        EfficientCovarianceMatrix operator=(const EfficientCovarianceMatrix & other)
        {
            mean = other.mean;
            covariance = other.covariance;
            dimensions = other.dimensions;
            mass = other.mass;
            // TODO: does currently not consider caching!
            
            trueCovariance = Eigen::MatrixXf::Zero(dimensions, dimensions);
            covarianceDeterminant = 0;
            cachedTrueCovariance = false;
            cachedDeterminant = false;
            
            trueMean = Eigen::VectorXf::Zero(dimensions);
            cachedTrueMean = false;
            
            return *this;
        }
        
        /**
         * Resets the mean and covariance to zero.
         */
        void reset()
        {
            mean = Eigen::VectorXf::Zero(dimensions);
            covariance = Eigen::MatrixXf::Zero(dimensions, dimensions);
            mass = 0;
            
            // Update caches.
            cachedTrueCovariance = false;
            cachedDeterminant = false;
            trueCovariance = Eigen::MatrixXf::Zero(dimensions, dimensions);
            covarianceDeterminant = 0;
            
            cachedTrueMean = false;
            trueMean = Eigen::VectorXf::Zero(dimensions);
        }
        
        /**
         * Get the number of samples.
         */
        int getMass()
        {
            return mass;
        }
        
        /**
         * Add a sample and update covariance and mean estimate.
         */
        void addOne(const DataPoint & x)
        {
            assert(x.rows() == mean.rows());
            assert(x.rows() == covariance.rows());
            assert(x.rows() == covariance.cols());

            for (int i = 0; i < x.rows(); i++)
            {
                // Update running estimate of mean.
                mean(i) += x(i);

                for (int j = 0; j < x.rows(); j++)
                {
                    // Update running estimate of covariance.
                    covariance(i, j) += x(i)*x(j);
                }
            }
            
            for (int i = 0; i < covariance.rows(); i++)
            {
                // Cannot be positive definite if diagonal values are negative:
                assert(covariance(i, i) > 0);

                for (int j = 0; j < i; j++)
                {
                    // Symmetry:
                    assert(covariance(i, j) == covariance(j, i));
                }
            }
            
            cachedTrueMean = false;
            cachedTrueCovariance = false;
            cachedDeterminant = false;
            
            mass += 1;
        }
        
        /**
         * Remove a sample and update covariance and mean estimate.
         */
        void subOne(const DataPoint & x)
        {
            assert(x.rows() == mean.rows());
            assert(x.rows() == covariance.rows());
            assert(x.rows() == covariance.cols());
            
            for (int i = 0; i < x.rows(); i++)
            {
                // Update running estimate of mean.
                mean(i) -= x(i);

                for (int j = 0; j < x.rows(); j++)
                {
                    // Update running estimate of covariance.
                    covariance(i, j) -= x(i)*x(j);
                }
            }

            for (int i = 0; i < covariance.rows(); i++)
            {
                // Cannot be positive definite if diagonal values are negative:
                assert(covariance(i, i) > 0);

                for (int j = 0; j < i; j++)
                {
                    // Symmetry:
                    assert(covariance(i, j) == covariance(j, i));
                }
            }
            
            cachedTrueMean = false;
            cachedTrueCovariance = false;
            cachedDeterminant = false;
            
            mass -= 1;
        }
        
        /**
         * Returns mean.
         */
        Eigen::VectorXf & getMean()
        {
            if (!cachedTrueMean)
            {
                trueMean = mean/mass;
                cachedTrueMean = true;
                
            }
            
            return trueMean;
        }
        
        /**
         * Returns true covariance matrix from estimates.
         */
        Eigen::MatrixXf & getCovariance()
        {
            if (!cachedTrueCovariance)
            {
                trueCovariance = (mean*mean.transpose())/(mass*mass); // (mean/mass)*(mean.transpose()/mass)
                trueCovariance = covariance/(mass - 1) - trueCovariance; // covariance/mass - (mean/mass)*(mean.transpose()/mass)
                
                for (int i = 0; i < trueCovariance.rows(); i++)
                {
                    // Cannot be positive definite if diagonal values are negative:
                    assert(trueCovariance(i, i) >= 0);
                    
                    for (int j = 0; j < i; j++)
                    {
                        // Symmetry:
                        assert(trueCovariance(i, j) == trueCovariance(j, i));
                    }
                }
                
                cachedTrueCovariance = true;
            }

            return trueCovariance;
        }
        
        /**
         * Returns covariance determinant;
         */
        float getDeterminant()
        {
            if (!cachedDeterminant)
            {
                covarianceDeterminant = getCovariance().determinant();
                
//                getCovariance();
//                covarianceDeterminant = 0;
//                
//                for (int i = 0; i < covariance.rows(); i++)
//                {
//                    covarianceDeterminant *= trueCovariance(i, i);
//                }
                
                cachedDeterminant = true;
            }

            return covarianceDeterminant;
        }
        
        /**
         * Get the entropy to determine split objective.
         */
        float getEntropy()
        {
            return mass*fastlog2(getDeterminant());
        }
        
    private:
        
        /**
         * Number of dimensions: dimension x dimension covariance matrix.
         */
        int dimensions;
        /**
         * Number of samples.
         */
        int mass;
        /**
         * Current estimate of dimension x dimension covariance matrix.
         */
        Eigen::MatrixXf covariance;
        /**
         * Current estimate of mean.
         */
        Eigen::VectorXf mean;
        /**
         * The true covariance is cached for reuse when setting a leaf's
         * Gaussian distribution.
         */
        bool cachedTrueCovariance;
        /**
         * The determinant is cached for the same reason as above.
         */
        bool cachedDeterminant;
        /**
         * Cached true covariance matrix.
         */
        Eigen::MatrixXf trueCovariance;
        /**
         * Cached covariance determinant.
         */
        float covarianceDeterminant;
        /**
         T* he mean is also cached.
         */
        bool cachedTrueMean;
        /**
         * The cached mean.
         */
        Eigen::VectorXf trueMean;
        
    };
}
#endif
 