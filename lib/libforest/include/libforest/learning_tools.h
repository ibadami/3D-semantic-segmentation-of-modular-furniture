#ifndef LIBF_LEARNING_TOOLS_H
#define LIBF_LEARNING_TOOLS_H

#include "data.h"

namespace libf {
    
    /**
     * This class can be used in order to sort the array of data point IDs by
     * a certain dimension
     */
    class FeatureComparator {
    public:
        /**
         * The feature dimension
         */
        int feature;
        /**
         * The data storage
         */
        AbstractDataStorage::ptr storage;

        /**
         * Compares two training examples
         */
        bool operator() (const int lhs, const int rhs) const
        {
            return storage->getDataPoint(lhs)(feature) < storage->getDataPoint(rhs)(feature);
        }
    };
    
    /**
     * Online decision trees are totally randomized, ie.e. the threshold at each
     * node is chosen randomly. Therefore, the tree has to know the ranges from
     * which to pick the thresholds.
     */
    class RandomThresholdGenerator {
    public:
        
        /**
         * Default constructor: use addFeatureRange to add the value range for
         * an additional feature. Features have to be added in the correct
         * order!
         */
        RandomThresholdGenerator() {};
        
        /**
         * Deduce the feature ranges from the given dataset.
         */
        RandomThresholdGenerator(AbstractDataStorage::ptr storage);
        
        /**
         * Adds a feature range. Note that the features have to be added in the 
         * correct order!
         */
        void addFeatureRange(float _min, float _max)
        {
            min.push_back(_min);
            max.push_back(_max);
        }
        
        /**
         * Adds the same range for num consecutive features.
         * Note that the features have to be added in the correct order!
         */
        void addFeatureRanges(int num, float _min, float _max)
        {
            for (int i = 0; i < num; i++)
            {
                min.push_back(_min);
                max.push_back(_max);
            }
        }
        
        /**
         * Returns minimum for a specific feature.
         */
        float getMin(int feature)
        {
            assert(feature >=0 && feature < getSize());
            return min[feature];
        }
        
        /**
         * Returns maximum for a specific feature.
         */
        float getMax(int feature)
        {
            assert(feature >= 0 && feature < getSize());
            return max[feature];
        }
        
        /**
         * Samples a value uniformly for the given feature.
         */
        float sample(int feature);
        
        /**
         * Returns the size of the generator (number of features).
         */
        int getSize()
        {
            return min.size();
        }
        
    protected:
        /**
         * Minimum value for each feature.
         */
        std::vector<float> min;
        /**
         * Maximum value for each feature.
         */
        std::vector<float> max;
    };
}
#endif