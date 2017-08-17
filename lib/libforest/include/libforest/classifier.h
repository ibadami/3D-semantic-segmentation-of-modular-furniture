#ifndef LIBF_CLASSIFIERS_H
#define LIBF_CLASSIFIERS_H

/**
 * This file contains the data structures for the classifiers. There are 
 * basically two kinds ot classifiers:
 * 1. Decision trees
 * 2. Random forests
 */

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include "tree.h"
#include "io.h"

namespace libf {
    /**
     * The base class for all classifiers. This allows use to use the evaluation
     * tools for both trees and forests. 
     */
    class AbstractClassifier {
    public:
        typedef std::shared_ptr<AbstractClassifier> ptr;
        
        virtual ~AbstractClassifier() {}
        
        /**
         * Assigns an integer class label to some data point
         */
        virtual int classify(const DataPoint & x) const;
        
        /**
         * Classifies an entire data set and uses the integer values. 
         */
        virtual void classify(AbstractDataStorage::ptr, std::vector<int> & results) const;
        
        /**
         * Returns the class posterior probability p(c|x).
         */
        virtual void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const = 0;
        
        virtual float getVotesFor1(const DataPoint & x) const { return 0; }
        
        /**
         * Reads the classifier from a stream. 
         * 
         * @param stream The stream to read the classifier from
         */
        virtual void read(std::istream & stream) = 0;
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the tree to.
         */
        virtual void write(std::ostream & stream) const = 0;
    };
    
    /**
     * This is the base class for all tree classifier node data classes. 
     */
    class TreeClassifierNodeData {
    public:
        /**
         * A histogram that represents a distribution over the class labels
         */
        std::vector<float> histogram;
    };
    
    /**
     * Overload the read binary method to also read DecisionTreeNodeData
     */
    template <>
    inline void readBinary(std::istream & stream, TreeClassifierNodeData & v)
    {
        readBinary(stream, v.histogram);
    }
    
    /**
     * Overload the write binary method to also write DecisionTreeNodeData
     */
    template <>
    inline void writeBinary(std::ostream & stream, const TreeClassifierNodeData & v)
    {
        writeBinary(stream, v.histogram);
    }
    
    
    /**
     * This is the base class for all tree classifiers
     */
    template <class Config, class Data>
    class AbstractTreeClassifier : public AbstractTree<Config, Data, AbstractClassifier> {
    public:
        
        virtual ~AbstractTreeClassifier() {}
        
        /**
         * Only accept template parameters that extend TreeClassifierNodeData.
         * Note: The double parentheses are needed.
         */
        BOOST_STATIC_ASSERT((boost::is_base_of<TreeClassifierNodeData, Data>::value));
        
        /**
         * Returns the class log posterior log(p(c | x)). The probabilities are
         * not normalized. 
         * 
         * @param x The data point for which the log posterior shall be evaluated. 
         * @param probabilities The vector of log posterior probabilities
         */
        void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const
        {
            // Get the leaf node
            const int leafNode = this->findLeafNode(x);
            probabilities = this->getNodeData(leafNode).histogram;
        }
    };
    
    /**
     * This is the base class for all forest classifiers. 
     */
    template <class TreeType>
    class AbstractForestClassifier : public AbstractForest<TreeType, AbstractClassifier> {};
    
    /**
     * This is the data each node in an online decision tree carries
     * 
     * These following statistics are saved as entropy histograms.
     * @see http://lrs.icg.tugraz.at/pubs/saffari_olcv_09.pdf
     */
    class OnlineDecisionTreeNodeData : public TreeClassifierNodeData {
    public:
        /**
         * The node's statistics saved as entropy histogram.
         */
        EfficientEntropyHistogram nodeStatistics;
        /**
         * Left child statistics for all splits.
         */
        std::vector<EfficientEntropyHistogram> leftChildStatistics;
        /**
         * Right child statistics for all splits.
         */
        std::vector<EfficientEntropyHistogram> rightChildStatistics;
        /**
         * Thresholds for each node.
         */
        std::vector< std::vector<float> > nodeThresholds; // TODO: This is really messy!
        /**
         * Features for all nodes.
         */
        std::vector<int> nodeFeatures;
    };
    
    /**
     * Overload the read binary method to also read OnlineDecisionTreeNodeData
     */
    template <>
    inline void readBinary(std::istream & stream, OnlineDecisionTreeNodeData & v)
    {
        // TODO: Implement this stuff
    }
    
    /**
     * Overload the write binary method to also write DecisionTreeNodeData
     */
    template <>
    inline void writeBinary(std::ostream & stream, const OnlineDecisionTreeNodeData & v)
    {
        // TODO: Implement this stuff
    }
    
    /**
     * This class represents a decision tree.
     */
    class DecisionTree : public AbstractAxisAlignedSplitTree< AbstractTreeClassifier<AxisAlignedSplitTreeNodeConfig, TreeClassifierNodeData> > {
    public:
        typedef std::shared_ptr<DecisionTree> ptr;
    };
    
    /**
     * This class represents an online decision tree.
     */
    class OnlineDecisionTree : public AbstractAxisAlignedSplitTree< AbstractTreeClassifier<AxisAlignedSplitTreeNodeConfig, OnlineDecisionTreeNodeData> > {
    public:
        typedef std::shared_ptr<OnlineDecisionTree> ptr;
    };
    
    /**
     * This class represents a projective decision tree.
     */
    class ProjectiveDecisionTree : public AbstractProjectiveSplitTree< AbstractTreeClassifier<ProjectiveSplitTreeNodeConfig, TreeClassifierNodeData> > {
    public:
        typedef std::shared_ptr<ProjectiveDecisionTree> ptr;
    };
    
    /**
     * This class implements random forest classifiers.
     */
    template <class TreeType>
    class RandomForest : public AbstractForestClassifier<TreeType> {
    public:
        typedef std::shared_ptr< RandomForest<TreeType> > ptr;
        
        /**
         * Only accept template parameters that extend AbstractClassifier.
         * Note: The double parentheses are needed.
         */
        BOOST_STATIC_ASSERT((boost::is_base_of<AbstractClassifier, TreeType>::value));
        
        virtual ~RandomForest() {}
        
        virtual float getVotesFor1(const DataPoint & x) const
        {
            float votes = 0;

            // Let the crowd decide
            for (int i = 0; i < this->getSize(); i++)
            {
                // Get the probabilities from the current tree
                if (this->getTree(i)->classify(x) == 1)
                {
                    votes += 1;
                }
            }
            return votes;
        }
        /**
         * Returns the class log posterior log(p(c | x)).
         * 
         * @param x The data point x to determine the posterior distribution of
         * @param probabilities A vector of log posterior probabilities
         */
        void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const
        {
            BOOST_ASSERT_MSG(this->getSize() > 0, "Cannot classify a point from an empty ensemble.");

            this->getTree(0)->classLogPosterior(x, probabilities);

            // Let the crowd decide
            for (int i = 1; i < this->getSize(); i++)
            {
                // Get the probabilities from the current tree
                std::vector<float> currentHist;
                this->getTree(i)->classLogPosterior(x, currentHist);

                BOOST_ASSERT(currentHist.size() > 0);

                // Accumulate the votes
                for (size_t c = 0; c < currentHist.size(); c++)
                {
                    probabilities[c] += currentHist[c];
                }
            }
        }
    };
    
    /**
     * This is a specialization for online random forests. It will be removed
     * once the learning process is refactored. 
     */
    class OnlineRandomForest : public RandomForest<OnlineDecisionTree> {
    public:
        typedef std::shared_ptr<OnlineRandomForest> ptr;
    };
    
    /**
     * This classifier can be used for boosting. The template class names the
     * actual classifier. 
     */
    template <class T>
    class WeakClassifier : public AbstractClassifier {
    public:
        typedef std::shared_ptr< WeakClassifier<T> > ptr;
        
        /**
         * Only accept template parameters that extend AbstractClassifier.
         * Note: The double parentheses are needed.
         */
        BOOST_STATIC_ASSERT((boost::is_base_of<AbstractClassifier, T>::value));
        
        WeakClassifier() : weight(0) {}
        
        /**
         * Sets the classifier
         * 
         * @param _classifier
         */
        void setClassifier(std::shared_ptr<T> _classifier)
        {
            classifier = _classifier;
        }
        
        /**
         * Returns the classifier
         * 
         * @return The classifier
         */
        std::shared_ptr<T> getClassifier() const
        {
            return classifier;
        }
        
        /**
         * Sets the weights
         * 
         * @param _weight 
         */
        void setWeight(float _weight)
        {
            weight = _weight;
        }
        
        /**
         * Returns the weight
         * 
         * @return weight
         */
        float getWeight() const
        {
            return weight;
        }
        
        /**
         * Reads the classifier from a stream. 
         * 
         * @param stream The stream to read the classifier from
         */
        void read(std::istream & stream)
        {
            readBinary(stream, weight);
            classifier = std::make_shared<T>();
            classifier->read(stream);
        }
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the tree to.
         */
        void write(std::ostream & stream) const
        {
            writeBinary(stream, weight);
            classifier->write(stream);
        }
        
        /**
         * Returns the class posterior probability p(c|x).
         */
        virtual void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const
        {
            classifier->classLogPosterior(x, probabilities);
        }
        
    private:
        /**
         * The actual classifier
         */
        std::shared_ptr<T> classifier;
        /**
         * The weight
         */
        float weight;
    };
    
    /**
     * A boosted random forest classifier.
     */
    template <class TreeType>
    class BoostedRandomForest : public AbstractForestClassifier< WeakClassifier<TreeType> > {
    public:
        typedef std::shared_ptr<BoostedRandomForest> ptr;
        
        virtual ~BoostedRandomForest() {}
        
        /**
         * Returns the class log posterior log(p(c | x)).
         * 
         * @param x The data point x to determine the posterior distribution of
         * @param probabilities A vector of log posterior probabilities
         */
        void classLogPosterior(const DataPoint & x, std::vector<float> & probabilities) const
        {
            BOOST_ASSERT_MSG(this->getSize() > 0, "Cannot classify a point from an empty ensemble.");
            
            // TODO: This can be done way more efficient
            // Determine the number of classes by looking at a histogram
            this->getTree(0)->classLogPosterior(x, probabilities);
            // Initialize the result vector
            const int C = static_cast<int>(probabilities.size());
            for (int c = 0; c < C; c++)
            {
                probabilities[c] = 0;
            }

            // Let the crowd decide
            for (int i = 0; i < this->getSize(); i++)
            {
                // Get the resulting label
                const int label = this->getTree(i)->classify(x);
                probabilities[label] += this->getTree(i)->getWeight();
            }
        }
    };
}
#endif