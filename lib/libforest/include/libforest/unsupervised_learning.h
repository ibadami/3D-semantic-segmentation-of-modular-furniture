#ifndef LIBF_UNSUPERVISED_LEARNING_H
#define	LIBF_UNSUPERVISED_LEARNING_H

#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include "estimators.h"
#include "learning.h"

namespace libf {
    /**
     * Forward declarations.
     */
    class DensityTree;
    class DensityTreeLearner;
    
    /**
     * This is the base class for all unsupervised learners.
     */
    template <class T>
    class UnsupervisedLearner {
    public:
        
        /**
         * Learns a classifier in an unsupervised fashion.
         */
        virtual std::shared_ptr<T> learn(AbstractDataStorage::ptr storage) = 0;

    };
    
    /**
     * Learner state for density tree learner.
     */
    class DensityTreeLearnerState : public AbstractLearnerState {
    public:
        DensityTreeLearnerState() : 
            node(0),
            depth(0), 
            maxDepth(0),
            samples(0),
            objective(0) {};
            
            /**
             * Current node.
             */
            int node;
            /**
             * Depth of node.
             */
            int depth;
            /**
             * Max depth allowed.
             */
            int maxDepth;
            /**
             * Number of samples at current node.
             */
            int samples;
            /**
             * Objective of split or not split.
             */
            float objective;
    };
    
    /**
     * Learnes a density tree on a given unlabeled dataset.
     */
    class DensityTreeLearner : 
            public AbstractDecisionTreeLearner<DensityTree, DensityTreeLearnerState>,
            public UnsupervisedLearner<DensityTree> {
    public:
        DensityTreeLearner() : AbstractDecisionTreeLearner() {};
                
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(DensityTree::ptr tree, const DensityTreeLearnerState & state);
        
        /**
         * Verbose callback for this learner.
         */
        static int verboseCallback(DensityTree::ptr tree, const DensityTreeLearnerState & state);
        
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        const static int ACTION_PROCESS_NODE = 3;
        const static int ACTION_INIT_NODE = 4;
        const static int ACTION_NOT_SPLIT_NODE = 5;
        const static int ACTION_NO_SPLIT_NODE = 6;
        
        /**
         * Learn a density tree.
         */
        DensityTree::ptr learn(AbstractDataStorage::ptr storage);

    private:
        /**
         * Updates the leaf node Gaussian estimates given the current covariance
         * and mean estimate.
         */
        void updateLeafNodeGaussian(Gaussian & gaussian, EfficientCovarianceMatrix & covariance);
        
    };
    
    /**
     * This class holds the current state of the density forest learning
     * algorithm.
     */
    class DensityForestLearnerState : public AbstractLearnerState {
    public:
        DensityForestLearnerState() : AbstractLearnerState(),
                tree(0),
                numTrees(0) {}
        
        /**
         * The current tree
         */
        int tree;
        /**
         * Number of learned trees.
         */
        int numTrees;
    };
    
    /**
     * This is an offline density forest learner.
     */
    class DensityForestLearner : public AbstractRandomForestLearner<DensityForest, DensityForestLearnerState>,
            public UnsupervisedLearner<DensityForest> {
    public:
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(DensityForest::ptr forest, const DensityForestLearnerState & state);
         
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        DensityForestLearner() : AbstractRandomForestLearner(),
                treeLearner(0) {}
        
        /**
         * Sets the decision tree learner
         */
        void setTreeLearner(DensityTreeLearner* _treeLearner)
        {
            treeLearner = _treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        DensityTreeLearner* getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Learns a forests. 
         */
        virtual DensityForest::ptr learn(AbstractDataStorage::ptr storage);

    protected:
        /**
         * The tree learner
         */
        DensityTreeLearner* treeLearner;
    };
    
    /**
     * This class holds the current state of the kernel density tree learner.
     */
    class KernelDensityTreeLearnerState : public AbstractLearnerState {
    public:
        KernelDensityTreeLearnerState() : 
            node(0),
            depth(0), 
            maxDepth(0),
            samples(0),
            objective(0) {};
            
            /**
             * Current node.
             */
            int node;
            /**
             * Depth of node.
             */
            int depth;
            /**
             * Max depth allowed.
             */
            int maxDepth;
            /**
             * Number of samples at current node.
             */
            int samples;
            /**
             * Objective of split or not split.
             */
            float objective;
    };
    
    /**
     * Learns a kernel density tree.
     */
    class KernelDensityTreeLearner : 
            public AbstractDecisionTreeLearner<KernelDensityTree, KernelDensityTreeLearnerState>,
            public UnsupervisedLearner<KernelDensityTree> {
    public:
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(KernelDensityTree::ptr tree, const KernelDensityTreeLearnerState & state);
        
        /**
         * Verbose callback for this learner.
         */
        static int verboseCallback(KernelDensityTree::ptr tree, const KernelDensityTreeLearnerState & state);  
        
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        const static int ACTION_PROCESS_NODE = 3;
        const static int ACTION_INIT_NODE = 4;
        const static int ACTION_NOT_SPLIT_NODE = 5;
        const static int ACTION_NO_SPLIT_NODE = 6;
        
        /**
         * Constructs a kernel density tree learner.
         */
        KernelDensityTreeLearner() : AbstractDecisionTreeLearner(),
                kernel(new MultivariateGaussianKernel()), 
                bandwidthSelectionMethod(KernelDensityEstimator::BANDWIDTH_RULE_OF_THUMB) {};
        
        /**
         * Sets the kernel to use.
         */
        void setKernel(MultivariateKernel* _kernel)
        {
            kernel = _kernel;
        }
        
        /**
         * Returns the used kernel.
         */
        MultivariateKernel* getKernel()
        {
            return kernel;
        }
               
        /**
         * Sets the bandwidth selection method to use.
         */
        void setBandwidthSelectionMethod(int _bandwidthSelectionMethod)
        {
            bandwidthSelectionMethod = _bandwidthSelectionMethod;
        }
        
        /**
         * Returns the bandwidth selection method to use.
         */
        int getBandwidthSelectionMethod()
        {
            return bandwidthSelectionMethod;
        }
        
        /**
         * Learns a kernel density tree.
         */
        virtual KernelDensityTree::ptr learn(AbstractDataStorage::ptr storage);
        
    private:
        /**
         * Initializes the leaf node estimator.
         */
        void initializeLeafNodeEstimator(Gaussian & gaussian,
                EfficientCovarianceMatrix & covariance,
                KernelDensityEstimator & estimator,
                const std::vector<int> & trainingExamples, 
                AbstractDataStorage::ptr storage);
        
        /**
         * Each leaf node has an associated kernel density estimator.
         */
        std::vector<KernelDensityEstimator> kernelEstimators;
        /**
         * Kernel used at the leaf node.
         */
        MultivariateKernel* kernel;
        /**
         * Bandwidth selection method to use in leaf nodes.
         */
        int bandwidthSelectionMethod;
        
    };
}

#endif	/* LIBF_UNSUPERVISED_LEARNING_H */

