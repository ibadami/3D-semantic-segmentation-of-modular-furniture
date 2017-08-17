#ifndef LIBF_CLASSIFIERLEARNINGOFFLINE_H
#define LIBF_CLASSIFIERLEARNINGOFFLINE_H

#include <functional>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <random>

#include "error_handling.h"
#include "data.h"
#include "classifier.h"
#include "learning.h"
#include "learning_tools.h"

namespace libf {
    class DecisionTreeLearnerState : public AbstractLearnerState {
    public:
        DecisionTreeLearnerState() : 
                AbstractLearnerState(), 
                total(0), 
                processed(0), 
                depth(0), 
                numNodes(0), 
                objective(0) {}

        /**
         * The total number of training examples
         */
        int total;
        /**
         * The number of processed examples
         */
        int processed;
        /**
         * The depth of the tree
         */
        int depth;
        /**
         * The total number of nodes
         */
        int numNodes;
        /**
         * The current objective
         */
        float objective;
    };
    
    /**
     * This is an ordinary offline decision tree learning algorithm. It learns the
     * tree using the information gain criterion.
     */
    class DecisionTreeLearner : 
            public AbstractDecisionTreeLearner<DecisionTree, DecisionTreeLearnerState>, 
            public OfflineLearnerInterface<DecisionTree> {
    public:
        
        /**
         * This is the learner state for the GUI
         */
        class State : public AbstractLearnerState {
        public:
            State() : 
                    AbstractLearnerState(), 
                    total(0), 
                    processed(0), 
                    depth(0), 
                    numNodes(0) {}

            /**
             * Resets the state.
             */
            void reset()
            {
                started = false;
                terminated = false;
                total = 0;
                processed = 0;
                depth = 0;
                numNodes = 0;
            }
            
            /**
             * The total number of training examples
             */
            int total;
            /**
             * The number of processed examples
             */
            int processed;
            /**
             * The depth of the tree
             */
            int depth;
            /**
             * The total number of nodes
             */
            int numNodes;
        };

        DecisionTreeLearner() : AbstractDecisionTreeLearner(),
                smoothingParameter(1),
                useBootstrap(false),
                numBootstrapExamples(1) {}
                
        /**
         * The default callback for this learner.
         * 
         * @param tree The current version of the learned tree
         * @param state The current state of the learning algorithm. 
         */
        static int defaultCallback(DecisionTree::ptr tree, const DecisionTreeLearnerState & state);
        
        /**
         * Creates the default GUI for this learner
         * 
         * @param state The current learner state
         */
        static void defaultGUI(const State & state);
        
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        
        /**
         * Sets the smoothing parameter. The smoothing parameter is the value 
         * the histograms at the leaf nodes are initialized with. 
         * 
         * @param _smoothingParameter The smoothing parameter
         */
        void setSmoothingParameter(float _smoothingParameter)
        {
            smoothingParameter = _smoothingParameter;
        }
        
        /**
         * Returns the smoothing parameter. 
         * 
         * @return The smoothing parameter
         */
        float getSmoothingParameter() const
        {
            return smoothingParameter;
        }
        
        /**
         * Sets whether or not bootstrapping shall be used. If true, then the
         * tree is learned on an iid sampled subset of the training data. 
         * 
         * @param _useBootstrap If true, the bootstrap sampling is performed
         */
        void setUseBootstrap(bool _useBootstrap)
        {
            useBootstrap = _useBootstrap;
        }

        /**
         * Returns whether or not bootstrapping is used
         * 
         * @return True if bootstrap sampling is performed
         */
        bool getUseBootstrap() const
        {
            return useBootstrap;
        }
        
        /**
         * Sets the number of samples to use for bootstrapping.
         * 
         * @param _numBootstrapExamples The number of bootstrap samples
         */
        void setNumBootstrapExamples(int _numBootstrapExamples)
        {
            numBootstrapExamples = _numBootstrapExamples;
        }
        
        /**
         * Returns the number of samples used for bootstrapping.
         * 
         * @return The number of bootstrap samples
         */
        int getNumBootstrapExamples() const
        {
            return numBootstrapExamples;
        }
        
        
        /**
         * Learns a decision tree on a data set.
         * 
         * @param storage The training set
         * @param state The learning state
         * @return The learned tree
         */
        virtual DecisionTree::ptr learn(AbstractDataStorage::ptr storage, State & state);
        
        /**
         * Learns a decision tree on a data set.
         * 
         * @param storage The training set
         * @return The learned tree
         */
        virtual DecisionTree::ptr learn(AbstractDataStorage::ptr storage)
        {
            State state;
            return this->learn(storage, state);
        }
        
    protected:
        /**
         * The smoothing parameter for the histograms
         */
        float smoothingParameter;
        /**
         * Whether or not bootstrapping shall be used
         */
        bool useBootstrap;
        /**
         * The number of bootstrap examples that shall be used.
         */
        int numBootstrapExamples;
    };
    
    
    /**
     * This is a projective decision tree learning algorithm. It learns the
     * tree using the information gain criterion.
     */
    class ProjectiveDecisionTreeLearner : 
            public AbstractDecisionTreeLearner<ProjectiveDecisionTree, DecisionTreeLearnerState>, 
            public OfflineLearnerInterface<ProjectiveDecisionTree> {
    public:
        ProjectiveDecisionTreeLearner() : AbstractDecisionTreeLearner(),
                smoothingParameter(1),
                useBootstrap(false),
                numBootstrapExamples(1) {}
                
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_SPLIT_NODE = 2;
        
        /**
         * Sets the smoothing parameter. The smoothing parameter is the value 
         * the histograms at the leaf nodes are initialized with. 
         * 
         * @param _smoothingParameter The smoothing parameter
         */
        void setSmoothingParameter(float _smoothingParameter)
        {
            smoothingParameter = _smoothingParameter;
        }
        
        /**
         * Returns the smoothing parameter. 
         * 
         * @return The smoothing parameter
         */
        float getSmoothingParameter() const
        {
            return smoothingParameter;
        }
        
        /**
         * Sets whether or not bootstrapping shall be used. If true, then the
         * tree is learned on an iid sampled subset of the training data. 
         * 
         * @param _useBootstrap If true, the bootstrap sampling is performed
         */
        void setUseBootstrap(bool _useBootstrap)
        {
            useBootstrap = _useBootstrap;
        }

        /**
         * Returns whether or not bootstrapping is used
         * 
         * @return True if bootstrap sampling is performed
         */
        bool getUseBootstrap() const
        {
            return useBootstrap;
        }
        
        /**
         * Sets the number of samples to use for bootstrapping.
         * 
         * @param _numBootstrapExamples The number of bootstrap samples
         */
        void setNumBootstrapExamples(int _numBootstrapExamples)
        {
            numBootstrapExamples = _numBootstrapExamples;
        }
        
        /**
         * Returns the number of samples used for bootstrapping.
         * 
         * @return The number of bootstrap samples
         */
        int getNumBootstrapExamples() const
        {
            return numBootstrapExamples;
        }
        
        /**
         * Learns a decision tree on a data set.
         * 
         * @param storage The training set
         * @return The learned tree
         */
        virtual ProjectiveDecisionTree::ptr learn(AbstractDataStorage::ptr storage);
        
    protected:
        /**
         * The smoothing parameter for the histograms
         */
        float smoothingParameter;
        /**
         * Whether or not bootstrapping shall be used
         */
        bool useBootstrap;
        /**
         * The number of bootstrap examples that shall be used.
         */
        int numBootstrapExamples;
    };
    
    /**
     * This class holds the current state of the random forest learning
     * algorithm.
     */
    class RandomForestLearnerState : public AbstractLearnerState {
    public:
        RandomForestLearnerState() : AbstractLearnerState(),
                tree(0),
                numTrees(0) {}
        
        /**
         * The current tree
         */
        int tree;
        /**
         * Total number of trees learned.
         */
        int numTrees;
    };
    
    /**
     * This is an offline random forest learner. T is the classifier learner. 
     */
    template <class L>
    class RandomForestLearner : 
            public AbstractRandomForestLearner<RandomForest<typename L::HypothesisType>, RandomForestLearnerState>,
            public OfflineLearnerInterface< RandomForest<typename L::HypothesisType> > {
    public:
        
        /**
         * This is the learner state for the GUI
         */
        class State : public AbstractLearnerState {
        public:
            State() : 
                    AbstractLearnerState(), 
                    total(0), 
                    processed(0) {}

            /**
             * The total number of trees to process
             */
            int total;
            /**
             * The number of learned trees
             */
            int processed;
            /**
             * The states of the individual tree learners (per thread)
             */
            std::vector<typename L::State> treeLearnerStates;
        };
        
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        /**
         * The default callback for this learner.
         */
        static int verboseCallback(typename RandomForest<typename L::HypothesisType>::ptr forest, const RandomForestLearnerState & state)
        {
            switch (state.action) {
                case ACTION_START_FOREST:
                    std::cout << "Start random forest training" << "\n";
                    break;
                case ACTION_START_TREE:
                    std::cout << std::setw(15) << std::left << "Start tree " 
                            << std::setw(4) << std::right << state.tree 
                            << " out of " 
                            << std::setw(4) << state.numTrees << "\n";
                    break;
                case ACTION_FINISH_TREE:
                    std::cout << std::setw(15) << std::left << "Finish tree " 
                            << std::setw(4) << std::right << state.tree 
                            << " out of " 
                            << std::setw(4) << state.numTrees << "\n";
                    break;
                case ACTION_FINISH_FOREST:
                    std::cout << "Finished forest in " << state.getPassedTime().count()/1000000. << "s\n";
                    break;
                default:
                    std::cout << "UNKNOWN ACTION CODE " << state.action << "\n";
                    break;
            }
            return 0;
        }
        
        static int defaultCallback(typename RandomForest<typename L::HypothesisType>::ptr forest, const RandomForestLearnerState & state)
        {
            switch (state.action) {
                case ACTION_START_FOREST:
                    std::cout << "Start random forest training" << "\n";
                    break;
                case ACTION_FINISH_FOREST:
                    std::cout << "Finished forest in " << state.getPassedTime().count()/1000000. << "s\n";
                    break;
            }

            return 0;
        }

        /**
         * Creates the default GUI for this learner
         * 
         * @param state The current learner state
         */
        static void defaultGUI(const State & state)
        {
            // Show the overall progress
            float progress = 0;
            if (state.total > 0)
            {
                progress = state.processed / static_cast<float>(state.total);
            }
            printw("Random Forest Progress: %4d/%4d Trees learned\n", state.processed, state.total);
            GUIUtil::printProgressBar(progress);
            printw("\n");
            
            for(size_t t = 0; t < state.treeLearnerStates.size(); t++)
            {
                printw("Thread %d\n", static_cast<int>(t)+1);
                L::defaultGUI(state.treeLearnerStates[t]);
            }
        }
        
        /**
         * Sets the decision tree learner
         */
        void setTreeLearner(const L & _treeLearner)
        {
            treeLearner = _treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        const L & getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        L & getTreeLearner()
        {
            return treeLearner;
        }
        
        /**
         * Learns a forests. 
         */
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage, State & state)
        {
            state.started = true;
            
            // Set up the empty random forest
            auto forest = std::make_shared< RandomForest<typename L::HypothesisType> >();

            // Set up the state for the call backs
            state.total = this->getNumTrees();
            state.treeLearnerStates.resize(this->getNumThreads());

            #pragma omp parallel for num_threads(this->numThreads)
            for (int i = 0; i < this->getNumTrees(); i++)
            {
                // Learn the tree
                auto tree = treeLearner.learn(storage, state.treeLearnerStates[omp_get_thread_num()]);
                
                // Add it to the forest
                #pragma omp critical
                {
                    state.processed++;
                    forest->addTree(tree);
                }
            }
            
            state.terminated = true;
            
            return forest;
        }

        /**
         * Learns a forests. 
         */
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage)
        {
            State state;
            return this->learn(storage, state);
        }
        
    protected:
        /**
         * The tree learner
         */
        L treeLearner;
    };
    
    /**
     * This is an online random forest learner. T is the classifier learner. 
     */
    template <class L>
    class OnlineRandomForestLearner : 
            public AbstractRandomForestLearner<RandomForest<typename L::HypothesisType>, RandomForestLearnerState>,
            public OnlineLearnerInterface< RandomForest<typename L::HypothesisType> > {
    public:
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        /**
         * The default callback for this learner.
         */
        static int verboseCallback(typename RandomForest<typename L::HypothesisType>::ptr forest, const RandomForestLearnerState & state)
        {
            switch (state.action) {
                case ACTION_START_FOREST:
                    std::cout << "Start random forest training" << "\n";
                    break;
                case ACTION_START_TREE:
                    std::cout << std::setw(15) << std::left << "Start tree " 
                            << std::setw(4) << std::right << state.tree 
                            << " out of " 
                            << std::setw(4) << state.numTrees << "\n";
                    break;
                case ACTION_FINISH_TREE:
                    std::cout << std::setw(15) << std::left << "Finish tree " 
                            << std::setw(4) << std::right << state.tree 
                            << " out of " 
                            << std::setw(4) << state.numTrees << "\n";
                    break;
                case ACTION_FINISH_FOREST:
                    std::cout << "Finished forest in " << state.getPassedTime().count()/1000000. << "s\n";
                    break;
                default:
                    std::cout << "UNKNOWN ACTION CODE " << state.action << "\n";
                    break;
            }
            return 0;
        }
        
        static int defaultCallback(typename RandomForest<typename L::HypothesisType>::ptr forest, const RandomForestLearnerState & state)
        {
            switch (state.action) {
                case OnlineRandomForestLearner::ACTION_START_FOREST:
                    std::cout << "Start random forest training" << "\n";
                    break;
                case OnlineRandomForestLearner::ACTION_FINISH_FOREST:
                    std::cout << "Finished forest in " << state.getPassedTime().count()/1000000. << "s\n";
                    break;
            }

            return 0;
        }

        /**
         * Sets the decision tree learner
         */
        void setTreeLearner(const L & _treeLearner)
        {
            treeLearner = _treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        const L & getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        L & getTreeLearner()
        {
            return treeLearner;
        }
        
        /**
         * Updates an already learned classifier. 
         * 
         * @param storage The storage to train the classifier on
         * @param classifier The base classifier
         * @return The trained classifier
         */
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(
            AbstractDataStorage::ptr storage, 
            typename RandomForest<typename L::HypothesisType>::ptr forest)
        {
            // Add the required number of trees if there are too few trees in 
            // the forest
            for (int i = forest->getSize(); i < this->getNumTrees(); i++)
            {
                auto tree = std::make_shared<typename L::HypothesisType>();
                // TODO: Move this to a factory or the constructor
                tree->addNode();
                
                forest->addTree(tree);
            }
            
            const int D = storage->getDimensionality();

            // Initialize variable importance values.
            // TODO: Update importance calculation
            // importance = std::vector<float>(D, 0.f);

            // Set up the state for the call backs
            RandomForestLearnerState state;
            state.tree = 0;
            state.numTrees = this->getNumTrees();
            state.action = ACTION_START_FOREST;

            this->evokeCallback(forest, 0, state);

            int treeStartCounter = 0; 
            int treeFinishCounter = 0; 

            #pragma omp parallel for num_threads(this->numThreads)
            for (int i = 0; i < this->numTrees; i++)
            {
                #pragma omp critical
                {
                    state.tree = ++treeStartCounter;
                    state.action = ACTION_START_TREE;

                    this->evokeCallback(forest, treeStartCounter - 1, state);
                }

                auto tree = forest->getTree(i);
                this->treeLearner.learn(storage, tree);
                
                #pragma omp critical
                {
                    state.tree = ++treeFinishCounter;
                    state.action = ACTION_FINISH_TREE;

                    this->evokeCallback(forest, treeFinishCounter - 1, state);

                    // Update variable importance.
                    for (int f = 0; f < D; ++f)
                    {
                        // TODO: Update importance calculation
                        // importance[f] += treeLearner.getImportance(f);
                    }
                }
            }

            state.tree = 0;
            state.action = ACTION_FINISH_FOREST;
            this->evokeCallback(forest, 0, state);

            return forest;
        }
        
        /**
         * Learns a forests. 
         */
        virtual typename RandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage)
        {
            auto forest = std::make_shared<RandomForest<typename L::HypothesisType> > ();
            return learn(storage, forest);
        }

    protected:
        /**
         * The tree learner
         */
        L treeLearner;
    };
    
    /**
     * This class holds the current state of the boosted random forest learning
     * algorithm.
     */
    class BoostedRandomForestLearnerState : public AbstractLearnerState {
    public:
        BoostedRandomForestLearnerState() : AbstractLearnerState(),
                tree(0),
                numTrees(0),
                error(0), 
                alpha(0) {}
        
        /**
         * The current tree
         */
        int tree;
        /**
         * Number of learned trees.
         */
        int numTrees;
        /**
         * The error value
         */
        float error;
        /**
         * The tree weight
         */
        float alpha;
    };
    
    /**
     * This is a random forest learner. 
     */
    template <class L>
    class BoostedRandomForestLearner : 
            public AbstractLearner<BoostedRandomForest<typename L::HypothesisType>, BoostedRandomForestLearnerState>, 
            public OfflineLearnerInterface< BoostedRandomForest<typename L::HypothesisType> > {
    public:
        typedef std::shared_ptr<BoostedRandomForestLearner<L> > ptr;
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(typename BoostedRandomForest<typename L::HypothesisType>::ptr forest, const BoostedRandomForestLearnerState & state)
        {
            switch (state.action) {
                case ACTION_START_FOREST:
                    std::cout << "Start boosted random forest training\n" << "\n";
                    break;
                case ACTION_START_TREE:
                    std::cout   << std::setw(15) << std::left << "Start tree " 
                                << std::setw(4) << std::right << state.tree 
                                << " out of " 
                                << std::setw(4) << state.numTrees << "\n";
                    break;
                case ACTION_FINISH_TREE:
                    std::cout   << std::setw(15) << std::left << "Finish tree " 
                                << std::setw(4) << std::right << state.tree 
                                << " out of " 
                                << std::setw(4) << state.numTrees
                                << " error = " << state.error 
                                << ", alpha = " << state.alpha << "\n";
                    break;
                case ACTION_FINISH_FOREST:
                    std::cout << "Finished boosted forest in " << state.getPassedTime().count()/1000000. << "s\n";
                    break;
                default:
                    std::cout << "UNKNOWN ACTION CODE " << state.action << "\n";
                    break;
            }

            return 0;
        }
        
        /**
         * These are the actions of the learning algorithm that are passed
         * to the callback functions.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_FINISH_TREE = 2;
        const static int ACTION_START_FOREST = 3;
        const static int ACTION_FINISH_FOREST = 4;
        
        /**
         * Sets the decision tree learner
         */
        void setTreeLearner(const L & _treeLearner)
        {
            treeLearner = _treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        const L & getTreeLearner() const
        {
            return treeLearner;
        }
        
        /**
         * Returns the decision tree learner
         */
        L & getTreeLearner()
        {
            return treeLearner;
        }
        
        /**
         * Learns a forests. 
         */
        virtual typename BoostedRandomForest<typename L::HypothesisType>::ptr learn(AbstractDataStorage::ptr storage)
        {
            // Set up the empty random forest
             auto forest = std::make_shared< BoostedRandomForest<typename L::HypothesisType> >();

            // Set up the state for the call backs
            BoostedRandomForestLearnerState state;
            state.numTrees = this->getNumTrees();
            state.tree = 0;
            state.action = ACTION_START_FOREST;

            this->evokeCallback(forest, 0, state);

            // Set up the weights for the data points
            const int N = storage->getSize();
            std::vector<float> dataWeights(N);
            std::vector<float> cumsum(N);
            std::vector<bool> misclassified(N);
            for (int n = 0; n < N; n++)
            {
                dataWeights[n] = 1.0f/N;
                cumsum[n] = (n+1) * 1.0f/N;
                misclassified[n] = false;
            }

            // We need this distribution in order to sample according to the weights
            std::random_device rd;
            std::mt19937 g(rd());
            std::uniform_real_distribution<float> U(0, 1);

            const int C = storage->getClasscount();

            int treeStartCounter = 0; 
            int treeFinishCounter = 0; 
            for (int i = 0; i < this->numTrees; i++)
            {
                state.tree = ++treeStartCounter;
                state.action = ACTION_START_TREE;
                this->evokeCallback(forest, treeStartCounter - 1, state);

                // Learn the tree
                // --------------

                // Sample data points according to the weights
                ReferenceDataStorage::ptr treeData = std::make_shared<ReferenceDataStorage>(storage);

                for (int n = 0; n < N; n++)
                {
                    const float u = U(g);
                    int index = 0;
                    while (u > cumsum[index] && index < N-1)
                    {
                        index++;
                    }
                    treeData->addDataPoint(index);
                }

                // Learn the tree
                auto tree = treeLearner.learn(treeData);

                // Calculate the error term
                float error = 0;
                for (int n = 0; n < N; n++)
                {
                    const int predictedLabel = tree->classify(storage->getDataPoint(n));
                    if (predictedLabel != storage->getClassLabel(n))
                    {
                        error += dataWeights[n];
                        misclassified[n] = true;
                    }
                    else
                    {
                        misclassified[n] = false;
                    }
                }

                // Compute the classifier weight
                const float alpha = std::log((1-error)/error) + std::log(C - 1);

                // Update the weights
                float total = 0;
                for (int n = 0; n < N; n++)
                {
                    if (misclassified[n])
                    {
                        dataWeights[n] *= std::exp(alpha);
                    }
                    total += dataWeights[n];
                }
                dataWeights[0] /= total;
                cumsum[0] = dataWeights[0];
                for (int n = 1; n < N; n++)
                {
                    dataWeights[n] /= total;
                    cumsum[n] = dataWeights[n] + cumsum[n-1];
                }

                // Create the weak classifier
                auto weakClassifier = std::make_shared<WeakClassifier<typename L::HypothesisType> >();
                weakClassifier->setClassifier(tree);
                weakClassifier->setWeight(alpha);
                
                // Add the classifier
                forest->addTree(weakClassifier);

                // --------------
                // Add it to the forest
                state.tree = ++treeFinishCounter;
                state.error = error;
                state.alpha = alpha;
                state.action = ACTION_FINISH_TREE;
                this->evokeCallback(forest, treeFinishCounter - 1, state);
            }

            state.tree = 0;
            state.action = ACTION_FINISH_FOREST;
            this->evokeCallback(forest, 0, state);

            return forest;
        }
        
        /**
         * Sets the number of trees. 
         */
        void setNumTrees(int _numTrees)
        {
            BOOST_ASSERT(_numTrees >= 1);
            numTrees = _numTrees;
        }
        
        /**
         * Returns the number of trees
         */
        int getNumTrees() const
        {
            return numTrees;
        }
        
    protected:
        /**
         * The number of trees that we shall learn
         */
        int numTrees;
        /**
         * The tree learner
         */
        L treeLearner;
    };
    
    class OnlineDecisionTreeLearnerState : public AbstractLearnerState {
    public:
        OnlineDecisionTreeLearnerState() : AbstractLearnerState(),
                node(0),
                objective(0), 
                depth(0) {}
        
        /**
         * Node id.
         */
        int node;
        /**
         * Samples of node.
         */
        int samples;
        /**
         * Objective of splitted node.
         */
        float objective;
        /**
         * Minimum require dobjective.
         */
        float minObjective;
        /**
         * Depth of spitted node.
         */
        int depth;
    };
    
    /**
     * Learn a decision tree online, either by passing a single sample at a time
     * or doing online batch learning.
     */
    class OnlineDecisionTreeLearner :
            public AbstractDecisionTreeLearner<OnlineDecisionTree, OnlineDecisionTreeLearnerState>,
            public OnlineLearnerInterface<OnlineDecisionTree> {
    public:
        OnlineDecisionTreeLearner() : AbstractDecisionTreeLearner(),
                smoothingParameter(1),
                useBootstrap(true),
                bootstrapLambda(1.f),
                numThresholds(2*numFeatures),
                minSplitObjective(1.f)
        {
            // Overwrite min split examples.
            minSplitExamples = 30;
            minChildSplitExamples = 15;
        }
        
        /**
         * The default callback for this learner.
         */
        static int defaultCallback(OnlineDecisionTree::ptr tree, const OnlineDecisionTreeLearnerState & state);
        
        /**
         * Verbose callback for this learner.
         */
        static int verboseCallback(OnlineDecisionTree::ptr tree, const OnlineDecisionTreeLearnerState & state);
        
        /**
         * Actions for the callback function.
         */
        const static int ACTION_START_TREE = 1;
        const static int ACTION_UPDATE_TREE = 2;
        const static int ACTION_INIT_NODE = 3;
        const static int ACTION_NOT_SPLITTING_NODE = 4;
        const static int ACTION_NOT_SPLITTING_OBJECTIVE_NODE = 5;
        const static int ACTION_SPLIT_NODE = 6;
        
        /**
         * Sets the smoothing parameter
         */
        void setSmoothingParameter(float _smoothingParameter)
        {
            smoothingParameter = _smoothingParameter;
        }
        
        /**
         * Returns the smoothing parameter
         */
        float getSmoothingParameter() const
        {
            return smoothingParameter;
        }
        
        /**
         * Sets whether or not bootstrapping shall be used
         */
        void setUseBootstrap(bool _useBootstrap)
        {
            useBootstrap = _useBootstrap;
        }

        /**
         * Returns whether or not bootstrapping is used
         */
        bool getUseBootstrap() const
        {
            return useBootstrap;
        }
        
        /**
         * Sets the minimum objective required for a split.
         */
        void setMinSplitObjective(float _minSplitObjective)
        {
            assert(_minSplitObjective > 0);
            minSplitObjective = _minSplitObjective;
        }
        
        /**
         * Returns the minimum objective required for a split.
         */
        float getMinSplitObjective() const
        {
            return minSplitObjective;
        }
        
        /**
         * Sets the number of thresholds randomly sampled for each node.
         */
        void setNumThresholds(int _numThresholds)
        {
            assert(_numThresholds > 0);
            numThresholds = _numThresholds;
        }
        
        /**
         * Returns the number of thresholds randomly sampled for each node.
         */
        int getNumThresholds() const
        {
            return numThresholds;
        }
        
        /**
         * Sets the threshold generator to use.
         */
        void setThresholdGenerator(RandomThresholdGenerator & _thresholdGenerator)
        {
            thresholdGenerator = _thresholdGenerator;
        }
        
        /**
         * Returns the used threshold generator.
         */
        RandomThresholdGenerator & getThresholdGenerator()
        {
            return thresholdGenerator;
        }
        
        /**
         * Learns a decision tree.
         */
        virtual OnlineDecisionTree::ptr learn(AbstractDataStorage::ptr storage);
        
        /**
         * Updates the given decision tree on the given data.
         */
        virtual OnlineDecisionTree::ptr learn(AbstractDataStorage::ptr storage, OnlineDecisionTree::ptr tree);
        
    protected:
        /**
         * For all splits, update left and right child statistics.
         */
        void updateSplitStatistics(std::vector<EfficientEntropyHistogram> & leftChildStatistics, 
                std::vector<EfficientEntropyHistogram> & rightChildStatistics, 
                const std::vector<int> & features,
                const std::vector< std::vector<float> > & thresholds, 
                const DataPoint & x, const int label);
        
        /**
         * The smoothing parameter for the histograms
         */
        float smoothingParameter;
        /**
         * Whether or not bootstrapping shall be used
         */
        bool useBootstrap;
        /**
         * Lambda used for poisson distribution for online bootstrapping.
         */
        float bootstrapLambda;
        /**
         * Number of thresholds randomly sampled. Together with the sampled
         * features these define the tests over which to optimize at
         * each node in online learning.
         * 
         * @see http://lrs.icg.tugraz.at/pubs/saffari_olcv_09.pdf
         */
        int numThresholds;
        /**
         * Minimum objective required for a node to split.
         */
        float minSplitObjective;
        /**
         * The generator to sample random thresholds.
         */
        RandomThresholdGenerator thresholdGenerator;
    };
}

#endif
