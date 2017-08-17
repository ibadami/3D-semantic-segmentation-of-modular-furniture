#ifndef LIBF_LEARNING_H
#define LIBF_LEARNING_H

#include <functional>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <random>
#include <thread>
#include "error_handling.h"
#include "data.h"
#include "classifier.h"
#include "ncurses.h"

namespace libf {
    /**
     * AbstractLearner: Combines all common functionality of offline and online
     * learners. It allows you to set a callback function that is called very n
     * iterations of the respective training algorithm.
     */
    template<class T, class S>
    class AbstractLearner {
    public:
        /**
         * Registers a callback function that is called every cycle iterations. 
         * 
         * @param callback The callback function
         * @param cycle The number of cycles in between the function calls
         */
        void addCallback(const std::function<int(std::shared_ptr<T>, const S &)> & callback, int cycle)
        {
            callbacks.push_back(callback);
            callbackCycles.push_back(cycle);
        }
        
    protected:
        /**
         * Calls the callbacks. The results of the callbacks are bitwise or'ed
         */
        int evokeCallback(std::shared_ptr<T> learnedObject, int iteration, const S & state) const
        {
            int result = 0;
            
            // Check all callbacks
            for (size_t i = 0; i < callbacks.size(); i++)
            {
                if ((iteration % callbackCycles[i]) == 0 )
                {
                    // It's time to call this function 
                    result = result | callbacks[i](learnedObject, state);
                }
            }
            
            return result;
        }
        
    private:
        /**
         * The callback functions.
         */
        std::vector<std::function<int(std::shared_ptr<T>, const S &)>  > callbacks;
        /**
         * The learning cycle. The callback is called very cycle iterations
         */
        std::vector<int> callbackCycles;
    };
    
    /**
     * Abstract learner state for measuring time and defining actions.
     */
    class AbstractLearnerState {
    public:
        AbstractLearnerState() : 
                action(0),
                startTime(std::chrono::high_resolution_clock::now()), 
                terminated(false), 
                started(false) {}
        
        /**
         * The current action
         */
        int action;
        /**
         * The start time
         */
        std::chrono::high_resolution_clock::time_point startTime;
        /**
         * Whether the learning has terminated
         */
        bool terminated;
        /**
         * Whether the learning has started yet
         */
        bool started;
        
        /**
         * Returns the passed time in microseconds
         * 
         * @return The time passed since instantiating the state object
         */
        std::chrono::microseconds getPassedTime() const
        {
            std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>( now - startTime );
        }
        
        /**
         * Returns the passed time in seconds
         * 
         * @return The time passed since instantiating the state object
         */
        float getPassedTimeInSeconds() const
        {
            return static_cast<float>(getPassedTime().count()/1000000);
        }
    };
    
    /**
     * This interface defines the API of offline learners. T is the learned
     * object. 
     */
    template<class T>
    class OfflineLearnerInterface {
    public:
        typedef T HypothesisType;
        
        /**
         * Learns a classifier.
         * 
         * @param storage The storage to train the classifier on
         * @return The trained classifier
         */
        virtual std::shared_ptr<T> learn(AbstractDataStorage::ptr storage) = 0;
    };
    
    /**
     * This interface defines the API of online learners. T is the learned
     * object. 
     */
    template<class T>
    class OnlineLearnerInterface {
    public:
        typedef T HypothesisType;
        
        /**
         * Learns a classifier.
         * 
         * @param storage The storage to train the classifier on
         * @return The trained classifier
         */
        virtual std::shared_ptr<T> learn(AbstractDataStorage::ptr storage) = 0;
        
        /**
         * Updates an already learned classifier. 
         * 
         * @param storage The storage to train the classifier on
         * @param classifier The base classifier
         * @return The trained classifier
         */
        virtual std::shared_ptr<T> learn(AbstractDataStorage::ptr storage, std::shared_ptr<T> classifier) = 0;
    };
    
    /**
     * This is an abstract decision tree learning providing functionality
     * needed for all decision tree learners (online or offline).
     */
    template<class M, class S>
    class AbstractDecisionTreeLearner : public AbstractLearner<M, S> {
    public:
        AbstractDecisionTreeLearner() : 
                numFeatures(10), 
                maxDepth(100), 
                minSplitExamples(3),
                minChildSplitExamples(1) {}
                
        /**
         * Sets the number of features that are required to perform a split. If 
         * there are less than the specified number of training examples at a 
         * node, it won't be split and becomes a leaf node. 
         * 
         * @param minSplitExamples The minimum number of examples required to split a node
         */
        void setMinSplitExamples(int minSplitExamples) 
        {
            BOOST_ASSERT(minSplitExamples >= 0);
            this->minSplitExamples = minSplitExamples;
        }

        /**
         * Returns the minimum number of training examples required in order
         * to split a node. 
         * 
         * @return The minimum number of training examples required to split a node
         */
        int getMinSplitExamples() const 
        {
            return minSplitExamples;
        }

        /**
         * Sets the maximum depth of a tree where the root node receives depth
         * 0. 
         * 
         * @param maxDepth the max depth
         */
        void setMaxDepth(int maxDepth) 
        {
            BOOST_ASSERT(maxDepth >= 0);
            
            this->maxDepth = maxDepth;
        }

        /**
         * Returns the maximum depth of a tree where the root node has depth 0. 
         * 
         * @return The maximum depth of a tree
         */
        int getMaxDepth() const 
        {
            return maxDepth;
        }

        /**
         * Sets the number of random features that shall be evaluated. If the
         * number of features equals the total number of dimensions, then we
         * train an ordinary decision tree. 
         * 
         * @param numFeatures The number of features to evaluate. 
         */
        void setNumFeatures(int numFeatures) 
        {
            BOOST_ASSERT(numFeatures >= 1);
            
            this->numFeatures = numFeatures;
        }

        /**
         * Returns the number of random features that shall be evaluated
         * 
         * @return The number of features to evaluate
         */
        int getNumFeatures() const 
        {
            return numFeatures;
        }
        
        /**
         * Sets the minimum number of examples that need to be in both child nodes
         * in order to perform a split. 
         * 
         * @param _minChildSplitExamples The required number of examples
         */
        void setMinChildSplitExamples(int _minChildSplitExamples)
        {
            BOOST_ASSERT(_minChildSplitExamples >= 0);
            minChildSplitExamples = _minChildSplitExamples;
        }
        
        /**
         * Returns the minimum number of examples that need to be in both child nodes
         * in order to perform a split. 
         * 
         * @return The required number of examples
         */
        int getMinChildSplitExamples() const
        {
            return minChildSplitExamples;
        }
        
    protected:
        
        /**
         * The number of random features that shall be evaluated for each 
         * split.
         */
        int numFeatures;
        /**
         * Maximum depth of the tree
         */
        int maxDepth;
        /**
         * The minimum number of training examples in a node that are required
         * in order to split the node further
         */
        int minSplitExamples;
        /**
         * The minimum number of examples that need to be in both child nodes
         * in order to perform a split. 
         */
        int minChildSplitExamples;
    };
    
    /**
     * This is a an abstract random forest learner providing functionality for
     * online and offline learning.
     */
    template<class M, class S>
    class AbstractRandomForestLearner : public AbstractLearner<M, S> {
    public:
        
        AbstractRandomForestLearner() : numTrees(8), numThreads(1) {}
        
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
        
        /**
         * Sets the number of threads
         */
        void setNumThreads(int _numThreads)
        {
            numThreads = _numThreads;
        }
        
        /**
         * Returns the number of threads
         */
        int getNumThreads() const
        {
            return numThreads;
        }
        
    protected:
        /**
         * The number of trees that we shall learn
         */
        int numTrees;
        /**
         * The number of threads that shall be used to learn the forest
         */
        int numThreads;
    };
    
    /**
     * This class creates a GUI that outputs the learner's state from time to 
     * time. The template parameter names the learning class. The class has
     * to have a subclass called "State". 
     */
    template <class L>
    class ConsoleGUI {
    public:
        ConsoleGUI(const typename L::State & state, const std::function<void(const typename L::State &)> & callback) : 
                state(state), 
                callback(callback)
        {
            workerThread = std::thread(& ConsoleGUI<L>::worker, this);
        }
        
        /**
         * This function creates the console output. 
         */
        void worker()
        {
            // Start the ncurses environment
            initscr();
            while (!state.terminated)
            {
                clear();
                // Did the learner start yet?
                if (!state.started)
                {
                    // Hm, this is odd
                    printw("The learner hasn't started yet.\n");
                    printw("Did you remember to call learn(storage, state) instead of learn(storage)?\n");
                }
                else
                {
                    printw("Runtime: %.2fs\n", state.getPassedTimeInSeconds());
                    callback(state);
                }
                
                refresh();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            endwin();
        }
        
        /**
         * Waits for the worker thread to finish
         */
        void join()
        {
            workerThread.join();
            std::cout << "Training completed in " << state.getPassedTimeInSeconds() << "s." << std::endl; 
        }
        
    protected:
        /**
         * The state that is watched
         */
        const typename L::State & state;
        /**
         * The callback function that creates the GUI
         */
        std::function<void(const typename L::State &)> callback;
        /**
         * The worker thread
         */
        std::thread workerThread;
    };
}

#endif
