#include "libforest/data.h"
#include "libforest/classifier.h"
#include "libforest/util.h"
#include "libforest/learning.h"
#include "libforest/learning_tools.h"
#include "libforest/classifier_learning.h"
#include "libforest/classifier_learning_tools.h"
#include "ncurses.h"

#include <algorithm>
#include <random>
#include <map>
#include <iomanip>
#include <queue>
#include <stack>

using namespace libf;

std::random_device rd;

////////////////////////////////////////////////////////////////////////////////
/// DecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////

/**
 * Updates the leaf node histograms using a smoothing parameter
 */
inline void updateLeafNodeHistogram(std::vector<float> & leafNodeHistogram, const EfficientEntropyHistogram & hist, float smoothing, bool useBootstrap)
{
    const int C = hist.getSize();
    
    leafNodeHistogram.resize(C);
    BOOST_ASSERT(leafNodeHistogram.size() > 0);
    
    if(!useBootstrap)
    {
        for (int c = 0; c < C; c++)
        {
            leafNodeHistogram[c] = std::log((hist.at(c) + smoothing)/(hist.getMass() + hist.getSize() * smoothing));
        }
    }
}

DecisionTree::ptr DecisionTreeLearner::learn(AbstractDataStorage::ptr dataStorage, State & state)
{
    state.reset();
    state.started = true;
    
    BOOST_ASSERT(numFeatures <= dataStorage->getDimensionality());
    
    AbstractDataStorage::ptr storage;
    // If we use bootstrap sampling, then this array contains the results of 
    // the sampler. We use it later in order to refine the leaf node histograms
    std::vector<bool> sampled;
    
    if (useBootstrap)
    {
        storage = dataStorage->bootstrap(numBootstrapExamples, sampled);
    }
    else
    {
        storage = dataStorage;
    }
    
    // Get the number of training examples and the dimensionality of the data set
    const int D = storage->getDimensionality();
    const int C = storage->getClasscount();
    
    state.total = storage->getSize();
    
    // Set up a new tree. 
    DecisionTree::ptr tree = std::make_shared<DecisionTree>();
    tree->addNode();
    
    // This is the list of nodes that still have to be split
    std::vector<int> splitStack;
    splitStack.reserve(static_cast<int>(fastlog2(storage->getSize())));
    
    // Add the root node to the list of nodes that still have to be split
    splitStack.push_back(0);
    
    // This matrix stores the training examples for certain nodes. 
    std::vector< int* > trainingExamples;
    std::vector< int > trainingExamplesSizes;
    trainingExamples.reserve(LIBF_GRAPH_BUFFER_SIZE);
    trainingExamplesSizes.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    // Add all training example to the root node
    trainingExamplesSizes.push_back(storage->getSize());
    trainingExamples.push_back(new int[trainingExamplesSizes[0]]);
    for (int n = 0; n < storage->getSize(); n++)
    {
        trainingExamples[0][n] = n;
    }
    
    // We use these arrays during training for the left and right histograms
    EfficientEntropyHistogram leftHistogram(C);
    EfficientEntropyHistogram rightHistogram(C);
    
    // We use this in order to sort the data points
    FeatureComparator cp;
    cp.storage = storage;
    
    // Set up a probability distribution over the features
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    // Set up the array of possible features, we use it in order to sample
    // the features without replacement
    std::vector<int> sampledFeatures(D);
    for (int d = 0; d < D; d++)
    {
        sampledFeatures[d] = d;
    }

    // Start training
    while (splitStack.size() > 0)
    {
        // Extract an element from the queue
        const int node = splitStack.back();
        splitStack.pop_back();
        
        state.numNodes = tree->getNumNodes();
        state.depth = std::max(state.depth, tree->getNodeConfig(node).getDepth());
        
        // Get the training example list
        int* trainingExampleList = trainingExamples[node];
        const int N = trainingExamplesSizes[node];

        // Set up the right histogram
        // Because we start with the threshold being at the left most position
        // The right child node contains all training examples
        
        EfficientEntropyHistogram hist(C);
        for (int m = 0; m < N; m++)
        {
            // Get the class label of this training example
            hist.addOne(storage->getClassLabel(trainingExampleList[m]));
        }

        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reached
        if (hist.getMass() < minSplitExamples || hist.isPure() || tree->getNodeConfig(node).getDepth() >= maxDepth)
        {
            // Resize and initialize the leaf node histogram
            updateLeafNodeHistogram(tree->getNodeData(node).histogram, hist, smoothingParameter, useBootstrap);
            BOOST_ASSERT(tree->getNodeData(node).histogram.size() > 0);
            state.processed += N;
            delete[] trainingExampleList;
            trainingExamples[node] = 0;
            continue;
        }
        
        // These are the parameters we optimize
        float bestThreshold = 0;
        int bestFeature = -1;
        float bestObjective = 1e35;
        int bestLeftMass = 0;
        int bestRightMass = N;

        // Sample random features
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::shuffle(sampledFeatures.begin(), sampledFeatures.end(), std::default_random_engine(seed));
        
        // Optimize over all features
        for (int f = 0; f < numFeatures; f++)
        {
            const int feature = sampledFeatures[f];
            
            cp.feature = feature;
            std::sort(trainingExampleList, trainingExampleList + N, cp);
            
            // Initialize the histograms
            leftHistogram.reset();
            rightHistogram = hist;
            
            float leftValue = storage->getDataPoint(trainingExampleList[0])(feature);
            int leftClass = storage->getClassLabel(trainingExampleList[0]);
            
            // Test different thresholds
            // Go over all examples in this node
            for (int m = 1; m < N; m++)
            {
                const int n = trainingExampleList[m];
                
                // Move the last point to the left histogram
                leftHistogram.addOne(leftClass);
                rightHistogram.subOne(leftClass);
                        
                // It does
                // Get the two feature values
                const float rightValue = storage->getDataPoint(n)(feature);
                
                // Skip this split, if the two points lie too close together
                const float diff = rightValue - leftValue;

                if (diff < 1e-6f*std::max(std::abs(rightValue+1e-6), std::abs(leftValue+1e-6)))
                {
                    leftValue = rightValue;
                    leftClass = storage->getClassLabel(n);
                    continue;
                }
                
                // Get the objective function
                const float localObjective = leftHistogram.getEntropy()
                        + rightHistogram.getEntropy();
                
                if (localObjective < bestObjective)
                {
                    // Get the threshold value
                    bestThreshold = (leftValue + rightValue);
                    bestFeature = feature;
                    bestObjective = localObjective;
                    bestLeftMass = leftHistogram.getMass();
                    bestRightMass = rightHistogram.getMass();
                }
                
                leftValue = rightValue;
                leftClass = storage->getClassLabel(n);
            }
        }
        
        // We spare the additional multiplication at each iteration.
        bestThreshold *= 0.5f;
        
        // Did we find good split values?
        if (bestFeature < 0 || bestLeftMass < minChildSplitExamples || bestRightMass < minChildSplitExamples)
        {
            // We didn't
            // Don't split
            updateLeafNodeHistogram(tree->getNodeData(node).histogram, hist, smoothingParameter, useBootstrap);
            BOOST_ASSERT(tree->getNodeData(node).histogram.size() > 0);
            state.processed += N;
            delete[] trainingExampleList;
            trainingExamples[node] = 0;
            continue;
        }
        
        // Set up the data lists for the child nodes
        trainingExamplesSizes.push_back(bestLeftMass);
        trainingExamplesSizes.push_back(bestRightMass);
        trainingExamples.push_back(new int[bestLeftMass]);
        trainingExamples.push_back(new int[bestRightMass]);
        
        int* leftList = trainingExamples[trainingExamples.size() - 2];
        int* rightList = trainingExamples[trainingExamples.size() - 1];
        
        // Sort the points
        for (int m = 0; m < N; m++)
        {
            const int n = trainingExampleList[m];
            const float featureValue = storage->getDataPoint(n)(bestFeature);
            
            BOOST_ASSERT(!std::isnan(featureValue));
            
            if (featureValue < bestThreshold)
            {
                leftList[--bestLeftMass] = n;
            }
            else
            {
                rightList[--bestRightMass] = n;
            }
        }
        
        // Ok, split the node
        tree->getNodeConfig(node).setThreshold(bestThreshold);
        tree->getNodeConfig(node).setSplitFeature(bestFeature);
        const int leftChild = tree->splitNode(node);
        
        // Prepare to split the child nodes
        splitStack.push_back(leftChild);
        splitStack.push_back(leftChild + 1);
        
        delete[] trainingExampleList;
        trainingExamples[node] = 0;
    }
    
    // If we use bootstrap, we use all the training examples for the 
    // histograms
    if (useBootstrap)
    {
        TreeLearningTools::updateHistograms(tree, dataStorage, smoothingParameter);
    }
    
    state.terminated = true;
    
    return tree;
}

int DecisionTreeLearner::defaultCallback(DecisionTree::ptr tree, const DecisionTreeLearnerState & state)
{
    switch (state.action) {
        case DecisionTreeLearner::ACTION_START_TREE:
            std::cout << "Start decision tree training" << "\n";
            break;
        case DecisionTreeLearner::ACTION_SPLIT_NODE:
            std::cout << std::setw(15) << std::left << "Split node:"
                    << "depth = " << std::setw(3) << std::right << state.depth
                    << ", objective = " << std::setw(6) << std::left
                    << std::setprecision(4) << state.objective << "\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state.action << "\n";
            break;
    }
    
    return 0;

}

void DecisionTreeLearner::defaultGUI(const State& state)
{
    printw("Nodes: %10d Depth: %10d\n", state.numNodes, state.depth);
    float p = 0;
    if (state.total > 0)
    {
        p = state.processed/static_cast<float>(state.total);
    }
    GUIUtil::printProgressBar(p);
}

////////////////////////////////////////////////////////////////////////////////
/// ProjectiveDecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////


ProjectiveDecisionTree::ptr ProjectiveDecisionTreeLearner::learn(AbstractDataStorage::ptr dataStorage)
{
    AbstractDataStorage::ptr storage;
    // If we use bootstrap sampling, then this array contains the results of 
    // the sampler. We use it later in order to refine the leaf node histograms
    std::vector<bool> sampled;
    
    if (useBootstrap)
    {
        storage = dataStorage->bootstrap(numBootstrapExamples, sampled);
    }
    else
    {
        storage = dataStorage;
    }
    
    // Get the number of training examples and the dimensionality of the data set
    const int D = storage->getDimensionality();
    const int C = storage->getClasscount();
    
    // Set up a new tree. 
    ProjectiveDecisionTree::ptr tree = std::make_shared<ProjectiveDecisionTree>();
    tree->addNode();
    
    // Set up the state for the call backs
    DecisionTreeLearnerState state;
    state.action = ACTION_START_TREE;
    
    evokeCallback(tree, 0, state);
    
    // This is the list of nodes that still have to be split
    std::stack<int> splitStack;
    //splitStack.reserve(static_cast<int>(fastlog2(storage->getSize())));
    
    // Add the root node to the list of nodes that still have to be split
    splitStack.push(0);
    
    // This matrix stores the training examples for certain nodes. 
    std::vector< int* > trainingExamples;
    std::vector< int > trainingExamplesSizes;
    trainingExamples.reserve(LIBF_GRAPH_BUFFER_SIZE);
    trainingExamplesSizes.reserve(LIBF_GRAPH_BUFFER_SIZE);
    
    // Saves the sum of impurity decrease achieved by each feature
    // TODO: Update importance calculation
    //importance = std::vector<float>(D, 0.f);
    
    // Add all training example to the root node
    trainingExamplesSizes.push_back(storage->getSize());
    trainingExamples.push_back(new int[trainingExamplesSizes[0]]);
    for (int n = 0; n < storage->getSize(); n++)
    {
        trainingExamples[0][n] = n;
    }
    
    // We use these arrays during training for the left and right histograms
    EfficientEntropyHistogram leftHistogram(C);
    EfficientEntropyHistogram rightHistogram(C);
    
    // Set up a probability distribution over the features
    std::mt19937 g(rd());
    std::uniform_int_distribution<int> dist(0, D-1);
    std::uniform_real_distribution<float> dist2(0, 1);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist3(-1, 1);
    const int F = 4;
    std::poisson_distribution<int> poisson(F);
    
    const float s = std::sqrt(D);
    std::vector<float> projectionValues(storage->getSize(), 0.0f);
    
    int numProjections = numFeatures;
    // Set up some random projections
    std::vector<DataPoint> projections(numProjections);
    
    std::vector<int> dimensions(D);
    for (int d = 0; d < D; d++) dimensions[d] = d;
    
    // Start training
    while (splitStack.size() > 0)
    {
        for (int f = 0; f < numProjections; f++)
        {
            projections[f] = DataPoint::Zero(D);
#if 1
                for (int d = 0; d < D; d++)
                {
                    const float u = dist2(g);
                    if (u <= 0.5/s)
                    {
                        projections[f](d) = -1;
                    }
                    if ( u <= 1/s)
                    {
                        projections[f](d) = dist3(g);
                    }
                }
#endif
#if 0
                projections[f](dist(g)) = 1;
                projections[f](dist(g)) = -1;
#endif
#if 0
                for (int d = 0; d < D; d++)
                {
                    projections[f](d) = normal(g);
                }
#endif
#if 0
                projections[f](dist(g)) = 1;
#endif
#if 0
                int nnz;
                do {
                    nnz = poisson(g);
                } while (nnz == 0);
                
                for (int l = 0; l < nnz; l++)
                {
                    projections[f](dist(g)) = 2*dist2(g) - 1;
                }
#endif
        }
        // Extract an element from the queue
        const int node = splitStack.top();
        splitStack.pop();
        
        // Get the training example list
        int* trainingExampleList = trainingExamples[node];
        const int N = trainingExamplesSizes[node];

        // Set up the right histogram
        // Because we start with the threshold being at the left most position
        // The right child node contains all training examples
        
        EfficientEntropyHistogram hist(C);
        for (int m = 0; m < N; m++)
        {
            // Get the class label of this training example
            hist.addOne(storage->getClassLabel(trainingExampleList[m]));
        }

        // Don't split this node
        //  If the number of examples is too small
        //  If the training examples are all of the same class
        //  If the maximum depth is reached
        if (hist.getMass() < minSplitExamples || hist.isPure() || tree->getNodeConfig(node).getDepth() >= maxDepth)
        {
            // Resize and initialize the leaf node histogram
            updateLeafNodeHistogram(tree->getNodeData(node).histogram, hist, smoothingParameter, useBootstrap);
            BOOST_ASSERT(tree->getNodeData(node).histogram.size() > 0);
            continue;
        }
        
        // These are the parameters we optimize
        float bestThreshold = 0;
        float bestObjective = 1e35;
        int bestLeftMass = 0;
        int bestRightMass = N;
        DataPoint bestProjection(D);

        // Optimize over all features
        for (int f = 0; f < numFeatures; f++)
        {
            // Set up the array of projection values
            for (int m = 0; m < N; m++)
            {
                const int n = trainingExampleList[m];
                projectionValues[n] = projections[f].adjoint()*storage->getDataPoint(n);
            }
            
            std::sort(trainingExampleList, trainingExampleList + N, [&projectionValues](const int lhs, const int rhs) -> bool {
                return projectionValues[lhs] < projectionValues[rhs];
            });
            
            // Initialize the histograms
            leftHistogram.reset();
            rightHistogram = hist;
            
            float leftValue = projectionValues[trainingExampleList[0]];
            int leftClass = storage->getClassLabel(trainingExampleList[0]);
            
            // Test different thresholds
            // Go over all examples in this node
            for (int m = 1; m < N; m++)
            {
                const int n = trainingExampleList[m];
                
                // Move the last point to the left histogram
                leftHistogram.addOne(leftClass);
                rightHistogram.subOne(leftClass);
                        
                // It does
                // Get the two feature values
                const float rightValue = projectionValues[n];
                
                // Skip this split, if the two points lie too close together
                const float diff = rightValue - leftValue;
                
                if (diff < 1e-6f)
                {
                    leftValue = rightValue;
                    leftClass = storage->getClassLabel(n);
                    continue;
                }
                
                // Get the objective function
                const float localObjective = leftHistogram.getEntropy()
                        + rightHistogram.getEntropy();
                
                if (localObjective < bestObjective)
                {
                    // Get the threshold value
                    bestThreshold = (leftValue + rightValue);
                    bestProjection = projections[f];
                    bestObjective = localObjective;
                    bestLeftMass = leftHistogram.getMass();
                    bestRightMass = rightHistogram.getMass();
                }
                
                leftValue = rightValue;
                leftClass = storage->getClassLabel(n);
            }
        }
        
        // We spare the additional multiplication at each iteration.
        bestThreshold *= 0.5f;
        
        // Did we find good split values?
        if (bestObjective > 1e20 || bestLeftMass < minChildSplitExamples || bestRightMass < minChildSplitExamples)
        {
            // We didn't
            // Don't split
            updateLeafNodeHistogram(tree->getNodeData(node).histogram, hist, smoothingParameter, useBootstrap);
            BOOST_ASSERT(tree->getNodeData(node).histogram.size() > 0);
            continue;
        }
        
        // Set up the data lists for the child nodes
        trainingExamplesSizes.push_back(bestLeftMass);
        trainingExamplesSizes.push_back(bestRightMass);
        trainingExamples.push_back(new int[bestLeftMass]);
        trainingExamples.push_back(new int[bestRightMass]);
        
        int* leftList = trainingExamples[trainingExamples.size() - 2];
        int* rightList = trainingExamples[trainingExamples.size() - 1];
        
        // Sort the points
        for (int m = 0; m < N; m++)
        {
            const int n = trainingExampleList[m];
            const float featureValue = bestProjection.adjoint()*storage->getDataPoint(n);
            
            if (featureValue < bestThreshold)
            {
                leftList[--bestLeftMass] = n;
            }
            else
            {
                rightList[--bestRightMass] = n;
            }
        }
        
        // Ok, split the node
        tree->getNodeConfig(node).setThreshold(bestThreshold);
        tree->getNodeConfig(node).getProjection() = bestProjection;
        const int leftChild = tree->splitNode(node);
        
        state.action = ACTION_SPLIT_NODE;
        state.depth = tree->getNodeConfig(node).getDepth();
        state.objective = bestObjective;
        
        evokeCallback(tree, 0, state);
        
        // Prepare to split the child nodes
        splitStack.push(leftChild);
        splitStack.push(leftChild + 1);
        
        delete[] trainingExampleList;
    }
    
    // If we use bootstrap, we use all the training examples for the 
    // histograms
    if (useBootstrap)
    {
        TreeLearningTools::updateHistograms(tree, dataStorage, smoothingParameter);
    }
    
    return tree;
}

////////////////////////////////////////////////////////////////////////////////
/// OnlineDecisionTreeLearner
////////////////////////////////////////////////////////////////////////////////

void OnlineDecisionTreeLearner::updateSplitStatistics(std::vector<EfficientEntropyHistogram> & leftChildStatistics, 
        std::vector<EfficientEntropyHistogram> & rightChildStatistics, 
        const std::vector<int> & features,
        const std::vector< std::vector<float> > & thresholds, 
        const DataPoint & x, const int label)
{
    for (int f = 0; f < numFeatures; f++)
    {
        // There may not be numThresholds thresholds yet!!!
        for (unsigned int t = 0; t < thresholds[f].size(); t++)
        {
            if (x(features[f]) < thresholds[f][t])
            {
                // x would fall into left child.
                leftChildStatistics[t + numThresholds*f].addOne(label);
            }
            else
            {
                // x would fall into right child.
                rightChildStatistics[t + numThresholds*f].addOne(label);
            }
        }
    }
}

OnlineDecisionTree::ptr OnlineDecisionTreeLearner::learn(AbstractDataStorage::ptr storage)
{
    OnlineDecisionTree::ptr tree = std::make_shared<OnlineDecisionTree>();
    tree->addNode();
    
    return learn(storage, tree);
}

OnlineDecisionTree::ptr OnlineDecisionTreeLearner::learn(AbstractDataStorage::ptr storage, OnlineDecisionTree::ptr tree)
{
    std::mt19937 g(rd());
    
    // Get the number of training examples and the dimensionality of the data set
    const int D = storage->getDimensionality();
    const int C = storage->getClasscount();
    const int N = storage->getSize();
    
    assert(numFeatures <= D);
    assert(thresholdGenerator.getSize() == D);
    
    // The tree must have at least the root note!
    assert(tree->getNumNodes() > 0);
    
    // Saves the sum of impurity decrease achieved by each feature
    // TODO: Update importance calculation
    // importance = std::vector<float>(D, 0.f);
    
    OnlineDecisionTreeLearnerState state;
    state.action = ACTION_START_TREE;
    
    evokeCallback(tree, 0, state);
    
    // Set up a list of all available features.
    numFeatures = std::min(numFeatures, D);
    std::vector<int> features(D);
    for (int f = 0; f < D; f++)
    {
        features[f] = f;
    }
    
    for (int n = 0; n < N; n++)
    {
        const DataPoint & x = storage->getDataPoint(n);
        const int label = storage->getClassLabel(n);
        const int leaf = tree->findLeafNode(x);
        const int depth = tree->getNodeConfig(leaf).getDepth();
        
        state.node = leaf;
        state.depth = depth;
        
        EfficientEntropyHistogram & nodeStatistics = tree->getNodeData(leaf).nodeStatistics;
        std::vector<int> & nodeFeatures = tree->getNodeData(leaf).nodeFeatures;
        std::vector< std::vector<float> > & nodeThresholds = tree->getNodeData(leaf).nodeThresholds;
        std::vector<EfficientEntropyHistogram> & leftChildStatistics = tree->getNodeData(leaf).leftChildStatistics;
        std::vector<EfficientEntropyHistogram> & rightChildStatistics = tree->getNodeData(leaf).rightChildStatistics;
        
        // This leaf node may be a fresh one.
        if (nodeStatistics.getSize() <= 0)
        {
            nodeStatistics.resize(C);
            
            leftChildStatistics.resize(numFeatures*numThresholds);
            rightChildStatistics.resize(numFeatures*numThresholds);
            
            nodeFeatures.resize(numFeatures, 0);
            nodeThresholds.resize(numFeatures);
            
            // Sample thresholds and features.
            std::shuffle(features.begin(), features.end(), std::default_random_engine(rd()));
            
            // Used to make sure, that non/trivial, different features are chosen.
//            int f_alt = numFeatures;
            
            for (int f = 0; f < numFeatures; f++)
            {
                // Try the first feature.
                nodeFeatures[f] = features[f];
                assert(nodeFeatures[f] >= 0 && nodeFeatures[f] < D);
                
                // This may be a trivial feature, so search for the next non/trivial
                // feature; make sure that all chosen features are different.
                const int M = 10;
                int m = 0;

                // TODO: this should not be necessary!
//                while(thresholdGenerator.getMin(nodeFeatures[f]) == thresholdGenerator.getMax(nodeFeatures[f])
//                        && m < M && f_alt < D)
//                {
//                    nodeFeatures[f] = features[f_alt];
//                    ++f_alt;
//                    ++m;
//                }
                        
                nodeThresholds[f].resize(numThresholds);
                
                for (int t = 0; t < numThresholds; t++)
                {
                    nodeThresholds[f][t] = thresholdGenerator.sample(nodeFeatures[f]);
                    
                    if (t > 0)
                    {
                        // Maximum 10 tries to get a better threshold.
                        m = 0;
                        while (std::abs(nodeThresholds[f][t] - nodeThresholds[f][t - 1]) < 1e-6f
                                && m < M)
                        {
                            nodeThresholds[f][t] = thresholdGenerator.sample(nodeFeatures[f]);
                            ++m;
                        }
                    }
                    
                    // Initialize left and right child statistic histograms.
                    leftChildStatistics[t + numThresholds*f].resize(C);
                    rightChildStatistics[t + numThresholds*f].resize(C);
                    
                    leftChildStatistics[t + numThresholds*f].reset();
                    rightChildStatistics[t + numThresholds*f].reset();
                }
            }
            
            state.action = ACTION_INIT_NODE;
            evokeCallback(tree, 0, state);
        }
        
        int K = 1;
        if (useBootstrap)
        {
            std::poisson_distribution<int> poisson(bootstrapLambda);
            K = poisson(g); // May also give zero.
        }
        
        for (int k = 0; k < K; k++)
        {
            // Update node statistics.
            nodeStatistics.addOne(label);
            // Update left and right node statistics for all splits.
            updateSplitStatistics(leftChildStatistics, rightChildStatistics, 
                    nodeFeatures, nodeThresholds, x, label);
        }
        
        state.node = leaf;
        state.depth = depth;
        state.samples = nodeStatistics.getMass();
        
        // As in offline learning, do not split this node
        // - if the number of examples is too small
        // - if the maximum depth is reached
        if (nodeStatistics.getMass() < minSplitExamples || nodeStatistics.isPure() 
                || depth >= maxDepth)
        {
            // Do not split, update leaf histogram according to new sample.
            updateLeafNodeHistogram(tree->getNodeData(leaf).histogram, nodeStatistics, smoothingParameter, false);
        
            state.action = ACTION_NOT_SPLITTING_NODE;  
            evokeCallback(tree, 0, state);
            
            continue;
        }
        
        // Get the best split.
        float bestObjective = 0;
        float bestThreshold = -1;
        float bestFeature = -1;
        
        for (int f = 0; f < numFeatures; f++)
        {
            for (int t = 0; t < numThresholds; t++)
            {
                const int leftMass = leftChildStatistics[t + numThresholds*f].getMass();
                const int rightMass = rightChildStatistics[t + numThresholds*f].getMass();
                
                if (leftMass > minChildSplitExamples && rightMass > minChildSplitExamples)
                {
                    const float localObjective = nodeStatistics.getEntropy()
                            - leftChildStatistics[t + numThresholds*f].getEntropy()
                            - rightChildStatistics[t + numThresholds*f].getEntropy();
                    
                    if (localObjective > bestObjective)
                    {
                        bestObjective = localObjective;
                        bestThreshold = t;
                        bestFeature = f;
                    }
                }
            }
        }
        
        // Split only if the minimum objective is obtained.
        if (bestObjective < minSplitObjective)
        {
            // Do not split, update leaf histogram according to new sample.
            updateLeafNodeHistogram(tree->getNodeData(leaf).histogram, 
                    nodeStatistics, smoothingParameter, false);
        
            state.action = ACTION_NOT_SPLITTING_OBJECTIVE_NODE;  
            state.objective = bestObjective;
            state.minObjective = minSplitObjective;
            
            evokeCallback(tree, 0, state);
            
            continue;
        }
        
        assert(bestFeature >= 0 && nodeFeatures[bestFeature] >= 0 
                && nodeFeatures[bestFeature] < D);
        
        // We split this node!
        tree->getNodeConfig(leaf).setThreshold(nodeThresholds[bestFeature][bestThreshold]); // Save the actual threshold value.
        tree->getNodeConfig(leaf).setSplitFeature(nodeFeatures[bestFeature]); // Save the index of the feature.
        
        const int leftChild = tree->splitNode(leaf);
        const int rightChild = leftChild  + 1;
        
        // This may be the last sample! So initialize the leaf node histograms!
        updateLeafNodeHistogram(tree->getNodeData(leftChild).histogram, 
                leftChildStatistics[bestThreshold + numThresholds*bestFeature], 
                smoothingParameter, false);
        
        updateLeafNodeHistogram(tree->getNodeData(rightChild).histogram, 
                rightChildStatistics[bestThreshold + numThresholds*bestFeature], 
                smoothingParameter, false);
        
        // Save best objective for variable importance.
        // TODO: Update importance calculation
        //++importance[bestFeature];
        
        // Clean up node at this is not a leaf anymore and statistics
        // are not required anymore.
        // nodeStatistics.clear();
        leftChildStatistics.clear();
        rightChildStatistics.clear();
        nodeThresholds.clear();
        nodeFeatures.clear();
        
        // Also clear the histogram as this node is not a leaf anymore!
        tree->getNodeData(leaf).histogram.clear();
        
        state.action = ACTION_SPLIT_NODE; 
        state.objective = bestObjective;
        
        evokeCallback(tree, 0, state);
    }
    
    return tree;
}

int OnlineDecisionTreeLearner::defaultCallback(OnlineDecisionTree::ptr tree, const OnlineDecisionTreeLearnerState & state)
{
    switch (state.action) {
//        case OnlineDecisionTreeLearner::ACTION_INIT_NODE:
//            std::cout << std::setw(30) << std::left << "Init node: "
//                    << "depth = " << std::setw(6) << state->depth << "\n";
//            break;
        case OnlineDecisionTreeLearner::ACTION_SPLIT_NODE:
            std::cout << std::setw(30) << std::left << "Split node: "
                    << "depth = " << std::setw(6) << state.depth
                    << "samples = " << std::setw(6) << state.samples
                    << "objective = " << std::setw(6)
                    << std::setprecision(3) << state.objective << "\n";
            break;
    }
    
    return 0;
}

int OnlineDecisionTreeLearner::verboseCallback(OnlineDecisionTree::ptr tree, const OnlineDecisionTreeLearnerState & state)
{
    switch (state.action) {
        case OnlineDecisionTreeLearner::ACTION_START_TREE:
            std::cout << "Start decision tree training." << "\n";
            break; 
        case OnlineDecisionTreeLearner::ACTION_INIT_NODE:
            std::cout << std::setw(30) << std::left << "Init node: "
                    << "depth = " << std::setw(6) << state.depth << "\n";
            break;
        case OnlineDecisionTreeLearner::ACTION_NOT_SPLITTING_NODE:
            std::cout << std::setw(30) << std::left << "Not splitting node: "
                    << "depth = " << std::setw(6) << state.depth
                    << "samples = " << std::setw(6) << state.samples << "\n";
            break;
        case OnlineDecisionTreeLearner::ACTION_NOT_SPLITTING_OBJECTIVE_NODE:
            std::cout << std::setw(30) << std::left << "Not splitting node: "
                    << "depth = " << std::setw(6) << state.depth
                    << "samples = " << std::setw(6) << state.samples
                    << "objective = " << std::setw(6)
                    << std::setprecision(3) << state.objective
                    << "min objective = " << std::setw(6)
                    << std::setprecision(3) << state.minObjective << "\n";
            break;
        case OnlineDecisionTreeLearner::ACTION_SPLIT_NODE:
            std::cout << std::setw(30) << std::left << "Split node: "
                    << "depth = " << std::setw(6) << state.depth
                    << "samples = " << std::setw(6) << state.samples
                    << "objective = " << std::setw(6)
                    << std::setprecision(3) << state.objective << "\n";
            break;
        default:
            std::cout << "UNKNOWN ACTION CODE " << state.action << "\n";
            break;
    }
    
    return 0;
}
