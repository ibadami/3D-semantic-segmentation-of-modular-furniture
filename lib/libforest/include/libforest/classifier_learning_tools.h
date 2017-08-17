#ifndef LIBF_CLASSIFIER_LEARNING_TOOLS_H
#define LIBF_CLASSIFIER_LEARNING_TOOLS_H

#include "classifier.h"

namespace libf {

    /**
     * These are some helpful tools for learning decision trees. 
     */
    class TreeLearningTools {
    public:
        /**
         * Updates the histograms at the leaf nodes of the given tree and updates
         * them based on the statistics of the given data storage. The old
         * histogram entries are deleted. 
         * 
         * @param tree The decision tree whose leaf nodes shall be updated
         * @param storage The data storage that shall be used to update the histograms
         * @param smoothingParameter The value with which the histograms are initialized
         */
        template <class T>
        static void updateHistograms(std::shared_ptr<T> tree, AbstractDataStorage::ptr storage, float smoothingParameter)
        {
            const int C = storage->getClasscount();

            // Reset all histograms
            for (int v = 0; v < tree->getNumNodes(); v++)
            {
                if (tree->getNodeConfig(v).isLeafNode())
                {
                    std::vector<float> & hist = tree->getNodeData(v).histogram;

                    // Make sure that hist is initialized.
                    hist.resize(C);

                    for (int c = 0; c < C; c++)
                    {
                        hist[c] = 0;
                    }
                }
            }


            // Compute the weights for each data point
            for (int n = 0; n < storage->getSize(); n++)
            {
                int leafNode = tree->findLeafNode(storage->getDataPoint(n));
                tree->getNodeData(leafNode).histogram[storage->getClassLabel(n)] += 1;
            }

            // Normalize the histograms
            for (int v = 0; v < tree->getNumNodes(); v++)
            {
                if (tree->getNodeConfig(v).isLeafNode())
                {
                    std::vector<float> & hist = tree->getNodeData(v).histogram;
                    float total = 0;
                    for (int c = 0; c < C; c++)
                    {
                        total += hist[c];
                    }
                    for (int c = 0; c < C; c++)
                    {
                        hist[c] = std::log((hist[c] + smoothingParameter)/(total + C*smoothingParameter));
                    }
                }
            }
        }
    };
}

#endif