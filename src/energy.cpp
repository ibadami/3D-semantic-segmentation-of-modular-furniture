#include <vector>

#include "parser/energy.h"
#include "libforest/libforest.h"
#include "parser/parser.h"
#include "gurobi_c++.h"

using namespace parser;


////////////////////////////////////////////////////////////////////////////////
//// Parser
////////////////////////////////////////////////////////////////////////////////

ParseTreeNode* ParserEnergy::parse(std::vector<ParseTreeNode*> nodes) const
{
    while (nodes.size() != 1)
    {
#if 0
        {
            cv::Mat demo(500,500,CV_8UC3);
            
            for (size_t j = 0; j < nodes.size(); j++)
            {
                demo = cv::Scalar(0);
                for (size_t i = 0; i < nodes.size(); i++)
                {
                    PlotUtil::plotRectangle(demo, nodes[i]->rect, cv::Scalar(255,255,0));
                }
                PlotUtil::plotRectangle(demo, nodes[j]->rect, cv::Scalar(255,0,255),3);
                cv::imshow("test", demo);
                cv::waitKey();
            }
        }
        std::cout << nodes.size() << std::endl;
#endif
        // Try to combine the nodes
        std::vector< std::vector<int> > clusters;
        combine(nodes, clusters);
        bool isMesh = true;
        // Did we merge two nodes?
        if (clusters.size() == 0)
        {
            isMesh = false;
            // Node, merge the two most similar nodes
            std::pair<size_t, size_t> bestMatch;
            float bestScore = -1;
            
            for (size_t i = 0; i < nodes.size(); i++)
            {
                for (size_t j = i+1; j < nodes.size(); j++)
                {
                    // Can we merge these two?
                    if (isValidMerge(std::vector<ParseTreeNode*>({ nodes[i], nodes[j] }), nodes))
                    {
                        float score =   std::abs(nodes[i]->rect.getWidth() - nodes[j]->rect.getWidth()) + 
                                        std::abs(nodes[i]->rect.getHeight() - nodes[j]->rect.getHeight());
                        
                        if (score < bestScore || bestScore < 0)
                        {
                            bestMatch.first = i;
                            bestMatch.second = j;
                            bestScore = score;
                        }
                    }
                }
            }
            
            // Well, the current state is not parse-able
            if (bestScore < 0)
            {
                for (size_t n = 0; n < nodes.size(); n++)
                {
                    delete nodes[n];
                }
                throw std::exception();
            }
            
            clusters.push_back( std::vector<int>({ static_cast<int>(bestMatch.first), static_cast<int>(bestMatch.second)}));
        }
        
        std::vector<int> mergedNodes;
        
        // Combine all nodes
        std::vector<ParseTreeNode*> newNodes;
        for (size_t i = 0; i < clusters.size(); i++)
        {
            // Create a new node for this group
            ParseTreeNode* node = new ParseTreeNode();
            node->mesh = isMesh;
            // Merge the rectangles
            std::vector<ParseTreeNode*> collection;
            for (size_t n = 0; n < clusters[i].size(); n++)
            {
                mergedNodes.push_back(clusters[i][n]);
                collection.push_back(nodes[clusters[i][n]]);
            }
            merge(collection, node->rect);
            node->children = collection;
            
            newNodes.push_back(node);
        }
        
        // Add the remaining nodes
        for (size_t n = 0; n < nodes.size(); n++)
        {
            if (std::find(mergedNodes.begin(), mergedNodes.end(), static_cast<int>(n)) == mergedNodes.end())
            {
                newNodes.push_back(nodes[n]);
            }
        }
        
        nodes = newNodes;
    }
    return nodes[0];
}

void ParserEnergy::visualize(ParseTreeNode* tree, const cv::Mat & image, const Segmentation & segmentation) const
{
    std::vector<ParseTreeNode*> nodes;
    nodes.push_back(tree);
    
    cv::Mat demo;
    image.copyTo(demo);
    while (true)
    {
        std::vector<Rectangle> unrectified, rectified;
        for (size_t n = 0; n < nodes.size(); n++)
        {
            unrectified.push_back(nodes[n]->rect);
            rectified.push_back(nodes[n]->rect);
        }
        
        CabinetParser parser;
        //parser.unrectifyParts(segmentation.regionOfInterest, unrectified, rectified);
        
        for (size_t n = 0; n < rectified.size(); n++)
        {
            PlotUtil::plotRectangle(demo, rectified[n], cv::Scalar(0,255,0), 2);
        }
        
        cv::imshow("test", demo);
        cv::waitKey();
        
        std::vector<ParseTreeNode*> newNodes;
        for (size_t n = 0; n < nodes.size(); n++)
        {
            if (nodes[n]->isTerminal())
            {
                newNodes.push_back(nodes[n]);
            }
            else
            {
                for (size_t c = 0; c < nodes[n]->children.size(); c++)
                {
                    newNodes.push_back(nodes[n]->children[c]);
                }
            }
        }
        // Check if all nodes are terminals
        bool allTerminals = true;
        for (size_t n = 0; n < nodes.size(); n++)
        {
            allTerminals = allTerminals && nodes[n]->isTerminal();
        }
        if (allTerminals)
        {
            break;
        }
        nodes = newNodes;
    }
    
}

void ParserEnergy::combine(const std::vector<ParseTreeNode*> nodes, std::vector< std::vector<int> > & clusters) const
{
    // Get the number of nodes
    const int N = static_cast<int>(nodes.size());
    
    // Set up a matrix of possible merges. 
    Eigen::MatrixXi possibleMerged = Eigen::MatrixXi::Zero(N,N);
    for (int n = 0; n < N; n++)
    {
        const Rectangle & r1 = nodes[n]->rect;
        for (int m = n+1; m < N; m++)
        {
            const Rectangle & r2 = nodes[m]->rect;
            
            if (parser::RectangleUtil::areSimilar(r1, r2, this->similarityThreshold))
            {
                possibleMerged(n,m) = 1;
                possibleMerged(m,n) = 1;
            }
            else
            {
                possibleMerged(n,m) = 0;
                possibleMerged(m,n) = 0;
            }
        }
    }
    
    // Compute the transitive closure of possibleMerged
    Eigen::MatrixXi closure = possibleMerged;
    for (int k = 0; k < N; k++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                closure(i,j) = std::max(closure(i,j), std::min(closure(i,k), closure(k,j)));
            }
        }
    }
    
    // Get the connected components
    std::vector< std::vector<int> > connectedComponents;
    std::vector<bool> assigned(N, false);
    for (int n = 0; n < N; n++)
    {
        // Has this node already been assigned to a connected component?
        if (!assigned[n])
        {
            // Nope, it has not
            // Create a new one
            connectedComponents.push_back(std::vector<int>());
            const size_t index = connectedComponents.size() - 1;

            for (int m = n; m < N; m++)
            {
                if (closure(n,m))
                {
                    connectedComponents[index].push_back(m);
                    assigned[m] = true;
                }
            }
        }
    }

    for(size_t k = 0; k < connectedComponents.size(); k++)
    {
        const std::vector<int> & component = connectedComponents[k];
        std::vector<int> mergeCandidates;
        
        // Set up a list of parts that can potentially be merged
        for (size_t _n = 0; _n < component.size(); _n++)
        {
            const int n = component[_n];

            // Search a merge partner for this part
            int partner = -1;
            
            // Do not consider this node if it's already in the candidate list
            if (std::find(mergeCandidates.begin(), mergeCandidates.end(), n) != mergeCandidates.end())
            {
                continue;
            }
            
            for (size_t _m = 0; _m < component.size(); _m++)
            {
                const int m = component[_m];

                // If the two rectangles can't be merged anyway, don't bother
                if (!possibleMerged(n,m))
                {
                    continue;
                }
                
                // Check if the two nodes can be merged
                if (isValidMerge(std::vector<ParseTreeNode*>({ nodes[n], nodes[m] }), nodes))
                {
                    // They can
                    partner = m;
                    break;
                }
            }
            
            // Did we find a partner?
            if (partner >= 0)
            {
                mergeCandidates.push_back(n);
                if (std::find(mergeCandidates.begin(), mergeCandidates.end(), partner) == mergeCandidates.end())
                {
                    mergeCandidates.push_back(partner);
                }
            }
        }
        
        const int M = static_cast<int>(mergeCandidates.size());
        
        // Create the candidate set of all possible merges
        std::vector< std::vector<int> > clusterCandidates;
        
        // Check all possible clusters
        Util::exhaustiveSubsetSearch<int>(mergeCandidates, [&clusterCandidates, &nodes, &closure, this](const std::vector<int> & cluster) -> void {
            // Is the cluster big enough?
            if (cluster.size() < 2)
            {
                // Nope
                return;
            }
            
            // Check if there is at least one partner for each element
            for (size_t n = 0; n < cluster.size(); n++)
            {
                for(size_t m = 0; m < cluster.size(); m++)
                {
                    //  Is this a partner?
                    if (!closure(static_cast<int>(cluster[n]), static_cast<int>(cluster[m])))
                    {
                        return;
                    }
                }
            }
            std::vector<ParseTreeNode*> realCluster(cluster.size());
            for (size_t n = 0; n < cluster.size(); n++)
            {
                realCluster[n] = nodes[cluster[n]];
            }
            
            // Check if its a valid cluster
            if (isValidMerge(realCluster, nodes))
            {
                // It is, add it to the list
                clusterCandidates.push_back(cluster);
            }
        });
        
        const int C = static_cast<int>(clusterCandidates.size());
        
        // Set up the matrix of conflicting clusters
        Eigen::MatrixXi conflictingClusters(C, C);
        std::vector<int> sizes(C);

        // Sort the cluster candidates by size (largest cluster up front)
        std::sort(clusterCandidates.begin(), clusterCandidates.end(), [](const std::vector<int> & lhs, const std::vector<int> & rhs) {
            return lhs.size() > rhs.size();
        });

        for (int m = 0; m < C; m++)
        {
            sizes[m] = static_cast<int>(clusterCandidates[m].size());
            conflictingClusters(m,m) = 0;
            for (int n = m+1; n < C; n++)
            {
                // Check if the two clusters are disjoint
                if (Util::areDisjoint(clusterCandidates[n],clusterCandidates[m]))
                {
                    conflictingClusters(n,m) = 0;
                    conflictingClusters(m,n) = 0;
                }
                else
                {
                    conflictingClusters(n,m) = 1;
                    conflictingClusters(m,n) = 1;
                }
            }
        }

        // Find the best clustering
        std::vector<int> bestClustering;

        auto isFeasible = [&conflictingClusters] (const std::vector<int> & clustering) -> bool {
            const auto S = clustering.size();
            
            for (size_t s = 0; s < S; s++)
            {
                for (size_t t = s+1; t < S; t++)
                {
                    if (conflictingClusters(clustering[s],clustering[t]))
                    {
                        return false;
                    }
                }
            }
            return true;
        };

        auto calcSize = [&sizes] (const std::vector<int> & clustering) -> int {
            const auto S = clustering.size();
            int totalSize = 0;
            
            for (size_t s = 0; s < S; s++)
            {
                totalSize += sizes[clustering[s]];
            }
            
            return totalSize;
        };

        // This is the recursive search
        std::function<void(std::vector<int>)> search;
        search = [&bestClustering, &isFeasible, &search, &calcSize, C, M] (std::vector<int> currentClustering) {
            // Is this a feasible cluster?
            if (isFeasible(currentClustering))
            {
                // Does it have the right size?
                if (calcSize(currentClustering) == M)
                {
                    // We can stop backtracking here
                    if (currentClustering.size() < bestClustering.size() || bestClustering.size() == 0)
                    {
                        bestClustering = currentClustering;
                    }
                }
                else
                {
                    // If the number of elements in this clustering is already higher than the
                    // best clustering, then there is no need to proceed searching
                    if (bestClustering.size() > 0 && currentClustering.size() > 0 && currentClustering.size() >= bestClustering.size())
                    {
                        return;
                    }
                    else
                    {
                        // We have to keep on searching
                        // The elements in the clustering are sorted, so we know where to keep
                        // searching
                        int maxElement = 0;
                        if (currentClustering.size() > 0)
                        {
                            maxElement = currentClustering[currentClustering.size() - 1] + 1;
                        }
                        for (int c = maxElement; c < C; c++)
                        {
                            std::vector<int> newClustering(currentClustering);
                            newClustering.push_back(c);
                            search(newClustering);
                        }
                    }
                }
            }
        };

        std::vector<int> initialClustering;
        search(initialClustering);

        for (size_t i = 0; i < bestClustering.size(); i++)
        {
            clusters.push_back(clusterCandidates[bestClustering[i]]);
        }
    }
}

bool ParserEnergy::isValidMerge(const std::vector<ParseTreeNode*> & nodes, const std::vector<ParseTreeNode*> & collection) const
{
    // Compute the merged rectangle
    Rectangle merged;
    merge(nodes, merged);
    
    // Check if the merge rectangle overlaps significantly with any other rectangle
    // in the collection
    for (size_t n = 0; n < collection.size(); n++)
    {
        if (std::find(nodes.begin(), nodes.end(), collection[n]) != nodes.end())
        {
            continue;
        }
        
        // Compute the relative intersection area
        Rectangle intersection;
        RectangleUtil::calcIntersection(collection[n]->rect, merged, intersection);
        
        const float relativeIntersectionArea = intersection.getArea()/collection[n]->rect.getArea();
        
        if (relativeIntersectionArea > 0.25)
        {
            return false;
        }
    }
    
    return true;
}

void ParserEnergy::merge(const std::vector<ParseTreeNode*> & nodes, Rectangle & mergedRectangle) const
{
    // Find the minimum/maximum x/y coordinates
    float minX = 1e10;
    float minY = 1e10;
    float maxX = -1e10;
    float maxY = -1e10;
    
    for (size_t i = 0; i < nodes.size(); i++)
    {
        minX = std::min(nodes[i]->rect.minX(), minX);
        minY = std::min(nodes[i]->rect.minY(), minY);
        maxX = std::max(nodes[i]->rect.maxX(), maxX);
        maxY = std::max(nodes[i]->rect.maxY(), maxY);
    }
    
    mergedRectangle[0][0] = minX;
    mergedRectangle[0][1] = minY;
    
    mergedRectangle[1][0] = maxX;
    mergedRectangle[1][1] = minY;
    
    mergedRectangle[2][0] = maxX;
    mergedRectangle[2][1] = maxY;
    
    mergedRectangle[3][0] = minX;
    mergedRectangle[3][1] = maxY;
}
