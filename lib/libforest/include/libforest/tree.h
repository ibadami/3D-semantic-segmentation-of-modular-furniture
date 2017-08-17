#ifndef LIBF_TREE_H
#define LIBF_TREE_H

#include <vector>
#include <functional>
#include <boost/static_assert.hpp>

#include "util.h"
#include "error_handling.h"
#include "io.h"
#include "data.h"

namespace libf {
    
    /**
     * This is the base class for all node configuration classes. 
     */
    class AbstractNodeConfig {
    public:
        AbstractNodeConfig() : depth(0), leftChild(0) {}
        template <class Config, class Data, class Base>
        friend class AbstractTree;
        
        virtual ~AbstractNodeConfig() {}
        
        /**
         * Returns true if the given node is a leaf node. 
         * 
         * @return True if the node is a leaf node
         */
        bool isLeafNode() const 
        {
            return leftChild == 0;
        }
        
        /**
         * Returns the index of the left child node for a node. 
         * 
         * @return The index of the left child node
         */
        int getLeftChild() const
        {
            return leftChild;
        }
        
        /**
         * Returns the index of the right child node for a node. 
         * 
         * @return The index of the right child node
         */
        int getRightChild() const
        {
            return leftChild + 1;
        }
        
        /**
         * Get depth of a node where the root node has depth 0. 
         * 
         * @return The depth of the node
         */
        int getDepth() const
        {
            return depth;
        }
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream The stream to read the tree from
         */
        virtual void read(std::istream & stream)
        {
            readBinary(stream, depth);
            readBinary(stream, leftChild);
        }
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the tree to.
         */
        virtual void write(std::ostream & stream) const
        {
            writeBinary(stream, depth);
            writeBinary(stream, leftChild);
        }
        
    private:
        
        /**
         * Sets depth of a node where the root node has depth 0. 
         * 
         * @param _depth The depth of the node
         */
        void setDepth(int _depth)
        {
            depth = _depth;
        }
        
        /**
         * Returns the index of the left child node for a node. 
         * 
         * @param _leftChild The index of the left child node
         */
        void setLeftChild(int _leftChild)
        {
            leftChild = _leftChild;
        }
        
        /**
         * The depth of the node.
         */
        int depth;
        /**
         * The left child node index of the node. If the left child node is 0, then 
         * this is a leaf node. The right child node is left + 1. 
         */
        int leftChild;
    };
    
    /**
     * This is the base class for all split trees. Each node in a tree has three
     * data unit associated with it:
     * 1. The node config (like split information, graph structure).
     * 2. The node data (custom application specific data e.g. histograms). 
     * 3. The base class (Either AbstractClassifier, AbstractEstimator, AbstractRegressor)
     */
    template <class Config, class Data, class Base>
    class AbstractTree : public Base {
    public:
        /**
         * Creates an empty split tree.
         */
        AbstractTree()
        {
            // Reserve some memory for the nodes
            // This speeds up training a bit
            nodes.reserve(LIBF_GRAPH_BUFFER_SIZE);
        }
        
        virtual ~AbstractTree() {}
        
        /**
         * Returns the total number of nodes. 
         * 
         * @return The total number of nodes
         */
        int getNumNodes() const
        {
            return static_cast<int>(nodes.size());
        }
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream The stream to read the tree from
         */
        virtual void read(std::istream & stream)
        {
            readBinary(stream, nodes);
        }
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the tree to.
         */
        virtual void write(std::ostream & stream) const
        {
            writeBinary(stream, nodes);
        }
        
        /**
         * Returns the node data. 
         * 
         * @param node The node index
         * @return A reference to the node data
         */
        const Data & getNodeData(int node) const
        {
            BOOST_ASSERT_MSG(0 <= node && node < getNumNodes(), "Invalid node index.");
            return nodes[node].second;
        }
        
        /**
         * Returns the node data. 
         * 
         * @param node The node index
         * @return A reference to the node data
         */
        Data & getNodeData(int node)
        {
            BOOST_ASSERT_MSG(0 <= node && node < getNumNodes(), "Invalid node index.");
            return nodes[node].second;
        }
        
        /**
         * Returns the node config. 
         * 
         * @param node The node index
         * @return A reference to the node config
         */
        const Config & getNodeConfig(int node) const
        {
            BOOST_ASSERT_MSG(0 <= node && node < getNumNodes(), "Invalid node index.");
            return nodes[node].first;
        }
        
        /**
         * Returns the node config. 
         * 
         * @param node The node index
         * @return A reference to the node config
         */
        Config & getNodeConfig(int node)
        {
            BOOST_ASSERT_MSG(0 <= node && node < getNumNodes(), "Invalid node index.");
            return nodes[node].first;
        }
        
        /**
         * Adds a new node. This method needs to be implemented by all trees.
         * 
         * @return The new node index
         */
        int addNode()
        {
            nodes.push_back(std::pair<Config, Data>(Config(), Data()));
            return static_cast<int>(nodes.size() - 1);
        }
        
        /**
         * Splits a child node and returns the index of the left child. 
         * 
         * @param node The node index
         * @return The index of the left child node
         */
        virtual int splitNode(int node)
        {
            // Make sure this is a valid node ID
            BOOST_ASSERT_MSG(0 <= node && node < this->getNumNodes(), "Invalid node index.");
            // Make sure this is a former leaf node
            BOOST_ASSERT_MSG(this->getNodeConfig(node).isLeafNode(), "Cannot split non-leaf node.");

            // Add the child nodes
            const int leftNode = this->addNode();
            const int rightNode = this->addNode();
            
            Config & config = this->getNodeConfig(node);
            
            // Update the depth
            const int depth = config.getDepth() + 1;
            this->getNodeConfig(leftNode).setDepth(depth);
            this->getNodeConfig(rightNode).setDepth(depth);

            // Set the child relation
            config.setLeftChild(leftNode);

            return leftNode;
        }
        
        /**
         * Passes the data point through the tree and returns the index of the
         * leaf node it ends up in. 
         * 
         * @param x The data point to pass down the tree
         * @return The index of the leaf node v ends up in
         */
        virtual int findLeafNode(const DataPoint & x) const = 0;
        
    private:
        /**
         * The node array
         */
        std::vector< std::pair<Config, Data> > nodes;
    };
    
    /**
     * This is the node configuration for axis aligned split trees.
     */
    class AxisAlignedSplitTreeNodeConfig : public AbstractNodeConfig {
    public:
        AxisAlignedSplitTreeNodeConfig() : AbstractNodeConfig(), splitFeature(0), threshold(0) {}
        
        virtual ~AxisAlignedSplitTreeNodeConfig() {}
        
        /**
         * Sets the split feature for a node
         * 
         * @param feature The feature dimension
         */
        void setSplitFeature(int feature)
        {
            BOOST_ASSERT_MSG(feature >= 0, "Invalid feature dimension.");
            
            splitFeature = feature;
        }
        
        /**
         * Returns the split feature for a node
         * 
         * @return The split feature
         */
        int getSplitFeature() const
        {
            return splitFeature;
        }
        
        /**
         * Sets the threshold for a node
         * 
         * @param _threshold The new threshold value
         */
        void setThreshold(float _threshold)
        {
            threshold = _threshold;
        }
        
        /**
         * Returns the threshold for a node
         * 
         * @return The threshold 
         */
        float getThreshold() const
        {
            return threshold;
        }
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream The stream to read the tree from
         */
        void read(std::istream & stream)
        {
            AbstractNodeConfig::read(stream);
            readBinary(stream, splitFeature);
            readBinary(stream, threshold);
        }
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the tree to.
         */
        void write(std::ostream & stream) const
        {
            AbstractNodeConfig::write(stream);
            writeBinary(stream, splitFeature);
            writeBinary(stream, threshold);
        }
        
    private:
        /**
         * The split feature of the node
         */
        int splitFeature;
        /**
         * The threshold of the node
         */
        float threshold;
    };
    
    /**
     * Overload the read binary method to also read AxisAlignedSplitTreeNodeConfig
     */
    inline void readBinary(std::istream & stream, AxisAlignedSplitTreeNodeConfig & v)
    {
        v.read(stream);
    }
    
    /**
     * Overload the write binary method to also write DecisionTreeNodeData
     */
    inline void writeBinary(std::ostream & stream, const AxisAlignedSplitTreeNodeConfig & v)
    {
        v.write(stream);
    }
    
    /**
     * This is a base class for trees that split the space using axis aligned
     * splits. Each node in the tree can carry some specified data. 
     */
    template <class Base>
    class AbstractAxisAlignedSplitTree : public Base {
    public:
        /**
         * Creates an empty split tree.
         */
        AbstractAxisAlignedSplitTree() : Base() {}
        
        /**
         * Destructor.
         */
        virtual ~AbstractAxisAlignedSplitTree() {}

        
        /**
         * Passes the data point through the tree and returns the index of the
         * leaf node it ends up in. 
         * 
         * @param x The data point to pass down the tree
         * @return The index of the leaf node v ends up in
         */
        virtual int findLeafNode(const DataPoint & x) const
        {
            // Select the root node as current node
            int node = 0;

            // Follow the tree until we hit a leaf node
            while (!this->getNodeConfig(node).isLeafNode())
            {
                const AxisAlignedSplitTreeNodeConfig & config = this->getNodeConfig(node);
                
                // Check the threshold
                if (x(config.getSplitFeature()) < config.getThreshold())
                {
                    // Go to the left
                    node = config.getLeftChild();
                }
                else
                {
                    // Go to the right
                    node = config.getRightChild();
                }
            }

            return node;
        }
    };
    
    /**
     * This is the node configuration for projective split trees.
     */
    class ProjectiveSplitTreeNodeConfig : public AbstractNodeConfig {
    public:
        ProjectiveSplitTreeNodeConfig() : AbstractNodeConfig(), threshold(0) {}
        
        virtual ~ProjectiveSplitTreeNodeConfig() {}
        
        /**
         * Returns the projection
         * 
         * @return The projection vector
         */
        DataPoint & getProjection()
        {
            return projection;
        }
        
        /**
         * Returns the projection
         * 
         * @return The projection vector
         */
        const DataPoint & getProjection() const
        {
            return projection;
        }
        
        /**
         * Sets the threshold for a node
         * 
         * @param _threshold The new threshold value
         */
        void setThreshold(float _threshold)
        {
            threshold = _threshold;
        }
        
        /**
         * Returns the threshold for a node
         * 
         * @return The threshold 
         */
        float getThreshold() const
        {
            return threshold;
        }
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream The stream to read the tree from
         */
        void read(std::istream & stream)
        {
            AbstractNodeConfig::read(stream);
            // TODO: Implement writeBinary for DataPoint
            // readBinary(stream, projection);
            readBinary(stream, threshold);
        }
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the tree to.
         */
        void write(std::ostream & stream) const
        {
            AbstractNodeConfig::write(stream);
            // TODO: Implement writeBinary for DataPoint
            // writeBinary(stream, projection);
            writeBinary(stream, threshold);
        }
        
    private:
        /**
         * The split projection
         */
        DataPoint projection;
        /**
         * The threshold of the node
         */
        float threshold;
    };
    
    /**
     * Overload the read binary method to also read ProjectiveSplitTreeNodeConfig
     */
    inline void readBinary(std::istream & stream, ProjectiveSplitTreeNodeConfig & v)
    {
        v.read(stream);
    }
    
    /**
     * Overload the write binary method to also write ProjectiveSplitTreeNodeConfig
     */
    inline void writeBinary(std::ostream & stream, const ProjectiveSplitTreeNodeConfig & v)
    {
        v.write(stream);
    }
    
    /**
     * This is a base class for trees that split the space using projective
     * splits. Each node in the tree can carry some specified data. 
     */
    template <class Base>
    class AbstractProjectiveSplitTree : public Base {
    public:
        /**
         * Creates an empty split tree.
         */
        AbstractProjectiveSplitTree() : Base() {}
        
        /**
         * Destructor.
         */
        virtual ~AbstractProjectiveSplitTree() {}
        
        /**
         * Passes the data point through the tree and returns the index of the
         * leaf node it ends up in. 
         * 
         * @param x The data point to pass down the tree
         * @return The index of the leaf node v ends up in
         */
        virtual int findLeafNode(const DataPoint & x) const
        {
            // Select the root node as current node
            int node = 0;

            // Follow the tree until we hit a leaf node
            while (!this->getNodeConfig(node).isLeafNode())
            {
                const ProjectiveSplitTreeNodeConfig & config = this->getNodeConfig(node);
                const float inner = config.getProjection().adjoint()*x;
                
                // Check the threshold
                if (inner < config.getThreshold())
                {
                    // Go to the left
                    node = config.getLeftChild();
                }
                else
                {
                    // Go to the right
                    node = config.getRightChild();
                }
            }

            return node;
        }
    };
    
    /**
     * This is the base class for all ensembles of trees. 
     */
    template <class TreeType, class Base>
    class AbstractForest : public Base {
    public:
        virtual ~AbstractForest() {}
        
        /**
         * Reads the tree from a stream. 
         * 
         * @param stream the stream to read the forest from.
         */
        virtual void read(std::istream & stream)
        {
            // Read the number of trees in this ensemble
            int size;
            readBinary(stream, size);

            // Read the trees
            for (int i = 0; i < size; i++)
            {
                std::shared_ptr<TreeType> tree = std::make_shared<TreeType>();

                tree->read(stream);
                addTree(tree);
            }
        }
        
        /**
         * Writes the tree to a stream
         * 
         * @param stream The stream to write the forest to. 
         */
        void write(std::ostream & stream) const
        {
            // Write the number of trees in this ensemble
            writeBinary(stream, getSize());

            // Write the individual trees
            for (int i = 0; i < getSize(); i++)
            {
                getTree(i)->write(stream);
            }
        }
        
        /**
         * Adds a tree to the ensemble
         * 
         * @param tree The tree to add to the ensemble
         */
        void addTree(std::shared_ptr<TreeType> tree)
        {
            trees.push_back(tree);
        }
        
        /**
         * Returns the number of trees
         * 
         * @return The number of trees in this ensemble
         */
        int getSize() const
        {
            return trees.size();
        }
        
        /**
         * Returns the i-th tree
         * 
         * @param i The index of the tree to return
         * @return The i-th tree
         */
        std::shared_ptr<TreeType> getTree(int i) const
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "Invalid tree index.");
            
            return trees[i];
        }
        
        /**
         * Removes the i-th tree from the ensemble. 
         * 
         * @param i The index of the tree
         */
        void removeTree(int i)
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "Invalid tree index.");
            
            // Remove it from the array
            trees.erase(trees.begin() + i);
        }
        
    private:
        /**
         * The individual decision trees. 
         */
        std::vector< std::shared_ptr<TreeType> > trees;
    };
}

#endif