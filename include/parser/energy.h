#ifndef ENERGY_H
#define ENERGY_H

#include "util.h"

namespace parser {
    
    class Segmentation;
    
    /**
     * This class represents a node in a parse tree
     */
    class ParseTreeNode {
    public:
        ParseTreeNode() : part(-1), mesh(false) {}
        virtual ~ParseTreeNode()
        {
            for (size_t n = 0; n < children.size(); n++)
            {
                delete children[n];
            }
        }
        
        /**
         * Returns true if the node is a terminal node
         */
        bool isTerminal() const 
        {
            return children.size() == 0;
        }
        
        /**
         * The rectangle of this node
         */
        Rectangle rect;
        /**
         * The children
         */
        std::vector<ParseTreeNode*> children;
        /**
         * Corresponding part
         */
        int part;
        /**
         * True if this is a mesh 
         */
        bool mesh;
    };
    
    /**
     * This class parses a set of rectangles. 
     */
    class ParserEnergy {
    public:
        ParserEnergy() : similarityThreshold(10) {}
        
        /**
         * Parses the given set of rectangles
         */
        ParseTreeNode* parse(std::vector<ParseTreeNode*> nodes) const;
        
        /**
         * Visualizes the parse tree
         */
        void visualize(ParseTreeNode* tree, const cv::Mat & image, const Segmentation & segmentation) const;
        
        float similarityThreshold;
    private:
        /**
         * Merged all possible nodes 
         */
        void combine(const std::vector<ParseTreeNode*> nodes, std::vector< std::vector<int> > & clusters) const;
        
        /**
         * Returns true if the given set of rectangles can be merged in a given
         * collection.
         */
        bool isValidMerge(const std::vector<ParseTreeNode*> & merge, const std::vector<ParseTreeNode*> & collection) const;
        
        /**
         * Compute the merged rectangle for a set of rectangles
         */
        void merge(const std::vector<ParseTreeNode*> & nodes, Rectangle & mergedRectangle) const;
        
        std::vector<Rectangle> rectangles;
    };
}

#endif