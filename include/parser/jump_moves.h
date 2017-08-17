#ifndef PARSER_PMCMC_H
#define PARSER_PMCMC_H

#include <random>
#include <opencv2/opencv.hpp>
#include "rjmcmc_sa.h"
#include "parser.h"

namespace parser
{
    /**
     * This is the type of the state vector for MCMC based optimization
     */
    typedef std::vector<int> MCMCParserStateType;

    /**
     * Jump move: splitting a rectangle into 2 new rectangles
     */
    class MCMCParserSplitMove : public SAMove {
    public:
        MCMCParserSplitMove(std::vector<Rectangle> rects) :
            g(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
            proposals(rects) {}

        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                );

    private:
        /**
         * Default random device
         */
        std::random_device rd;
        /**
         * This is the random number generator
         */
        std::mt19937 g;

        /**
         * These are the rectangle proposals
         */
        std::vector<Rectangle> proposals;

    };

    /**
     * Jump move: Merging 2 rectangles into 1 rectangle
     */
    class MCMCParserMergeMove : public SAMove {
    public:
        MCMCParserMergeMove(std::vector<Rectangle> rects) :
            g(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
            proposals(rects) {}

        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                );

    private:
        /**
         * Default random device
         */
        std::random_device rd;
        /**
         * This is the random number generator
         */
        std::mt19937 g;

        /**
         * These are the rectangle proposals
         */
        std::vector<Rectangle> proposals;

    };

    

#if 0
    /**
     * Splits a rectangle into two new rectangles
     */
    class MCMCParserSplitMove : public SAMove {
    public:
        MCMCParserSplitMove(int numParts) : g(std::chrono::high_resolution_clock::now().time_since_epoch().count()), dist(0, numParts - 1) {}

        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                );

    private:
        /**
         * Default random device
         */
        std::random_device rd;
        /**
         * This is the random number generator
         */
        std::mt19937 g;
        /**
         * This is a distribution over the parts
         */
        std::uniform_int_distribution<int> dist;

    };

    /**
     * Merges two rectangles into a new rectangle
     */
    class MCMCParserMergeMove : public SAMove {
    public:
        MCMCParserMergeMove(std::vector<Rectangle> proposals, Eigen::MatrixXi widthMergeable, Eigen::MatrixXi heightMergeable )
            : widthMergeable(widthMergeable), heightMergeable(heightMergeable), proposals(proposals), g(std::chrono::high_resolution_clock::now().time_since_epoch().count()), dist(0, proposals.size() - 1) {}

        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                 );
        int computeMergeableRects(const MCMCParserStateType state, const int mergeType, std::vector<int> & mergeSetR1, std::vector<int> & mergeSetR2);

    private:
        /**
         * Default random device
         */
        std::random_device rd;
        /**
         * This is the random number generator
         */
        std::mt19937 g;
        /**
         * This is a distribution over the parts
         */
        std::uniform_int_distribution<int> dist;
        /**
         * This is the proposal pool
         */
        std::vector<Rectangle> proposals;
        /**
         * This is the matrix of width  mergeable rectangles (common width)
         */
        //Eigen::SparseMatrix<int> widthMergeable;
        Eigen::MatrixXi widthMergeable;
        /**
         * This is the matrix of height mergeable rectangles (common height)
         */
        Eigen::MatrixXi heightMergeable;
        //Eigen::SparseMatrix<int> heightMergeable;

    };

#endif

    /**
     * Birth Move: Add a new rectangle to the current state, keeping all other rectangles fixed,
     * Exchanges a tree with some other tree with an extra rectangle
     */
    class MCMCParserBirthMove : public SAMove {
    public:
        MCMCParserBirthMove(std::vector<Part> partHypotheses, Eigen::MatrixXi overlapPairs, int numRectClusters)
            : numProposals(partHypotheses.size()), overlapPairs(overlapPairs), numClusters(numRectClusters),
              g(std::chrono::high_resolution_clock::now().time_since_epoch().count()), dist(0, partHypotheses.size() - 1) {}

        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                 );

    private:
        /**
         * Default random device
         */
        std::random_device rd;
        /**
         * This is the random number generator
         */
        std::mt19937 g;
        /**
         * This is a distribution over the parts
         */
        std::uniform_int_distribution<int> dist;
        /**
         * This is a number of parts/proposals
         */
        int numProposals;
        /**
         * This is a matrix indicating pairs of overlap above a threshold
         */
        Eigen::MatrixXi overlapPairs;
        /**
         * This is a vector of rectangle proposal parts
         */
        std::vector<Part> partHypotheses;
        /**
         * Number of non redundant Rectangles in the proposal pool
         */
        int numClusters;

        };

    /**
     * Death Move: Removes a rectangle from the current state keeping all other rectangles fixed.
     * Exchanges a tree with some other tree with one rectangle less
     */
    class MCMCParserDeathMove : public SAMove {
    public:
        MCMCParserDeathMove(std::vector<Part> partHypotheses, Eigen::MatrixXi overlapPairs, int numRectClusters) : partHypotheses(partHypotheses), numProposals(partHypotheses.size()),
            overlapPairs(overlapPairs), g(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
            dist(0, partHypotheses.size() - 1), numClusters(numRectClusters) {}

        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                 );
        int computeDissimilarity(
            const MCMCParserStateType & state,
                std::vector<int> & disRectsNum
                );

    private:
        /**
         * Default random device
         */
        std::random_device rd;
        /**
         * This is the random number generator
         */
        std::mt19937 g;
        /**
         * This is a distribution over the parts
         */
        std::uniform_int_distribution<int> dist;
        /**
         * This is the number of proposals
         */
        int numProposals;
        /**
         * This is distribution for selecting from the roulette
         */
        std::uniform_real_distribution<float> rouletteDist;
        /**
         * This is a matrix indicating pairs of overlap above a threshold
         */
        Eigen::MatrixXi overlapPairs;
        /**
         * This is a vector of rectangle proposal parts
         */
        std::vector<Part> partHypotheses;
        /**
         * The number of rectangle clusters in the proposal pool
         */
        int numClusters;

    };
    

}//namespace PARSER
#endif
