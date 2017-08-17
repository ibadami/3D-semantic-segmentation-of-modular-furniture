#ifndef PARSER_MOVES_DIFFUSE_H
#define PARSER_MOVES_DIFFUSE_H

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
    * This is the simulated annealing callback. It just logs everything to the
    * console
    */
    class MCMCParserCallback : public SACallback {
    public:
        MCMCParserCallback(const std::vector<Part> & parts) : parts(parts) {}

        /**
        * The function that is called
        */
        virtual int callback(
            const MCMCParserStateType & state,
            float energy,
            const MCMCParserStateType & bestState,
            float bestEnergy,
            int iteration,
            float temperature);

    private:
        /**
        * The list of parts
        */
        std::vector<Part> parts;
    };


    /**
    * Exchanges a tree with some other tree
    */
    class MCMCParserExchangeMove : public SAMove {
    public:
        MCMCParserExchangeMove(std::vector<Part> partHypotheses) : partHypotheses(partHypotheses),
            g(std::chrono::high_resolution_clock::now().time_since_epoch().count()), dist(0, partHypotheses.size()-1) {}

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
        * This is distribution over the number of parts
        */
        std::uniform_int_distribution<int> dist;

        /**
        * This is the vector of parts
        */
        std::vector<Part> partHypotheses;


    };

    /**
     * Exchanges a tree with some other tree DATA-DRIVEN by weighted sampling (russian roulette)
     */
    class MCMCParserDDExchangeMove : public SAMove {
    public:
        MCMCParserDDExchangeMove(std::vector<Part> partHypotheses, Eigen::MatrixXi overlapPairs70) : overlapPairs70(overlapPairs70),
            numProposals(partHypotheses.size()), partHypotheses(partHypotheses), rouletteDist(0, 1),
            g(std::chrono::high_resolution_clock::now().time_since_epoch().count()), dist(0, partHypotheses.size()-1) {}

        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                );
        /*
         * Find structurally similar rectangles for exchange
         **/
        int computeSimilarRects(const int rectIDx, std::vector<int> & similarRects);
        /*
         * Weighted sampling strategy
         **/
        int computeRussianRoulette(const MCMCParserStateType state);

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
         * This is distribution over the number of parts
         */
        std::uniform_int_distribution<int> dist;
        /**
         * This is distribution for selecting from the roulette
         */
        std::uniform_real_distribution<float> rouletteDist;
        /**
         * This is rectangle parts
         */
        std::vector<Part> partHypotheses;
        /**
         * This is the number of proposals
         */
        int numProposals;
        /**
         * This is the matrix with overlap>70% proposals and < 100%
         */
        Eigen::MatrixXi overlapPairs70;

    };


    /**
     * Dimension preserving Diffusion move: update the center location
     */
    class MCMCParserUpdateCenterDiffuseMove : public SAMove {
    public:
        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                );
    };

    /**
     * Dimension preserving Diffusion move: update the rectangle width
     */
    class MCMCParserUpdateWidthDiffuseMove : public SAMove {
    public:
        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                );
    };

    /**
     * Dimension preserving Diffusion move: update the rectangle height
     */
    class MCMCParserUpdateHeightDiffuseMove : public SAMove {
    public:
        /**
         * Computes the move
         */
        void move(
            const MCMCParserStateType & state,
            MCMCParserStateType & newState,
            float & logAcceptRatio
                );
    };

    /**
     * Switches the label of a randomly selected part; fixed structure move
     *Preserves dimensions
     */
    class MCMCParserLabelDiffuseMove : public SAMove {
    public:
        MCMCParserLabelDiffuseMove(std::vector<Part> parts) :
            partHypotheses(parts), g(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {}

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
        /*
         * the weighted and labelled rectangles
        */
        std::vector<Part> partHypotheses;
    };


}//namespace PARSER


#endif // PARSER_MOVES_DIFFUSE_H
