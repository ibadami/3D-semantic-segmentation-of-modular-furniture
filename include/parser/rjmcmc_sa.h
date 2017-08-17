#ifndef PARSER_MCMC_H
#define PARSER_MCMC_H

#include <vector>
#include <random>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "parser.h"

namespace parser {

    typedef std::vector<int> MCMCParserStateType;

    /**
     * This is the cooling schedule interface. 
     */
    class SACoolingSchedule {
    public:
        /**
         * Calculates the next temperature based on the iteration and the 
         * current temperature. 
         */
        virtual float calcTemperature(int iteration, float temperature) const = 0;
        
        /**
         * Returns the start temperature
         */
        virtual float getStartTemperature() const = 0;
        
        /**
         * Returns the end temperature
         */
        virtual float getEndTemperature() const = 0;
    };



    /**
     * This is a geometric cooling schedule: t_k+1 = t_k * alpha.
     */
    class GeometricCoolingSchedule { // : public SACoolingSchedule {
    public:
        GeometricCoolingSchedule() : startTemperature(100), endTemperature(1), alpha(0.8f) {}

        /**
         * Sets the start temperature
         */
        void setStartTemperature(float temp)
        {
            startTemperature = temp;
        }

        /**
         * Returns the start temperature
         */
        float getStartTemperature() const
        {
            return startTemperature;
        }

        /**
         * Sets the end temperature
         */
        void setEndTemperature(float temp)
        {
            endTemperature = temp;
        }

        /**
         * Returns the end temperature
         */
        float getEndTemperature() const
        {
            return endTemperature;
        }

        /**
         * Calculates the next temperature based on the iteration and the
         * current temperature.
         */
        float calcTemperature(int, float temperature) const
        {
            return temperature*alpha;
        }

        /**
         * Sets alpha
         */
        void setAlpha(float _alpha)
        {
            alpha = _alpha;
        }

        /**
         * Returns alpha
         */
        float getAlpha() const
        {
            return alpha;
        }

    private:
        /**
         * The start temperature
         */
        float startTemperature;
        /**
         * The end temperature
         */
        float endTemperature;
        /**
         * Cooling schedule alpha
         */
        float alpha;
    };


    
    /**
     * This is the interface one has to implement for a callback function for
     * SA.
     */
    class SACallback {
    public:
        /**
         * The function that is called
         */
        virtual int callback(const MCMCParserStateType & state, float energy, const MCMCParserStateType & bestState, float bestEnergy, int iteration, float temperature) = 0;
    };



    
    /**
     * This is the interface one has to implement for a SA move.
     */
    class SAMove {
    public:
        /**
         * Computes the move
         */
        virtual void move(const MCMCParserStateType & state, MCMCParserStateType & newState, float & improvement) = 0;
    };


    
    /**
     * This is the interface one has to implement for a SA energy function.
     */
    class SAEnergy {
    public:
        /**
         * Computes the energy
         */
        virtual float energy(const MCMCParserStateType & state, std::vector<float> & moveProbabilities) = 0;
    };
    



    /**
    * This is the energy function for the pruning optimization
    */
    class MCMCParserEnergy {
    public:
        MCMCParserEnergy() {}
        MCMCParserEnergy(
            const std::vector<Part> & parts,
            const std::vector<float> & rectAreas,
            const Eigen::MatrixXi & overlapPairs,
            const Eigen::MatrixXf & overlapArea,
            const cv::Mat & image
                ): parts(parts), areas(rectAreas), overlapConflicts(overlapPairs), overlaps(overlapArea), image(image){}

    /**
     * Computes the energy
     */
    float energy(const MCMCParserStateType & state,
                 std::vector<float>& moveProbabilities,
                 const std::vector<float> & areas,
                 const Eigen::MatrixXi & overlapPairs,
                 const Eigen::MatrixXf & overlapArea,
                 std::vector<Part>& parts
                 );
    /**
     * To update the move probabilities in each state
     */

    void updateMoveProbabilities(const float coveredArea, std::vector<float>& moveProbabilities, const int numRects);
    /**
     * Energy to trim weird shaped structures
     */

    int computeFormFactorEnergy(const MCMCParserStateType & state);

    /**
     * Factorial computation
     */
    int factorial(int n);
    /**
     * Permutation
     */
    int nC2(int n);

#if 0
    /**
     * Computes the area covering of the state
     */
    float coverArea(const MCMCParserStateType & state);
#endif
    void setOverlapArea(Eigen::MatrixXf rectOverlaps)
    {
        overlaps = rectOverlaps;
    }

    void setAreas(std::vector<float> rectAreas)
    {
        for(size_t i=0; i<rectAreas.size(); ++i)
        {
            areas.push_back(rectAreas[i]);
        }
    }

    void setOverlapPairs(Eigen::MatrixXi rectOverlapConflicts)
    {
        overlapConflicts = rectOverlapConflicts;
    }

    float lastLabelEnergy;
    float lastWeightEnergy;
    float lastLayoutVarianceEnergy;
    float lastOverlapEnergy;
    float lastCoverEnergy;
    float lastStateSizeEnergy;

    /**
     * This are all part hypotheses
     */
    std::vector<Part> parts;
    /**
     * The image
     */
    cv::Mat image;
    /**
     * This is the conflict matrix
     */
    Eigen::MatrixXi overlapConflicts;
    /**
     * The relative area of the individual parts
     */
    std::vector<float> areas;
    /**
     * The area overlaps of the individual parts
     */
    Eigen::MatrixXf overlaps;

    float temperature;
};


    /**
     * This class implements general simulated annealing. T is the type of
     * the state variable. We assume that the proposal distribution is symmetric.
     * i.e.
     *   q(i|j) = q(j|i)
     */
    class SimulatedAnnealing {
    public:
        SimulatedAnnealing(std::vector<Part> & parts, std::vector<Rectangle> & proposals, const cv::Mat & gradMag,
                           std::vector<float> & areas, Eigen::MatrixXi & overlapPairs, Eigen::MatrixXi & overlapPairs70,
                           Eigen::MatrixXf & overlapArea, cv::Mat cannyEdges

        ) : gradMag(gradMag), partHypotheses(parts), areas(areas), proposals(proposals), numInnerLoops(500),
            maxNoUpdateIterations(5000), overlapPairs(overlapPairs), overlapPairs70(overlapPairs70), overlapArea(overlapArea), cannyEdges(cannyEdges) {};

        /**
         * Adds a move
         */
        void addMove(SAMove* move, float prob)
        {
            moves.push_back(move);
            moveProbabilities.push_back(prob);
        }
        
        /**
         * Sets the cooling schedule
         */
        void setCoolingSchedule(GeometricCoolingSchedule schedule)
        {
            coolingSchedule = schedule;
        }
        
        /**
         * Returns the cooling schedule
         */
        const GeometricCoolingSchedule & getCoolingSchedule() const
        {
            return coolingSchedule;
        }
        
        /**
         * Returns the cooling schedule
         */
        GeometricCoolingSchedule & getCoolingSchedule()
        {
            return coolingSchedule;
        }
        
        /**
         * Adds a callback function
         */
        void addCallback(SACallback* callback)
        {
            callbacks.push_back(callback);
        }
        
        /**
         * Sets the number of inner loops
         */
        void setNumInnerLoops(int _numInnerLoops)
        {
            numInnerLoops = _numInnerLoops;
        }


        /**
         * Returns the number of inner loops
         */
        int getNumInnerLoops() const
        {
            return numInnerLoops;
        }
        
        /**
         * Sets the energy function
         */
        void setEnergyFunction(MCMCParserEnergy function)
        {
            energyFunction = function;
        }
        
        /**
         * Sets maxNoUpdateIterations
         */
        void setMaxNoUpdateIterations(int _maxNoUpdateIterations)
        {
            maxNoUpdateIterations = _maxNoUpdateIterations;
        }
        
        /**
         * Returns maxNoUpdateIterations
         */
        int getMaxNoUpdateIterations() const
        {
            return maxNoUpdateIterations;
        }

        /**
         * Optimizes the error function for a given initialization. 
         */
        float optimize(MCMCParserStateType & state);
        /**
         * To display the params in console:debug purpose
         */
        void displaySimAnnealParams();

        /**
         * To display the state in a window:debug purpose
         */
        void plotMarkovChainState(const MCMCParserStateType state, const std::vector<Part> partHypotheses);

        /**
         * Selecting the type of move
         */
        int selectRJMCMCMoveType(const float u);

        /**
         * Split one rectangle into two rectangles
         */
        void split1Rectangle(const MCMCParserStateType state, const int splitPart, const Part originalPartB4Split,
                             std::vector<Part> & partHypotheses, Part & splitPartR1, Part & splitPartR2);

        /**
         * Update the proposal matrices during split move
         */
        void updateMatricesSplit(const std::vector<Part> partHypotheses, std::vector<float> & areas,
                                 const Part splitPartR1, const Part splitPartR2, const float imageArea,
                                 Eigen::MatrixXi& overlapPairs, Eigen::MatrixXf& overlapArea);
        /**
         * Check whether a pair of rectangles are mergeable
         */
        void computeMergeability(const MCMCParserStateType state, const std::vector<Part> partHypotheses,
                                 int & rectIdx1, int & rectIdx2);
        /**
         * Merge 2 rectangles into a single rectangle
         */
        void merge2Rectangles(const MCMCParserStateType state, std::vector<Part> & partHypotheses, const int rectIdx1,
                              const int rectIdx2, Part & mergedPart);
        /**
         * Update the proposal matrices during merge move
         */
        void updateMatricesMerge(const std::vector<Part> partHypotheses, const Rectangle mergedRect,
                                 Eigen::MatrixXi& overlapPairs, Eigen::MatrixXf& overlapArea);        
        /**
         * Update Location Move
         */
        void diffuseCenterLoc(const Rectangle originalCenterRect, Rectangle & modifiedCenterRect);
        /**
         * Update the proposal matrices during updateLoc move
         */
        void updateMatricesCenterLocDiffuse(const MCMCParserStateType state,
                                                                const std::vector<Part> partHypotheses,
                                                                const Rectangle modifiedCenterRect,
                                                                const int updateCenterPart,
                                                                Eigen::MatrixXi& overlapPairs,
                                                                Eigen::MatrixXf& overlapArea);

        /**
         * UpdateWidth Move
         */
        void diffuseRectWidth(const Rectangle originalWidthRect, Rectangle & modifiedWidthRect);
        /**
         * Update the proposal matrices during updateWidth move
         */
        void updateMatricesWidthDiffuse(const MCMCParserStateType state,
                                                                const std::vector<Part> partHypotheses,
                                                                std::vector<float> & areas,
                                                                const Rectangle modifiedWidthRect,
                                                                const int updateWidthPart,
                                                                Eigen::MatrixXi& overlapPairs,
                                                                Eigen::MatrixXf& overlapArea,
                                                                const float imageArea);
        /**
         * UpdateHeight move
         */
        void diffuseRectHeight(const Rectangle originalHeightRect, Rectangle & modifiedHeightRect);
        /**
         * Update the proposal matrices during updateHeight move
         */
        void updateMatricesHeightDiffuse(const MCMCParserStateType state,
                                                                const std::vector<Part> partHypotheses,
                                                                std::vector<float> & areas,
                                                                const Rectangle modifiedHeightRect,
                                                                const int updateHeightPart,
                                                                Eigen::MatrixXi& overlapPairs,
                                                                Eigen::MatrixXf& overlapArea,
                                                                const float imageArea);

    private:
        /**
         * These are the registered moves
         */
        std::vector<SAMove*> moves;
        /**
         * The probability for choosing this move
         */
        std::vector<float> moveProbabilities;
        /**
         * The cooling schedule that determines the temperature
         */
        GeometricCoolingSchedule coolingSchedule;
        /**
         * The error function
         */
        MCMCParserEnergy energyFunction;
        /**
         * The callback functions
         */
        std::vector<SACallback*> callbacks;
        /**
         * The number of inner loops
         */
        int numInnerLoops;
        /**
         * If the optimum does not change for more than this iterations
         * of the outer loop, optimization is terminated.
         */
        int maxNoUpdateIterations;
        /**
         * The rectangle parts
         */
        std::vector<Part> partHypotheses;
        /**
         * The rectangle proposals
         */
        std::vector<Rectangle> proposals;
        /**
         * The relative area of the individual parts
         */
        std::vector<float> areas;
        /**
         * The binary matrix indicating pairs with overlap greater that a threshold
         */
        Eigen::MatrixXi overlapPairs;
        /**
         * The binary matrix indicating pairs with overlap greater that a 70%
         */
        Eigen::MatrixXi overlapPairs70;
        /**
         * The matrix indicating exact overlap between any two rectangles
         */
        Eigen::MatrixXf overlapArea;
        /**
         * Gradient Magnitude Image
         */
        cv::Mat gradMag;
        /**
         * Canny edge Image
         */
        cv::Mat cannyEdges;

    };

}
#endif
