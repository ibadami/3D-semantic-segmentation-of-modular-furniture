#include "parser/jump_moves.h"

using namespace parser;

static std::random_device rd;

////////////////////////////////////////////////////////////////////////////////
//// MCMCParserSplitMove
////////////////////////////////////////////////////////////////////////////////
void MCMCParserSplitMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;
    logAcceptRatio = 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
//// MCMCParserMergeMove
////////////////////////////////////////////////////////////////////////////////
void MCMCParserMergeMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;
    logAcceptRatio = 0.0f;
}

#if 0
////////////////////////////////////////////////////////////////////////////////
//// MCMCParserSplitMove
////////////////////////////////////////////////////////////////////////////////
void MCMCParserSplitMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio )
{
    // Copy all other parts
    newState = state;

    // Set up a distribution over the current state
    std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));

#if 0// TO DO: Include data driven rectangle selection for split, split type and split location
    // Choose a rectangle to split
    const int splitPart = stateDist(g);

    // Set up a distribution over height 0 or Width 1 split
    std::uniform_int_distribution<int> hwDist(0,1);
    // Choose type of split
    const int splitType = hwDist(g);

    // Set up a distribution over where to split
    std::uniform_real_distribution<float> uDist(0.01f,1);// 0.0f and 1 excluded to prevent singularity of Jacobian
    // Choose position of split
    float splitPos = uDist(g);

    Rectangle afterSplitR1, afterSplitR2;

    float widthB4Split  = proposals[splitPart].getWidth();
    float heightB4Split = proposals[splitPart].getHeight();// clockwise order starting from top left

    // Create two new rectangles.
    if(splitType == HEIGHT_SPLIT)//Height split, same width
    {
        //afterSplitR1 is the top rectangle
        afterSplitR1[0] = proposals[splitPart][0];
        afterSplitR1[1] = proposals[splitPart][1];
        afterSplitR1[2] = afterSplitR1[1];
        afterSplitR1[2][1] += splitPos*heightB4Split;
        afterSplitR1[3] = afterSplitR1[0];
        afterSplitR1[3][1] += splitPos*heightB4Split;

        //afterSplitR2 is the bottom rectangle
        afterSplitR2[0] = afterSplitR1[3];
        afterSplitR2[1] = afterSplitR1[2];
        afterSplitR2[2] = proposals[splitPart][2];
        afterSplitR2[3] = proposals[splitPart][3];
    }
    else if(splitType == WIDTH_SPLIT)//Width split, same height
    {
        //afterSplitR1 is the left rectangle
        afterSplitR1[0] = proposals[splitPart][0];
        afterSplitR1[1] = afterSplitR1[0];
        afterSplitR1[1][0] += splitPos*widthB4Split;
        afterSplitR1[3] = proposals[splitPart][3];
        afterSplitR1[2] = afterSplitR1[3];
        afterSplitR1[2][0] += splitPos*widthB4Split;

        //afterSplitR2 is the right rectangle
        afterSplitR2[0] = afterSplitR1[1];;
        afterSplitR2[1] = proposals[splitPart][1];
        afterSplitR2[2] = proposals[splitPart][2];
        afterSplitR2[3] = afterSplitR1[2];
    }
    else
    {
        std::cout<<"Invalid split move"<<std::endl;
    }

    //std::cout<<proposals.size()<<std::endl;
    proposals.push_back(afterSplitR1);
    proposals.push_back(afterSplitR2);
    //std::cout<<proposals.size()<<std::endl;
    updateProposals(proposals);
    // TO DO: update the proposal matrices if reuired (if accepted?)
    newState.erase(newState.begin()+ splitPart);
    newState.push_back(proposals.size()-1);// confirm
    newState.push_back(proposals.size()-2);

    setsplitProperties(splitType, splitPos);

    float proposalFactor = std::log(state.size()/numMergeablePairs);

    int sizeB4Split;
    if(splitType == WIDTH_SPLIT)
    {
        sizeB4Split = widthB4Split;
    }
    else if(splitType == HEIGHT_SPLIT)
    {
        sizeB4Split = heightB4Split;
    }

    float factorJacobian = std::log( (splitPos-1)*sizeB4Split );// TO DO: Recheck: log(Negative value d-1)// MISTAKE

#endif

    // Choose a tree to add
    const int addPart = dist(g);
    // Choose a tree to replace
    const int replacePart = stateDist(g);

    newState[replacePart] = addPart;

    float proposalFactor = 0.0f;
    float factorJacobian = 0.0f;

    logAcceptRatio = proposalFactor + factorJacobian;
}


////////////////////////////////////////////////////////////////////////////////
//// MCMCParserMergeMove
////////////////////////////////////////////////////////////////////////////////

void MCMCParserMergeMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;

    // Set up a distribution over height 0 or Width 1 split
    std::uniform_int_distribution<int> hwDist(0,1);//could be data driven based on the relative number of height-width mergeability pairs
    // Choose type of split
    const int mergeType = hwDist(g);

    // function with current state as input
    // outputs: number of mergeable pairs, select a pair at random and return them
    // which is R1 and which is R2  proper indication

    Rectangle toMergeR1, toMergeR2, mergedRect;
    std::vector<int> mergeSetR1, mergeSetR2;

    int numMergeablePairs = computeMergeableRects(state, mergeType, mergeSetR1, mergeSetR2);// find it out from the current state and mergeability matrix
    //std::cout<<"numMergeablePairs: "<<numMergeablePairs<<std::endl;

    if(numMergeablePairs > 0)
    {
        std::uniform_int_distribution<int> mergeableDist(0, numMergeablePairs-1 );
        int mergePairIdx = mergeableDist(g);
        toMergeR1 = proposals[mergeSetR1[mergePairIdx]];
        toMergeR2 = proposals[mergeSetR2[mergePairIdx]];

    }

    int indexR1, indexR2;

#if 0

    if(mergeType == WIDTH_MERGE)// Merging by width, same width
    {
        //choose index 1 and index 2 based on width mergeability matrix
        widthMergeable
        numMergeable =
        std::uniform_int_distribution<int> mergeableDist(0, static_cast<int>(numMergeable - 1));
        indexR1 =
        indexR2 =
        toMergeR1 = proposals[index1];
        toMergeR1 = proposals[index2];

        // assume toMergeR1 is on top of toMergeR2
        mergedRect[0] = toMergeR1[0];// 4 possible merge strategies which makes sure that the result of merge is also a rectangle
        mergedRect[1] = toMergeR1[1];
        mergedRect[2][0] = toMergeR1[1][0];
        mergedRect[2][1] = toMergeR2[2][1];
        mergedRect[3][0] = toMergeR1[0][0];
        mergedRect[3][1] = toMergeR2[3][1];
    }
    else if(mergeType == HEIGHT_MERGE)// Merging by height, same height
    {
        //choose index 1 and index 2 based on height mergeability matrix
        heightMergeable
        numMergeable =
        std::uniform_int_distribution<int> mergeableDist(0, static_cast<int>(numMergeable - 1));
        indexR1 =
        indexR2 =
        toMergeR1 = proposals[index1];
        toMergeR1 = proposals[index2];

        // Assume toMergeR1 is on left of toMergeR2
        mergedRect[0] = toMergeR1[0];
        mergedRect[1][0] = toMergeR2[1][0];
        mergedRect[1][1] = toMergeR1[1][1];
        mergedRect[2][0] = toMergeR2[2][0];
        mergedRect[2][1] = toMergeR1[2][1];
        mergedRect[3] = toMergeR1[3];
    }
    else
    {
        std::cout<<"Invalid merge move"<<std::endl;
    }
    // temporarily add the merged rectangle to the proposals. let the algorithm accpt or reject it.
    // if accepted add that to the set of proposal permanently.
    //Dont forget to delete the two merged rectangles permanently from the proposals (if accepted)
    proposals.push_back(mergedRect);// size of the proposals changed. this must be updated in acceptance criteria also

    int mergePart1Idx, mergePart2Idx;
    for(size_t i = 0; i < newState.size(); i++)
    {
        if(newState[i] == index1)
            mergePart1Idx = i;
        if(newState[i] == index21)
            mergePart2Idx = i;
    }

    newState.erase(newState.begin()+ mergePart1Idx);
    newState.erase(newState.begin()+ mergePart2Idx);
    newState.push_back(proposals.size()-1);//confirm

    float proposalFactor = std::log(numMergeablePairs/newState.size());

    int mergeR1Size, mergeR2Size;
    if(splitType == WIDTH_MERGE)
    {
        mergeR1Size = toMergeR1.getWidth;
        mergeR2Size = toMergeR2.getWidth;
    }
    else if(splitType == HEIGHT_MERGE)
    {
        mergeR1Size = toMergeR1.getHeight;
        mergeR2Size = toMergeR2.getHeight;
    }

    float factorJacobian = std::log( -1*mergeR1Size/(mergeR1Size + mergeR2Size)/(mergeR1Size + mergeR2Size));
    // TO DO: Recheck: log(Negative value d-1)// MISTAKE



#endif

    // Set up a distribution over the current state
    std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));

    // Choose a tree to add
    const int addPart = dist(g);
    // Choose a tree to replace
    const int replacePart = stateDist(g);

    newState[replacePart] = addPart;

    float proposalFactor = 0.0f;
    float factorJacobian = 0.0f;

    logAcceptRatio = proposalFactor + factorJacobian;

}


int MCMCParserMergeMove::computeMergeableRects(const MCMCParserStateType state, const int mergeType, std::vector<int> & mergeSetR1, std::vector<int> & mergeSetR2)
{

    int numMergeablePairs = 0;
    Eigen::MatrixXi mergeableMat(heightMergeable.rows(),heightMergeable.cols());

    if(mergeType == HEIGHT_MERGE)
    {
        mergeableMat = heightMergeable;
    }
    else if(mergeType == WIDTH_MERGE)
    {
        mergeableMat = widthMergeable;
    }


    for(int _sr = 0; _sr < state.size(); _sr++)
    {
        for(int _sc = 0; _sc < state.size(); _sc++)
        {
            if(_sr != _sc && mergeableMat(state[_sr],state[_sc]) == 1)
            {
                mergeSetR1.push_back(state[_sr]);
                mergeSetR2.push_back(state[_sc]);
            }
        }
    }

    numMergeablePairs = mergeSetR1.size();
    return numMergeablePairs;
}

#endif
////////////////////////////////////////////////////////////////////////////////
//// MCMCParserBirthMove: Increase of dimensions
////////////////////////////////////////////////////////////////////////////////
void MCMCParserBirthMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;
    float proposalFactor = 0.0f;
    float factorJacobian = 0.0f;

#if DATA_DRIVEN_BIRTH_DEATH //Data-driven birth // not tested

    MCMCParserDeathMove objD(partHypotheses,overlapPairs);
    std::vector<int> dissimilarRects;
    int numDissimilarRects = objD.computeDissimilarity(state, dissimilarRects);

    if(numDissimilarRects > 0)
    {
        std::uniform_int_distribution<int> dissimilarDist(0, static_cast<int>(numDissimilarRects - 1));
        // Choose a tree to add
        const int addPart = dissimilarDist(g);
        newState.push_back(dissimilarRects[addPart]);
        proposalFactor = std::log( (double) numDissimilarRects / newState.size()  );// Proposal ratio
    }

#else
    const int addPart = dist(g);//random selection
    newState.push_back(addPart);
    //proposalFactor = std::log((double)numProposals/(newState.size()));// Proposal ratio
    //proposalFactor = std::log((double)numClusters /(newState.size()));// Proposal ratio
    proposalFactor = std::log( numClusters );//proposal ratio factor

#endif

    logAcceptRatio = proposalFactor + factorJacobian;//in logarithm

}

/////////////////////////////
//// MCMCParserDeath: Decrease of dimension
/////////////////////////////

void MCMCParserDeathMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;

    // Make probability of death move absolutely zero when state size = 1 so that it wont go to size = 0 state
    //if(state.size() > 1)
    {
        float factorJacobian = 0.0f;
        float proposalFactor = 0.0f;
        int deathPart;

    #if ROULETTE_DEATH

        std::vector<Part> posteriorInState;
        for(size_t i = 0; i < state.size(); i++)
        {
            posteriorInState.push_back(partHypotheses[state[i]].posterior);
        }

        std::vector<float> rouletteRussian;

        computeRussianRoulette(posteriorInState, rouletteRussian);
        const float rouletteNum = rouletteDist(g);

        for(size_t i = state.size(); i>0; i--)
        {
            if( rouletteNum >= rouletteRussian[i-1] && rouletteNum <= rouletteRussian[i] )
            {
                deathPart = i-1;
                //std::cout<<" added part: "<<deathPart<<" out of :"<<state.size()<<" with roulette number: "<<rouletteNum<<std::endl;
                break;
            }
        }
    #else

        // Set up a distribution over the current state
        std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));

        // Choose a tree to replace
        deathPart = stateDist(g);//random selection
    #endif

        newState.erase(newState.begin() + deathPart);


    #if DATA_DRIVEN_BIRTH_DEATH

        std::vector<int> dissimilarRects;
        int numDissimilarRects = computeDissimilarity(newState, dissimilarRects);

        if(numDissimilarRects>0)
        {
            //std::cout<<"Normal DD death case detected"<<std::endl;
            proposalFactor = std::log( (double) state.size() / numDissimilarRects );// Proposal ratio
        }
        else if(numDissimilarRects==0)
        {
            //std::cout<<"Limiting death case detected"<<std::endl;
            //float proposalFactor = 0.0f;
        }
    #else
        //proposalFactor = std::log((double)(state.size())/numProposals);// Proposal ratio
        //proposalFactor = std::log( (double)state.size()/numClusters );
        proposalFactor = std::log( state.size() );

    #endif

        logAcceptRatio = proposalFactor + factorJacobian;
    }

}

///////////////////////////////////////
/*
 * For data driven moves
*/

int MCMCParserDeathMove::computeDissimilarity(
    const MCMCParserStateType & state,
        std::vector<int> & dissimilarRects)
{
    // compute all the rectangles dissimilar from the current state rectangles
    Eigen::MatrixXi overlapRects = Eigen::MatrixXi::Zero(static_cast<int>(1),static_cast<int>(numProposals));
    dissimilarRects.resize(0);

    for (size_t i = 0; i < state.size(); i++)
    {
        overlapRects += overlapPairs.row(state[i]);
    }

#if 0
    std::cout<<overlapRects.rows()<<" x "<<overlapRects.cols()<<std::endl;
    std::cout<<overlapRects<<std::endl;
#endif

    for (size_t i = 0; i < numProposals; i++)
    {
        if(overlapRects(0,i) == 0)
        {
            dissimilarRects.push_back(i);
        }
    }

    int numDissimilarRects = dissimilarRects.size();
    return numDissimilarRects;
}
