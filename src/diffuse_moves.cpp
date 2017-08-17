#include "parser/diffuse_moves.h"
#include <math.h>

using namespace parser;

static std::random_device rd;

////////////////////////////////////////////////////////////////////////////////
//// MCMCParserCallback
////////////////////////////////////////////////////////////////////////////////

int MCMCParserCallback::callback(
    const MCMCParserStateType & state,
    float energy,
    const MCMCParserStateType & bestState,
    float bestEnergy,
    int iteration,
    float temperature)
{
    std::cout <<
            "iteration: " << iteration <<
            " energy: " << energy <<
            " best: " << bestEnergy <<
            " temp: " << temperature <<
            " state: " << state.size() << std::endl;

    cv::Mat demo(500, 500, CV_8UC3);
    demo = cv::Scalar(0);

    for (size_t h = 0; h < state.size(); h++)
    {
        cv::Scalar color;
        switch (parts[state[h]].label)
        {
            case 0:
                color = cv::Scalar(0,0,255);
                break;
            case 1:
                color = cv::Scalar(0,255,0);
                break;
            case 2:
                color = cv::Scalar(0,255,255);
                break;
        }
        PlotUtil::plotRectangle(demo, parts[state[h]].rect, color);
    }

    cv::imshow("test", demo);
    cv::waitKey(1);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//// MCMCParserExchangeMove: No dimension change
////////////////////////////////////////////////////////////////////////////////

void MCMCParserExchangeMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;

    int replacePart;

    // Set up a distribution over the current state
    std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));

    // Choose a tree to replace
    replacePart = stateDist(g);//random selection

    int addPart;
    addPart = dist(g);//random

    newState[replacePart] = addPart;

    float proposalFactor = 0.0f;
    //float proposalFactor = std::log((double) ( ( partHypotheses[addPart].posterior ) / ( std::max(0.01f,partHypotheses[state[replacePart]].posterior)) ) );
    logAcceptRatio = proposalFactor;
}


// DATA-DRIVEN EXCHANGE

void MCMCParserDDExchangeMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;

    int replacePart, addPart;

    // Set up a distribution over the current state
    std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));


#if ROULETTE_EXCHANGE// weihted sampling
    //Rectangles with lower weights in the state are highly probable to be picked
    // Calculate roulette based on the state

    if(state.size()>0)// TO DO:Remove this if condition when probability of moves is implemented Make sure that state never
        //reaches zero dimension by making zero probability for death move when state size = 0
    {
        replacePart = computeRussianRoulette(state);// Weighted sampling
    }
    else
    {
        // Choose a tree to replace
        replacePart = stateDist(g);

    }

#else

    // Choose a tree to replace
    replacePart = stateDist(g);//random
#if 0
    if(state.size()==0)
        std::cout<<"at zero dimension Replacing part "<<replacePart<<std::endl;
#endif

#endif


    int rectIDx = state[replacePart];
    std::vector<int> similarRects;
    //data-drivenness
    int numSimilarRects = computeSimilarRects(rectIDx, similarRects);

    if(numSimilarRects > 0)// similar rectangles exist, so exchange in a data driven way
    {
        std::uniform_int_distribution<int> similarRectsDist(0, static_cast<int>(numSimilarRects - 1));
        addPart = similarRects[similarRectsDist(g)];
    }
    else // no similar rectangles
    {
        //keep this state, dont replace
        addPart = state[replacePart];
    }

    newState[replacePart] = addPart;
    float proposalFactor = 0.0f;
    //float proposalFactor = std::log((double) ( ( partHypotheses[addPart].posterior ) / ( std::max(0.01f,partHypotheses[state[replacePart]].posterior)) ) );
    // avoid singularity with parts of zero probability
    logAcceptRatio = proposalFactor;
}

/*
 * Similarity check: for data drivenness
*/

int MCMCParserDDExchangeMove::computeSimilarRects(const int rectIDx, std::vector<int> & similarRects)
{
    // compute all the rectangles dissimilar from the input rectangle
    Eigen::MatrixXi overlapRects = Eigen::MatrixXi::Zero(static_cast<int>(1),static_cast<int>(numProposals));
    similarRects.resize(0);

    overlapRects = overlapPairs70.row(rectIDx);

    for (size_t i = 0; i < numProposals; i++)
    {
        if(overlapRects(0,i) != 0)
        {
            similarRects.push_back(i);
        }
    }

    int numSimilarRects = similarRects.size();
    return numSimilarRects;

}

int MCMCParserDDExchangeMove::computeRussianRoulette(const MCMCParserStateType state)
{

    int replacePart;
    float total = 0.0f;

    for(size_t i= 0; i<state.size(); i++ )
    {
        total += (1.0f - partHypotheses[state[i]].posterior); // least posterior parts must be selected more frequently
    }

    float prevIndex = 0.0f;
    std::vector<float> rouletteRussian;
    rouletteRussian.push_back(0.0f);

    for(size_t i= 0; i<state.size(); i++ )
    {
         //special case when  all the posteriors are 1.0f
        if(total == 0)
        {
            rouletteRussian.push_back(((double) 1/state.size())  + prevIndex);
        }
        else
        {
            rouletteRussian.push_back(static_cast<float>( (1.0f- partHypotheses[state[i]].posterior)/total) + prevIndex);
        }

        prevIndex = rouletteRussian[i+1];
    }

    const float rouletteNum = rouletteDist(g);

    for(size_t i = state.size(); i > 0; i--)// TO DO:Make sure that state size > 0 (prevent death move going to null states)
    {
        if( rouletteNum >= rouletteRussian[i-1] && rouletteNum <= rouletteRussian[i] )
        {
            replacePart = i-1;
            break;
        }
    }

    return replacePart;
}

////////////////////////////////////////////////////////////////////////////////
//// MCMCParserUpdateCenterDiffuseMove
////////////////////////////////////////////////////////////////////////////////
void MCMCParserUpdateCenterDiffuseMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;
    logAcceptRatio = 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
//// MCMCParserUpdateWidthDiffuseMove
////////////////////////////////////////////////////////////////////////////////
void MCMCParserUpdateWidthDiffuseMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;
    logAcceptRatio = 0.0f;
}

////////////////////////////////////////////////////////////////////////////////
//// MCMCParserUpdateHeightDiffuseMove
////////////////////////////////////////////////////////////////////////////////
void MCMCParserUpdateHeightDiffuseMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;
    logAcceptRatio = 0.0f;
}


////////////////////////////////////////////////////////////////////////////////
//// MCMCParserLabelDiffuseMove
////////////////////////////////////////////////////////////////////////////////

void MCMCParserLabelDiffuseMove::move(
    const MCMCParserStateType & state,
    MCMCParserStateType & newState,
    float & logAcceptRatio)
{
    // Copy all other parts
    newState = state;

    int replacePart;
#if 0
    float minPosterior = 1.0f;

    for(int k = 0; k < newState.size(); k++ )
    {
        std::min(minPosterior, partHypotheses[newState[k]].posterior);
        if(minPosterior == partHypotheses[newState[k]].posterior)
        {
            replacePart = k;
        }
    }
#endif

#if 1
    // Set up a distribution over the current state
    std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));

    // Choose a label to change
    replacePart = stateDist(g);
#endif

    // Determine the label of the part
    const int label = state[replacePart]%3;

    // Select a new label
    std::uniform_int_distribution<int> labelDist(0, 2);

    newState[replacePart] = state[replacePart] - label + labelDist(g);

    logAcceptRatio = 0.0f;
}
