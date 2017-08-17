#include "parser/rjmcmc_sa.h"
#include <math.h>

using namespace parser;

/*
 * Module to compute the combinatorial energy function in each state
 */



////////////////////////////////////////////////////////////////////////////////
//// MCMCParserEnergy
////////////////////////////////////////////////////////////////////////////////
#if 0
/*
 * Compute area coverage of any state
 */
float MCMCParserEnergy::coverArea(const MCMCParserStateType & state)
{
    float coverArea = 0.0f;
    for (size_t n = 0; n < state.size(); n++)
    {
        coverArea += areas[state[n]];
        for (size_t m = n+1; m < state.size(); m++)
        {
            coverArea -= overlaps(static_cast<int>(state[n]), static_cast<int>(state[m]));
        }
    }

    return coverArea;
}
#endif


/*
 * Updating move probabilities across iterations
 */

void MCMCParserEnergy::updateMoveProbabilities(const float coveredArea, std::vector<float>& moveProbabilities, const int numRects)
{
    // Covering Area is just an approximation
#if 0
    if(coveredArea >= 0.95f)
    {
        moveProbabilities[BIRTH_MOVE_IDX] = 0.0001f;
        moveProbabilities[EXCHANGE_DD_MOVE_IDX] = INIT_PROB_EXCHANGE_DATADRIVEN + INIT_PROB_BIRTH - 0.0001f;
    }
    else
    {
        moveProbabilities[BIRTH_MOVE_IDX] = INIT_PROB_BIRTH;
        moveProbabilities[EXCHANGE_DD_MOVE_IDX] = INIT_PROB_EXCHANGE_DATADRIVEN;
    }
#endif



#if 0
    if(coveredArea >= 0.95f)
    {
        moveProbabilities[BIRTH_MOVE_IDX] = 0.0001f;
        moveProbabilities[DEATH_MOVE_IDX] = 0.05f - 0.0001f;
        moveProbabilities[LABEL_DIFFUSE_MOVE_IDX] = 0.4f;
        //moveProbabilities[UPDATE_CENTER_MOVE_IDX] = 0.0f;
        moveProbabilities[EXCHANGE_DD_MOVE_IDX] = 0.25f;
        moveProbabilities[EXCHANGE_MOVE_IDX] = 0.3f;
    }
    else if(coveredArea >= 0.7f)
    {
        moveProbabilities[BIRTH_MOVE_IDX] = 0.05f;
        moveProbabilities[DEATH_MOVE_IDX] = 0.1f;
        moveProbabilities[LABEL_DIFFUSE_MOVE_IDX] = 0.35f;
        //moveProbabilities[UPDATE_CENTER_MOVE_IDX] = 0.0f;
        moveProbabilities[EXCHANGE_DD_MOVE_IDX] = 0.20f;
        moveProbabilities[EXCHANGE_MOVE_IDX] = 0.30f;
    }
    else if(coveredArea >= 0.4f)
    {
        moveProbabilities[BIRTH_MOVE_IDX] = 0.1f;
        moveProbabilities[DEATH_MOVE_IDX] = 0.1f;
        moveProbabilities[LABEL_DIFFUSE_MOVE_IDX] = 0.15f;
        //moveProbabilities[UPDATE_CENTER_MOVE_IDX] = 0.0f;
        moveProbabilities[EXCHANGE_DD_MOVE_IDX] = 0.25f;
        moveProbabilities[EXCHANGE_MOVE_IDX] = 0.4f;
    }
    else
    {
        moveProbabilities[BIRTH_MOVE_IDX] = INIT_PROB_BIRTH;
        moveProbabilities[DEATH_MOVE_IDX] = INIT_PROB_DEATH;
        moveProbabilities[LABEL_DIFFUSE_MOVE_IDX] = INIT_PROB_LABEL_DIFFUSE;
        //moveProbabilities[UPDATE_CENTER_MOVE_IDX] = INIT_PROB_UPDATE_CENTER;
        moveProbabilities[EXCHANGE_DD_MOVE_IDX] = INIT_PROB_EXCHANGE_DATADRIVEN;
        moveProbabilities[EXCHANGE_MOVE_IDX] = INIT_PROB_EXCHANGE_RANDOM;
    }

#endif

#if 0
    float birthDecay;

    if(coveredArea >= 0.95f)
    {
        float exchangeDDBoost = 0.10f, labelDiffuseBoost = 0.45f, updateLocCenterBoost = 0.0f;
        birthDecay = INIT_PROB_BIRTH - 0.00001f;//float birthDecay =  0.0f;
        moveProbabilities[BIRTH_MOVE_IDX] = INIT_PROB_BIRTH - birthDecay;// Cannot be zero as it leads to singularity in acceptance ratio equation
        moveProbabilities[LABEL_DIFFUSE_MOVE_IDX] = INIT_PROB_LABEL_DIFFUSE + labelDiffuseBoost;// very efficient diffusion, need not be even 10%
        moveProbabilities[UPDATE_CENTER_MOVE_IDX] = INIT_PROB_UPDATE_CENTER + updateLocCenterBoost;
        moveProbabilities[EXCHANGE_DD_MOVE_IDX] = INIT_PROB_EXCHANGE_DATADRIVEN + exchangeDDBoost;
        moveProbabilities[EXCHANGE_MOVE_IDX] = INIT_PROB_EXCHANGE_RANDOM - labelDiffuseBoost - exchangeDDBoost - updateLocCenterBoost
                                                 + birthDecay;
    }
    #if 0
    else if(coveredArea >= 0.50f)
    {
        birthDecay = INIT_PROB_BIRTH - 0.1f;
        moveProbabilities[BIRTH_MOVE_IDX] = INIT_PROB_BIRTH - birthDecay;// Cannot be zero as it leads to singularity in acceptance ratio equation
        moveProbabilities[LABEL_DIFFUSE_MOVE_IDX] = INIT_PROB_LABEL_DIFFUSE;
        moveProbabilities[UPDATE_CENTER_MOVE_IDX] = INIT_PROB_UPDATE_CENTER;
        moveProbabilities[EXCHANGE_DD_MOVE_IDX] = INIT_PROB_EXCHANGE_DATADRIVEN;
        moveProbabilities[EXCHANGE_MOVE_IDX] = INIT_PROB_EXCHANGE_RANDOM;
    }
    #endif
    else
    {
        moveProbabilities[BIRTH_MOVE_IDX] = INIT_PROB_BIRTH;
        moveProbabilities[LABEL_DIFFUSE_MOVE_IDX] = INIT_PROB_LABEL_DIFFUSE;
        moveProbabilities[UPDATE_CENTER_MOVE_IDX] = INIT_PROB_UPDATE_CENTER;
        moveProbabilities[EXCHANGE_DD_MOVE_IDX] = INIT_PROB_EXCHANGE_DATADRIVEN;
        moveProbabilities[EXCHANGE_MOVE_IDX] = INIT_PROB_EXCHANGE_RANDOM;
    }
#endif

#if 1
    /*
     * Death move not probable for state with single rectangle or less
     */

    if(numRects < 2)
    {
        float deathBoost = moveProbabilities[DEATH_MOVE_IDX] - 0.0001f;
        moveProbabilities[DEATH_MOVE_IDX] = 0.0001f; // Cannot be zero as it leads to singularity in acceptance ratio equation
        moveProbabilities[EXCHANGE_MOVE_IDX] += deathBoost;
    }
    else
    {
        moveProbabilities[DEATH_MOVE_IDX] = INIT_PROB_DEATH;
        moveProbabilities[EXCHANGE_MOVE_IDX] = INIT_PROB_EXCHANGE_RANDOM;
    }
#endif

}


/*
 * Combinatorial Energy Function
 */

float MCMCParserEnergy::energy(const MCMCParserStateType & state, std::vector<float>& moveProbabilities,
                               const std::vector<float> & areas,
                               const Eigen::MatrixXi & overlapConflicts,
                               const Eigen::MatrixXf & overlaps,
                               std::vector<Part>& parts
                               )
{
    // Check if this is a valid rectangle covering
    float coveredArea = 0.0f;
    int overlapCounts = 0;
    float overLapEnergy = 0.0f;

    int numRects = state.size();

#if DEBUG_MODE_ON
    std::cout<<"No of rows in overlap matrix : "<<overlapConflicts.rows()<<std::endl;
    std::cout<<"No of columns in overlap matrix : "<<overlapConflicts.cols()<<std::endl;
    std::cout<<"No of columns in parts : "<<parts.size()<<std::endl;
#endif


    for (size_t n = 0; n < numRects; n++)
    {
        coveredArea += areas[state[n]];

        for (size_t m = n+1; m < numRects; m++)
        {
        #if DEBUG_MODE_ON
            if(state[m] >= overlapConflicts.rows() || state[n] >= overlapConflicts.rows() )
                std::cout<<"invalid overlap detected...matrix updation failure. The matrices still has size : "<<overlapConflicts.rows()<<std::endl;
            #endif

            if (overlapConflicts(static_cast<int>(state[n]), static_cast<int>(state[m])))
            {
                overlapCounts++;
            }

            overLapEnergy+= overlaps(static_cast<int>(state[n]), static_cast<int>(state[m]));
        }
    }

    if(isnan(overLapEnergy))
    {
        //std::cout<<overlaps.col(parts.size()-1)<<std::endl;
    }

    /*
     * penalising with overalpa above overlap parameter
     */

    if (overlapCounts > 0.0f)// Will reject higly improbable birth moves also
    {
    #if DEBUG_MODE_ON
        std::cout<<"No of rectangle pairs with overlap : "<<overlapCounts<<std::endl;
    #endif
        const float energy = overlapCounts*1000;
        return energy;
    }


#if 0// if enabled can give faster solution but less accurate
    if(coveredArea < 0.7f)
    {
        const float energy = 100000000;
        return energy;
    }
#endif

    // Compute the parse graph
    // Set up the terminal nodes
    std::vector<ParseTreeNode*> nodes(numRects);
    for (size_t n = 0; n < numRects; n++)
    {
        nodes[n] = new ParseTreeNode();
        nodes[n]->rect = parts[state[n]].rect;
        nodes[n]->part = state[n];
    }

    ParserEnergy parserEnergy;
    ParseTreeNode* tree;
    try {
        tree = parserEnergy.parse(nodes);
    } catch(...) {
        const float energy = 10000;
        return energy;
    }

    // Traverse the tree
    std::vector<float> weightsVec;
    float weightsSum = 0.0f;
    float labelEnergy = 0.0f;
    std::vector<ParseTreeNode*> queue;
    queue.push_back(tree);
    float varianceError = 0.0f;
    int nodeCount = 0, meshCount = 0;
    while (queue.size() > 0)
    {
        ParseTreeNode* node = queue.back();
        queue.pop_back();
        nodeCount++;

        // Is this a terminal node?
        if (node->isTerminal())
        {
            // Add the weight (1 - posterior)
            weightsVec.push_back(1.0f - parts[node->part].posterior);
            weightsSum += (1.0f - parts[node->part].posterior);
        }

        // If this is a mesh, then add penalties for all labels that do
        // not match
        if (node->mesh)
        {
            meshCount++;
            int numChildren = node->children.size();
            float labelPenalty = 0.0f;

            if(numChildren>1)
            {
                labelPenalty = (double) 1/nC2(numChildren);
            }


            for (size_t c1 = 0; c1 < node->children.size(); c1++)
            {
                if (!node->children[c1]->isTerminal())
                {
                    continue;
                }
                for (size_t c2 = c1 + 1; c2 < node->children.size(); c2++)
                {
                    if (!node->children[c2]->isTerminal())
                    {
                        continue;
                    }

                    if (parts[node->children[c2]->part].label != parts[node->children[c1]->part].label)
                    {
                        labelEnergy += labelPenalty;
                    }
                }
            }
        }
        float meanWidth = 0;
        float meanHeight = 0;

        for (size_t c1 = 0; c1 < node->children.size(); c1++)
        {
            meanWidth += node->children[c1]->rect.getWidth()/image.cols;
            meanHeight += node->children[c1]->rect.getHeight()/image.rows;
        }

        meanWidth /= node->children.size();
        meanHeight /= node->children.size();

        for (size_t c1 = 0; c1 < node->children.size(); c1++)
        {
            const float temp1 = node->children[c1]->rect.getWidth()/image.cols - meanWidth;
            const float temp2 = node->children[c1]->rect.getHeight()/image.rows - meanHeight;
            varianceError += temp1*temp1 + temp2*temp2;
        }

        for (size_t i = 0; i < node->children.size(); i++)
        {
            queue.push_back(node->children[i]);
        }
    }

    delete tree;

    if(numRects == 0)// hard Constraint // might no longer be necessary as it is taken care in death move
    {
        const float energy = 10000000;
        return energy;
    }
    /*
     * Updating the probabilities in each state
     */

    updateMoveProbabilities(coveredArea, moveProbabilities, numRects);

#if 0 // Poisson Distribution
    double lambda = 5.0;
    float stateSizeEnergy = pow(lambda,static_cast<double>(numRects)) * exp(-1*lambda) / factorial(numRects);

    for(int i = 0; i < 25; i++)
    {
        float stateSizeEnergy = pow(lambda,static_cast<double>(i)) * exp(-1*lambda) / factorial(i);
        std::cout<<"State size: "<<i<<" State Size Energy: "<<stateSizeEnergy<<std::endl;
    }
#endif

    // Penalises inconsistency of labels within a group
    if(meshCount > 0)
    {
        lastLabelEnergy = labelEnergy/meshCount;
    }
    else
    {
        lastLabelEnergy = 0.0f;
    }

#if 0
    // Penalises parts with less weights (median, one or two bad weights fine as long as the majority has good weights)
    std::sort(weightsVec.begin(), weightsVec.end());

    if(numRects % 2 == 0)// even
        lastWeightEnergy = 0.5f*weightsVec[numRects/2] + 0.5f*weightsVec[(numRects/2)+1];//normalised 0-1
    else
        lastWeightEnergy = weightsVec[(numRects+1)/2];//normalised 0-1

#endif

    lastWeightEnergy = weightsSum/numRects;

    // Penalises inconsistency of structure within a group
    lastLayoutVarianceEnergy = varianceError/2/nodeCount; //normalised 0-1

    // Penalises overlap between parts
    if(numRects>1)
    {
        lastOverlapEnergy = (double) overLapEnergy/nC2(numRects)/MAX_OVERLAP;//normalised 0-1
        // TO DO: over normalised. nC2(numRects) is a very high value but theoretically correct
    }
    else
    {
        lastOverlapEnergy = 0.0f;
    }
    // Penalises drop in full covering
    lastCoverEnergy = 1.0f*(1.0f-coveredArea);//normalised ~(0-1)

    // Prefers states with more number of parts
    lastStateSizeEnergy =  (double) numRects/OPTIMUM_RECTS - 1.0f;//normalised ~(0-1)

    //lastFormFactorEnergy = computeFormFactorEnergy(state);

#if DEBUG_MODE_ON
    // Check for normalization issues if any
    if( lastLabelEnergy > 1.0f)
        std::cout<<"Invalid Label energy: "<<lastLabelEnergy<<std::endl;
    if( lastWeightEnergy > 1.0f)
        std::cout<<"Invalid Weights energy: "<<lastWeightEnergy<<std::endl;
    if( lastOverlapEnergy > 1.0f)
        std::cout<<"Invalid Overlaps energy: "<<lastOverlapEnergy<<std::endl;
    if( lastLayoutVarianceEnergy > 1.0f)
        std::cout<<"Invalid variance energy: "<<lastLayoutVarianceEnergy<<std::endl;
    std::cout<<"Temperature: "<<temperature<<std::endl;
    std::cout<<"Label Energy: "<<lastLabelEnergy<<std::endl;
    std::cout<<"Covered Area: "<<coveredArea<<std::endl;
#endif

    /*
     * Compute the total energy (6 equally weighted terms)
   */
   float energy =          lastCoverEnergy //ROI has zero cover energy
                          + lastOverlapEnergy// size = 1 has no overlap
                          + lastLabelEnergy  // size = 1 has zero label energy
                          + lastWeightEnergy
                          + lastLayoutVarianceEnergy// size = 1 has zero variance energy
                          - lastStateSizeEnergy;

#if DEBUG_MODE_ON
    std::cout<<"Cover Energy: "<<lastCoverEnergy<<std::endl;
    std::cout<<"Overlap Energy: "<<lastOverlapEnergy<<std::endl;
    std::cout<<"Label Variance Energy: "<<lastLabelEnergy<<std::endl;
    std::cout<<"Layout Variance Energy: "<<lastLayoutVarianceEnergy<<std::endl;
    std::cout<<"Weight Energy: "<<lastWeightEnergy<<std::endl;
#endif

    /*
     * Penalising the singularity at state with just bounding box rectangle
     */

    if(state.size() == 1)

    {
        energy += 0.05f;
        energy += 0.10f;
        energy += 0.05f;
        energy += 0.0f;
        energy += 0.05f;
    }
    return energy;
}

/*
 * Fom factor energy: to trim weird structures
 */

int MCMCParserEnergy::computeFormFactorEnergy(const MCMCParserStateType & state)
{
    float formFactor = 0.0f;

    for(size_t i = 0; i < state.size(); i++)
    {
        float height = parts[state[i]].rect.getHeight();
        float width  = parts[state[i]].rect.getWidth();
        formFactor += (double)(4*height*width/(height+width)/(height+width));
    }

    formFactor = (double)(formFactor/state.size());// per rectangle form-factor
    return formFactor;
}


/*
 * Factorial
 */

int MCMCParserEnergy::factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}


int MCMCParserEnergy::nC2(int n)
{
  return (n-1) * n /2;
}

