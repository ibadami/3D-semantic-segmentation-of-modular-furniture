#include "parser/rjmcmc_sa.h"
#include <math.h>

using namespace parser;

static std::random_device rd;
/*
 * Update the proposal Matrices in Height update move
 */
void SimulatedAnnealing::updateMatricesHeightDiffuse(const MCMCParserStateType state,
                                                        const std::vector<Part> partHypotheses,
                                                        std::vector<float> & areas,
                                                        const Rectangle modifiedHeightRect,
                                                        const int updateHeightPart,
                                                        Eigen::MatrixXi& overlapPairs,
                                                        Eigen::MatrixXf& overlapArea,
                                                        const float imageArea
                                                     )
{
    for (size_t m = 0; m < partHypotheses.size(); m++)
    {
        // Compute the intersection rectangle
        Rectangle intersection;
        RectangleUtil::calcIntersection(modifiedHeightRect, partHypotheses[m].rect, intersection);
        const float intersectionScore = intersection.getArea()/(std::min(modifiedHeightRect.getArea(), partHypotheses[m].rect.getArea()));
        if (intersectionScore > MAX_OVERLAP)
        {
            overlapPairs(static_cast<int>(state[updateHeightPart]),static_cast<int>(m)) = 1;
            overlapPairs(static_cast<int>(m),static_cast<int>(state[updateHeightPart])) = 1;
        }
        /*if (intersectionScore > 0.7f && intersectionScore < 1.0f)// Perfect matches also left out
        {
            overlapPairs70(static_cast<int>(state[updateHeightPart]),static_cast<int>(m)) = 1;
            overlapPairs70(static_cast<int>(m),static_cast<int>(state[updateHeightPart])) = 1;
        }*/
        overlapArea(static_cast<int>(state[updateHeightPart]),static_cast<int>(m)) = intersectionScore;
        overlapArea(static_cast<int>(m),static_cast<int>(state[updateHeightPart])) = intersectionScore;
    }

    //Update the Area matrix
    areas[state[updateHeightPart]] = modifiedHeightRect.getArea();
    areas[state[updateHeightPart]] /= imageArea;

    switch(partHypotheses[state[updateHeightPart]].label)
    {
    /*
     * Update all three labels in the matrix for each rectangle
     *for area, overlap indicator matrix and extent of overlap matrix
     */

        case 0://door
            areas[state[updateHeightPart] + static_cast<int>(1)] = areas[state[updateHeightPart]];
            areas[state[updateHeightPart] + static_cast<int>(2)] = areas[state[updateHeightPart]];

            overlapPairs.row(state[updateHeightPart] + static_cast<int>(1)) = overlapPairs.row(state[updateHeightPart]);
            overlapPairs.row(state[updateHeightPart] + static_cast<int>(2)) = overlapPairs.row(state[updateHeightPart]);
            //overlapPairs70.row(state[updateHeightPart] + static_cast<int>(1)) = overlapPairs70.row(state[updateHeightPart]);
            //overlapPairs70.row(state[updateHeightPart] + static_cast<int>(2)) = overlapPairs70.row(state[updateHeightPart]);
            overlapArea.row(state[updateHeightPart] + static_cast<int>(1)) = overlapArea.row(state[updateHeightPart]);
            overlapArea.row(state[updateHeightPart] + static_cast<int>(2)) = overlapArea.row(state[updateHeightPart]);

            overlapPairs.col(state[updateHeightPart] + static_cast<int>(1)) = overlapPairs.col(state[updateHeightPart]);
            overlapPairs.col(state[updateHeightPart] + static_cast<int>(2)) = overlapPairs.col(state[updateHeightPart]);
            //overlapPairs70.col(state[updateHeightPart] + static_cast<int>(1)) = overlapPairs70.col(state[updateHeightPart]);
            //overlapPairs70.col(state[updateHeightPart] + static_cast<int>(2)) = overlapPairs70.col(state[updateHeightPart]);
            overlapArea.col(state[updateHeightPart] + static_cast<int>(1)) = overlapArea.col(state[updateHeightPart]);
            overlapArea.col(state[updateHeightPart] + static_cast<int>(2)) = overlapArea.col(state[updateHeightPart]);
            break;

        case 1://drawer
            areas[state[updateHeightPart] - static_cast<int>(1)] = areas[state[updateHeightPart]];
            areas[state[updateHeightPart] + static_cast<int>(1)] = areas[state[updateHeightPart]];
            overlapPairs.row(state[updateHeightPart] + static_cast<int>(1)) = overlapPairs.row(state[updateHeightPart]);
            overlapPairs.row(state[updateHeightPart] - static_cast<int>(1)) = overlapPairs.row(state[updateHeightPart]);
            //overlapPairs70.row(state[updateHeightPart] + static_cast<int>(1)) = overlapPairs70.row(state[updateHeightPart]);
            //overlapPairs70.row(state[updateHeightPart] - static_cast<int>(1)) = overlapPairs70.row(state[updateHeightPart]);
            overlapArea.row(state[updateHeightPart] + static_cast<int>(1)) = overlapArea.row(state[updateHeightPart]);
            overlapArea.row(state[updateHeightPart] - static_cast<int>(1)) = overlapArea.row(state[updateHeightPart]);

            overlapPairs.col(state[updateHeightPart] + static_cast<int>(1)) = overlapPairs.col(state[updateHeightPart]);
            overlapPairs.col(state[updateHeightPart] - static_cast<int>(1)) = overlapPairs.col(state[updateHeightPart]);
            //overlapPairs70.col(state[updateHeightPart] + static_cast<int>(1)) = overlapPairs70.col(state[updateHeightPart]);
            //overlapPairs70.col(state[updateHeightPart] - static_cast<int>(1)) = overlapPairs70.col(state[updateHeightPart]);
            overlapArea.col(state[updateHeightPart] + static_cast<int>(1)) = overlapArea.col(state[updateHeightPart]);
            overlapArea.col(state[updateHeightPart] - static_cast<int>(1)) = overlapArea.col(state[updateHeightPart]);
            break;

        case 2://shelf
            areas[state[updateHeightPart] - static_cast<int>(1)] = areas[state[updateHeightPart]];
            areas[state[updateHeightPart] - static_cast<int>(2)] = areas[state[updateHeightPart]];

            overlapPairs.row(state[updateHeightPart] - static_cast<int>(1)) = overlapPairs.row(state[updateHeightPart]);
            overlapPairs.row(state[updateHeightPart] - static_cast<int>(2)) = overlapPairs.row(state[updateHeightPart]);
            //overlapPairs70.row(state[updateHeightPart] - static_cast<int>(1)) = overlapPairs70.row(state[updateHeightPart]);
            //overlapPairs70.row(state[updateHeightPart] - static_cast<int>(2)) = overlapPairs70.row(state[updateHeightPart]);
            overlapArea.row(state[updateHeightPart] - static_cast<int>(1)) = overlapArea.row(state[updateHeightPart]);
            overlapArea.row(state[updateHeightPart] - static_cast<int>(2)) = overlapArea.row(state[updateHeightPart]);

            overlapPairs.col(state[updateHeightPart] - static_cast<int>(1)) = overlapPairs.col(state[updateHeightPart]);
            overlapPairs.col(state[updateHeightPart] - static_cast<int>(2)) = overlapPairs.col(state[updateHeightPart]);
            //overlapPairs70.col(state[updateHeightPart] - static_cast<int>(1)) = overlapPairs70.col(state[updateHeightPart]);
            //overlapPairs70.col(state[updateHeightPart] - static_cast<int>(2)) = overlapPairs70.col(state[updateHeightPart]);
            overlapArea.col(state[updateHeightPart] - static_cast<int>(1)) = overlapArea.col(state[updateHeightPart]);
            overlapArea.col(state[updateHeightPart] - static_cast<int>(2)) = overlapArea.col(state[updateHeightPart]);
            break;
    }

#if 0
    std::cout<<"True Area b4 update : "<<areas[state[updateHeightPart]]<<std::endl;
    std::cout<<"ROI Area : "<<imageArea<<std::endl;
    std::cout<<"Rect area : "<<modifiedHeightRect.getArea()<<std::endl;
    std::cout<<"Rect area/ ROI area : "<<modifiedHeightRect.getArea()/imageArea<<std::endl;
#endif

    //TO DO : update Appearance Likelihood: ignore
    //TO DO : update shape prior : ignore
    //TO DO : Recalculate rectangle weight : ignore

}


void SimulatedAnnealing::updateMatricesWidthDiffuse(const MCMCParserStateType state,
                                                        const std::vector<Part> partHypotheses,
                                                        std::vector<float> & areas,
                                                        const Rectangle modifiedWidthRect,
                                                        const int updateWidthPart,
                                                        Eigen::MatrixXi& overlapPairs,
                                                        Eigen::MatrixXf& overlapArea,
                                                        const float imageArea
                                                    )
{
    for (size_t m = 0; m < partHypotheses.size(); m++)
    {
        // Compute the intersection rectangle
        Rectangle intersection;
        RectangleUtil::calcIntersection(modifiedWidthRect, partHypotheses[m].rect, intersection);
        const float intersectionScore = intersection.getArea()/(std::min(modifiedWidthRect.getArea(), partHypotheses[m].rect.getArea()));
        if (intersectionScore > MAX_OVERLAP)
        {
            overlapPairs(static_cast<int>(state[updateWidthPart]),static_cast<int>(m)) = 1;
            overlapPairs(static_cast<int>(m),static_cast<int>(state[updateWidthPart])) = 1;
        }
        /*if (intersectionScore > 0.7f && intersectionScore < 1.0f)// Perfect matches also left out
        {
            overlapPairs70(static_cast<int>(state[updateWidthPart]),static_cast<int>(m)) = 1;
            overlapPairs70(static_cast<int>(m),static_cast<int>(state[updateWidthPart])) = 1;
        }*/
        overlapArea(static_cast<int>(state[updateWidthPart]),static_cast<int>(m)) = intersectionScore;
        overlapArea(static_cast<int>(m),static_cast<int>(state[updateWidthPart])) = intersectionScore;
    }

    //Update Area Matrix
    areas[state[updateWidthPart]] = modifiedWidthRect.getArea();
    areas[state[updateWidthPart]] /= imageArea;

    switch(partHypotheses[state[updateWidthPart]].label)
    {
    /*
     * Update all three labels in the matrix for each rectangle
     *for area, overlap indicator matrix and extent of overlap matrix
     */

        case 0://door
            areas[state[updateWidthPart] + static_cast<int>(1)] = areas[state[updateWidthPart]];
            areas[state[updateWidthPart] + static_cast<int>(2)] = areas[state[updateWidthPart]];

            overlapPairs.row(state[updateWidthPart] + static_cast<int>(1)) = overlapPairs.row(state[updateWidthPart]);
            overlapPairs.row(state[updateWidthPart] + static_cast<int>(2)) = overlapPairs.row(state[updateWidthPart]);
            //overlapPairs70.row(state[updateWidthPart] + static_cast<int>(1)) = overlapPairs70.row(state[updateWidthPart]);
            //overlapPairs70.row(state[updateWidthPart] + static_cast<int>(2)) = overlapPairs70.row(state[updateWidthPart]);
            overlapArea.row(state[updateWidthPart] + static_cast<int>(1)) = overlapArea.row(state[updateWidthPart]);
            overlapArea.row(state[updateWidthPart] + static_cast<int>(2)) = overlapArea.row(state[updateWidthPart]);

            overlapPairs.col(state[updateWidthPart] + static_cast<int>(1)) = overlapPairs.col(state[updateWidthPart]);
            overlapPairs.col(state[updateWidthPart] + static_cast<int>(2)) = overlapPairs.col(state[updateWidthPart]);
            //overlapPairs70.col(state[updateWidthPart] + static_cast<int>(1)) = overlapPairs70.col(state[updateWidthPart]);
            //overlapPairs70.col(state[updateWidthPart] + static_cast<int>(2)) = overlapPairs70.col(state[updateWidthPart]);
            overlapArea.col(state[updateWidthPart] + static_cast<int>(1)) = overlapArea.col(state[updateWidthPart]);
            overlapArea.col(state[updateWidthPart] + static_cast<int>(2)) = overlapArea.col(state[updateWidthPart]);
            break;

        case 1://drawer
            areas[state[updateWidthPart] - static_cast<int>(1)] = areas[state[updateWidthPart]];
            areas[state[updateWidthPart] + static_cast<int>(1)] = areas[state[updateWidthPart]];
            overlapPairs.row(state[updateWidthPart] + static_cast<int>(1)) = overlapPairs.row(state[updateWidthPart]);
            overlapPairs.row(state[updateWidthPart] - static_cast<int>(1)) = overlapPairs.row(state[updateWidthPart]);
            //overlapPairs70.row(state[updateWidthPart] + static_cast<int>(1)) = overlapPairs70.row(state[updateWidthPart]);
            //overlapPairs70.row(state[updateWidthPart] - static_cast<int>(1)) = overlapPairs70.row(state[updateWidthPart]);
            overlapArea.row(state[updateWidthPart] + static_cast<int>(1)) = overlapArea.row(state[updateWidthPart]);
            overlapArea.row(state[updateWidthPart] - static_cast<int>(1)) = overlapArea.row(state[updateWidthPart]);

            overlapPairs.col(state[updateWidthPart] + static_cast<int>(1)) = overlapPairs.col(state[updateWidthPart]);
            overlapPairs.col(state[updateWidthPart] - static_cast<int>(1)) = overlapPairs.col(state[updateWidthPart]);
            //overlapPairs70.col(state[updateWidthPart] + static_cast<int>(1)) = overlapPairs70.col(state[updateWidthPart]);
            //overlapPairs70.col(state[updateWidthPart] - static_cast<int>(1)) = overlapPairs70.col(state[updateWidthPart]);
            overlapArea.col(state[updateWidthPart] + static_cast<int>(1)) = overlapArea.col(state[updateWidthPart]);
            overlapArea.col(state[updateWidthPart] - static_cast<int>(1)) = overlapArea.col(state[updateWidthPart]);
            break;

        case 2://shelf
            areas[state[updateWidthPart] - static_cast<int>(1)] = areas[state[updateWidthPart]];
            areas[state[updateWidthPart] - static_cast<int>(2)] = areas[state[updateWidthPart]];

            overlapPairs.row(state[updateWidthPart] - static_cast<int>(1)) = overlapPairs.row(state[updateWidthPart]);
            overlapPairs.row(state[updateWidthPart] - static_cast<int>(2)) = overlapPairs.row(state[updateWidthPart]);
            //overlapPairs70.row(state[updateWidthPart] - static_cast<int>(1)) = overlapPairs70.row(state[updateWidthPart]);
            //overlapPairs70.row(state[updateWidthPart] - static_cast<int>(2)) = overlapPairs70.row(state[updateWidthPart]);
            overlapArea.row(state[updateWidthPart] - static_cast<int>(1)) = overlapArea.row(state[updateWidthPart]);
            overlapArea.row(state[updateWidthPart] - static_cast<int>(2)) = overlapArea.row(state[updateWidthPart]);

            overlapPairs.col(state[updateWidthPart] - static_cast<int>(1)) = overlapPairs.col(state[updateWidthPart]);
            overlapPairs.col(state[updateWidthPart] - static_cast<int>(2)) = overlapPairs.col(state[updateWidthPart]);
            //overlapPairs70.col(state[updateWidthPart] - static_cast<int>(1)) = overlapPairs70.col(state[updateWidthPart]);
            //overlapPairs70.col(state[updateWidthPart] - static_cast<int>(2)) = overlapPairs70.col(state[updateWidthPart]);
            overlapArea.col(state[updateWidthPart] - static_cast<int>(1)) = overlapArea.col(state[updateWidthPart]);
            overlapArea.col(state[updateWidthPart] - static_cast<int>(2)) = overlapArea.col(state[updateWidthPart]);
            break;
    }

}


void SimulatedAnnealing::updateMatricesCenterLocDiffuse(const MCMCParserStateType state,
                                                        const std::vector<Part> partHypotheses,
                                                        const Rectangle modifiedCenterRect,
                                                        const int updateCenterPart,
                                                        Eigen::MatrixXi& overlapPairs,
                                                        Eigen::MatrixXf& overlapArea)
{
    for (size_t m = 0; m < partHypotheses.size(); m++)
    {
        // Compute the intersection rectangle
        Rectangle intersection;
        RectangleUtil::calcIntersection(modifiedCenterRect, partHypotheses[m].rect, intersection);
        const float intersectionScore = intersection.getArea()/(std::min(modifiedCenterRect.getArea(), partHypotheses[m].rect.getArea()));
        if (intersectionScore > MAX_OVERLAP)
        {
            overlapPairs(static_cast<int>(state[updateCenterPart]),static_cast<int>(m)) = 1;
            overlapPairs(static_cast<int>(m),static_cast<int>(state[updateCenterPart])) = 1;
        }
        /*if (intersectionScore > 0.7f && intersectionScore < 1.0f)// Perfect matches also left out
        {
            overlapPairs70(static_cast<int>(state[updateCenterPart]),static_cast<int>(m)) = 1;
            overlapPairs70(static_cast<int>(m),static_cast<int>(state[updateCenterPart])) = 1;
        }*/
        overlapArea(static_cast<int>(state[updateCenterPart]),static_cast<int>(m)) = intersectionScore;
        overlapArea(static_cast<int>(m),static_cast<int>(state[updateCenterPart])) = intersectionScore;
    }
    /*
     * Update all three labels in the matrix for each rectangle
     *for area, overlap indicator matrix and extent of overlap matrix
     */

    switch(partHypotheses[state[updateCenterPart]].label)
    {
        case 0://door
            overlapPairs.row(state[updateCenterPart] + static_cast<int>(1)) = overlapPairs.row(state[updateCenterPart]);
            overlapPairs.row(state[updateCenterPart] + static_cast<int>(2)) = overlapPairs.row(state[updateCenterPart]);
            //overlapPairs70.row(state[updateCenterPart] + static_cast<int>(1)) = overlapPairs70.row(state[updateCenterPart]);
            //overlapPairs70.row(state[updateCenterPart] + static_cast<int>(2)) = overlapPairs70.row(state[updateCenterPart]);
            overlapArea.row(state[updateCenterPart] + static_cast<int>(1)) = overlapArea.row(state[updateCenterPart]);
            overlapArea.row(state[updateCenterPart] + static_cast<int>(2)) = overlapArea.row(state[updateCenterPart]);

            overlapPairs.col(state[updateCenterPart] + static_cast<int>(1)) = overlapPairs.col(state[updateCenterPart]);
            overlapPairs.col(state[updateCenterPart] + static_cast<int>(2)) = overlapPairs.col(state[updateCenterPart]);
            //overlapPairs70.col(state[updateCenterPart] + static_cast<int>(1)) = overlapPairs70.col(state[updateCenterPart]);
            //overlapPairs70.col(state[updateCenterPart] + static_cast<int>(2)) = overlapPairs70.col(state[updateCenterPart]);
            overlapArea.col(state[updateCenterPart] + static_cast<int>(1)) = overlapArea.col(state[updateCenterPart]);
            overlapArea.col(state[updateCenterPart] + static_cast<int>(2)) = overlapArea.col(state[updateCenterPart]);
            break;

        case 1://drawer
            overlapPairs.row(state[updateCenterPart] + static_cast<int>(1)) = overlapPairs.row(state[updateCenterPart]);
            overlapPairs.row(state[updateCenterPart] - static_cast<int>(1)) = overlapPairs.row(state[updateCenterPart]);
            //overlapPairs70.row(state[updateCenterPart] + static_cast<int>(1)) = overlapPairs70.row(state[updateCenterPart]);
            //overlapPairs70.row(state[updateCenterPart] - static_cast<int>(1)) = overlapPairs70.row(state[updateCenterPart]);
            overlapArea.row(state[updateCenterPart] + static_cast<int>(1)) = overlapArea.row(state[updateCenterPart]);
            overlapArea.row(state[updateCenterPart] - static_cast<int>(1)) = overlapArea.row(state[updateCenterPart]);

            overlapPairs.col(state[updateCenterPart] + static_cast<int>(1)) = overlapPairs.col(state[updateCenterPart]);
            overlapPairs.col(state[updateCenterPart] - static_cast<int>(1)) = overlapPairs.col(state[updateCenterPart]);
            //overlapPairs70.col(state[updateCenterPart] + static_cast<int>(1)) = overlapPairs70.col(state[updateCenterPart]);
            //overlapPairs70.col(state[updateCenterPart] - static_cast<int>(1)) = overlapPairs70.col(state[updateCenterPart]);
            overlapArea.col(state[updateCenterPart] + static_cast<int>(1)) = overlapArea.col(state[updateCenterPart]);
            overlapArea.col(state[updateCenterPart] - static_cast<int>(1)) = overlapArea.col(state[updateCenterPart]);
            break;

        case 2://shelf
            overlapPairs.row(state[updateCenterPart] - static_cast<int>(1)) = overlapPairs.row(state[updateCenterPart]);
            overlapPairs.row(state[updateCenterPart] - static_cast<int>(2)) = overlapPairs.row(state[updateCenterPart]);
            //overlapPairs70.row(state[updateCenterPart] - static_cast<int>(1)) = overlapPairs70.row(state[updateCenterPart]);
            //overlapPairs70.row(state[updateCenterPart] - static_cast<int>(2)) = overlapPairs70.row(state[updateCenterPart]);
            overlapArea.row(state[updateCenterPart] - static_cast<int>(1)) = overlapArea.row(state[updateCenterPart]);
            overlapArea.row(state[updateCenterPart] - static_cast<int>(2)) = overlapArea.row(state[updateCenterPart]);

            overlapPairs.col(state[updateCenterPart] - static_cast<int>(1)) = overlapPairs.col(state[updateCenterPart]);
            overlapPairs.col(state[updateCenterPart] - static_cast<int>(2)) = overlapPairs.col(state[updateCenterPart]);
            //overlapPairs70.col(state[updateCenterPart] - static_cast<int>(1)) = overlapPairs70.col(state[updateCenterPart]);
            //overlapPairs70.col(state[updateCenterPart] - static_cast<int>(2)) = overlapPairs70.col(state[updateCenterPart]);
            overlapArea.col(state[updateCenterPart] - static_cast<int>(1)) = overlapArea.col(state[updateCenterPart]);
            overlapArea.col(state[updateCenterPart] - static_cast<int>(2)) = overlapArea.col(state[updateCenterPart]);
            break;
    }


}


/*
 * Upadting Height of rectangle: Diffusion Move
*/
void SimulatedAnnealing::diffuseRectHeight(const Rectangle originalHeightRect, Rectangle & modifiedHeightRect)
{
    // TO DO: Set boundary conditions so that the new rectangle wont go out of the ROI
    // TO DO: shape prior remains same but appearance changes
    std::uniform_int_distribution<int> heightChangeDist(static_cast<int>(-1.0f*WH_DIFFUSE_FACTOR*originalHeightRect.getHeight()), static_cast<int>(1.0f*WH_DIFFUSE_FACTOR*originalHeightRect.getHeight())); // +/-5% of height
    std::mt19937 g(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    modifiedHeightRect = originalHeightRect;
    //random updates
    int heightDeltaTop = heightChangeDist(g);
    int heightDeltaBot = heightChangeDist(g);

#if DEBUG_MODE_ON
    std::cout<<"Height Delta Top is : "<<heightDeltaTop<<std::endl;
    std::cout<<"Height Delta Bot is : "<<heightDeltaBot<<std::endl;
#endif
    //New rectangle with changed height
    modifiedHeightRect[0][1] += heightDeltaTop;
    modifiedHeightRect[1][1] += heightDeltaTop;
    modifiedHeightRect[2][1] += heightDeltaBot;
    modifiedHeightRect[3][1] += heightDeltaBot;
}


/*
 * Upadting Width of rectangle: Diffusion Move
*/
void SimulatedAnnealing::diffuseRectWidth(const Rectangle originalWidthRect, Rectangle & modifiedWidthRect)
{
    // TO DO: Set boundary conditions so that the new rectangle wont go out of the ROI
    // TO DO: shape prior remains same but appearance changes
    std::uniform_int_distribution<int> widthChangeDist(static_cast<int>(-1.0f*WH_DIFFUSE_FACTOR*originalWidthRect.getWidth()), static_cast<int>(1.0f*WH_DIFFUSE_FACTOR*originalWidthRect.getWidth())); // +/-5% of width
    std::mt19937 g(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    modifiedWidthRect = originalWidthRect;

    //random updates
    int widthDeltaLeft = widthChangeDist(g);
    int widthDeltaRight = widthChangeDist(g);

    //New rectangle with changed width
    modifiedWidthRect[0][0] += widthDeltaLeft;
    modifiedWidthRect[1][0] += widthDeltaRight;
    modifiedWidthRect[2][0] += widthDeltaRight;
    modifiedWidthRect[3][0] += widthDeltaLeft;
}


/*
 * Upadting Center Location of rectangle: Diffusion Move
*/
void SimulatedAnnealing::diffuseCenterLoc(const Rectangle originalCenterRect, Rectangle & modifiedCenterRect)
{
    // TO DO: Set boundary conditions so that the new rectangle wont go out of the ROI
    // TO DO: shape prior remains same but appearance changes
    std::uniform_int_distribution<int> heightChangeDist(static_cast<int>(-1.0f*CENTER_DIFFUSE_FACTOR*originalCenterRect.getHeight()), static_cast<int>(1.0f*CENTER_DIFFUSE_FACTOR*originalCenterRect.getHeight())); // +/-2.5% of height
    std::uniform_int_distribution<int> widthChangeDist(static_cast<int>(-1.0f*CENTER_DIFFUSE_FACTOR*originalCenterRect.getWidth()), static_cast<int>(1.0f*CENTER_DIFFUSE_FACTOR*originalCenterRect.getWidth())); // +/-2.5% of width
    std::mt19937 g(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    modifiedCenterRect = originalCenterRect;
    //Random update
    int heightDelta = heightChangeDist(g);
    int widthDelta = widthChangeDist(g);

    //New rectangle with updated location
    modifiedCenterRect[0][0] += widthDelta;
    modifiedCenterRect[0][1] += heightDelta;

    modifiedCenterRect[1][0] += widthDelta;
    modifiedCenterRect[1][1] += heightDelta;

    modifiedCenterRect[2][0] += widthDelta;
    modifiedCenterRect[2][1] += heightDelta;

    modifiedCenterRect[3][0] += widthDelta;
    modifiedCenterRect[3][1] += heightDelta;
}


#if 0
void SimulatedAnnealing::updateAppearanceLikelihood()
{
    // Load the appearance codebook
    std::vector<Eigen::MatrixXf> codebooks(5);
    std::ifstream res("codebook.dat");
    for (int l = 0; l < 3; l++)
    {
        libf::readBinary(res, codebooks[l]);
        std::stringstream ss;
        ss << l << "_codebook.csv";
        std::ofstream o(ss.str());
        o << codebooks[l];
        o.close();
    }
    res.close();

    float reconstructionErrors[3];
    libf::DataPoint p, p2;
    CabinetParser parserObj;
    parserObj.extractDiscretizedAppearanceDataGM(gradMag, modifiedRect, p, p2);

    for (int l = 0; l < 3; l++)
    {
        // Get the reconstruction error
        reconstructionErrors[l] = parserObj.calcCodebookError(codebooks[l], p);
    }

    //Re-calculate for all three labels, but keep the current label

    int l = 0;
    partHypotheses[state[updatePart]].likelihood = exp(-reconstructionErrors[l]/0.01f);;
    partHypotheses[state[updatePart]].posterior = partHypotheses[state[updatePart]].posterior;

}
#endif



/*
 * Split a rectangle into two sub rectangles
*/
void SimulatedAnnealing::split1Rectangle(const MCMCParserStateType state, const int splitPart,
                                         const Part originalPartB4Split,
                                         std::vector<Part> & partHypotheses,
                                         Part & splitPartR1, Part & splitPartR2)
{
    splitPartR1 = originalPartB4Split;
    splitPartR2 = originalPartB4Split;
    //Randomized split location
    std::uniform_int_distribution<int> splitDist(0, partHypotheses[state[splitPart]].projProf.size()-1 );
    std::mt19937 g(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    int splitPos = partHypotheses[state[splitPart]].projProf[splitDist(g)];
    //Edge Projection profile
    splitPartR1.projProf.clear(); //New rectangle 1
    splitPartR2.projProf.clear();// New Rectangle 2

    for(int i = 0; i< originalPartB4Split.projProf.size(); i++)
    {
        if(originalPartB4Split.projProf[i] < splitPos)
        {
            splitPartR1.projProf.push_back(originalPartB4Split.projProf[i]);
        }
        else if(originalPartB4Split.projProf[i] > splitPos)
        {
            splitPartR2.projProf.push_back(originalPartB4Split.projProf[i]);
        }
    }

    if((splitPartR1.projProf.size() + splitPartR2.projProf.size() + 1) !=  originalPartB4Split.projProf.size())
    {
#if DEBUG_MODE_ON
        std::cout<<"Error: Splitting the edge profiles failed"<<std::endl;
        std::cout<<"before split: "<<originalPartB4Split.projProf.size()<<std::endl;
        std::cout<<"after split: "<<splitPartR1.projProf.size()<<std::endl;
        std::cout<<"after split: "<<splitPartR2.projProf.size()<<std::endl;
#endif
    }

    if(splitPartR1.rect.getArea() == 0 or splitPartR2.rect.getArea() == 0)
    {
        std::cout<<"Invalid split..zero area detected"<<std::endl;
    }

#if DEBUG_MODE_ON
    std::cout<<"original Part area 1 : "<<originalPartB4Split.rect.getArea()<<std::endl;
    std::cout<<"Split Part R1 area 1 : "<<splitPartR1.rect.getArea()<<std::endl;
    std::cout<<"Split Part R2 area 1 : "<<splitPartR2.rect.getArea()<<std::endl;

    std::cout<<"split position is : "<<splitPos<<" out of : "<<originalPartB4Split.rect.getWidth()<<std::endl;
#endif

#if DEBUG_MODE_ON
    std::cout<<"sub move: diffuse "<<std::endl;
#endif

    //Splitting
    splitPartR1.rect[1][0] = splitPos;
    splitPartR1.rect[2][0] = splitPos;

    splitPartR2.rect[0][0] = splitPos + 1;
    splitPartR2.rect[3][0] = splitPos + 1;

#if DEBUG_MODE_ON
    std::cout<<"original Part area 2 : "<<originalPartB4Split.rect.getArea()<<std::endl;
    std::cout<<"Split Part R1 area 2 : "<<splitPartR1.rect.getArea()<<std::endl;
    std::cout<<"Split Part R2 area 2 : "<<splitPartR2.rect.getArea()<<std::endl;
#endif

    if(originalPartB4Split.rect.getArea() != ( splitPartR1.rect.getArea() + splitPartR2.rect.getArea() ) )
    {
#if DEBUG_MODE_ON
        std::cout<<"Invalid split move detected"<<std::endl;
#endif
    }

#if DEBUG_MODE_ON
    std::cout<<"sub move: birth "<<std::endl;
#endif

#if DEBUG_MODE_ON
    std::cout<<"No of parts before split "<<partHypotheses.size()<<std::endl;
#endif

    //partHypotheses[state[splitPart]] = splitPartR1;
    partHypotheses.push_back(splitPartR1);
    partHypotheses.push_back(splitPartR2);

#if DEBUG_MODE_ON
    std::cout<<"No of parts after split "<<partHypotheses.size()<<std::endl;
#endif

}



/*
 * Updating the proposal matrices of rectangle after split Move
*/
void SimulatedAnnealing::updateMatricesSplit(const std::vector<Part> partHypotheses, std::vector<float> & areas,
                                             const Part splitPartR1, const Part splitPartR2, const float imageArea,
                                             Eigen::MatrixXi& overlapPairs, Eigen::MatrixXf& overlapArea)
{
    //Initialize
    Eigen::MatrixXi overlapPairsNew = Eigen::MatrixXi::Zero(static_cast<int>(partHypotheses.size()),static_cast<int>(partHypotheses.size()));
    //Eigen::MatrixXi overlapPairs70New = Eigen::MatrixXi::Zero(static_cast<int>(partHypotheses.size()),static_cast<int>(partHypotheses.size()));
    Eigen::MatrixXf overlapAreaNew = Eigen::MatrixXf::Zero(static_cast<int>(partHypotheses.size()),static_cast<int>(partHypotheses.size()));

    for (size_t m = 0; m < partHypotheses.size(); m++)
    {
        // Compute the intersection rectangle
        Rectangle intersection;
        RectangleUtil::calcIntersection(splitPartR1.rect, partHypotheses[m].rect, intersection);
        const float intersectionScore = intersection.getArea()/(std::min(splitPartR1.rect.getArea(), partHypotheses[m].rect.getArea()));
        if (intersectionScore > MAX_OVERLAP)
        {
            overlapPairsNew(static_cast<int>(partHypotheses.size()-2),static_cast<int>(m)) = 1;
            overlapPairsNew(static_cast<int>(m),static_cast<int>(partHypotheses.size()-2)) = 1;
        }
        /*if (intersectionScore > 0.7f && intersectionScore < 1.0f)// Perfect matches also left out
        {
            overlapPairs70New(static_cast<int>(partHypotheses.size()-2),static_cast<int>(m)) = 1;
            overlapPairs70New(static_cast<int>(m),static_cast<int>(partHypotheses.size()-2)) = 1;
        }*/
        if(isnan(intersectionScore))
        {
#if DEBUG_MODE_ON
            std::cout<<"NaN detected in matrix"<<std::endl;
            std::cout<<"intersection Area: "<<intersection.getArea()<<std::endl;
            std::cout<<"Split part r1 area: "<<splitPartR1.rect.getArea()<<std::endl;
            std::cout<<"part Hypotheses Area: "<<partHypotheses[m].rect.getArea()<<std::endl;
            std::cout<<"part Hypotheses index: "<<m<<" out of "<<partHypotheses.size()<<std::endl;
#endif
        }
        overlapAreaNew(static_cast<int>(partHypotheses.size()-2),static_cast<int>(m)) = intersectionScore;
        overlapAreaNew(static_cast<int>(m),static_cast<int>(partHypotheses.size()-2)) = intersectionScore;
    }


    for (size_t m = 0; m < partHypotheses.size(); m++)
    {
        // Compute the intersection rectangle
        Rectangle intersection;
        RectangleUtil::calcIntersection(splitPartR2.rect, partHypotheses[m].rect, intersection);
        const float intersectionScore = intersection.getArea()/(std::min(splitPartR2.rect.getArea(), partHypotheses[m].rect.getArea()));
        if (intersectionScore > MAX_OVERLAP)
        {
            overlapPairsNew(static_cast<int>(partHypotheses.size()-1),static_cast<int>(m)) = 1;
            overlapPairsNew(static_cast<int>(m),static_cast<int>(partHypotheses.size()-1)) = 1;
        }
        /*if (intersectionScore > 0.7f && intersectionScore < 1.0f)// Perfect matches also left out
        {
            overlapPairs70New(static_cast<int>(partHypotheses.size()-1),static_cast<int>(m)) = 1;
            overlapPairs70New(static_cast<int>(m),static_cast<int>(partHypotheses.size()-1)) = 1;
        }*/
        overlapAreaNew(static_cast<int>(partHypotheses.size()-1),static_cast<int>(m)) = intersectionScore;
        overlapAreaNew(static_cast<int>(m),static_cast<int>(partHypotheses.size()-1)) = intersectionScore;
    }

    overlapPairsNew.block(0,0,partHypotheses.size()-3,partHypotheses.size()-3) = overlapPairs;
    //overlapPairs70New.block(0,0,partHypotheses.size()-3,partHypotheses.size()-3) = overlapPairs70;
    overlapAreaNew.block(0,0,partHypotheses.size()-3,partHypotheses.size()-3) = overlapArea;

    overlapPairs = overlapPairsNew;
    //overlapPairs70 = overlapPairs70New;
    overlapArea = overlapAreaNew;


    //Update Area Matrix
    float r1Area = splitPartR1.rect.getArea();
    r1Area /= imageArea;
    float r2Area = splitPartR2.rect.getArea();
    r2Area /= imageArea;
    areas.push_back(r1Area);
    areas.push_back(r2Area);

}


/*
 * Updating Proposal Matrices after Merge Move
*/
void SimulatedAnnealing::updateMatricesMerge(const std::vector<Part> partHypotheses, const Rectangle mergedRect,
                                             Eigen::MatrixXi& overlapPairs, Eigen::MatrixXf& overlapArea)
{
    //Initialize
    Eigen::MatrixXi overlapPairsMergeNew = Eigen::MatrixXi::Zero(static_cast<int>(partHypotheses.size()),static_cast<int>(partHypotheses.size()));
    //Eigen::MatrixXi overlapPairs70MergeNew = Eigen::MatrixXi::Zero(static_cast<int>(partHypotheses.size()),static_cast<int>(partHypotheses.size()));
    Eigen::MatrixXf overlapAreaMergeNew = Eigen::MatrixXf::Zero(static_cast<int>(partHypotheses.size()),static_cast<int>(partHypotheses.size()));

    for (size_t m = 0; m < partHypotheses.size(); m++)
    {
        // Compute the intersection rectangle
        Rectangle intersection;
        RectangleUtil::calcIntersection(mergedRect, partHypotheses[m].rect, intersection);
        const float intersectionScore = intersection.getArea()/(std::min(mergedRect.getArea(), partHypotheses[m].rect.getArea()));
        if (intersectionScore > MAX_OVERLAP)
        {
            overlapPairsMergeNew(static_cast<int>(partHypotheses.size()-1),static_cast<int>(m)) = 1;
            overlapPairsMergeNew(static_cast<int>(m),static_cast<int>(partHypotheses.size()-1)) = 1;
        }
        /*if (intersectionScore > 0.7f && intersectionScore < 1.0f)// Perfect matches also left out
        {
            overlapPairs70MergeNew(static_cast<int>(partHypotheses.size()-1),static_cast<int>(m)) = 1;
            overlapPairs70MergeNew(static_cast<int>(m),static_cast<int>(partHypotheses.size()-1)) = 1;
        }*/
        if(isnan(intersectionScore))
        {
            std::cout<<"NaN detected in matrix"<<std::endl;
            std::cout<<"intersection Area: "<<intersection.getArea()<<std::endl;
            std::cout<<"Split part r1 area: "<<mergedRect.getArea()<<std::endl;
            std::cout<<"part Hypotheses Area: "<<partHypotheses[m].rect.getArea()<<std::endl;
            std::cout<<"part Hypotheses index: "<<m<<" out of "<<partHypotheses.size()<<std::endl;

        }
        overlapAreaMergeNew(static_cast<int>(partHypotheses.size()-1),static_cast<int>(m)) = intersectionScore;
        overlapAreaMergeNew(static_cast<int>(m),static_cast<int>(partHypotheses.size()-1)) = intersectionScore;
    }

    overlapPairsMergeNew.block(0,0,partHypotheses.size()-2,partHypotheses.size()-2) = overlapPairs;
    //overlapPairs70New.block(0,0,partHypotheses.size()-2,partHypotheses.size()-2) = overlapPairs70;
    overlapAreaMergeNew.block(0,0,partHypotheses.size()-2,partHypotheses.size()-2) = overlapArea;

    overlapPairs = overlapPairsMergeNew;
    //overlapPairs70 = overlapPairs70MergeNew;
    overlapArea = overlapAreaMergeNew;

}

/*
 * Merging two rectangles into a single new rectangle
*/

void SimulatedAnnealing::merge2Rectangles(const MCMCParserStateType state, std::vector<Part> & partHypotheses,
                                          const int rectIdx1, const int rectIdx2, Part & mergedPart)
{
    //TO DO: find the indices of two rectangles being merged
    Rectangle toMergeRectR1 = partHypotheses[state[rectIdx1]].rect;
    Rectangle toMergeRectR2 = partHypotheses[state[rectIdx2]].rect;
    Rectangle mergedRect = toMergeRectR1;// so that y coordinates will be automatically assigned

    //The outer rectangle os preferred
    mergedRect[0][0] = std::min(toMergeRectR1[0][0],toMergeRectR2[0][0]);//TL x coord
    mergedRect[3][0] = std::min(toMergeRectR1[3][0],toMergeRectR2[3][0]);//BL x coord
    mergedRect[1][0] = std::max(toMergeRectR1[1][0],toMergeRectR2[1][0]);//TL x coord
    mergedRect[2][0] = std::max(toMergeRectR1[2][0],toMergeRectR2[2][0]);//BL x coord

    if(mergedRect.getArea() != (toMergeRectR1.getArea() + toMergeRectR2.getArea()))
    {
        std::cout<<"Invalid merge move detected"<<std::endl;
        std::cout<<"rect 1 area: "<<toMergeRectR1.getArea()<<std::endl;
        std::cout<<"rect 2 area: "<<toMergeRectR2.getArea()<<std::endl;
        std::cout<<"area after merge: "<<mergedRect.getArea()<<std::endl;
    }

    mergedPart = partHypotheses[state[rectIdx1]];//TO DO: change this
    mergedPart.rect = mergedRect;

    partHypotheses.push_back(mergedPart);
}

/*
 * Not all rectangle pairs are mergeable: Finding mergeability
*/

void SimulatedAnnealing::computeMergeability(const MCMCParserStateType state, const std::vector<Part> partHypotheses,
                                             int & rectIdx1, int & rectIdx2)
{
    std::vector<int> mergeSet1, mergeSet2;

    mergeSet1.clear();
    mergeSet2.clear();
    int mergeFlag = 0;

    /*
     * Three criteria for merging
    */

    if(state.size() > 1)
    {
        for(int i = 0; i< state.size(); i++)
        {
            for(int j = i+1; j< state.size(); j++)
            {
                if(  std::abs(partHypotheses[state[i]].rect.getHeight() - partHypotheses[state[j]].rect.getHeight() ) < MERGE_ALLOWANCE
                     && // height criteria
                     (std::abs( partHypotheses[state[i]].rect.getCenter()[0] - partHypotheses[state[j]].rect.getCenter()[0] ) - partHypotheses[state[i]].rect.getWidth()/2 - partHypotheses[state[j]].rect.getWidth()/2)  < MERGE_ALLOWANCE
                     && // center distance criteria
                    std::abs( partHypotheses[state[i]].rect.getCenter()[1] - partHypotheses[state[j]].rect.getCenter()[1] ) < MERGE_ALLOWANCE
                     ) // center aligning criteria
                {
                    //std::cout<<"width merging possible "<<std::endl;
                    mergeFlag = 1;

                    mergeSet1.push_back(static_cast<int>(i));
                    mergeSet2.push_back(static_cast<int>(j));
                }
            }
        }
    }

    if(mergeSet1.size()>0)
    {
        std::cout<<"width mergeable pairs: "<<mergeSet1.size()<<" out of : "<<state.size()*(state.size()-1)/2<<std::endl;
        plotMarkovChainState(state,partHypotheses);
    }

    std::uniform_int_distribution<int> mergeDist(static_cast<int>(0), static_cast<int>(mergeSet1.size()-1) );
    std::mt19937 g(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    int mergeIdx = mergeDist(g);

    /*
     * Index of mergeable rectangles
    */
    rectIdx1 = mergeSet1[mergeIdx];
    rectIdx2 = mergeSet2[mergeIdx];

}

/*
 * Select the type of the move
*/

int SimulatedAnnealing::selectRJMCMCMoveType(const float u)
{
    int randomMove = -1;
    float probSum = 0;
    for (size_t i = 0; i < moveProbabilities.size(); i++)
    {
        if (probSum <= u && u < probSum+moveProbabilities[i])
        {
            randomMove = static_cast<int>(i);
            break;
        }
        else
        {
            probSum += moveProbabilities[i];
        }
    }
    // Did we find a move?
    if (randomMove < 0)
    {
        // Nope, then we sampled the last move
        randomMove = static_cast<int>(moveProbabilities.size() - 1);
    }

    return randomMove;
}

/*
 * Parameter Display
*/

void SimulatedAnnealing::displaySimAnnealParams()
{
    std::cout<<std::endl;
    std::cout<<"Simulated Annealing Parameter Settings "<<std::endl;
    std::cout<<"Inner loops : "<<numInnerLoops<<std::endl;
    std::cout<<"min non-update (convergence) iterations : "<<maxNoUpdateIterations<<std::endl;
    std::cout<<"Starting Temperature : "<<MAX_TEMP<<std::endl;
    std::cout<<"Final Temperature : "<<MIN_TEMP<<std::endl;
    std::cout<<"Alpha (cooling) : "<<ALPHA<<std::endl;
    std::cout<<std::endl;
}


/*
 * Plot the state on a window: debug purpose
*/

void SimulatedAnnealing::plotMarkovChainState(const MCMCParserStateType state, const std::vector<Part> partHypotheses)
{
    cv::Mat demo(cannyEdges.rows, cannyEdges.cols, CV_8UC3);
    cannyEdges.copyTo(demo);
    cv::cvtColor(demo, demo, CV_GRAY2BGR);

    for (size_t h = 0; h < state.size(); h++)
    {
        switch (partHypotheses[state[h]].label)
        {
            case 0:
                PlotUtil::plotRectangle(demo, partHypotheses[state[h]].rect, cv::Scalar(0,0,255), 2);
                break;
            case 1:
                PlotUtil::plotRectangle(demo, partHypotheses[state[h]].rect, cv::Scalar(0,255,0), 2);
                break;
            case 2:
                PlotUtil::plotRectangle(demo, partHypotheses[state[h]].rect, cv::Scalar(255,0,0), 2);
            break;
        }
    }
        cv::imshow("test", demo);
        cv::waitKey(0);
}

/*
 * The trans-dimensional optimization function using simulated annealing variant of rjMCMC
  * Joint optimization over structure and class labels
*/


float SimulatedAnnealing::optimize(MCMCParserStateType & state)
{
    //std::random_device rd;
    std::mt19937 g(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    // Set up a uniform distribution. We need this in order to accept
    // hill climbing steps and choose the moves
    std::uniform_real_distribution<float> uniformDist(0,1);

    // Get the temperature
    float temperature = coolingSchedule.getStartTemperature();

    // Get the current error
    float currentEnergy = energyFunction.energy(state, moveProbabilities, areas, overlapPairs, overlapArea, partHypotheses);

    // Keep track on the optimum
    float bestEnergy = currentEnergy;
    MCMCParserStateType bestState = state;

    // Start the optimization
    int iteration = 0;
    int noUpdateIterations = 0;

    displaySimAnnealParams();

    // Track the move selection: debug purpose
    int exchangeCount = 0, specialExchangeCount = 0, totalExchangeCount = 0;
    int birthCount = 0, specialBirthCount = 0, totalBirthCount = 0;
    int deathCount = 0, specialDeathCount = 0, totalDeathCount = 0;
    int splitCount = 0, specialSplitCount = 0, totalSplitCount = 0;
    int mergeCount = 0, specialMergeCount = 0, totalMergeCount = 0;
    int labelDiffuseCount = 0, specialLabelDiffuseCount = 0, totalLabelDiffuseCount = 0;
    int exchangeDDCount = 0, specialExchangeDDCount = 0, totalExchangeDDCount = 0;
    int updateCenterCount = 0, specialUpdateCenterCount = 0, totalUpdateCenterCount = 0;
    int updateWidthCount = 0, specialUpdateWidthCount = 0, totalUpdateWidthCount = 0;
    int updateHeightCount = 0, specialUpdateHeightCount = 0, totalUpdateHeightCount = 0;

    float posteriorFactor = 0.0f, moveProbfactor = 0.0f;

    while (temperature > coolingSchedule.getEndTemperature())
    {
        iteration++;
        temperature = coolingSchedule.calcTemperature(iteration, temperature);
        bool update = false;
        for (int inner = 0; inner < numInnerLoops; inner++)
        {

            // Choose a move at random
            const float u = uniformDist(g);
            int randomMove = selectRJMCMCMoveType(u);

            // Get the result of the move
            MCMCParserStateType newState;
            float logAcceptRatio;
            moves[randomMove]->move(state, newState, logAcceptRatio);

            Rectangle originalCenterRect, originalWidthRect, originalHeightRect;
            Part originalPartB4Split, originalPartB4Merge;
            Part splitPartR1, splitPartR2;
            int updateCenterPart, updateWidthPart, updateHeightPart, metaSplitPart, splitPart, mergePart;

            float imageArea = gradMag.rows;
            imageArea *= gradMag.cols;

            Eigen::MatrixXi originalOverlapPairsSplit, originalOverlapPairsMerge, originalOverlapPairsCenter, originalOverlapPairsWidth, originalOverlapPairsHeight;
            //Eigen::MatrixXi originalOverlapPairs70Split, originalOverlapPairs70Merge, originalOverlapPairs70Center, originalOverlapPairs70Width, originalOverlapPairs70Height;
            Eigen::MatrixXf originalOverlapAreaSplit, originalOverlapAreaMerge, originalOverlapAreaCenter, originalOverlapAreaWidth, originalOverlapAreaHeight;

            //Move specific acceptance ratios
            if(randomMove == EXCHANGE_MOVE_IDX)//EXCHANGE MOVE
            {
            #if DEBUG_MODE_ON
                std::cout<<"exchange move "<<std::endl;
            #endif
                moveProbfactor = 0.0f;
                totalExchangeCount++;
            }
            else if(randomMove == BIRTH_MOVE_IDX)//BIRTH MOVE
            {
            #if DEBUG_MODE_ON
                std::cout<<"birth move "<<std::endl;
            #endif
                moveProbfactor = std::log((double)moveProbabilities[DEATH_MOVE_IDX]/moveProbabilities[BIRTH_MOVE_IDX]);// Move probability ratio
                totalBirthCount++;
            }
            else if(randomMove == DEATH_MOVE_IDX)//DEATH MOVE
            {
            #if DEBUG_MODE_ON
                std::cout<<"death move "<<std::endl;
            #endif
                moveProbfactor = std::log((double)moveProbabilities[BIRTH_MOVE_IDX]/moveProbabilities[DEATH_MOVE_IDX]);//Move probability ratio
                totalDeathCount++;
            }
            else if(randomMove == SPLIT_MOVE_IDX)//SPLIT MOVE
            {

            #if DEBUG_MODE_ON
                std::cout<<"split move "<<std::endl;
            #endif

                // data driven selection: select only any of the rectangle which can be split
                std::vector<int> metaState;
                for(int i = 0; i<state.size(); i++)
                {
                    if(partHypotheses[state[i]].projProf.size()>0)
                        metaState.push_back(i);
                }

                if(metaState.size() > 0)
                {
                    // elect the rectangle to be split in the current state by weighted sampling
                    std::uniform_int_distribution<int> stateDist(0, static_cast<int>(metaState.size() - 1));
                    // TO DO: there should be a minimum width constraint to split
                    metaSplitPart = stateDist(g);
                    splitPart = metaState[metaSplitPart];

                    // the part to be split is selected
                    originalPartB4Split = partHypotheses[state[splitPart]];
                #if DEBUG_MODE_ON
                    std::cout<<partHypotheses[state[splitPart]].projProf.size()<<" splits possible"<<std::endl;
                #endif

                    //Split the selected rectangle into two new rectangles
                    split1Rectangle(state, splitPart, originalPartB4Split, partHypotheses, splitPartR1, splitPartR2);

                    // Update the matrix entries corresponding to the newly formed rectangles
                    originalOverlapPairsSplit = overlapPairs;
                    //originalOverlapPairs70Split = overlapPairs70;
                    originalOverlapAreaSplit = overlapArea;
                #if DEBUG_MODE_ON
                    std::cout<<"Size of overlap pair binary matrix before split "<<overlapPairs.rows()<<" x "<<overlapPairs.cols()<<std::endl;
                    std::cout<<"Size of overlap Area float matrix before split "<<overlapArea.rows()<<" x "<<overlapArea.cols()<<std::endl;
                #endif
                    updateMatricesSplit(partHypotheses, areas, splitPartR1, splitPartR2, imageArea, overlapPairs, overlapArea);
                #if DEBUG_MODE_ON
                    std::cout<<"Size of overlap pair binary matrix after split "<<overlapPairs.rows()<<" x "<<overlapPairs.cols()<<std::endl;
                    std::cout<<"Size of overlap Area float matrix after split "<<overlapArea.rows()<<" x "<<overlapArea.cols()<<std::endl;
                #endif

                #if DEBUG_MODE_ON
                    std::cout<<"Size of state before split "<<newState.size()<<std::endl;
                #endif

                    //Update the state: erase old rectangle and add two new rectangles
                    newState.erase(newState.begin() + splitPart);// or splitPart-1 confirm?
                    newState.push_back(partHypotheses.size()-1);
                    newState.push_back(partHypotheses.size()-2);

                #if DEBUG_MODE_ON
                    std::cout<<"Size of state after split "<<newState.size()<<std::endl;
                #endif
                    // TO DO: Update the Edge Proj Profile of the new rectangles
                    // TO DO: Update Labels: shape prior and appearance
                }
                else
                {
#if DEBUG_MODE_ON
                    std::cout<<"split move not possible with the selected rectangle"<<std::endl;
#endif
                }

                moveProbfactor =  0.0f;
                totalSplitCount++;
            }
            else if(randomMove == MERGE_MOVE_IDX)//MERGE MOVE
            {

            #if DEBUG_MODE_ON
                std::cout<<"merge move "<<std::endl;
            #endif

                if(newState.size() > 1)
                {
                    int rectIdx1,rectIdx2;
                    computeMergeability(state, partHypotheses, rectIdx1, rectIdx2);
                #if DEBUG_MODE_ON
                    std::cout<<"rect 1 Idx: "<<rectIdx1<<" out of : "<<newState.size()<<std::endl;
                    std::cout<<"rect 2 Idx: "<<rectIdx2<<" out of : "<<newState.size()<<std::endl;
                #endif

                    Part mergedPart;
                    merge2Rectangles(state, partHypotheses, rectIdx1, rectIdx2, mergedPart);
                    Rectangle mergedRect = mergedPart.rect;
                    float areaR = mergedRect.getArea();
                    areaR /= imageArea;
                    areas.push_back(areaR);

                    //update the overlap matrices
                    originalOverlapPairsMerge = overlapPairs;
                    //originalOverlapPairs70Merge = overlapPairs70;
                    originalOverlapAreaMerge = overlapArea;

                #if DEBUG_MODE_ON
                    std::cout<<"Size of overlap pair binary matrix before split "<<overlapPairs.rows()<<" x "<<overlapPairs.cols()<<std::endl;
                    std::cout<<"Size of overlap Area float matrix before split "<<overlapArea.rows()<<" x "<<overlapArea.cols()<<std::endl;
                #endif

                    updateMatricesMerge(partHypotheses, mergedRect, overlapPairs, overlapArea);

                    newState.push_back(partHypotheses.size()-1);
                    newState.erase(newState.begin() + rectIdx1);
                    newState.erase(newState.begin() + rectIdx2 - 1);// the location decreases by one with previos erase
                    // TO DO: Add merged Rect to the part hypothesis
                    // TO DO: Add merged Rect to the new state and delete the old two
                    //TO DO : update all 3 labels
                    //TO DO : update overlap matrix binary, overlap 70 binary, absolute overlap
                    //TO DO : Area matix will also change
                    //TO DO : update Appearance Likelihood
                    //TO DO : update required for shape prior
                    //TO DO : Recalculate rectangle weight for each class
                }
                else
                {
                    std::cout<<"No merge move possible..state only has a single rectangle"<<std::endl;
                }

                moveProbfactor =  0.0f;
                totalMergeCount++;
            }
#if 0
            else if(randomMove == SPLIT_MOVE_IDX)//SPLIT MOVE
            {
            #if DEBUG_MODE_ON
                std::cout<<"split move "<<std::endl;
            #endif
                moveProbfactor = std::log((double)moveProbabilities[MERGE_MOVE_IDX]/moveProbabilities[SPLIT_MOVE_IDX]);//Move probability ratio
                totalSplitCount++;
            }
            else if(randomMove == MERGE_MOVE_IDX)//MERGE MOVE
            {
            #if DEBUG_MODE_ON
                std::cout<<"merge move "<<std::endl;
            #endif
                moveProbfactor = std::log((double)moveProbabilities[SPLIT_MOVE_IDX]/moveProbabilities[MERGE_MOVE_IDX]);//Move probability ratio
                totalMergeCount++;
            }
#endif
            else if(randomMove == LABEL_DIFFUSE_MOVE_IDX)//EXCHANGE MOVE
            {
            #if DEBUG_MODE_ON
                std::cout<<"label diffuse move "<<std::endl;
            #endif
                moveProbfactor = 0.0f;
                totalLabelDiffuseCount++;
            }
            else if(randomMove == EXCHANGE_DD_MOVE_IDX)//EXCHANGE MOVE
            {
            #if DEBUG_MODE_ON
                std::cout<<"DD exchange move "<<std::endl;
            #endif
                moveProbfactor = 0.0f;
                totalExchangeDDCount++;
            }
            else if(randomMove == UPDATE_CENTER_MOVE_IDX)//EXCHANGE MOVE
            {
                // Set up a distribution over the current state
                std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));
                // Choose a tree to replace
                updateCenterPart = stateDist(g);
            #if DEBUG_MODE_ON
                std::cout<<"update center move "<<std::endl;
            #endif

                originalCenterRect = partHypotheses[state[updateCenterPart]].rect;
                Rectangle modifiedCenterRect;
                diffuseCenterLoc(originalCenterRect,modifiedCenterRect);

#if 0
                updateAppearanceLikelihood();
#endif
                //TO DO : update all 3 labels
                partHypotheses[state[updateCenterPart]].rect = modifiedCenterRect;
                //update the overlap matrices
                originalOverlapPairsCenter = overlapPairs;
                //originalOverlapPairs70Center = overlapPairs70;
                originalOverlapAreaCenter = overlapArea;
                updateMatricesCenterLocDiffuse(state,
                                               partHypotheses,
                                               modifiedCenterRect,
                                               updateCenterPart,
                                               overlapPairs,
                                               overlapArea);

                //TO DO : update Appearance Likelihood: ignore
                //no update required for shape prior
                //TO DO : Recalculate rectangle weight : ignore
                moveProbfactor = 0.0f;
                totalUpdateCenterCount++;
            }
            else if(randomMove == UPDATE_WIDTH_MOVE_IDX)//EXCHANGE MOVE
            {

                // Set up a distribution over the current state
                std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));
                // Choose a tree to replace
                updateWidthPart = stateDist(g);
            #if DEBUG_MODE_ON
                std::cout<<"update width move "<<std::endl;
            #endif
                originalWidthRect = partHypotheses[state[updateWidthPart]].rect;
                Rectangle modifiedWidthRect;
                diffuseRectWidth(originalWidthRect, modifiedWidthRect);

                partHypotheses[state[updateWidthPart]].rect = modifiedWidthRect;
                //TO DO : update all 3 labels

                //update the overlap matrices
                originalOverlapPairsWidth = overlapPairs;
                //originalOverlapPairs70Width = overlapPairs70;
                originalOverlapAreaWidth = overlapArea;

                updateMatricesWidthDiffuse(state,
                                               partHypotheses,
                                               areas,
                                               modifiedWidthRect,
                                               updateWidthPart,
                                               overlapPairs,
                                               overlapArea,
                                               imageArea
                                           );

                //TO DO : update Appearance Likelihood: ignore
                //TO DO : update shape prior : ignore
                //TO DO : Recalculate rectangle weight : ignore
                moveProbfactor = 0.0f;
                totalUpdateWidthCount++;
            }
            else if(randomMove == UPDATE_HEIGHT_MOVE_IDX)//Height UPDATE MOVE
            {
                // Set up a distribution over the current state
                std::uniform_int_distribution<int> stateDist(0, static_cast<int>(state.size() - 1));

                // Choose a tree to replace
                updateHeightPart = stateDist(g);

            #if DEBUG_MODE_ON
                std::cout<<"update height move "<<std::endl;
            #endif
                //std::cout<<"Normal update Part is : "<<updateHeightPart<<" out of :"<<proposals.size()<<std::endl;

                originalHeightRect = partHypotheses[state[updateHeightPart]].rect;
                Rectangle modifiedHeightRect;
                diffuseRectHeight(originalHeightRect, modifiedHeightRect);

                partHypotheses[state[updateHeightPart]].rect = modifiedHeightRect;
                //TO DO : update all 3 labels

                //update the overlap matrices
                originalOverlapPairsHeight = overlapPairs;
                //originalOverlapPairs70Height = overlapPairs70;
                originalOverlapAreaHeight = overlapArea;

                updateMatricesHeightDiffuse(state,
                                               partHypotheses,
                                               areas,
                                               modifiedHeightRect,
                                               updateHeightPart,
                                               overlapPairs,
                                               overlapArea,
                                               imageArea);

                moveProbfactor = 0.0f;
                totalUpdateHeightCount++;
            }
            else
            {
                std::cout<<"invalid move detected.. quitting"<<std::endl;
                break;
            }
            // Calculate the new error and the improvement
            const float newError = energyFunction.energy(newState, moveProbabilities, areas, overlapPairs, overlapArea, partHypotheses);

            //Check energy gradient
            posteriorFactor = newError - currentEnergy;
            //posterior ratio
            logAcceptRatio += (posteriorFactor/temperature);
            //move probability ratio
            logAcceptRatio += moveProbfactor;

            // What to do?
            if (logAcceptRatio <= 0)
            {
                // We improved the energy, accept this step
                currentEnergy = newError;
                state = newState;
#if DEBUG_MODE_ON
                plotMarkovChainState(state, partHypotheses);
                std::cout <<"blind accept temperature: "<<temperature<<" energy: "<<newError<< "\n";
#endif

#if 0
                float areaCoverage = energyFunction.coverArea(state);
                if(areaCoverage < 0.0f)
                {
                    std::cout<<"Area coverage is: "<<areaCoverage<<std::endl;

                }
#endif

                if(randomMove == EXCHANGE_MOVE_IDX)
                {
                    exchangeCount++;
                }
                else if(randomMove == BIRTH_MOVE_IDX)
                {
                    birthCount++;
                }
                else if(randomMove == DEATH_MOVE_IDX)
                {
                    deathCount++;
                }
                else if(randomMove == SPLIT_MOVE_IDX)
                {
                    splitCount++;
                    //TO DO: update proposal pool, add 2 new, delete old
                    //the proposal pool, index of the 3 rects must be available here
                }
                else if(randomMove == MERGE_MOVE_IDX)
                {
                    mergeCount++;
                    //TO DO: update proposal pool, add 1 new, delete 2 old
                    //the proposal pool, index of the 3 rects must be available here
                }
                else if(randomMove == LABEL_DIFFUSE_MOVE_IDX)
                {
                    labelDiffuseCount++;
                }
                else if(randomMove == EXCHANGE_DD_MOVE_IDX)
                {
                    exchangeDDCount++;
                }
                else if(randomMove == UPDATE_CENTER_MOVE_IDX)
                {
                    updateCenterCount++;
                }
                else if(randomMove == UPDATE_WIDTH_MOVE_IDX)
                {
                    updateWidthCount++;
                }
                else if(randomMove == UPDATE_HEIGHT_MOVE_IDX)
                {
                    updateHeightCount++;
                }
                else
                {
                    std::cout<<"invalid state accepted.. quitting"<<std::endl;
                    break;
                }
            }
            else
            {
                // We did not improve. Accept the step with a certain
                // probability
                const float u = uniformDist(g);
                if (std::log(u) <= -logAcceptRatio)
                {
                    currentEnergy = newError;
                    state = newState;

#if DEBUG_MODE_ON
                    plotMarkovChainState(state, partHypotheses);
                    std::cout <<"conditional accept temperature: "<<temperature<<" energy: "<<newError<< "\n";
#endif

#if 0
                    float areaCoverage = energyFunction.coverArea(state);
                    if(areaCoverage < 0.0f)
                    {
                        std::cout<<"Area coverage is: "<<areaCoverage<<std::endl;

                    }
#endif

                    if(randomMove == EXCHANGE_MOVE_IDX)
                    {
                        specialExchangeCount++;
                    }
                    else if(randomMove == BIRTH_MOVE_IDX)
                    {
                        specialBirthCount++;
                    }
                    else if(randomMove == DEATH_MOVE_IDX)
                    {
                        specialDeathCount++;
                    }
                    else if(randomMove == SPLIT_MOVE_IDX)
                    {
                        specialSplitCount++;
                        //TO DO: update proposal pool, add 2 new, delete old
                        //the proposal pool, index of the 3 rects must be available here
                    }
                    else if(randomMove == MERGE_MOVE_IDX)
                    {
                        specialMergeCount++;
                        //TO DO: update proposal pool, add 1 new, delete 2 old
                        //the proposal pool, index of the 3 rects must be available here
                    }
                    else if(randomMove == LABEL_DIFFUSE_MOVE_IDX)
                    {
                        specialLabelDiffuseCount++;
                    }
                    else if(randomMove == EXCHANGE_DD_MOVE_IDX)
                    {
                        specialExchangeDDCount++;
                    }
                    else if(randomMove == UPDATE_CENTER_MOVE_IDX)
                    {
                        specialUpdateCenterCount++;
                    }
                    else if(randomMove == UPDATE_WIDTH_MOVE_IDX)
                    {
                        specialUpdateWidthCount++;
                    }
                    else if(randomMove == UPDATE_HEIGHT_MOVE_IDX)
                    {
                        specialUpdateHeightCount++;
                    }
                    else
                    {
                        std::cout<<"invalid state accepted.. quitting"<<std::endl;
                        break;
                    }
                }
                else
                {

#if DEBUG_MODE_ON
                plotMarkovChainState(state, partHypotheses);
                std::cout <<"reject temperature: "<<temperature<<" energy: "<<newError<< "\n";
#endif
                    //Cancell all move specific updations
                    if(randomMove == SPLIT_MOVE_IDX)
                    {
                        //partHypotheses[state[splitPart]] = originalPartB4Split;
                        partHypotheses.pop_back();
                        partHypotheses.pop_back();
                        //areas[state[splitPart]] = originalPartB4Split.getArea()/image.rows()/image.cols();
                        areas.pop_back();
                        areas.pop_back();
                        overlapPairs = originalOverlapPairsSplit;
                        //overlapPairs70 = originalOverlapPairs70Split;
                        overlapArea = originalOverlapAreaSplit;

                    }
                    else if(randomMove == MERGE_MOVE_IDX)
                    {
                        partHypotheses.pop_back();
                        areas.pop_back();
                        overlapPairs = originalOverlapPairsMerge;
                        //overlapPairs70 = originalOverlapPairs70Merge;
                        overlapArea = originalOverlapAreaMerge;
                        //partHypotheses[state[updateCenterPart]].rect = originalCenterRect;
                        //std::cout<<"Rejected update Part is : "<<replacePart<<" out of :"<<proposals.size()<<std::endl;
                    }
                    else if(randomMove == UPDATE_CENTER_MOVE_IDX)
                    {
                        partHypotheses[state[updateCenterPart]].rect = originalCenterRect;
                        overlapPairs = originalOverlapPairsCenter;
                        //overlapPairs70 = originalOverlapPairs70Center;
                        overlapArea = originalOverlapAreaCenter;
                        //std::cout<<"Rejected update Part is : "<<replacePart<<" out of :"<<proposals.size()<<std::endl;
                    }
                    else if(randomMove == UPDATE_WIDTH_MOVE_IDX)
                    {
                        partHypotheses[state[updateWidthPart]].rect = originalWidthRect;
                        areas[state[updateWidthPart]] = originalWidthRect.getArea();
                        areas[state[updateWidthPart]] /= imageArea;
                        switch(partHypotheses[state[updateWidthPart]].label)
                        {
                            case 0:
                                areas[state[updateWidthPart] + static_cast<int>(1)] = areas[state[updateWidthPart]];
                                areas[state[updateWidthPart] + static_cast<int>(2)] = areas[state[updateWidthPart]];
                                break;
                            case 1:
                                areas[state[updateWidthPart] - static_cast<int>(1)] = areas[state[updateWidthPart]];
                                areas[state[updateWidthPart] + static_cast<int>(1)] = areas[state[updateWidthPart]];
                                break;
                            case 2:
                                areas[state[updateWidthPart] - static_cast<int>(1)] = areas[state[updateWidthPart]];
                                areas[state[updateWidthPart] - static_cast<int>(2)] = areas[state[updateWidthPart]];
                                break;
                        }

                        overlapPairs = originalOverlapPairsWidth;
                        //overlapPairs70 = originalOverlapPairs70Width;
                        overlapArea = originalOverlapAreaWidth;
                    }
                    else if(randomMove == UPDATE_HEIGHT_MOVE_IDX)
                    {
                        partHypotheses[state[updateHeightPart]].rect = originalHeightRect;
                        areas[state[updateHeightPart]] = originalHeightRect.getArea();
                        areas[state[updateHeightPart]] /= imageArea;
                        switch(partHypotheses[state[updateHeightPart]].label)
                        {
                            case 0:
                                areas[state[updateHeightPart] + static_cast<int>(1)] = areas[state[updateHeightPart]];
                                areas[state[updateHeightPart] + static_cast<int>(2)] = areas[state[updateHeightPart]];
                                break;
                            case 1:
                                areas[state[updateHeightPart] - static_cast<int>(1)] = areas[state[updateHeightPart]];
                                areas[state[updateHeightPart] + static_cast<int>(1)] = areas[state[updateHeightPart]];
                                break;
                            case 2:
                                areas[state[updateHeightPart] - static_cast<int>(1)] = areas[state[updateHeightPart]];
                                areas[state[updateHeightPart] - static_cast<int>(2)] = areas[state[updateHeightPart]];
                                break;
                        }
#if 0
                        std::cout<<"Reject ROI Area : "<<imageArea<<std::endl;
                        std::cout<<"Reject Rect area : "<<originalHeightRect.getArea()<<std::endl;
                        std::cout<<"Reject Rect area/ ROI area : "<<originalHeightRect.getArea()/imageArea<<std::endl;
#endif
                        overlapPairs = originalOverlapPairsHeight;
                        //overlapPairs70 = originalOverlapPairs70Height;
                        overlapArea = originalOverlapAreaHeight;
                    }
                }
            }

            if (currentEnergy < bestEnergy)
            {
                bestEnergy = currentEnergy;
                bestState = state;
                update = true;
            }
        }
        if (!update)
        {
            noUpdateIterations++;
        }
        else
        {
            noUpdateIterations = 0;
        }

        // Call the callback functions
#if 1
        int result = 0;
        for (size_t i = 0; i < callbacks.size(); i++)
        {
            result = std::min(result, callbacks[i]->callback(state, currentEnergy, bestState, bestEnergy, iteration, temperature));
        }

        if (result < 0)
        {
            break;
        }

#endif
        if (noUpdateIterations >= maxNoUpdateIterations)
        {
            break;
        }
    }

    /*
     * Move statistics: for debug purpose
    */

    std::cout<<"No of effective best state no updates: "<<noUpdateIterations<<std::endl;
    std::cout<<"Convergence temperature: "<<temperature<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Optimization complete"<<std::endl<<std::endl;
    std::cout<<std::endl;

    std::cout<<"RJMCMC Move Statistics:"<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Exchange: "<<exchangeCount + specialExchangeCount<<" ("<<specialExchangeCount<<") accepted out of "<<totalExchangeCount<<std::endl;
    std::cout<<"Exchange Acceptance Rate: "<<100*(double)(exchangeCount + specialExchangeCount)/totalExchangeCount<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Birth: "<<birthCount + specialBirthCount<<" ("<<specialBirthCount<<") accepted out of "<<totalBirthCount<<std::endl;
    std::cout<<"Birth Acceptance Rate: "<<100*(double)(birthCount + specialBirthCount)/totalBirthCount<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Death: "<<deathCount + specialDeathCount<<" ("<<specialDeathCount<<") accepted out of "<<totalDeathCount<<std::endl;
    std::cout<<"Death Acceptance Rate: "<<100*(double)(deathCount + specialDeathCount)/totalDeathCount<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Split: "<<splitCount + specialSplitCount<<" ("<<specialSplitCount<<") accepted out of "<<totalSplitCount<<std::endl;
    std::cout<<"Split Acceptance Rate: "<<100*(double)(splitCount + specialSplitCount)/totalSplitCount<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Merge: "<<mergeCount + specialMergeCount<<" ("<<specialMergeCount<<") accepted out of "<<totalMergeCount<<std::endl;
    std::cout<<"Merge Acceptance Rate: "<<100*(double)(mergeCount + specialMergeCount)/totalMergeCount<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Label Diffuse: "<<labelDiffuseCount + specialLabelDiffuseCount<<" ("<<specialLabelDiffuseCount<<") accepted out of "<<totalLabelDiffuseCount<<std::endl;
    std::cout<<"Label Diffuse Acceptance Rate: "<<100*(double)(labelDiffuseCount + specialLabelDiffuseCount)/totalLabelDiffuseCount<<std::endl;
    std::cout<<std::endl;

    std::cout<<"DD Exchange: "<<exchangeDDCount + specialExchangeDDCount<<" ("<<specialExchangeDDCount<<") accepted out of "<<totalExchangeDDCount<<std::endl;
    std::cout<<"DD E Acceptance Rate: "<<100*(double)(exchangeDDCount + specialExchangeDDCount)/totalExchangeDDCount<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Center Loc Update (Diffuse): "<<updateCenterCount + specialUpdateCenterCount<<" ("<<specialUpdateCenterCount<<") accepted out of "<<totalUpdateCenterCount<<std::endl;
    std::cout<<"Center Loc Update (Diffuse) Acceptance Rate: "<<100*(double)(updateCenterCount + specialUpdateCenterCount)/totalUpdateCenterCount<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Width Update (Diffuse): "<<updateWidthCount + specialUpdateWidthCount<<" ("<<specialUpdateWidthCount<<") accepted out of "<<totalUpdateWidthCount<<std::endl;
    std::cout<<"Width Update (Diffuse) Acceptance Rate: "<<100*(double)(updateWidthCount + specialUpdateWidthCount)/totalUpdateWidthCount<<std::endl;
    std::cout<<std::endl;

    std::cout<<"Height Update (Diffuse): "<<updateHeightCount + specialUpdateHeightCount<<" ("<<specialUpdateHeightCount<<") accepted out of "<<totalUpdateHeightCount<<std::endl;
    std::cout<<"Height Update (Diffuse) Acceptance Rate: "<<100*(double)(updateHeightCount + specialUpdateHeightCount)/totalUpdateHeightCount<<std::endl;
    std::cout<<std::endl;

    state = bestState;
    return bestEnergy;
}
