#include "libforest/util.h"
#include <random>
#include "ncurses.h"

static std::random_device rd;

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// Util
////////////////////////////////////////////////////////////////////////////////

void Util::generateRandomPermutation(int N, std::vector<int> & sigma)
{
    // Set up the initial state
    sigma.resize(N);
    for (int n = 0; n < N; n++)
    {
        sigma[n] = n;
    }
    
    // Randomize the permutation
    std::shuffle(sigma.begin(), sigma.end(), std::default_random_engine(rd()));
}

////////////////////////////////////////////////////////////////////////////////
/// GUIUtil
////////////////////////////////////////////////////////////////////////////////

void GUIUtil::printProgressBar(float percentage)
{
    // Determine the width of the window
    int mrow, mcol;
    getmaxyx(stdscr, mrow, mcol);
    
    // Calculate the number of progress bar segments
    // 8 = 2 spacers, 4 characters for the percentage, 1 blank space, 1 line feed
    int progressBarWidth = mcol - 8;
    
    printw("[");
    
    for (int k = 0; k < progressBarWidth; k++)
    {
        const float p = k/static_cast<float>(progressBarWidth);
        if (p <= percentage)
        {
            printw("=");
        }
        else
        {
            printw(" ");
        }
    }
    
    printw("] %3.0f%%\n", percentage * 100);
}