#include <random>

#include "libforest/learning_tools.h"

using namespace libf;

static std::default_random_engine rd;
static std::mt19937 g(rd());

////////////////////////////////////////////////////////////////////////////////
/// RandomThresholdGenerator
////////////////////////////////////////////////////////////////////////////////

RandomThresholdGenerator::RandomThresholdGenerator(AbstractDataStorage::ptr storage)
{
    const int D = storage->getDimensionality();
    const int N = storage->getSize();
    
    min = std::vector<float>(D, 1e35f);
    max = std::vector<float>(D, -1e35f);
    
    for (int n = 0; n < N; ++n)
    {
        // Retrieve the datapoint to check all features.
        const DataPoint & x = storage->getDataPoint(n);
        
        for (int d = 0; d < D; d++)
        {
            if (x(d) < min[d])
            {
                min[d] = x(d);
            }
            if (x(d) > max[d])
            {
                max[d] = x(d);
            }
        }
    }    
}

float RandomThresholdGenerator::sample(int feature)
{
    // assert(feature >= 0 && feature < getSize());
    std::uniform_real_distribution<float> dist(min[feature], max[feature]);
    
    return dist(g);
}
