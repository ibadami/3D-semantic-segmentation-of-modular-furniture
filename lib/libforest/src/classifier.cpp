#include "libforest/classifier.h"
#include "libforest/data.h"
#include "libforest/io.h"
#include "libforest/util.h"
#include <ios>
#include <iostream>
#include <string>
#include <cmath>

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// Classifier
////////////////////////////////////////////////////////////////////////////////

void AbstractClassifier::classify(AbstractDataStorage::ptr storage, std::vector<int> & results) const
{
    // Clean the result set
    results.resize(storage->getSize());
    
    // Classify each individual data point
    for (int i = 0; i < storage->getSize(); i++)
    {
        results[i] = classify(storage->getDataPoint(i));
    }
}

int AbstractClassifier::classify(const DataPoint & x) const
{
    // Get the class posterior
    std::vector<float> posterior;
    classLogPosterior(x, posterior);
    
    return Util::argMax(posterior);
}
