#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

using namespace libf;

/**
 * Command line tool for visualizing the pixel importance for datasets where
 * images are given as raw input (e.g. USPS, MNIST).
 * 
 * Usage:
 * $ ./examples/cli_pixel_importance --help
 * Allowed options:
 *   --help                                produce help message
 *   --file-train arg                      path to train DAT file
 *   --file-test arg                       path to test DAT file
 *   --output-image arg (=pixel_importance.png)
 *                                         output image visualizing pixel 
 *   --output-width arg (=10)              width of output image (assumed to
 *                                         be square)
 *                                         importance
 *   --num-features arg (=10)              number of features to use (set to 
 *                                         dimensionality of data to learn 
 *                                         deterministically)
 *   --use-bootstrap                       use bootstrapping for training
 *   --num-trees arg (=100)                number of trees in forest
 *   --max-depth arg (=100)                maximum depth of trees
 *   --num-threads arg (=1)                number of threads for learning
 */
int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("file-train", boost::program_options::value<std::string>(), "path to train DAT file")
        ("file-test", boost::program_options::value<std::string>(), "path to test DAT file")
        ("output-image", boost::program_options::value<std::string>()->default_value("pixel_importance.png"), "output image visualizing pixel importance")
        ("output-width", boost::program_options::value<int>()->default_value(16), "width of output image (assumed to be square)")
        ("num-features", boost::program_options::value<int>()->default_value(10), "number of features to use (set to dimensionality of data to learn deterministically)")
        ("use-bootstrap", "use bootstrapping for training")
        ("num-trees", boost::program_options::value<int>()->default_value(100), "number of trees in forest")
        ("max-depth", boost::program_options::value<int>()->default_value(100), "maximum depth of trees")
        ("num-threads", boost::program_options::value<int>()->default_value(1), "number of threads for learning");

    boost::program_options::positional_options_description positionals;
    positionals.add("file-train", 1);
    positionals.add("file-test", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end())
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    boost::filesystem::path mnistTrainDat(parameters["file-train"].as<std::string>());
    if (!boost::filesystem::is_regular_file(mnistTrainDat))
    {
        std::cout << "Train DAT file does not exist at the specified location." << std::endl;
        return 1;
    }
    
    boost::filesystem::path mnistTestDat(parameters["file-test"].as<std::string>());
    if (!boost::filesystem::is_regular_file(mnistTestDat))
    {
        std::cout << "Test DAT file does not exist at the specified location." << std::endl;
        return 1;
    }
    
    bool useBootstrap = parameters.find("use-bootstrap") != parameters.end();
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataStorage::ptr storageT = DataStorage::Factory::create();
    
    LibforestDataReader reader;
    reader.read(mnistTrainDat.string(), storageT);
    reader.read(mnistTestDat.string(), storage);
    
    // Important for sorted datasets!
    storageT->randPermute();
    
    std::cout << "Training Data" << std::endl;
    storageT->dumpInformation();
    
    RandomForestLearner forestLearner;
    forestLearner.addCallback(RandomForestLearner::defaultCallback, 1);
    
    forestLearner.getTreeLearner().setUseBootstrap(useBootstrap);
    forestLearner.getTreeLearner().setMaxDepth(parameters["max-depth"].as<int>());
    forestLearner.getTreeLearner().setNumFeatures(parameters["num-features"].as<int>());
    
    forestLearner.setNumTrees(parameters["num-trees"].as<int>());
    forestLearner.setNumThreads(parameters["num-threads"].as<int>());
    
    RandomForest::ptr forest = forestLearner.learn(storage);
    
    AccuracyTool accuracyTool;
    accuracyTool.measureAndPrint(forest, storageT);
    
    ConfusionMatrixTool confusionMatrixTool;
    confusionMatrixTool.measureAndPrint(forest, storageT);
    
    // USPS images are 16 x 16, MNIST are 28 x 28.
    PixelImportanceTool piTool;
    piTool.measureAndPrint(&forestLearner);
    piTool.measureAndSave(&forestLearner, boost::filesystem::path(parameters["output-image"].as<std::string>()), parameters["output-width"].as<int>());
    
    return 0;
}
