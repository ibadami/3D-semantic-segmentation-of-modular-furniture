/**
 * This file contains a very simple command line interface. 
 */

#include <iostream>
#include <chrono>

#include "parser/parser.h"

/**
 * Trains the pipeline on the data in the specified directory.
 */
int train(int argc, const char** argv);

/**
 * Parses a single image
 */
int parse(int argc, const char** argv);

/**
 * Tests individual components
 */
int test(int argc, const char** argv);

/**
 * Exports some visualizations
 */
int exportVisualization(int argc, const char** argv);

/**
 * Generates the edge map data. 
 */
int createGeneralEdgeDetectorSet(int argc, const char** argv);

/**
 * Exports the rectified images
 */
int exportRectifiedImages(int argc, const char** argv);

/**
 * Visualizes the parse graph
 */ 
int visualizeParseGraph(int argc, const char** argv);

/**
 * Exports the discretized appearance descriptors
 */
int exportAppearanceDescriptors(int argc, const char** argv);

int main(int argc, const char** argv)
{
    // There must be at least one argument
    if (argc < 2)
    {
        std::cout << "Please specify a function: evaluate, train, parse." << std::endl;
        return 1;
    }
    
    std::string function(argv[1]);
    
    // Run the specified command
    if (function == "evaluate")
    {
        std::cout << "Not implemented." << std::endl;
        return 2;
    }
    else if (function == "train")
    {
        return train(argc, argv);
    }
    else if (function == "parse")
    {
        return parse(argc, argv);
    }
    else if (function == "test")
    {
        return test(argc, argv);
    }
    else if (function == "createGeneralEdgeDetectorSet")
    {
        return createGeneralEdgeDetectorSet(argc, argv);
    }
    else if (function == "exportVisualization")
    {
        return exportVisualization(argc, argv);
    }
    else if (function == "exportRectifiedImages")
    {
        return exportRectifiedImages(argc, argv);
    }
    else if (function == "visualizeParseGraph")
    {
        return visualizeParseGraph(argc, argv);
    }
    else if (function == "exportAppearanceDescriptors")
    {
        return exportAppearanceDescriptors(argc, argv);
    }
    else
    {
        std::cout << "Unknown function." << std::endl;
        return 3;
    }
    return 0;
}

int train(int argc, const char** argv)
{
    // There must be a directory
    if (argc != 3)
    {
        std::cout << "Please specify a directory: $ bin train [directory]" << std::endl;
        return 1;
    }
    
    std::string directory(argv[2]);
    
    parser::CabinetParser parser;
    parser.train(directory);
    
    return 0;
}

int test(int argc, const char** argv)
{
    // There must be a directory
    if (argc != 3)
    {
        std::cout << "Please specify a directory: $ bin test [directory]" << std::endl;
        return 1;
    }
    
    std::string directory(argv[2]);
    
    parser::CabinetParser parser;
    parser.test(directory);
    
    return 0;
}

#include "parser/energy.h"

int parse(int argc, const char** argv)
{
    // You have to specify a directory and a number
    if (argc != 4)
    {
        std::cout << "Please specify a directory and an image number: $ bin train [directory] [number]" << std::endl;
        return 1;
    }
    
    std::string directory(argv[2]);
    std::string number(argv[3]);
    
    // Load the image
    std::string imageFile = directory + number + ".JPG";
    cv::Mat image = cv::imread(imageFile, 1);
    
    std::string imageDFile = directory + number + "_depth.png";
    cv::Mat imageD = cv::imread(imageDFile, 1);

    // Load the auxilary file
    std::string auxFile = directory + number + "_annotation.json";
    parser::Segmentation segmentation;
    segmentation.readAnnotationFile(auxFile);
    
    parser::CabinetParser parser;
    
    // Parse the image
    std::vector<parser::Part> result;
    parser.parse(image, imageD, segmentation.regionOfInterest, result);
    
    cv::Mat demo;
    parser.visualizeSegmentation(image, segmentation.regionOfInterest, result, demo);
    cv::imwrite("last_result.png", demo);
    parser::Util::imshow(demo);

    return 0;
}

int createGeneralEdgeDetectorSet(int argc, const char** argv)
{
    // You have to specify a directory and a number
    if (argc != 5)
    {
        std::cout << "Please specify an input and an output directory: $ bin createGeneralEdgeDetectorSet [input directory] [image directory] [groundtruth directory]" << std::endl;
        return 1;
    }
    
    std::string inputDirectory(argv[2]);
    std::string imageDirectory(argv[3]);
    std::string groundtruthDirectory(argv[4]);
    
    parser::CabinetParser parser;
    parser.createGeneralEdgeDetectorSet(inputDirectory, imageDirectory, groundtruthDirectory);
    
    return 0;
}

int exportVisualization(int argc, const char** argv)
{
    // You have to specify a directory and a number
    if (argc != 4)
    {
        std::cout << "Please specify a directory and an image number: $ bin exportVisualization [directory] [number]" << std::endl;
        return 1;
    }
    
    std::string directory(argv[2]);
    std::string number(argv[3]);
    
    // Load the image
    std::string imageFile = directory + number + ".JPG";
    cv::Mat image = cv::imread(imageFile, 1);
    
    // Load the auxilary file
    std::string auxFile = directory + number + "_annotation.json";
    parser::Segmentation segmentation;
    segmentation.readAnnotationFile(auxFile);
    
    parser::CabinetParser parser;
    parser.exportEdgeDistributionFromGT(image, segmentation);
    return 0;
}

int exportRectifiedImages(int argc, const char** argv)
{
    // You have to specify an input directory
    if (argc != 3)
    {
        std::cout << "Please specify a directory: $ bin exportRectifiedImages [directory]" << std::endl;
        return 1;
    }
    
    std::string directory(argv[2]);
    
    parser::CabinetParser parser;
    parser.exportRectifiedImages(directory);
    
    return 0;
}

int visualizeParseGraph(int argc, const char** argv)
{
    // You have to specify a directory and a number
    if (argc != 4)
    {
        std::cout << "Please specify a directory and an image number: $ bin visualizeParseGraph [directory] [number]" << std::endl;
        return 1;
    }
    
    std::string directory(argv[2]);
    std::string number(argv[3]);
    
    // Load the image
    std::string imageFile = directory + number + ".JPG";
    cv::Mat image = cv::imread(imageFile, 1);
    
    // Load the auxilary file
    std::string auxFile = directory + number + "_annotation.json";
    parser::Segmentation segmentation;
    segmentation.readAnnotationFile(auxFile);
    
    parser::CabinetParser parser;
    std::vector<parser::Rectangle> rectified;
    parser.rectifyParts(segmentation.regionOfInterest, segmentation.parts, rectified);
    std::vector<parser::ParseTreeNode*> nodes(rectified.size());
    for (size_t n = 0; n < rectified.size(); n++)
    {
        nodes[n] = new parser::ParseTreeNode();
        nodes[n]->rect = rectified[n];
    }

    parser::ParserEnergy parserEnergy;
    parser::ParseTreeNode* tree;
    try {
        auto start = std::chrono::high_resolution_clock::now();
        tree = parserEnergy.parse(nodes);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
        std::cout << duration.count() << "ms\n";
        parserEnergy.visualize(tree, image, segmentation);
    } catch(...) {
        return 1;
    }

    return 0;
}

int exportAppearanceDescriptors(int argc, const char** argv)
{
    // You have to specify a directory and an output filename
    if (argc != 4)
    {
        std::cout << "Please specify a directory and an output filename: $ bin exportAppearanceDescriptors [directory] [out file]" << std::endl;
        return 1;
    }
    
    std::string directory(argv[2]);
    std::string outFile(argv[3]);
    
    parser::CabinetParser parser;
    
    // Load the directory content
    std::vector< std::tuple<cv::Mat, parser::Segmentation, cv::Mat> > trainingData;
    parser.loadImage(directory, trainingData);
    
    // Get the training set
    libf::DataStorage::ptr trainingSet = libf::DataStorage::Factory::create();
    parser.extractDiscretizedAppearanceDistributions(trainingSet, trainingData);
    
    // Save the result
    std::ofstream os(outFile);
    if (!os.is_open())
    {
        std::cout << "Cannot open file." << std::endl;
        return 1;
    }
    
    libf::CSVDataWriter writer;
    writer.write(os, trainingSet);
    os.close();
    
    return 0;
}
