#include <iostream>
#include "libforest/libforest.h"
#include <chrono>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace libf;

/**
 * Command line tool for converting between CSV and DAT files.
 * 
 * Usage:
 * $ ./examples/cli_convert --help
 * Allowed options:
 *   --help                   produce help message
 *   --in-file arg            path to input file
 *   --out-file arg           path to output file
 *   --csv-to-dat             convert CSV to DAT
 *   --csv-label-col arg (=0) CSV column for label
 */
int main(int argc, const char** argv)
{
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("in-file", boost::program_options::value<std::string>(), "path to input file")
        ("out-file", boost::program_options::value<std::string>(), "path to output file")
        ("csv-to-dat", "convert CSV to DAT")
        ("csv-label-col", boost::program_options::value<int>()->default_value(0), "CSV column for label")
        ("csv-separator", boost::program_options::value<std::string>()->default_value(" "), "CSV column separator");
    
    boost::program_options::positional_options_description positionals;
    positionals.add("in-file", 1);
    positionals.add("out-file", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end())
    {
        std::cout << desc << std::endl;
        return 1;
    }
    
    boost::filesystem::path inFile(parameters["in-file"].as<std::string>());
    if (!boost::filesystem::is_regular_file(inFile))
    {
        std::cout << "Input file does not exist." << std::endl;
        return 1;
    }
    
    boost::filesystem::path outFile(parameters["out-file"].as<std::string>());
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    if (parameters.find("csv-to-dat") != parameters.end())
    {
        CSVDataReader reader;
        reader.setClassLabelColumnIndex(parameters["csv-label-col"].as<int>());
        reader.setReadClassLabels(true);
        reader.setColumnSeparator(parameters["csv-separator"].as<std::string>());
        reader.read(inFile.string(), storage);
        LibforestDataWriter writer;
        writer.write(outFile.string(), storage);
    }
    // Other conversions not possible yet.
    
    return 0;
}
