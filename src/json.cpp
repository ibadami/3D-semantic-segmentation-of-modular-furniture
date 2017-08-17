#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "parser/json.h"
#include "parser/types.h"

using namespace parser;

////////////////////////////////////////////////////////////////////////////////
//// JSONFactory functions

JSON::ptr JSON::Factory::createBlank()
{
    return ptr(new rapidjson::Document);
}

JSON::ptr JSON::Factory::loadFromFile(const std::string& _fileName)
{
    // Read the file into a string
    std::stringstream ss;
    std::ifstream istream;
    istream.open(_fileName.c_str(), std::ios::binary);
    
    if (!istream.is_open())
    {
        throw ParserException("Cannot open json file.");
    }
    
    ss << istream.rdbuf();
    istream.close();
    
    // Parse the string
    return JSON::Factory::loadFromString(ss.str());
}

JSON::ptr JSON::Factory::loadFromString(const std::string& _str)
{
    JSON::ptr json = JSON::Factory::createBlank();
    
    if (json->Parse<0>(_str.c_str()).HasParseError())
    {
        throw ParserException("Invalid JSON structure.");
    }
    
    return json;
}


