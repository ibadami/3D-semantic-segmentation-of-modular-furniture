/* 
 * File:   json.h
 * Author: pohlen
 *
 * Created on June 30, 2014, 2:47 PM
 */

#ifndef JSON_H
#define	JSON_H

#include <memory>

#include "rapidjson/document.h"

namespace parser {
    /**
     * A factory for json objects. 
     */
    class JSON {
    public:
        typedef std::shared_ptr<rapidjson::Document> ptr;
        
        class Factory {
        public:
            /**
             * Creates a new blank document
             * 
             * @return Blank document
             */
            static ptr createBlank();
            
            /**
             * Loads a json object from a file
             * 
             * @param _fileName The file name
             * @return the json object
             */
            static ptr loadFromFile(const std::string & _fileName);

            /**
             * Loads a json object from a string
             * 
             * @param _str The string containing the json code
             * @return the json object
             */
            static ptr loadFromString(const std::string & _str);
        };
    };
}

#endif	/* JSON_H */

