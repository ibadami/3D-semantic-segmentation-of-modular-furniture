#ifndef LIBF_DATA_H
#define LIBF_DATA_H

#include <vector>
#include <utility>
#include <map>
#include <cassert>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <functional>
#include "error_handling.h"

namespace libf {
    /**
     * Forward declarations
     */
    class DataStorage;
    class ReferenceDataStorage;
    
    /**
     * We use eigen3 vectors for data points. This allows us to build quickly
     * implement new features that rely on linear algebra. 
     */
    typedef Eigen::VectorXf DataPoint;
    
        
    /**
     * This is the label for points without a label
     */
#define LIBF_NO_LABEL -9999
    
    /**
     * This is a class label map. The internal data storage works using integer
     * class labels. When loading a data set from a file, the class labels are
     * transformed from strings to integers. This map stores the relation 
     * between them.
     * 
     * How this map works:
     * - All string class labels get a preliminary class label when they are added
     * - Once all labels have been added, the final integer class label are computed
     * 
     * This has a reason: We want the integer class labels to be independent of 
     * the order in which they are read. In order for this to work, we have to
     * know all class labels. 
     */
    class ClassLabelMap {
    public:
        /**
         * Returns the number of classes.
         * 
         * @return The total number of classes
         */
        int getClassCount() const
        {
            return static_cast<int>(inverseLabelMap.size());
        }
        
        /**
         * Returns the integer class label for a given string class label
         * 
         * @param label The class label to return the integer label for
         * @return The corresponding int class label
         */
        int getClassLabel(const std::string & label) const
        {
            BOOST_ASSERT_MSG(labelMap.find(label) != labelMap.end(), "Invalid string class label.");
            return labelMap.find(label)->second;
        }
        
        /**
         * Returns the string class label for a given integer class label.
         * 
         * @param label The label to return the string class label for
         * @return The string class label
         */
        const std::string & getClassLabel(int label) const
        {
            BOOST_ASSERT_MSG(0 <= label && label < getClassCount(), "Invalid integer class label.");
            return inverseLabelMap[label];
        }
        
        /**
         * Adds a string class label and returns a primarily integer class label. 
         * 
         * @param label The new class label 
         * @return The preliminary int class label of the newly added label
         */
        int addClassLabel(const std::string & label)
        {
            // Did we already see the label before?
            if (labelMap.find(label) == labelMap.end())
            {
                // Nope, add it 
                inverseLabelMap.push_back(label);
                labelMap[label] = static_cast<int>(inverseLabelMap.size() - 1);
            }
            return labelMap[label];
        }
        
        /**
         * Computes the final integer class labels and returns a mapping from the 
         * preliminary class labels to the final class labels. 
         * 
         * @param intLabelMap A map [preliminary label] -> [final label]
         */
        void computeIntClassLabels(std::vector<int> & intLabelMap);
        
        /**
         * Writes the class label map to a stream. 
         * 
         * @param stream The stream to write the map to
         */
        void write(std::ostream & stream) const;
        
        /**
         * Reads the class label map from a file
         * 
         * @param stream The stream to read the map from
         */
        void read(std::istream & stream);
        
    private:
        /**
         * The actual map
         */
        std::map<std::string, int> labelMap;
        /**
         * The inverse map for fast class label prediction
         */
        std::vector<std::string> inverseLabelMap;
    };
    
    /**
     * This is the base class of all data storages. 
     */
    class AbstractDataStorage : public std::enable_shared_from_this<AbstractDataStorage> {
    public:
        typedef std::shared_ptr<AbstractDataStorage> ptr;
        typedef std::shared_ptr<const AbstractDataStorage> const_ptr;
        
        /**
         * Returns the i-th class label. 
         * 
         * @param i The data point index
         * @return The class label of the i-th data point
         */
        virtual int getClassLabel(int i) const = 0;
        
        /**
         * Returns the number of classes. 
         * 
         * @return The number of observed classes
         */
        virtual int getClasscount() const = 0;
        
        /**
         * Returns the i-th vector from the storage
         * 
         * @param i The index of the data point to return
         * @return the i-th data point
         */
        virtual const DataPoint & getDataPoint(int i) const = 0;
        
        /**
         * Returns the number of data points. 
         * 
         * @return The number of data points in this data storage
         */
        virtual int getSize() const = 0;
        
        /**
         * Returns the dimensionality of the data storage. 
         * 
         * @return The dimensionality of the data storage.
         */
        int getDimensionality() const;

        /**
         * Returns true if there are unlabeled data points in the storage. 
         * 
         * @return True if there are unlabeled points.
         */
        bool containsUnlabeledPoints() const;

        /**
         * Create a DataStorage for an excerpt of thisDataStorage.
         * 
         * @param begin The index of the first data point
         * @param end The index of the last data point
         * @return A data storage with the bootstrapped examples
         */
        AbstractDataStorage::ptr excerpt(int begin, int end) const;
        
        /**
         * Bootstrap-samples the data storage. 
         * 
         * @param N The number of data points to sample
         * @param sampled Array of flags. sampled[i] == true <=> point i was sampled
         * @param referenceStorage The shallow storage to put the data points in
         */
        AbstractDataStorage::ptr bootstrap(int N, std::vector<bool> & sampled) const;
        
        /**
         * Permutes the data points according to some permutation. Please 
         * notice that this will also change reference data storage that depend
         * on this storage.
         * 
         * @param permutation A given permutation.
         */
        virtual void permute(const std::vector<int> & permutation) = 0;
        
        /**
         * Permutes the data points randomly.  Please 
         * notice that this will also change reference data storage that depend
         * on this storage.
         */
        void randPermute();
        
        /**
         * Dumps information about the data storage
         * 
         * @param stream The stream to dump the information to
         */
        virtual void dumpInformation(std::ostream & stream = std::cout);
        
        /**
         * Creates a reference copy. 
         * 
         * @return A reference copy. 
         */
        AbstractDataStorage::ptr copy() const
        {
            return excerpt(0, getSize() - 1);
        }
        
        /**
         * Creates a hard copy of the data storage copying all entries. This
         * can be very expensive. 
         * 
         * @return A hard copy of the data storage
         */
        std::shared_ptr<DataStorage> hardCopy() const;
        
        /**
         * Creates a reference storage with a subset of data points from this 
         * storage. The subset is defined by all points for which the given
         * callback function evaluates to true. 
         */
        AbstractDataStorage::ptr select(const std::function<bool(const DataPoint &, int)> & f) const;
    };
    
    /**
     * This is a storage for labeled and unlabeled data. Data points without 
     * a label get the class label LIBF_NO_LABEL. This allows us to have mixed storages
     * with missing labels. 
     */
    class DataStorage : public AbstractDataStorage {
    public:
        typedef std::shared_ptr<DataStorage> ptr;
        
        /**
         * Initializes an empty data storage
         */
        DataStorage() : classcount(0) {}
        
        /**
         * Copy constructor. 
         * Please be aware that this can be very expensive. 
         * 
         * @param other The data storage to copy
         */
        DataStorage(const DataStorage & other)
        {
            dataPoints = other.dataPoints;
            classcount = other.classcount;
        }
        
        /**
         * Returns the i-th class label. 
         * 
         * @param i The data point index
         * @return The class label of the i-th data point
         */
        int getClassLabel(int i) const
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "Data point index out of bounds.");
            return classLabels[i];
        }
        
        /**
         * Returns the i-th class label. 
         * Do not simply change the class label of a point. This function should
         * only be used by library developers.
         * 
         * @param i The data point index
         * @return The class label of the i-th data point
         * @internal
         */
        int & getClassLabel(int i) 
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "Data point index out of bounds.");
            return classLabels[i];
        }
        
        /**
         * Returns the number of classes. 
         * 
         * @return The number of observed classes
         */
        int getClasscount() const
        {
            return classcount;
        }
        
        /**
         * Returns the i-th vector from the storage
         * 
         * @param i The index of the data point to return
         * @return the i-th data point
         */
        const DataPoint & getDataPoint(int i) const
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "The data point index is out of bounds.");
            return dataPoints[i];
        }
        
        /**
         * Returns the i-th vector from the storage. 
         * Do not alter the size of a data point. This will break the data 
         * storage. 
         * 
         * @param i The index of the data point to return
         * @return the i-th data point
         * @internal
         */
        DataPoint & getDataPoint(int i)
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "The data point index is out of bounds.");
            return dataPoints[i];
        }
        
        /**
         * Returns the number of data points. 
         * 
         * @return The number of data points in this data storage
         */
        int getSize() const
        {
            return dataPoints.size();
        }
        
        /**
         * Add a single data point without a label.
         * 
         * @param point The point to add to the storage
         */
        void addDataPoint(const DataPoint & point)
        {
            // Check if the dimensionality is correct
            BOOST_ASSERT_MSG(getSize() == 0 || getDataPoint(0).rows() == point.rows(), "The dimensionality of the new point does not match the one of the existing points.");
            
            dataPoints.push_back(point);
            classLabels.push_back(LIBF_NO_LABEL);
        }
        
        /**
         * Adds a single data point with a label. 
         * 
         * @param point The point to add to the storage
         * @param label The class label of the point
         */
        void addDataPoint(const DataPoint & point, int label)
        {
            BOOST_ASSERT_MSG(label >= 0 || label == LIBF_NO_LABEL, "The class labels must be consecutive and non-negative.");
            // Check if the dimensionality is correct
            BOOST_ASSERT_MSG(getSize() == 0 || getDataPoint(0).size() == point.size(), "The dimensionality of the new point does not match the one of the existing points.");
            
            dataPoints.push_back(point);
            classLabels.push_back(label);
            if (label >= classcount)
            {
                classcount = label + 1;
            }
        }
        
        /**
         * Adds all data points from the given storage to this one.
         * 
         * @param storage the storage to copy data points from
         */
        void addDataPoints(AbstractDataStorage::ptr storage);
        
        /**
         * Permutes the data points according to some permutation. Please 
         * notice that this will also change reference data storage that depend
         * on this storage.
         * 
         * @param permutation A given permutation.
         */
        void permute(const std::vector<int> & permutation);
        
        /**
         * A factory class for this data storage class. 
         */
        class Factory {
        public:
            /**
             * Creates a new empty data storage. 
             * 
             * @return New empty data storage
             */
            static DataStorage::ptr create()
            {
                return std::make_shared<DataStorage>();
            }
        };
        
    protected:
        /**
         * This is a list of data points. 
         */
        std::vector< DataPoint > dataPoints;
        /**
         * The total number of classes
         */
        int classcount;
        /**
         * These are the corresponding class labels to the data points
         */
        std::vector<int> classLabels;
    };
    
    /**
     * This is a shallow data storage that contains references to data points
     * from another data storage. We use this in order to perform efficient
     * bootstrap sampling. Obviously, this one only works as long as the other
     * storage is still alive. 
     * 
     * This should only be used by library developers. 
     * 
     * @internal
     */
    class ReferenceDataStorage : public AbstractDataStorage {
    public:
        typedef std::shared_ptr<ReferenceDataStorage> ptr;
        
        ReferenceDataStorage(AbstractDataStorage::const_ptr dataStorage) : dataStorage(dataStorage) {}
        
        /**
         * Returns the i-th class label. 
         * 
         * @param i The data point index
         * @return The class label of the i-th data point
         */
        int getClassLabel(int i) const
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "Data point index out of bounds.");
            return dataStorage->getClassLabel(dataPointIndices[i]);
        }
        
        /**
         * Returns the number of classes. 
         * 
         * @return The number of observed classes
         */
        int getClasscount() const
        {
            return dataStorage->getClasscount();
        }
        
        /**
         * Returns the i-th vector from the storage
         * 
         * @param i The index of the data point to return
         * @return the i-th data point
         */
        const DataPoint & getDataPoint(int i) const
        {
            BOOST_ASSERT_MSG(0 <= i && i < getSize(), "The data point index is out of bounds.");
            return dataStorage->getDataPoint(dataPointIndices[i]);
        }
        
        /**
         * Returns the number of data points. 
         * 
         * @return The number of data points in this data storage
         */
        int getSize() const
        {
            return dataPointIndices.size();
        }
        
        /**
         * Add a single data point.
         * 
         * @param n The index of the data point to add
         */
        void addDataPoint(int n)
        {
            BOOST_ASSERT_MSG(0 <= n && n < dataStorage->getSize(), "Invalid data point index from reference storage.");
            dataPointIndices.push_back(n);
        }
        
        /**
         * Permutes the data points according to some permutation. Please 
         * notice that this will also change reference data storage that depend
         * on this storage.
         * 
         * @param permutation A given permutation.
         */
        void permute(const std::vector<int> & permutation);
        
    private:
        /**
         * The referenced data point indices
         */
        std::vector<int> dataPointIndices;
        /**
         * The reference data storage
         */
        AbstractDataStorage::const_ptr dataStorage;
    };
    
    /**
     * This is the interface that has to be implemented if you wish to implement
     * a custom data provider. 
     */
    class AbstractDataReader {
    public:
        virtual ~AbstractDataReader() {}
        
        /**
         * Reads a labeled dataset from a stream.
         * 
         * @param stream The stream to read the data from
         * @param dataStorage The data storage to add the read data points to
         */
        virtual void read(std::istream & stream, DataStorage::ptr dataStorage) = 0;
        
        /**
         * Reads a labeled dataset from a file.
         * 
         * @param filename The name of the file that shall be read
         * @param dataStorage The data storage to add the read data points to
         */
        virtual void read(const std::string & filename, DataStorage::ptr dataStorage) throw(IOException);
    };
    
    /**
     * This data provider reads data from a local CSV file. 
     */
    class CSVDataReader : public AbstractDataReader {
    public:
        using AbstractDataReader::read;    

        /**
         * Constructor
         */
        CSVDataReader() : readClassLabels(true), classLabelColumnIndex(0), columnSeparator(",") {}
        
        /**
         * Destructor.
         */
        virtual ~CSVDataReader() {}
        
        /**
         * Reads a dataset from a stream.
         * 
         * @param stream The stream to read the data from
         * @param dataStorage The data storage to add the read data points to
         */
        virtual void read(std::istream & stream, DataStorage::ptr dataStorage);
        
        /**
         * Sets whether or not class labels shall be read from the file. 
         * 
         * @param _readClassLabels Whether of not class labels shall be read. True => Read labels
         */
        void setReadClassLabels(bool _readClassLabels)
        {
            readClassLabels = _readClassLabels;
        }
        
        /**
         * Returns whether or not class labels shall be read from the file. 
         * 
         * @return True if class labels are read
         */
        bool getReadClassLabels() const
        {
            return readClassLabels;
        }
        
        /**
         * Sets the class label column index. 
         * 
         * @param _classLabelColumnIndex The new index
         */
        void setClassLabelColumnIndex(int _classLabelColumnIndex)
        {
            BOOST_ASSERT_MSG(_classLabelColumnIndex >= 0, "The class label index must be non-negative.");
            classLabelColumnIndex = _classLabelColumnIndex;
        }
        
        /**
         * Returns the class label column index. 
         * 
         * @return The class label column index
         */
        int getClassLabelColumnIndex() const
        {
            return classLabelColumnIndex;
        }
        
        /**
         * Sets the column separator. 
         * 
         * @param _columnSeparator The new column separator
         */
        void setColumnSeparator(const std::string & _columnSeparator)
        {
            columnSeparator = _columnSeparator;
        }
        
        /**
         * Returns the column separator. 
         * 
         * @return The column separator
         */
        const std::string & getColumnSeparator() const
        {
            return columnSeparator;
        }
        
    private:
        /**
         * Parses a single line.
         * 
         * @param line The string of the line
         * @param result All entries of the line
         */
        void parseLine(const std::string & line, std::vector<float> & result) const;
        
        /**
         * If true, class labels are read from the file. 
         */
        bool readClassLabels;
        /**
         * The index of the column that contains the class label
         */
        int classLabelColumnIndex;
        /**
         * Separator used between columns; default usually is ','
         */
        std::string columnSeparator;
    };
    
    /**
     * This data provider reads data from a local LIBSVM file. 
     */
    class LIBSVMDataReader : public AbstractDataReader {
    public:
        using AbstractDataReader::read;
        LIBSVMDataReader() : convertBinaryLabels(false) {}
        
        virtual ~LIBSVMDataReader() {}
        
        /**
         * Reads a dataset from a stream.
         * 
         * @param stream The stream to read the data from
         * @param dataStorage The data storage to add the read data points to
         */
        virtual void read(std::istream & stream, DataStorage::ptr dataStorage);
        
        /**
         * Sets whether or not binary labels shall be converted from -1 and 1
         * to 0 and 1.
         * 
         * @param _convertBinaryLabels If true, the labels are converted. 
         */
        void setConvertBinaryLabels(bool _convertBinaryLabels)
        {
            convertBinaryLabels = _convertBinaryLabels;
        }
        
        /**
         * Returns whether or not binary labels shall be converted. 
         * 
         * @return If true, binary labels are converted
         */
        bool getConvertBinaryLabels() const
        {
            return convertBinaryLabels;
        }
    private:
        /**
         * Parses a single line and feeds.
         * 
         * @param line The string of the line
         * @param result The class label is stored in first, the dimension:feature list is stored in second
         */
        void parseLine(const std::string & line, std::pair<int, std::vector< std::pair<int, float> > > & result) const;
        
        /**
         * If true, the labels -1 and 1 are converted to 0 and 1
         */
        bool convertBinaryLabels;
    };
    
    /**
     * Reads the data set from a binary libforest format. This is the fastest
     * way to load a data set. 
     */
    class LibforestDataReader : public AbstractDataReader {
    public:
        using AbstractDataReader::read;
        
        LibforestDataReader() {}
        
        /**
         * Reads a labeled dataset from a stream.
         */
        virtual void read(std::istream & stream, DataStorage::ptr dataStorage);
        
    private:
        /**
         * Reads a single data point from a stream
         * 
         * @param stream The data stream 
         * @param v The data point where the data shall be saved into
         */
        void readDataPoint(std::istream & stream, DataPoint & v);
    };
    
    /**
     * This is the basic class for a data writer.
     */
    class AbstractDataWriter {
    public:
        /**
         * Writes the data to a stream. 
         */
        virtual void write(std::ostream & stream, DataStorage::ptr dataStorage) = 0;
        
        /**
         * Writes the data to a file and add them to the data storage. 
         */
        virtual void write(const std::string & filename, DataStorage::ptr dataStorage) throw(IOException);
    };
    
    
    /**
     * This data provider reads data from a local CSV file. 
     */
    class CSVDataWriter : public AbstractDataWriter {
    public:
        using AbstractDataWriter::write;
        
        CSVDataWriter() : writeClassLabels(true), classLabelColumnIndex(0), columnSeparator(",") {}
        
        /**
         * Writes the data to a stream. 
         */
        virtual void write(std::ostream & stream, DataStorage::ptr dataStorage);
        
        /**
         * Sets whether or not class labels shall be written to the file. 
         * 
         * @param _writeClassLabels Whether of not class labels shall be read. True => Read labels
         */
        void setWriteClassLabels(bool _writeClassLabels)
        {
            writeClassLabels = _writeClassLabels;
        }
        
        /**
         * Returns whether or not class labels shall be written to the file. 
         * 
         * @return True if class labels are written
         */
        bool getWriteClassLabels() const
        {
            return writeClassLabels;
        }
        
        /**
         * Sets the class label column index. 
         * 
         * @param _classLabelColumnIndex The new index
         */
        void setClassLabelColumnIndex(int _classLabelColumnIndex)
        {
            BOOST_ASSERT_MSG(_classLabelColumnIndex >= 0, "The class label index must be non-negative.");
            classLabelColumnIndex = _classLabelColumnIndex;
        }
        
        /**
         * Returns the class label column index. 
         * 
         * @return The class label column index
         */
        int getClassLabelColumnIndex() const
        {
            return classLabelColumnIndex;
        }
        
        /**
         * Sets the column separator. 
         * 
         * @param _columnSeparator The new column separator
         */
        void setColumnSeparator(const std::string & _columnSeparator)
        {
            columnSeparator = _columnSeparator;
        }
        
        /**
         * Returns the column separator. 
         * 
         * @return The column separator
         */
        const std::string & getColumnSeparator() const
        {
            return columnSeparator;
        }
        
    private:
        /**
         * If true, class labels are written to the file. 
         */
        bool writeClassLabels;
        /**
         * The index of the column that contains the class label
         */
        int classLabelColumnIndex;
        /**
         * Separator used between columns; default usually is ','
         */
        std::string columnSeparator;
    };
    
    /**
     * Writes the data set to a binary libforest format. This is the fastest
     * way to save a data set. 
     */
    class LibforestDataWriter : public AbstractDataWriter {
    public:
        using AbstractDataWriter::write;
        
        LibforestDataWriter() {}
        
        /**
         * Writes the data to a stream. 
         */
        virtual void write(std::ostream & stream, DataStorage::ptr dataStorage);
        
    private:
        /**
         * Writes a single data point to a stream
         * 
         * @param stream The data stream 
         * @param v The data point where the data shall be saved into
         */
        void writeDataPoint(std::ostream & stream, DataPoint & v);
    };
    
    /**
     * This data provider write data to a local LIBSVM file. 
     */
    class LIBSVMDataWriter : public AbstractDataWriter {
    public:
        using AbstractDataWriter::write;
        
        /**
         * Writes the data to a stream. 
         */
        virtual void write(std::ostream & stream, DataStorage::ptr dataStorage);
        
        /**
         * Writers a single data point to a stream
         * 
         * @param stream The data stream 
         * @param v The data point where the data shall be saved into
         */
        void writeDataPoint(std::ostream & stream, const DataPoint & v);
    };
}

#endif