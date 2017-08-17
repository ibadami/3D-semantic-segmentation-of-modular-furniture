
#include <random>
#include <fstream>

#include "gtest/gtest.h"
#include "libforest/data.h"
#include "libforest/io.h"

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// Unit tests for the class "ClassLabelMap"
////////////////////////////////////////////////////////////////////////////////

TEST(ClassLabelMap, getClassCount)
{
    ClassLabelMap map;
    
    const std::string label("test");
    
    ASSERT_EQ(map.getClassCount(), 0);
    
    map.addClassLabel(label);
    
    ASSERT_EQ(map.getClassCount(), 1);
}

TEST(ClassLabelMap, getClassLabel_stringIncorrect)
{
    ClassLabelMap map;
    const std::string label("test");
    
    ASSERT_THROW(map.getClassLabel(label), AssertionException);
}

TEST(ClassLabelMap, getClassLabel_stringCorrect)
{
    ClassLabelMap map;
    
    int label = map.addClassLabel("test");
    
    ASSERT_EQ(map.getClassLabel("test"), label);
}

TEST(ClassLabelMap, getClassLabel_intIncorrect)
{
    ClassLabelMap map;
    
    ASSERT_THROW(map.getClassLabel(0), AssertionException);
}

TEST(ClassLabelMap, getClassLabel_intCorrect)
{
    ClassLabelMap map;
    
    const std::string strLabel("test");
    int label = map.addClassLabel(strLabel);
    
    ASSERT_EQ(map.getClassLabel(label), strLabel);
}

TEST(ClassLabelMap, addClassLabel_twice1)
{
    ClassLabelMap map;
    
    int label1 = map.addClassLabel("test");
    int label2 = map.addClassLabel("test");
    
    ASSERT_EQ(label1, label2);
}

TEST(ClassLabelMap, addClassLabel_twice2)
{
    ClassLabelMap map;
    
    int label1 = map.addClassLabel("test");
    int label2 = map.addClassLabel("test2");
    
    ASSERT_NE(label1, label2);
}

/**
 * Tests if the final integer class labels really are independent of the order
 * in which the string labels are added.
 */
TEST(ClassLabelMap, computeIntClassLabels_independency)
{
    ClassLabelMap map1;
    ClassLabelMap map2;
    
    map1.addClassLabel("c");
    map1.addClassLabel("a");
    map1.addClassLabel("b");
    map2.addClassLabel("b");
    map2.addClassLabel("a");
    map2.addClassLabel("c");
    
    std::vector<int> permutation1, permutation2;
    map1.computeIntClassLabels(permutation1);
    map2.computeIntClassLabels(permutation2);
    
    ASSERT_EQ(map1.getClassLabel("a"), map2.getClassLabel("a"));
    ASSERT_EQ(map1.getClassLabel("b"), map2.getClassLabel("b"));
    ASSERT_EQ(map1.getClassLabel("c"), map2.getClassLabel("c"));
    ASSERT_EQ(map1.getClassLabel(0), map2.getClassLabel(0));
    ASSERT_EQ(map1.getClassLabel(1), map2.getClassLabel(1));
    ASSERT_EQ(map1.getClassLabel(2), map2.getClassLabel(2));
}

TEST(ClassLabelMap, readWrite1)
{
    ClassLabelMap map1;
    ClassLabelMap map2;
    
    map1.addClassLabel("c");
    map1.addClassLabel("a");
    map1.addClassLabel("b");
    
    write("classLabelMap.dat", map1);
    read("classLabelMap.dat", map2);
    
    ASSERT_EQ(map1.getClassCount(), map2.getClassCount());
    ASSERT_EQ(map1.getClassLabel("a"), map2.getClassLabel("a"));
    ASSERT_EQ(map1.getClassLabel("b"), map2.getClassLabel("b"));
    ASSERT_EQ(map1.getClassLabel("c"), map2.getClassLabel("c"));
    ASSERT_EQ(map1.getClassLabel(0), map2.getClassLabel(0));
    ASSERT_EQ(map1.getClassLabel(1), map2.getClassLabel(1));
    ASSERT_EQ(map1.getClassLabel(2), map2.getClassLabel(2));
}

TEST(ClassLabelMap, readWrite2)
{
    ClassLabelMap map1;
    ClassLabelMap map2;
    
    map1.addClassLabel("c");
    map1.addClassLabel("a");
    map1.addClassLabel("b");
    
    write("classLabelMap.dat", map1);
    read("classLabelMap.dat", map2);
    
    std::vector<int> permutation1, permutation2;
    map1.computeIntClassLabels(permutation1);
    map2.computeIntClassLabels(permutation2);
    
    ASSERT_EQ(map1.getClassCount(), map2.getClassCount());
    ASSERT_EQ(map1.getClassLabel("a"), map2.getClassLabel("a"));
    ASSERT_EQ(map1.getClassLabel("b"), map2.getClassLabel("b"));
    ASSERT_EQ(map1.getClassLabel("c"), map2.getClassLabel("c"));
    ASSERT_EQ(map1.getClassLabel(0), map2.getClassLabel(0));
    ASSERT_EQ(map1.getClassLabel(1), map2.getClassLabel(1));
    ASSERT_EQ(map1.getClassLabel(2), map2.getClassLabel(2));
}

////////////////////////////////////////////////////////////////////////////////
/// Unit tests for the class "DataStorage"
////////////////////////////////////////////////////////////////////////////////

TEST(DataStorage, getClassLabel_invalidIndex)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    ASSERT_THROW(storage->getClassLabel(0), AssertionException);
}

TEST(DataStorage, getClassLabel_validIndex)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    DataPoint x(1);
    storage->addDataPoint(x, 1);
    
    ASSERT_EQ(storage->getClassLabel(0), 1);
}

TEST(DataStorage, getClassLabel_validIndex2)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    DataPoint x(1);
    storage->addDataPoint(x);
    
    ASSERT_EQ(storage->getClassLabel(0), LIBF_NO_LABEL);
}

TEST(DataStorage, getClasscount_noLabel)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    DataPoint x(1);
    storage->addDataPoint(x);
    
    ASSERT_EQ(storage->getClasscount(), 0);
}

TEST(DataStorage, getClasscount_label)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    DataPoint x(1);
    storage->addDataPoint(x, 0);
    storage->addDataPoint(x, 4);
    storage->addDataPoint(x, 9);
    
    ASSERT_EQ(storage->getClasscount(), 10);
}

TEST(DataStorage, getDataPoint_invalidIndex)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    ASSERT_THROW(storage->getDataPoint(0), AssertionException);
}

TEST(DataStorage, getDataPoint_validIndex)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    DataPoint x(1);
    storage->addDataPoint(x);
    
    ASSERT_EQ(storage->getDataPoint(0), x);
}

TEST(DataStorage, getSize)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    ASSERT_EQ(storage->getSize(), 0);
    
    DataPoint x(1);
    storage->addDataPoint(x);
    
    ASSERT_EQ(storage->getSize(), 1);
}

TEST(DataStorage, getDimensionality)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    ASSERT_EQ(storage->getDimensionality(), 0);
    
    DataPoint x(3);
    storage->addDataPoint(x);
    
    ASSERT_EQ(storage->getDimensionality(), 3);
}

TEST(DataStorage, containsUnlabeledPoints_noPoints)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    ASSERT_FALSE(storage->containsUnlabeledPoints());
}

TEST(DataStorage, containsUnlabeledPoints_unlabeledPoints)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(3);
    storage->addDataPoint(x);
    ASSERT_TRUE(storage->containsUnlabeledPoints());
}

TEST(DataStorage, containsUnlabeledPoints_labeledPoints)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(3);
    storage->addDataPoint(x, 0);
    ASSERT_FALSE(storage->containsUnlabeledPoints());
}

TEST(DataStorage, containsUnlabeledPoints_mixedLabeledPoints)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(3);
    storage->addDataPoint(x, 0);
    storage->addDataPoint(x);
    ASSERT_TRUE(storage->containsUnlabeledPoints());
}

TEST(DataStorage, excerpt)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1), y(1), z(1), a(1), b(1);
    x(0) = 1; y(0) = 2; z(0) = 3; a(0) = 4; b(0) = 5;
    
    storage->addDataPoint(x);
    storage->addDataPoint(y, 0);
    storage->addDataPoint(z, 1);
    storage->addDataPoint(a);
    storage->addDataPoint(b, 2);
    
    AbstractDataStorage::ptr newStorage = storage->excerpt(1, 3);
    
    ASSERT_EQ(newStorage->getSize(), 3);
    ASSERT_EQ(newStorage->getClassLabel(0), 0);
    ASSERT_EQ(newStorage->getClassLabel(1), 1);
    ASSERT_EQ(newStorage->getClassLabel(2), LIBF_NO_LABEL);
    ASSERT_EQ(newStorage->getDataPoint(0), y);
    ASSERT_EQ(newStorage->getDataPoint(1), z);
    ASSERT_EQ(newStorage->getDataPoint(2), a);
}

TEST(DataStorage, bootstrap)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1), y(1);
    x(0) = 1; y(0) = 2; 
    
    storage->addDataPoint(x);
    storage->addDataPoint(y, 0);
    
    for (int c = 0; c < 100; c++)
    {
        std::vector<bool> sampled;
        AbstractDataStorage::ptr newStorage = storage->bootstrap(100, sampled);
        
        ASSERT_EQ(newStorage->getSize(), 100);
        ASSERT_EQ(static_cast<int>(sampled.size()), 2);
        
        for (int n = 0; n < newStorage->getSize(); n++)
        {
            ASSERT_TRUE(newStorage->getDataPoint(n) == x || newStorage->getDataPoint(n) == y);
            if (newStorage->getDataPoint(n) == x)
            {
                ASSERT_EQ(newStorage->getClassLabel(n), LIBF_NO_LABEL);
            }
            else
            {
                ASSERT_EQ(newStorage->getClassLabel(n), 0);
            }
        }
    }
}

TEST(DataStorage, permute)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1), y(1), z(1);
    x(0) = 1; y(0) = 2, z(0) = 3;
    
    storage->addDataPoint(x);
    storage->addDataPoint(y, 0);
    storage->addDataPoint(z, 2);
    
    std::vector<int> sigma({2,1,0});
    
    storage->permute(sigma);
    
    ASSERT_EQ(storage->getDataPoint(0), z);
    ASSERT_EQ(storage->getClassLabel(0), 2);
    ASSERT_EQ(storage->getDataPoint(1), y);
    ASSERT_EQ(storage->getClassLabel(1), 0);
    ASSERT_EQ(storage->getDataPoint(2), x);
    ASSERT_EQ(storage->getClassLabel(2), LIBF_NO_LABEL);
}

TEST(DataStorage, copy)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1), y(1), z(1);
    x(0) = 1; y(0) = 2, z(0) = 3;
    
    storage->addDataPoint(x);
    storage->addDataPoint(y, 0);
    storage->addDataPoint(z, 2);
    
    AbstractDataStorage::ptr copy = storage->copy();
    
    ASSERT_EQ(copy->getSize(), 3);
    ASSERT_EQ(copy->getDataPoint(0), x);
    ASSERT_EQ(copy->getClassLabel(0), LIBF_NO_LABEL);
    ASSERT_EQ(copy->getDataPoint(1), y);
    ASSERT_EQ(copy->getClassLabel(1), 0);
    ASSERT_EQ(copy->getDataPoint(2), z);
    ASSERT_EQ(copy->getClassLabel(2), 2);
}

TEST(DataStorage, addDataPoint_invalidDimension)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1), y(2);
    
    storage->addDataPoint(x);
    
    ASSERT_THROW(storage->addDataPoint(y, 0), AssertionException);
    ASSERT_THROW(storage->addDataPoint(y), AssertionException);
}

TEST(DataStorage, addDataPoint_invalidLabel)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1);
    
    storage->addDataPoint(x);
    
    ASSERT_THROW(storage->addDataPoint(x, -1), AssertionException);
}

TEST(DataStorage, select)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1), y(1), z(1);
    x(0) = 1; y(0) = 2, z(0) = 3;
    
    storage->addDataPoint(x);
    storage->addDataPoint(y, 0);
    storage->addDataPoint(z, 2);
    
    AbstractDataStorage::ptr selection = storage->select([](const DataPoint & x, int c) {
        return c == 2;
    });
    
    ASSERT_EQ(selection->getSize(), 1);
    ASSERT_EQ(selection->getDataPoint(0), z);
    ASSERT_EQ(selection->getClassLabel(0), 2);
}

TEST(DataStorage, addDataPoints)
{
    DataStorage::ptr storage1 = DataStorage::Factory::create();
    DataStorage::ptr storage2 = DataStorage::Factory::create();
    DataPoint x(1), y(1), z(1);
    x(0) = 1; y(0) = 2, z(0) = 3;
    
    storage1->addDataPoint(x);
    storage1->addDataPoint(y, 0);
    storage2->addDataPoint(z, 2);
    
    storage1->addDataPoints(storage2);
    
    ASSERT_EQ(storage1->getSize(), 3);
    ASSERT_EQ(storage1->getDataPoint(2), z);
    ASSERT_EQ(storage1->getClassLabel(2), 2);
}

////////////////////////////////////////////////////////////////////////////////
/// Unit tests for the class "ReferenceDataStorage"
////////////////////////////////////////////////////////////////////////////////

TEST(ReferenceDataStorage, getClassLabel_invalidIndex)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1);
    for (int n = 0; n < 10; n++) storage->addDataPoint(x);
    
    ReferenceDataStorage::ptr refStorage = std::make_shared<ReferenceDataStorage>(storage);
    
    ASSERT_THROW(refStorage->getClassLabel(0), AssertionException);
}

TEST(ReferenceDataStorage, getClassLabel_validIndex)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1);
    for (int n = 0; n < 10; n++) storage->addDataPoint(x);
    
    ReferenceDataStorage::ptr refStorage = std::make_shared<ReferenceDataStorage>(storage);
    
    refStorage->addDataPoint(0);
    
    ASSERT_EQ(refStorage->getClassLabel(0), storage->getClassLabel(0));
}

TEST(ReferenceDataStorage, getClasscount)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1);
    for (int n = 0; n < 10; n++) storage->addDataPoint(x);
    
    ReferenceDataStorage::ptr refStorage = std::make_shared<ReferenceDataStorage>(storage);
    
    ASSERT_EQ(refStorage->getClasscount(), storage->getClasscount());
}

TEST(ReferenceDataStorage, getDataPoint_invalidIndex)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1);
    for (int n = 0; n < 10; n++) storage->addDataPoint(x);
    
    ReferenceDataStorage::ptr refStorage = std::make_shared<ReferenceDataStorage>(storage);
    
    ASSERT_THROW(refStorage->addDataPoint(10), AssertionException);
}

TEST(ReferenceDataStorage, getDataPoint_validIndex)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1);
    for (int n = 0; n < 10; n++) storage->addDataPoint(x);
    
    ReferenceDataStorage::ptr refStorage = std::make_shared<ReferenceDataStorage>(storage);
    
    refStorage->addDataPoint(9);
    
    ASSERT_EQ(refStorage->getDataPoint(0), storage->getDataPoint(9));
    ASSERT_EQ(refStorage->getClassLabel(0), storage->getClassLabel(9));
}

TEST(ReferenceDataStorage, getSize)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1);
    for (int n = 0; n < 10; n++) storage->addDataPoint(x);
    
    ReferenceDataStorage::ptr refStorage = std::make_shared<ReferenceDataStorage>(storage);
    
    ASSERT_EQ(refStorage->getSize(), 0);
    refStorage->addDataPoint(9);
    ASSERT_EQ(refStorage->getSize(), 1);
}

TEST(ReferenceDataStorage, permute)
{
    DataStorage::ptr storage = DataStorage::Factory::create();
    DataPoint x(1), y(1), z(1);
    x(0) = 1; y(0) = 2; z(0) = 3;
    
    storage->addDataPoint(x, 0);
    storage->addDataPoint(y, 1);
    storage->addDataPoint(z, 2);
    
    ReferenceDataStorage::ptr refStorage = std::make_shared<ReferenceDataStorage>(storage);
    for(int n = 0; n < 3; n++) refStorage->addDataPoint(n);
    
    std::vector<int> sigma({1,0,2});
    
    refStorage->permute(sigma);
    
    ASSERT_EQ(refStorage->getClassLabel(0), 1);
    ASSERT_EQ(refStorage->getClassLabel(1), 0);
    ASSERT_EQ(refStorage->getClassLabel(2), 2);
    ASSERT_EQ(refStorage->getDataPoint(0), y);
    ASSERT_EQ(refStorage->getDataPoint(1), x);
    ASSERT_EQ(refStorage->getDataPoint(2), z);
}

////////////////////////////////////////////////////////////////////////////////
/// Unit tests for the class "CSVDataReader" and "CSVDataWriter"
////////////////////////////////////////////////////////////////////////////////

TEST(CSVData, readWrite_labeledDataComma)
{
    // Create a data set
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> entryDist(0.0f, 10.0f);
    std::uniform_int_distribution<int> labelDist(0, 30);
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    const int N = 1000;
    const int D = 80;
    
    for (int n = 0; n < N; n++)
    {
        DataPoint x(D);
        
        for (int d = 0; d < D; d++)
        {
            x(d) = entryDist(g);
        }
        
        storage->addDataPoint(x, labelDist(g));
    }
    
    CSVDataReader reader;
    CSVDataWriter writer;
    
    reader.setClassLabelColumnIndex(10);
    writer.setClassLabelColumnIndex(10);
    reader.setReadClassLabels(true);
    writer.setWriteClassLabels(true);
    reader.setColumnSeparator(",");
    writer.setColumnSeparator(",");
    
    writer.write("data.csv", storage);
    
    DataStorage::ptr readStorage = DataStorage::Factory::create();
    
    reader.read("data.csv", readStorage);
    
    for (int n = 0; n < N; n++)
    {
        ASSERT_EQ(readStorage->getClassLabel(n), storage->getClassLabel(n));
        for (int d = 0; d < D; d++)
        {
            ASSERT_FLOAT_EQ(readStorage->getDataPoint(n)(d), storage->getDataPoint(n)(d));
        }
    }
}

TEST(CSVData, read_labeledDataComma)
{
    // Create a data small CSV file
    std::ofstream stream("data.csv");
    
    ASSERT_TRUE(stream.is_open());
    
    stream << "0, 1, 2" << std::endl;
    stream << "1, 4, 8" << std::endl;
    stream << "0, 2, 10.5" << std::endl;
    
    stream.close();
    
    CSVDataReader reader;
    
    reader.setClassLabelColumnIndex(0);
    reader.setReadClassLabels(true);
    reader.setColumnSeparator(",");
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    reader.read("data.csv", storage);
    
    ASSERT_EQ(storage->getSize(), 3);
    ASSERT_EQ(storage->getClassLabel(0), 0);
    ASSERT_EQ(storage->getClassLabel(1), 1);
    ASSERT_EQ(storage->getClassLabel(2), 0);
    
    ASSERT_FLOAT_EQ(storage->getDataPoint(0)(0), 1.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(0)(1), 2.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(1)(0), 4.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(1)(1), 8.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(2)(0), 2.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(2)(1), 10.5f);
}

TEST(CSVData, readWrite_unlabeledDataComma)
{
    // Create a data set
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> entryDist(0.0f, 10.0f);
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    const int N = 1000;
    const int D = 80;
    
    for (int n = 0; n < N; n++)
    {
        DataPoint x(D);
        
        for (int d = 0; d < D; d++)
        {
            x(d) = entryDist(g);
        }
        
        storage->addDataPoint(x);
    }
    
    CSVDataReader reader;
    CSVDataWriter writer;
    
    reader.setReadClassLabels(false);
    writer.setWriteClassLabels(false);
    reader.setColumnSeparator(",");
    writer.setColumnSeparator(",");
    
    writer.write("data.csv", storage);
    
    DataStorage::ptr readStorage = DataStorage::Factory::create();
    
    reader.read("data.csv", readStorage);
    
    for (int n = 0; n < N; n++)
    {
        ASSERT_EQ(readStorage->getClassLabel(n), storage->getClassLabel(n));
        for (int d = 0; d < D; d++)
        {
            ASSERT_FLOAT_EQ(readStorage->getDataPoint(n)(d), storage->getDataPoint(n)(d));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Unit tests for the class "LibforestDataWriter" and "LibforestDataReader"
////////////////////////////////////////////////////////////////////////////////

TEST(LibforestData, readWrite_labeledData)
{
    // Create a data set
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> entryDist(0.0f, 10.0f);
    std::uniform_int_distribution<int> labelDist(0, 30);
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    const int N = 1000;
    const int D = 80;
    
    for (int n = 0; n < N; n++)
    {
        DataPoint x(D);
        
        for (int d = 0; d < D; d++)
        {
            x(d) = entryDist(g);
        }
        
        storage->addDataPoint(x, labelDist(g));
    }
    
    LibforestDataReader reader;
    LibforestDataWriter writer;
    
    writer.write("data.dat", storage);
    
    DataStorage::ptr readStorage = DataStorage::Factory::create();
    
    reader.read("data.dat", readStorage);
    
    for (int n = 0; n < N; n++)
    {
        ASSERT_EQ(readStorage->getClassLabel(n), storage->getClassLabel(n));
        for (int d = 0; d < D; d++)
        {
            ASSERT_FLOAT_EQ(readStorage->getDataPoint(n)(d), storage->getDataPoint(n)(d));
        }
    }
}

TEST(LibforestData, readWrite_unlabeledData)
{
    // Create a data set
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> entryDist(0.0f, 10.0f);
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    const int N = 1000;
    const int D = 80;
    
    for (int n = 0; n < N; n++)
    {
        DataPoint x(D);
        
        for (int d = 0; d < D; d++)
        {
            x(d) = entryDist(g);
        }
        
        storage->addDataPoint(x);
    }
    
    LibforestDataReader reader;
    LibforestDataWriter writer;
    
    writer.write("data.dat", storage);
    
    DataStorage::ptr readStorage = DataStorage::Factory::create();
    
    reader.read("data.dat", readStorage);
    
    for (int n = 0; n < N; n++)
    {
        ASSERT_EQ(readStorage->getClassLabel(n), storage->getClassLabel(n));
        for (int d = 0; d < D; d++)
        {
            ASSERT_FLOAT_EQ(readStorage->getDataPoint(n)(d), storage->getDataPoint(n)(d));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Unit tests for the class "LIBSVMDataWriter" and "LIBSVMDataReader"
////////////////////////////////////////////////////////////////////////////////

TEST(LIBSVM, readWrite_labeledData)
{
    // Create a data set
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> entryDist(0.0f, 10.0f);
    std::uniform_int_distribution<int> labelDist(0, 30);
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    const int N = 1000;
    const int D = 80;
    
    for (int n = 0; n < N; n++)
    {
        DataPoint x(D);
        
        for (int d = 0; d < D; d++)
        {
            x(d) = entryDist(g);
        }
        
        storage->addDataPoint(x, labelDist(g));
    }
    
    LIBSVMDataReader reader;
    LIBSVMDataWriter writer;
    
    writer.write("data.txt", storage);
    
    DataStorage::ptr readStorage = DataStorage::Factory::create();
    
    reader.read("data.txt", readStorage);
    
    ASSERT_EQ(readStorage->getSize(), N);
    for (int n = 0; n < N; n++)
    {
        ASSERT_EQ(readStorage->getClassLabel(n), storage->getClassLabel(n));
        for (int d = 0; d < D; d++)
        {
            ASSERT_FLOAT_EQ(readStorage->getDataPoint(n)(d), storage->getDataPoint(n)(d));
        }
    }
}

TEST(LIBSVM, read_labeledData)
{
    // Create a data small CSV file
    std::ofstream stream("data.txt");
    
    ASSERT_TRUE(stream.is_open());
    
    stream << "1 1:1 2:2" << std::endl;
    stream << "2 1:4 2:8" << std::endl;
    stream << "1 1:2 2:10.5" << std::endl;
    
    stream.close();
    
    LIBSVMDataReader reader;
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    reader.read("data.txt", storage);
    
    ASSERT_EQ(storage->getSize(), 3);
    ASSERT_EQ(storage->getClassLabel(0), 0);
    ASSERT_EQ(storage->getClassLabel(1), 1);
    ASSERT_EQ(storage->getClassLabel(2), 0);
    
    ASSERT_FLOAT_EQ(storage->getDataPoint(0)(0), 1.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(0)(1), 2.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(1)(0), 4.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(1)(1), 8.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(2)(0), 2.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(2)(1), 10.5f);
}


TEST(LIBSVM, read_labeledDataZeros)
{
    // Create a data small CSV file
    std::ofstream stream("data.txt");
    
    ASSERT_TRUE(stream.is_open());
    
    stream << "1.00000 1:1 2:2" << std::endl;
    stream << "2.0000 1:4 2:8" << std::endl;
    stream << "1.000 1:2 2:10.5" << std::endl;
    
    stream.close();
    
    LIBSVMDataReader reader;
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    reader.read("data.txt", storage);
    
    ASSERT_EQ(storage->getSize(), 3);
    ASSERT_EQ(storage->getClassLabel(0), 0);
    ASSERT_EQ(storage->getClassLabel(1), 1);
    ASSERT_EQ(storage->getClassLabel(2), 0);
    
    ASSERT_FLOAT_EQ(storage->getDataPoint(0)(0), 1.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(0)(1), 2.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(1)(0), 4.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(1)(1), 8.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(2)(0), 2.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(2)(1), 10.5f);
}


TEST(LIBSVM, read_binaryData)
{
    // Create a data small CSV file
    std::ofstream stream("data.txt");
    
    ASSERT_TRUE(stream.is_open());
    
    stream << "-1 1:1 2:2" << std::endl;
    stream << "1 1:4 2:8" << std::endl;
    stream << "-1 1:2 2:10.5" << std::endl;
    
    stream.close();
    
    LIBSVMDataReader reader;
    reader.setConvertBinaryLabels(true);
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    reader.read("data.txt", storage);
    
    ASSERT_EQ(storage->getSize(), 3);
    ASSERT_EQ(storage->getClassLabel(0), 0);
    ASSERT_EQ(storage->getClassLabel(1), 1);
    ASSERT_EQ(storage->getClassLabel(2), 0);
    
    ASSERT_FLOAT_EQ(storage->getDataPoint(0)(0), 1.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(0)(1), 2.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(1)(0), 4.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(1)(1), 8.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(2)(0), 2.0f);
    ASSERT_FLOAT_EQ(storage->getDataPoint(2)(1), 10.5f);
}

TEST(LIBSVM, readWrite_unlabeledData)
{
    // Create a data set
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution<float> entryDist(0.0f, 10.0f);
    
    DataStorage::ptr storage = DataStorage::Factory::create();
    
    const int N = 1000;
    const int D = 80;
    
    for (int n = 0; n < N; n++)
    {
        DataPoint x(D);
        
        for (int d = 0; d < D; d++)
        {
            x(d) = entryDist(g);
        }
        
        storage->addDataPoint(x);
    }
    
    LIBSVMDataReader reader;
    LIBSVMDataWriter writer;
    
    writer.write("data.txt", storage);
    
    DataStorage::ptr readStorage = DataStorage::Factory::create();
    
    reader.read("data.txt", readStorage);
    
    for (int n = 0; n < N; n++)
    {
        ASSERT_EQ(readStorage->getClassLabel(n), storage->getClassLabel(n));
        for (int d = 0; d < D; d++)
        {
            ASSERT_FLOAT_EQ(readStorage->getDataPoint(n)(d), storage->getDataPoint(n)(d));
        }
    }
}

