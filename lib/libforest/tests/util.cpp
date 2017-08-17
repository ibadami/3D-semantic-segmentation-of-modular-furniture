
#include "gtest/gtest.h"
#include "libforest/util.h"

using namespace libf;

////////////////////////////////////////////////////////////////////////////////
/// Unit tests for the class "Util"
////////////////////////////////////////////////////////////////////////////////

/**
 * Tests if an exception is thrown when the size of the permutation and the 
 * input size do not match.
 */
TEST(Util, permute_invalidPermutationSize)
{
    std::vector<int> v({1,2,3});
    std::vector<int> sigma({2,1});
    std::vector<int> u;
    
    ASSERT_THROW(Util::permute(sigma, v, u), AssertionException);
}

/**
 * Tests if an exception is thrown when the input vector is the same as the 
 * output vector.
 */
TEST(Util, permute_inputSameAsOutput)
{
    std::vector<int> v({1,2,3});
    std::vector<int> sigma({2,1});
    
    ASSERT_THROW(Util::permute(sigma, v, v), AssertionException);
}

/**
 * Tests if a vector is correctly permuted. 
 */
TEST(Util, permute_correctlyPermute)
{
    std::vector<int> v({4,5,6});
    std::vector<int> sigma({1,0,2});
    std::vector<int> u;
    
    Util::permute(sigma, v, u);
    ASSERT_EQ(u, std::vector<int>({5,4,6}));
}

/**
 * Tests if an exception is thrown when we pass an invalid permutation:
 * Image out of upper range
 */
TEST(Util, permute_invalidPermutation_1)
{
    std::vector<int> v({4,5,6});
    std::vector<int> sigma({1,0,3});
    std::vector<int> u;
    
    ASSERT_THROW(Util::permute(sigma, v, v), AssertionException);
}


/**
 * Tests if an exception is thrown when we pass an invalid permutation:
 * Duplicate mapping
 */
TEST(Util, permute_invalidPermutation_2)
{
    std::vector<int> v({4,5,6});
    std::vector<int> sigma({1,0,1});
    std::vector<int> u;
    
    ASSERT_THROW(Util::permute(sigma, v, v), AssertionException);
}

/**
 * Tests if an exception is thrown when we pass an invalid permutation:
 * Image out of lower range
 */
TEST(Util, permute_invalidPermutation_3)
{
    std::vector<int> v({4,5,6});
    std::vector<int> sigma({1,0,-1});
    std::vector<int> u;
    
    ASSERT_THROW(Util::permute(sigma, v, v), AssertionException);
}

/**
 * Tests if the hamming distance can be computed
 */
TEST(Util, hammingDist_correctly_1)
{
    std::vector<int> v({4,5,6});
    std::vector<int> u({6,5,4});
    
    size_t distance = Util::hammingDist(v, u);
    
    ASSERT_EQ(static_cast<int>(distance), 2);
}

/**
 * Tests if the hamming distance can be computed
 */
TEST(Util, hammingDist_correctly_2)
{
    std::vector<int> v({4,5,6});
    std::vector<int> u({6,5});
    
    size_t distance = Util::hammingDist(v, u);
    
    ASSERT_EQ(static_cast<int>(distance), 2);
}

/**
 * Tests if the hamming distance can be computed
 */
TEST(Util, hammingDist_correctly_3)
{
    std::vector<int> v({1,2,3,4,5,6});
    std::vector<int> u({1,2,3,4,5});
    
    size_t distance = Util::hammingDist(v, u);
    
    ASSERT_EQ(static_cast<int>(distance), 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Unit tests for the class "EfficientEntropyHistogram"
////////////////////////////////////////////////////////////////////////////////

TEST(EfficientEntropyHistogram, copyConstructor)
{
    // Check if everything is copied correctly
    EfficientEntropyHistogram hist1(5);
    
    // Fill the histogram
    hist1.addOne(0);
    hist1.addOne(1);
    hist1.addOne(1);
    hist1.addOne(2);
    hist1.addOne(2);
    hist1.addOne(2);
    hist1.addOne(3);
    hist1.addOne(3);
    hist1.addOne(3);
    hist1.addOne(3);
    hist1.addOne(4);
    hist1.addOne(4);
    hist1.addOne(4);
    hist1.addOne(4);
    hist1.addOne(4);
    hist1.addOne(4);
    
    // Copy the histogram
    EfficientEntropyHistogram hist2(hist1);
    
    // Check if the two are equal
    ASSERT_EQ(hist1.at(0), hist2.at(0));
    ASSERT_EQ(hist1.at(1), hist2.at(1));
    ASSERT_EQ(hist1.at(2), hist2.at(2));
    ASSERT_EQ(hist1.at(3), hist2.at(3));
    ASSERT_EQ(hist1.at(4), hist2.at(4));
    ASSERT_EQ(hist2.getSize(), 5);
    ASSERT_FLOAT_EQ(hist1.getMass(), hist2.getMass());
    ASSERT_FLOAT_EQ(hist1.getEntropy(), hist2.getEntropy());
    
    // Check if everything has been reallocated
    hist1.addOne(4);
    ASSERT_NE(hist1.at(4), hist2.at(4));
}

TEST(EfficientEntropyHistogram, assignmentOperator)
{
    // Check if everything is copied correctly
    EfficientEntropyHistogram hist1(5);
    
    // Fill the histogram
    hist1.addOne(0);
    hist1.addOne(1);
    hist1.addOne(1);
    hist1.addOne(2);
    hist1.addOne(2);
    hist1.addOne(2);
    hist1.addOne(3);
    hist1.addOne(3);
    hist1.addOne(3);
    hist1.addOne(3);
    hist1.addOne(4);
    hist1.addOne(4);
    hist1.addOne(4);
    hist1.addOne(4);
    hist1.addOne(4);
    hist1.addOne(4);
    
    // Copy the histogram
    EfficientEntropyHistogram hist2 = hist1;
    
    // Check if the two are equal
    ASSERT_EQ(hist1.getSize(), hist2.getSize());
    ASSERT_EQ(hist1.at(0), hist2.at(0));
    ASSERT_EQ(hist1.at(1), hist2.at(1));
    ASSERT_EQ(hist1.at(2), hist2.at(2));
    ASSERT_EQ(hist1.at(3), hist2.at(3));
    ASSERT_EQ(hist1.at(4), hist2.at(4));
    ASSERT_EQ(hist2.getSize(), 5);
    ASSERT_FLOAT_EQ(hist1.getMass(), hist2.getMass());
    ASSERT_FLOAT_EQ(hist1.getEntropy(), hist2.getEntropy());
    
    // Check if everything has been reallocated
    hist1.addOne(4);
    ASSERT_NE(hist1.at(4), hist2.at(4));
}

TEST(EfficientEntropyHistogram, initConstructor_1)
{
    EfficientEntropyHistogram hist1(5);
    
    // Check if the two are equal
    ASSERT_EQ(hist1.at(0), 0);
    ASSERT_EQ(hist1.at(1), 0);
    ASSERT_EQ(hist1.at(2), 0);
    ASSERT_EQ(hist1.at(3), 0);
    ASSERT_EQ(hist1.at(4), 0);
    ASSERT_EQ(hist1.getSize(), 5);
    ASSERT_FLOAT_EQ(hist1.getMass(), 0.0f);
    ASSERT_FLOAT_EQ(hist1.getEntropy(), 0.0f);
}

TEST(EfficientEntropyHistogram, initConstructor_2)
{
    EfficientEntropyHistogram hist1;
    
    // Check if the two are equal
    ASSERT_EQ(hist1.getSize(), 0);
    ASSERT_FLOAT_EQ(hist1.getMass(), 0.0f);
    ASSERT_FLOAT_EQ(hist1.getEntropy(), 0.0f);
}

TEST(EfficientEntropyHistogram, resize_toZero)
{
    EfficientEntropyHistogram hist(5);
    
    ASSERT_EQ(hist.getSize(), 5);
    
    hist.resize(0);
    
    ASSERT_EQ(hist.getSize(), 0);
    ASSERT_FLOAT_EQ(hist.getMass(), 0.0f);
    ASSERT_FLOAT_EQ(hist.getEntropy(), 0.0f);
}

TEST(EfficientEntropyHistogram, resize_toN)
{
    EfficientEntropyHistogram hist;
    
    ASSERT_EQ(hist.getSize(), 0);
    ASSERT_FLOAT_EQ(hist.getMass(), 0.0f);
    ASSERT_FLOAT_EQ(hist.getEntropy(), 0.0f);
    
    hist.resize(5);
    
    ASSERT_EQ(hist.at(0), 0);
    ASSERT_EQ(hist.at(1), 0);
    ASSERT_EQ(hist.at(2), 0);
    ASSERT_EQ(hist.at(3), 0);
    ASSERT_EQ(hist.at(4), 0);
    ASSERT_EQ(hist.getSize(), 5);
    ASSERT_FLOAT_EQ(hist.getMass(), 0.0f);
    ASSERT_FLOAT_EQ(hist.getEntropy(), 0.0f);
}

TEST(EfficientEntropyHistogram, reset)
{
    EfficientEntropyHistogram hist(5);
    
    hist.addOne(1);
    
    ASSERT_EQ(hist.at(0), 0);
    ASSERT_EQ(hist.at(1), 1);
    ASSERT_EQ(hist.at(2), 0);
    ASSERT_EQ(hist.at(3), 0);
    ASSERT_EQ(hist.at(4), 0);
    ASSERT_EQ(hist.getSize(), 5);
    ASSERT_FLOAT_EQ(hist.getMass(), 1.0f);
    
    hist.reset();
    
    ASSERT_EQ(hist.at(0), 0);
    ASSERT_EQ(hist.at(1), 0);
    ASSERT_EQ(hist.at(2), 0);
    ASSERT_EQ(hist.at(3), 0);
    ASSERT_EQ(hist.at(4), 0);
    ASSERT_EQ(hist.getSize(), 5);
    ASSERT_FLOAT_EQ(hist.getMass(), 0.0f);
    ASSERT_FLOAT_EQ(hist.getEntropy(), 0.0f);
}

TEST(EfficientEntropyHistogram, at_invalidIndex_1)
{
    EfficientEntropyHistogram hist(5);
    
    ASSERT_THROW(hist.at(-1), AssertionException);
}

TEST(EfficientEntropyHistogram, at_invalidIndex_2)
{
    EfficientEntropyHistogram hist(5);
    
    ASSERT_THROW(hist.at(6), AssertionException);
}

TEST(EfficientEntropyHistogram, addOne_invalidIndex_1)
{
    EfficientEntropyHistogram hist(5);
    
    ASSERT_THROW(hist.addOne(-1), AssertionException);
}

TEST(EfficientEntropyHistogram, addOne_invalidIndex_2)
{
    EfficientEntropyHistogram hist(5);
    
    ASSERT_THROW(hist.addOne(6), AssertionException);
}

TEST(EfficientEntropyHistogram, addOne_correctCalculations)
{
    EfficientEntropyHistogram hist(5);
    
    hist.addOne(1);
    hist.addOne(1);
    hist.addOne(1);
    hist.addOne(2);
    hist.addOne(2);
    
    ASSERT_EQ(hist.at(1), 3);
    ASSERT_EQ(hist.at(2), 2);
    ASSERT_FLOAT_EQ(hist.getMass(), 5.0f);
    ASSERT_NEAR(hist.getEntropy(), 4.8548f, 1e-3f);
}

TEST(EfficientEntropyHistogram, subOne_invalidIndex_1)
{
    EfficientEntropyHistogram hist(5);
    
    ASSERT_THROW(hist.subOne(-1), AssertionException);
}

TEST(EfficientEntropyHistogram, subOne_invalidIndex_2)
{
    EfficientEntropyHistogram hist(5);
    
    ASSERT_THROW(hist.subOne(6), AssertionException);
}

TEST(EfficientEntropyHistogram, subOne_emptyBin)
{
    EfficientEntropyHistogram hist(5);
    
    ASSERT_THROW(hist.subOne(4), AssertionException);
}

TEST(EfficientEntropyHistogram, subOne_correctCalculations)
{
    EfficientEntropyHistogram hist(5);
    
    hist.addOne(1);
    hist.addOne(1);
    hist.addOne(1);
    hist.addOne(1);
    hist.addOne(2);
    hist.addOne(2);
    hist.subOne(1);
    
    ASSERT_EQ(hist.at(1), 3);
    ASSERT_EQ(hist.at(2), 2);
    ASSERT_FLOAT_EQ(hist.getMass(), 5.0f);
    ASSERT_NEAR(hist.getEntropy(), 4.8548f, 1e-3f);
}

TEST(EfficientEntropyHistogram, isPure_empty)
{
    EfficientEntropyHistogram hist(5);
    
    ASSERT_TRUE(hist.isPure());
}

TEST(EfficientEntropyHistogram, isPure_singleNonEmptyBin)
{
    EfficientEntropyHistogram hist(5);
    
    hist.addOne(4);
    
    ASSERT_TRUE(hist.isPure());
}

TEST(EfficientEntropyHistogram, isPure_multipleNonEmptyBin)
{
    EfficientEntropyHistogram hist(5);
    
    hist.addOne(4);
    hist.addOne(3);
    
    ASSERT_FALSE(hist.isPure());
}
