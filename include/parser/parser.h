#ifndef PARSER_PARSER_H
#define PARSER_PARSER_H

/**
 * This file contains the definition of the parser. The entire pipeline works
 * as follows:
 * Input:  Image of a cabinet and region of interest
 * 
 * Pipeline: 
 *         1) Rectify the region of interest
 *         2) Apply the edge detection random forest to each pixel in order to
 *            get an edge map
 *         3) Find horizontal and vertical lines using the probabilisitc Hough 
 *            transform on the edge map
 *         4) Find rectangles by random sampling 
 *         5) Compute the segmentation using rjMCMC Simulated Annealing
 *         6) Classify the found parts
 * 
 * Output: A list of pairs: <rectangle, class label> that corresponds to the 
 *         segmentation.
 */

#include "types.h"
#include "processing.h"
#include "libforest/libforest.h"
#include "energy.h"
#include <vector>
#include <utility>
#include <Eigen/Sparse>


/**
 * Detailed command line displays
 */
#define DEBUG_MODE_ON 0

/**
 * Use depth information or not
 */
#define INCLUDE_DEPTH 1

/**
 * Quick Convergence Configuration
 */
#define DUMMY_MCMC_LOGIC 0


/**
 * Corrupting Bounding Box Experiment
 */
#define ROI_CORRUPTION 0

#define CORRUPT_PIXELS 5 // extent of corruption

/**
 * Whether to augment the proposal pool or not
 */
#define SPLIT_MERGE_AUGMENT 1

/**
 * Split augment parameter
 */
#define PROJ_PROF_THRESH 160

/**
 * Allowance(in pixels) to merge rectangles in MERGE move
 */
#define MERGE_ALLOWANCE 20 // Should be more for natural images

/**
 * Initial probability of each (reversible) move
 */

#define INIT_PROB_BIRTH 0.20f
#define INIT_PROB_DEATH 0.02f
#define INIT_PROB_SPLIT 0.0f
#define INIT_PROB_MERGE 0.00f
#define INIT_PROB_LABEL_DIFFUSE 0.0f
#define INIT_PROB_EXCHANGE_DATADRIVEN 0.0f
#define INIT_PROB_UPDATE_CENTER 0.00f
#define INIT_PROB_UPDATE_WIDTH 0.00f
#define INIT_PROB_UPDATE_HEIGHT 0.00f
#define INIT_PROB_EXCHANGE_RANDOM (1.0f - INIT_PROB_BIRTH - INIT_PROB_DEATH- INIT_PROB_SPLIT - INIT_PROB_MERGE - INIT_PROB_EXCHANGE_DATADRIVEN - INIT_PROB_UPDATE_CENTER - INIT_PROB_UPDATE_WIDTH - INIT_PROB_UPDATE_HEIGHT - INIT_PROB_LABEL_DIFFUSE)

/**
 * Simulated Annealing Parameters (Default Architecture)
 */

#define MAX_TEMP 100
#define MIN_TEMP 1e-2
#define ALPHA 0.995f
#define MAX_UPDATE_ITER 2500
#define NUM_INNER_LOOPS 1000



/**
 * Simulated Annealing Parameters (2Phase Architecture)
 */

#define MAX_TEMP_WARMUP 100
#define MIN_TEMP_WARMUP 1e-1
#define ALPHA_WARMUP 0.95f
#define MAX_UPDATE_ITER_WARMUP 500
#define NUM_INNER_LOOPS_WARMUP 100

#define MAX_TEMP_MASTER 10
#define MIN_TEMP_MASTER 1e-2
#define ALPHA_MASTER 0.995f
#define MAX_UPDATE_ITER_MASTER 2500
#define NUM_INNER_LOOPS_MASTER 100


/**
 * move indices
 */
#define EXCHANGE_MOVE_IDX 0
#define BIRTH_MOVE_IDX 1
#define DEATH_MOVE_IDX 2
#define SPLIT_MOVE_IDX 3
#define MERGE_MOVE_IDX 4
#define LABEL_DIFFUSE_MOVE_IDX 5
#define EXCHANGE_DD_MOVE_IDX 6
#define UPDATE_CENTER_MOVE_IDX 7
#define UPDATE_WIDTH_MOVE_IDX 8
#define UPDATE_HEIGHT_MOVE_IDX 9

/**
 * choose: data driven birth or random birth
 */
#define DATA_DRIVEN_BIRTH_DEATH 0

/**
 * extend of diffusion in update moves
 */
#define WH_DIFFUSE_FACTOR 2.5f
#define CENTER_DIFFUSE_FACTOR 5.0f

/**
 * choose: weighted sampling or not
 */
#define ROULETTE_EXCHANGE 1
#define ROULETTE_DEATH 0
/**
 * choose: optimum rectangle clusters
 */
#define OPTIMUM_RECTS 10
/**
 * Redundancy removal IOU threshold
 */
#define CLUSTER_MAX_IOU 0.95f
/**
 * Bound for appearance likelihood
 */
#define numPartsCBUpperBound 400
/**
 * Overlap Parameter
 */
#define MAX_OVERLAP 0.15f
/**
 * Augmentation in Appearance
 */
#define APPEARANCE_AUGMENT 0
/**
 * If this is true, then the parser generates debug output during the parsing
 * process
 */
#define VERBOSE_MODE 1
/**
 * This is the size of the patches that are used for the edge detector
 */
#define PATCH_SIZE 17
/**
 * Steepness of Probabilistic SVM Sigmoid
 */
#define STEEPNESS 2.0f
/**
 * The number of channels used for the edge detector
 */
#define EDGE_DETECTOR_CHANNELS 4
typedef cv::Vec<float, EDGE_DETECTOR_CHANNELS> EdgeDetectorVec;

/**
 * Depth Rectification boundary jitter
 */
#define JITTERFACTOR 5

/**
 * The channels
 */
#if 0

#define EDGE_DETECTOR_CHANNEL_L 0
#define EDGE_DETECTOR_CHANNEL_U 1
#define EDGE_DETECTOR_CHANNEL_V 2
#define EDGE_DETECTOR_CHANNEL_INTENSITY 3
#define EDGE_DETECTOR_CHANNEL_XDERIV 4
#define EDGE_DETECTOR_CHANNEL_YDERIV 5
#define EDGE_DETECTOR_CHANNEL_GM 6

#else

#define EDGE_DETECTOR_CHANNEL_INTENSITY 0
#define EDGE_DETECTOR_CHANNEL_GM 1
#define EDGE_DETECTOR_CHANNEL_XDERIV 2
#define EDGE_DETECTOR_CHANNEL_YDERIV 3

#endif

struct Vec3f
{
    float v[3];

    Vec3f() {}
    Vec3f(float x, float y, float z)
    {
        v[0] = x; v[1] = y; v[2] = z;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Print Matrix
 */

static Vec3f operator +(const Vec3f &a, const Vec3f &b) { return Vec3f(a.v[0] + b.v[0], a.v[1] + b.v[1], a.v[2] + b.v[2]); }
static Vec3f operator -(const Vec3f &a, const Vec3f &b) { return Vec3f(a.v[0] - b.v[0], a.v[1] - b.v[1], a.v[2] - b.v[2]); }
static Vec3f operator *(float s, const Vec3f &a)        { return Vec3f(s * a.v[0], s * a.v[1], s * a.v[2]); }

static Vec3f &operator -=(Vec3f &a, const Vec3f &b)     { a.v[0] -= b.v[0]; a.v[1] -= b.v[1]; a.v[2] -= b.v[2]; return a; }

static float dot(const Vec3f &a, const Vec3f &b)        { return a.v[0]*b.v[0] + a.v[1]*b.v[1] + a.v[2]*b.v[2]; }
static Vec3f normalize(const Vec3f &in)                 { return (1.0f / sqrtf(dot(in, in))) * in; }


struct Mat33f
{
    Vec3f col[3];
};
static void print_mat(const char *name, const Mat33f &mat)
{
    printf("%s=[\n", name);
    for(int i=0; i < 3; i++) {
        for(int j=0; j < 3; j++)
            printf(" %10.6f%c", mat.col[j].v[i], (j == 2) ? ';' : ',');

        printf("\n");
    }
    printf("];\n");
}
//////////////////////////////////////////////////////////////////////////////
namespace parser {
    
    /**
     * This class holds the segmentation of an image. 
     */
    class Segmentation {
    public:
        /**
         * The region of interest
         */
        Rectangle regionOfInterest;
        /**
         * The functional parts
         */
        std::vector<Rectangle> parts;
        /**
         * The part labels
         */
        std::vector<int> labels;
        /**
         * The actual image file
         */
        std::string file;
        /**
         * The image id
         */
        std::string id;
        
        /**
         * Loads the segmentation from the annotation file.
         */
        void readAnnotationFile(const std::string & filename);
        
        /**
         * Loads the region of interest from the auxiliary file.
         */
        void readAuxiliaryFile(const std::string & filename);
    };
    
    /**
     * This class represents a single part. 
     */
    class Part {
    public:
        Part() : label(0), likelihood(0.0f), meanDepth(0.0f), shapePrior(0.0f), posterior(0.0f) {}
        
        /**
         * The rectified rectangle
         */
        Rectangle rect;
        /**
         * The class label
         */
        int label;
        /**
         * The visual weight p(P|I) 
         */
        float posterior;
        /**
        * The appearance likelihood
        */
        float likelihood;
        /**
         * The shape Prior Probability 
         */
        float shapePrior;
        /**
         * The mean Depth value 
         */
        float meanDepth;
        /**
         * The mean Depth value
         */
        std::vector<int> projProf;
        /**
         * The mean Depth value
         */
        std::vector<int> projProfTyp;
    };
    
    /**
     * This class parses an image and returns the segmentation.
     */

    class CabinetParser {
    public:

        /**
         * Computes the segmentation of the image given the region of interest
         */

        void parse(const cv::Mat & image, const cv::Mat & imageDepth, const Rectangle & region, std::vector<Part> & parts);
        
        /**
         * Trains the parser on the given data.
         */
        void train(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Trains the parser on the images of a given directory.
         */
        void train(const std::string & directory);
        
        /**
         * Evaluates the performance of individual components on these images
         */
        void test(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Evaluates the performance of individual components on the images
         * from a directory.
         */
        void test(const std::string & directory);
        
        /**
         * Generates the training data for arbitrary edge detectors:
         * 1. The rectified image
         * 2. The rectified edge map
         */
        void createGeneralEdgeDetectorSet(const std::string & inputDirectory, const std::string & imageOutputDirectory, const std::string & groundTruthDirectory);
        
        /**
         * Visualizes a segmentation
         */
        void visualizeSegmentation(const cv::Mat & image, const Rectangle & ROI, const std::vector<Part> & parts, cv::Mat & display);
        
        /**
         * Exports the relative edge distributions for the GT rectangles.
         */
        void exportEdgeDistributionFromGT(const cv::Mat & image, const Segmentation & segmentation);
        
        /**
         * Evaluates the performance of individual components on these images
         */
        void exportRectifiedImages(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Evaluates the performance of individual components on the images
         * from a directory.
         */
        void exportRectifiedImages(const std::string & directory);
        
        /**
         * Computes the final precision/recall curve for the semantic segmentation
         */
        void computePrecisionRecallCurve(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Visualizes an appearance distribution
         */
        void visualizeAppearanceDistribution(const libf::DataPoint & p, cv::Mat & img);
        
        /**
         * Visualizes a floating point image
         */
        void visualizeFloatImage(const cv::Mat & img) const;
        
        /**
         * Removes the pixels from the edge map that were not used for rectangle
         * detection
         */
        void removeNonRectanglePixels(const std::vector<Rectangle> & rectangles, cv::Mat & edgeMap);
        
        /**
         * Removes the pixels from the edge map that were not used for rectangle
         * detection
         */
        void removeNonRectanglePixels2(const std::vector<Rectangle> & rectangles, cv::Mat & edgeMap);
        
        /**
         * Removes the pixels from the edge map that were not used for rectangle
         * detection
         */
        void removeNonRectanglePixels3(const std::vector<Rectangle> & rectangles, cv::Mat & edgeMap);
        
        /**
         * Plots the rectified edge map for an annotated image. 
         */
        void plotRectifiedEdgeMap(const cv::Mat & image, const Segmentation & segmentation, cv::Mat & edges);
        
        /**
         * Plots the rectified segmentation for an annotated image. 
         */
        void plotRectifiedSegmentation(const cv::Mat & image, const Segmentation & segmentation, cv::Mat & segImg);
        
        /**
         * Extracts the multi channel image from the region of interest. 
         */
        void extractRectifiedMultiChannelImage(const cv::Mat & image, const Rectangle & region, cv::Mat & out);
        
        /**
         * Visualizes the multi channel image.
         */
        void visualizeMultiChannelImage(const cv::Mat & image) const;
        
        /**
         * Extracts an image patch from the multi channel image. Pixels outside the
         * image get a value of 0.
         */
        void extractPatch(const cv::Mat & multiChannelImage, const cv::Vec2i & point, libf::DataPoint & dataPoint, int orientation);

        void extractPatchFlipped(const cv::Mat & multiChannelImage, const cv::Vec2i & point, libf::DataPoint & dataPoint);
        
        /**
         * One time computation of all proposal based matrices
         */
        void computeProposalMatrices(const std::vector<Rectangle> proposals, const float imageArea, Eigen::MatrixXi & overlapPairs, Eigen::MatrixXi & overlapPairs50,
                std::vector<float> & areas, Eigen::MatrixXf & overlapArea, Eigen::MatrixXi & widthMergeable, Eigen::MatrixXi & heightMergeable);
        /**
         * Loads annotated images from a directory.
         */
        void loadImage(const std::string & directory, std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Extracts the patches for the edge detector for training/test
         */
        void extractEdgeDetectorPatches(libf::DataStorage::ptr trainingSet, const std::vector< std::pair<cv::Mat, Segmentation > > & images);
        
        /**
         * Trains the edge detecting forest on a set of images and their annotations. 
         */
        void trainEdgeDetector(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Tests the edge detecting forest on a set of images and their annotations. 
         */
        void testEdgeDetector(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Trains the part prior model. It's a score on the size of the part.
         */
        void trainPartPrior(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);

        /**
         * SVM prior model
         */
        void probSVMPrior(libf::DataStorage::ptr dataStorage);

        /**
         * Test bench for SVM prior model
         */
        void evaluateProbSVMPrior(libf::DataStorage::ptr dataStorage);

        /**
         * Train probabilistic SVM using sigmoid
         */
        void trainSVM(cv::Mat & trainDataSVM, int partCount[3]);
        
        /**
         * Applies the learned edge detector to the multi channel image
         */
        void applyEdgeDetector(const cv::Mat & multiChannelImage, cv::Mat & out, int depthFlag);
        
        /**
         * Maps a set of rectangles in the original image to their rectified
         * form.
         */
        void rectifyParts(const Rectangle & regionOfInterest, const std::vector<Rectangle> & partsIn, std::vector<Rectangle> & partsOut);
        
        /**
         * Maps a set of rectangles in the rectified image to their original
         * form.
         */
        void unrectifyParts(const Rectangle & regionOfInterest, const std::vector<Rectangle> & partsIn, std::vector<Rectangle> & partsOut);
        
        /**
         * Performs line detection on a binary image using the probabilistic Hough
         * transform. 
         */
        void detectLines(const cv::Mat & image, std::vector<LineSegment> & resultH, std::vector<LineSegment> & resultV);
        
        /**
         * Performs rectangle detection on a binary image using a random sampling
         * strategy.
         */
        void detectRectangles(const cv::Mat & image, const cv::Mat & cannyEdges, std::vector<Rectangle> & result);
        
        /**
         * Selects a subset of the hypotheses rectangles are parts.
         */
        void selectParts(const cv::Mat & mcImage, const cv::Mat & depthImage, const cv::Mat & edgeImage, std::vector<Rectangle> & hypotheses, std::vector<Part> & result);
        
        /**
         * Extracts the discretized kernel distributions for the appearance.
         */
        void extractDiscretizedAppearanceData(const cv::Mat & rectifiedEdgeImage, const Rectangle & part, libf::DataPoint & p);
        
        /**
         * Extracts the discretized kernel distributions for the appearance.
         */
        void extractDiscretizedAppearanceDataGM(const cv::Mat & gradMag, const Rectangle & part, libf::DataPoint & p, libf::DataPoint& p2);
        
        /**
         * Extracts the edge projection profile of each part
         */
        void extractEdgeProjectionProfile(const cv::Mat& gradMag, const Rectangle & part, std::vector<int> & edgeProjProfile, std::vector<int> & indexWH);

        /**
         * Split Augmentation: Width then Height
         */
        void augmentRectanglesSplitWH(const cv::Mat gradMag, std::vector<Rectangle> & hypotheses, const double imageArea);
        /**
         * Split Augmentation: Height then Width
         */
        void augmentRectanglesSplitHW(const cv::Mat gradMag, std::vector<Rectangle> & hypotheses, const double imageArea);
        /**
         * Merge Augmentation: Width then Height
         */
        void augmentRectanglesMergeWH(std::vector<Rectangle> & hypotheses);
        /**
         * Merge Augmentation: Height then Width
         */
        void augmentRectanglesMergeHW(std::vector<Rectangle> & hypotheses);
        /**
         * Augmentation: Redundancy removal
         */
        void removeRedundantRects(std::vector<Rectangle> & hypotheses, const float maxIOU_thresh);
        /**
         * Extracts the discretized kernel distributions for the appearance.
         */
        void extractDiscretizedAppearanceDistributions(libf::DataStorage::ptr trainingSet, const std::vector< std::tuple<cv::Mat,  Segmentation, cv::Mat > > & images);
        
        /**
         * Trains the appearance codebook
         */
        void trainAppearanceCodeBook(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Trains the appearance codebook
         */
        void testAppearanceCodeBook(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Prunes the set of detected rectangles using the packing/cover numbers. 
         */
        void pruneRectangles(float roiSize, const std::vector<Rectangle> & rectangles, int & packingNumber, int & coverNumber, std::vector<Rectangle> & preselection);
        
        /**
         * Creates the ground truth edge map
         */
        void createGroundtruthEdgeMap(const cv::Mat & image, const Segmentation & segmentation, cv::Mat & edges);
        
        /**
         * Evaluates the edge detector
         */
        void evaluateEdgeDetector(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Evaluates the rectangle detector
         */
        void evaluateRectangleDetector(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Evaluates the rectangle pruning
         */
        //void evaluateRectanglePruning(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Determines the values of the latent variables for a given codebook 
         * and an observed distribution. 
         */
        void determineLatentVariables(const Eigen::MatrixXf & codebook, const Eigen::MatrixXf & x, Eigen::MatrixXf & pi);
        
        /**
         * Learns a codebook
         */
        void learnCodebook(libf::AbstractDataStorage::ptr storage, Eigen::MatrixXf & codebook, int K);
        
        /**
         * Returns the error of fitting x to the codebook
         */
        float calcCodebookError(const Eigen::MatrixXf & codebook, const Eigen::VectorXf & x);
        
        /**
         * Extracts detected rectangles with their labels (part or background).
         */
        void extractRectangleData(libf::DataStorage::ptr dataStorage, const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Rectifies the depth using biliear interpolation
         */
        void rectifyDepthBilinear(const cv::Mat & unRectifiedImage, cv::Mat & rectifiedImage);
        void rectifyDepth(const cv::Mat & unRectifiedRGB, const cv::Mat & unRectifiedDepth, cv::Mat & rectifiedImage);
        
        /**
         * Compute the Orthogonal coordinate system
         */
        void computeCordSys(Eigen::Matrix3f &transform_MGS, Mat33f &main3normals);

        /**
         * Gram Schmidt orthonormalization
         */
        void modified_gram_schmidt(Mat33f &out, const Mat33f &in);

        /**
         * Transform (Rotate and translate) the point cloud
         */
        void transformPCL(Eigen::Matrix3f &transform_GM);

        /**
         * Extract mean depth of each part
         */
        float extractMeanPartDepth(const cv::Mat & rectifiedDepth, const Rectangle rectifiedRect);

        /**
         * Evaluates the segmentation
         */
        void evaluateSegmentation(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
        /**
         * Extracts the rectified appearances as images
         */
        //void extractPartAppearances(const std::vector< std::tuple<cv::Mat, Segmentation, cv::Mat > > & images);
        
    public:
        /**
         * This class captures all the parameters that can be tuned. 
         */
        class Parameters {
        public:
            Parameters() : rectifiedROISize(500) {}
            
            /**
             * This is the size of the rectified regions of interest
             */
            int rectifiedROISize;
        };
        
        Parameters parameters;
    };
}
#endif
