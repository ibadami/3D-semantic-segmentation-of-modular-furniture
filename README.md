The code was tested on Ubuntu 14.04. 

Dependencies: eigen, PCL, Gurobi, OpenCV
--------------------------COMPILING----------------------------------
Following two commands to be run in the terminal from the build directory:
cmake -DCMAKE_BUILD_TYPE=Release ..
make

--------------------------RUNNING THE EXE----------------------------------
Run the following command line from the build folder

Training:
for eg: train with all images in the folder '../data/depth/160/crossValidate/set4/train/'
./bin/cli train ../data/depth/160/crossValidate/set4/train/
All the training models must be stored in the build directory before testing or parsing
Each training can be separately enabled in parser.cpp
1) trainEdgeDetector
2) trainPartPrior
3) trainAppearanceCodeBook



Parsing a single image:
for example to parse an image named '728.JPG' in the folder ../data/depth/160/crossValidate/set4/test
./bin/cli parse ../data/depth/160/crossValidate/set4/test/ 728
the result image will be stored with the name 'last_result.png' in the build directory




Testing:
for eg: test on all images in the folder '../data/depth/160/crossValidate/set4/test/'
./bin/cli test ../data/depth/160/crossValidate/set4/test/
The image results will be in 'results' folder in build directory and quantitative results in 'results.txt' in the build directory

--------------------------PARAMETER SETTINGS----------------------------------
the very critical parameters are RDT and maxIOU

RDT = 6 for WACV 17 synthetic dataset
RDT = 1 for WACV 17 real kinect dataset
RDT = 3 for WACV 16 dataset
RDT (rectangleDetectionThreshold) can be adjusted in parser.cpp


maxIOU = 0.95, can be adjusted in detector.h 
--------------------------PARSER SETTINGS----------------------------------
Adjust in the header file parser.h

Following are the options:

DEBUG_MODE_ON:  if 1, displays the debugging outputs in the command prompt

INCLUDE_DEPTH: if 1, includes depth information in the whole pipeline

DUMMY_MCMC_LOGIC: if 1, enables the 2 phase architecture and deactivtes the default architecture

ROI_CORRUPTION: if 1 enables corrpting the Region of interest bounding box, the extent of corruption
 (in pixels) can be specified using CORRUPT_PIXELS

SPLIT_MERGE_AUGMENT: if 1, enables proposal pool (split and merge) augmentation 
The split augment parameter can be specified using PROJ_PROF_THRESH
and merging allowance (in pixels) can be specified using MERGE_ALLOWANCE


/**
 * Initial probability of each (reversible) move can be specified using the following
 */

INIT_PROB_BIRTH
INIT_PROB_DEATH
INIT_PROB_SPLIT
INIT_PROB_MERGE
INIT_PROB_LABEL_DIFFUSE: switch move
INIT_PROB_EXCHANGE_DATADRIVEN: data driven exchange
INIT_PROB_UPDATE_CENTER
INIT_PROB_UPDATE_WIDTH
INIT_PROB_UPDATE_HEIGHT
INIT_PROB_EXCHANGE_RANDOM: random exchange

/**
 * Simulated Annealing Parameters (Default Architecture)
 */

MAX_TEMP // Starting temperature
MIN_TEMP // ending temperature
ALPHA // Geometric cooling parameter
MAX_UPDATE_ITER // To check convergence
NUM_INNER_LOOPS // iterations at each temperature 



/**
 * Simulated Annealing Parameters (2Phase Architecture)
 */

//Phase 1 parameters
MAX_TEMP_WARMUP
MIN_TEMP_WARMUP
ALPHA_WARMUP
MAX_UPDATE_ITER_WARMUP
NUM_INNER_LOOPS_WARMUP

//Phase 2 parameters
MAX_TEMP_MASTER
MIN_TEMP_MASTER
ALPHA_MASTER
MAX_UPDATE_ITER_MASTER
NUM_INNER_LOOPS_MASTER


/**
 * move indices
 */
EXCHANGE_MOVE_IDX
BIRTH_MOVE_IDX
DEATH_MOVE_IDX
SPLIT_MOVE_IDX
MERGE_MOVE_IDX
LABEL_DIFFUSE_MOVE_IDX
EXCHANGE_DD_MOVE_IDX
UPDATE_CENTER_MOVE_IDX
UPDATE_WIDTH_MOVE_IDX
UPDATE_HEIGHT_MOVE_IDX


/**
 * extend of diffusion in update moves
 */
WH_DIFFUSE_FACTOR : for diffusing width and height
CENTER_DIFFUSE_FACTOR : for diffusing center location

/**
 * choose: weighted sampling or not
 */
ROULETTE_EXCHANGE: If 1, weighted sampling (russian roulette) is enabled

/**
 * Redundancy removal IOU threshold
 */
CLUSTER_MAX_IOU // measure to check similarity of rectangles in redundancy removal (proposal pool augmentation)

/**
 * Upper Bound for appearance likelihood
 */
numPartsCBUpperBound // Only if running in low specification systems
/
**
 * Maximum permitted Overlap above which penlising happens
 */
MAX_OVERLAP

/**
 * Augmentation in Appearance
 */

APPEARANCE_AUGMENT
/**
 * If this is true, then the parser generates debug output during the parsing
 * process
 */

VERBOSE_MODE
/**
 * This is the size of the patches that are used for the edge detector
 */

PATCH_SIZE
/**
 * Steepness of Probabilistic SVM Sigmoid
 */

STEEPNESS
------------------------------------------------------------------------------


------------------------------------------------------------------------------

A note on shape prior: Form factor was excluded in the evaluation of Proposed approach in 3d as 7 dimensions were enough. But for 2d, to strenthen the shape prior it was includedin the trained models.
------------------------------------------------------------------------------
A note on testing on real images: The current depth rectification is done using bilinear interpolation which works perfectly for synthetic images but not the best for real images especially for the ones with large amount of transformation. The authentic rectification is to implement the incomplete module using normal computation, clustering, gram schmidt orthonormalization and then cancelling the transformation based on the computed co-ordinate system. This section is incomplete.
Recommended: RDT = 1 for real images
------------------------------------------------------------------------------
To add noise to the synthetic images use use addKinectNoise.m in scripts folder
------------------------------------------------------------------------------
To get real png images (RGB and depth) use oni2PNG.cpp

