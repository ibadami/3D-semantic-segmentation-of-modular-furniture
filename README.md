The code is tested on Ubuntu 14.04.

Download RGB-D furniture models and their annotations from [here](https://www.vision.rwth-aachen.de/page/furniture).

#### Dependencies: 
eigen, PCL, Gurobi, OpenCV

#### Compiling:
Following two commands to be run in the terminal from the build directory:

`cmake -DCMAKE_BUILD_TYPE=Release ..`  
`make`

Note: change the required directory path in CMakeList file.
#### Training:

##### Train with all the images: 
`./bin/cli train ../data/depth/160/crossValidate/set4/train/`

All the training models must be stored in the build directory before testing
Each training can be separately enabled in parser.cpp
1) trainEdgeDetector
2) trainPartPrior
3) trainAppearanceCodeBook


##### Parsing a single image:
for example to parse an image named '728.JPG' in the folder ../data/depth/160/crossValidate/set4/test

`./bin/cli parse ../data/depth/160/crossValidate/set4/test/ 728`

the resulting image will be stored with the name 'last_result.png' in the build directory.

#### Testing:

###### Test all the images:
'./bin/cli test ../data/depth/160/crossValidate/set4/test/'

The results will be saved in 'results' folder and the quantitative results in 'results.txt' in the build directory.

#### Parameter settings:
RDT(rectangleDetectionThreshold) and maxIOU are critical parameters for the segmentation.

- RDT = 6 for WACV 17 synthetic dataset
- RDT = 1 for WACV 17 real kinect dataset
- RDT = 3 for WACV 16 dataset
- maxIOU = 0.95

Note: RDT can be adjusted in parser.cpp and maxIOU can be adjusted in detector.h  
Recommended: RDT = 1 for real images


###### More parameter options

- DEBUG_MODE_ON:  if 1, displays the debugging outputs in the command prompt

- INCLUDE_DEPTH: if 1, includes depth information in the whole pipeline

- DUMMY_MCMC_LOGIC: if 1, enables the 2 phase architecture and deactivtes the default architecture

- ROI_CORRUPTION: if 1 enables corrpting the Region of interest bounding box, the extent of corruption
 (in pixels) can be specified using CORRUPT_PIXELS

- SPLIT_MERGE_AUGMENT: if 1, enables proposal pool (split and merge) augmentation 
The split augment parameter can be specified using PROJ_PROF_THRESH
and merging allowance (in pixels) can be specified using MERGE_ALLOWANCE

###### Initial probability of each (reversible) move can be specified using the following

1) INIT_PROB_BIRTH
2) INIT_PROB_DEATH
3) INIT_PROB_SPLIT
4) INIT_PROB_MERGE
5) INIT_PROB_LABEL_DIFFUSE: switch move
6) INIT_PROB_EXCHANGE_DATADRIVEN: data driven exchange
7) INIT_PROB_UPDATE_CENTER
8) INIT_PROB_UPDATE_WIDTH
9) INIT_PROB_UPDATE_HEIGHT
10) INIT_PROB_EXCHANGE_RANDOM: random exchange  

###### Simulated Annealing Parameters (Default Architecture)

1) MAX_TEMP // Starting temperature
2) MIN_TEMP // ending temperature
3) ALPHA // Geometric cooling parameter
4) MAX_UPDATE_ITER // To check convergence
5) NUM_INNER_LOOPS // iterations at each temperature 

###### Simulated Annealing Parameters (2Phase Architecture)

###### # Phase 1 parameters
1) MAX_TEMP_WARMUP
2) MIN_TEMP_WARMUP
3) ALPHA_WARMUP
4) MAX_UPDATE_ITER_WARMUP
5) NUM_INNER_LOOPS_WARMUP

###### # Phase 2 parameters
1) MAX_TEMP_MASTER
2) MIN_TEMP_MASTER
3) ALPHA_MASTER
4) MAX_UPDATE_ITER_MASTER
5) NUM_INNER_LOOPS_MASTER

###### Sampling move indices

1) EXCHANGE_MOVE_IDX
2) BIRTH_MOVE_IDX
3) DEATH_MOVE_IDX
4) SPLIT_MOVE_IDX
5) MERGE_MOVE_IDX
6) LABEL_DIFFUSE_MOVE_IDX
7) EXCHANGE_DD_MOVE_IDX
8) UPDATE_CENTER_MOVE_IDX
9) UPDATE_WIDTH_MOVE_IDX
10) UPDATE_HEIGHT_MOVE_IDX

###### Diffusion in update moves

1) WH_DIFFUSE_FACTOR : for diffusing width and height
2) CENTER_DIFFUSE_FACTOR : for diffusing center location

###### Weighted sampling

- ROULETTE_EXCHANGE: If 1, weighted sampling (russian roulette) is enabled

###### Redundancy removal IOU threshold
- CLUSTER_MAX_IOU // measure to check similarity of rectangles in redundancy removal (proposal pool augmentation)

###### Upper Bound for appearance likelihood
- numPartsCBUpperBound // Only if running in low specification systems

###### Maximum permitted Overlap above which penlising happens
- MAX_OVERLAP
###### Augmentation in Appearance
- APPEARANCE_AUGMENT: If this is true, then the parser generates debug output during the parsing process
 

###### Misc
- VERBOSE_MODE: This is the size of the patches that are used for the edge detector
- PATCH_SIZE: Steepness of Probabilistic SVM Sigmoid


Note on shape prior: Form factor was excluded in the evaluation of Proposed approach in 3d as 7 dimensions were enough. But for 2d, to strenthen the shape prior it was includedin the trained models.

Note on testing on real images: The current depth rectification is done using bilinear interpolation which works perfectly for synthetic images but not the best for real images especially for the ones with large amount of transformation. The authentic rectification is to implement the incomplete module using normal computation, clustering, gram schmidt orthonormalization and then cancelling the transformation based on the computed co-ordinate system. This section is incomplete.

To add noise to the synthetic images use use addKinectNoise.m in scripts folder
To get real png images (RGB and depth) use oni2PNG.cpp