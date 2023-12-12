# WORKFLOW FOR TRAINING

# Our feature extractor (E) gets source domain images (192,192) slices and downsampled target domain images (384,384) ↓ (192,192)
# * The size ratio (second MRI / first MRI) for each dimension (x, y, z) are actually (1.7179487179487178, 1.7169811320754718, 1.7142857142857142) 
# Given the dimensions of our first domain is (117,159,126)
# And the dimensions of second domain are (201,273,216)
# Current model adapts to 192x192. Given the nature of pixelshuffle later used, we decide on doing x2 to (384,384) instead of individual ratios ≈ x1.71.

### 1 - Module E ### 

## We input here images with dimensionality (192,192).

## Options for Module E:
# - ASPPnet 
#   + Works good with multiple scales
#   - Might lose sharp details
# - U-Net down section thingy
#   + Works well with less data. Works well to learn sharp details
#   - Encoder, have to deal with skip connections. Not proven for domain adaptation, have to experiment on multiple scales. 
# - DeepLabv3+ thingy, using ASPPnet instead of SPP
#   + Tries to tackle by combining the advantages of both methods: multiscale contextual information and sharper boundaries.
#   - Encoder, have to deal with skip connections. Not proven for domain adaptation, have to experiment.

## We would get apriori a feature map of half the source domain size, being (96,96).

### 2 ### 

## We use the feature maps from step 1 as inputs.

### 2.1 - Module R ###

## Here we are doing a little Generative Adversial Training

## Options
# - PixelShuffle: probably going for this one for the Feature Pyramid
# - Some diffusion thingy
# - Generator

# Pixel-level discriminator → we input the source composite images and the target domain images
## Five Convolutional Layers → {64,128,256,512,1}, except for the last one, each followed by a leaky ReLU

# We produce high resolution target domain with detailed features, having learned the style specifics


### 2.2 - Module S Part 1 ###

## The actual segmentation module

## Options
# - Three convolutional layers with kernels 3x3x3 and 9x9, with a softmax at the end → Feature Pyramid Network for Object Detention
# - Adapt the upper going U-Net thingy to accept the lateral connections for multiscalling and sharp detection (?)

# * ?? Could we connect the ASPPnet to the pyramid and then run that throught a u net or smt ??

### 3 - Module S Part 2 ###

## We use labelled source domain images andcomposite images for supervised learning

## We use unlabelled target domain images for self supervised learning

## We perform label correction on source domain labels based on segmentation results of the composite images 

## Feature Distribution of both domains are further aligned by training agains a Output-Space Domian Discriminator. 
## Where we input the segmentation predictions of both domains.