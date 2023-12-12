# WORKFLOW FOR TRAINING

# Our feature extractor (E) gets source domain images and downsampled target domain images
## Options for E:
# - ASPPnet 
#   + Works good with multiple scales
#   - Might lose sharp details
# - U-Net down section thingy
#   + Works well with less data. Works well to learn sharp details
#   - Encoder, have to deal with skip connections. Not proven for domain adaptation, have to experiment on multiple scales. 
# - DeepLabv3+ thingy, using ASPPnet instead of SPP
#   + Tries to tackle by combining the advantages of both methods: multiscale contextual information and sharper boundaries.
#   - Encoder, have to deal with skip connections. Not proven for domain adaptation, have to experiment.

# 