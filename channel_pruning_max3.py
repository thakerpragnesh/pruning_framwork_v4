#!/usr/bin/env python
# coding: utf-8
"""
Iterative structured channel (filter) pruning using the "Max3" saliency
criterion: in each conv layer, the output channels whose kernels' summed
top-3-magnitude weights are smallest are considered least important and are
pruned first (see facilitate_pruning.compute_max3_saliency_score_channel).
Each round prunes the *current* (already pruned + fine-tuned) model, then
fine-tunes the resulting smaller network, so pruning decisions in round N+1
are based on the weights actually trained in round N.
"""
import facilitate_pruning as fp
from pruning_app import PruningEnvironment

env = PruningEnvironment(save_prefix='vgg16_IntelIc_Prune')

if __name__ == '__main__':
    env.run(fp.compute_max3_saliency_score_channel)
