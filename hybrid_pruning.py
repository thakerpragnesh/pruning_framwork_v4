#!/usr/bin/env python
# coding: utf-8
"""
Hybrid structured channel pruning: chains different pruning criteria across
rounds instead of applying the same one repeatedly, since a stronger
saliency criterion and a redundancy/similarity criterion tend to catch
different kinds of prunable channels. Each round prunes the *current*
(already pruned + fine-tuned) model with whichever criterion the active
stage specifies, then fine-tunes the resulting smaller network, so pruning
decisions in round N+1 are based on the weights actually trained in round N
-- exactly like the single-criterion scripts, just with `score_fn` allowed
to change between rounds (see pruning_driver.hybrid_channel_pruning).

The stage sequence below (4 rounds of Max3 saliency, then 3 rounds of
K-means similarity) substitutes for the thesis's best-found sequence ("4x
L1 saliency-channel -> 3x L1 similarity-channel -> 1x L1 saliency-kernel"):
Max3 stands in for plain L1 saliency since it was the stronger channel-level
saliency scorer, and the trailing kernel-level stage is dropped since
kernel-level pruning is out of scope for this framework right now.
"""
import functools

import facilitate_pruning as fp
from pruning_app import PruningEnvironment

env = PruningEnvironment(save_prefix='vgg16_IntelIc_Prune_hybrid')

distance_metric = env.config.get('Pruning', 'distance_metric', fallback='manhattan')
kmeans_score_fn = functools.partial(fp.compute_kmeans_similarity_score_channel, distance=distance_metric)

stages = [
    (fp.compute_max3_saliency_score_channel, 4),
    (kmeans_score_fn, 3),
]

if __name__ == '__main__':
    env.run_hybrid(stages)
