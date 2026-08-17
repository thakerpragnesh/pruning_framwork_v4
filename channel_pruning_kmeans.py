#!/usr/bin/env python
# coding: utf-8
"""
Iterative structured channel (filter) pruning using a K-means redundancy
criterion: in each conv layer, output channels are clustered by similarity
and, within each cluster, every channel except the largest-L1-norm one is
considered redundant and pruned (see
facilitate_pruning.compute_kmeans_similarity_score_channel). The distance
metric used for clustering is configurable via config.ini's
[Pruning] distance_metric (manhattan/euclidean/cosine); manhattan was the
best-performing metric in evaluation. Each round prunes the *current*
(already pruned + fine-tuned) model, then fine-tunes the resulting smaller
network, so pruning decisions in round N+1 are based on the weights
actually trained in round N.
"""
import functools

import facilitate_pruning as fp
from pruning_app import PruningEnvironment

env = PruningEnvironment(save_prefix='vgg16_IntelIc_Prune')

distance_metric = env.config.get('Pruning', 'distance_metric', fallback='manhattan')
kmeans_score_fn = functools.partial(fp.compute_kmeans_similarity_score_channel, distance=distance_metric)

if __name__ == '__main__':
    env.run(kmeans_score_fn)
