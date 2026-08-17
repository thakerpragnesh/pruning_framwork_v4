#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch

# In[2]: Channel (filter) importance by weight-magnitude criterion.
# Ranks each output channel of a conv layer by its Lp-norm and returns the
# indices of the `prune_amount` least-salient (smallest-norm) channels,
# weakest first. This is the standard magnitude/saliency pruning criterion
# (cf. Li et al., "Pruning Filters for Efficient ConvNets").
def compute_saliency_score_channel(tensor_t, n=1, prune_amount=1):
    out_channels = tensor_t.shape[0]
    prune_amount = max(0, min(prune_amount, out_channels - 1))
    channel_norm = torch.norm(tensor_t.reshape(out_channels, -1), p=n, dim=1)
    order = torch.argsort(channel_norm)
    return order[:prune_amount].tolist()


# In[3]: Channel (filter) importance by redundancy criterion.
# Normalizes every output channel to unit norm, computes pairwise distance
# between channels, and returns the indices of the `prune_amount` channels
# that are closest to some other channel in the layer (i.e. most redundant /
# most "similar" to another surviving channel), most redundant first.
def compute_distance_score_channel(tensor_t, n=1, prune_amount=1):
    out_channels = tensor_t.shape[0]
    prune_amount = max(0, min(prune_amount, out_channels - 1))
    flat = tensor_t.reshape(out_channels, -1)
    norms = torch.norm(flat, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = flat / norms
    dist = torch.cdist(normalized, normalized, p=n)
    dist.fill_diagonal_(float('inf'))
    nearest_dist, _ = torch.min(dist, dim=1)
    order = torch.argsort(nearest_dist)
    return order[:prune_amount].tolist()


# In[3b]: Channel importance by "Max3" saliency criterion (thesis Ch. 4.3).
# For each output channel, for every input-channel kernel slice, take the 3
# largest-magnitude weights and sum them; sum those per-kernel scores across
# all input channels to get the channel's saliency score. Rationale: a
# channel's most discriminative signal is assumed to be carried by each
# kernel's few strongest activations rather than its overall magnitude, so
# this can rank a channel highly even when its plain L1 norm is unremarkable.
def compute_max3_saliency_score_channel(tensor_t, prune_amount=1):
    out_channels, in_channels = tensor_t.shape[0], tensor_t.shape[1]
    prune_amount = max(0, min(prune_amount, out_channels - 1))
    flat = tensor_t.reshape(out_channels, in_channels, -1).abs()
    k = min(3, flat.shape[-1])
    channel_score = flat.topk(k, dim=-1).values.sum(dim=(-1, -2))
    order = torch.argsort(channel_score)
    return order[:prune_amount].tolist()


# In[3c]: Channel importance via Singular Value Decomposition (thesis
# Ch. 4.5). Each output channel's weight slice (in_channels x kh x kw) is
# reshaped to a 2D matrix (in_channels x kh*kw); its singular values are
# summed to get a single "energy" score for that channel -- channels whose
# kernels carry less total energy across their singular directions are
# pruned first. Batched via torch.linalg.svdvals so every channel in the
# layer is decomposed in one call.
def compute_svd_saliency_score_channel(tensor_t, prune_amount=1):
    out_channels, in_channels = tensor_t.shape[0], tensor_t.shape[1]
    prune_amount = max(0, min(prune_amount, out_channels - 1))
    flat = tensor_t.reshape(out_channels, in_channels, -1)
    singular_values = torch.linalg.svdvals(flat)
    channel_score = singular_values.sum(dim=-1)
    order = torch.argsort(channel_score)
    return order[:prune_amount].tolist()


# In[3d]: Channel importance via K-means redundancy clustering (thesis
# Ch. 5.2/6.2). Flattens each output channel to a feature vector, clusters
# them into (out_channels - prune_amount) groups using the given distance
# metric, keeps the largest-L1-norm channel from each cluster, and marks the
# rest for pruning -- channels that are similar to some other surviving
# channel are considered redundant. `distance` selects the metric used only
# for the cluster-assignment step (centroids are still updated by mean):
# 'manhattan' (L1, the best-performing metric in the thesis), 'euclidean',
# or 'cosine'.
def _pairwise_distance(a, b, distance='manhattan'):
    if distance == 'euclidean':
        return torch.cdist(a, b, p=2)
    if distance == 'manhattan':
        return torch.cdist(a, b, p=1)
    if distance == 'cosine':
        a_n = torch.nn.functional.normalize(a, dim=-1)
        b_n = torch.nn.functional.normalize(b, dim=-1)
        return 1 - a_n @ b_n.t()
    raise ValueError(f"Unknown distance metric: {distance!r}")


def _kmeans_cluster_channels(features, k, distance='manhattan', iters=25):
    n = features.shape[0]
    k = max(1, min(k, n))
    centroids = features[torch.randperm(n)[:k]].clone()

    assignments = torch.full((n,), -1, dtype=torch.long)
    for it in range(iters):
        dist = _pairwise_distance(features, centroids, distance=distance)
        new_assignments = torch.argmin(dist, dim=1)
        converged = it > 0 and torch.equal(new_assignments, assignments)
        assignments = new_assignments
        if converged:
            break

        for c in range(k):
            members = features[assignments == c]
            if len(members) > 0:
                centroids[c] = members.mean(dim=0)
            else:
                # Re-seed an empty cluster from a random point so it has a
                # chance to pick up a member (a point at distance 0 from its
                # own value) on the next assignment step.
                centroids[c] = features[torch.randint(0, n, (1,))].squeeze(0)
    return assignments


def compute_kmeans_similarity_score_channel(tensor_t, prune_amount=1, distance='manhattan'):
    out_channels = tensor_t.shape[0]
    prune_amount = max(0, min(prune_amount, out_channels - 1))
    if prune_amount == 0:
        return []
    k = out_channels - prune_amount

    flat = tensor_t.reshape(out_channels, -1)
    channel_norm = torch.norm(flat, p=1, dim=1)
    assignments = _kmeans_cluster_channels(flat, k, distance=distance)

    prune_indices = []
    for c in range(k):
        members = (assignments == c).nonzero(as_tuple=True)[0]
        if len(members) == 0:
            continue
        keep_idx = members[torch.argmax(channel_norm[members])].item()
        prune_indices.extend(m for m in members.tolist() if m != keep_idx)

    # A cluster can end up empty (all candidate points reassigned elsewhere
    # before convergence), which would prune more than `prune_amount`
    # channels since no representative survives for it. Guard by keeping
    # only the `prune_amount` weakest (smallest-norm) candidates.
    prune_indices.sort(key=lambda idx: channel_norm[idx].item())
    return prune_indices[:prune_amount]


# In[3]:
def compute_saliency_score_kernel(tensor_t, n=1, dim_to_keep=[0, 1], prune_amount=1):
    # dims = all axes, except for the one identified by `dim`
    dim_to_prune = list(range(tensor_t.dim()))  # initially it has all dims

    # remove dim which we want to keep from dimensions to prune
    for i in range(len(dim_to_keep)):
        dim_to_prune.remove(dim_to_keep[i])

    size = tensor_t.shape
    norm = torch.norm(tensor_t, p=n, dim=dim_to_prune)
    kernel_list_saliency = []
    size = norm.shape
    kl = -1
    max_value = 0
    max_idx = 0
    for i in range(size[0]):
        for j in range(size[1]):
            if (kl+1) < prune_amount:
                kernel_list_saliency.append([i, j, norm[i][j]])
                kl += 1
                if kernel_list_saliency[kl][2] > max_value:
                    max_value = kernel_list_saliency[prune_amount][2]
                    max_idx = kl
            else:
                if norm[i][j] < max_value:
                    kernel_list_saliency.pop(max_idx)
                    kernel_list_saliency.append([i, j, norm[i][j]])
                    max_value = 0
                    max_idx = 0
                    for rang_idx in range(prune_amount):
                        if max_value < kernel_list_saliency[rang_idx][2]:
                            max_value = kernel_list_saliency[rang_idx][2]
                            max_idx = rang_idx

    return kernel_list_saliency


# In[2]: 
def compute_distance_score_kernel(tensor_t, n=1, dim_to_keep=[0, 1], prune_amount=1):
    # dims = all axes, except for the one identified by `dim`
    dim_to_prune = list(range(tensor_t.dim()))  # initially it has all dims
    # remove dim which we want to keep from dimensions to prune
    for i in range(len(dim_to_keep)):
        dim_to_prune.remove(dim_to_keep[i])

    size = tensor_t.shape
    module_buffer = torch.zeros_like(tensor_t)

    # shape of norm should be equal to multiplication of dim to keep values
    norm = torch.norm(tensor_t, p=n, dim=dim_to_prune)
    size = tensor_t.shape
    for i in range(size[0]):
        for j in range(size[1]):
            module_buffer[i][j] = tensor_t[i][j] / norm[i][j]

    dist = torch.zeros(size[1], size[0], size[0])

    kernel_list_distance = []
    for j in range(size[1]):
        idx_tuple = []
        print('.', end='')
        max_value = -1
        max_idx = -1
        for i1 in range(size[0]):
            for i2 in range((i1 + 1), size[0]):
                dist[j][i1][i2] = torch.norm((module_buffer[i1][j] - module_buffer[i2][j]), p=1)
                dist[j][i2][i1] = dist[j][i1][i2]


                if len(idx_tuple) < prune_amount:
                    idx_tuple.append([j, i1, i2, dist[j][i1][i2]])
                    idx = len(idx_tuple) - 1
                    if max_value < idx_tuple[idx][3]:
                        max_value = idx_tuple[idx][3]
                        max_idx = idx
                    continue

                if dist[j][i1][i2] < max_value:
                    del idx_tuple[max_idx]
                    idx_tuple.append([j, i1, i2, dist[j][i1][i2]])

                    max_value = idx_tuple[0][3]
                    max_idx = 0
                    for new_max_idx in range(1, len(idx_tuple)):
                        if max_value < idx_tuple[new_max_idx][3]:
                            max_value = idx_tuple[new_max_idx][3]
                            max_idx = new_max_idx

        kernel_list_distance.append(idx_tuple)
    return kernel_list_distance


# In[6]:
def deep_copy_kernelwise(destination_model, source_model):
    for i in range(len(source_model.features)):
        if str(source_model.features[i]).find('Conv') != -1:
            size_org = source_model.features[i]._parameters['weight'].shape
            size_new = destination_model.features[i]._parameters['weight'].shape
            for fin_org in range(size_org[1]):
                j = 0
                fin_new = fin_org
                for fout in range(size_org[0]):
                    if torch.norm(source_model.features[i]._parameters['weight'][fout][fin_org]) != 0:
                        fin_new += 1
                        if j >= size_new[0] or fin_new >= size_new[1]:
                            break
                        t = source_model.features[i]._parameters['weight'][fout][fin_org]
                        destination_model.features[i]._parameters['weight'][j][fin_new] = t
                        j = j + 1


# In[7]: Copy surviving (non-zeroed) output channels of each pruned conv
# layer in source_model into the smaller destination_model, in order.
#
# Because pruning a layer zeroes out entire output channels (filters), the
# NEXT conv layer's input-channel dimension must be sliced using the same
# surviving-channel indices its predecessor kept -- otherwise the copied
# weight tensor's input dimension (still full-size in source_model) will not
# match destination_model's already-shrunk input dimension. The previous
# version of this function ignored that, copying the full input-channel
# slice unmodified, which raises a shape-mismatch error for every conv layer
# beyond the first. BatchNorm parameters (weight/bias/running stats), which
# are also indexed per output-channel, are copied the same way; skipping
# them (as before) meant every fine-tuning run started from freshly
# re-initialized BatchNorm statistics instead of the trained ones.
#
# The assertions below are deliberate: this function's correctness was never
# actually confirmed during the original (week-long, V100) training runs --
# a shape mismatch here fails loudly and immediately rather than silently,
# so if `feature_list` bookkeeping and the actual surviving-channel count
# ever diverge again, the very next run says so on round 1 instead of
# quietly copying/training on a model that doesn't mean what its recorded
# parameter/FLOPs counts claim.
def deep_model_copy_channelwise(source_model, destination_model, feature_list):
    prev_surviving_in = None  # channel indices kept by the previous conv layer
    with torch.no_grad():
        for l in range(len(source_model.features)):
            layer_str = str(source_model.features[l])
            if layer_str.find('Conv') != -1:
                src_weight = source_model.features[l]._parameters['weight']
                dst_weight = destination_model.features[l]._parameters['weight']
                src_bias = source_model.features[l]._parameters.get('bias')
                dst_bias = destination_model.features[l]._parameters.get('bias')

                out_org = src_weight.shape[0]
                surviving_out = [o for o in range(out_org) if torch.norm(src_weight[o]) != 0]

                assert len(surviving_out) == dst_weight.shape[0], (
                    f"deep_model_copy_channelwise: layer {l} has {len(surviving_out)} "
                    f"surviving (non-zero) channels in source_model, but destination_model "
                    f"was built expecting {dst_weight.shape[0]} -- feature_list bookkeeping "
                    f"and the actual pruned-channel count have diverged.")

                for new_o, old_o in enumerate(surviving_out):
                    src_slice = src_weight[old_o]
                    if prev_surviving_in is not None:
                        src_slice = src_slice[prev_surviving_in]
                    assert src_slice.shape == dst_weight[new_o].shape, (
                        f"deep_model_copy_channelwise: layer {l} channel {old_o}->{new_o} "
                        f"shape mismatch: source slice {tuple(src_slice.shape)} vs "
                        f"destination {tuple(dst_weight[new_o].shape)}.")
                    dst_weight[new_o] = src_slice
                    if src_bias is not None and dst_bias is not None:
                        dst_bias[new_o] = src_bias[old_o]

                prev_surviving_in = surviving_out
            elif layer_str.find('BatchNorm') != -1 and prev_surviving_in is not None:
                src_bn = source_model.features[l]
                dst_bn = destination_model.features[l]
                assert len(prev_surviving_in) == dst_bn.weight.shape[0], (
                    f"deep_model_copy_channelwise: BatchNorm layer {l} expects "
                    f"{dst_bn.weight.shape[0]} channels but the preceding conv layer kept "
                    f"{len(prev_surviving_in)} -- feature_list bookkeeping has diverged.")
                for new_o, old_o in enumerate(prev_surviving_in):
                    dst_bn.weight[new_o] = src_bn.weight[old_o]
                    dst_bn.bias[new_o] = src_bn.bias[old_o]
                    dst_bn.running_mean[new_o] = src_bn.running_mean[old_o]
                    dst_bn.running_var[new_o] = src_bn.running_var[old_o]


# In[]