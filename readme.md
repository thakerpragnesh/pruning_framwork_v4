# pruning_framwork_v4

Structured channel pruning for VGG16, implementing the methods from my
Ph.D. thesis on neural network compression — six peer-reviewed
publications including two IEEE Access journal articles (see the papers'
DOIs below). This is the canonical repo for the CNN pruning work.

For the Transformer extension of these same ideas (BERT FFN pruning,
attention-head redundancy), see
[`transformer_pruning`](https://github.com/thakerpragnesh/transformer_pruning).
A separate package, [`structured-pruning`](https://github.com/thakerpragnesh/structured-pruning),
has a more thoroughly tested reference implementation of the mask-then-compress
pruning workflow used here, plus an earlier version of the Transformer work —
but it only covers Max-k/L1/L2/random criteria, not the full method set below.

## Criteria implemented

| Criterion | Script | Method |
|---|---|---|
| Max-3 saliency | `channel_pruning_max3.py` | Sum of top-3 kernel magnitudes per channel (thesis Ch. 4.3) |
| L1 saliency | `channel_pruning_saliency.py` | Plain L1 norm per channel |
| Distance/redundancy | `channel_pruning_distance.py` | Repeatedly prunes the lower-norm member of the closest remaining channel pair (thesis Ch. 5.1) |
| SVD | `channel_pruning_svd.py` | Prunes lowest singular-value-energy channels |
| K-means | `channel_pruning_kmeans.py` | Clusters channels (Manhattan/Euclidean/Cosine, `config.ini`'s `distance_metric`), keeps the highest-norm channel per cluster |
| Hybrid | `hybrid_pruning.py` | Chains criteria across rounds — 4 rounds Max-3, then 3 rounds K-means, standing in for the thesis's best-found sequence |

Each is a ~15-line entry point (`env.run(fp.compute_..._channel)`) — all
shared setup (config parsing, paths, logging, device, dataloaders, model
loading) lives once in `pruning_app.py::PruningEnvironment`, and the
prune → shrink → copy → fine-tune loop lives once in `pruning_driver.py`.

## Running it

Edit `config.ini` for your dataset paths, then:

```bash
python channel_pruning_max3.py       # or _saliency, _distance, _svd, _kmeans
python hybrid_pruning.py
```

## Architecture notes

- **Structural surgery, not masking-only.** Each round physically shrinks
  `Conv2d`/`BatchNorm2d` layers to the surviving channels
  (`facilitate_pruning.py::deep_model_copy_channelwise`), with explicit
  shape assertions so a `feature_list` bookkeeping error fails loudly on
  round 1 instead of silently training garbage.
- **VGG16-BN throughout.** The source model and every round's rebuilt
  model both use `vgg16_bn`, so `.features` layer indices stay aligned
  between them — a real bug in an earlier version of this code came from
  mixing `vgg16`/`vgg16bn`, which shifts indices by however many BatchNorm
  layers exist.
- **K-means uses k-means++ initialization**, not uniform random centroid
  selection — with plain random init, a near-duplicate channel pair is
  very likely to both become separate initial centroids, and once that
  happens neither can ever be reassigned to the other's cluster (a point's
  distance to itself is always 0, beating even a near-identical neighbor).
  k-means++ biases centroid selection away from points close to existing
  centroids, which avoids this.

## Publications

See `CHANGES.md` for the full defect-fix history of this codebase. Six
publications total; the two IEEE Access journal articles cover the Max-3
saliency criterion and a custom channel-regularization loss, both
implemented here.

## License

Not yet specified — add one before treating this as reusable by others.
