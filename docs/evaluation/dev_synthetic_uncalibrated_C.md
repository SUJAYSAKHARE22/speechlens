### Speaker C vs reference A

Recordings: 16

| IoU threshold | precision | recall | F1 |
|---|---|---|---|
| iou>=0.3 | 0.88 | 0.786 | 0.83 |
| iou>=0.5 | 0.76 | 0.679 | 0.717 |

- matched mean IoU: **0.832**; median start / end error: **15.4 / 12.8 ms**
- type accuracy among matched: **0.955** (family: 1.0)
- ideal (flawless) recordings: {'n': 1, 'false_regions_per_recording': 0.0, 'mean_score': 92.5}
- score vs ladder level: Spearman **-1.0**, pairwise ordering **1.0**, category attribution **0.9**
- mean score by level: {0: 92.5, 1: 90.9, 2: 87.3, 3: 75.0, 4: 72.1, 5: 49.1}

| flaw type | recall |
|---|---|
| drag | 0.667 |
| misplaced_pause | 1.0 |
| missing_pause | 0.5 |
| monotone | 1.0 |
| mumble | 0.5 |
| overlong_pause | 1.0 |
| rush | 0.75 |
| trail_off | 1.0 |
| uptalk | 0.333 |
| volume_collapse | 1.0 |

| severity | recall |
|---|---|
| high (>=0.7) | 0.917 |
| low (<0.4) | 0.0 |
| mid (0.4-0.7) | 0.733 |
