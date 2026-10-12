### Speaker C vs reference A

Recordings: 16

| IoU threshold | precision | recall | F1 |
|---|---|---|---|
| iou>=0.3 | 0.875 | 0.75 | 0.808 |
| iou>=0.5 | 0.75 | 0.643 | 0.692 |

- matched mean IoU: **0.838**; median start / end error: **13.7 / 12.6 ms**
- type accuracy among matched: **1.0** (family: 1.0)
- ideal (flawless) recordings: {'n': 1, 'false_regions_per_recording': 0.0, 'mean_score': 94.7}
- score vs ladder level: Spearman **-1.0**, pairwise ordering **1.0**, category attribution **0.9**
- mean score by level: {0: 94.7, 1: 92.9, 2: 89.4, 3: 77.4, 4: 74.6, 5: 51.6}

| flaw type | recall |
|---|---|
| drag | 0.667 |
| misplaced_pause | 1.0 |
| missing_pause | 0.5 |
| monotone | 1.0 |
| mumble | 0.5 |
| overlong_pause | 1.0 |
| rush | 0.75 |
| trail_off | 0.667 |
| uptalk | 0.333 |
| volume_collapse | 1.0 |

| severity | recall |
|---|---|
| high (>=0.7) | 0.833 |
| low (<0.4) | 0.0 |
| mid (0.4-0.7) | 0.733 |
