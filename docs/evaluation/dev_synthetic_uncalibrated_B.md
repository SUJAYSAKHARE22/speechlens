### Speaker B vs reference A

Recordings: 16

| IoU threshold | precision | recall | F1 |
|---|---|---|---|
| iou>=0.3 | 1.0 | 0.821 | 0.902 |
| iou>=0.5 | 1.0 | 0.821 | 0.902 |

- matched mean IoU: **0.91**; median start / end error: **12.0 / 18.1 ms**
- type accuracy among matched: **1.0** (family: 1.0)
- ideal (flawless) recordings: {'n': 1, 'false_regions_per_recording': 0.0, 'mean_score': 91.3}
- score vs ladder level: Spearman **-1.0**, pairwise ordering **1.0**, category attribution **0.9**
- mean score by level: {0: 91.3, 1: 88.1, 2: 85.5, 3: 82.1, 4: 70.4, 5: 52.3}

| flaw type | recall |
|---|---|
| drag | 0.8 |
| misplaced_pause | 1.0 |
| missing_pause | 1.0 |
| monotone | 1.0 |
| mumble | 0.5 |
| overlong_pause | 1.0 |
| rush | 0.6 |
| trail_off | 1.0 |
| uptalk | 0.5 |
| volume_collapse | 1.0 |

| severity | recall |
|---|---|
| high (>=0.7) | 1.0 |
| low (<0.4) | 0.0 |
| mid (0.4-0.7) | 0.786 |
