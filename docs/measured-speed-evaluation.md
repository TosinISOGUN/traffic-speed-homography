# Evaluating against measured speeds

`scripts/evaluate.py` scores the pipeline's per-vehicle speeds against measured ground truth. It is separate from the binary-label evaluation behind the manuscript's Table 1 (see the main README).

## 1. Run the pipeline

```bash
python scripts/estimate_speed.py --source data/vehicles.mp4
```

This writes `results/video/per_vehicle_speeds.csv` and `annotated.mp4`, where each vehicle is labelled `#id`.

## 2. Measure true speeds

Any independent source works: radar, GPS, or timing a vehicle between road markers of known spacing. For `vehicles.mp4`, the roadside posts are 50 m apart, so a vehicle's speed is 50 m divided by the time it takes to travel from one post to the next. This does not depend on the homography.

Step through the video and, for each vehicle, note the frame where it reaches one post and the frame where it reaches a later post. Time the same point on the vehicle at both posts (for example, where its front wheels meet the post). Save the frames with the vehicle's tracker ID:

```csv
tracker_id,frame_start,frame_end,posts_crossed
3,112,171,1
5,140,184,1
```

`posts_crossed` is the number of 50 m gaps between the two frames (default 1). Frame numbers match the `frame` column of `results/video/frame_log.csv`, counted from 0.

```bash
python scripts/timings_to_ground_truth.py --timings my_timings.csv     # -> results/video/ground_truth.csv
```

## 3. Evaluate

```bash
python scripts/evaluate.py --ground-truth results/video/ground_truth.csv --tol-kmh 5
```

This writes `metrics.json`, `table1_metrics.csv`, `per_vehicle_outcomes.csv` and `fig5_metrics.png` to `results/evaluation/`. Vehicles you did not time are left out of the score, and the script notes how many tracks it ignored.

For more control, `results/video/ground_truth_template.csv` has one row per tracked vehicle. Fill in `true_speed_kmh` by hand and:

* leave it blank for a track that is not a real vehicle (it counts as spurious);
* list several IDs as `3;7` if the tracker switched IDs on one vehicle;
* add a row with a blank `tracker_id` for a vehicle that was never tracked (it counts as missed).

## Metric definitions

With measured speeds there is no natural "true negative", so the metrics are vehicle-level. An estimate is **correct** when it is within the tolerance (default ±5 km/h; `--tol-rel 0.1` adds a ±10% rule).

| Count | Meaning |
|---|---|
| TP | Real vehicle, estimate correct |
| FP | Estimate wrong, or estimate on a spurious track |
| FN | Real vehicle missed, or estimate wrong |

accuracy = TP / (real vehicles + spurious tracks), precision = TP / (TP + FP), recall = TP / (TP + FN), and F1 is their harmonic mean. These differ from the binary definitions behind Table 1, so scores from the two are not directly comparable. The script also reports MAE, RMSE and MAPE.
