# Master Thesis - Model Training Results

## Training Times

| Model | Parameters (M) | Trainable Params (M) | Inference Time - Single (ms) | Inference Time - Batch (ms) | Inference Time - Per Sample (ms) | Training Epoch Time (sec) | Training Epoch Time (min) | Memory (MB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mobilenetv3_small_100 | 1.528 | 1.528 | 4.48 | 18.61 | 0.073 | 25.11 | 0.42 | 27.94 |
| regnetx_002 | 2.319 | 2.319 | 4.30 | 32.89 | 0.128 | 25.81 | 0.43 | 32.10 |
| regnety_002 | 2.798 | 2.798 | 6.44 | 35.52 | 0.139 | 26.51 | 0.44 | 33.94 |
| ghostnet_100 | 3.914 | 3.914 | 8.49 | 57.10 | 0.223 | 30.59 | 0.51 | 40.56 |
| mnasnet_100 | 3.115 | 3.115 | 4.33 | 60.20 | 0.235 | 30.61 | 0.51 | 28.69 |
| efficientnet_lite0 | 3.384 | 3.384 | 4.39 | 72.29 | 0.282 | 37.08 | 0.62 | 43.22 |
| mobilevit_xs | 1.937 | 1.937 | 7.29 | 146.74 | 0.573 | 55.98 | 0.93 | 40.63 |

## Notes

Training times extracted from TensorFlow event files using [src/extract_training_time.py](src/extract_training_time.py).

To regenerate this table:
```bash
uv run python src/extract_training_time.py --results-dir ./results
```
ok