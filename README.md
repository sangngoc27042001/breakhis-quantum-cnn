# Master Thesis - Model Training Results

## Training Times

| Model | Training Time | Duration (seconds) |
|-------|--------------|-------------------|
| densenet169_20251220_081918 | 1h 28m 53s | 5333.37 |
| efficientnetv2b3_20251219_224900 | 45m 28s | 2728.25 |
| mobilenetv3large_20251219_210637 | 21m 27s | 1287.64 |
| nasnetmobile_20251219_213208 | 1h 4m 2s | 3842.63 |
| vgg16_20251219_235830 | 1h 17m 44s | 4664.87 |

## Notes

Training times extracted from TensorFlow event files using [src/extract_training_time.py](src/extract_training_time.py).

To regenerate this table:
```bash
uv run python src/extract_training_time.py --results-dir ./results
```
