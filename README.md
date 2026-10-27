# Greenpod Labs Assignment 2 — Orange Image Classification (Adapted)

> **Dataset disclosure:** The original assignment asks for avocado ripening-stage classification. The supplied ZIP is an orange image dataset with four folders/classes: `fresh`, `blackspot`, `canker`, and `grenning`. This project therefore classifies orange condition/disease categories, **not avocado ripening stages**. Confirm with the evaluator whether this substitution is acceptable.

## Approaches
1. **Classical computer vision:** HSV color histograms + HOG texture/shape features, classified using an RBF-kernel SVM.
2. **Deep learning:** A compact CNN trained from scratch with augmentation.

## Dataset layout expected
Extract the supplied ZIP so that this project folder (or `--data-dir`) contains:
```
train/
  fresh/ blackspot/ canker/ grenning/
test/
  fresh/ blackspot/ canker/ grenning/
```
The supplied ZIP already follows this layout.

## Run
Python 3.10+ recommended.

For Streamlit Community Cloud, `requirements.txt` uses the headless OpenCV package so the app does not depend on Linux GUI libraries such as `libGL.so.1`.

```bash
pip install -r requirements.txt
python train_and_evaluate.py --data-dir "/path/to/extracted/dataset" --output-dir outputs --epochs 8
```

On the first run, the script trains both approaches, writes a classification report and confusion-matrix plots, saves the CNN weights, and prints a sample prediction. It uses the provided `test/` folder only for final evaluation; training validation is split from `train/`.

## Important evaluation notes
- `grenning` is kept exactly as named in the dataset; verify whether it means “greening” and whether labels are consistently applied.
- Report macro-F1 alongside accuracy because class counts are uneven.
- A test folder with only 99 images makes metrics uncertain. Treat results as a small benchmark, not proof of production readiness.
- The script checks unreadable files and image dimensions. It does not automatically remove visually unusual images; inspect any flagged cases before excluding them.
- Do not claim the model is production-ready without testing on images from different phones, lighting, backgrounds, farms, and capture conditions.

## Quick inference with the trained SVM baseline
After training, run:
```bash
python predict_image.py --image "/path/to/your/orange.jpg" --model outputs/svm_hog_hsv.joblib
```

## Baseline results included
The report includes measured SVM test metrics and a separately trained **one-epoch CNN baseline** on the same test split. The one-epoch CNN is intentionally undertrained; run `train_and_evaluate.py` for the full augmentation/validation-based CNN training and final model comparison. Never present the quick CNN score as a tuned model result.

The report also includes a measured three-epoch quick CNN baseline in `outputs/quick_cnn_three_epoch.json`. The main `train_and_evaluate.py` script is the fuller pipeline for validation-based CNN training; its output should replace the quick baseline in the final model-selection discussion after you run it.
