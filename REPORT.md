# Greenpod Labs — Assignment 2
## Computer Vision Classification of Orange Condition Categories

**Candidate project:** Two-approach image classification pipeline  
**Dataset disclosure:** The original assignment asks for *avocado ripening-stage detection/classification*. The supplied ZIP is an *orange* image dataset with four labels (`fresh`, `blackspot`, `canker`, `grenning`). This submission is adapted to the supplied data and does **not** solve avocado ripening classification. Please disclose this difference to the evaluator and ask whether the substitution is acceptable.

## 1. Objective
Classify an input orange image into one of four dataset labels: `fresh`, `blackspot`, `canker`, or `grenning`. This is image-level classification, not object detection: the model predicts a class for the whole image and does not draw bounding boxes.

## 2. Dataset understanding
The supplied archive has `train/` and `test/` folders. Inventory scan found **1,090 readable images**: 991 training images and 99 test images. No unreadable images were found during the scan.

| Split | blackspot | canker | fresh | grenning | Total |
|---|---:|---:|---:|---:|---:|
| Train | 184 | 179 | 281 | 347 | 991 |
| Test | 22 | 22 | 33 | 22 | 99 |

The training split is imbalanced: `grenning` is the largest class (347; about 35.0%) and `canker` is the smallest (179; about 18.1%). The test set is also small, with 22–33 examples per class. Consequently, a single mistake changes per-class recall by about 3–5 percentage points. Accuracy alone is not enough; macro-F1 and per-class recall are important.

### Data quality, outliers, and bias
- The scan found 150 distinct image dimensions; 776 images are 800×800, while others vary. The code resizes images to a common input size.
- No images were unreadable. The simple outlier rule (dimension below 64 px or aspect ratio outside 0.5–2.0) flagged zero images. This does **not** prove there are no mislabeled, duplicated, blurry, or visually unusual images.
- Labels may be noisy or ambiguous, particularly `grenning` (possibly intended as “greening”); preserve the supplied label in code but verify its meaning with the dataset owner.
- Potential biases: background, camera, lighting, image source, fruit size, and capture angle may correlate with class. Random train/test splits can overstate real-world performance if near-duplicate images or source-specific patterns occur in both splits. A duplicate/near-duplicate check and a source-separated split would improve confidence.
- Do not delete flagged images automatically. Inspect them manually and document any exclusions.

## 3. Approach 1 — Classical computer vision (HSV histogram + HOG + SVM)
**Pipeline:** resize image → convert to HSV and extract a normalized color histogram → extract HOG texture/edge descriptors → standardize features → RBF-kernel SVM with balanced class weights.

**Why selected:** Color can carry useful information about orange appearance, while HOG captures edges and texture. SVM is suitable for a small-to-medium dataset and offers a strong baseline without requiring a GPU. Its limitations are sensitivity to lighting/background and limited ability to learn complex visual representations.

### Measured test results
The SVM was trained on the supplied training split and evaluated on the 99-image test split.

| Metric | Result |
|---|---:|
| Accuracy | 90.91% |
| Macro precision | 90.30% |
| Macro recall | 90.15% |
| Macro F1 | 90.03% |
| Weighted F1 | 90.80% |

Per-class results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| blackspot | 0.800 | 0.909 | 0.851 | 22 |
| canker | 0.842 | 0.727 | 0.780 | 22 |
| fresh | 0.970 | 0.970 | 0.970 | 33 |
| grenning | 1.000 | 1.000 | 1.000 | 22 |

The weakest recall is `canker` at 72.7%, meaning 6 of 22 canker examples were missed. That class-specific error matters more than the overall accuracy may suggest. The perfect score on `grenning` is encouraging but should not be treated as reliable proof because there are only 22 test examples.

## 4. Approach 2 — Compact CNN
The code implements a small convolutional neural network with convolution, batch normalization, ReLU, pooling, dropout, and a four-class output. It uses 64×64 resized images, data augmentation (horizontal flips, small rotations, and mild color changes), inverse-frequency class weights, and a validation split from the training data. The best epoch is selected using validation macro-F1, then evaluated on the test set.

**Why selected:** CNNs learn spatial features directly from pixels rather than relying on hand-crafted descriptors. Augmentation helps with modest variations in orientation and lighting. A compact CNN is easier to run than a large pretrained network, but training from scratch on only 991 training images can overfit and may be less robust than transfer learning.

**Quick CNN baseline measured:** A separate compact CNN (3 convolution blocks, 64×64 inputs) was trained for **three epochs** on the full training split and evaluated on the same 99-image test split. It achieved **75.76% accuracy**, **72.73% macro recall**, **65.93% macro-F1**, and **83.67% macro precision**. This is a quick, lightly trained baseline—not the final tuned CNN. The full `train_and_evaluate.py` pipeline trains a compact CNN with augmentation, class weighting, and validation-based epoch selection; run it to generate the final CNN metrics and `model_comparison.csv`.

## 5. Metrics and model-selection logic
- **Accuracy:** easy overall summary, but can hide class-specific failures.
- **Precision:** among images predicted as a class, how many were correct; useful when false alarms are costly.
- **Recall:** among actual examples of a class, how many were detected; important when missing a diseased fruit is costly.
- **Macro-F1:** computes F1 per class and averages classes equally, preventing the largest class from dominating the headline score.
- **Confusion matrix:** shows which categories are confused, helping diagnose overlapping visual symptoms.

Measured quick comparison (same 99-image test split): SVM — 90.91% accuracy and 90.03% macro-F1; three-epoch CNN baseline — 75.76% accuracy and 65.93% macro-F1. The CNN is still lightly trained and is not fully tuned, so the SVM is the stronger model in the measured baseline. Run the full script before final submission for a validation-selected CNN and final model comparison; do not choose only from accuracy.

## 6. Example run
A sample inference was run on `test/blackspot/b (1).jpg`:
- Ground-truth folder label: `blackspot`
- SVM prediction: `blackspot`
- Correct: yes

This is only one example, not a measure of general performance. The script writes an example prediction JSON and, after full execution, includes CNN prediction and confidence values.

## 7. Production performance
The SVM is lightweight at inference time: image resize, feature extraction, and one SVM prediction. A compact CNN can also run on CPU, especially at 64×64 resolution, and can be accelerated with a GPU. Real production latency must be measured on the target device with preprocessing included.

The reported SVM test accuracy is 90.9% on only 99 test images. It is not a production guarantee. Before deployment, test on independently collected images across different phones, lighting, backgrounds, farms, fruit varieties, and disease severity. Add confidence thresholds and a human-review/“uncertain” path; monitor errors and data drift.

## 8. Drawbacks
- Dataset differs from the requested avocado dataset.
- Small dataset and small test set create high uncertainty.
- Possible class-label ambiguity and unverified label quality.
- Potential source/background leakage or near-duplicate images.
- SVM uses hand-crafted features and may fail under lighting/background shifts.
- CNN trained from scratch may overfit and requires tuning.
- No bounding boxes, segmentation, or fruit localization; whole-image classification only.

## 9. What I would improve with more time
1. Obtain the correct avocado dataset or written approval to use the orange dataset.
2. Verify class definitions and manually audit mislabeled/ambiguous examples.
3. Run perceptual-hash duplicate detection and group splits by source/session to reduce leakage.
4. Use stratified cross-validation on the training split and report confidence intervals.
5. Compare transfer learning (MobileNetV3/EfficientNet-B0) against the compact CNN.
6. Tune augmentation, learning rate, class weights, and SVM hyperparameters using validation data only.
7. Evaluate on a larger external test set and report latency, memory, calibration, and per-class failure cases.
8. Add a review threshold and log uncertain predictions rather than forcing every image into a class.

## 10. Where most time was spent
Most effort went into understanding the archive and class distribution, implementing two distinct pipelines, and setting up a fair evaluation workflow with a held-out test set, macro-F1, per-class recall, and confusion matrices. A key remaining step is completing the CNN run and inserting its measured metrics into the comparison; those values must come from an actual run, not an estimate.

## 11. Reproducibility
From the extracted dataset root:
```bash
pip install -r requirements.txt
python train_and_evaluate.py --data-dir "/path/to/extracted/dataset" --output-dir outputs --epochs 5
```
The script saves data inventory, class counts, confusion matrices, classification reports, model weights, comparison CSV/JSON, and an example prediction. It uses the test split only for final evaluation and derives validation data from `train/`.
