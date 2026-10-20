# Week 8 — Neural Networks for Object Classification

## Overview

This project explores a simple feed-forward neural network for object classification using image features extracted from the **YCB-Video dataset**.

The main goal is to understand the fundamentals of neural networks, implement a first neural-network classifier, compare it with classical machine learning models, and investigate how feature representation affects classification performance.

## Objectives

- Understand the basic architecture of a feed-forward neural network.
- Train a neural network using handcrafted image features.
- Compare neural-network classification with Logistic Regression.
- Evaluate models using scene-based cross-validation.
- Investigate overfitting and class-specific classification errors.
- Compare color histograms, HOG features, and combined features.

## Dataset

The experiments use images from YCB-Video test scenes `000048` to `000059`.

- **Total samples:** 2,700
- **Number of classes:** 9
- **Development scenes:** `000048`–`000058`
- **Held-out test scene:** `000059`

The held-out scene is reserved for a final evaluation after model development and selection.

### Object Classes

| Class ID | Object |
|---:|---|
| 1 | Master Chef Can |
| 2 | Cracker Box |
| 3 | Sugar Box |
| 4 | Tomato Soup Can |
| 6 | Tuna Fish Can |
| 9 | Potted Meat Can |
| 11 | Pitcher Base |
| 12 | Bleach Cleanser |
| 15 | Power Drill |

## Feature Extraction

Three feature representations were investigated:

| Feature representation | Dimensions | Description |
|---|---:|---|
| HSV color histogram | 4,096 | Represents the distribution of image colors |
| HOG | 1,764 | Represents local image gradients and shape-related patterns |
| Combined features | 5,860 | Concatenation of color histogram and HOG features |

These are handcrafted features extracted from images, rather than raw pixels supplied directly to the neural network.

## Neural Network

The first neural-network classifier uses a simple fully connected architecture:

```text
Input: 4,096 color histogram features
              |
       Dense layer (128)
              |
            ReLU
              |
       Dense layer (9)
              |
        Class logits
```

**Training configuration**

- Framework: PyTorch
- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Learning rate: `0.001`
- Batch size: `64`
- Random seed: `42`
- Training duration: 5 epochs per development fold in the scene-based comparison

The architecture contains **525,577 trainable parameters**.

## Evaluation Strategy

The models were evaluated using Leave-One-Scene-Out (LOSO) cross-validation on development scenes `000048`–`000058`.

For each fold, one scene was held out for validation while the remaining development scenes were used for training. This evaluates how well the models generalize to scenes not included in their training data.

The final test scene, `000059`, is kept separate from this development evaluation.

Metrics include:

- Accuracy
- Macro-F1 score
- Per-class precision, recall, and F1-score
- Confusion matrices

## Results

### Logistic Regression: Feature Ablation

| Feature representation | Accuracy | Macro-F1 |
|---|---:|---:|
| Original color histograms | 72.30% | 0.6271 |
| Standardized color histograms | **75.14%** | **0.6907** |
| Standardized HOG features | 20.17% | 0.1820 |
| Standardized combined features | 73.29% | 0.6803 |

Standardized color histograms achieved the strongest overall performance among the tested Logistic Regression configurations.

### Logistic Regression vs. Neural Network

Using the original color histogram features and the same development scenes:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 72.30% | 0.6271 |
| Feed-forward Neural Network | **75.70%** | **0.6625** |

The neural network achieved higher overall accuracy and macro-F1 in this experiment. However, some object classes remained difficult to recognize, showing that overall accuracy alone does not fully describe model performance.

*Note: These results are from development-scene evaluation, not from the reserved final test scene.*

## Key Findings

- The neural network slightly outperformed Logistic Regression in the initial scene-based comparison.
- Training accuracy can become very high while validation performance remains substantially lower, indicating overfitting.
- Standardizing color histogram features improved Logistic Regression performance in the feature-ablation experiment.
- HOG features alone performed poorly under the tested configuration; this does not establish that HOG is inherently ineffective.
- Some visually similar objects remain difficult to distinguish using global color histograms.
- Performance varies considerably across scenes, highlighting the importance of scene-based evaluation.

## Project Structure

```text
MLInternship_week8/
├── data/
│   └── ycbv/
│       └── test/
├── notebooks/
│   └── ycbv_neural_network_classification.ipynb
├── results/
│   ├── week8_validation_diagnostic.png
│   ├── week8_per_class_analysis.png
│   ├── week8_full_confusion_matrices.png
│   ├── week8_full_class_report.txt
│   ├── week8_grouped_cv_confusion_matrices.png
│   ├── week8_grouped_cv_report.txt
│   ├── week8_class_confusion_investigation.png
│   ├── week8_feature_ablation.png
│   └── week8_feature_ablation.txt
├── report.md
└── README.md
```

*The listed files reflect the current project artifacts; filenames may change as experiments are updated.*

## How to Run

1. Clone or download this repository.
2. Set up a Python virtual environment.
3. Install the required dependencies.
4. Ensure the YCB-Video data and the feature arrays generated during the previous week's work are available at the paths expected by the notebook.
5. Open and run `notebooks/ycbv_neural_network_classification.ipynb`.

Typical dependencies include:

- Python
- NumPy
- Matplotlib
- scikit-learn
- PyTorch

Exact package versions and data paths should match the notebook environment.

## Limitations and Next Steps

- Standardize neural-network inputs using statistics fitted only on each training fold.
- Compare Logistic Regression and the neural network under a fully controlled evaluation setup.
- Investigate misclassified examples, particularly Cracker Box and Potted Meat Can.
- Explore regularization and early stopping to reduce overfitting.
- Evaluate the selected model on the reserved scene `000059` only after development decisions are finalized.

## Detailed Report

See [`report.md`](report.md) for the extended analysis, experimental findings, limitations, and discussion.

---

**Project:** ML Internship — Week 8  
**Topic:** Neural Networks and Classical Object Classification  
**Dataset:** YCB-Video
