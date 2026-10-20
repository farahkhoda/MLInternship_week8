# Week 8 Report — Neural Networks and Deep Learning

## 1. Objective

The main objective of Week 8 was to understand the fundamentals of neural networks and implement a simple neural-network classifier for the YCB-Video dataset.

The practical work focused on:
- Building a feed-forward neural network using PyTorch.
- Understanding the input layer, hidden layer, ReLU activation, output layer, loss function, and optimizer.
- Comparing a neural network with a Logistic Regression baseline.
- Evaluating generalization across different scenes.
- Investigating how different feature representations affect classification performance.

## 2. Dataset and Features

The dataset contains 2,700 samples from YCB-Video scenes `000048`–`000059`, covering nine object classes.

The following handcrafted feature representations were used:

| Feature Representation | Dimensions | Description |
|---|---:|---|
| Color histogram | 4,096 | HSV color histogram |
| HOG | 1,764 | Histogram of Oriented Gradients extracted from masked grayscale images |
| Combined | 5,860 | Concatenation of color and HOG features |

The nine classes are Master Chef Can, Cracker Box, Sugar Box, Tomato Soup Can, Tuna Fish Can, Potted Meat Can, Pitcher Base, Bleach Cleanser, and Power Drill.

Scene `000059` was reserved for final testing. Development experiments used scenes `000048`–`000058`.

## 3. Neural Network Architecture

A fully connected feed-forward neural network was implemented using PyTorch.

```text
Input: 4,096 features
        ↓
Dense Layer: 128 neurons
        ↓
ReLU Activation
        ↓
Output Layer: 9 classes
```

The model contains 525,577 trainable parameters.

Training configuration:
- Loss function: Cross-Entropy Loss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Random seed: 42

The output layer produces nine logits, and the class with the highest score is selected as the prediction.

## 4. Training Behavior

During the initial training experiment, training accuracy increased to approximately 99.68%, while validation accuracy stopped improving and validation loss increased.

This behavior indicates overfitting: the network learned the training samples very well but did not generalize equally well to the validation scene.

The validation scene contained only two classes, Master Chef Can and Tuna Fish Can. Therefore, its results were useful for diagnosing training behavior but were not sufficient to evaluate performance across all nine classes.

## 5. Scene-Based Evaluation

Leave-One-Scene-Out (LOSO) cross-validation was performed using scenes `000048`–`000058`. In each fold, one scene was held out for evaluation while the remaining development scenes were used for training.

### Logistic Regression vs. Neural Network

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression — original color features | 72.30% | 0.6271 |
| Neural Network — original color features | 75.70% | 0.6625 |

The neural network improved accuracy by 3.40 percentage points and macro F1 by 0.0354 compared with the original-color Logistic Regression baseline.

However, this comparison does not establish that neural networks are always superior. Further controlled experiments are needed to compare models under consistent preprocessing and tuning procedures.

## 6. Feature Representation Experiments

Four feature representations were evaluated using Logistic Regression with the same LOSO strategy.

| Representation | Dimensions | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Original color features | 4,096 | 72.30% | 0.6271 |
| Standardized color features | 4,096 | 75.14% | 0.6907 |
| Standardized HOG features | 1,764 | 20.17% | 0.1820 |
| Standardized combined features | 5,860 | 73.29% | 0.6803 |

Standardized color features achieved the best overall performance, with 75.14% accuracy and a macro F1 score of 0.6907.

Standardization improved the original color baseline by 2.84 percentage points in accuracy. HOG features alone performed poorly in this experiment, while combining HOG and color features improved recall for some classes but did not outperform standardized color features overall.

The scaler was fitted within each training fold to avoid data leakage from the held-out scene.

## 7. Error Analysis

Two classes presented particularly difficult classification challenges.

### Cracker Box
- Original-color Logistic Regression recall: 0%.
- Standardized-color Logistic Regression recall: 50.0%.
- Combined-feature Logistic Regression recall: 54.7%.

Standardization and combined features improved recognition of this class.

### Potted Meat Can
- Original-color Logistic Regression recall: 0%.
- Neural-network recall: 0%.
- Standardized-color Logistic Regression recall: 14.7%.
- Combined-feature Logistic Regression recall: 2.0%.

Potted Meat Can remained difficult to distinguish from other objects. Possible contributing factors include visual similarity, limitations of global color histograms, and the limited spatial information in the current features. These explanations require further investigation.

## 8. Key Findings

1. A working feed-forward neural-network classifier was implemented using PyTorch.
2. The initial training experiment demonstrated a clear overfitting pattern.
3. LOSO cross-validation provided a more meaningful assessment of generalization across scenes.
4. The neural network outperformed the original-color Logistic Regression baseline on accuracy and macro F1 in the current experiment.
5. Standardized color histograms with Logistic Regression achieved the best overall results among the tested feature representations.
6. HOG alone performed poorly in the current setup, while combining HOG and color improved some class-specific results.
7. Cracker Box and Potted Meat Can remain important targets for further error analysis.

## 9. Limitations

- The models use handcrafted features rather than features learned directly from raw images.
- The initial validation scene contained only two of the nine classes.
- LOSO results cover development scenes and do not represent the reserved final test scene.
- The neural network and Logistic Regression have not been exhaustively tuned under an identical hyperparameter-search protocol.
- Performance may depend on preprocessing, image masks, feature scaling, and training settings.

## 10. Next Steps

1. Evaluate the neural network using standardized color features, with scaling fitted separately inside each training fold.
2. Compare the neural network and Logistic Regression using identical LOSO folds and metrics.
3. Inspect correctly and incorrectly classified examples of Cracker Box and Potted Meat Can.
4. Test a small number of justified regularization strategies, such as weight decay or early stopping.
5. Finalize the model using development data, then evaluate it on the reserved scene `000059`.

## Conclusion

Week 8 established a neural-network baseline and a scene-based evaluation workflow for YCB-Video classification. The neural network improved upon the original-color Logistic Regression baseline, but the feature-ablation experiment showed that standardized color histograms with Logistic Regression achieved the strongest overall performance among the tested combinations.

The results also demonstrated that additional features do not necessarily improve overall performance and that classification quality varies substantially across object classes. Further controlled experiments and class-specific error analysis are needed before the final evaluation on scene `000059`.
