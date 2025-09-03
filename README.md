# Enhanced Iris Recognition System

A comprehensive machine learning system for iris recognition using advanced feature extraction techniques and multiple classification algorithms.

## Overview

This project implements an enhanced iris recognition system that achieves **88.50% accuracy** on the MMU Iris Database. The system uses sophisticated feature extraction methods combined with machine learning classifiers to identify individuals based on their iris patterns.

## Key Features

- **Advanced Feature Extraction**: Multiple feature extraction techniques, including Gabor filters, HOG-like features, wavelet transforms, and enhanced Local Binary Patterns (LBP)
- **Robust Preprocessing**: Histogram equalisation, image resizing, and noise reduction
- **Multiple ML Models**: Support Vector Machines, Random Forest, Gradient Boosting, K-Nearest Neighbours, and Logistic Regression
- **Hyperparameter Optimisation**: GridSearchCV with cross-validation for optimal model performance
- **Adaptive Cross-Validation**: Automatically adjusts CV folds based on dataset characteristics
- **Comprehensive Analysis**: Performance visualisation and detailed model comparison

## Dataset

- **Source**: MMU Iris Database
- **Total Samples**: 450 iris images
- **Classes**: 90 unique iris identities (45 persons × 2 eyes)
- **Samples per Class**: 5 images each
- **Image Format**: BMP files

## Performance Results

### Best Model: Logistic Regression
- **Test Accuracy**: 88.50%
- **Cross-Validation Score**: 81.61% ± 2.49%
- **Best Parameters**: C=10, solver='lbfgs'

### Model Comparison
| Model | Test Accuracy | CV Score | Best Parameters |
|-------|--------------|----------|-----------------|
| Logistic Regression | 88.50% | 81.61% ± 2.49% | C=10, solver='lbfgs' |
| Random Forest | 86.73% | 62.62% ± 1.10% | max_depth=None, n_estimators=200 |
| SVM Linear | 84.07% | 75.68% ± 4.33% | C=0.1 |
| SVM RBF | 79.65% | 60.55% ± 4.28% | C=10, gamma='scale' |
| KNN | 70.80% | 66.78% ± 3.15% | n_neighbors=3, weights='distance' |
| Gradient Boosting | 23.01% | 18.99% ± 1.03% | learning_rate=0.1, n_estimators=100 |

## Technical Architecture

### Feature Extraction Pipeline

1. **Image Preprocessing**
   - Resize to 128×128 pixels
   - Histogram equalisation for contrast enhancement
   - Grayscale conversion

2. **Feature Types Extracted**
   - **Histogram Features**: 512-bin intensity histogram
   - **Statistical Features**: Mean, standard deviation, skewness, kurtosis
   - **Enhanced LBP**: 256-dimensional Local Binary Pattern features
   - **Gabor Features**: 36 features from multiple orientations and frequencies
   - **HOG-like Features**: 36-bin histogram of oriented gradients
   - **Wavelet Features**: 6 features from frequency decomposition
   - **Edge Features**: Edge density and contour analysis
   - **GLCM Features**: Gray-level co-occurrence matrix approximation
   - **Hu Moments**: 7 invariant moment features

3. **Feature Processing**
   - Zero-variance feature removal
   - StandardScaler normalisation
   - SelectKBest feature selection (top 300 features)
   - Final feature vector: 300 dimensions

### Machine Learning Pipeline

1. **Data Split**: 75% training, 25% testing with stratification
2. **Cross-Validation**: Adaptive 3-fold stratified CV
3. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
4. **Model Evaluation**: Accuracy, confusion matrices, feature importance

## Installation & Requirements

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import cv2
```

## Usage

1. **Setup Dataset Path**
   ```python
   dataset_path = "/path/to/MMU-Iris-Database"
   ```

2. **Run Feature Extraction**
   ```python
   # Extract features from all iris images
   features = extract_enhanced_features(image_path)
   ```

3. **Train Models**
   ```python
   # Train multiple models with hyperparameter tuning
   best_models = train_all_models(X_train, y_train)
   ```

4. **Evaluate Performance**
   ```python
   # Get accuracy and detailed metrics
   accuracy = evaluate_model(best_model, X_test, y_test)
   ```

## Key Improvements Over Basic Systems

- **7.02 percentage point improvement** over baseline (81.48% → 88.50%)
- Advanced feature extraction increases discriminative power
- Histogram equalisation improves image quality consistency
- Feature selection reduces overfitting and computational complexity
- Hyperparameter optimisation ensures optimal model configuration
- Adaptive cross-validation handles class imbalance effectively

## System Specifications

- **Total Features Extracted**: 872 dimensions
- **Features After Filtering**: 610 dimensions
- **Selected Features**: 300 dimensions
- **Training Samples**: 337
- **Test Samples**: 113
- **Classes**: 90
- **Cross-Validation**: 3-fold stratified

## Limitations & Considerations

1. **Small Dataset**: Only 5 samples per class limit generalisation
2. **Controlled Environment**: A Dataset from a single source may not generalise to real-world conditions
3. **Class Imbalance Handling**: Small class sizes require careful cross-validation strategy
4. **Computational Complexity**: Feature extraction is computationally intensive

## Future Enhancements

- Deep learning approaches (CNNs)
- Data augmentation to increase sample size
- Real-time processing optimisation
- Multi-database evaluation
- Ensemble methods combining multiple feature types

## Performance Analysis

The system demonstrates robust performance with:
- **Good Generalisation**: CV scores align reasonably with test accuracy
- **Feature Importance**: Random Forest analysis shows distributed feature importance
- **Model Diversity**: Different algorithms excel in different aspects
- **Stability**: Low standard deviation in cross-validation scores

## Conclusion

This enhanced iris recognition system successfully achieves high accuracy through sophisticated feature engineering and machine learning techniques. The combination of multiple feature extraction methods with optimised classification algorithms provides a solid foundation for iris-based biometric identification.

---

**Performance Level**: Good (88.50% accuracy)  
**Improvement**: +7.02 percentage points over baseline  
**Best Model**: Logistic Regression with L-BFGS optimization
