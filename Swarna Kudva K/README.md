# CytoAutoCluster: A Semi-Supervised Approach to Cell Classification

## Introduction
CytoAutoCluster focuses on clustering cells into distinct groups based on their features using semi-supervised learning. This purely computational approach aims to overcome challenges posed by unlabeled cytometry data and high-dimensional features by efficiently utilizing labeled data and sophisticated learning techniques.

---

## Problem Overview
Clustering cytometry data faces several challenges:
- **Lack of Labels**: The majority of cytometry data is unlabeled.
- **Data Collection Difficulty**: Sufficient labeled data is challenging to obtain.
- **Complex Features**: High-dimensional data complicates feature extraction and classification.

---

## Objectives
The primary goal of CytoAutoCluster is to:
1. Learn meaningful features from a limited labeled dataset.
2. Apply learned features to classify unlabeled data efficiently.
3. Develop a computational system capable of scalable clustering.

---

## Approach: Semi-Supervised Learning
Semi-supervised learning leverages both labeled and unlabeled data:
- **Labeled Data**: Guides the learning process with explicit groupings.
- **Unlabeled Data**: Helps in discovering patterns and enhancing the model's generalizability.
This balance enables robust feature learning and classification with limited labeled resources.

---

## Dataset Selection
- **Accepted Dataset**: *Levine CytOF 32 Dimensional Data* (60% unlabeled, feature-rich for semi-supervised learning).
- **Rejected Dataset**: *CellCnn Learning Disease-Associated Cell Subsets* (focused on medical applications outside the project's scope).

---

## Key Components

### Classification of Labeled and Unlabeled Data
Combining labeled and unlabeled data balances guidance and discovery. Labeled data informs the model of class boundaries, while unlabeled data reveals broader patterns.

### Logistic Regression
A statistical model for binary or multi-class classification, logistic regression uses the sigmoid function and cross-entropy loss for efficient and interpretable predictions.

### XGBoost
XGBoost employs gradient boosting to build accurate and robust models. Its regularization capabilities and handling of missing values make it ideal for large-scale datasets.

### Logistic Regression and XGBoost Loss
- **Logistic Regression Loss**: Binary cross-entropy minimizes prediction-label divergence.
- **XGBoost Loss**: Gradient-boosted log loss optimizes iteratively for higher accuracy.

### Encoder Model
Encoders compress data into latent representations, preserving meaningful features for downstream tasks like clustering or classification.

### Semi-Supervised Learning
This paradigm uses limited labeled data alongside vast unlabeled data, employing techniques like pseudo-labeling and consistency regularization for efficient learning.

### Performance Metrics: Accuracy and AUROC
- **Accuracy**: Measures overall prediction correctness but may mislead with imbalanced datasets.
- **AUROC**: Assesses class distinction across thresholds, offering a robust performance metric.

### Dimensionality Reduction: PCA and t-SNE
- **PCA**: Simplifies data by preserving variance in fewer dimensions.
- **t-SNE After Encoder**: Visualizes clusters and patterns in encoded data by preserving local similarities.

### Gradio Integration
Gradio facilitates interactive model demonstrations with customizable web interfaces, enabling real-time input-output visualization.

---

## Implementation Steps
1. **Dataset Loading**: Load and explore the *Levine CytOF* dataset in a Google Colab environment.
2. **Preprocessing**: Handle null values, analyze correlations, and prepare data for modeling.
3. **Dimensionality Reduction**: Apply PCA and t-SNE to simplify and visualize data structure.
4. **Modeling**: Train logistic regression, XGBoost, and encoder-based semi-supervised models.
5. **Evaluation**: Use accuracy and AUROC metrics for performance assessment.
6. **Deployment**: Deploy the model interface using Gradio for real-time interaction.

---

## Visualizations and Insights
- **Histograms**: Visualize feature distributions.
- **Correlation Matrix**: Identify relationships among features.
- **Box Plots**: Examine skewness, kurtosis, and outliers.
- **t-SNE Visualizations**: Explore high-dimensional clustering.

---

## Conclusion
CytoAutoCluster demonstrates a computational approach to cell classification by leveraging semi-supervised learning. This project highlights the effective use of dimensionality reduction, advanced learning techniques, and interactive tools like Gradio for efficient and interpretable clustering.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

For more details, please refer to the [documentation](./Documentation).


To include these references in your README.md file for your GitHub repository, you can format them like this:

## References
Levine, J.H., et al.
Data-Driven Phenotypic Dissection of AML.

Kim, B., et al.
VIME: Value Imputation and Mask Estimation.
