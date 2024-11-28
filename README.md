# CytoAutoCluster: A Semi-Supervised Approach to Cell Classification

## Project Overview
CytoAutoCluster is an innovative framework designed to improve cell population clustering in cytometry data. By leveraging semi-supervised learning, the project addresses key challenges in biomedical research, such as high-dimensional data, limited labels, and data variability, to deliver robust and interpretable clustering results.

---

## Problem Overview
Clustering cytometry data faces several challenges:
- **Lack of Labels**: The majority of cytometry data is unlabeled.
- **Data Collection Difficulty**: Sufficient labeled data is challenging to obtain.
- **Complex Features**: High-dimensional data complicates feature extraction and classification.

---

## Features
-**Semi-Supervised Learning**: Combines labeled and unlabeled data for enhanced clustering accuracy.
-**Dimensionality Reduction**: Integrates PCA and t-SNE for intuitive visualizations and data clarity.
-**High Performance**: Processes large-scale, high-dimensional datasets efficiently.
-**Actionable Insights**: Generates clear and interpretable cluster outputs for better decision-making.
-**Scalability**: Designed to handle complex biomedical datasets.

---

## Objectives
The primary goal of CytoAutoCluster is to:
1. Implement a semi-supervised framework for effective clustering.
2. Achieve high accuracy while minimizing reliance on labeled data.
3. Generate intuitive visualizations for complex data.

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

## Results
- Improved clustering accuracy using a semi-supervised approach.
- Reduced dependency on annotated data, making it suitable for data-scarce domains.
- Enhanced interpretability of noisy, high-dimensional cytometry datasets.
- Delivered actionable insights via intuitive cluster visualizations.

---

## Conclusion
CytoAutoCluster bridges the gap between data scarcity and the need for accurate, scalable clustering in cytometry. By leveraging semi-supervised learning and dimensionality reduction, this framework empowers researchers to analyze high-dimensional data effectively, delivering actionable insights into cellular populations. With a focus on precision, interpretability, and scalability, CytoAutoCluster is a step forward in biomedical data analysis, paving the way for more robust and efficient clustering methodologies.

---




## References
Levine, J.H., et al.
Data-Driven Phenotypic Dissection of AML.

Kim, B., et al.
VIME: Value Imputation and Mask Estimation.
