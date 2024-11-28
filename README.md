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
- **Semi-Supervised Learning**: Combines labeled and unlabeled data for enhanced clustering accuracy.
- **Dimensionality Reduction**: Integrates PCA and t-SNE for intuitive visualizations and data clarity.
- **High Performance**: Processes large-scale, high-dimensional datasets efficiently.
- **Actionable Insights**: Generates clear and interpretable cluster outputs for better decision-making.
- **Scalability**: Designed to handle complex biomedical datasets.

---

## Objectives
The primary goal of CytoAutoCluster is to:
1. Implement a semi-supervised framework for effective clustering.
2. Achieve high accuracy while minimizing reliance on labeled data.
3. Generate intuitive visualizations for complex data.

---
## Key Components

### Classification of Labeled and Unlabeled Data  
Combining **labeled** and **unlabeled data** ensures a balance between guidance and discovery. Labeled data provides clear class boundaries, while unlabeled data reveals broader patterns, enhancing the clustering model's robustness.

### Logistic Regression  
**Logistic Regression** is a simple yet powerful classification technique. It predicts binary or multi-class outcomes using the sigmoid function and cross-entropy loss, delivering interpretable results and serving as a baseline for comparison.

### XGBoost  
**XGBoost** uses gradient boosting to construct accurate and robust models. Its ability to handle missing values, regularization features, and scalability make it ideal for large, complex datasets.

### Logistic Regression and XGBoost Loss Functions  
- **Logistic Regression Loss**: Uses binary cross-entropy to minimize the divergence between predictions and true labels.  
- **XGBoost Loss**: Optimizes gradient-boosted log loss iteratively, refining predictions for improved accuracy.

### Encoder Model  
Encoders compress high-dimensional data into compact latent representations, preserving meaningful features for downstream tasks such as clustering and classification.

### Semi-Supervised Learning  
This project incorporates **semi-supervised learning** to utilize both labeled and unlabeled data efficiently. Techniques like **pseudo-labeling** and **consistency regularization** ensure robust learning, reducing the need for extensive labeled datasets.

### Performance Metrics: Accuracy and AUROC  
- **Accuracy**: Measures the overall correctness of predictions but can be misleading with imbalanced data.  
- **AUROC (Area Under Receiver Operating Characteristic Curve)**: Evaluates the model's ability to distinguish between classes across various thresholds, providing a robust and balanced metric.

### Dimensionality Reduction: PCA and t-SNE  
- **PCA (Principal Component Analysis)**: Reduces dimensionality while preserving the most significant variance in the data.  
- **t-SNE After Encoder**: Creates visualizations by preserving local data relationships, revealing meaningful patterns and clusters.

### Gradio Integration  
**Gradio** provides an interactive, web-based interface for real-time model demonstration. Users can input custom data and observe outputs, making it easier to understand the model's functionality and performance.


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
