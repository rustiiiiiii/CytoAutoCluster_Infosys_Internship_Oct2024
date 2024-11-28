# **CytoAutoCluster**

## Project Overview
CytoAutoCluster is an innovative solution designed to enhance the analysis of high-dimensional cytometry data by clustering cells based on unique, identifiable characteristics. Leveraging semi-supervised learning techniques, it integrates both labeled and unlabeled data to improve clustering accuracy and computational efficiency. This cutting-edge approach provides meaningful, actionable insights into complex cellular data, advancing research in the field of biomedical science.

## Key Features
- **Semi-Supervised Learning**:  
  Harnesses the power of both labeled and unlabeled data to improve clustering accuracy, minimizing the need for extensive labeled datasets and enhancing model 
  performance.

- **Efficient Cell Grouping**:  
  Segments cells into distinct clusters based on nuanced, identifiable features, aiding in the interpretation of complex datasets and improving accuracy.

- **Optimized for Performance**:  
  Designed for large-scale, high-dimensional datasets, offering fast, precise processing with advanced clustering algorithms.

- **Interpretability**:  
  Provides clear, understandable visualizations of cluster distributions, enabling researchers to gain actionable insights and make informed decisions.

- **Scalability**:  
  Built to handle large, complex datasets efficiently, maintaining speed and accuracy without performance degradation.

## Why Choose CytoAutoCluster?

- **Biomedical Research**:  
  Optimizes cellular population studies in cytometry and related fields, streamlining complex analyses to support advancements in biomedical research.

- **Data-Scarce Applications**:  
  Effectively handles environments with limited labeled data by leveraging semi-supervised learning, overcoming the challenge of scarce data availability.

- **High-Dimensional Data**:  
  Simplifies complex, high-dimensional datasets through advanced dimensionality reduction techniques, enhancing interpretability and accuracy.

- **Precision and Efficiency**:  
  Delivers rapid, high-quality clustering results, even when faced with noisy, incomplete, or large datasets, ensuring meaningful and actionable insights.

## Problem Overview

Cytometry generates vast, high-dimensional datasets that pose significant challenges for traditional clustering methods. Key issues include:

- **High Dimensionality**:  
  Complex data is difficult to interpret without advanced dimensionality reduction techniques.

- **Scarcity of Labeled Data**:  
  Limited availability of labeled datasets due to the high cost or difficulty of obtaining them restricts the effectiveness of supervised learning approaches.

- **Noise and Variability**:  
  Biological differences introduce substantial noise and variability, impacting the accuracy of traditional clustering methods.

  These challenges require a novel solution that combines efficiency, accuracy, and minimal reliance on labeled data. CytoAutoCluster addresses these issues by 
  leveraging robust machine learning techniques, including semi-supervised learning, to enhance both clustering accuracy and computational efficiency in the 
  analysis of cytometry data.

## Project Objectives

- **Semi-Supervised Learning Framework**:  
  Utilize both labeled and unlabeled data to enhance clustering performance, ensuring robust and effective clustering.

- **Accuracy Enhancement**:  
  Leverage deep learning techniques to improve the accuracy of the clustering model, ensuring precise classification of cell groupings.

- **Labeling Efficiency**:  
  Minimize dependency on labeled datasets, reducing the need for extensive labeled data while maintaining high-quality results.

- **Interpretability**:  
  Generate clear, meaningful visualizations and insights into data clusters, enabling researchers to make informed decisions.

- **Scalability**:  
  Build a framework optimized to handle large-scale, high-dimensional datasets efficiently without compromising performance.
## Features Breakdown

### Dimensionality Reduction Techniques
- **Principal Component Analysis (PCA)**: Reduces dimensionality while retaining the variance in the data for improved visualization and clustering.
- **t-SNE**: Uses non-linear dimensionality reduction to preserve the local structure of the data in a lower-dimensional space, aiding in visualization.

### Semi-Supervised Learning Techniques
- **Consistency Regularization**: Improves model stability by applying perturbations to the input data and ensuring consistent predictions.
- **Entropy Minimization**: Encourages confident predictions for unlabeled data, helping the model make reliable decisions even with limited labeled data.
- **Binary Masking**: Focuses the model's attention on the most relevant features of the data, improving the learning process by ignoring irrelevant parts.

## Methodology

### Data Preparation:
- Cleaned and pre-processed high-dimensional cytometry data.
- Conducted exploratory data analysis (EDA) including histograms, boxplots, and correlation matrices to understand data patterns and relationships.

### Dimensionality Reduction:
- Applied **Principal Component Analysis (PCA)** and **t-SNE** to reduce the dimensionality and enhance data visualization, aiding in cluster identification.

### Semi-Supervised Learning Framework:
- Utilized **binary masking** and **consistency regularization** techniques to enhance model robustness.
- Developed a self-supervised function for feature extraction, improving clustering accuracy by leveraging both labeled and unlabeled data.

### Model Training:
- Trained baseline models such as **Logistic Regression** and **XGBoost** for initial clustering results.
- Built a custom **semi-supervised deep learning model** to boost clustering performance and improve predictive accuracy.

### Visualization:
- Generated meaningful and interpretable cluster visualizations using **Matplotlib** and **Seaborn**, helping to better understand the data and the resulting clusters.

## Technical Details

### Key Techniques
- **Kurtosis & Skewness Analysis**: Used to assess the data distribution and identify outliers.
- **Masking and Corruption**: Introduced noise into the data to make the model more robust and improve its generalization.
- **Cluster Validation Metrics**: Employed metrics such as **Silhouette Score**, **Purity Score**, and **Adjusted Rand Index** for evaluating the quality of the clustering results.

### Tools & Frameworks
- **Python**: The core programming language used for development.
- **Pandas & NumPy**: Essential for data manipulation and processing.
- **Matplotlib & Seaborn**: Utilized for creating visualizations of data and clustering results.
- **Scikit-learn**: Used for dimensionality reduction, regression models, and traditional machine learning techniques.
- **XGBoost**: Used for advanced gradient boosting models.
- **TensorFlow**: The framework for implementing deep learning and semi-supervised learning models.

## Results

- Achieved improved clustering accuracy by leveraging **semi-supervised learning** techniques.
- Reduced dependency on labeled datasets through methods like **consistency regularization** and **entropy minimization**.
- Demonstrated the model's ability to handle **noisy, high-dimensional data** with minimal preprocessing.
- Provided clear, intuitive, and interpretable visualizations for better understanding of clustering results.

## Future Scope

- **Multi-Class Clustering**: Extend capabilities to handle tasks involving multiple cell populations.
- **Advanced Architectures**: Explore hybrid models combining **autoencoders** and **graph neural networks**.
- **Real-Time Analysis**: Develop tools for **live cytometry data clustering** and real-time processing.
- **Application Integration**: Integrate CytoAutoCluster seamlessly into **biomedical platforms** for enhanced data analysis.

## How to Use
## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/rustiiiiiii/CytoAutoCluster_Infosys_Internship_Oct2024
   ```
3. Install Dependencies:
  ```bash  
  pip install -r requirements.txt
  ```
  
5. Run the Main Script:
   ```bash
   python main.py
   ```

## Contact
For any inquiries or collaboration opportunities:
- Email: [rsakashkumar@gmail.com](rsakashkumar@gmail.com)
- LinkedIn : [Akash](https://www.linkedin.com/in/akash-kumar-71667a224)
- GitHub : [Akash](https://github.com/Akasha005)

## References
- Levine, J.H., et al.
  Data-Driven Phenotypic Dissection of AML.[Read the Paper](https://www.sciencedirect.com/science/article/pii/S0092867415006376)
- Kim, B., et al.
  VIME: Value Imputation and Mask Estimation. [Read the Paper](https://arxiv.org/pdf/2006.05278)

  
