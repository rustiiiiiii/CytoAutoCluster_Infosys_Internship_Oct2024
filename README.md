# CYTOAUTOCLUSTER

# INTRODUCTION :

CytoAutoCluster is a project designed to integrate semi-supervised learning techniques into cytometry workflows. By utilizing both labeled and unlabeled data, this project aims to develop a robust clustering algorithm capable of identifying underlying patterns in cytometric data.

This approach offers:

* Improved accuracy in cell classification.
* Reduced dependence on large, labeled datasets, which are often costly and time-consuming to 
  produce.
By combining the strengths of machine learning and cytometry, CytoAutoCluster seeks to enhance the efficiency and effectiveness of cytometric analyses.

# Problem Overview

Cytometry data is both abundant and complex, offering rich insights into biological systems. However, its high-dimensional nature poses significant barriers to effective analysis and clustering. Traditional methods often fail to meet the demands of modern cytometry workflows.

# Key Challenges
1. High Dimensionality
   Cytometric datasets often contain dozens of parameters per sample, creating a multi- 
   dimensional space that is challenging to analyze, visualize, and interpret.

2. Limited Labeled Data
   Creating labeled datasets in cytometry is resource-intensive, requiring domain expertise and
   significant manual effort. This limitation restricts the applicability of supervised 
   learning techniques.

3. Biological Noise and Variability
   The inherent variability in biological samples and experimental setups introduces noise, 
   complicating the identification of meaningful patterns and relationships.

4. Scalability Issues
   Traditional clustering algorithms like k-means or hierarchical clustering struggle to scale 
   effectively when dealing with large, high-dimensional datasets typical of cytometry.

# Why CytoAutoCluster?

1. Revolutionizes Cytometry Analysis: Bridges gaps in traditional clustering by tackling high- 
   dimensional complexity.
2. Optimized Learning: Minimizes reliance on labeled data using advanced semi-supervised 
   techniques.
3. Enhanced Interpretability: Provides meaningful visualizations for actionable insights.
4. Scalability: Efficiently handles large, complex datasets.
   
# Methodology

1. Data Preparation:
* Cleaned and preprocessed high-dimensional cytometry datasets by addressing missing values, removing duplicates, and standardizing features to ensure consistency.
* Conducted Exploratory Data Analysis (EDA) using:
  * Histograms: Explored data distributions by grouping values into bins and observing frequencies.
  * Boxplots: Identified outliers and visualized feature variability.
  * Correlation Matrices: Assessed relationships between features to detect multicollinearity or meaningful patterns.
  * Kurtosis and Skewness: Evaluated the distribution shapes to understand asymmetry and "tailedness."
  * Pairplots: Visualized pairwise feature interactions to identify clusters or trends.
  
2. Dimensionality Reduction:
* Standardized features to avoid any single feature's dominance.
* Used Principal Component Analysis (PCA) to reduce dimensions while retaining variance for efficient downstream processing.
* Applied t-SNE for visualizing clusters in 2D or 3D spaces, preserving local data structures for better interpretability.

3. Data Augmentation:
* Consistency Regularization: Stabilized model predictions by introducing small perturbations to inputs.
* Entropy Minimization: Reduced prediction uncertainty, boosting the model's confidence in its outputs.
* Binary Masking: Focused on relevant data regions, improving the model's effectiveness with unlabeled data.
* Corruption of Data: Introduced noise and intentional changes to test and enhance model robustness.

4. Classification and Regression:

* Implemented train-test splits to separate data for model training and evaluation, ensuring reliable generalization.
* Used Logistic Regression for baseline binary classification tasks, leveraging its simplicity and probabilistic outputs.
* Employed XGBoost for gradient-boosted decision trees, providing powerful and efficient performance on complex datasets.
* Measured model accuracy with Log Loss, penalizing incorrect predictions heavily to emphasize confident outputs.

5. Semi-Supervised Learning Framework:

* Incorporated labeled and unlabeled data using:
* Encoders to transform categorical data into numerical formats for analysis.
* Binary Masking to focus attention on relevant data regions.
* Consistency Regularization for model stability.
* Entropy Minimization for confidence enhancement in outputs.
* Developed a custom self-supervised feature extraction module to uncover robust patterns.

# Gradio Interface:
* Integrated Gradio for creating interactive, web-based interfaces to test and showcase the machine learning model's functionality.

6. Visualization and Insights:
* Created detailed visualizations with Matplotlib and Seaborn for clear interpretation of clustering outcomes.
* Highlighted actionable insights from clustering to validate model performance and extract meaningful patterns.

# Results
* Enhanced Accuracy: Improved clustering performance through advanced semi-supervised learning 
  approaches.
* Reduced Label Dependency: Achieved reliable clustering outcomes while minimizing reliance on 
  labeled datasets by leveraging techniques like consistency regularization and entropy 
  minimization.
* Robust to Noise: Demonstrated resilience in handling noisy and high-dimensional data with 
  minimal preprocessing efforts.
* Intuitive Visualizations: Delivered clear, interpretable clustering visualizations, aiding 
  in data-driven insights.

 # How to Use:

1.Clone the repository:

```bash
   git clone https://github.com/HariPriya//CytoAutoCluster.git
```

2.Navigate to the project directory:

```bash
   cd CytoAutoCluster
```

3.Install dependencies:

```bash
   pip install -r requirements.txt
```

4.Run the main script:

```bash
   python CytoAutoCluster.ipynb
```

# Technologies & Frameworks
* Programming: Python
* Libraries:
    * Data Analysis: Pandas, NumPy
    * Visualization: Matplotlib, Seaborn
    * Modeling: Scikit-learn, XGBoost, TensorFlow

# Future Scope
1. Personalized Medicine: Integrate CytoAutoCluster with patient-specific data for tailored 
   disease signatures and treatment strategies.
2. Cross-Dataset Generalization: Improve model robustness by enabling generalization across 
   different datasets and conditions.
3. Automated Preprocessing: Add automated tools for outlier detection, normalization, and 
   feature selection to enhance data quality.
4. Single-Cell Genomics Integration: Combine cytometry with genomic data for deeper insights 
   into cellular phenotypes.
5. Longitudinal Data Analysis: Develop techniques to analyze cytometry data over time, aiding 
   in disease progression and treatment monitoring

# License
  This project is licensed under the MIT License. See the LICENSE file for details.

# References
1. *Levine, J.H., et al.*  : [Data-Driven Phenotypic Dissection of AML](https://www.sciencedirect.com/science/article/pii/S0092867415006376)

2. *Kim, B., et al.*  : [VIME: Value Imputation and Mask Estimation](https://arxiv.org/pdf/2006.05278)

3. *Céline Hudelot, Myriam Tam*  :  [Deep Semi-Supervised Learning](https://arxiv.org/pdf/2006.05278)
