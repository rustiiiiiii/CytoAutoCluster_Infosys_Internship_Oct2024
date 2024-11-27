
# CytoAutoCluster: Revolutionizing Cytometry Data Analysis

## Project Overview

**CytoAutoCluster** is an innovative solution that aims to enhance the analysis of cytometry data by applying **semi-supervised learning** techniques. By leveraging both labeled and unlabeled data, the project seeks to improve clustering accuracy and computational efficiency in high-dimensional cytometry datasets. This approach is designed to provide insightful and actionable results in the field of biomedical research.

## Key Features

- **Semi-Supervised Clustering**: Uses both labeled and unlabeled data to enhance model performance.
- **Optimized Clustering Algorithms**: Efficiently groups cells based on their characteristics, improving accuracy.
- **High-Performance Design**: Capable of processing large datasets with speed and precision.
- **Enhanced Interpretability**: Provides clear and visual insights into clustered data.
- **Scalability**: Designed to handle complex datasets without compromising speed or accuracy.

## Why CytoAutoCluster?

**CytoAutoCluster** is an excellent choice for:
- **Biomedical Applications**: Optimizing cellular population studies in cytometry.
- **Low Data Availability**: Overcoming the challenge of limited labeled data.
- **Handling Complex Data**: Using dimensionality reduction for simplifying high-dimensional data.
- **Fast and Accurate Clustering**: Ensuring high-quality clustering even in noisy data environments.

## Problem Overview

Cytometry generates complex, high-dimensional datasets, presenting significant challenges for traditional clustering techniques:
- **High Dimensionality**: Data is difficult to interpret without reduction techniques.
- **Limited Labeled Data**: The lack of sufficient labeled data restricts the use of traditional supervised learning.
- **Noise and Variability**: Biological variability introduces significant noise, making data analysis challenging.

CytoAutoCluster addresses these challenges by combining robust machine learning techniques with semi-supervised learning, enhancing both accuracy and efficiency in the analysis of cytometry data.

## Project Objectives

1. **Develop a Semi-Supervised Learning Framework**: Combine both labeled and unlabeled data for effective clustering.
2. **Improve Clustering Accuracy**: Use deep learning to enhance the accuracy of the clustering model.
3. **Reduce the Need for Labeled Data**: Minimize dependency on labeled datasets for training the model.
4. **Provide Insights**: Generate clear and interpretable visualizations of cell groupings.
5. **Ensure Scalability**: Build a framework that can handle large-scale datasets without performance loss.

## Features Breakdown

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reduces dimensionality while retaining variance for better visualization and clustering.
- **t-SNE**: Uses non-linear dimensionality reduction to preserve local structure in the data.

### Semi-Supervised Learning Techniques
- **Consistency Regularization**: Helps improve the stability of the model with input perturbations.
- **Entropy Minimization**: Encourages more confident predictions for unlabeled data.
- **Masking Techniques**: Focuses the model’s learning capacity on the most relevant portions of the data.

## Methodology

1. **Data Preparation**:
   - Clean and preprocess high-dimensional cytometry data.
   - Conduct exploratory data analysis (EDA) using histograms, boxplots, and correlation matrices.

2. **Dimensionality Reduction**:
   - Apply PCA and t-SNE for dimensionality reduction to visualize the data better.

3. **Semi-Supervised Learning**:
   - Implement consistency regularization and entropy minimization to leverage unlabeled data.
   - Design a self-supervised function to extract features for improved clustering.

4. **Model Training**:
   - Train a series of baseline models, including Logistic Regression and XGBoost.
   - Develop a semi-supervised deep learning model to boost clustering performance.

5. **Visualization**:
   - Use Matplotlib and Seaborn to generate interpretable visualizations of the clusters.

## Technical Overview

### Core Techniques
- **Skewness & Kurtosis Analysis**: To evaluate data distribution and detect outliers.
- **Corruption and Masking**: Introduce noise into the data to make the model more robust.
- **Cluster Validation**: Metrics like Silhouette Score, Purity Score, and Adjusted Rand Index are used for validating clustering results.

### Tools & Frameworks
- **Python**: Core programming language.
- **Pandas & NumPy**: For data manipulation.
- **Matplotlib & Seaborn**: For visualizing data and clustering results.
- **Scikit-learn**: For dimensionality reduction and classification models.
- **XGBoost**: For gradient boosting models.
- **TensorFlow**: For deep learning and semi-supervised learning implementations.

## Results

- Achieved **improved clustering accuracy** using semi-supervised learning.
- Demonstrated how **less reliance on labeled data** can still yield reliable clustering results.
- Showcased **robust performance** in noisy, high-dimensional datasets.
- Provided **clear and meaningful visualizations** for interpretability.

## Future Directions

1. **Expanding to Multi-Class Clustering**: Handle clustering tasks with multiple cell populations.
2. **Hybrid Models**: Combine advanced architectures such as **autoencoders** and **graph neural networks**.
3. **Real-Time Cytometry Data Processing**: Develop real-time clustering solutions for live data analysis.
4. **Integration with Biomedical Platforms**: Extend this framework to integrate seamlessly with other biomedical data analysis platforms.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rustiiiiiii/CytoAutoCluster_Infosys_Internship_Oct2024/tree/Anuj-kadam
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Main Script**:
   ```bash
   python main.py
   ```

4. **Explore the Results** in the `output` folder, which contains visualizations and analysis outputs.

## Contact

For any inquiries or collaboration opportunities, please reach out to:

- **Email**: [anujkadam3554@gmail.com](mailto:anujkadam3554@gmail.com)
- **LinkedIn**: [Anuj Kadam](https://www.linkedin.com/in/anuj-kadam3554?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- **GitHub**: [github.com/rustiiiiiii/CytoAutoCluster_Infosys_Internship_Oct2024/tree/Anuj-kadam](https://github.com/rustiiiiiii/CytoAutoCluster_Infosys_Internship_Oct2024/tree/Anuj-kadam)

## References

1. **Levine, J.H., et al.**  
   *Data-Driven Phenotypic Dissection of AML*.  
   [Read the Paper](https://www.sciencedirect.com/science/article/pii/S0092867415006376)

2. **Kim, B., et al.**  
   *VIME: Value Imputation and Mask Estimation*.  
   [Read the Paper](https://arxiv.org/pdf/2006.05278)

---

### CytoAutoCluster: Empowering the future of cytometry with intelligent, scalable clustering. 🌟


