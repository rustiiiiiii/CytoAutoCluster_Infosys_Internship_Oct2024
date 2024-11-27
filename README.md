# CytoAutoCluster 📊🔬  
Enhancing Cytometry with Deep Learning  

## 🚀 Project Overview  
In the data-intensive realm of modern science, **CytoAutoCluster** emerges as a cutting-edge solution for clustering cells based on unique, identifiable characteristics. Leveraging **semi-supervised learning techniques**, this project aims to improve clustering accuracy and computational efficiency, providing meaningful insights into cellular data.  

---

## 🌟 Key Features  

- **Semi-Supervised Learning**: Harnesses the power of labeled and unlabeled data to enhance clustering accuracy.  
- **Efficient Cell Grouping**: Segments cells into distinct clusters based on nuanced features.  
- **Optimized for Performance**: Designed to handle large datasets with speed and precision.  
- **Interpretability**: Visualizes and explains cluster distributions with clarity.  
- **Scalability**: Built to manage large, complex datasets without compromising performance.  

---

## 🔎 Why Choose CytoAutoCluster?  

CytoAutoCluster is ideal for:  

- **Biomedical Research**: Streamlining cellular population studies in cytometry.  
- **Data-Scarce Applications**: Overcoming limitations of labeled datasets.  
- **High-Dimensional Data**: Simplifying complex datasets with advanced dimensionality reduction techniques.  
- **Precision and Efficiency**: Designed for meaningful and rapid clustering, even in noisy datasets.  

---

## 📚 Problem Overview  

Cytometry produces vast, high-dimensional datasets, posing challenges for traditional clustering methods. Key issues include:  

- **High Dimensionality**: Complex data is hard to interpret.  
- **Scarcity of Labeled Data**: Limited labeled datasets restrict supervised learning.  
- **Noise and Variability**: Biological differences introduce significant noise.  

These challenges necessitate a novel approach combining efficiency, accuracy, and minimal reliance on labeled data.  

---

## 🎯 Objectives  

- **Semi-Supervised Framework**: Leverage both labeled and unlabeled data for robust clustering.  
- **Accuracy Enhancement**: Employ deep learning techniques for improved classification.  
- **Labeling Efficiency**: Minimize dependency on labeled data.  
- **Interpretability**: Provide meaningful visualizations and insights into cell groupings.  
- **Scalability**: Ensure the framework handles large datasets efficiently.  

---

## 🛠️ Features Breakdown  

### Dimensionality Reduction Techniques  
- **PCA**: Captures variance and reduces dimensions for better visualization.  
- **t-SNE**: Preserves local data structure in low-dimensional space.  

### Semi-Supervised Learning  
- **Consistency Regularization**: Maintains model stability with input perturbations.  
- **Entropy Minimization**: Encourages confident predictions on unlabeled data.  
- **Binary Masking**: Focuses learning on relevant parts of the data.  

---

## 🧬 Methodology  

### Data Preparation:  
- Cleaned and pre-processed high-dimensional cytometry data.  
- Performed exploratory data analysis (EDA) using histograms, boxplots, and correlation matrices.  

### Dimensionality Reduction:  
- Applied PCA and t-SNE to reduce dimensions and visualize clusters.  

### Semi-Supervised Learning Framework:  
- Utilized binary masking and consistency regularization.  
- Developed a self-supervised function for feature extraction and robust clustering.  

### Model Training:  
- Trained baseline models: Logistic Regression and XGBoost.  
- Built a custom semi-supervised deep learning model to improve clustering accuracy.  

### Visualization:  
- Generated clear cluster visualizations using Matplotlib and Seaborn for better interpretability.  

---

## 🧪 Technical Details  

### 📈 Key Techniques  
- **Kurtosis & Skewness Analysis**: To assess data distribution and outliers.  
- **Masking and Corruption**: Introduced noise for robust learning.  
- **Cluster Validation Metrics**: Used Silhouette Score, Purity Score, and Adjusted Rand Index.  

### 🛠️ Tools & Frameworks  
- **Python**: Core programming language.  
- **Pandas & NumPy**: For data manipulation.  
- **Matplotlib & Seaborn**: For data visualization.  
- **Scikit-learn**: For dimensionality reduction and regression models.  
- **XGBoost**: For advanced gradient boosting.  
- **TensorFlow**: For semi-supervised deep learning implementation.  

---

## 📊 Results  

- Achieved higher clustering accuracy by leveraging semi-supervised learning techniques.  
- Reduced dependency on labeled datasets by utilizing consistency regularization and entropy minimization.  
- Demonstrated efficient handling of noisy, high-dimensional data with minimal preprocessing.  
- Provided intuitive and interpretable clustering visualizations.
- **Logistic Regression Log Loss**: `0.1037`
- **XGBoost Log Loss**: `0.0621`
- **Semi-Supervised Learning Accuracy**: `91.65%`
- **Semi-Supervised Learning AUROC**: `0.9944`

---
## Dependencies

- Python 3.7+
- TensorFlow 2.x
- Scikit-learn
- XGBoost
- Matplotlib
- Pandas
- NumPy
---


## 📜 Future Scope  

- **Multi-Class Clustering**: Extend to tasks involving multiple cell populations.  
- **Advanced Architectures**: Explore hybrid models combining autoencoders and graph neural networks.  
- **Real-Time Analysis**: Develop tools for live cytometry data clustering.  
- **Application Integration**: Integrate the framework with biomedical platforms for seamless analysis.  

---

## 💻 How to Use  

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/Amanjayswal960/CytoAutoCluster.git
 
2. **Navigate to the project directory:**  
   ```bash
   cd CytoAutoCluster
   ```  
3. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  
4. **Run the main script:**  
   ```bash
   python CytoAutoCluster.ipynb
   ```  

Access visualizations and results in the `output` directory.  

---

## 📬 Contact  

For queries or collaborations, please reach out via:  

- **Email**: [amanjayswal960@gmail.com](mailto:amanjayswal960@gmail.com)  
- **LinkedIn**: [Aman Jayswal](https://linkedin.com/in/aman-jayswal)  
- **GitHub**: [github.com/Amanjayswal960](https://github.com/Amanjayswal960) 

---

## 📜 References  

1. **Levine, J.H., et al.**  
   *Data-Driven Phenotypic Dissection of AML.*  
   [Read the Paper](#)  

2. **Kim, B., et al.**  
   *VIME: Value Imputation and Mask Estimation.*  
   [Read the Paper](#)  

---

**CytoAutoCluster**: Bridging the gap between data scarcity and efficient deep clustering. 🌟  
``
---



This pipeline demonstrates how unlabeled data can enhance performance by leveraging a combination of self-supervised and semi-supervised methods.
