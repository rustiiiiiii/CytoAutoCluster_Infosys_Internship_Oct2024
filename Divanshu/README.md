# CytoAutoCluster 📊🔬  
**Enhancing Cytometry with Deep Learning**  
*Contributed by: Divanshu*

## 🚀 Project Overview  
CytoAutoCluster is a cutting-edge solution for clustering cells based on unique, identifiable characteristics. Leveraging advanced deep learning and semi-supervised learning techniques, this project improves clustering accuracy and computational efficiency, providing meaningful insights into cellular data.

## 🌟 Key Features  
- **Semi-Supervised Learning:** Uses labeled and unlabeled data to improve clustering accuracy while minimizing the need for extensive labeled datasets.
- **Efficient Cell Grouping:** Segments cells into distinct clusters based on nuanced features, aiding in the interpretation of complex datasets.
- **Optimized for Performance:** Designed for large-scale datasets, offering fast and precise processing.
- **Interpretability:** Provides clear and understandable visualizations of clusters, enabling researchers to make informed decisions.
- **Scalability:** Built to handle high-dimensional datasets without performance degradation.

## 🔎 Why Choose CytoAutoCluster?  
CytoAutoCluster is perfect for:  
- **Biomedical Research:** Facilitates cellular population studies in cytometry and related fields.  
- **Data-Scarce Applications:** Effective in environments with limited labeled data.  
- **High-Dimensional Data:** Reduces complexity through advanced dimensionality reduction methods.  
- **Precision and Efficiency:** Offers meaningful, rapid clustering results even when faced with noisy or incomplete data.

## 📚 Problem Overview  
Cytometry often results in vast, high-dimensional datasets that are difficult to analyze using traditional clustering methods. Key challenges include:  
- **High Dimensionality:** Complex data that can be difficult to interpret.  
- **Scarcity of Labeled Data:** Limitations due to the high cost or difficulty in obtaining labeled datasets.  
- **Noise and Variability:** Biological differences introduce significant noise, affecting the accuracy of clustering.  

These challenges call for a new approach—one that combines efficiency, accuracy, and minimal reliance on labeled data.

## 🎯 Objectives  
- **Semi-Supervised Framework:** Utilize both labeled and unlabeled data to enhance clustering performance.
- **Accuracy Enhancement:** Improve clustering accuracy with deep learning techniques.
- **Labeling Efficiency:** Reduce the dependency on labeled datasets while maintaining high quality.
- **Interpretability:** Provide clear visualizations and insights into data clusters.
- **Scalability:** Ensure the system is optimized for large datasets.

## 🛠️ Features Breakdown  
- **Dimensionality Reduction Techniques:**  
  - **PCA**: Reduces dimensions while maintaining key data variance.  
  - **t-SNE**: Helps preserve the local structure of the data in lower dimensions.  
- **Semi-Supervised Learning:**  
  - **Consistency Regularization**: Stabilizes model predictions through input perturbations.  
  - **Entropy Minimization**: Ensures confident predictions for unlabeled data.  
  - **Binary Masking**: Focuses the model's attention on relevant features.

## 🧬 Methodology  
- **Data Preparation:**  
  - Cleaned and pre-processed high-dimensional cytometry data.  
  - Conducted exploratory data analysis (EDA) including histograms, boxplots, and correlation matrices.
  
- **Dimensionality Reduction:**  
  - Applied **PCA** and **t-SNE** for dimensionality reduction and visualization.

- **Semi-Supervised Learning Framework:**  
  - Used **binary masking** and **consistency regularization** to improve model robustness.  
  - Developed a self-supervised function for feature extraction and clustering.

- **Model Training:**  
  - Trained baseline models (e.g., Logistic Regression, XGBoost).  
  - Built a custom deep learning model to further improve clustering.

- **Visualization:**  
  - Generated meaningful cluster visualizations using **Matplotlib** and **Seaborn**.

## 🧪 Technical Details  
- **Key Techniques:**  
  - **Kurtosis & Skewness Analysis**: To assess the distribution and identify outliers.  
  - **Cluster Validation Metrics**: Silhouette Score, Purity Score, Adjusted Rand Index.  

- **Tools & Frameworks:**  
  - **Python**: Core language for development.  
  - **Pandas & NumPy**: For data manipulation and processing.  
  - **Scikit-learn**: For dimensionality reduction and traditional machine learning models.  
  - **TensorFlow**: For implementing the deep learning framework.

## 📊 Results  
- Achieved **higher clustering accuracy** by leveraging semi-supervised learning techniques.  
- Reduced dependency on labeled datasets through **consistency regularization** and **entropy minimization**.  
- Demonstrated the model's ability to handle noisy, high-dimensional data with minimal preprocessing.  
- Provided **intuitive visualizations** for clearer interpretation of clustering results.

## 📜 Future Scope  
- **Multi-Class Clustering:** Expand capabilities to handle tasks with multiple cell populations.  
- **Advanced Architectures:** Explore hybrid models integrating **autoencoders** and **graph neural networks**.  
- **Real-Time Analysis:** Develop tools for live cytometry data clustering.  
- **Application Integration:** Integrate CytoAutoCluster into biomedical platforms for seamless analysis.

## 💻 How to Use  
1. **Clone the repository**:  
    ```bash  
    git clone https://github.com/rustiiiiiii/CytoAutoCluster_Infosys_Internship_Oct2024.git  
    ```  
2. **Navigate to the project directory**:  
    ```bash  
    cd CytoAutoCluster  
    ```  
3. **Install dependencies**:  
    ```bash  
    pip install -r requirements.txt  
    ```  
4. **Run the main script**:  
    ```bash  
    python main.py  
    ```  
5. **Access results**: Results and visualizations will be saved in the `output/` directory.

## 📬 Contact  
For any inquiries or collaboration opportunities, feel free to reach out:  
- **Email**: divanshuvaish96@gmail.com  
- **LinkedIn**: [Divanshu](https://www.linkedin.com/in/divanshu-658a18217/)  
- **GitHub**: [Divanshu](https://github.com/Divanshu7)

## 📜 References  
- Levine, J.H., et al. *Data-Driven Phenotypic Dissection of AML.* [Read the Paper](https://doi.org/...)
- Kim, B., et al. *VIME: Value Imputation and Mask Estimation.* [Read the Paper](https://doi.org/...)

---
### 🔐 Recording:
  [PLAY](https://drive.google.com/file/d/1pw3Mf88lYRvW13J8E8Nxx-MlDdiGOabw/view)          [DOWNLOAD](https://drive.google.com/uc?export=download&id=1pw3Mf88lYRvW13J8E8Nxx-MlDdiGOabw)

### 🔐 Terms of Use  
**License**: This project is licensed under the [MIT License](https://github.com/rustiiiiiii/CytoAutoCluster_Infosys_Internship_Oct2024/blob/Divanshu/Divanshu/LICENCE), but unauthorized use, distribution, or modification of this code for commercial purposes is strictly prohibited without prior written consent.

This README structure is ready to be added to your repository to showcase the details of your project and to inform users about how to properly use and contribute to the project.
