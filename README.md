
# **CytoAutoCluster**

## **Overview**
This project focuses on implementing a self-supervised learning pipeline to work with labeled and unlabeled data. The core objective is to train models on both data types, evaluate their performance, and visualize the results using t-SNE. A user-friendly Gradio interface is added for real-time exploration of the results.

---

## **Key Features**
- **Data Preprocessing**:
  - Standardization of data for uniform scaling.
  - Splitting data into labeled and unlabeled sets for experimentation.

- **Self-Supervised Learning**:
  - Encoder-predictor architecture for handling unlabeled data.
  - Binary mask-based corruption applied to simulate missing data.

- **t-SNE Visualization**:
  - Dimensionality reduction of labeled and unlabeled data.
  - Combined visualization of clusters for insights into data patterns.

- **Performance Metrics**:
  - Accuracy and AUROC calculations for both binary and multiclass classification tasks.

- **Gradio Interface**:
  - Interactive interface to visualize predictions and t-SNE outputs in real time.

---

## **Technologies Used**
- **Python Libraries**:
  - TensorFlow/Keras
  - Scikit-learn
  - Matplotlib and Seaborn for visualizations
  - Gradio for interactive interface
- **Dimensionality Reduction**:
  - t-SNE for 2D representation of high-dimensional data.
- **Git Integration**:
  - Organized codebase for reproducibility.

---

## **Project Workflow**
1. **Data Preparation**:
   - Preprocess labeled and unlabeled datasets.
   - Encode labels and scale features for model compatibility.

2. **Model Training**:
   - Self-supervised encoder trained with masked and corrupted inputs.
   - Predictor for labeled data trained alongside.

3. **Performance Evaluation**:
   - Accuracy and AUROC metrics calculated on test data.
   - Binary and multiclass performance compared.

4. **Visualization**:
   - t-SNE used to observe data clustering.
   - Combined representation of labeled and unlabeled predictions.

5. **Gradio Interface**:
   - Developed for model interaction and prediction visualization.

---

## **How to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/rustiiiiiii/CytoAutoCluster_Infosys_Internship_Oct2024.git
   

## **t-SNE Visualization**
![t-SNE Visualization](./assets/tsne_visualization.png)

This visualization demonstrates clustering of labeled and unlabeled data, highlighting the model's capability to integrate and represent both datasets cohesively.

---

## **Future Enhancements**
- Extend the Gradio interface for more detailed analysis.
- Explore advanced unsupervised learning techniques (e.g., VAEs, GANs).
- Integrate additional datasets to generalize performance.

---

##  **Contact Detail**

**Name : Aniket Rahile**\
**Email**: aniketrahile1@gmail.com \
**GitHub**:  https://github.com/Ani-R3 

---

## **Acknowledgments**
This project leverages concepts of self-supervised learning, dimensionality reduction, and visualization to bridge the gap between labeled and unlabeled datasets. Special thanks to [Gradio](https://gradio.app/) for providing a seamless interface library.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more information.

