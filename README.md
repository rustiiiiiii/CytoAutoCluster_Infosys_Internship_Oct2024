**CytoAutoCluster: Semi-Supervised Deep Approach for Cytometry Data Analysis
Introduction**
CytoAutoCluster introduces a cutting-edge semi-supervised learning framework for data analysis of high-dimensional mass cytometry (CyTOF). Leveraging self-supervised learning, autoencoders, and robust clustering techniques, this project addresses the challenges of noise, label scarcity, and high dimensionality in cytometry datasets.
This framework is particularly suited for immunology, cancer research, and disease diagnostics applications.

**Key Objectives**
Perform comprehensive data preprocessing and exploratory analysis.
Develop and evaluate semi-supervised clustering algorithms tailored for high-dimensional cytometry data.
Benchmark clustering results against manually gated clusters for accuracy and interpretability.
Dataset Overview
**Source:** Levine et al. (2015), a publicly available benchmark dataset hosted on Cytobank.
**Characteristics:**

Cells (n): 265,627
Markers (p): 32
Manually Labeled Cells: 39% (104,184 cells)
Unlabeled Cells: 61% (161,443 cells)
Clusters (k): 14
Markers Used:

Manual Gating: CD3, CD4, CD7, CD8, HLA-DR, CD123, CD235a/b, etc.
Additional Markers: CD10, CD45RA, CD56, among others.
Methodology
Data Preprocessing
Normalization: Standardized feature distributions using StandardScaler.
Python
Copy code
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
Exploratory Data Analysis: Visualized marker distributions, cluster imbalances, and missing values using histograms and t-SNE plots.
Data Masking: Simulated partially labeled scenarios for real-world clustering evaluation.
Clustering Techniques
Autoencoder-based Dimensionality Reduction: Captured latent representations of high-dimensional data.
t-SNE Visualization: Mapped data to lower dimensions for intuitive cluster identification.
Semi-Supervised Learning
Model Architecture:
Input Dimension: 37 features
Hidden Layers: Fully connected layers with ReLU activation
Outputs: Mask estimation (Binary Cross-Entropy Loss) and feature estimation (MSE Loss)
Training Parameters:
Batch Size: 128, Epochs: 50, Optimizer: RMSprop
Final Loss Metrics:
Feature Estimation Loss: 0.6936
Mask Estimation Loss: 0.8807
Supervised Fine-Tuning
Logistic Regression: Achieved Log Loss = 0.0299, ensuring efficient and interpretable predictions.
XGBoost: Delivered superior performance with Log Loss = 0.0039, showcasing robustness in handling imbalanced data.
Performance Evaluation
Accuracy: 93.54%
AUROC: 99.09%
Visualization
Cluster Distribution: Showcased imbalances in manually gated clusters.
t-SNE Plots: Highlighted well-separated clusters post-encoder predictions.
Gradio Interface
A user-friendly Gradio-based interface was developed to streamline data visualization and prediction workflows.
Key Features:

Prediction Function: Encodes unlabeled data and predicts cell types.
t-SNE Visualization: Maps high-dimensional data into two dimensions for intuitive analysis.
Dynamic Inputs: Allows interactive selection of data subsets for analysis.
Challenges and Solutions
Noisy Data:

Challenge: Instrument variability and sample preparation introduced noise.
Solution: Implemented scaling, normalization, and data augmentation techniques.
Label Imbalance:

Challenge: Rare cell populations were underrepresented.
Solution: Used semi-supervised learning, class weights, and synthetic oversampling for better balance.
High Dimensionality:

Challenge: Thousands of features increased computational costs.
Solution: Applied dimensionality reduction with autoencoders and feature selection.
Results
Quantitative Metrics
Log Loss: Evaluated classification accuracy.
Accuracy: Measured model performance on labeled datasets.
Key Visualizations
t-SNE Plots: Demonstrated clear separations between predicted clusters.
Marker Distributions: Highlighted significant markers for cell type identification.
Future Work
Extend to Multi-Omics Data: Incorporate genomics, transcriptomics, and proteomics for integrated analysis.
Handle Diverse Cytometry Modalities: Adapt the framework to other cytometry types like flow cytometry.
Enhance Scalability: Utilize distributed computing to process datasets with millions of cells.
Domain Adaptation: Address cross-laboratory variability with transfer learning techniques.
**Conclusion**
The CytoAutoCluster project introduces a semi-supervised clustering framework that bridges the gap between labeled and unlabeled data in cytometry analysis. Its ability to identify rare cell populations and provide interpretable results positions it as a transformative tool for biomedical research and diagnostics.

**References**
Levine, J. H., et al. (2015). Data-Driven Phenotypic Dissection of AML. Cell, 162, 184–197.
Publicly available cytometry datasets (e.g., Cytobank, Kaggle).
