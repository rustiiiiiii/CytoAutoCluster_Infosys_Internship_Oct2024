# CytoAutoCluster: A Semi-Supervised Deep Learning Framework for Cytometry Data Analysis

## Overview
CytoAutoCluster is an advanced deep learning framework designed to tackle challenges in cytometry data analysis, including high dimensionality, missing labels, and noise. It leverages semi-supervised learning, dimensionality reduction, and clustering techniques with a focus on interactive data exploration using Gradio.

---

## Repository Structure

```plaintext
main/
├── Mohana/                        # Supporting folder for custom scripts or files
├── CytoAutoClusterCode.docx       # Detailed documentation and project notes
├── ENCODER_AND_GRADIO.ipynb       # Jupyter notebook combining encoder training and Gradio interface
├── README.md                      # Project documentation (this file)
├── cytoAutoCluster_BeforeEncoder.ipynb  # Notebook for initial data exploration and preprocessing
├── LICENSE                        # Licensing information for the project


---

## Key Features
1. **Data Cleaning and Preprocessing**:
   - Removal of irrelevant columns (e.g., `Time`, `Cell_length`).
   - Imputation of missing feature values using column means.
   - Normalization with `StandardScaler`.

2. **Dimensionality Reduction**:
   - **PCA**: Reduced data dimensions while retaining 95% variance.
   - **t-SNE**: Generated 2D/3D visualizations for cluster analysis.

3. **Self-Supervised Learning**:
   - Trained an autoencoder to handle 30% masked data.
   - Extracted robust latent features for clustering.

4. **Clustering**:
   - Algorithms: **k-Means** and **Hierarchical Clustering**.
   - Evaluation Metrics: Adjusted Rand Index, Silhouette Score.

5. **Gradio Interface**:
   - Upload datasets and preprocess them interactively.
   - Adjust clustering parameters (e.g., number of clusters, masking ratio).
   - Visualize PCA and t-SNE clustering results in real-time.
   - Monitor performance metrics and reconstruction loss.

---

## Dataset Information
- **Dataset Name**: Levine32Dimensional
- **Source**: Kaggle
- **Properties**:
  - **Rows**: 265,627 (cells)
  - **Columns**: 32 markers (features)
  - **Labeled Data**: 104,184 rows (39%)
  - **Unlabeled Data**: 161,443 rows (61%)

### Dataset Challenges
- High dimensionality makes clustering computationally expensive.
- A significant proportion (61%) of missing labels is ideal for semi-supervised learning approaches.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MOHANAL/CytoAutoCluster.git
   cd CytoAutoCluster
2.Install required dependencies:
  ```bash
  pip install -r requirements.txt



