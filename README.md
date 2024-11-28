# CytoAutoCluster

Welcome to **CytoAutoCluster**, a project developed as part of the Infosys Internship (October 2024). This repository contains code and resources for implementing a semi-supervised machine learning pipeline for deep clustering on cytometry data, specifically leveraging the 'Levine 32 dim' dataset.

---

## Features

- **Semi-Supervised Learning**: Implements methods for training models on labeled and unlabeled cytometry data.
- **Deep Clustering**: Utilizes advanced clustering techniques to identify distinct cell populations.
- **Visualization Tools**: Includes t-SNE and correlation analysis for insightful data exploration.
- **Flexible Configurations**: Parameterized training for enhanced customization.

---

## Repository Structure

### Branch: `Swarna_Umasankar`
All work for this project resides in the `Swarna_Umasankar` branch. **Note**: This branch is independent of the `main` branch and will not be merged.

### Key Files
- **`Code.ipynb`**: Contains the implementation of the semi-supervised deep clustering pipeline.
- **`data/`**: Placeholder for the Levine 32 dim dataset (not included due to size or privacy constraints).

---

## Prerequisites

To run the code, ensure you have the following dependencies installed:

- Python 3.8+
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tensorflow` or `pytorch` (as applicable)
  - `umap-learn`

You can install the necessary packages using:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Clone the Repository
```bash
git clone https://github.com/rustiiiiiii/CytoAutoCluster_Infosys_Internship_Oct2024.git
cd CytoAutoCluster_Infosys_Internship_Oct2024
git checkout Swarna_Umasankar
```

### Step 2: Add Dataset
Place the 'Levine 32 dim' dataset in the `data/` directory.

### Step 3: Run the Notebook
Open the `Code.ipynb` notebook in Jupyter or your preferred environment and execute the cells step-by-step.

---

## Methodology

### Data Preprocessing
- Scaling of features.
- Splitting into labeled and unlabeled subsets based on the presence of labels.

### Semi-Supervised Training
- Custom self-supervised function (`self_supervised`) for learning representations.

### Clustering
- Deep clustering using state-of-the-art algorithms.

### Visualization
- t-SNE for dimensionality reduction.
- Correlation matrix analysis for feature exploration.

---

## Results
- Improved accuracy in identifying cell populations.
- Enhanced interpretability through meaningful visualizations.

---

## Contributions

Contributions to this project are welcome. If you would like to contribute:
1. Fork this repository.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request to the `Swarna_Umasankar` branch.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

Special thanks to Infosys and mentors for their guidance during the internship. The dataset used, 'Levine 32 dim,' is a crucial resource for this research.
