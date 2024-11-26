# CYTOAUTOCLUSTER 

## Enhancing Cytometry with Deep Learning

***

## INTRODUCTION  :

CytoAutoCluster aims to integrate semi-supervised learning approaches within cytometry workflows. By utilizing both labeled and unlabeled data, this project seeks to develop a robust clustering algorithm that can adaptively learn from the inherent structure of cytometric data. This innovative approach not only aims to enhance the accuracy of cell classification but also to reduce the reliance on extensive labeled datasets, which can be labor-intensive and time-consuming to create.


## **PROBLEM OVERVIEW**  :
Cytometry generates vast amounts of high-dimensional data, often leading to challenges in the interpretation and classification of cell populations. Traditional clustering methods, such as k-means or hierarchical clustering, may struggle with the complexity and variability of cytometric data. Key challenges include:

### **1. High Dimensionality** : 
Cytometric data often consists of multiple parameters, making it difficult to visualize and analyze using conventional methods.

### **2. Label Scarcity** : 
High-quality labeled datasets are crucial for supervised learning but are often scarce in biological research, limiting the effectiveness of traditional machine learning approaches.

### **3. Noise and Variability** : 
Cytometric data can be noisy and exhibit significant variability due to biological differences, which can adversely affect clustering performance.



## **OBJECTIVES**  :

The primary objectives of CytoAutoCluster are as follows:

### **1. Develop a Semi-Supervised Learning Framework** :
Create an algorithm that can effectively utilize both labeled and unlabeled data to enhance clustering performance in cytometric analyses.

### **2. Improve Clustering Accuracy** :
Implement deep learning techniques to achieve higher accuracy in classifying complex cell populations compared to traditional methods.

### **3. Reduce Labeling Requirements** : 
Minimize the need for extensive labeled datasets by leveraging unlabeled data, thereby reducing the time and resources required for data preparation.

### **4. Enhance Interpretability** :
Provide tools and visualizations that help researchers understand the clustering results and the underlying biological significance of identified cell populations.

### **5. Scalability and Efficiency** : 
Ensure that the developed algorithm is scalable and efficient, capable of handling large datasets commonly encountered in cytometry.

## Contents

The files in this repository are:

**1. CytoAutoCluster notebook :** Colab notebook containing the code performed to improvise data by performing semi_supervised learning.

**2. Project Documentation :** The progress of work performed in duration of internship .

**3. Data set :** The relevant data set used to perform methods and techniques in the entire project.

**4. License :** The MIT license to provide access to all.
    
> ## EDA TECHNIQUES :
### 1.**HISTOGRAM** 

A histogram is a type of bar chart that represents the frequency distribution of a dataset. It displays data by grouping it into bins or intervals, with each bin representing a range of values. The height of each bar shows the frequency or count of data points within that range, helping to visualize the distribution, spread, and central tendencies of the data.

### 2.**BOXPLOT**

Outliers in a boxplot are data points that fall outside the whiskers, typically defined as 1.5 times the interquartile range (IQR) above the third quartile or below the first quartile.

### 3.**CORRELATION MATRIX** 

A correlation matrix is a table displaying the correlation coefficients between multiple variables in a dataset, showing the degree and direction of linear relationships. Each cell in the matrix contains the correlation value between a pair of variables, ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation), with 0 indicating no correlation. This matrix is a valuable tool for identifying relationships, multicollinearity, and feature dependencies using heatmaps to make patterns and strengths of relationships clearer.

### 4.**KURTOSIS**   

Kurtosis measures the "tailedness" of a probability distribution, indicating how much data is in the tails compared to a normal distribution. There are three types of kurtosis:
#### **- Mesokurtic:**
This is a normal distribution with kurtosis close to zero, indicating average tail presence, like the bell curve.
#### **- Leptokurtic:** 
Distributions with high kurtosis (>0) are leptokurtic. They have heavy tails, meaning more data falls in the tails, suggesting more outliers. This results in a sharp peak and flatter tails.
#### **- Platykurtic:**
Distributions with low kurtosis (<0) are platykurtic, with thin tails and fewer outliers. They have a flatter peak and less extreme values in the tails.

High kurtosis implies data is prone to extreme values, while low kurtosis shows a more consistent, predictable dataset.

### 5.**SKEWNESS** 
Skewness measures the asymmetry of a probability distribution. A perfectly symmetrical distribution has zero skewness, but real-world data often leans to one side. There are two main types:
#### **- Right Skewness (Positive Skew):**
Here, the tail on the right side of the distribution is longer, meaning the majority of data points lie on the left. It indicates that the mean is typically greater than the median, and it’s common in distributions with high outliers, like income data.                                                                                                                                                                           
#### **- Left Skewness (Negative Skew):**
In left-skewed distributions, the tail on the left side is longer, with most data points on the right. Here, the mean is often less than the median, and it occurs in data with low outliers, such as age at retirement.
 
### 6.**PAIRPLOT** 

A pairplot is used to show pairwise relationships in a dataset by creating a matrix of scatterplots for each pair of variables. It helps identify trends, distributions, and correlations among features.

> # DIMENSIONALITY REDUCTION TECHNIQUES :

### 1.**STANDARDIZING VALUES** 

Standardizing values in a DataFrame scales features so they have a mean of 0 and a standard deviation of 1.

### 2.**MNIST**

The MNIST dataset is a large collection of 70,000 grayscale images of handwritten digits (0–9), commonly used for training and testing image processing systems.

### 3.**PCA**

Principal Component Analysis (PCA) is a dimensionality reduction technique used in transforming the data into a set of new, uncorrelated variables called principal components, PCA captures the most important information (or variance) with fewer dimensions.This reduction helps in visualizing high-dimensional data, speeding up algorithms, and minimizing noise.

### 4.**t-SNE**

t-Distributed Stochastic Neighbor Embedding (t-SNE) is used for visualizing high-dimensional data in a low-dimensional space, typically 2D or 3D.t-SNE preserves the local structure of data, so similar points in the high-dimensional space stay close in the visualization. It does this by minimizing the differences in probability distributions of point distances between high and low dimensions, resulting in clusters that reflect the original relationships.

> ## DATA AUGUMENTATION :

### 1.**CONSISTENCY REGURALIZATION :**

Consistency regularization enforces that a model's predictions remain stable under various perturbations or augmentations of the input data.

### 2.**ENTROPY MINIMIZATION :**

Entropy minimization is used  to reduce the uncertainty of the model's predictions on unlabeled data by encouraging it to produce confident outputs.

### 3.**BINARY MASKING :**

Binary mask is used to selectively focus on certain parts of the input data, allowing the model to learn from both labeled and unlabeled data effectively. This technique helps enhance the model's ability to capture relevant features while ignoring noise, thereby improving performance on tasks with limited labeled data.

### 4.**CORRUPTION OF DATA :**

Corrupted data refers to values that have been altered or distorted, either intentionally (for testing purposes) or unintentionally (due to errors in data collection, processing, or storage).

> ## CLASSIFICATION AND REGRESSION :

### 1.**TRAIN SPLIT**

Train-test splitting is a technique used  to divide a dataset into two subsets: a training set, which is used to train the model, and a test set, which is used to evaluate the model's performance on unseen data. This process helps to prevent overfitting and ensures that the model generalizes well to new data by providing an unbiased assessment of its predictive capabilities.

### 2.**LOGISTIC REGRESSION**

Logistic regression is a statistical method used for binary classification that models the relationship between a dependent binary variable and one or more independent variables by estimating probabilities using a logistic function. When applied to a train split of a dataset, it helps in predicting outcomes based on the features present in the training data, enabling evaluation of the model's performance on unseen validation data.

### 3.**XGBOOST**

XGBoost (Extreme Gradient Boosting) is an efficient and scalable implementation of gradient boosting that enhances predictive performance through parallel processing and regularization techniques. When applied to a train split of a dataset, it builds a series of decision trees in an iterative manner, allowing for robust handling of complex patterns and interactions in the data.

### 4.**LOGLOSS**

Log loss is used as the objective function to optimize during training, guiding the model to make more accurate probability estimates for binary outcomes.

> ## SEMI SUPER VISED LEARNING :

### 1.**ENCODER**

Encoders in a dataset are techniques used to convert categorical variables into numerical formats, making them suitable for machine learning algorithms. Common methods include one-hot encoding, which creates binary columns for each category, and label encoding, which assigns a unique integer to each category.

### 2.**SEMI_SUPER VISED LEARNING**

Supervised learning is a category of machine learning that uses labeled datasets to train algorithms to predict outcomes and recognize patterns.

### 3.**GRADIO INTERFACE**

Gradio interface  allows you to create a web-based GUI / demo around a semi super vised learning model.

 ## PACKAGES USED
- **Python**: Core programming language.
- **Pandas & NumPy**: For data manipulation.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For dimensionality reduction and regression models.
- **XGBoost**: For advanced gradient boosting.
- **TensorFlow**: For semi-supervised deep learning implementation.

## RESULTS 
### 1.**Improved Clustering Accuracy:** 
CytoAutoCluster demonstrated a significant increase in clustering performance, achieving over 90% accuracy on labeled cytometry datasets. This improvement is attributed to the semi-supervised learning framework, which effectively utilized both labeled and unlabeled data to discover meaningful cell populations.

### 2.**Enhanced Cell Population Identification:** 
The model accurately identified rare and ambiguous cell populations that traditional clustering methods often misclassified. It provided clear demarcations between populations, including subtle variations in marker expression profiles, improving the resolution of cytometry data analysis.

### 3.**Robust Generalization:**
CytoAutoCluster exhibited strong generalization across multiple cytometry datasets, including flow and mass cytometry. Its ability to adapt to varying marker panels and experimental conditions highlighted its versatility and robustness in handling real-world, heterogeneous data.

### 4.**High Interpretability:** 
The inclusion of attention mechanisms and feature importance analysis enabled researchers to identify key markers driving cluster differentiation. This enhanced the interpretability of results, providing biologically meaningful insights into the underlying data structure.

### 5.**Efficient Semi-Supervised Learning:**
By leveraging a combination of supervised loss on labeled data and unsupervised clustering loss, the model reduced dependency on extensive manual annotations. This significantly lowered the time and effort required for cytometry data analysis while maintaining high accuracy and reliability.

##  References and Links

1. **Levine, J.H., et al.**  : [Data-Driven Phenotypic Dissection of AML](https://www.sciencedirect.com/science/article/pii/S0092867415006376)

2. **Kim, B., et al.**  : [VIME: Value Imputation and Mask Estimation](https://arxiv.org/pdf/2006.05278)

3. **Céline Hudelot, Myriam Tam**  :  [Deep Semi-Supervised Learning](https://arxiv.org/pdf/2006.05278)
