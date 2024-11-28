# Cytoauto-Clustering-
CytoAuto Cluster is a project focused on applying semi-supervised learning techniques to improve the accuracy and interpretability of cell population identification in cytometry data. Cytometry, particularly flow or mass cytometry, is widely used in biological and medical research for analyzing the characteristics of cells.  

•	 This semi-supervised learning framework combines labeled and unlabeled data to train a classification model. It leverages a pre-trained encoder for feature representation and uses both supervised and unsupervised objectives during training. The model employs an iterative optimization process where the supervised loss (on labeled data) and an unsupervised loss (on unlabeled data) are minimized.
how to use the model.  

input  

1 index of the starting row of subset of  unlabeled input dataset  

2 index of the ending row of  subset of unlabeled input dataset  

output  

1 tsne visualzation of given unlabeled input dataframe  

2 labels for give unlabeled input dataframe  


