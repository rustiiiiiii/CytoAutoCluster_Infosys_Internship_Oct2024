# Readme for Self-Supervised Semi-Supervised Learning Framework

This project implements a pipeline for self-supervised and semi-supervised learning using a combination of feature encoders and machine learning models. The system uses unlabeled data to improve classification performance by leveraging self-supervised pretraining and semi-supervised fine-tuning.

---

## Workflow Overview

1. **Data Preprocessing**:
   - Columns like `Event`, `Time`, `file_number`, and `event_number` are dropped.
   - Labeled and unlabeled datasets are separated.
   - Features (`x_labeled`, `x_unlabeled`) and labels (`y_labeled`) are prepared.
   - Data is scaled using `StandardScaler`.

2. **Self-Supervised Pretraining**:
   - A self-supervised encoder model is trained to learn meaningful representations of the data.
   - The encoder is trained using unlabeled data with a masking and corruption mechanism.
   - Model outputs:
     - **Mask Estimation**: Predicts which features are masked.
     - **Feature Estimation**: Predicts original feature values.

3. **Supervised Training**:
   - Encoded features from the pretrained encoder are used to train two classifiers:
     - Logistic Regression
     - XGBoost
   - Performance is evaluated using **Log Loss**.

4. **Semi-Supervised Learning**:
   - A supervised neural network is trained with both labeled and unlabeled data.
   - Variance-based regularization is applied to enforce consistency on the unlabeled data.
   - Model performance is evaluated using **Accuracy** and **AUROC**.

5. **Prediction on Unlabeled Data**:
   - The encoder and the trained predictor model are used to generate class predictions for unlabeled data.

6. **Visualization**:
   - t-SNE is applied to visualize the feature space learned by the encoder.

---

## Code Walkthrough

### 1. **Preprocessing**

- **Masking and Corruption**:
  ```python
  def binary_mask(masking_probability, data_shape):
      mask = np.random.binomial(1, masking_probability, data_shape)
      return mask

  def corruption(mask, data, df_shuffle):
      corrupted_data = data * (1 - mask) + df_shuffle * mask
      return corrupted_data
  ```

- **Data Scaling**:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)
  ```

### 2. **Self-Supervised Pretraining**

- **Encoder Architecture**:
  ```python
  from keras.models import Model
  from keras.layers import Input, Dense

  def self_supervised(x_unlab, masking_probability, alpha, parameters):
      input_layer = Input(shape=(x_unlab.shape[1],))
      h = Dense(x_unlab.shape[1], activation='relu')(input_layer)
      output1 = Dense(x_unlab.shape[1], activation='sigmoid', name='mask_estimation')(h)
      output2 = Dense(x_unlab.shape[1], activation='sigmoid', name='feature_estimation')(h)
      model = Model(inputs=input_layer, outputs=[output1, output2])
      return model
  ```

- **Training the Encoder**:
  ```python
  model.compile(optimizer='rmsprop',
                loss={'mask_estimation': 'binary_crossentropy',
                      'feature_estimation': 'mean_squared_error'},
                loss_weights={'mask_estimation': 1.0, 'feature_estimation': alpha})
  model.fit(x_unlabeled_corrupted, {
      'mask_estimation': mask_new,
      'feature_estimation': x_unlab
  }, epochs=parameters['epochs'], batch_size=parameters['batch_size'])
  ```

### 3. **Supervised Training**

- **Logistic Regression**:
  ```python
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import log_loss

  log_reg = LogisticRegression(max_iter=1000)
  log_reg.fit(x_train_scaled_encoded, y_train)
  y_encoded_log_reg = log_reg.predict_proba(x_test_scaled_encoded)
  log_reg_loss = log_loss(y_test, y_encoded_log_reg)
  ```

- **XGBoost**:
  ```python
  import xgboost as xgb
  xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
  xgb_model.fit(x_train_scaled_encoded, y_train)
  ```

### 4. **Semi-Supervised Learning**

- **Supervised Neural Network**:
  ```python
  def model(input_dimension, hidden_dimension, label_dimension, activation=tf.nn.relu):
      inputs = tf.keras.Input(shape=input_dimension)
      x = layers.Dense(hidden_dimension, activation=activation)(inputs)
      x = layers.Dense(hidden_dimension, activation=activation)(x)
      y_logit = layers.Dense(label_dimension, activation=None)(x)
      y = layers.Activation('softmax')(y_logit)
      return tf.keras.Model(inputs=inputs, outputs=[y_logit, y])
  ```

- **Training with Variance Regularization**:
  ```python
  def train(feature_batch, label_batch, unlabeled_feature_batch, model, beta, supv_loss_fn, optimizer):
      with tf.GradientTape() as tape:
          y_logit, _ = model(feature_batch, training=True)
          y_loss = supv_loss_fn(label_batch, y_logit)
          unlabeled_y_logit, _ = model(unlabeled_feature_batch, training=True)
          _, variance = tf.nn.moments(unlabeled_y_logit, axes=0)
          unlabeled_y_loss = tf.reduce_mean(variance)
          total_loss = y_loss + beta * unlabeled_y_loss
      grads = tape.gradient(total_loss, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))
  ```

### 5. **Performance Metrics**

- **Accuracy and AUROC**:
  ```python
  from sklearn.metrics import accuracy_score, roc_auc_score

  def perf_metric(y_true, y_pred, metric='accuracy'):
      if metric == 'accuracy':
          return accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
      elif metric == 'AUROC':
          return roc_auc_score(y_true, y_pred, multi_class='ovr')
  ```

### 6. **Visualization**

- **t-SNE**:
  ```python
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
  data_tsne = tsne.fit_transform(df_pred)
  plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
  plt.title('t-SNE Visualization')
  plt.show()
  ```

---

## Results

- **Logistic Regression Log Loss**: `0.1037`
- **XGBoost Log Loss**: `0.0621`
- **Semi-Supervised Learning Accuracy**: `91.65%`
- **Semi-Supervised Learning AUROC**: `0.9944`

---

## Files and Directories

- `models/encoder_model1.keras`: Pretrained encoder.
- `notebooks/`: Jupyter notebooks for experimentation.
- `scripts/`: Python scripts for model training and evaluation.
- `README.md`: Documentation.

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

This pipeline demonstrates how unlabeled data can enhance performance by leveraging a combination of self-supervised and semi-supervised methods.
