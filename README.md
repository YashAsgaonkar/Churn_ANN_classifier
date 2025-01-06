# Churn ANN Classifier  

This repository contains a **Neural Network-based Classifier** built using the **Churn Dataset** to predict customer churn. The project demonstrates the use of advanced data preprocessing techniques, including **label encoding** and **one-hot encoding**, and showcases the power of an Artificial Neural Network (ANN) for classification tasks.

## About the Project  

Customer churn is a key performance indicator for many businesses. This project utilizes an ANN to classify whether a customer is likely to churn or not based on various attributes. The classifier has been deployed using **Streamlit**, providing an interactive user experience.  

### Features  
- **Data Preprocessing:** Applied `LabelEncoder` for categorical features and `OneHotEncoder` for multi-class features.  
- **ANN Architecture:** A simple but effective ANN trained for churn classification.  
- **Streamlit Deployment:** Easily accessible through a web-based application.  
- **Interpretability:** Clear preprocessing and structured ANN model for reproducibility.  

### Live Demo  
Try the classifier here: [Churn ANN Classifier App](https://churnannclassifier-yashasgaonkar.streamlit.app/)  

---

## Dataset  

The **Churn Dataset** contains the following:  
- **Customer Attributes:** Demographics, account information, and behavior.  
- **Target Variable:** Indicates whether a customer has churned (`Yes`/`No`).  

### Data Preprocessing  
1. **Label Encoding:** For binary categorical columns, e.g., `Yes/No`.  
2. **One-Hot Encoding:** For multi-class categorical columns to prevent bias.  
3. **Feature Scaling:** Normalized numerical data to improve ANN performance.  

---

## Model Architecture  

The ANN consists of:  
- **Input Layer:** Takes in the preprocessed data.  
- **Hidden Layers:** Two dense layers with ReLU activation.  
- **Output Layer:** Single neuron with sigmoid activation for binary classification.  

### Optimizer & Loss Function  
- **Optimizer:** Adam  
- **Loss Function:** Binary Cross-Entropy  

---
