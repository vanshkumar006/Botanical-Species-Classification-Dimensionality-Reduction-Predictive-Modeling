Botanical Species Classification: PCA & Predictive Modeling

This project implements a complete Machine Learning pipeline to classify botanical species (Iris flowers) using the K-Nearest Neighbors (KNN) algorithm.It features advanced data visualization through Principal Component Analysis (PCA) and automated hyperparameter tuning to achieve high predictive accuracy.

🚀 Overview
The goal of this project is to accurately predict the species of an Iris flower based on its physical measurements: Sepal Length, Sepal Width, Petal Length, and Petal Width. The model distinguishes between three species: Setosa, Versicolor, and Virginica.

✨ Key Features
Dimensionality Reduction (PCA): Compresses 4D feature space into 2D for intuitive visual cluster analysis.
Exploratory Data Analysis (EDA): Detailed correlation heatmaps and pairplots to understand feature relationships.
Hyperparameter Tuning: Uses GridSearchCV to automatically find the optimal number of neighbors (K).
Robust Preprocessing: Implements StandardScaler to ensure feature uniformity for distance-based calculations.
Live Prediction System: Includes a ready-to-use function for real-time inference on new flower data.

🛠️ Tech Stack
Language: Python
Libraries: Pandas, NumPy, Matplotlib, Seaborn
Machine Learning: Scikit-Learn

📊 Project Workflow
Data Acquisition: Loads data from local CSV or falls back to the Seaborn library.
EDA: Visualizes feature correlations and species separation.
PCA: Projects 4-dimensional data onto 2 components (capturing ~95.8% variance).
Scaling & Splitting: Prepares data for training with a stratified 80/20 split.
Model Tuning: Runs a GridSearch across K-values (1-25).
Evaluation: Generates a Confusion Matrix and Accuracy Score.

📈 Results
Optimal K: 5
Test Accuracy: 93.33%
Variance Explained by PCA: 95.80%

💻 How to Use
Clone the repository.
Ensure you have the required libraries: pip install pandas numpy matplotlib seaborn scikit-learn
Run the script: python main.py
Use the predict_flower() function in the script to test your own measurements!





