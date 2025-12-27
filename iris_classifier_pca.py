import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# SciKit-Learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Make the plots look nice
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# --- Step 1: Get the Data ---
print("\n--- Step 1: Loading Data ---")

# I'm using a try/except block here. If you have the CSV, great.
# If not, it automatically downloads it from the seaborn library so the code doesn't crash.
try:
    df = pd.read_csv('Iris_data.csv')
    # If the CSV has an 'Id' column, it messes up the training, so let's drop it.
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'], axis=1)
    print("Success: Loaded data from 'Iris_data.csv'")
except FileNotFoundError:
    df = sns.load_dataset('iris')
    df.rename(columns={'species': 'Species'}, inplace=True)
    print("Note: Local CSV not found. Loaded dataset from Seaborn library.")

# Quick sanity check
print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['Species'].value_counts()}")

# --- Step 2: Exploratory Data Analysis (EDA) ---
print("\n--- Step 2: Analyzing Data ---")

# Let's check for correlations. 
# We only want numeric columns because you can't calculate correlation on text (Species).
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('How features correlate with each other')
plt.show()

# Pairplot is the best way to see if the species are separable
print("Generating Pairplot... (Check the popup window)")
sns.pairplot(df, hue='Species', palette='viridis', markers=["o", "s", "D"])
plt.show()

# --- Step 3: Dimensionality Reduction (PCA) ---
# It's hard to visualize 4 dimensions (Sepal L/W, Petal L/W). 
# PCA squashes them into 2 dimensions so we can see the clusters on a flat screen.
print("\n--- Step 3: Running PCA (Visualization) ---")

X_raw = df.drop('Species', axis=1)
y = df['Species']

# PCA requires scaling first
scaler_pca = StandardScaler()
X_scaled_pca = scaler_pca.fit_transform(X_raw)

# Compress to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_pca)

# Let's plot the result
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Species'] = y
variance = pca.explained_variance_ratio_

print(f"PCA captured {sum(variance)*100:.2f}% of the information in just 2 dimensions.")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=pca_df, s=100, palette='viridis')
plt.title('Iris Dataset: 4D data projected to 2D')
plt.show()

# --- Step 4: Preprocessing for the Model ---
print("\n--- Step 4: Preparing for Training ---")

# 1. Split: 80% for training, 20% for testing
# We use 'stratify' to ensure we have an equal mix of flower types in both sets
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

# 2. Scaling: CRITICAL for KNN
# KNN calculates distance. If one feature is measured in cm and another in mm, 
# the model will get confused. Scaling fixes this.
scaler_ml = StandardScaler()
X_train_scaled = scaler_ml.fit_transform(X_train)
X_test_scaled = scaler_ml.transform(X_test) # Note: We only transform test data, never fit!

# --- Step 5: Training & Tuning (GridSearch) ---
print("\n--- Step 5: Finding the best Model (GridSearch) ---")

# We don't know what 'K' (number of neighbors) is best. Is it 3? 5? 10?
# Let's test everything from 1 to 25.
param_grid = {'n_neighbors': np.arange(1, 25)}

knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Train the model
knn_grid.fit(X_train_scaled, y_train)

best_k = knn_grid.best_params_['n_neighbors']
best_model = knn_grid.best_estimator_

print(f"Training complete! The AI decided that K={best_k} is the best setting.")

# --- Step 6: Evaluation ---
print("\n--- Step 6: Final Test Results ---")

y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy on Test Data: {accuracy*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title('Confusion Matrix (Where did the model get confused?)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# --- Step 7: Live Prediction System ---
print("\n--- Step 7: Let's try it out! ---")

def predict_flower(sepal_l, sepal_w, petal_l, petal_w):
    """
    Helper function to predict a single flower.
    """
    # Create the array
    new_flower = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    # Scale it using the SAME scaler we used for training
    new_flower_scaled = scaler_ml.transform(new_flower)
    # Predict
    prediction = best_model.predict(new_flower_scaled)
    return prediction[0]

# Testing some fake data
sample_1 = predict_flower(5.1, 3.5, 1.4, 0.2) # Should be Setosa
sample_2 = predict_flower(6.5, 3.0, 5.2, 2.0) # Should be Virginica

print(f"Flower 1 [Small] is predicted to be: {sample_1}")
print(f"Flower 2 [Big]   is predicted to be: {sample_2}")