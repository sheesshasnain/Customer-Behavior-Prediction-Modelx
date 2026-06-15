import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

print("\nChecking dataset...")
df = pd.read_csv("dataset.csv")

print("\nShape of Dataset:")
print("Rows, Columns:", df.shape)

print("\nPreprocessing: Handling Missing Values...")
data_filled = df.copy()
data_filled = data_filled.fillna(data_filled.mean(numeric_only=True))

def remove_noise(df):
    return df

data_cleaned = remove_noise(data_filled)
data_cleaned = data_cleaned.drop_duplicates()

cleaned_file_path = "cleaned_dataset.csv"
data_cleaned.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data saved to {cleaned_file_path}")

print("\nCorrelation Matrix:")  
plt.figure(figsize=(22, 22))  
sns.heatmap(data_cleaned.corr(numeric_only=True), annot=True, cmap='cividis', fmt='.2f') 
plt.title("Correlation Heatmap")  
plt.show()  

print("\nBox Plots of Features:")
plt.figure(figsize=(18, 12))
sns.boxplot(data=data_cleaned.select_dtypes(include=np.number))
plt.xticks(rotation=90)
plt.title("Box Plot of Numeric Features")
plt.show()

print("\n--- Histograms of Numeric Features ---")
numeric_columns = data_cleaned.select_dtypes(include=np.number).columns
num_cols = len(numeric_columns)
cols_per_row = 3
rows = (num_cols + cols_per_row - 1) // cols_per_row
fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, rows * 4))
axes = axes.flatten()

for i, column in enumerate(numeric_columns):
    sns.histplot(data_cleaned[column], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f"Histogram of {column}")

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


print("\nNormalization:")
scaler_minmax = MinMaxScaler()
data_cleaned[numeric_columns] = scaler_minmax.fit_transform(data_cleaned[numeric_columns])

print("\nData After Normalization:")
print(data_cleaned.head())

categorical_cols = data_cleaned.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data_cleaned[col] = le.fit_transform(data_cleaned[col])
    label_encoders[col] = le

data_cleaned['feature_sum'] = data_cleaned[numeric_columns].sum(axis=1)
data_cleaned['feature_mean'] = data_cleaned[numeric_columns].mean(axis=1)

if data_cleaned.shape[1] > 20 and 'target' in data_cleaned.columns:
    pca = PCA(n_components=0.95)
    pca_result = pca.fit_transform(data_cleaned.drop('target', axis=1))

X = data_cleaned.drop('target', axis=1)
y = data_cleaned['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

metrics = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    metrics.append([name, acc, prec, rec, f1])

    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print(classification_report(y_test, y_pred))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='cividis')
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

print("\n--- Cross-Validation Scores (5-fold) ---")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} CV Accuracy Scores: {scores}")
    print(f"Mean CV Accuracy: {scores.mean():.4f}\n")

metrics_df = pd.DataFrame(metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
metrics_df.set_index('Model', inplace=True)
metrics_df.plot(kind='bar', figsize=(10, 6))

plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

rf = models["Random Forest"]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))

print(f"\n\n")
from matplotlib.cm import get_cmap
cmap = get_cmap('Spectral') 
colors = [cmap(i / len(importances)) for i in range(len(importances))] 

plt.title("Feature Importances") 
plt.bar(range(X.shape[1]), importances[indices], color=[colors[i] for i in indices])  
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout() 
plt.show()