import pandas as pd  # used for handling data, like reading CSV files and working with tables
import numpy as np  # used for numerical operations and arrays
import matplotlib.pyplot as plt  # helps in making plots and charts
import seaborn as sns  # makes better-looking visualizations
from sklearn.model_selection import train_test_split, cross_val_score  # used to split data and check model accuracy
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler  # used to prepare and scale data
from sklearn.impute import SimpleImputer  # helps fill in missing values
from sklearn.ensemble import RandomForestClassifier  # one of the machine learning models we'll use
from sklearn.linear_model import LogisticRegression  # another ML model, simpler than Random Forest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix  # used to check how well our model works
from sklearn.decomposition import PCA  # used to reduce the number of features
import joblib  # allows saving and loading models
import warnings  # helps to hide warnings
warnings.filterwarnings('ignore')  # ignore any warning messages

print("\nChecking dataset...")  # just a message showing we're loading the dataset
df = pd.read_csv("dataset.csv")  # reading the CSV file into a DataFrame

print("\nShape of Dataset:")  # print the shape of the dataset
print("Rows, Columns:", df.shape)  # shows number of rows and columns

print("\nPreprocessing: Handling Missing Values...")  # message about cleaning missing values
data_filled = df.copy()  # create a copy of the original data
data_filled = data_filled.fillna(data_filled.mean(numeric_only=True))  # fill missing values with the column mean

def remove_noise(df):  # this function is for removing noise, but it's empty for now
    return df  # returns the same data for now

data_cleaned = remove_noise(data_filled)  # calling the dummy noise removal function
data_cleaned = data_cleaned.drop_duplicates()  # remove duplicate rows

cleaned_file_path = "cleaned_dataset.csv"  # set file name for cleaned data
data_cleaned.to_csv(cleaned_file_path, index=False)  # save cleaned data to a CSV file
print(f"\nCleaned data saved to {cleaned_file_path}")  # print confirmation message

print("\nCorrelation Matrix:")  # now we check how different features are related to each other
plt.figure(figsize=(22, 22))  # set the figure size for heatmap
sns.heatmap(data_cleaned.corr(numeric_only=True), annot=True, cmap='cividis', fmt='.2f')  # draw heatmap with correlation values
plt.title("Correlation Heatmap")  # set title
plt.show()  # show the plot

print("\nBox Plots of Features:")  # next, draw box plots to find outliers
plt.figure(figsize=(18, 12))  # set figure size
sns.boxplot(data=data_cleaned.select_dtypes(include=np.number))  # draw boxplots for numeric columns
plt.xticks(rotation=90)  # rotate x-axis labels to fit properly
plt.title("Box Plot of Numeric Features")  # title of the plot
plt.show()  # show the box plot

print("\n--- Histograms of Numeric Features ---")  # draw histograms to see data distribution
numeric_columns = data_cleaned.select_dtypes(include=np.number).columns  # get numeric columns
num_cols = len(numeric_columns)  # count how many numeric columns
cols_per_row = 3  # how many plots in each row
rows = (num_cols + cols_per_row - 1) // cols_per_row  # calculate number of rows needed
fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, rows * 4))  # make subplots
axes = axes.flatten()  # flatten in case it’s a 2D array

for i, column in enumerate(numeric_columns):  # go through each numeric column
    sns.histplot(data_cleaned[column], kde=True, ax=axes[i], color='skyblue')  # draw histogram with line (KDE)
    axes[i].set_title(f"Histogram of {column}")  # title for each histogram

for j in range(i + 1, len(axes)):  # hide any extra empty plots
    axes[j].axis('off')  # turn off axis for empty plots

plt.tight_layout()  # adjust spacing
plt.show()  # show histograms

print("\nNormalization:")  # normalize values to bring them between 0 and 1
scaler_minmax = MinMaxScaler()  # create MinMaxScaler
data_cleaned[numeric_columns] = scaler_minmax.fit_transform(data_cleaned[numeric_columns])  # apply scaling to numeric columns

print("\nData After Normalization:")  # print data after normalization
print(data_cleaned.head())  # show top rows

categorical_cols = data_cleaned.select_dtypes(include='object').columns  # find non-numeric columns (like text)
label_encoders = {}  # to store encoders
for col in categorical_cols:  # go through each text column
    le = LabelEncoder()  # create encoder
    data_cleaned[col] = le.fit_transform(data_cleaned[col])  # convert text to numbers
    label_encoders[col] = le  # save encoder

data_cleaned['feature_sum'] = data_cleaned[numeric_columns].sum(axis=1)  # create new feature that sums all features
data_cleaned['feature_mean'] = data_cleaned[numeric_columns].mean(axis=1)  # another feature with mean of all features

if data_cleaned.shape[1] > 20 and 'target' in data_cleaned.columns:  # if too many features and 'target' exists
    pca = PCA(n_components=0.95)  # reduce features to keep 95% of info
    pca_result = pca.fit_transform(data_cleaned.drop('target', axis=1))  # apply PCA (except target column)

X = data_cleaned.drop('target', axis=1)  # features for training
y = data_cleaned['target']  # target column (what we want to predict)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split into train/test data

models = {  # dictionary of models to train
    "Logistic Regression": LogisticRegression(max_iter=300),  # basic model
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)  # stronger model with multiple trees
}

metrics = []  # store performance of models

for name, model in models.items():  # train and test each model
    model.fit(X_train, y_train)  # train the model
    y_pred = model.predict(X_test)  # make predictions
    acc = accuracy_score(y_test, y_pred)  # check accuracy
    prec = precision_score(y_test, y_pred, average='weighted')  # check precision
    rec = recall_score(y_test, y_pred, average='weighted')  # check recall
    f1 = f1_score(y_test, y_pred, average='weighted')  # check F1 score
    metrics.append([name, acc, prec, rec, f1])  # save model scores

    print(f"\n--- {name} ---")  # model name
    print("Accuracy:", acc)  # show accuracy
    print("Precision:", prec)  # show precision
    print("Recall:", rec)  # show recall
    print("F1 Score:", f1)  # show F1 score
    print(classification_report(y_test, y_pred))  # detailed performance report

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='cividis')  # draw confusion matrix
    plt.title(f"Confusion Matrix: {name}")  # set title
    plt.xlabel("Predicted")  # x-axis label
    plt.ylabel("Actual")  # y-axis label
    plt.show()  # show plot

print("\n--- Cross-Validation Scores (5-fold) ---")  # now test models using cross-validation
for name, model in models.items():  # for each model
    scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
    print(f"{name} CV Accuracy Scores: {scores}")  # print all scores
    print(f"Mean CV Accuracy: {scores.mean():.4f}\n")  # print average score

metrics_df = pd.DataFrame(metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])  # make dataframe of results
metrics_df.set_index('Model', inplace=True)  # set model names as index
metrics_df.plot(kind='bar', figsize=(10, 6))  # draw bar chart for comparison

plt.title('Model Performance Comparison')  # chart title
plt.ylabel('Score')  # y-axis label
plt.ylim(0, 1)  # score range
plt.xticks(rotation=0)  # keep model names straight
plt.legend(loc='upper right')  # legend position
plt.tight_layout()  # adjust layout
plt.show()  # show chart

rf = models["Random Forest"]  # get Random Forest model
importances = rf.feature_importances_  # get feature importances
indices = np.argsort(importances)[::-1]  # sort importances in descending order
plt.figure(figsize=(12, 6))  # set figure size

print(f"\n\n")  # print blank line
from matplotlib.cm import get_cmap  # import color map
cmap = get_cmap('Spectral')  # use 'Spectral' color map
colors = [cmap(i / len(importances)) for i in range(len(importances))]  # assign colors

plt.title("Feature Importances")  # chart title
plt.bar(range(X.shape[1]), importances[indices], color=[colors[i] for i in indices])  # draw bars with colors
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)  # set x labels
plt.tight_layout()  # fix layout
plt.show()  # show chart