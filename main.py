import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

dataset = pd.read_csv("dataset/spambase.csv")
print(dataset.head(10))
column_sum = dataset.groupby(by="spam", as_index=False).sum()
X = dataset.drop(columns="spam")
y = dataset["spam"]
plot = False
print(f"X es de tipo: {type(X)}")
print(f"El shape es de X: {X.shape}")
print(f"El label es de tipo: {type(y)}")
print(f"El shape es de Y: {y.shape}")
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("Observaciones de X_train:", X_train.shape[0])
print("Observaciones de y_train:", y_train.size)
print("Observaciones de X_test:", len(X_test))
print("Observaciones de y_test:", len(y_test))
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
if plot == True:
    sns.pairplot(data=dataset, diag_kind="hist",
                 hue="spam", palette="cool", corner=True)
    plt.savefig("images/pairplot.png")
# --- TF-IDF Analysis Code to Add ---
print("--- TF-IDF Analysis ---")

# Separate the dataset into negative (0) and positive (1) reviews
# and drop the 'REVIEW_VALUE' column from these subsets
dataset_neg = dataset[dataset["REVIEW_VALUE"]
                      == 0].drop(columns="REVIEW_VALUE")
dataset_pos = dataset[dataset["REVIEW_VALUE"]
                      == 1].drop(columns="REVIEW_VALUE")

# Calculate the mean TF-IDF for each word in negative reviews
mean_tfidf_neg = dataset_neg.mean().sort_values(ascending=False)
# Calculate the mean TF-IDF for each word in positive reviews
mean_tfidf_pos = dataset_pos.mean().sort_values(ascending=False)

# Print the top 5 words with the highest average TF-IDF in each class
print("\n🔴 Palabras más representativas de la clase NEGATIVA:")
print(mean_tfidf_neg.head(5))

print("\n🟢 Palabras más representativas de la clase POSITIVA:")
print(mean_tfidf_pos.head(5))

# Clean up temporary variables to free memory
del mean_tfidf_neg, mean_tfidf_pos, dataset_neg, dataset_pos
print("--- TF-IDF Analysis Complete ---")
# --- End of TF-IDF Analysis Code ---
