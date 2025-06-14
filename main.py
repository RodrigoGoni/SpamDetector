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

word_freq_cols = dataset.columns[0:48]  # Columnas para palabras
char_freq_cols = dataset.columns[48:54]  # Columnas para caracteres
spam_df = dataset[dataset["spam"] == 1].drop(columns="spam")
non_spam_df = dataset[dataset["spam"] == 0].drop(columns="spam")
# Calcular la media de frecuencia para palabras en SPAM y NO SPAM
mean_word_freq_spam = spam_df[word_freq_cols].mean(
).sort_values(ascending=False)
mean_word_freq_non_spam = non_spam_df[word_freq_cols].mean(
).sort_values(ascending=False)

# Calcular la media de frecuencia para símbolos en SPAM y NO SPAM
mean_char_freq_spam = spam_df[char_freq_cols].mean(
).sort_values(ascending=False)
mean_char_freq_non_spam = non_spam_df[char_freq_cols].mean(
).sort_values(ascending=False)

print("--- Análisis de Frecuencia de Palabras y Símbolos ---")

print("\n✅ Top 10 palabras más frecuentes en correos SPAM:")
print(mean_word_freq_spam.head(10))

print("\n✅ Top 10 palabras más frecuentes en correos NO SPAM:")
print(mean_word_freq_non_spam.head(10))

print("\n✅ Top 10 símbolos más frecuentes en correos SPAM:")
print(mean_char_freq_spam.head(6))  # Hay 6 columnas de símbolos, no 10

print("\n✅ Top 10 símbolos más frecuentes en correos NO SPAM:")
print(mean_char_freq_non_spam.head(6))  # Hay 6 columnas de símbolos, no 10

# Determinar palabras o símbolos en común y llamativos
# Puedes hacer esto comparando las listas de los top N elementos.
# Por ejemplo, para palabras en común:
top_spam_words = set(mean_word_freq_spam.head(10).index)
top_non_spam_words = set(mean_word_freq_non_spam.head(10).index)
common_words = top_spam_words.intersection(top_non_spam_words)

print(
    f"\n¿Hay palabras en común en el top 10 de SPAM y NO SPAM? {'Sí' if common_words else 'No'}")
if common_words:
    print(f"Palabras en común: {list(common_words)}")

# Para símbolos en común:
top_spam_chars = set(mean_char_freq_spam.head(6).index)  # Ajustado a 6
top_non_spam_chars = set(mean_char_freq_non_spam.head(6).index)  # Ajustado a 6
common_chars = top_spam_chars.intersection(top_non_spam_chars)

print(
    f"\n¿Hay símbolos en común en el top 6 de SPAM y NO SPAM? {'Sí' if common_chars else 'No'}")
if common_chars:
    print(f"Símbolos en común: {list(common_chars)}")

print("\nConsideraciones sobre palabras/símbolos llamativos:")
print("- Las palabras o símbolos con una frecuencia promedio significativamente más alta en una categoría que en otra suelen ser más discriminatorias.")
print("- Por ejemplo, si 'free' aparece mucho más en SPAM que en NO SPAM, es llamativo.")
print("- Símbolos como '!', '$' o '#' pueden ser más frecuentes en SPAM.")
# --- End of TF-IDF Analysis Code ---
