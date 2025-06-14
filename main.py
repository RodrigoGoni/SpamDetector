from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
import pandas as pd
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
plot = True
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
col_sel = [
    'word_freq_free',
    'word_freq_george',
    'word_freq_your',
    'char_freq_!',
    'word_freq_our',
    'spam'
]
dataset_visualization = dataset[col_sel]
if plot == True:
    sns.pairplot(data=dataset_visualization, diag_kind="kde",
                 hue="spam", palette="cool", corner=True)

    plt.show()
# --- TF-IDF Analysis Code to Add ---
print("--- TF-IDF Analysis ---")

word_freq_cols = dataset.columns[0:48]
char_freq_cols = dataset.columns[48:54]
spam_df = dataset[dataset["spam"] == 1].drop(columns="spam")
non_spam_df = dataset[dataset["spam"] == 0].drop(columns="spam")
mean_word_freq_spam = spam_df[word_freq_cols].mean(
).sort_values(ascending=False)
mean_word_freq_non_spam = non_spam_df[word_freq_cols].mean(
).sort_values(ascending=False)

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
print(mean_char_freq_spam.head(10))

print("\n✅ Top 10 símbolos más frecuentes en correos NO SPAM:")
print(mean_char_freq_non_spam.head(10))
top_spam_words = set(mean_word_freq_spam.head(10).index)
top_non_spam_words = set(mean_word_freq_non_spam.head(10).index)
common_words = top_spam_words.intersection(top_non_spam_words)

print(
    f"\n¿Hay palabras en común en el top 10 de SPAM y NO SPAM? {'Sí' if common_words else 'No'}")
if common_words:
    print(f"Palabras en común: {list(common_words)}")

top_spam_chars = set(mean_char_freq_spam.head(10).index)
top_non_spam_chars = set(mean_char_freq_non_spam.head(10).index)
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
print("\n--- Naive Bayes Classifier Training ---")
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

print("Modelo de Naive Bayes Multinomial entrenado exitosamente.")
accuracy_train = naive_bayes_model.score(X_train, y_train)
accuracy_validation = naive_bayes_model.score(X_test, y_test)

print(f"La exactitud de entrenamiento es {accuracy_train}")
print(f"La exactitud de validación es {accuracy_validation}")

sc = StandardScaler()

X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
print("\n--- Logistic Regression Classifier Training ---")
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train_scaled, y_train)

print("Modelo de Regresión Logística entrenado exitosamente con datos normalizados.")

accuracy_train_lr = logistic_regression_model.score(X_train_scaled, y_train)
accuracy_validation_lr = logistic_regression_model.score(X_test_scaled, y_test)

print(
    f"La exactitud de entrenamiento (Regresión Logística) es {accuracy_train_lr}")
print(
    f"La exactitud de validación (Regresión Logística) es {accuracy_validation_lr}")

# --- Model Evaluation (Optional but Recommended) ---
y_pred_nb = naive_bayes_model.predict(X_test)
y_pred_lr = logistic_regression_model.predict(X_test_scaled)

print("\n--- Model Evaluation ---")
print(f"Accuracy (Naive Bayes): {accuracy_score(y_test, y_pred_nb)}")
print(f"Accuracy (Logistic Regression): {accuracy_score(y_test, y_pred_lr)}")

# Precision scores
print(f"Precision (Naive Bayes): {precision_score(y_test, y_pred_nb)}")
print(f"Precision (Logistic Regression): {precision_score(y_test, y_pred_lr)}")

# Recall scores
print(f"Recall (Naive Bayes): {recall_score(y_test, y_pred_nb)}")
print(f"Recall (Logistic Regression): {recall_score(y_test, y_pred_lr)}")


# --- Confusion Matrix Visualization ---
print("\n--- Confusion Matrices ---")


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()


# Plotting for Naive Bayes
print("Mostrando Matriz de Confusión para Naive Bayes...")
plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes")

# Plotting for Logistic Regression
print("Mostrando Matriz de Confusión para Regresión Logística...")
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")

print("\n--- Análisis Detallado de Errores y Métricas ---")

# Calcular y mostrar las matrices de confusión
cm_nb = confusion_matrix(y_test, y_pred_nb)
cm_lr = confusion_matrix(y_test, y_pred_lr)

print("\nMatriz de Confusión - Naive Bayes:")
print(cm_nb)

print("\nMatriz de Confusión - Regresión Logística:")
print(cm_lr)


# Análisis de Curva ROC y AUC

# --- Naive Bayes ---
y_prob_nb = naive_bayes_model.predict_proba(X_test)[:, 1]
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_test, y_prob_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)


# --- Logistic Regression ---
y_prob_lr = logistic_regression_model.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

print(f"\nAUC (Naive Bayes): {roc_auc_nb:.4f}")
print(f"AUC (Regresión Logística): {roc_auc_lr:.4f}")

# --- Graficar ambas curvas ROC ---
plt.figure(figsize=(10, 7))
plt.plot(fpr_nb, tpr_nb, color='blue', lw=2,
         label=f'ROC curve Naive Bayes (AUC = {roc_auc_nb:.2f})')
plt.plot(fpr_lr, tpr_lr, color='red', lw=2,
         label=f'ROC curve Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Comparación de Modelos')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
