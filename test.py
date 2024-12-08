import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from toy_script import load_data

# Spécifier le chemin du fichier
file_path = "preprocess_data_3freq.txt"

# Charger les données sans inclure les titres
X_raw = pd.read_csv(file_path, sep="\t")
x_unusable,y_raw, X_test_unusable =load_data(data_path='./')

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(X_train, y_train)

# Prédire sur les données de test
y_pred = clf.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)