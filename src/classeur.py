import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# on commence par charger le jeu de données qu'on a crée dans le dossier data
df = pd.read_csv("data/exemples.csv")

# bien séparer texte et catégorie pour avoir un bon rapport final
X = df["texte"]
y = df["categorie"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# transformer le texte en vecteur
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# entrainement du modele 
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# affichage des resulatats
y_pred = model.predict(X_test_tfidf)
print("Résultatss :\n")
print(classification_report(y_test, y_pred))