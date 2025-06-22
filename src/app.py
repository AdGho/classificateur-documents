import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# charger les données
df = pd.read_csv("data/exemples.csv")
X = df["texte"]
y = df["categorie"]

vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_vect, y) # entrainement des dononées

# fonction de prédiction
def predire_categorie(texte):
    if not texte.strip():
        return " Texte vide. Essaie encore."
    vect = vectorizer.transform([texte])
    prediction = model.predict(vect)[0]
    return f" Catégorie prédite : {prediction}"

# interface visuelle
iface = gr.Interface(
    fn=predire_categorie,
    inputs=gr.Textbox(lines=2, placeholder="Ex : Nous cherchons un développeur Java...", label="  Entrez un texte à classer"),
    outputs=gr.Textbox(label=" Résultat", lines=1),
    title="Classificateur de Documents",
    description="Écris une phrase et le modèle prédit la catégorie (RH, Technique, commercial)."
)

iface.launch()
