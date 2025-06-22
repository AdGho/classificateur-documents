# Classificateur de Documents

Ce projet est un outil simple de classification automatique de texte en trois catégories :
- RH
- Technique
- Commercial

Il utilise un modèle de machine learning (`LogisticRegression`) entraîné sur un jeu de données.  
Une interface Web est disponible via **Gradio** pour tester le modèle facilement.

---

## Lancer le projet

### 1. Cloner le repo

```bash
git clone https://github.com/AdGho/classificateur-documents.git
cd classificateur-documents
```

### 2. Créer et activer l’environnement Python

```bash
python -m venv env
source env/Scripts/activate  
# Avec Git Bash sous Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l’application Gradio

```bash
python src/app.py
```
Un lien sera retourné par l'editeur de code, le copier et le coller dans un navigateur.


---

## Exemple de texte à tester

- "Nous recrutons un développeur back-end"
- "Le serveur de production a planté ce matin"
- "Les offres commerciales sont à mettre à jour"

---

## Outils utilisés

- Python
- pandas
- scikit-learn
- Gradio

---

## Choix techniques

- TF-IDF pour la vectorisation des textes : 
Cela permet de transformer chaque phrase en vecteur numérique en tenant compte à la fois de la fréquence des mots dans le texte et de leur rareté dans le corpus.
C’est une solution adaptée à un petit jeu de données comme dans notre cas

- Modèle de classification LogisticRegression :
Le classificateur repose sur une régression logistique, un modèle efficace pour tâches de classification linéaire.
L'entrainement est rapide donc est idéal pour des test locaux.

- Interface Gradio (Bpnus) :
Gradio permet de créer une interface Web similaire a Streamlit.
Elle facilite l'interaction avec un public au profil non technique.


---

## Ce que j’ajouterais avec plus de temps

- Enrichissement automatique du dataset avec des données externes
- Export des prédictions en CSV
- Interface avancée avec filtres ou upload de fichier texte
- Permettre à l’utilisateur de téléverser un .txt ou un .csv contenant plusieurs lignes à classer.
- Ajout d’un système de feedback utilisateur
- Ajouter une page pour afficher la matrice de confusion, précision, rappel par classe.
- Offrir une alternative à Gradio pour intégrer le modèle dans une app existante via ou Flask (choix de la tech a confirmer)