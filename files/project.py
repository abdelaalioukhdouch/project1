# -*- coding: utf-8 -*-

from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


app = Flask(__name__)

# Chargement des données et préparation

df = pd.read_excel("database/Full Dataset.xlsx")
print(df.columns)
df.drop(['ID', 'Manufacturing_location', 'Use_location'], axis=1, inplace=True)

# ... Le reste de votre code pour l'exploration des données, l'entraînement du modèle, etc.

# Routes Flask

@app.route('/')
def index():
    # Code pour générer les visualisations
    # Utilisez les fonctions de visualisation que vous avez définies dans votre code précédent

    # Exemple de visualisation en utilisant Matplotlib
    plt.figure(figsize=(8, 6))
    df['EI'].value_counts().plot(kind='pie', autopct='%.2f%%')
    plt.title('Distribution du label EI (Environmental Impact)', fontweight='bold')
    plt.savefig('static/pie_chart.png')

    # Exemple de visualisation en utilisant Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Type', y='Count', data=df['Type'].value_counts().reset_index().rename(columns={'index': 'Type', 'Type': 'Count'}))
    plt.xlabel('Type de vêtement')
    plt.ylabel('Nombre de vêtements')
    plt.title('Nombre de vêtements par type')
    plt.savefig('static/bar_chart.png')

    # Exemple d'affichage du modèle d'arbre de décision
    plt.figure(figsize=(15, 10))
    # plot_tree(clf, feature_names=features, filled=True)
    plt.savefig('static/decision_tree.png')

    # Rendu du template HTML avec les visualisations
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
