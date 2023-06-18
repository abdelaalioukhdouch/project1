#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
import threading

import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any GUI backend
import seaborn as sns



app = Flask(__name__)
df = pd.read_excel("database/Full Dataset.xlsx")
df.head()

print("Nombre de lignes :", df.shape[0])
print("Nombre de colonnes :", df.shape[1])
df.info()
df.isna().sum().sum()
df.drop(['ID', 'Manufacturing_location', 'Use_location'], axis=1, inplace=True)
df.describe()
#sns.displot(df,x = 'Manufacturing_location',aspect = 5/5)

df['Recycled_content'].value_counts()
df['Recylability_label'].value_counts()
df['Reused_content'].value_counts()
df['Reusability_label'].value_counts()
df['Material_label'].value_counts()





def bar_h(df,X) :
    # Compter le nombre de chaque type de vêtement
    counts = df[X].value_counts()
    # Créer un graphique en barres horizontales
    plt.barh(counts.index, counts.values)
    # Ajouter des étiquettes et un titre
    plt.xlabel('Nombre de vêtements')
    plt.ylabel(f'{X}')
    plt.title(f'Nombre de vêtements par {X}')
    # Afficher le graphique
    #plt.savefig('static/nbr_vet.png')
    return plt

# Plotting
plt1 = bar_h(df, 'Type')
plt2 = bar_h(df, 'Washing_instruction')
plt3 = bar_h(df, 'Drying_instruction')
plt1.savefig('static/bar_type.png')
plt2.savefig('static/bar_washing.png')
plt3.savefig('static/bar_drying.png')


def generate_plots():
    
    # Bar Graph
    plt.figure(figsize=(10, 6))
    bar_h(df, 'Type')
    plt.close()
    
    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Type', y='EI')
    plt.xlabel('Type')
    plt.ylabel('EI')
    plt.title('Distribution of EI by Type')
plt.savefig('static/boxplot_category.png')
    

def plot():
    # Exemple de visualisation en utilisant Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df['Type'].value_counts().index, y=df['Type'].value_counts().values)
    plt.xlabel('Type de vêtement')
    plt.ylabel('Nombre de vêtements')
    plt.title('Nombre de vêtements par type')
plt.savefig('static/bar_chart.png')

def fun():
    plt.figure(figsize=(6,6))
    df['EI'].value_counts().plot(kind='pie', autopct='%.2f%%')
    plt.title('Distribution du label EI (Environmental Impact)', fontweight='bold', fontsize=15)
    plt.savefig('static/Ei.png')

def corr():
   #matrice de correlation
    corr = df.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, annot=True, cmap='rocket_r', linewidths=1, linecolor='white')
    plt.title("Matrice des corrélations", fontsize=18, fontweight='bold')
    plt.show()
    plt.savefig('static/corr.png')

#définir le nombre de composantes principales
n_components = 10
#liste avec composantes
x_list = range(1, n_components+1)

#créer une copie du df en ne conservant que les colonnes numériques :
df = df.select_dtypes(include=['float64', 'int64'])

#matrice X
X = df.iloc[:, :-1].values

y = df['EI']
y

#nom des variables à expliquer
features = df.iloc[:, :-1].columns
features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = preprocessing.StandardScaler()
std_scale = scaler.fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)
pd.DataFrame(X_train_std).describe().round(2).iloc[1:3:, : ]



# On instancie notre ACP :
pca = PCA(n_components=n_components)

# On l'entraîne et transforme sur les données :
pca.fit(X_train_std)
df_pca = pca.transform(X_train_std)

pca.explained_variance_ratio_

pcs = pd.DataFrame(pca.components_)
pcs.columns = features
pcs.index = [f"F{i}" for i in x_list]
pcs.round(2)
pcs.T


# ACP : Analyse par composantes principales
def var():

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = preprocessing.StandardScaler()
    std_scale = scaler.fit(X_train)
    X_train_std = std_scale.transform(X_train)
    X_test_std = std_scale.transform(X_test)

    pd.DataFrame(X_train_std).describe().round(2).iloc[1:3:, : ]

    #définir le nombre de composantes principales
    n_components = 10

    # On instancie notre ACP :
    pca = PCA(n_components=n_components)

    # On l'entraîne et transforme sur les données :
    pca.fit(X_train_std)
    df_pca = pca.transform(X_train_std)


    pca.explained_variance_ratio_

    #variances par composantes principales
    scree = (pca.explained_variance_ratio_*100).round(2)

    #variances cumulées
    scree_cum = scree.cumsum().round()

    #liste avec composantes
    x_list = range(1, n_components+1)

    #éboulis des valeurs propres
    plt.bar(x_list, scree)
    plt.plot(x_list, scree_cum,c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.savefig('static/var.png')

def correlation_graph(pca, x_y, features) :
    """Affiche le graphe des correlations"""

    # Extrait x et y
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante :
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0,
                pca.components_[x, i],
                pca.components_[y, i],
                head_width=0.07,
                head_length=0.07,
                width=0.02, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                pca.components_[y, i] + 0.05,
                features[i])

    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # J'ai copié collé le code sans le lire
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.savefig('static/cor.png')

    # définition des axes x et y (2 premières composantes) :
    x_y = (0,1)
    x_y
# graphique
correlation_graph(pca, (1,2), features)


def arbre():
    # Tester des différents paramètres min_samples_leaf
    param_grid = {'min_samples_leaf': range(5, 100, 5)}
    # Instancier le classifieur d'arbre de decisions avec un crière de Gini et best splitter :
    dtc = DecisionTreeClassifier(criterion='gini', splitter='best')
    # Rechercher le meilleur paramètre min_samples_leaf et ajuster le modèle aux données d'apprentissage :
    grid_search = GridSearchCV(dtc, param_grid=param_grid, cv=10)
    grid_search.fit(X_train_std, y_train)
    # Enregistrer n_samples_leaf optimal
    n_samples = grid_search.best_params_['min_samples_leaf']

    print("Best min_samples_leaf:", n_samples)

    # Appliquer la classification par arbre de décision avec le meilleur hyperparamètre
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
    # Ajuster le modèle aux données d'apprentissage :
    clf.fit(X_train_std, y_train)
    # Faire des prédictions sur les données du test
    y_pred = clf.predict(X_test_std)
    # Accuracy score
    print('Accuracy:', round(accuracy_score(y_test, y_pred),2))
    # Visualiser l'arbre de decision
    plt.figure(figsize=(20,20))
    plot_tree(clf, fontsize=8)
    plt.savefig('static/arbre.png')


def stat():
    # visualiser les scores des caractéristiques
    feature_scores = pd.Series(rfc.feature_importances_, index=features).sort_values(ascending=False)

    # Créer d'un diagramme à barres
    plt.figure(figsize=(12,10))
    sns.barplot(x=feature_scores, y=feature_scores.index)

    # Ajouter des étiquettes au graphique
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')

    # Ajouter un titre au graphique
    plt.title("Variables les plus importantes de la Forêt Aléatoire", fontsize=15, fontweight='bold')

    # Visualize the graph
    plt.savefig('static/stat')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graphs')
def generate_graphs():
    # Start a background thread to generate the plots
    thread = threading.Thread(target=generate_plots)
    thread.start()

    return render_template('graphs.html')


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
