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

"""# Importation du jeu de données"""

app = Flask(__name__)
df = pd.read_excel("database/Full Dataset.xlsx")
df.head()

"""# Traitement du jeu de données"""

print("Nombre de lignes :", df.shape[0])
print("Nombre de colonnes :", df.shape[1])

"""Regardons les informations générales, telles que les noms des colonnes, leurs types et le nombre de données non-nulles :"""

df.info()

"""**Nous avons une colonne qui indique le type de vêtement dont il s'agit, des données sur la quantité de chaque matériel pour ce vêtement, des labels selon l'ACP de ce produit tels que production, toxicité (chemicals), réutilisabilité et recyclabilité, le local de fabrication et d'utilisation, la distance entre ces deux derniers et finalement la classification environnementale (EI - Environmental Impact)**.

`Manufacturing_location` est la seule colonne avec des données manquantes. Regardons pour combien d'observations :
"""

df.isna().sum().sum()

"""Puisque les données manquantes représentent une grande partie de notre ensemble de données, et sachant que nous avons la colonne `Transporation_distance` qui est finalement plus significative que les lieux de fabrication et d'utilisation du produit, **nous allons supprimer les colonnes `Manufacturing_location` et `Use_location`**. Puisque la **colonne `ID`** correspond à l'index, nous allons la supprimer également :"""

df.drop(['ID', 'Manufacturing_location', 'Use_location'], axis=1, inplace=True)

"""Regardons maintenant quelques inforamtions statistiques de chaque colonne numérique (moyenne, écart-type, minimum, maximum, intercartilles, count) :"""

df.describe()


#Fonction pour afficher un graphique en barres horizontales :

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
    return plt
# Plotting
plt1 = bar_h(df, 'Type')
plt1.savefig('static/bar_type.png')
plt2 = bar_h(df, 'Washing_instruction')
plt2.savefig('static/bar_washing.png')
plt3 = bar_h(df, 'Drying_instruction')
plt3.savefig('static/bar_drying.png')

"""Ces variables qualitatives n'ont pas d'impact significatif sur l'impact environnemental du vêtement : le type de lavage est une conséquence plutôt de la composition générale des matériaux de fabrication d'une pièce, et le type de vêtement n'est qu'une classification qui n'a pas de corrélation directe avec la variable qu'on cherche à expliquer.
Ainsi, **nous n'allons pas utiliser les variables qualitatives pour notre modèle de Machine Learning**. Cela nous évite également de passer par un processus d'encodage via le module OneHot Encoder de Scikit-Learn, pour les variables non ordinales, et Label Encoding pour les variables ordinales.

Regardons le colonnes `Recycled_content`, `Recylability_label`, `Reused_content`, `Reusability_label` et `Material_label` :
"""

df['Recycled_content'].value_counts()

df['Recylability_label'].value_counts()

df['Reused_content'].value_counts()

df['Reusability_label'].value_counts()

df['Material_label'].value_counts()

"""Nous pouvons constater que la plupart des vêtements ne contiennent aucun matériau recyclé lors de leur fabrication et qu'ils ne peuvent pas non plus être réutilisés.
Regardons maintenat notre **variable à expliquer**, l'impact environnemental (`EI` - Environmental Impact) :
"""

def distrubution():

    plt.figure(figsize=(6,6))
    df['EI'].value_counts().plot(kind='pie', autopct='%.2f%%')
    plt.title('Distribution du label EI (Environmental Impact)', fontweight='bold', fontsize=15)
    #plt.show()
    print('test')
    plt.savefig('static/pie.png')

"""Il s'agit d'une étiquette allant de 1 à 5 selon l'impact environnemental du produit. La plupart des vêtements de notre base de données ont une classification entre 3 et 5, c'est-à-dire, sont dans une catégorie de grand impact environnemental.

Examinons la matrice de corrélation, qui nous donne les R-carrés ou les coefficients de détermination entre les variables, pour voir si nous pouvons en tirer des conclusions significatives :
"""
def cor_matrix():
    #matrice de correlation
    corr = df.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, annot=True, cmap='rocket_r', linewidths=1, linecolor='white')
    plt.title("Matrice des corrélations", fontsize=18, fontweight='bold')
    plt.savefig('static/cor_matrix.png')
    #plt.show()

"""Nous pouvons voir que **l'étiquette de l'impact environnemental (`EI`) augmente en fonction de l'impact négatif : plus elle est grande, moins l'entreprise est éco-responsable**.  <br>
Nous pouvons voir que **le pourcentage de coton biologique (`cotton_organique`) est négativement corrélé à l'`IE`, ainsi que toutes les variables `label`**. <br>Comme cet ensemble de données provient d'une étude antérieure dans laquelle l'une des variables d'une paire de variables explicatives fortement corrélées entre elles a été supprimée, nous n'avons pas besoin de le faire pour que notre modèle ne soit pas biaisé.

Nous avons une quantité importante de variables. Avant de créer un modéle d'apprentissage supervisé, il faut que nous regardions s'il est possible de diminuer le nombre de variables initiales pour obtenir un un espace de dimension réduite en perdant le moins possible d'informations. Pour ça, nous allons appliquer une ACP.

Le premier pas, tout d'abord, est de **diviser notre jeu de données en variables explicatives, variable à expliquer, en d'entraînement et jeu de test**.

# Data Split

D'un coté X la matrice des données (**variables explicatives**) :
"""

#créer une copie du df en ne conservant que les colonnes numériques :
df = df.select_dtypes(include=['float64', 'int64'])

#matrice X
X = df.iloc[:, :-1].values

#nom des variables à expliquer
features = df.iloc[:, :-1].columns
features

"""Nous enregistrons les différentes catégories de la **variable à expliquer** dans une nouvelle variable, y :"""

y = df['EI']
y

"""On découpe nos données en **jeu d'entraînement et jeu de test (20% du total)** :"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""On va **centrer-réduire** nos données :"""

scaler = preprocessing.StandardScaler()
std_scale = scaler.fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

"""On confirme que la moyenne est à 0 et l'écart-type à 1, c'est-à-dire, que nos données sont bien centrées-réduites :"""

pd.DataFrame(X_train_std).describe().round(2).iloc[1:3:, : ]

"""# ACP : Analyse par composantes principales

L'**ACP** ou **PCA** (pour Principal Component Analysis) a deux objectifs principaux. Elle permet d'étudier :

* La **variabilité entre les individus**, c'est-à-dire quelles sont les différences et les ressemblances entre individus ;

* Les **liaisons entre les variables** : y a-t-il des groupes de variables très corrélées entre elles, qui peuvent être regroupées en de nouvelles variables synthétiques ?

Pour savoir **combien de composantes principales étudier**, il faut créer un diagramme d’éboulis des valeurs propres.
En ACP, quand on projette les données sur les axes principaux d’inertie, ceux-ci sont ordonnés selon l’inertie du nuage de points projeté : de la plus grande à la plus petite. Quand on additionne les inerties associées à tous les axes, on obtient l’inertie totale du nuage des individus. L’éboulis des valeurs propres est donc ce diagramme qui décrit le pourcentage d’inertie totale associé à chaque axe. On peut également afficher la somme cumulée des inerties, une courbe qui part de l’origine et qui arrive à 100 % après avoir parcouru tous les axes.
Nous allons travailler ici sur les 10 premières composantes :
"""

#définir le nombre de composantes principales
n_components = 10

# On instancie notre ACP :
pca = PCA(n_components=n_components)

# On l'entraîne et transforme sur les données :
pca.fit(X_train_std)
df_pca = pca.transform(X_train_std)

"""Intéressons nous maintenant à la variance captée par chaque nouvelle composante :"""

pca.explained_variance_ratio_

"""Ici, on voit que la 1ère composante capte presque 13% de la variance de nos données initiales, la 2ème 6.71%, etc etc. Nous pouvons voir cela dans notre **éboulis des valeurs propres** :"""

#variances par composantes principales
scree = (pca.explained_variance_ratio_*100).round(2)

#variances cumulées
scree_cum = scree.cumsum().round()

#liste avec composantes
x_list = range(1, n_components+1)

def val_p():
    #éboulis des valeurs propres
    plt.bar(x_list, scree)
    plt.plot(x_list, scree_cum,c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    #plt.show(block=False)

"""On a en bleu la variance de chaque nouvelle composante, et en rouge la variance cumulée. On voit ici qu'on a que'environ 55% de la variance expliquée par les 10 premières composantes. Cela nous indique que toutes nos variables doivent être comprises dans notre modèle. Nous pouvons voir la contribution de chaque variable à chaque axe principal d'inertie par la table ci-dessus :"""

pcs = pd.DataFrame(pca.components_)
pcs.columns = features
pcs.index = [f"F{i}" for i in x_list]
pcs.round(2)
pcs.T

"""Répresentons cela par un heatmap :"""

fig, ax = plt.subplots(figsize=(20, 6))
sns.heatmap(pcs.T, vmin=-1, vmax=1, annot=True, cmap="coolwarm", fmt="0.2f")
#plt.show()

"""Et finalement par un **cercle de corrélations**, qui nous permet d'étudier la liaison entre les variables qui définissent l'impact environnemental d'un vêtement selon les plans principaux d'inertie. Créons notre cercle des corrélations pour F1 et F2 :"""

def correlation_graph(pca,
                      x_y,
                      features) :
    """Affiche le graphe des correlations

    Positional arguments :
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

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
    #plt.show(block=False)

# définition des axes x et y (2 premières composantes) :
x_y = (0,1)
x_y

# graphique
correlation_graph(pca, x_y, features)

"""Et pour F2 et F3 :"""

correlation_graph(pca, (1,2), features)

"""**On constate qu'il est effectivement difficile de retirer des conclusions sur un possible résumé de nos données. Toutes les variables étant importantes, nous n'allons pas obtenir un espace de dimension réduit.**

**Notre but étant d'effectuer une tâche de classification, nous pouvons essayer d'utiliser un algorithme différent qui pourrait être mieux adapté à nos données, comme les forêts aléatoires ou les réseaux de neurones.**

**Nous allons commencer par un algorithme d'arbre de décisions, pour après appliquer Random FOREST et évaluer la performance de chaque algorithme.**

# Arbre de décisions

**Un arbre de décisions est un algorithme d'apprentissage supervisé non paramétrique, qui est utilisé à la fois pour les tâches de classification et régression**. Il a une **structure hiérarchique**, une structure arborescente, qui se compose d'un noeud racine, de branches, de nœuds interne et de noeuds feuille.

Pour la classification,  à chacune de ces itérations, l'algorithme d'entraînement va rajouter la décision qu'il lui semble le mieux de rajouter. Pour ce faire, il va  tester et évaluer la qualité de toutes les nouvelles décisions qu'il est possible d'ajouter à l'arbre en calculant le score Gini. Le score Gini est un score qui a été spécialement inventé afin de réaliser la sélection des nouvelles branches dans un arbre de décision.

Le score "Gini", est compris entre zéro et 1.  Il s'agit d'une valeur numérique indiquant la probabilité que l' arbre se trompe lors de la prise d'une décision (par exemple qu'il choisit la classe "A" alors que la vraie classe c'est "B") - si le jeu de données était libellé qie sur la base de la distribution de ses classes. Une branche sera rajoutée à l'arbre si parmi toutes les branches qu'il est possible de créer cette dernière présente le score Gini maximal.

Le process complet de construction d'un arbre de décision, est, donc :

   1- À l'initialisation, l'arbre est totalement vide.

   2- Le score de toutes les décisions qu'il est possible de prendre  est calculé.

   3- La décision qui présente le score Gini maximal est choisie comme racine

   4-Tant qu'il est possible de faire un split et que le critère d'arrêt n'est pas respecté :

            5- Pour chaque décision qu'il est possible d'ajouter à l'arbre; Faire 6.

                             6- Calcul du score Gini de la décision courante

            7-Sélection de la décision admettant le score max et ajout de celle-ci à l'arbre


Nous allons faire appel à la bibliothèque scikit-learn pour entraîner un modèle d'arbre de décisions sur notre jeu de données. Pour choisir les meilleurs paramètres à utiliser dans notre modèle, nous allons utiliser la validation croisée à travers GridSearchCV :
"""

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

"""**Nous avons obtenu un accuracy score de 0.85 : notre modèle est donc très performant**. Nous pouvons le visualizer sur l'image ci-dessus :"""
def arbre():
    # Visualiser l'arbre de decision
    plt.figure(figsize=(20,20))
    plot_tree(clf, fontsize=8)
    #plt.show()

"""Même si notre modèle est très performant, il y a un possible "problème" quand on utilise qu'un seul arbre de decision : notre modèle tend a overfitter très facilement, en restant trop collé aux données qu'on lui a fourni pour faire une classification. C'est pour cette raison qu'on passera a un deuxième modèle, qui va être au fait une perfecction des arbres de decisions : une forêt aléatoire.

# Forêts aléatoires

**Un random forest est constitué d'un ensemble d'arbres de décision indépendants. Chaque arbre dispose d'une vision parcellaire du problème du fait d'un double tirage aléatoire : un tirage aléatoire avec remplacement sur les observations** (bootstrap).

L'algorithme d'entraînement du Random Forest construit chacun des arbres de décisions qui le composent, en les entraînant tous avec un sous-ensemble des données du problème.

Il choisit aléatoirement des données auxquelles une partie des arbres de décisions n'auront pas accès tandis qu'une autre y aura accès afin de les rendre totalement aveugles à ces dernières et de s'assurer que tous les arbres de décision aient bien une expérience différente du problème.

Une fois l'entraînement de tout les arbres de décision terminés, le Random Forest prend ses décisions relativement au problème de classification ou de régression à résoudre, en faisant voter tous les arbres de décisions qui le compose. La décision de la majorité l'emporte alors.

Entraînons notre modèle de Random Forest à l'aide de Sckiti-Learn, encore une fois. Commençons par choisir au hasard les paramètres n_estimators et max_depth, et en définissant la racine carrée du nombre total de caractéristiques comme le nombre de caractéristiques que l'algorithme de la forêt aléatoire peut rechercher à partir de chaque point de division d'un arbre :
"""

# Définir le classificateur de la forêt aléatoire
rfc = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=10, random_state=42)

# Ajuster le classificateur aux données d'entraînement
rfc.fit(X_train_std, y_train)

# Prédire sur la base des données de test
y_pred = rfc.predict(X_test_std)

# Accuracy score
print('Accuracy:', round(accuracy_score(y_test, y_pred),2))

"""**Nous avons déjà un accuracy score meilleur que celui d'un seul arbre de decision : 0.87.**

Testons une autre configuration de n_estimators et max_depth :
"""

# Définir le classificateur de la forêt aléatoire
rfc = RandomForestClassifier(n_estimators=10, max_features='sqrt', max_depth=100, random_state=42)

# Ajuster le classificateur aux données d'entraînement
rfc.fit(X_train_std, y_train)

# Prédire sur la base des données de test
y_pred = rfc.predict(X_test_std)

#Accuracy score
print('Accuracy:', round(accuracy_score(y_test, y_pred),2))

"""**Nous avons un accuracy score encore mieux : 0.89** ! Notre modèle est donc plus performant avec un n_estimators=10 et max_depth=100, c'est-à-dire, des arbres très profonds.

On va maintenant utiliser la variable "importance des caractéristiques" pour afficher les scores d'importance des variables explicatives et les visualizer dans un graphique à barres horizontales :
"""
# visualiser les scores des caractéristiques
feature_scores = pd.Series(rfc.feature_importances_, index=features).sort_values(ascending=False)


def score():
    # Créer d'un diagramme à barres
    plt.figure(figsize=(12,10))
    sns.barplot(x=feature_scores, y=feature_scores.index)

    # Ajouter des étiquettes au graphique
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')

    # Ajouter un titre au graphique
    plt.title("Variables les plus importantes de la Forêt Aléatoire", fontsize=15, fontweight='bold')

    # Visualize the graph
    #plt.show()

"""**Nous allons maintenant voir si notre modèle s'amèlieure encore plus si on garde que les variables qui nous sont vraiment intéressantes pour le modèle**, on va dire les 28 variables dont le feature_scores est visible sur le graphique ci-dessus :"""

# Déclarer le vecteur de caractéristiques et la variable cible
X_new = df[pd.DataFrame(feature_scores).reset_index().head(28)['index'].tolist()]
y_new = df['EI']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size = 0.2, random_state = 42)

# Centrer-reduire le nouvel ensemble de variables
std_scale_new = scaler.fit(X_train_new)
X_train_std_new = std_scale_new.transform(X_train_new)
X_test_std_new = std_scale_new.transform(X_test_new)

# Définir le classificateur de la forêt aléatoire
rfc_new = RandomForestClassifier(n_estimators=10, max_features='sqrt', max_depth=100, random_state=42)

# Ajuster le classificateur aux données d'entraînement
rfc_new.fit(X_train_std_new, y_train_new)

# Prédire sur la base des données de test
y_pred_new = rfc_new.predict(X_test_std_new)

#Accuracy score
print('Accuracy:', round(accuracy_score(y_test_new, y_pred_new),2))

"""**L'accuracy score a diminué avec le feature selection effectué. Cela nous confirme, comme nous avios vu lors de l'essai de l'application de l'ACP, que toutes nos variables sont significatives pour notre modèle. Notre modèle choisi est donc celui dont l'accuracy score est égal à 0.89**


Regardons, maintenat, la **matrice de confusion** pour avoir une idée de la qualité de prédiction pour chaque catégorie :
"""

# Definir les labels
labels = [1, 2, 3, 4, 5]

# Créer une matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=labels)

def  matrice_confusion():
    # Visualiser matrice de confusion
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot()
    plt.title("Matrice de confusion du modèle de Forêt Aléatoire", fontweight='bold')
    #plt.show()

"""# Conclusion

Nous constatons que tous les vêtements ayant un `EI` de 1 ont été classés dans la catégorie 1. Il en va de même pour la catégorie 2.
Ainsi, **notre modèle est capable de prédire 100 % des vêtements ayant le plus faible impact environnemental**.  <br>
La variance est un peu plus importante pour les vêtements situés au milieu, puisque 6 vêtements ont été classés dans la catégorie 2 et 6 dans la catégorie 4 alors qu'ils appartenaient à la catégorie 3, ce qui représente 18 % des vêtements de la catégorie 3 qui ont été mal classés.
8 vêtements ont été classés dans la catégorie 5 alors qu'ils appartenaient à la catégorie 4, mais cela ne représente que 8,6 % des vêtements de la catégorie 4, ce qui reste une très bonne prédiction.
Enfin, 4 vêtements de la catégorie 5 ont été classés dans la catégorie 4, ce qui représente 12 % des vêtements de cette catégorie.
D'une manière générale, **notre modèle reste très performant avec des classifications pertinentes et un très bon accuracy score**.
"""



distrubution()
plt.savefig('static/pie.png')

cor_matrix()
plt.savefig('static/cor_matrix.png')

val_p()
plt.savefig('static/scree_plot.png')

arbre()
plt.savefig('static/arbre.png')

score()
plt.savefig('static/score.png')

matrice_confusion()
plt.savefig('static/confusion_matrix.png')


correlation_graph(pca, [0, 1], features)
plt.savefig('static/correlation_plot.png')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graphs')
def generate_graphs():

    return render_template('graphs.html')


if __name__ == '__main__':
    app.run(threaded=True, debug=True)
