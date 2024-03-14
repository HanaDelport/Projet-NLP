# Projet-NLP
Github pour le projet de NLP | Groupe : Saona Moussard - Hana Delport

## Introduction

Notre objectif est d'entrainer un modèle permettant de distinguer le niveau de toxicité de commentaires postées sur internet.
Pour ce faire, nous utilisons le dataset d'entraînement "train" fourni.

## Importations

Dans un premier temps, nous importons les librairies nécessaires à la réalisation du modèle ainsi que le dataset que nous avons téléchargé sur Google Drive au préalable.

Les libraires importées sont:
- numpy pour la lecture des fichiers .csv
- matplotlib pour la visualisation des statistiques
- keras pour la partie liée au Natural Language Processing ainsi que les layers nécessaires pour le modèle

## Étude du jeu de données

Avant de commencer à travailler sur le dataset, il est utile de regarder en quoi se compose ce dernier.
C'est pourquoi nous a semblé pertinent de construire un graphique afin de visualiser le pourcentage du nombre d'instances pour chaque classe de toxicité.

Initialement, le dataset est composé de 8 colonnes:
- un "id" unique pour chaque commentaire
- puis le "comment_text"
- et enfin les niveaux de toxicité suivants: "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate" de valeur 0 ou 1

Nous ajoutons une colonne "clean_comments" pour les commentaires n'ayant aucun 1 dans les colonnes de toxicité.

La création de la liste *list_classes* permet d'avoir la liste des classes séparément du train dataset tandis que la liste *list_sentences_train* permet d'avoir seulement les commentaires.
Afin de compter le nombre d'instances de chaque classe de toxicité sur l'entièreté du dataset, nous utilisons la fonction *sum()* sur le dataset *data[list_classes]* que nous divisons ensuite par la longueur du dataset.

Nous obtenons ainsi le graphique suivant:
[insérer image]

Nous remarquons ainsi qu'à peine 2% des commentaires du dataset sont considérés comme toxique.

## Préparation des données

Nous devons maintenant préparer les données afin qu'elles puissent être évaluées par le modèle que nous créerons ultérieurement.

Après un choix arbitraire, nous avons décider de prendre un vocabulaire maximum à 20 000 mots.

Nous utilisons *list_sentences_train* (appelée X) en input et les valeurs des colonnes spécifiant les toxicité *data[data.columns[2:8]].values* (appelée y) en output.
Nous ne voulons pas ici la colonne "clean_comments" par soucis de complexité.

Nous utilisons la fonction *TextVectorization* qui permet de:
- standardiser chaque exemple (mettre le texte en minuscule et enlever la ponctuation)
- séparer chaque exemple en plusieurs mots
- recombiner ces mots en tokens
- indexer les tokens, c'est à dire associer une valeur unique à chaque token
- transformer chaque exemple avec cet index afin de les transformer en vecteur d'entiers ou de flottants

Nous avons décider d'utiliser un vecteur d'entiers d'une longueur maximum à 1 800 mots pour plus de facilité.

La fonction *adapt* permet d'obtenir tout le vocabulaire compris dans la liste X.
Nous plaçons la liste tokenized dans la variable *tokenized_text*.

### Création d'un dataset

En utilisant la méthode MCSHBAP (map, cache, shuffle, batch et prefetch) avec tensorflow, nous avons créer une pipeline pour le dataset avant de faciliter l'entraînement de notre modèle.

Nous séparons ensuite notre liste de tokens et notre liste de labels recherchés.

Ensuite, nous séparons les batches en batches de validation (ici, 70% des groupes), de validation (20%) et de testing (10%).
La fonction *skip* permet de ne pas utiliser certaines partitions.

## Entraînement du modèle

Nous créons un modèle séquentiel avec les layers suivants:
- 1 embedding layer
- 1 bidirectional en utilisant LSTM layer
- 3 dense layers avec 'relu' en activation
- *Ici, le layer flatten n'est pas utile*
- 1 dense layer avec 'sigmoid' en activation pour les 6 labels

'BinaryCrossentropy' est ici utilisable car les 6 classes sont considérées comme différentes et non liées.

Avec mon ordinateur (Saona), chaque epoch dure plus de 2h30 pour se compléter avec tout le dataset.
J'en ai fait 3...

Un historique des epochs permet de les comparer entre eux.

## Itération de la modélisation

Nous pouvons maintenant tester notre modèle.

Les inputs test doivent être vectoriser et mis en batch pour que cela fonctionne.

L'utilisation des fonctions *Precision()*, *Recall()* et *CategoricalAccuracy()' permettent d'obtenir les statistiques du modèle.
Afin d'obtenir les résultats, nous créons une boucle afin qu'après chaque batch, les valeurs associées aux fonctions ci-dessus soient modifiées en conséquence après une comparaison entre les résultats réels et prédictifs. 

## Résultats
