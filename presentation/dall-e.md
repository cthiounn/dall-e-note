---

slideOptions:
  transition: fade
  theme: white
  
---


# Présentation de DALL-E

---

1. Qu'est-ce que DALL-E ?
2. Modèle général des réseaux de neurones
3. Application au NLP
4. Application au computer vision

---

## Qu'est-ce que DALL-E ?

----

DALL-E est un modèle entrainé par OpenAI qui :
* prend en entrée un texte (et une image)
* génère en sortie des images correspondant aux entrées


----

DALL-E permet générer de nouvelles images selon une description et éventuellement d'une image initiale :

* en modifiant des caractéristiques d'une image existante (attribut d'un objet ; vue ou perspective)
* en composant une nouvelle image
* tout en comprenant le contexte et des connaissances ou référence du monde (géographique ; temporelle)
* "auto-apprendre de nouveaux concepts"

----


DALL-E est un modèle de transformation du langage (texte et image) :
* peut prendre en entrée 1280 tokens
* utilise GPT-3
* génère 512 images (CLIP filtre et donne les 32 meilleurs)


----

DALL-E est un modèle de deep learning mélant :
* NLP avec GPT-3
* Computer Vision en utilisant et en générant des images


---

## Modèle général des réseaux de neurones

----

Un neurone est :
* une entité qui prend en entrée des signaux et qui renvoie un signal en sortie
* combiné à d'autres, permet de modéliser n'importe quelle fonction (théorème d'approximation universelle)
* permet d'apprendre des comportements complexes par entraînement
* "presque assimilé" à une regression logistique binaire

----

### Le perceptron

* input : entiers
* fonction agrégation : CL avec poids
* fonction d'activation : sigmoïde, reLU, etc
* renvoie 0 ou 1

----

Combiné à plusieurs, peut modéliser :
* des portes logiques (ET; OU; NON)
* fournir une prédiction sur des données linéairement séparables
* "apprendre" un comportement par la correction des poids, pas à pas de chaque neurone

----

### Le perceptron à plusieurs couches (MLP)

* Plusieurs couches où les signaux traversent unidirectionnellement (feedforward)
* Un algorithme permet de minimiser l'erreur (loss fonction)
* en corrigeant les poids utilisés par les neurones en ordre inverse (backpropagation)
* par la méthode de la descente du gradient (propriétés liées à la différentiabilité de la fonction de coût)

---

## Application au NLP

----

Les réseaux de neurones peuvent trouver une utilisation dans le domaine de la linguistique.
Pour cela, il faut :
* pouvoir donner une quantité à un mot (token d'un vocabulaire)
* ainsi, être capable de se représenter tout un vocabulaire dans un espace de dimension n
* avec une notion de similarité/dissimilarité (~distance)

----

### Vectorisation de mots

* autrement appelée "Méthode de prolongement de mots" (Word embedding - word2vec)
* Buts : 
    * associer un mot avec un vecteur dans R^n
    * calculer si deux mots sont proches

----

* Pour cela, hypothèse de sémantique distributionnelle :
    * deux mots qui présentent des contextes distributionnelles proches sont sémantiquement proches
    * deux mots dont les vecteurs sont proches sont sémantiquement
    * e.g : "j'aime les animaux et ils m'apportent du bonheur", "Particulièrement j'adore les chiens et ils sont mignons", "les chiens sont des animaux"

----

```
!pip install gensim
from gensim.models import word2vec

phrases=["j'aime les animaux et ils m'apportent du bonheur" ,"Particulièrement j'adore les chiens et ils sont mignons", "les chiens sont des animaux"]

model = word2vec.Word2Vec([i.replace(',','').split() for i in phrases], vector_size=5, window=5,min_count=1, workers=1)

print("vecteur pour animaux",model.wv['animaux'])
print("vecteur pour j'aime",model.wv["j'aime"])
print("vecteur pour chiens",model.wv['chiens'])

print(model.wv.most_similar(positive='animaux',topn=5))
```

----

```
vecteur pour animaux [-0.06810813 -0.01891556  0.11536513 -0.15044144 -0.07872652]
vecteur pour j'aime [ 0.04938513 -0.01775371  0.11067609 -0.0548633   0.04522192]
vecteur pour chiens [ 0.1475969  -0.03065392 -0.09074011  0.13107255 -0.09719098]
[('des', 0.9323765635490417), ('sont', 0.7742035388946533), ('ils', 0.7669143080711365), ("j'aime", 0.46942147612571716), ("m'apportent", 0.4595804810523987)]
```

----

### Calcul d'une proximité

* Similarité cosinus
* distance euclidienne
* distance de Jaccard

----

### NLP dans le cas DALL-E

* voir approfondissement sur GPT2/3 et réseau de neurones récurrents

---

## Application au computer vision

----

* Buts : 
  * travailler dans le champ de l'image/vidéo
  * discriminer/classifier
  * générer/débruiter/transformer une image

----

### Réseau de neurones convolutifs (CNN)

* particularités :
  * raisonnement au pixel = grande volumétrie
  * hypothèse de dépendance spacio-temporelle
* même principe pour les mots, on souhaite quantifier une image dans R^n

----

### Principe général du CNN

Trois phases/espaces

* [Convolution] une partie réduction de dimension avec l'application de noyaux de convolution
* arrivée sur un petit espace de dimension R^n et travaux usuel [Couche intégralement connectée] (E.g. classification par des réseaux de neurones classiques)
* [Déconvolution] si besoin, une partie restitution de l'image, en appliquant les opérations inverses (transposées)

----

### Réduction de dimension par convolution

----

### Computer Vision dans DALL-E

* voir VAE


---

## Notions "avancées"

* Transformeurs
* Mécanisme d'attention