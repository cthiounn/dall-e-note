# NLP

*     https://jalammar.github.io/illustrated-gpt2/

* GPT-2 decoder-only transformer
* transformer-based language model trained on a massive dataset
* self-attention layer


## GPT-2 et ma modélisation du langage

### Qu'est-ce qu'un modèle de langue ?

* clavier de smartphone (e.g. SwiftKey) qui prédit le prochain mot
* GPT2 = prédiction du prochain mot ?
* entrainement/apprentissage par 40GB de données  (WebText), basé sur des textes sur Interner, crawlé par les chercheurs d'OpenAI
* Volumétrie de SwiftKey : 78 MB ; version mini ("small") de GPT2 : 500 MB ; version max ("extra-large") de GPT2 : 6.5 GB

* Démonstration (AllenAI GPT-2 Explorer):  https://demo.allennlp.org/next-token-lm
* Corpus en anglais

### Transformeurs pour la modélisation du langage (Transformers)

* encoder / decoder
* GPT-2 pile de decoder ; BERT pile d'encoder ; Transformer XL pile de recurrent decoder
* Différences de version de GPT-2 = nombre de decoder (48 pour l'extra, 12 pour le mini)

### GPT-2 Vs BERT

* decoder vs encoder
* GPT-2 prédit un mot (token) à la fois, comme les modèles traditionnels de langage
* autoregression (="récursivité en avant") : prédiction d'un mot => nouvelle phrase résultat => pris comme en entrée => nouveau mot prédit etc.  (force des réseaux de neurones récursifs [RNN])
* GPT-2, TransformerXL et XLNet sont nativement autoregressif <> BERT ne l'est pas

### Etude de l'évolution historique des transformeurs

* papier initial : https://arxiv.org/abs/1706.03762

* Initialement, un bloc d'encoder peut prendre jusqu'à 512 mots
* Il est composé d'un réseau de neurones "Feed Forward" et d'un bloc "Self-attention"
* un bloc de decoder est une variation du bloc d'encoder, avec l'ajout d'un bloc d'encoder/decoder de Self-attention et l'utilisation d'un bloc "Self-attention" à masque (l'algorithme ne regarde qu'une partie des inputs)
* (NB : le modèle BERT n'utilise pas de masque)

* modèle à bloc de decoder
* le premier modèle utilisait 6 blocs identiques de decoder (sans encoder/decoder de Self-attention), pouvant prendre en compte 4000 mots

### Dans les entrailles de la machine GPT-2

* GPT-2 peut traiter 1024 mots
* On peut lancer GPT-2 en mode (rambling), il génère ainsi des exemples de phrases avec un thème ou non (generating unconditional samples/ interactive conditional)
* A partir d'un "token input" (start token =\<s>), GPT-2 génère un mot le plus probable "The" qui passe dans tous les decoders, parmi son corpus de vocabulaire de 50k mots. Et ainsi de suite.
* S'il prend toujours le mot le plus probable, le modèle peut être coincé dans une boucle où la seule façon d'en sortir est de prendre le deuxième ou troisième mot probable
* les blocs de decoder se sont déjà spécialisées à l'issue de la première passe, pour que la deuxième passe n'interprète plus le "token input"

### De la fine dentelle

* https://fr.wikipedia.org/wiki/Word_embedding
* matrice d'imbrication (vectorisation de mots)

* self-attention = calculer des références à l'aide de probabilité. e.g. "Le sac ne rentre pas car il est trop grand", "il" refère au "sac"
* = comprendre à partir du contexte la référence
* (Query/key/value) Question/clé/valeur