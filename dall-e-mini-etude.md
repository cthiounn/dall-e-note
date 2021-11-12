# Etude de l'architecture du mini dall-e à travers le code

* https://github.com/borisdayma/dalle-mini.git

```
  |-app
  |  |-app.py
  |  |-gradio
  |  |  |-app_gradio.py
  |  |  |-app_gradio_ngrok.py
  |  |  |-requirements.txt
  |  |-img
  |  |  |-loading.gif
  |-CITATION.cff
  |-dalle_mini
  |  |-backend.py
  |  |-dataset.py
  |  |-helpers.py
  |  |-model.py
  |  |-text.py
  |  |-__init__.py
  |-dev
  |  |-data
  |  |  |-CC12M_downloader.py
  |  |  |-CC3M_downloader.py
  |  |  |-README.md
  |  |-encoding
  |  |  |-vqgan-jax-encoding-streaming.ipynb
  |  |  |-vqgan-jax-encoding-webdataset.ipynb
  |  |  |-vqgan-jax-encoding-with-captions.ipynb
  |  |  |-vqgan-jax-encoding-yfcc100m.ipynb
  |  |  |-vqgan-jax-encoding.ipynb
  |  |-environment.yaml
  |  |-inference
  |  |  |-inference_pipeline.ipynb
  |  |  |-README.md
  |  |  |-samples.txt
  |  |  |-wandb-backend.ipynb
  |  |  |-wandb-examples-from-backend.py
  |  |  |-wandb-examples.py
  |  |-README.md
  |  |-requirements.txt
  |  |-seq2seq
  |  |  |-do_big_run.sh
  |  |  |-do_small_run.sh
  |  |  |-run_seq2seq_flax.py
  |  |  |-sweep.yaml
  |  |-vqgan
  |  |  |-JAX_VQGAN_f16_16384_Reconstruction.ipynb
  |-img
  |  |-logo.png
  |-LICENSE
  |-README.md
  |-setup.cfg
  |-setup.py
```

## App avec gradio

* expose le modèle statistique au travers d'une interface : https://gradio.app/

* utilise gradio (UI), flax (réseau de neurones avec JAX), transformers
    * https://github.com/gradio-app/gradio
    * https://github.com/google/flax
    * https://github.com/huggingface/transformers

* deux interfaces, l'un avec ngrok et l'autre sans
    * post la requete au backend (get_images_from_ngrok) qui effectue le travail
    *  l'autre fait le traitement

### 1 - génération des images à partir du texte
```
flow :
prompt -> p=tokenizer(prompt) pour -> i=model.generate(p) -> vqgan.decode(i) -> images
```
* tokenizer = BartTokenizer.from_pretrained
* model = CustomFlaxBartForConditionalGeneration.from_pretrained    [dalle/model.py]
* vqgan = VQModel.from_pretrained

### 2 - classement par CLIP

```
flow :
i=prompt+images -> p=processor(i) -> clip(p) -> sort by scores

```

* processor = CLIPProcessor.from_pretrained
* clip = FlaxCLIPModel.from_pretrained


## Dalle-mini

### model.py

```
import flax.linen as nn

from transformers.models.bart.modeling_flax_bart import (
    FlaxBartModule,
    FlaxBartForConditionalGenerationModule,
    FlaxBartForConditionalGeneration,
    FlaxBartEncoder,
    FlaxBartDecoder
)

from transformers import BartConfig
```
* deux embeddings : shared (utilisé par l'encoder) + decoder-only
* lm_head : transformation linéaire (nn.dense)

## Dev

### data



* https://github.com/google-research-datasets/conceptual-12m 


> utilitaires pour télécharger et transformer les images CC3M et CC12M, qui sont des images avec description

* CC3M_downloader.py
* usage : python CC3M_downloader.py Train-GCC-training.tsv training

* input : fichier tsv avec les noms des fichiers + répertoire où se trouve les fichiers
* résultat : sauvegarde les images transformées par torchvision.transforms  (RGB, redimensionnement avec interpolation Lanczos)



* CC12M_downloader.py
* meme chose mais avec les images après 10 000 000 ?

### encoding

> plusieurs notebook pour encoder les images selon différents cas d'utilisation

* vqgan-jax-encoding-streaming.ipynb
> en mode streaming (dalle-mini/YFCC100M_OpenAI_subset) avec VQGAN préentrainé sur flax-community/vqgan_f16_16384
* vqgan-jax-encoding-webdataset.ipynb
>  avec un webdataset et VQGAN préentrainé sur flax-community/vqgan_f16_16384
* vqgan-jax-encoding-with-captions.ipynb
> avec un CaptionDataset (CC12M) avec VQGAN préentrainé sur flax-community/vqgan_f16_16384
* vqgan-jax-encoding-yfcc100m.ipynb
> avec un CaptionDataset (YFCC100M_OpenAI_subset) avec VQGAN préentrainé sur flax-community/vqgan_f16_16384
* vqgan-jax-encoding.ipynb
> encodage avec VQGAN préentrainé sur valhalla/vqgan-imagenet-f16-1024

### inference

### seq2seq

### vqgan

> un notebook pour tester l'encodage et le decodage à partir du modèle VQGAN préentrainé flax-community/vqgan_f16_16384