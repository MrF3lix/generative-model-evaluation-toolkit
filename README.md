# Generative Model Evaluation Toolkit

A toolkit for robust evaluation of generative models.

## Highlights

- 🚀 A complete evaluation toolkit for generative models
- ✅ Correct generative model evaluations obtained by trainable classifiers 
- 💡 Finally compare different generative models without arbitray biases
- 📱 Independent of the generated modality (Text, Image, Audio, etc.)
- 🔬 Based on scientific [advancements](https://ojs.aaai.org/index.php/ICWSM/article/view/31411) in quantification methods

## Evaluation Steps

1. Generate a test set using your generative model
```bash
uv run generate
```
2. Manually Annotate a subsample (n=100 depending on the task).
```bash
uv run annotate
```
> **Note**: *If you provide the confusion matrix for you classifier you can skip this step.*
3. Evaluate your generative model
```bash
uv run evaluate
```


## Prerequisites

- Python package and project manager [`UV`](https://github.com/astral-sh/uv)

## Installation

If you want to use the `cgeval` library to implement in your own pipelines follow the [`cgeval` library](#cgeval-library) guide.

If you want to use this exact toolkit follow these steps:

1. Install prerequisits
   1. Python package and project manager [`UV`](https://github.com/astral-sh/uv)
2. Clone Repository
3. Run `uv sync` in the root folder of the repository
4. Create your custom `config.yaml` file to define your pipeline.

## Architecture

Read more about the [architecture](./docs/architecture.md) and decisions during the development of this tooklit.

## Config

The `config.yaml` file is split into five sections:

### `env`
Configurations related to the environment.

|Name|Type / Options|Default|Description|
|-|-|-|-|
|`device`|`cpu`</br>`cuda`</br>`mps`|`cpu`|Defines on which device the weights and samples are stored during processing.|

### `model`
Configurations related to the generative model.

|Name|Type / Options|Default|Description|
|-|-|-|-|
|`type`|`llm`</br>`diffusion`</br>`ollama`||Descibes the type of the model that is used. From it the modality is also inferred.|
|`url`|String||Required if type is `ollama`. Endpoint of the ollama REST API|
|`name`|String||Local Path, Huggingface Name, Ollama Model Name|

### `dataset`
Configurations related to the evaluation dataset.

|Name|Type / Options|Default|Description|
|-|-|-|-|
|`type`|`local_image`</br>`hf`||Describes the type and origin of the dataset.|
|`name`|String||Local Path, Huggingface Name|
|`batch_size`|Number|None|If provided batches the dataset. Otherwise a single batch is used.|
|`samples`|Number|None|If provided only the specified number of sampels are taken from the dataset. Otherwise all samples are used.|

### `classifier`
Configurations related to the classifier. 
Multiple classifiers can be used.

|Name|Type / Options|Default|Description|
|-|-|-|-|
|`type`|`llm`</br>`diffusion`</br>`ollama`||Descibes the type of the model that is used. From it the modality is also inferred.|
|`url`|String||Required if type is `ollama`. Endpoint of the ollama REST API|
|`name`|String||Local Path, Huggingface Name, Ollama Model Name|
|`output`|`class`</br>`logits`||Defines if the classifier outputs logits or class labels directly.|
|`labels`|String[]||List of labels the classifier can assign.|


### `method`
Configurations related to the quantification method that is applied.

|Name|Type / Options|Default|Description|
|-|-|-|-|
|`method`|`CC`||Quantification Method used|

### Sample
Here is an example of the configuration shown.

```yaml
env:
  device: cuda
model:
  type: llm
  name: mistralai/Mistral-7B-Instruct-v0.1
dataset:
  type: hf
  name: Sp1786/multiclass-sentiment-analysis-dataset
  batch_size: 25
classifier:
  type: ollama
  url: http://localhost:11434/api/chat
  name: llama3
  output: class
  labels: 
    - positive
    - neutral
    - negative
evaluation:
  method: CC
```

The generative model that is being evaluated is a LLM running on the local machine using the `transformers` library from HuggingFace.

The `multiclass-sentiment-analysis-dataset` dataset from HuggingFace is used to evaluate the model.

As the classifier a local version of `llama3` throught the Ollama application is used.

The naive evaluation method used is `CC` (Classify and Count).


## `cgeval` library

Learn more about the [`cgeval` library](./package/cgeval/README.md).