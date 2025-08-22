# Generative Model Evaluation Toolkit

This toolkit was developed as part of a project to showcase how conditional generation can be evaluationed beyond the Classify & Count (CC) method.

- 🚀 A complete evaluation toolkit for conditional generation models
- 📱 Independent of the generated modality (Text, Image, Audio, etc.)
- ✅ Achieve robust and unbiased estimates of the true performance
- 💡 Finally compare different generative models without arbitray biases

The full report can be found in [VT2_controlled_text_generation_evaluation.pdf](./VT2_controlled_text_generation_evaluation.pdf).

## Abstract

Attribute control in controllable generation is typically evaluated using pretrained classifiers.
It has been shown that the common Classify & Count (CC) method leads to biased and inconsistent results.
Estimates are found to vary significantly across different classifiers.
In this project, Attribute Control Success (ACS) estimation is framed as a quantification task.
A hybrid Bayesian method, called Bayesian Classify & Count (BCC), is applied, in which classifier predictions are combined with a small number of human labels used for calibration.

To evaluate the method, a dual modality benchmark containing both text and image samples was collected.
It consists of 600 human annotated samples and 60′000 metric annotated samples. Through experiments, it is shown that the BCC method produces robust estimates across both text and text-to-image generation tasks.
It is also shown that the BCC method enables consistent pairwise comparisons of model performance across different classifiers, yielding stable rankings among generators.

Furthermore, the information gain from metric annotations is quantified, highlighting the added value of human annotations over metric-based ones.
Thus, a more reliable and principled alternative to existing evaluation practices is provided.

As part of this project, a toolkit was developed that can be used—either in parts or as a whole—to estimate ACS on various generative models.
Support is provided for both text and text-to-image generation in combination with binary attribute control.

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

## How to run it

1. Generate a test set using your generative model
```bash
uv run generate
```
2. Manually Annotate a subsample (n=100 depending on the task).
```bash
uv run annotate
```
3. Evaluate your generative model using pre-trained classifiers
```bash
uv run evaluate
```
4. Quantify & correct the biased results
```bash
uv run quantify
```

The tasks and their dependencies are descirbed in the section [Tasks](#tasks)


## Config

The `config.yaml` file is split into five sections:

### `experiment`
Configurations related to the environment.

| Name          | Type / Options | Default | Description                                                  |
| ------------- | -------------- | ------- | ------------------------------------------------------------ |
| `name`        | String         |         | Used to name the results folder and identify the experiment. |
| `report_path` | String         |         | Folder where the reports are stored in.                      |

### `env`
Configurations related to the environment.

| Name     | Type / Options             | Default | Description                                                                   |
| -------- | -------------------------- | ------- | ----------------------------------------------------------------------------- |
| `device` | `cpu`</br>`cuda`</br>`mps` | `cpu`   | Defines on which device the weights and samples are stored during processing. |

### `model`
Configurations related to the generative model.

| Name          | Type / Options                     | Default | Description                                                                                      |
| ------------- | ---------------------------------- | ------- | ------------------------------------------------------------------------------------------------ |
| `type`        | `llm`</br>`diffusion`</br>`ollama` |         | Descibes the type of the model that is used. From it the modality is also inferred.              |
| `url`         | String                             |         | Required if type is `ollama`. Endpoint of the ollama REST API                                    |
| `name`        | String                             |         | Local Path, Huggingface Name, Ollama Model Name                                                  |
| `samples`     | Number                             |         | Number of samples that should be generated                                                       |
| `base_prompt` | String                             |         | Base prompt that is used for the generation                                                      |
| `labels`      | List                               |         | List of labels that should be used during the generation, each label has a `name` and a `ratio`. |

### `dataset`
Configurations related to the evaluation dataset.

| Name         | Type / Options                          | Default | Description                                                                                                  |
| ------------ | --------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `type`       | `local_image`</br>`local_text`</br>`hf` |         | Describes the type and origin of the dataset.                                                                |
| `name`       | String                                  |         | Local Path, Huggingface Name                                                                                 |
| `batch_size` | Number                                  | None    | If provided batches the dataset. Otherwise a single batch is used.                                           |
| `samples`    | Number                                  | None    | If provided only the specified number of sampels are taken from the dataset. Otherwise all samples are used. |

### `classifier`
Configurations related to the classifier. 
Multiple classifiers can be used.

| Name     | Type / Options                                        | Default | Description                                                                         |
| -------- | ----------------------------------------------------- | ------- | ----------------------------------------------------------------------------------- |
| `type`   | `llm`</br>`diffusion`</br>`ollama`</br>`transformers` |         | Descibes the type of the model that is used. From it the modality is also inferred. |
| `url`    | String                                                |         | Required if type is `ollama`. Endpoint of the ollama REST API                       |
| `name`   | String                                                |         | Local Path, Huggingface Name, Ollama Model Name                                     |
| `output` | `class`</br>`logits`                                  |         | Defines if the classifier outputs logits or class labels directly.                  |
| `labels` | String[]                                              |         | List of labels the classifier can assign.                                           |


### `method`
Configurations related to the quantification method that is applied.

| Name     | Type / Options   | Default | Description                                                                       |
| -------- | ---------------- | ------- | --------------------------------------------------------------------------------- |
| `method` | `CC`             |         | Quantification Method used                                                        |
| `method` | `Classification` |         | Creates a standard classification report. Requires a dataset with a ground truth. |

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

## Tasks
The toolkit consists of three tasks.

1. Generate
2. Annotate
3. Evaluate


### Generate

Generates a dataset using a provided model, class distributions, and expected sample size.

The generated dataset looks like this:

| $I$   | $o=\pi(i)$ |
| ----- | ---------- |
| $i_1$ | $o_1$      |
| ...   | ...        |
| $i_n$ | $o_n$      |

The first column $I$ is the input, usually the condition, to the generative model $\pi$.
The second column describes the generated output based on the given input condition $i$.

The following options are required for the generation task:

- $\pi$: Generative model
- $n$: Number of samples to generate
- Distribution of the input features


### Annotate

Helps to annotate a subsample of the generated dataset.

The annotated dataset then looks like this:

| $I$   | $o=\pi(i)$ | $\Phi$     |
| ----- | ---------- | ---------- |
| $i_1$ | $o_1$      | $\omega_1$ |
| ...   | ...        | ...        |
| $i_k$ | $o_k$      | $\omega_k$ |


- $k$: Number of samples that will should an annotation

### Evaluate
### Quantify

Extends the dataset with a column for each classifier $\Mu$.


| $I$       | $o=\pi(i)$ | $\Phi$     | $\Mu_b$   |
| --------- | ---------- | ---------- | --------- |
| $i_1$     | $o_1$      | $\omega_1$ | $m_1$     |
| ...       | ...        | ...        | ...       |
| $i_k$     | $o_k$      | $\omega_k$ | $m_k$     |
|           |            |            |
| $i_{k+1}$ | $o_{k+1}$  |            | $m_{k+1}$ |
| ...       | ...        |            | ...       |
| $i_{n}$   | $o_{n}$    |            | $m_{n}$   |

- $n$: Number of samples
- $k$: Number of samples that have an oracle evaluation

From this table the different parameters for the quantification methods can be computed.

## `cgeval` library

Learn more about the [`cgeval` library](./package/cgeval/README.md).