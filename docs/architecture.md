# Architecture

The evaluation toolkit is build on the following sketch.

![](./assets/library.dio.svg)

The toolkit has three main tasks:

1. **Generate**
   
   Generate a dataset that can be used for evaluation. The type of model is not decisive, any modality will work as long as the generation is dependent on a condition. For example in machine translation the condition is the language of the generated output.

2. **Annotate**

    Supports the developer to annotate a subsample of the generated dataset. This annotated sample is seen as the ground truth and will be used to calibrate the classifier.
    If the turth table of the classifier is already known this step can be omitted.

3. **Evaluate**

    This is the heart of the toolkit. It offers different quantitative methods to evaluate the generative models using the outputs from the classifier as well as the manually annotated samples to calibrate the classifiers.

Each of these tasks can be run independently.