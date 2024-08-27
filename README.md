# Variational Inference applied to a Deep Network

* Trained on JAX 0.4.23 and Tensorflow Probability 0.23.0


## Contents
* [Repository Files](#repository-files)
* [Model](#model)
* [Variational Inference](#variational-inference)
* [Dataset](#dataset)
* [Results](#results)


## Repository Files

* jax_advi.py - This file contains the JAX ADVI fit function
* jax_train_and_test.py - This file contains the preprocessing functions, and the training and testing scripts for the JAX model.
* jax_vi_model.py - This file contains the JAX model
* jax_vi_weights.pkl - This file contains the trained weights for the JAX model.
<br><br>
* tfp_advi.py - This file contains the TFP ADVI fit function
* tfp_train_and_test.py - This file contains the preprocessing functions, and the training and testing scripts for TFP model.
* tfp_vi_model.py - This file contains the TFP model
* tfp_vi_weights.pkl - This file contains the trained weights for the TFP model


## Model

The model is implemented in both JAX and Tensorflow Probability (TFP). It consists of a sequence of linear layers with a Categorical distribution layer at the end. The priors of the weights and biases are defined as normal distributions. The log probability function is defined as a weighted sum of the log probability of the prior and the log probability of the likelihood function. More weight is placed on the log probability of the likelihood function.


## Variational Inference

This model is trained using Automatic Differentiaion Variational Inferenece (ADVI). Here, ADVI is implemented using both JAX and Tensorflow Probability (TFP). The surrogate posterior is defined as a MultivariateNormalDiag distribution.

For the TFP version of ADVI, ideas were taken from the following Tensorflow Probablity Tutorial:

```Bibtex
https://tensorflow.org/probability/examples/Modeling_with_JointDistribution
```

The Jax version of ADVI is based on the book "Probabilistic Machine Learning: Advanced Topics" by Kevin P. Murphy and its associated github repository, Bijax.

Some important differences between this implementation of ADVI and the implemntation in the Bijax repository are described below:

* <u>Prior & Likelihood</u>:<br>
In the Bijax repository, in the loss step, the prior and the likelihood function are provided separately. In this project, the prior and likelihood function are passed to the loss step as a single function, "model_log_prob_fn."

* <u>Loss Function</u>:<br>
In this project, the loss value is calculated as a weighted sum. Here, the log-probability value of the surrogate posterior is given a weight of 0.1 and the "model_log_prob" has a weight of 1.

* <u>Loss Function</u>:<br>
In this project, the surrogate posterior's scale_diagonal is initialized with a normal distribution and then scaled by a factor of 0.1.

Link to the Bijax Repository:
```Bibtex
https://github.com/patel-zeel/bijax
```
Reference to the Probabilistic Machine Learning book:
```Bibtex
@book{pml2Book,
  author = "Kevin P. Murphy",
  title = "Probabilistic Machine Learning: Advanced Topics",
  publisher = "MIT Press",
  year = 2023,
  url = "http://probml.github.io/book2"
}
```


## Dataset

The two models were trained on the Census Income dataset. This dataset provides information for 14 features for each instance. The goal is to predict whether the income level exceeds $50,000 for each instance. Both models were trained on 20,000 instances from the train dataset.

Reference to the dataset:
```Bibtex
@misc{misc_census_income_20,
  author       = {Kohavi,Ron},
  title        = {{Census Income}},
  year         = {1996},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5GP7S}
}
```


## Results

The test dataset has 16,282 instance. The TFP model achieved an accuracy of 84.3% and the JAX model achieved an accuracy of 84.2% on the test dataset.<br> 
The model's accuracy could be improved by:
* increasing the dimensions of the custom linear layers
* using more training data
* using a larger sample size in ADVI
