import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import pickle

layers = tf.keras.layers
tfd = tfp.distributions

from tfp_vi_model import get_likelihood_fn, get_model
from tfp_advi import tfp_vi_fit, unravel_func



def get_dataframe(file_path, test_data=False):
    headers = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income"
        ]

    df = pd.read_csv(file_path, header=None, names=headers)

    # for training
    if not test_data:
        # to shuffle the data
        df = df.sample(frac=1)
        df = df.loc[:20000]

    return df



def get_dataset(df, categorical_vocab, test_data=False):
    """
    This function splits the data into inputs and labels and preprocesses them.
    """
    if test_data:
        df = df.loc[1:] # skip first line
        labels = df.loc[:, "income"].apply(lambda x: 0 if x==" <=50K." else 1)  # extra dot
    else:
        labels = df.loc[:, "income"].apply(lambda x: 0 if x==" <=50K" else 1)
    labels = tf.constant(labels, dtype=tf.int32)

    numeric_dataset = df.loc[:, ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]]
    numeric_dataset = tf.constant(numeric_dataset, dtype=tf.float32)

    # normalizes the numeric data
    numeric_data_norm = tf.linalg.norm(numeric_dataset, axis=0) + 1e-6
    numeric_dataset /= numeric_data_norm

    categorical_features = ["workclass", "education", "marital_status", "occupation",
                                "relationship", "race", "sex", "native_country"]

    # uses StringLookup to convert categorical_features into tokens
    for feature in categorical_features:
        lookup = layers.StringLookup(vocabulary=categorical_vocab[feature])
        df.loc[:, feature] = lookup(df.loc[:, feature])
    categorical_dataset = tf.constant(df.loc[:, categorical_features], dtype=tf.int32)

    return [numeric_dataset, categorical_dataset], labels



def get_train_data(file_path):
    train_df = get_dataframe(file_path)

    categorical_features = ["workclass", "education", "marital_status", "occupation",
                                "relationship", "race", "sex", "native_country"]

    # gets a StringLookup layer for categorical features and the length of each vocabulary
    categorical_num, categorical_vocab = {}, {}
    for feature in categorical_features:
        vocab = sorted(list(train_df.loc[:, feature].unique()))
        categorical_num[feature] = len(vocab) + 1   # for 1 oov_index
        categorical_vocab[feature] = vocab

    train_x, train_y = get_dataset(train_df, categorical_vocab)
    return train_x, train_y, categorical_num, categorical_vocab



def get_test_data(file_path, categorical_vocab):
    test_df = get_dataframe(file_path, test_data=True)
    return get_dataset(test_df, categorical_vocab, test_data=True)



def run_training():
    """
    This script trains the tfp_vi model
    """

    train_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # train_path = "census/adult.data"

    train_x, train_y, categorical_num, categorical_vocab = get_train_data(train_path)
    prior, log_prob_fn, num_prior_parts = get_model(categorical_num, train_x, train_y)
    event_shapes = prior.event_shape

    # Training the model
    posterior, loss = tfp_vi_fit(prior, log_prob_fn, num_prior_parts, num_steps=500, sample_size=150, learning_rate=0.01)

    # save the posterior distribution (weights)
    weights_dict = {"loc": posterior.loc,
                    "scale": posterior._scale_diag,                     
                    "event_shapes": event_shapes,
                    "vocab_dict": categorical_vocab}    
    pickle.dump(weights_dict, open("tfp_vi_weights_1.pkl", "wb"))



def test_step(posterior, event_shapes, test_x, test_y):    
    # posterior_sample = tf.math.reduce_mean(posterior.sample(1000), axis=0)

    # takes the mean instead of sampling
    posterior_sample = posterior.loc

    # flattens the posterior samples
    posterior_sample = unravel_func(posterior_sample, event_shapes)

    # gets the likelihood function using the test data
    test_likelihood_fn = get_likelihood_fn(test_x)

    # applies the trained weights to the likelihood function and samples the output
    test_dist = test_likelihood_fn(posterior_sample)
    test_pred = test_dist.sample(seed=0)

    m = tf.keras.metrics.BinaryAccuracy()
    m.update_state(test_y, test_pred)
    return m.result()



def run_test():
    """
    This script tests the tfp_vi model
    """
    test_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    # test_path = "census/adult.test"

    weights_dict = pickle.load(open("tfp_vi_weights.pkl", "rb"))
    posterior = tfd.MultivariateNormalDiag(loc=weights_dict["loc"], scale_diag=weights_dict["scale"])
    categorical_vocab = weights_dict["vocab_dict"]
    event_shapes = weights_dict["event_shapes"]

    test_x, test_y = get_test_data(test_path, categorical_vocab)
    test_accuracy = test_step(posterior, event_shapes, test_x, test_y)
    return test_accuracy




