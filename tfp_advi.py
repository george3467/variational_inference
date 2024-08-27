
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow_probability.python.mcmc.transformed_kernel import make_transformed_log_prob
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors


def ravel_func(input, event_shape):
    """
    This function performs a similar task as the jax.flatten_util.ravel_pytree function.
    It flattens a pytree into a flat vector.
    """
    def reshape_func(section, event_shape):

        # gets the number of batch dimensions (usually 1 dimension)
        num_batch_dims = tf.size(np.shape(section)) - tf.size(event_shape)

        # gets the batch dimensions
        batch_dims = tf.shape(section)[:num_batch_dims]

        # flattens the remaining dimensions after the batch dimensions
        # shape = (batch_dims, -1)
        shape = tf.concat([batch_dims, [-1]], axis=-1)
        return tf.reshape(section, shape)
    
    # iterates for each section in the event_shape
    x = tf.nest.map_structure(reshape_func, input, event_shape)

    # concatenates flattened parts together into a single vector
    return tf.concat(tf.nest.flatten(x), axis=-1)


def unravel_func(input, event_shape):
    """
    This function performs a similar task as the unflatten function returned from the
    jax.flatten_util.ravel_pytree function.
    It unflattens a 1D vector into a pytree with the original structure.
    """
    # gets the flattened lengths for each part in the event_shape
    splits = [tf.math.reduce_prod(shape) for shape in tf.nest.flatten(event_shape)]

    # splits the flattened vector according to the flattened lengths of each part
    input = tf.split(input, splits, axis=-1)

    def reshape_func(section, event_shape):

        # any dimension before the last dimension are the batch dimensions
        batch_dims = tf.shape(section)[:-1]

        # the original shape before ravel_func was applied
        shape = tf.concat([batch_dims, event_shape], axis=-1)

        # reshape to the original shape
        return tf.reshape(section, shape)
    
    # iterates for each section in the event_shape
    return tf.nest.map_structure(reshape_func, input, event_shape)


def tfp_vi_fit(prior, log_prob_fn, num_prior_parts, num_steps, sample_size, learning_rate=1e-3, mean_field=True):
    """
    This function fits a surrogate posterior to our model. 
    """
    
    # setting a bijector for each part of the prior
    bijectors = [tfb.Identity() for i in range(num_prior_parts)]

    # creating a new function that applies the bijector to the prior parts of the log_prob_fn
    transform_log_prob = make_transformed_log_prob(log_prob_fn, bijectors, direction='forward', enable_bijector_caching=False)

    # samples the prior and gets its event shape
    prior_sample = prior.sample()
    event_shapes = prior.event_shape

    # flattens the prior sample
    prior_flat_sample = ravel_func(prior_sample, event_shapes)
    prior_length = tf.shape(prior_flat_sample)

    # initializes the mean of the posterior
    initial_value = tf.random.normal(prior_length)
    loc = tf.Variable(initial_value)

    # the covariance will only be a diagonal matrix
    if mean_field:
        scale = tfp.util.TransformedVariable(0.1 * tf.ones(prior_length), tfb.Softplus()) 
        posterior = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
    
    # the covariance will be a lower triangular matrix
    else:
        cov_param = tfp.util.TransformedVariable(0.1 * tf.eye(prior_length.numpy()[0]), tfb.Softplus())
        posterior = tfd.MultivariateNormalTriL(loc=loc, scale_tril=cov_param)

    # this function will unflatten the input before calculating the log_prob
    log_prob_func = lambda x: 0.1 * transform_log_prob(*unravel_func(x, event_shapes))

    # fits the surrogate posterior to the log_probability function
    loss = tfp.vi.fit_surrogate_posterior(log_prob_func, 
                                          surrogate_posterior=posterior, 
                                          optimizer=tf.keras.optimizers.Adam(learning_rate),
                                          num_steps=num_steps,
                                          sample_size=sample_size,
                                          )
    
    return posterior, loss