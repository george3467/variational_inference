
import tensorflow_probability as tfp
import jax
import optax

random = jax.random
jnp = jax.numpy

tfpj = tfp.substrates.jax
tfdj = tfpj.distributions
tfbj = tfpj.bijectors



def get_loss_func(posterior_bijector, unravel_func, model_log_prob_fn, n_samples):
    """
    This function calculates the variational inference loss
    """
    def loss_func(surrogate_posterior, seed):

        # flattens the surrogate poster
        flat_posterior, unflatten_func = jax.tree_util.tree_flatten(surrogate_posterior)

        # applies the bijectors to their respective parts in the posterior
        transformed_posterior = jax.tree_util.tree_map(lambda section, bijector: bijector(section), 
                                                        flat_posterior, posterior_bijector)
        
        # unflattens the posterior to its original structure to make the surrogate posterior
        surrogate_posterior = jax.tree_util.tree_unflatten(unflatten_func, transformed_posterior)

        # takes n samples from the surrogate posterior
        samples = surrogate_posterior.sample(seed=seed, sample_shape=(n_samples,))
        
        def loss_loop(sample):
            posterior_log_prob = surrogate_posterior.log_prob(sample)

            # unflattens the sample to the prior stucture and gets its log_prob
            unravel_sample = unravel_func(sample)   
            model_log_prob = model_log_prob_fn(*unravel_sample)

            # variational inference loss
            return 0.1 * posterior_log_prob - model_log_prob

        # iterates over the n samples of the surrogate posterior
        results = jax.vmap(loss_loop)(samples).mean()

        return results
    
    return loss_func


def train_loop(loss_func, posterior, optimizer, n_epochs):
    """
    This function trains the posterior by minimizing the loss
    """
    value_and_grad = jax.value_and_grad(loss_func)
    state = optimizer.init(posterior)
    seeds = jax.random.split(random.PRNGKey(0), n_epochs)

    def one_step(prev_info, seed):
        posterior, state = prev_info
        loss, gradients = value_and_grad(posterior, seed=seed)
        updates, state = optimizer.update(gradients, state)
        posterior = optax.apply_updates(posterior, updates)
        return (posterior, state), (loss, posterior)
    
    (posterior, _), (loss_history, posterior_history) = jax.lax.scan(one_step, (posterior, state), xs=seeds)
    return posterior, loss_history, posterior_history


def jax_vi_fit(prior, model_log_prob_fn, n_epochs, n_samples):
    """
    This function runs through the variational inference algorithm.
    """
    prior_samples = prior.sample(seed = random.PRNGKey(0))

    # flattens the prior sample
    array, unravel_func = jax.flatten_util.ravel_pytree(prior_samples)

    # legnth of the flat prior sample
    prior_length = len(array)

    seed1, seed2 = random.split(random.PRNGKey(0), 2)

    # the covariance will be a diagonal matrix
    surrogate_posterior = tfdj.MultivariateNormalDiag(loc=jax.random.normal(seed1, (prior_length,)), 
                                                        scale_diag=jax.random.normal(seed2, (prior_length,)) * 0.1)
    posterior_bijector = [tfbj.Identity(), tfbj.Exp()]

    loss_func = get_loss_func(posterior_bijector, unravel_func, model_log_prob_fn, n_samples)
    optimizer = optax.adam(learning_rate=0.01)

    posterior, loss_history, posterior_history = train_loop(loss_func, surrogate_posterior, optimizer, n_epochs)
    return posterior, loss_history, unravel_func



