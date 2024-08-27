
import tensorflow_probability as tfp
import tensorflow as tf
import jax

layers = tf.keras.layers
random = jax.random
jnp = jax.numpy

tfpj = tfp.substrates.jax
tfdj = tfpj.distributions


def get_likelihood_fn(input_data):
    numeric_data, categorical_data = input_data

    def likelihood_fn(prior_weights):
        W_work, W_edu, W_marital, W_occ, W_rel, W_race, W_sex, W_country, W_1, bias_1, W_2, bias_2, W_3, bias_3, W_4, bias_4, W_5, bias_5 = prior_weights

        categorical_Ws =    [W_work, W_edu, W_marital, W_occ, W_rel, W_race, W_sex, W_country]
        W_list =            [W_1, W_2, W_3, W_4, W_5]
        bias_list =         [bias_1, bias_2, bias_3, bias_4, bias_5]

        categorical_features = []

        # gather prior values from categorical data indices
        for i, W in enumerate(categorical_Ws):
            features = jnp.take(W, categorical_data[:, i], axis=0)
            categorical_features.append(features)
        categorical_features = jnp.concatenate(categorical_features, axis=-1)
        
        # Applies a Linear layer: y = W * x + bias and concatenates the categorical features 
        x = jnp.matmul(numeric_data, W_list[0])
        x = jnp.concatenate([x, categorical_features], axis=-1)
        x += bias_list[0]

        # Applies a sequence of Linear layers
        num_layers = len(W_list) - 1
        for i in range(num_layers):
            x = jnp.matmul(x, W_list[i+1])
            x += bias_list[i+1]

        output = tfdj.Independent(tfdj.Categorical(logits=x), reinterpreted_batch_ndims=1)
        return output

    return likelihood_fn


def get_model(categorical_num, train_x, train_y):
    """
    The model is designed specifically for this dataset.
    """
    num_numeric = 5
    num_categories = 8
    num_labels = 2

    scale = 1
    categorical_unit = 16
    numerical_units = 64
    combined_units = numerical_units + categorical_unit * num_categories
    units = [128, 128, 32, num_labels]

    prior = tfdj.JointDistributionSequentialAutoBatched([

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((categorical_num["workclass"], categorical_unit)),
                                    scale_diag = jnp.ones((categorical_num["workclass"], categorical_unit)) * scale),               # W_workclass

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((categorical_num["education"], categorical_unit)),
                                    scale_diag = jnp.ones((categorical_num["education"], categorical_unit)) * scale),               # W_education

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((categorical_num["marital_status"], categorical_unit)),
                                    scale_diag = jnp.ones((categorical_num["marital_status"], categorical_unit)) * scale),          # W_marital

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((categorical_num["occupation"], categorical_unit)),
                                    scale_diag = jnp.ones((categorical_num["occupation"], categorical_unit)) * scale),              # W_occupation

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((categorical_num["relationship"], categorical_unit)),
                                    scale_diag = jnp.ones((categorical_num["relationship"], categorical_unit)) * scale),            # W_relationship

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((categorical_num["race"], categorical_unit)),
                                    scale_diag = jnp.ones((categorical_num["race"], categorical_unit)) * scale),                    # W_race

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((categorical_num["sex"], categorical_unit)),
                                    scale_diag = jnp.ones((categorical_num["sex"], categorical_unit)) * scale),                     # W_sex
                                    
        tfdj.MultivariateNormalDiag(loc = jnp.zeros((categorical_num["native_country"], categorical_unit)),
                                    scale_diag = jnp.ones((categorical_num["native_country"], categorical_unit)) * scale),          # W_country

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((num_numeric, numerical_units)), 
                                    scale_diag = jnp.ones((num_numeric, numerical_units)) * scale),                                 # W_1
        tfdj.MultivariateNormalDiag(loc = jnp.zeros((1, combined_units)), scale_diag = jnp.ones((1, combined_units)) * scale),      # bias_1

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((combined_units, units[0])), 
                                    scale_diag = jnp.ones((combined_units, units[0])) * scale),                                     # W_2
        tfdj.MultivariateNormalDiag(loc = jnp.zeros((1, units[0])), scale_diag = jnp.ones((1, units[0])) * scale),                  # bias_2

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((units[0], units[1])), scale_diag = jnp.ones((units[0], units[1])) * scale),    # W_3
        tfdj.MultivariateNormalDiag(loc = jnp.zeros((1, units[1])), scale_diag = jnp.ones((1, units[1])) * scale),                  # bias_3

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((units[1], units[2])), scale_diag = jnp.ones((units[1], units[2])) * scale),    # W_4
        tfdj.MultivariateNormalDiag(loc = jnp.zeros((1, units[2])), scale_diag = jnp.ones((1, units[2])) * scale),                  # bias_4

        tfdj.MultivariateNormalDiag(loc = jnp.zeros((units[2], units[3])), scale_diag = jnp.ones((units[2], units[3])) * scale),    # W_5
        tfdj.MultivariateNormalDiag(loc = jnp.zeros((1, units[3])), scale_diag = jnp.ones((1, units[3])) * scale),                  # bias_5
    ])

    # gets the likelihood function using the training input data
    likelihood_fn = get_likelihood_fn(train_x)

    # this function takes prior samples as input 
    model_log_prob_fn = lambda *x: 0.1 * prior.log_prob(*x) + likelihood_fn(x).log_prob(train_y,)
    
    # print(model_log_prob_fn(*prior.sample(seed = random.PRNGKey(0))))

    return prior, model_log_prob_fn







