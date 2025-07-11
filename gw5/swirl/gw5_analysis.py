import numpy as np
from scipy.optimize import linear_sum_assignment

import jax
import jax.numpy as jnp
from jax import vmap
import itertools

def get_reward_nm(trans_probs, R_params, apply_fn):
    n_states, n_actions, _ = trans_probs.shape
    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        # Combine one-hot encodings
        return apply_fn({'params': R_params}, one_hot_input)  # Apply the network to get R(hidden, a)
        
    reward_net = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)
    return reward_net

def get_reward_m(trans_probs, R_params, apply_fn):
    n_states, n_actions, _ = trans_probs.shape
    reshape_func = lambda x: (jnp.tile(jnp.expand_dims(x, axis=-1), (1,) * (x.ndim) + (n_states,)) / n_states).reshape(*x.shape[:-1], x.shape[-1] * x.shape[-1])
    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        # Combine one-hot encodings
        return apply_fn({'params': R_params}, reshape_func(one_hot_input))  # Apply the network to get R(hidden, a)
        
    reward_net = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)
    return reward_net


def compute_accuracy(zs_pred, zs_true):
    # Flatten the arrays to 1D
    zs_pred_flat = np.array(zs_pred).flatten()
    zs_true_flat = np.array(zs_true).flatten()

    # Get the union of unique labels from both arrays
    labels = np.unique(np.concatenate((zs_true_flat, zs_pred_flat)))
    K = len(labels)

    # Map each label to a unique index
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((K, K), dtype=int)

    # Populate the confusion matrix
    for t_label, p_label in zip(zs_true_flat, zs_pred_flat):
        idx_true = label_to_index[t_label]
        idx_pred = label_to_index[p_label]
        confusion_matrix[idx_true, idx_pred] += 1

    # Apply the Hungarian algorithm to maximize correct label assignments
    cost_matrix = -confusion_matrix  # Negate for maximization
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create a mapping from predicted labels to true labels
    mapping = {}
    for idx_true, idx_pred in zip(row_ind, col_ind):
        label_true = labels[idx_true]
        label_pred = labels[idx_pred]
        mapping[label_pred] = label_true

    # Apply the optimal label mapping to the predicted labels
    zs_pred_mapped = np.vectorize(lambda x: mapping.get(x, x))(zs_pred_flat)
    zs_pred_mapped = zs_pred_mapped.reshape(zs_pred.shape)

    # Calculate the accuracy
    accuracy = np.mean(zs_pred_mapped == zs_true)

    return accuracy

def nan_corr(x, y):
    # Mask NaN values
    x = x.flatten()
    y = y.flatten()
    mask = ~np.isnan(x) & ~np.isnan(y)
    
    # Apply mask to both x and y
    x_valid = x[mask]
    y_valid = y[mask]
    
    # Compute the Pearson correlation coefficient for valid values
    if len(x_valid) == 0 or len(y_valid) == 0:
        return np.nan  # Return NaN if no valid values
    corr = np.corrcoef(x_valid, y_valid)[0, 1]
    
    return round(corr, 3)

def best_perm_corr(x, y):
    # Store the best correlation and best permutation
    best_corr = -np.inf
    best_perm = None
    
    # Generate all possible permutations of y along axis 0
    for perm in itertools.permutations(y):
        perm = np.array(perm)
        
        # Compute the correlation for the current permutation
        corr = nan_corr(x, perm)
        
        # Check if this is the best correlation so far
        if corr > best_corr:
            best_corr = corr
            best_perm = perm
    
    return best_corr, best_perm

def calibrate_reward(reward_learnt, RG):
    
    # Compute the mean of the difference along axes (1, 2), keeping the dimensions
    mean_diff = np.nanmean(RG - reward_learnt, axis=(1, 2), keepdims=True)
    
    # Update reward_filtered_net
    updated_reward = reward_learnt + mean_diff
    
    return updated_reward

