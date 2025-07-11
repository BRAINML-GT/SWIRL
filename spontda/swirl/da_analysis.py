import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap


def get_reward_nm(trans_probs, R_params, apply_fn):
    n_states, n_actions, _ = trans_probs.shape
    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        # Combine one-hot encodings
        return apply_fn({'params': R_params}, one_hot_input)
        
    reward_net = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)
    return reward_net

def get_reward_m(trans_probs, R_params, apply_fn):
    n_states, n_actions, _ = trans_probs.shape
    reshape_func = lambda x: (jnp.tile(jnp.expand_dims(x, axis=-1), (1,) * (x.ndim) + (n_states,)) / n_states).reshape(*x.shape[:-1], x.shape[-1] * x.shape[-1])
    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        # Combine one-hot encodings
        return apply_fn({'params': R_params}, reshape_func(one_hot_input))
        
    reward_net = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)
    return reward_net

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

def zscore(res_R, axis=1):
    matrix = res_R
    column_means = np.nanmean(matrix, axis=axis, keepdims=True)

    column_stddevs = np.nanstd(matrix, axis=axis, keepdims=True)

    centered_matrix = matrix - column_means

    z_score_R_sa = centered_matrix / column_stddevs

    return z_score_R_sa

def get_DA_Rsa(pred_zs, xs, acs, DAs, K, C):
    DA_R = np.zeros((K, C, C))
    count = np.zeros((K, C, C))

    for trial_id in range(xs.shape[0]):
        for j in range(xs.shape[1]-1):
                DA_R[int(pred_zs[trial_id, j]), int(xs[trial_id, j]), int(acs[trial_id, j])] += DAs[trial_id, j]
                count[int(pred_zs[trial_id, j]), int(xs[trial_id, j]), int(acs[trial_id, j])] += 1
    DA_R = DA_R/count
    return DA_R

def get_corr(learnt_R, DA_R, C):
    learnt_R[:, np.arange(C), np.arange(C)] = np.nan
    DA_R[:, np.arange(C), np.arange(C)] = np.nan

    corr_list = []
    for i in range(DA_R.shape[0]):
        corr_list.append(nan_corr(learnt_R[i], DA_R[i]))
    return corr_list, DA_R, learnt_R