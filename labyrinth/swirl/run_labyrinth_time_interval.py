import numpy as np
import numpy.random as npr

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
jax.config.update("jax_enable_x64", True)

import sys
if len(sys.argv) < 2:
    seed = 10
    K = 3
else:
    seed = int(sys.argv[1])
    K = int(sys.argv[2])

D_obs = 1
D_latent = 1
C = 127
folder = '../data'
save_folder = '../results'
import os
os.makedirs(save_folder, exist_ok=True)

acs = np.load(folder + '/time_interval_emissions500new.npy')[:, :, 1]
xs = np.load(folder + '/time_interval_emissions500new.npy')[:, :, 0]
trans_probs = np.load(folder + '/trans_probs.npy')

test_indices = np.arange(0, xs.shape[0], 5)
train_indices = np.setdiff1d(np.arange(xs.shape[0]), test_indices)

test_xs, test_acs = xs[test_indices], acs[test_indices]
train_xs, train_acs = xs[train_indices], acs[train_indices]

def compute_next_state_map(trans_probs, n_state, n_action):
    next_state_map = -1*np.ones((n_state, 4), dtype=int)
    
    # Loop through trans_probs to find valid previous states for each x
    for x in range(n_state):
        for next_x in range(n_state):
            for a in range(n_action):
                if trans_probs[x, a, next_x] > 0:
                        next_state_map[x][a] = next_x
    
    return next_state_map

def compute_prev_state_map(trans_probs, n_state, n_action):
    prev_state_map = {x: [] for x in range(n_state)}
    
    # Loop through trans_probs to find valid previous states for each x
    for x in range(n_state):
        for prev_x in range(n_state):
                if np.sum(trans_probs[prev_x, :, x]) > 0:
                        prev_state_map[x].append(prev_x)
    
    return prev_state_map


def construct_new_trans_probs_limited(trans_probs, prev_state_map, next_state_map, n_state, n_action):
    new_trans_probs = np.zeros((n_state * 4, n_action, n_state * 4))
    invalid_indices = np.ones((n_state, 4), dtype=bool)
    for x in range(n_state):
        for prev_x_i in np.arange(4):
            if prev_x_i < len(prev_state_map[x]):
                invalid_indices[x, prev_x_i] = False
            new_state = x * 4 + prev_x_i
            for a in range(n_action):
                next_x = next_state_map[x, a]
                new_next_state = next_x * 4 + prev_state_map[next_x].index(x)
                new_trans_probs[new_state, a, new_next_state] = trans_probs[x, a, next_x]
    return new_trans_probs, invalid_indices

n_state, n_action, _ = trans_probs.shape
# Compute the prev_state_map based on the transitions
next_state_map = compute_next_state_map(trans_probs, n_state, n_action)
prev_state_map = compute_prev_state_map(trans_probs, n_state, n_action)

# Generate new transition probabilities
new_trans_probs, invalid_indices = construct_new_trans_probs_limited(trans_probs, prev_state_map, next_state_map, n_state, n_action)


r1 = np.zeros((127, 4))
r2 = np.zeros((127, 4))
np.random.seed(seed)
if K == 3:
    r3 = npr.rand(127)[:, None]
    r3 = jnp.tile(r3, (1, 4))
    R_start = np.array([r1, r2, r3])
    R_start2 = R_start.mean(axis=-1)
elif K == 2:
    r3 = npr.rand(127)[:, None]
    r3 = jnp.tile(r3, (1, 4))
    R_start = np.array([r1, r3])
    R_start2 = R_start.mean(axis=-1)

# from ssm.swirl import ARHMMs
# npr.seed(seed)
# arhmm_s = ARHMMs(D_obs, K, D_latent, C,
#              transitions="mlprecurrent",
#              dynamics="arcategorical",
#              single_subspace=True)
# list_x = [row for row in train_xs[:, :, np.newaxis].astype(int)]
# lls_arhmm = arhmm_s.initialize(list_x, num_init_iters=100)
# init_start = arhmm_s.init_state_distn.initial_state_distn
# logpi0_start = arhmm_s.init_state_distn.log_pi0
# log_Ps_start = arhmm_s.transitions.log_Ps
# Rs_start = arhmm_s.transitions.W1, arhmm_s.transitions.b1, arhmm_s.transitions.W2, arhmm_s.transitions.b2
# np.savez(folder + '/time_interval_' + str(K) + '_' + str(seed) + '_arhmm_s.npz', init_start=init_start, logpi0_start=logpi0_start, log_Ps_start=log_Ps_start, W1_start=Rs_start[0], b1_start=Rs_start[1], W2_start=Rs_start[2], b2_start=Rs_start[3])

arhmm_s_params = np.load(folder + '/time_interval_' + str(K) + '_10_arhmm_s.npz', allow_pickle=True)
logpi0_start = arhmm_s_params['logpi0_start']
log_Ps_start = arhmm_s_params['log_Ps_start']
Rs_start = arhmm_s_params['W1_start'], arhmm_s_params['b1_start'], arhmm_s_params['W2_start'], arhmm_s_params['b2_start']

def preprocess_xs_prev_np(xs_list, xs_prev_list, prev_state_map):
    prev_indices_list = []
    for xs, xs_prev in zip(xs_list, xs_prev_list):
        prev_indices = []
        for x, prev_x in zip(xs, xs_prev):
            try:
                prev_indices.append(prev_state_map[x].index(prev_x))
            except ValueError:
                print(x, prev_x)
        prev_indices_list.append(prev_indices)
    return np.array(prev_indices_list)

def one_hot_jax(z, K):
    z = jnp.atleast_1d(z).astype(int)
    shp = z.shape
    N = z.size
    zoh = jnp.zeros((N, K))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    zoh = jnp.reshape(zoh, shp + (K,))
    return zoh

def one_hot_jax2(z, z_prev, K):
    z = z * 4 + z_prev
    z = jnp.atleast_1d(z).astype(int)
    K2 = K * 4
    shp = z.shape
    N = z.size
    zoh = jnp.zeros((N, K2))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    zoh = jnp.reshape(zoh, shp + (K2,))
    return zoh

n_states, n_actions, _ = trans_probs.shape
def one_hotx_partial(xs):
    return one_hot_jax(xs[:, None], n_states)
def one_hotx2_partial(xs, xs_prev):
    return one_hot_jax2(xs[:, None], xs_prev[:, None], n_states)
def one_hota_partial(acs):
    return one_hot_jax(acs[:, None], n_actions)

train_xs_prev = preprocess_xs_prev_np(train_xs[:, 1:], train_xs[:, :-1], prev_state_map)
train_xohs = vmap(one_hotx_partial)(train_xs[:, 1:])
train_xohs2 = vmap(one_hotx2_partial)(train_xs[:, 1:], train_xs_prev)
train_aohs = vmap(one_hota_partial)(train_acs[:, 1:])

test_xs_prev = preprocess_xs_prev_np(test_xs[:, 1:], test_xs[:, :-1], prev_state_map)
test_xohs = vmap(one_hotx_partial)(test_xs[:, 1:])
test_xohs2 = vmap(one_hotx2_partial)(test_xs[:, 1:], test_xs_prev)
test_aohs = vmap(one_hota_partial)(test_acs[:, 1:])

all_xs_prev = preprocess_xs_prev_np(xs[:, 1:], xs[:, :-1], prev_state_map)
all_xohs = vmap(one_hotx_partial)(xs[:, 1:])
all_xohs_prev = vmap(one_hotx_partial)(xs[:, :-1])
all_xohs2 = vmap(one_hotx2_partial)(xs[:, 1:], all_xs_prev)
all_aohs = vmap(one_hota_partial)(acs[:, 1:])


from swirl_func import jax_e_step_batch_temp, pi0_m_step, trans_m_step_jax_scipy2, emit_m_step_jax_scipy_temp_reg

def em_train_temp(logpi0, log_Ps, Rs, rewards, temps, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch_temp(pi0, log_Ps, Rs, rewards, trans_probs, temps, train_xohs, train_aohs)
        print(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        print(new_logpi0)

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_scipy2(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_rewards = emit_m_step_jax_scipy_temp_reg(rewards, trans_probs, temps,(all_gamma_jax, all_xi_jax), jnp.array(train_xohs), jnp.array(train_aohs))
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list

def normalize(reward, indices=[0]):
    out = reward.copy()
    if isinstance(indices, int):
        indices = [indices]
    for k in indices:
        x = out[k, 0, :]
        x = (x - x.min())**4
        x = x / x.max()
        out[k, 0, :] = x
    return out

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
temps = jnp.array([1] + [1] * (K - 1))

# S-1
new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(jnp.array(logpi0_start), jnp.array(log_Ps_start), Rs_start, jnp.array(R_start2)[:, None], temps, 50, init=False, trans=False)
# jnp.savez(save_folder + '/time_interval_' + str(K) + '_' + str(seed) + '_NM_labyrinth1_init.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)
new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(jnp.array(new_logpi0), jnp.array(new_log_Ps), new_Rs, jnp.array(new_reward), temps, 30)

new_reward = normalize(new_reward)

new_reward = new_reward[[1, 0, 2], ...]
jnp.savez(save_folder + '/time_interval_' + str(K) + '_' + str(seed) + '_NM_labyrinth1.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)