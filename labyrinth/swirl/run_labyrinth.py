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

acs = np.load(folder + '/emissions500new.npy')[:, :, 1]
xs = np.load(folder + '/emissions500new.npy')[:, :, 0]
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


# Initialize reward from SWIRL learnt params on time_interval data
time_interval_learnt_params = jnp.load(save_folder + '/time_interval_' + str(K) + '_' + str(10) + '_NM_labyrinth1.npz', allow_pickle=True)
init_reward = time_interval_learnt_params['new_reward']

from plot_labyrinth import normalize
r1 = init_reward[0].T
r1 = normalize(r1)
r1 = np.tile(r1, (1, 4))
r2 = init_reward[1].T
r2 = normalize(r2)
r2 = np.tile(r2, (1, 4))
np.random.seed(seed)
if K == 3:
    r3 = npr.rand(127)[:, None]
    r3 = jnp.tile(r3, (1, 4))
    R_start = np.array([r1, r2, r3])
    R_start2 = R_start.mean(axis=-1)
elif K == 2:
    R_start = np.array([r1, r2])
    R_start2 = R_start.mean(axis=-1)
elif K == 4:
    r3 = npr.rand(127)[:, None]
    r3 = jnp.tile(r3, (1, 4))
    r4 = npr.rand(127)[:, None]
    r4 = jnp.tile(r4, (1, 4))
    R_start = np.array([r1, r2, r3, r4])
    R_start2 = R_start.mean(axis=-1)
elif K == 5:
    r3 = npr.rand(127)[:, None]
    r3 = jnp.tile(r3, (1, 4))
    r4 = npr.rand(127)[:, None]
    r4 = jnp.tile(r4, (1, 4))
    r5 = npr.rand(127)[:, None]
    r5 = jnp.tile(r5, (1, 4))
    R_start = np.array([r1, r2, r3, r4, r5])
    R_start2 = R_start.mean(axis=-1)

# Initialize other params from arhmm_s
arhmm_s_params = np.load(folder + '/' + str(K) + 'arhmm_s_params.npz')
logpi0_start = arhmm_s_params['arr_0']
log_Ps_start = arhmm_s_params['arr_1']
Rs_start = arhmm_s_params['arr_2'], arhmm_s_params['arr_3'], arhmm_s_params['arr_4'], arhmm_s_params['arr_5']

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


from swirl_func import jax_e_step_batch_labyrinth, jax_e_step_batch2_labyrinth, jax_e_step_batch_temp, jax_e_step_batch2_temp, pi0_m_step, trans_m_step_jax_scipy2, emit_m_step_jax_scipy2_labyrinth, emit_m_step_jax_scipy2_temp

def em_train_labyrinth(logpi0, log_Ps, Rs, rewards, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch_labyrinth(pi0, log_Ps, Rs, rewards, trans_probs, train_xohs, train_aohs)
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
            new_rewards = emit_m_step_jax_scipy2_labyrinth(rewards, trans_probs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs), jnp.array(train_aohs))
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list

def em_train2_labyrinth(logpi0, log_Ps, Rs, rewards, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch2_labyrinth(pi0, log_Ps, Rs, rewards, new_trans_probs, train_xohs, train_xohs2, train_aohs)
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
            new_rewards = emit_m_step_jax_scipy2_labyrinth(rewards, new_trans_probs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs2), jnp.array(train_aohs))
           
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list

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
            new_rewards = emit_m_step_jax_scipy2_temp(rewards, trans_probs, temps,(all_gamma_jax, all_xi_jax), jnp.array(train_xohs), jnp.array(train_aohs))
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list

def em_train2_temp(logpi0, log_Ps, Rs, rewards, temps, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jax_e_step_batch2_temp(pi0, log_Ps, Rs, rewards, new_trans_probs, temps, train_xohs, train_xohs2, train_aohs)
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
            new_rewards = emit_m_step_jax_scipy2_temp(rewards, new_trans_probs, temps, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs2), jnp.array(train_aohs))
        else:
            new_rewards = rewards

        logpi0, log_Ps, Rs, rewards = new_logpi0, new_log_Ps, new_Rs, new_rewards
    return logpi0, log_Ps, Rs, rewards, LL_list

from swirl_func import _viterbi_JAX, forward, vi_temp, comp_ll_jax, comp_transP
def comp_LLloss(pi0, trans_Ps, lls):
    alphas_list = vmap(partial(forward, jnp.array(pi0)))(trans_Ps, lls)
    return jnp.sum(jax_logsumexp(alphas_list[:, -1], axis=-1))
def learnt_LL1(logpi0, log_Ps, Rs, rewards, temps):
    n_states, n_actions, _ = trans_probs.shape
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    pi, _, _ = vmap(partial(vi_temp, trans_probs))(rewards_sa, temps)
    logemit = jnp.log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(all_xohs))
    new_lls_jax_vmap_train = vmap(partial(comp_ll_jax, logemit))(jnp.array(train_xohs), jnp.array(train_aohs))
    new_trans_Ps_vmap_train = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(train_xohs))
    new_lls_jax_vmap_test = vmap(partial(comp_ll_jax, logemit))(jnp.array(test_xohs), jnp.array(test_aohs))
    new_trans_Ps_vmap_test = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(test_xohs))
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap) / (all_xohs.shape[0]*all_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_train, new_lls_jax_vmap_train) / (train_xohs.shape[0]*train_xohs.shape[1]),  comp_LLloss(pi0, new_trans_Ps_vmap_test, new_lls_jax_vmap_test) / (test_xohs.shape[0]*test_xohs.shape[1]), jax_path_vmap
def learnt_LL2(logpi0, log_Ps, Rs, rewards, temps):
    n_states, n_actions, _ = new_trans_probs.shape
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    pi, _, _ = vmap(partial(vi_temp, new_trans_probs))(rewards_sa, temps)
    logemit = jnp.log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs2), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(all_xohs))
    new_lls_jax_vmap_train = vmap(partial(comp_ll_jax, logemit))(jnp.array(train_xohs2), jnp.array(train_aohs))
    new_trans_Ps_vmap_train = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(train_xohs))
    new_lls_jax_vmap_test = vmap(partial(comp_ll_jax, logemit))(jnp.array(test_xohs2), jnp.array(test_aohs))
    new_trans_Ps_vmap_test = vmap(partial(comp_transP, jnp.array(log_Ps), Rs))(jnp.array(test_xohs))
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap) / (all_xohs.shape[0]*all_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_train, new_lls_jax_vmap_train) / (train_xohs.shape[0]*train_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_test, new_lls_jax_vmap_test) / (test_xohs.shape[0]*test_xohs.shape[1]), jax_path_vmap


from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
temps = jnp.array([0.01] + [1] * (K - 1))

# S-2
new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = em_train2_labyrinth(jnp.array(logpi0_start), jnp.array(log_Ps_start), Rs_start, jnp.array(R_start)[:, None].reshape(K, 1, 127*4), 50)
# jnp.savez(save_folder + '/' + str(K) + '_' + str(seed) + '_NM_labyrinth2_init.npz', new_logpi0=new_logpi02, new_log_Ps=new_log_Ps2, new_Rs=np.array(new_Rs2, dtype=object), new_reward=new_reward2, LL_list=LL_list2, temps=temps)
new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = em_train2_temp(jnp.array(new_logpi02), jnp.array(new_log_Ps2), new_Rs2, jnp.array(new_reward2), temps, 30)
jnp.savez(save_folder + '/' + str(K) + '_' + str(seed) + '_NM_labyrinth2.npz', new_logpi0=new_logpi02, new_log_Ps=new_log_Ps2, new_Rs=np.array(new_Rs2, dtype=object), new_reward=new_reward2, LL_list=LL_list2, temps=temps)

# # S-1
new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_labyrinth(jnp.array(logpi0_start), jnp.array(log_Ps_start), Rs_start, jnp.array(R_start2)[:, None], 50)
# jnp.savez(save_folder + '/' + str(K) + '_' + str(seed) + '_NM_labyrinth1_init.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)
new_logpi0, new_log_Ps, new_Rs, new_reward, LL_list = em_train_temp(jnp.array(new_logpi0), jnp.array(new_log_Ps), new_Rs, jnp.array(new_reward), temps, 30)
jnp.savez(save_folder + '/' + str(K) + '_' + str(seed) + '_NM_labyrinth1.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=np.array(new_Rs, dtype=object), new_reward=new_reward, LL_list=LL_list, temps=temps)

# Load S-2 params
params2 = jnp.load(save_folder + '/' + str(K) + '_' + str(seed) + '_NM_labyrinth2.npz', allow_pickle=True)
new_logpi02, new_log_Ps2, new_Rs2, new_reward2, LL_list2 = params2['new_logpi0'], params2['new_log_Ps'], params2['new_Rs'], params2['new_reward'], params2['LL_list']

LL, train_LL, test_LL, jax_path_vmap = learnt_LL2(new_logpi02, new_log_Ps2, new_Rs2, new_reward2, temps)
print(LL, train_LL, test_LL)

from plot_labyrinth import plot_trajs, PlotMazeFunction
maze_info = np.load(folder + '/maze_info.npz', allow_pickle=True)
m_wa, m_ru, m_xc, m_yc = maze_info['m_wa'], maze_info['m_ru'], maze_info['m_xc'], maze_info['m_yc']
xy_list = np.load(folder + '/xy_list500new.npy', allow_pickle=True)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
reward2_filtered = np.copy(new_reward2[:, 0]).reshape((K, C, 4))
reward2_filtered[:, invalid_indices] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(19,6), dpi=400)
title_list = ['Water', 'Home', 'Explore']
color_list = ['blue', 'brown', 'lightblue']
color_options = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    (0.814, 0.661, 0.885, 0.9)
]
for i in range(3):
    converted_map = np.nanmean(reward2_filtered[i], -1) 
    PlotMazeFunction(converted_map, title_list[i], m_wa, m_ru, m_xc, m_yc, numcol='blue', figsize=6, selected_color=color_options[i], axes=axes[i])

norm = plt.Normalize(vmin=0, vmax=1)  # Normalize values between 0 and 1
import matplotlib.colors as mcolors

for i in range(3):
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(1, 1, 1, 1), color_options[i]])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, ticks=[0, 1])
    cbar.ax.tick_params(labelsize=12)

plt.savefig(save_folder + '/fig_all_reward_maps_labyrinth.pdf', bbox_inches='tight')


learnt_zs = np.array(jax_path_vmap)
fig, axs = plt.subplots(1, 3, figsize=(18,6), dpi=400)
axs, lines_list = plot_trajs(m_wa, learnt_zs, xy_list, axs=axs)
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(lines_list[-1], cax=cax)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Start', 'End'])
cbar.ax.tick_params(labelsize=18)

plt.savefig(save_folder + '/fig_all_trajs_labyrinth.pdf', bbox_inches='tight')