import numpy as np
import numpy.random as npr
from scipy.special import logsumexp

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import sys
if len(sys.argv) < 2:
    seed = 12345
else:
    seed = int(sys.argv[1])

K = 2
D = 1
C = 25
folder = '../data'
save_folder = '../results'

trans_probs = np.load(folder + '/trans_probs.npy')

zs = np.load(folder + '/zs.npy')[:200]
xs = np.load(folder + '/xs.npy')[:200]
acs = np.load(folder + '/acs.npy')[:200]


test_indices = np.arange(0, xs.shape[0], 5).astype(int)
train_indices = np.setdiff1d(np.arange(xs.shape[0]), test_indices).astype(int)

test_xs, test_acs = xs[test_indices].astype(int), acs[test_indices].astype(int)
train_xs, train_acs = xs[train_indices].astype(int), acs[train_indices].astype(int)
test_zs, train_zs = zs[test_indices].astype(int), zs[train_indices].astype(int)



n_states, n_actions, _ = trans_probs.shape
def one_hot_jax(z, K):
    z = jnp.atleast_1d(z).astype(int)
    shp = z.shape
    N = z.size
    zoh = jnp.zeros((N, K))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    zoh = jnp.reshape(zoh, shp + (K,))
    return zoh

def one_hot_jax2(z, z_prev, K):
    z = z * K + z_prev
    z = jnp.atleast_1d(z).astype(int)
    K2 = K * K
    shp = z.shape
    N = z.size
    zoh = jnp.zeros((N, K2))
    zoh = zoh.at[jnp.arange(N), jnp.ravel(z)].set(1)
    zoh = jnp.reshape(zoh, shp + (K2,))
    return zoh

def one_hotx_partial(xs):
    return one_hot_jax(xs[:, None], n_states)
def one_hotx2_partial(xs, xs_prev):
    return one_hot_jax2(xs[:, None], xs_prev[:, None], n_states)
def one_hota_partial(acs):
    return one_hot_jax(acs[:, None], n_actions)

train_xohs = vmap(one_hotx_partial)(train_xs)
train_xohs2 = vmap(one_hotx2_partial)(train_xs, jnp.roll(train_xs, 1))
train_aohs = vmap(one_hota_partial)(train_acs)

all_xohs = vmap(one_hotx_partial)(xs)
all_xohs2 = vmap(one_hotx2_partial)(xs, jnp.roll(xs, 1))
all_aohs = vmap(one_hota_partial)(acs)

test_xohs = vmap(one_hotx_partial)(test_xs)
test_xohs2 = vmap(one_hotx2_partial)(test_xs, jnp.roll(test_xs, 1))
test_aohs = vmap(one_hota_partial)(test_acs)

npr.seed(seed)
logpi0_start = np.array([0.5, 0.5])
Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
Ps /= Ps.sum(axis=1, keepdims=True)
log_Ps_start = np.log(Ps)
Rs_start = np.zeros((C, 1, K))


from swirl_func import pi0_m_step, trans_m_step_jax_jaxopt
n_states, n_actions, _ = trans_probs.shape


import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

class MLP(nn.Module):
    subnet_size: int
    hidden_size: int
    output_size: int
    n_hidden: int
    expand: bool

    def setup(self):
        self.subnet = nn.Dense(self.subnet_size)
        self.dense1 = nn.Dense(self.hidden_size)
        # self.dense1b = nn.Dense(5)
        self.dense2 = nn.Dense(self.n_hidden)
        if self.expand:
            self.reshape_func = lambda x: jnp.tile(jnp.expand_dims(x, axis=-1), (1,) * (x.ndim) + (C,)) / C
        else:
            self.reshape_func = lambda x: x.reshape(*x.shape[:-1], C, C)

    def __call__(self, x):
        x = self.reshape_func(x)
        x = jax.vmap(self.subnet, in_axes=-1)(x)
        x = x.reshape(*x.shape[1:-1], x.shape[0] * x.shape[-1])
        x = self.dense1(x)
        x = nn.leaky_relu(x)
        x = self.dense2(x) 
        x = nn.leaky_relu(x)
        x = jnp.expand_dims(x, axis=-1)
        x = jnp.tile(x, (1, self.output_size))
        return x


# Create the model
def create_model(rng, subnet_size, n_hidden, input_size, hidden_size, output_size, expand):
    model = MLP(subnet_size=subnet_size, hidden_size=hidden_size, output_size=output_size, n_hidden=n_hidden, expand=expand)
    params = model.init(rng, jnp.ones((1, input_size)))['params']  # Initialize model params with an example input
    return model, params

# Training state to hold the model parameters and optimizer state
def create_train_state(rng, subnet_size, learning_rate, n_hidden, input_size, hidden_size, output_size, expand=False):
    model, params = create_model(rng, subnet_size, n_hidden, input_size, hidden_size, output_size, expand)
    tx = optax.adam(learning_rate)  # Adam optimizer
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


rng = jax.random.PRNGKey(0)
input_size = C*C
subnet_size = 4
hidden_size = 16
output_size = 5
n_hidden = K
learning_rate = 5e-3

# Initialize the model and training state
R_state = create_train_state(rng, subnet_size, learning_rate, n_hidden, input_size, hidden_size, output_size, expand=False)

R_state2 = create_train_state(rng, subnet_size, learning_rate, n_hidden, input_size, hidden_size, output_size, expand=False)

n_state, n_action, _ = trans_probs.shape
new_trans_probs = np.zeros((n_state * n_state, n_action, n_state * n_state))

# new transition matrix
for s_prev in range(n_state):
    for s in range(n_state):
        for a in range(n_action):
            for s_prime in range(n_state):
                if trans_probs[s, a, s_prime] > 0:
                    new_trans_probs[s * n_state + s_prev, a, s_prime * n_state + s] = trans_probs[s, a, s_prime]

from swirl_func import comp_transP, forward, backward, expected_states, comp_ll_jax, vinet, vinet_expand
from swirl_func import emit_m_step_jaxnet_optax2, emit_m_step_jaxnet_optax2_expand, jaxnet_e_step_batch2, jaxnet_e_step_batch

def em_train_net2(logpi0, log_Ps, Rs, R_state, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jaxnet_e_step_batch2(pi0, log_Ps, Rs, R_state, new_trans_probs, train_xohs, train_xohs2, train_aohs)
        print(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        print(new_logpi0)

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_jaxopt(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_R_state = emit_m_step_jaxnet_optax2_expand(R_state, jnp.array(trans_probs), all_gamma_jax, jnp.array(train_xohs), jnp.array(train_aohs), num_iters=800)
            new_R_state = emit_m_step_jaxnet_optax2(new_R_state, jnp.array(new_trans_probs), all_gamma_jax, jnp.array(train_xohs2), jnp.array(train_aohs), num_iters=200)
        else:
            new_R_state = R_state
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        logpi0, log_Ps, Rs, R_state = new_logpi0, new_log_Ps, new_Rs, new_R_state
    return logpi0, log_Ps, Rs, R_state, LL_list

def em_train_net(logpi0, log_Ps, Rs, R_state, iter=100, init=True, trans=True, emit=True):
    LL_list = []
    for i in range(iter):
        print(i)
        pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
        all_gamma_jax, all_xi_jax, all_jax_alphas = jaxnet_e_step_batch(pi0, log_Ps, Rs, R_state, trans_probs, train_xohs, train_aohs)
        print(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))

        if init == True:
            new_logpi0 = pi0_m_step(all_gamma_jax)
        else:
            new_logpi0 = logpi0
        print(new_logpi0)

        if trans == True:
            new_log_Ps, new_Rs = trans_m_step_jax_jaxopt(log_Ps, Rs, (all_gamma_jax, all_xi_jax), jnp.array(train_xohs))
        else:
            new_log_Ps, new_Rs = log_Ps, Rs

        if emit == True:
            new_R_state = emit_m_step_jaxnet_optax2_expand(R_state, jnp.array(trans_probs), all_gamma_jax, jnp.array(train_xohs), jnp.array(train_aohs), num_iters=800)
        else:
            new_R_state = R_state
        LL_list.append(jnp.sum(jax_logsumexp(all_jax_alphas[:, -1], axis=-1)))
        logpi0, log_Ps, Rs, R_state = new_logpi0, new_log_Ps, new_Rs, new_R_state
    return logpi0, log_Ps, Rs, R_state, LL_list

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)




new_logpi0, new_log_Ps, new_Rs, new_R_state, LL_list = em_train_net2(jnp.array(logpi0_start), jnp.array(log_Ps_start), jnp.array(Rs_start), R_state, 50)
jnp.savez(save_folder + '/' + str(seed) + '_NM_gw5_net2.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=new_Rs, new_R_state=new_R_state.params, LL_list=LL_list)


from swirl_func import _viterbi_JAX
def comp_LLloss(pi0, trans_Ps, lls):
    alphas_list = vmap(partial(forward, jnp.array(pi0)))(trans_Ps, lls)
    print(alphas_list.shape)
    return jnp.sum(jax_logsumexp(alphas_list[:, -1], axis=-1))
def learnt_LL1(logpi0, log_Ps, Rs, params, apply_fn):
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    pi, _, _ = vinet_expand(trans_probs, params, apply_fn)
    logemit = jnp.log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(all_xohs))
    new_lls_jax_vmap_test = vmap(partial(comp_ll_jax, logemit))(jnp.array(test_xohs), jnp.array(test_aohs))
    new_trans_Ps_vmap_test = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(test_xohs))
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    jax_path_vmap_test = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap_test), jnp.array(new_lls_jax_vmap_test))
    return comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap) / (all_xohs.shape[0]*all_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_test, new_lls_jax_vmap_test) / (test_xohs.shape[0]*test_xohs.shape[1]), jax_path_vmap, jax_path_vmap_test
def learnt_LL2(logpi0, log_Ps, Rs, params, apply_fn):
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    pi, _, _ = vinet(new_trans_probs, params, apply_fn)
    logemit = jnp.log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs2), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(all_xohs))
    new_lls_jax_vmap_test = vmap(partial(comp_ll_jax, logemit))(jnp.array(test_xohs2), jnp.array(test_aohs))
    new_trans_Ps_vmap_test = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(test_xohs))
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    jax_path_vmap_test = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap_test), jnp.array(new_lls_jax_vmap_test))
    return comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap) / (all_xohs.shape[0]*all_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_test, new_lls_jax_vmap_test) / (test_xohs.shape[0]*test_xohs.shape[1]), jax_path_vmap, jax_path_vmap_test

from gw5_analysis import get_reward_nm, get_reward_m, compute_accuracy, calibrate_reward, best_perm_corr
reward_nm2 = get_reward_nm(new_trans_probs, new_R_state.params, new_R_state.apply_fn)
ll2, tll2, learnt_zs2, learnt_zs2_test = learnt_LL2(new_logpi0, new_log_Ps, new_Rs, new_R_state.params, new_R_state.apply_fn)
acc2 = compute_accuracy(learnt_zs2, zs)
test_acc2 = compute_accuracy(learnt_zs2_test, test_zs)

RG = np.load(folder + '/RG.npy')
invalid_transitions = np.all(trans_probs == 0, axis=1)
RG_filtered = np.copy(RG)
RG_filtered[:, invalid_transitions] = np.nan
reward_nm2_filtered = np.copy(reward_nm2).mean(-1).reshape((K, C, C))
reward_nm2_filtered[:, invalid_transitions] = np.nan
reward_nm2_filtered = calibrate_reward(reward_nm2_filtered, RG_filtered)
best_corr2, reward_nm2_filtered = best_perm_corr(RG_filtered, reward_nm2_filtered)

print('S2 acc:', acc2, 'S2 test acc:', test_acc2, 'S2 R corr:', best_corr2)


new_logpi0, new_log_Ps, new_Rs, new_R_state, LL_list = em_train_net(jnp.array(logpi0_start), jnp.array(log_Ps_start), jnp.array(Rs_start), R_state2, 50)
jnp.savez(save_folder + '/' + str(seed) + '_NM_gw5_net1.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=new_Rs, new_R_state=new_R_state.params, LL_list=LL_list)

reward_m1 = get_reward_m(trans_probs, new_R_state.params, new_R_state.apply_fn)
ll1, tll1, learnt_zs1, learnt_zs1_test = learnt_LL1(new_logpi0, new_log_Ps, new_Rs, new_R_state.params, new_R_state.apply_fn)
acc1 = compute_accuracy(learnt_zs1, zs)
test_acc1 = compute_accuracy(learnt_zs1_test, test_zs)
reward_m1_filtered = np.copy(reward_m1).mean(-1).reshape((K, C))[:, :, None]
reward_m1_filtered = np.tile(reward_m1_filtered, (1, 1, C))
reward_m1_filtered = calibrate_reward(reward_m1_filtered, RG_filtered)
best_corr1, reward_m1_filtered = best_perm_corr(RG_filtered, reward_m1_filtered)

print('S1 acc:', acc1, 'S1 test acc:', test_acc1, 'S1 R corr:', best_corr1)





