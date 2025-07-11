import numpy as np
import numpy.random as npr

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

import sys
if len(sys.argv) < 3:
    K = 5
    seed = 30
else:
    K = int(sys.argv[1])
    seed = int(sys.argv[2])


D_obs = 1
D_latent = 1
C = 9
folder = '../data/'
save_folder = '../results/'

seqs = np.load(folder + '/seqs.npy')
for i in range(seqs.shape[0]):
    for j in range(seqs.shape[1]):
        if seqs[i, j] == 4:
            seqs[i, j] = 0
        elif seqs[i, j] > 4:
            seqs[i, j] = seqs[i, j] - 1
z_features = np.load(folder + '/z_features.npy')

trajs = []
raw_DAs = []
for i in range(seqs.shape[0]):
    traj = []
    raw_DA = []

    for j in range(seqs.shape[1]-1):
        if seqs[i, j] < C and seqs[i, j+1] < C:
            traj.append([seqs[i, j], seqs[i, j+1], 1, seqs[i, j+1]])
            raw_DA.append(z_features[i, j+1])
            
    if len(traj) > 300:
        trajs.append(traj[:300])
        raw_DAs.append(raw_DA[:300])

trajs = np.array(trajs)
raw_DAs = np.array(raw_DAs)


xs = trajs[:, :, 0]
acs = trajs[:, :, 1]
trans_probs = np.load(folder + '/trans_probs.npy')[:9, :9, :9]

test_indices = np.arange(0, xs.shape[0], 5)
train_indices = np.setdiff1d(np.arange(xs.shape[0]), test_indices)

test_xs, test_acs = xs[test_indices], acs[test_indices]
train_xs, train_acs = xs[train_indices], acs[train_indices]


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
    return zoh, z

def one_hotx_partial(xs):
    return one_hot_jax(xs[:, None], n_states)
def one_hotx2_partial(xs, xs_prev):
    return one_hot_jax2(xs[:, None], xs_prev[:, None], n_states)
def one_hota_partial(acs):
    return one_hot_jax(acs[:, None], n_actions)

train_xohs = vmap(one_hotx_partial)(train_xs)
train_xohs2, train_xs2 = vmap(one_hotx2_partial)(train_xs, jnp.roll(train_xs, 1))
train_aohs = vmap(one_hota_partial)(train_acs)

all_xohs = vmap(one_hotx_partial)(xs)
all_xohs2, all_xs2 = vmap(one_hotx2_partial)(xs, jnp.roll(xs, 1))
all_aohs = vmap(one_hota_partial)(acs)

test_xohs = vmap(one_hotx_partial)(test_xs)
test_xohs2, test_xs2 = vmap(one_hotx2_partial)(test_xs, jnp.roll(test_xs, 1))
test_aohs = vmap(one_hota_partial)(test_acs)

# from ssm.swirl import ARHMMs
# npr.seed(seed)
# arhmm_s = ARHMMs(D_obs, K, D_latent, C,
#              transitions="recurrent",
#              dynamics="arcategorical",
#              single_subspace=True)
# list_x = [row for row in train_xs[:, :, np.newaxis].astype(int)]
# lls_arhmm = arhmm_s.initialize(list_x, num_init_iters=100)
# init_start = arhmm_s.init_state_distn.initial_state_distn
# logpi0_start = arhmm_s.init_state_distn.log_pi0
# log_Ps_start = arhmm_s.transitions.log_Ps
# Rs_start = arhmm_s.transitions.Rs
# np.savez(folder + str(K) + '_' + str(seed) + '_arhmm_s.npz', init_start=init_start, logpi0_start=logpi0_start, log_Ps_start=log_Ps_start, Rs_start=Rs_start)

arhmm_params = np.load(folder + str(K) + '_' + str(seed) + '_arhmm_s.npz', allow_pickle=True)
init_start = arhmm_params['init_start']
logpi0_start = arhmm_params['logpi0_start']
log_Ps_start = arhmm_params['log_Ps_start']
Rs_start = arhmm_params['Rs_start']



from swirl_func import pi0_m_step, trans_m_step_jax_jaxopt, emit_m_step_jaxnet_optax2, emit_m_step_jaxnet_optax2_expand, jaxnet_e_step_batch2, jaxnet_e_step_batch
n_states, n_actions, _ = trans_probs.shape


from flax import linen as nn
from flax.training import train_state

class MLP(nn.Module):
    subnet_size: int
    hidden_size: int
    output_size: int
    n_hidden: int
    expand: bool

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.n_hidden*C)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.leaky_relu(x)
        x = self.dense2(x) 
        X = nn.tanh(x)
        x = x.reshape((K, C))
        return x

# Create the model
def create_model(rng, subnet_size, n_hidden, input_size, hidden_size, output_size, expand):
    model = MLP(subnet_size=subnet_size, hidden_size=hidden_size, output_size=output_size, n_hidden=n_hidden, expand=expand)
    params = model.init(rng, jnp.ones((1, input_size)))['params']
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
output_size = C
n_hidden = K
learning_rate = 5e-3

# Initialize the model and training state
R_state = create_train_state(rng, subnet_size, learning_rate, n_hidden, input_size, hidden_size, output_size, expand=False)

R_state2 = create_train_state(rng, subnet_size, learning_rate, n_hidden, input_size, hidden_size, output_size, expand=False)

n_state, n_action, _ = trans_probs.shape
new_trans_probs = np.zeros((n_state * n_state, n_action, n_state * n_state))
for s_prev in range(n_state):
    for s in range(n_state):
        for a in range(n_action):
            for s_prime in range(n_state):
                if trans_probs[s, a, s_prime] > 0:
                    new_trans_probs[s * n_state + s_prev, a, s_prime * n_state + s] = trans_probs[s, a, s_prime]



def em_train_jaxopt_netadam2(train_xohs2, logpi0, log_Ps, Rs, R_state, iter=100, init=True, trans=True, emit=True):
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

def em_train_jaxopt_netadam(logpi0, log_Ps, Rs, R_state, iter=100, init=True, trans=True, emit=True):
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

# S-2
new_logpi02, new_log_Ps2, new_Rs2, new_R_state2, LL_list2 = em_train_jaxopt_netadam2(train_xohs2, jnp.array(logpi0_start), jnp.array(log_Ps_start), jnp.array(Rs_start), R_state, 50, init=False, trans=False)
# jnp.savez(folder + '/DAar1_rand/' + str(K) + '_' + str(seed) + '_NM_DAsa_net2_Ronly.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=new_Rs, new_R_state=new_R_state.params, LL_list=LL_list)
new_logpi02, new_log_Ps2, new_Rs2, new_R_state2, LL_list2 = em_train_jaxopt_netadam2(train_xohs2, jnp.array(new_logpi02), jnp.array(new_log_Ps2), jnp.array(new_Rs2), new_R_state2, 50)
jnp.savez(save_folder + str(K) + '_' + str(seed) + '_NM_DAsa_net2.npz', new_logpi0=new_logpi02, new_log_Ps=new_log_Ps2, new_Rs=new_Rs2, new_R_state=new_R_state2.params, LL_list=LL_list2)

# S-1
new_logpi0, new_log_Ps, new_Rs, new_R_state, LL_list = em_train_jaxopt_netadam(jnp.array(logpi0_start), jnp.array(log_Ps_start), jnp.array(Rs_start), R_state2, 50, init=False, trans=False)
# jnp.savez(folder + '/DAar1_rand/' + str(K) + '_' + str(seed) + '_NM_DAsa_net1_Ronly.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=new_Rs, new_R_state=new_R_state.params, LL_list=LL_list)
new_logpi0, new_log_Ps, new_Rs, new_R_state, LL_list = em_train_jaxopt_netadam(jnp.array(new_logpi0), jnp.array(new_log_Ps), jnp.array(new_Rs), new_R_state, 50)
jnp.savez(save_folder + str(K) + '_' + str(seed) + '_NM_DAsa_net1.npz', new_logpi0=new_logpi0, new_log_Ps=new_log_Ps, new_Rs=new_Rs, new_R_state=new_R_state.params, LL_list=LL_list)

# Load S-1
learnt_params = np.load(save_folder + str(K) + '_' + str(seed) + '_NM_DAsa_net1.npz', allow_pickle=True)
new_logpi0, new_log_Ps, new_Rs, new_R_state = learnt_params['new_logpi0'], learnt_params['new_log_Ps'], learnt_params['new_Rs'], learnt_params['new_R_state']
from swirl_func import vinet_expand, comp_ll_jax, comp_transP, _viterbi_JAX, forward
def comp_LLloss(pi0, trans_Ps, lls):
    alphas_list = vmap(partial(forward, jnp.array(pi0)))(trans_Ps, lls)
    return jnp.sum(jax_logsumexp(alphas_list[:, -1], axis=-1))
def learnt_LL21(logpi0, log_Ps, Rs, params, apply_fn):
    pi0 = jnp.exp(logpi0 - jax_logsumexp(logpi0))
    pi, _, _ = vinet_expand(trans_probs, params, apply_fn)
    logemit = jnp.log(pi)
    new_lls_jax_vmap = vmap(partial(comp_ll_jax, logemit))(jnp.array(all_xohs), jnp.array(all_aohs))
    new_trans_Ps_vmap = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(all_xohs))
    new_lls_jax_vmap_test = vmap(partial(comp_ll_jax, logemit))(jnp.array(test_xohs), jnp.array(test_aohs))
    new_trans_Ps_vmap_test = vmap(partial(comp_transP, jnp.array(log_Ps), jnp.array(Rs)))(jnp.array(test_xohs))
    jax_path_vmap = vmap(partial(_viterbi_JAX, jnp.array(pi0)))(jnp.array(new_trans_Ps_vmap), jnp.array(new_lls_jax_vmap))
    return comp_LLloss(pi0, new_trans_Ps_vmap, new_lls_jax_vmap) / (all_xohs.shape[0]*all_xohs.shape[1]), comp_LLloss(pi0, new_trans_Ps_vmap_test, new_lls_jax_vmap_test) / (test_xohs.shape[0]*test_xohs.shape[1]), jax_path_vmap

from da_analysis import get_reward_m, get_DA_Rsa, get_corr, zscore

reward_m1 = get_reward_m(trans_probs, new_R_state.item(), R_state.apply_fn)
reward_m1_filtered = np.copy(reward_m1).reshape((K, C, C))
ll1, tll1, learnt_zs1 = learnt_LL21(new_logpi0, new_log_Ps, new_Rs, new_R_state.item(), R_state.apply_fn)
DA_Rsa1 = get_DA_Rsa(learnt_zs1, xs, acs, raw_DAs, K, C)
z_DA_Rsa1 = zscore(DA_Rsa1, axis=2)
z_corr1, z_DA_Rsa1, z_learnt_Rsa1 = get_corr(np.copy(reward_m1_filtered), np.copy(z_DA_Rsa1), C)
learnt_R = np.copy(reward_m1_filtered)
DA_R_corr = np.copy(z_corr1)

import matplotlib.pyplot as plt

x = ['Pause', 'Walk', '↑', '←', 'Groom', '→', 'Sniff', 'Sniff↑', 'Run']
n_hidden = K
fig, axes = plt.subplots(1, n_hidden, figsize=(4 * n_hidden, 4), dpi=400)

def normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

for j in range(n_hidden):
    ax = axes[j]
    plot_data = learnt_R[j] + np.eye(C) / (np.eye(C) - 1) - np.eye(C) / (np.eye(C) - 1)
    im = ax.imshow(normalize(plot_data), cmap='viridis')
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, rotation=0, fontsize=6)
    ax.set_ylabel('state')
    ax.set_xlabel('action')
    ax.set_yticks(np.arange(len(x)))
    ax.set_yticklabels(x, fontsize=6)
    ax.set_title(f'hidden {j + 1} \n DA corr {DA_R_corr[j]}')
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax)

plt.suptitle(f'Heatmaps of R and DA for n_hidden = {n_hidden}')
plt.tight_layout()
plt.savefig(save_folder + '/' + str(seed) + '_fig_Rsa_DA.pdf', bbox_inches='tight', dpi=400)


plt.figure(dpi=400)
im = plt.imshow(learnt_zs1 + 1, aspect='auto', cmap="inferno", vmin=0, vmax=n_hidden-1+1)
# Add a legend for the hidden states
hidden_state_colors = plt.cm.magma(np.linspace(0, 1, n_hidden+1))
for idx in range(n_hidden):
    plt.plot([], [], color=hidden_state_colors[idx+1], label=f'h{idx+1}')
plt.grid(False)
# plt.legend(loc='upper right', fontsize=6)
plt.savefig(save_folder + '/' + str(seed) + '_fig_segments_DA.svg', bbox_inches='tight')




  