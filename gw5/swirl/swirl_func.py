import numpy as np
import numpy.random as npr
from scipy.special import logsumexp

#JAX
import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")


def vinet(trans_probs, R_params, apply_fn, discount=0.95):
    n_states, n_actions, _ = trans_probs.shape

    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        # Combine one-hot encodings
        return apply_fn({'params': R_params}, one_hot_input)  # Apply the network to get R(hidden, a)
        
    reward = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)

    @partial(jax.jit, static_argnames=['threshold'])
    def jax_soft_find_policynet(transition_probabilities, reward, threshold=100):
        n_states, n_actions = reward.shape
        v = jnp.ones(n_states)

        Q = jnp.ones((n_states, n_actions))

        def scan_iter(carry, inputs):

            v, Q = carry

            new_Q = vmap(vmap(lambda r_sa, tp_sa: r_sa + discount * jnp.dot(tp_sa, v)))(reward, transition_probabilities)
            new_v = jax.scipy.special.logsumexp(new_Q, axis=1)

            return (new_v, new_Q), 0

        (v, Q), _ = lax.scan(scan_iter, (v, Q), jnp.arange(threshold))

        policy = jnp.exp(Q)/jnp.exp(Q).sum(axis=1, keepdims=True)
        return v, Q, policy

    VG, QG, piG = vmap(partial(jax_soft_find_policynet, jnp.array(trans_probs)))(reward)
    
    return piG, QG, VG

def vinet_expand(trans_probs, R_params, apply_fn, discount=0.95):
    n_states, n_actions, _ = trans_probs.shape
    reshape_func = lambda x: (jnp.tile(jnp.expand_dims(x, axis=-1), (1,) * (x.ndim) + (n_states,)) / n_states).reshape(*x.shape[:-1], x.shape[-1] * x.shape[-1])

    def get_reward_single(curr_s):
        one_hot_input = jax.nn.one_hot(curr_s, n_states)
        # Combine one-hot encodings
        return apply_fn({'params': R_params}, reshape_func(one_hot_input))
        
    reward = vmap(get_reward_single)(jnp.arange(n_states)).transpose(1, 0, 2)

    @partial(jax.jit, static_argnames=['threshold'])
    def jax_soft_find_policynet(transition_probabilities, reward, threshold=100):
        n_states, n_actions = reward.shape
        v = jnp.ones(n_states)

        Q = jnp.ones((n_states, n_actions))

        def scan_iter(carry, inputs):

            v, Q = carry

            new_Q = vmap(vmap(lambda r_sa, tp_sa: r_sa + discount * jnp.dot(tp_sa, v)))(reward, transition_probabilities)
            new_v = jax.scipy.special.logsumexp(new_Q, axis=1)

            return (new_v, new_Q), 0

        (v, Q), _ = lax.scan(scan_iter, (v, Q), jnp.arange(threshold))

        policy = jnp.exp(Q)/jnp.exp(Q).sum(axis=1, keepdims=True)
        return v, Q, policy

    VG, QG, piG = vmap(partial(jax_soft_find_policynet, jnp.array(trans_probs)))(reward)
    
    return piG, QG, VG

def comp_ll_jax(logits, one_hot_x, one_hot_a):
    logits = logits - jax_logsumexp(logits, axis=-1, keepdims=True)
    transition_probs = jnp.einsum('...i,...ij->...j', one_hot_x, logits)
    lls = jnp.sum(one_hot_a * transition_probs, axis=-1)
    return lls

def comp_log_transP(log_Ps, Rs, one_hot_x):
    T = one_hot_x.shape[0]
    log_Ps = jnp.tile(log_Ps[None, :, :], (T-1, 1, 1))
    log_Ps = log_Ps + jnp.dot(one_hot_x[:-1, 0, :], Rs[:, 0, :])[:, None, :]
    return log_Ps - jax_logsumexp(log_Ps, axis=2, keepdims=True)
    
def comp_transP(log_Ps, Rs, one_hot_x):
    return jnp.exp(comp_log_transP(log_Ps, Rs, one_hot_x))
    
def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K))
    zoh[np.arange(N), np.arange(K)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K,))
    return zoh

def one_hot2(z, z_prev, K):
    z = z * K + z_prev
    z = np.atleast_1d(z).astype(int)
    K2 = K * K
    assert np.all(z >= 0) and np.all(z < K2)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K2))
    zoh[np.arange(N), np.arange(K2)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K2,))
    return zoh

def comp_ll_transP(xs, acs, logemit_learnt, model):
    lls = []
    trans_Ps = []
    for (x, ac) in zip(xs, acs):
        variational_mean = x.astype(int)[:, np.newaxis]
        n_ac = ac.astype(int)[:, np.newaxis]
        trans = model.transitions.transition_matrices(variational_mean, None, None, None)
        log_likes = model.dynamics.log_likelihoods(variational_mean, n_ac, None, np.ones_like(variational_mean, dtype=bool), None)
        lls.append(log_likes)
        trans_Ps.append(trans)
    return np.array(lls), np.array(trans_Ps)

def _viterbi_JAX(pi0, Ps, ll):
    T, K = ll.shape

    # Check if the transition matrices are stationary or time-varying (hetero)
    hetero = (Ps.shape[0] == T-1)

    def score_fn(carry, inputs):
        score_next_t = carry
        t = inputs

        vals = jnp.log(Ps[t * hetero]) + score_next_t + ll[t+1]
        
        def comp_arg_score_over_K(vals):
            return jnp.argmax(vals), jnp.max(vals)
        
        arg_next_t, score_t = vmap(comp_arg_score_over_K)(vals)

        return score_t[None, :], (score_t, arg_next_t)
    
    _, (scores, args) = lax.scan(score_fn, jnp.zeros((1, K)), jnp.arange(T-2,-1,-1))

    z0 = jnp.array([(scores[-1] + jnp.log(pi0) + ll[0]).argmax()])
    
    def scan_z(carry, inputs):
        z_prev_t = carry
        arg_next_t = inputs

        z_t = arg_next_t[z_prev_t.astype(int)]

        return z_t, z_t

    _, z = lax.scan(scan_z, z0, args[::-1])

    return jnp.concatenate([z0, z.squeeze()]).astype(int)

def forward(pi0, Ps, log_likes):
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    alpha0 = jnp.log(pi0) + log_likes[0]
    def scan_body(carry, inputs):
        alpha_prev = carry
        Ps_t, log_like_t = inputs
        m = jnp.max(alpha_prev)
        alpha_t = jnp.log(jnp.dot(jnp.exp(alpha_prev - m), Ps_t)) + m + log_like_t
        return alpha_t, alpha_t
    _, alphas = lax.scan(scan_body, alpha0, (Ps, log_likes[1:]))
    return jnp.concatenate([alpha0[None, :], alphas])

def backward(Ps, log_likes):
    T = log_likes.shape[0]
    K = log_likes.shape[1]
    betaT = jnp.zeros((K))

    def scan_body(carry, inputs):
        beta_next = carry
        Ps_t, log_like_next = inputs
        tmp = log_like_next + beta_next
        m = jnp.max(tmp)
        beta_t = jnp.log(jnp.dot(Ps_t, jnp.exp(tmp - m))) + m

        return beta_t, beta_t
        
    _, betas = lax.scan(scan_body, betaT, (Ps[::-1], log_likes[1:][::-1]))

    return jnp.concatenate([betas[::-1], betaT[None, :]])

def expected_states(alphas, betas, Ps, ll):
    T, K = ll.shape

    expected_states = alphas + betas
    expected_states -= jax_logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = jnp.exp(expected_states)

    log_Ps = jnp.log(Ps)

    expected_joints = alphas[:-1,:,None] + betas[1:,None,:] + ll[1:,None,:] + log_Ps
    expected_joints -= expected_joints.max((1,2))[:,None, None]
    expected_joints = jnp.exp(expected_joints)
    expected_joints /= expected_joints.sum((1,2))[:,None,None]

    return expected_states, expected_joints

def jaxnet_e_step_logpi(pi0, log_Ps, Rs, logemit, trans_probs, xoh, aoh):
    Ps_jax = comp_transP(jnp.array(log_Ps), jnp.array(Rs), jnp.array(xoh))
    log_likes_jax = comp_ll_jax(jnp.array(logemit), jnp.array(xoh), jnp.array(aoh))
    alpha_jax = forward(pi0, Ps_jax, log_likes_jax)
    beta_jax = backward(Ps_jax, log_likes_jax)
    gamma_jax, xi_jax = expected_states(alpha_jax, beta_jax, Ps_jax, log_likes_jax)
    return gamma_jax, xi_jax, alpha_jax

def jaxnet_e_step_batch(pi0, log_Ps, Rs, R_state, trans_probs, xoh_list, aoh_list):
    pi, _, _ = vinet_expand(trans_probs, R_state.params, R_state.apply_fn)
    pi = jax.lax.stop_gradient(pi)
    logemit = jnp.log(pi)
    gamma_jax_list, xi_jax_list, alpha_jax_list = vmap(partial(jaxnet_e_step_logpi, jnp.array(pi0), jnp.array(log_Ps), jnp.array(Rs), jnp.array(logemit), jnp.array(trans_probs)))(jnp.array(xoh_list), jnp.array(aoh_list))
    
    return gamma_jax_list, xi_jax_list, alpha_jax_list

def jaxnet_e_step_logpi2(pi0, log_Ps, Rs, logemit, trans_probs, xoh, xoh2, aoh):
    Ps_jax = comp_transP(jnp.array(log_Ps), jnp.array(Rs), jnp.array(xoh))
    log_likes_jax = comp_ll_jax(jnp.array(logemit), jnp.array(xoh2), jnp.array(aoh))
    alpha_jax = forward(pi0, Ps_jax, log_likes_jax)
    beta_jax = backward(Ps_jax, log_likes_jax)
    gamma_jax, xi_jax = expected_states(alpha_jax, beta_jax, Ps_jax, log_likes_jax)
    return gamma_jax, xi_jax, alpha_jax

def jaxnet_e_step_batch2(pi0, log_Ps, Rs, R_state, trans_probs, xoh_list, xoh_list2, aoh_list):
    pi, _, _ = vinet(trans_probs, R_state.params, R_state.apply_fn)
    pi = jax.lax.stop_gradient(pi)
    logemit = jnp.log(pi)
    gamma_jax_list, xi_jax_list, alpha_jax_list = vmap(partial(jaxnet_e_step_logpi2, jnp.array(pi0), jnp.array(log_Ps), jnp.array(Rs), jnp.array(logemit), jnp.array(trans_probs)))(jnp.array(xoh_list), jnp.array(xoh_list2), jnp.array(aoh_list))
    
    return gamma_jax_list, xi_jax_list, alpha_jax_list

def pi0_m_step_numpy(expectations):
    pi0 = sum([Ez[0] for Ez, _ in expectations]) + 1e-8
    return np.log(pi0 / pi0.sum())


from jaxopt import BFGS, LBFGS

def trans_m_step_jax_jaxopt(log_Ps, Rs, expectations, one_hot_xs, num_iters=1000, **kwargs):
    
    # Maximize the expected log joint
    def _expected_log_joint(params, expectations):
        log_Ps, Rs = params
        def scan_func(carry, inputs):
            elbo = carry
            one_hot_x, expected_states, expected_joints = inputs
            log_trans = comp_log_transP(log_Ps, Rs, one_hot_x)
            return elbo + jnp.sum(expected_joints * log_trans), 0

        elbo, _ = lax.scan(scan_func, 0, (one_hot_xs, expectations[0], expectations[1]))
        return elbo

    T = sum([one_hot_x.shape[0] for one_hot_x in one_hot_xs])
    # Define the objective function for BFGS
    def objective(params):
        elbo = _expected_log_joint(params, expectations)
        return -elbo / T

    # Initial parameter values
    initial_params = (log_Ps, Rs)
    
    # Set up BFGS optimizer from jaxopt
    bfgs = LBFGS(fun=objective, maxiter=num_iters, tol=1e-4, stop_if_linesearch_fails=True)
    # opt = optax.adam(5e-3)
    # adam = OptaxSolver(opt=opt, fun=objective, maxiter=1000)

    # Run the optimization
    optimal_params = bfgs.run(init_params=initial_params)
    # optimal_params = adam.run(init_params=initial_params)

    return optimal_params.params

def trans_m_step_jax_optax(log_Ps, Rs, expectations, one_hot_xs, num_iters=1000, learning_rate=5e-3, **kwargs):

    # Maximize the expected log joint
    def _expected_log_joint(params, expectations):
        log_Ps, Rs = params
        def scan_func(carry, inputs):
            elbo = carry
            one_hot_x, expected_states, expected_joints = inputs
            log_trans = comp_log_transP(log_Ps, Rs, one_hot_x)
            return elbo + jnp.sum(expected_joints * log_trans), 0

        elbo, _ = lax.scan(scan_func, 0, (one_hot_xs, expectations[0], expectations[1]))
        return elbo

    T = sum([one_hot_x.shape[0] for one_hot_x in one_hot_xs])
    # Define the objective function for Adam
    def objective(params):
        elbo = _expected_log_joint(params, expectations)
        return -elbo / T

    # Initial parameter values
    initial_params = (log_Ps, Rs)

    # Set up Adam optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Optimization loop
    params = initial_params
    for _ in range(num_iters):
        params, opt_state, loss = step(params, opt_state)

    return params

import optax

def emit_m_step_jaxnet_optax2(R_state, trans_probs, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    
    def _expected_log_joint(params, apply_fn, expectations):
        pi, _, _ = vinet(trans_probs, params, apply_fn)
        logemit = jnp.log(pi)

        def single_elbo(one_hot_x, one_hot_a, expected_states):

            lls = comp_ll_jax(logemit, one_hot_x, one_hot_a) 

            return jnp.sum(expected_states * lls)

        elbo = jax.vmap(single_elbo)(one_hot_xs, one_hot_acs, expectations)
        
        return jnp.sum(elbo)
        
    T = sum([one_hot_x.shape[0] for one_hot_x in one_hot_xs])

    def loss_fn(params):
        elbo = _expected_log_joint(params, R_state.apply_fn, expectations)
        return -elbo / T

    @jax.jit
    def step(R_state):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(R_state.params)
        new_R_state = R_state.apply_gradients(grads=grads)
        return new_R_state

    for iter_time in range(num_iters):
        # print('iter ', iter_time)
        R_state = step(R_state)

    return R_state

def emit_m_step_jaxnet_optax2_expand(R_state, trans_probs, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    
    def _expected_log_joint(params, apply_fn, expectations):
        pi, _, _ = vinet_expand(trans_probs, params, apply_fn)
        logemit = jnp.log(pi)
        def single_elbo(one_hot_x, one_hot_a, expected_states):
            lls = comp_ll_jax(logemit, one_hot_x, one_hot_a) 
            return jnp.sum(expected_states * lls)

        elbo = jax.vmap(single_elbo)(one_hot_xs, one_hot_acs, expectations)
        
        return jnp.sum(elbo)
        
    T = sum([one_hot_x.shape[0] for one_hot_x in one_hot_xs])

    def loss_fn(params):
        elbo = _expected_log_joint(params, R_state.apply_fn, expectations)
        return -elbo / T

    @jax.jit
    def step(R_state):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(R_state.params)
        new_R_state = R_state.apply_gradients(grads=grads)
        return new_R_state

    for iter_time in range(num_iters):
        # print('iter ', iter_time)
        R_state = step(R_state)

    return R_state

def pi0_m_step(gammas):
    pi0 = sum(gammas[:, 0]) + 1e-8
    return jnp.log(pi0 / pi0.sum())

