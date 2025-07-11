import numpy as np
from scipy.optimize import minimize

#JAX
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax import grad as jax_grad
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")


@partial(jax.jit, static_argnames=['threshold'])
def jax_soft_find_policy(transition_probabilities, reward, discount,
                threshold=100):
    n_states, n_actions = reward.shape
    v = jnp.ones(n_states)

    Q = jnp.ones((n_states, n_actions))


    def scan_iter(carry, inputs):

        v, Q = carry

        new_Q = vmap(vmap(lambda r_sa, tp_sa: r_sa + discount * jnp.dot(tp_sa, v)))(reward, transition_probabilities)
        new_v = jax.scipy.special.logsumexp(new_Q, axis=1)

        return (new_v, new_Q), 0

    (v, Q), _ = lax.scan(scan_iter, (v, Q), jnp.arange(threshold))

    Q -= Q.max(axis=1).reshape((n_states, 1))

    policy = jnp.exp(Q)/jnp.exp(Q).sum(axis=1, keepdims=True)
    
    return v, Q, policy

def vi(trans_prob, reward, discount=0.95):
    VG, QG, piG = jax_soft_find_policy(jnp.array(trans_prob), reward, discount=discount)
    return piG, QG, VG

@partial(jax.jit, static_argnames=['threshold'])
def jax_soft_find_policy_temp(transition_probabilities, reward, discount, temp, threshold=100):
    n_states, n_actions = reward.shape
    v = jnp.ones(n_states)
    Q = jnp.ones((n_states, n_actions))

    def scan_iter(carry, inputs):
        v, Q = carry
        new_Q = vmap(vmap(lambda r_sa, tp_sa: r_sa + discount * jnp.dot(tp_sa, v)))(
            reward, transition_probabilities)
        new_v = temp * jax.scipy.special.logsumexp(new_Q / temp, axis=1)
        return (new_v, new_Q), 0

    (v, Q), _ = lax.scan(scan_iter, (v, Q), jnp.arange(threshold))
    policy = jnp.exp((Q - v[:, None]) / temp)
    policy += 1e-8
    policy /= jnp.sum(policy, axis=1, keepdims=True)
    return v, Q, policy

def vi_temp(trans_prob, reward, temp, discount=0.95):
    VG, QG, piG = jax_soft_find_policy_temp(
        jnp.array(trans_prob), reward, discount=discount, temp=temp)
    return piG, QG, VG

def comp_ll_jax(logits, one_hot_x, one_hot_a):
    logits = logits - jax_logsumexp(logits, axis=-1, keepdims=True)
    transition_probs = jnp.einsum('...i,...ij->...j', one_hot_x, logits)
    lls = jnp.sum(one_hot_a * transition_probs, axis=-1)
    return lls

def sigmoid_jax(x):
    return 1 / (1 + jnp.exp(-x))

def comp_log_transP(log_Ps, Wbs, one_hot_x):
    T = one_hot_x.shape[0]
    W1, b1, W2, b2 = Wbs
    log_Ps = jnp.tile(log_Ps[None, :, :], (T-1, 1, 1))

    Z1 = jnp.dot(one_hot_x[:-1, 0, :], W1) + b1
    A1 = sigmoid_jax(Z1)
    Z2 = jnp.dot(A1, W2) + b2
    log_Ps = log_Ps + Z2[:, None, :]
    return log_Ps - jax_logsumexp(log_Ps, axis=2, keepdims=True)
    
def comp_transP(log_Ps, Wbs, one_hot_x):
    return jnp.exp(comp_log_transP(log_Ps, Wbs, one_hot_x))

def _viterbi_JAX(pi0, Ps, ll):
    """
    Implements the Viterbi algorithm in JAX.
    """
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

def jax_e_step_logpi2(pi0, log_Ps, Rs, logemit, trans_probs, xoh, xoh2, aoh):
    Ps_jax = comp_transP(jnp.array(log_Ps), Rs, jnp.array(xoh))
    log_likes_jax = comp_ll_jax(jnp.array(logemit), jnp.array(xoh2), jnp.array(aoh))
    alpha_jax = forward(pi0, Ps_jax, log_likes_jax)
    beta_jax = backward(Ps_jax, log_likes_jax)
    gamma_jax, xi_jax = expected_states(alpha_jax, beta_jax, Ps_jax, log_likes_jax)
    return gamma_jax, xi_jax, alpha_jax

def jax_e_step_logpi(pi0, log_Ps, Rs, logemit, trans_probs, xoh, aoh):
    Ps_jax = comp_transP(jnp.array(log_Ps), Rs, jnp.array(xoh))
    log_likes_jax = comp_ll_jax(jnp.array(logemit), jnp.array(xoh), jnp.array(aoh))
    alpha_jax = forward(pi0, Ps_jax, log_likes_jax)
    beta_jax = backward(Ps_jax, log_likes_jax)
    gamma_jax, xi_jax = expected_states(alpha_jax, beta_jax, Ps_jax, log_likes_jax)
    return gamma_jax, xi_jax, alpha_jax

def jax_e_step_batch2_labyrinth(pi0, log_Ps, Rs, rewards, trans_probs, xoh_list, xoh_list2, aoh_list):
    n_states, n_actions = trans_probs.shape[0], trans_probs.shape[1]
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    pi, _, _ = vmap(partial(vi, trans_probs))(rewards_sa)
    logemit = pi
    gamma_jax_list, xi_jax_list, alpha_jax_list = vmap(partial(jax_e_step_logpi2, jnp.array(pi0), jnp.array(log_Ps), Rs, jnp.array(logemit), jnp.array(trans_probs)))(jnp.array(xoh_list), jnp.array(xoh_list2), jnp.array(aoh_list))
    return gamma_jax_list, xi_jax_list, alpha_jax_list

def jax_e_step_batch_labyrinth(pi0, log_Ps, Rs, rewards, trans_probs, xoh_list, aoh_list):
    n_states, n_actions = trans_probs.shape[0], trans_probs.shape[1]
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    pi, _, _ = vmap(partial(vi, trans_probs))(rewards_sa)
    logemit = pi
    gamma_jax_list, xi_jax_list, alpha_jax_list = vmap(partial(jax_e_step_logpi, jnp.array(pi0), jnp.array(log_Ps), Rs, jnp.array(logemit), jnp.array(trans_probs)))(jnp.array(xoh_list), jnp.array(aoh_list))
    return gamma_jax_list, xi_jax_list, alpha_jax_list

def jax_e_step_batch2_temp(pi0, log_Ps, Rs, rewards, trans_probs, temps, xoh_list, xoh_list2, aoh_list):
    n_states, n_actions = trans_probs.shape[0], trans_probs.shape[1]
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    pi, _, _ = vmap(partial(vi_temp, trans_probs))(rewards_sa, temps)
    logemit = jnp.log(pi)
    gamma_jax_list, xi_jax_list, alpha_jax_list = vmap(partial(jax_e_step_logpi2, jnp.array(pi0), jnp.array(log_Ps), Rs, jnp.array(logemit), jnp.array(trans_probs)))(jnp.array(xoh_list), jnp.array(xoh_list2), jnp.array(aoh_list))
    return gamma_jax_list, xi_jax_list, alpha_jax_list

def jax_e_step_batch_temp(pi0, log_Ps, Rs, rewards, trans_probs, temps, xoh_list, aoh_list):
    n_states, n_actions = trans_probs.shape[0], trans_probs.shape[1]
    rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
    pi, _, _ = vmap(partial(vi_temp, trans_probs))(rewards_sa, temps)
    logemit = jnp.log(pi)
    gamma_jax_list, xi_jax_list, alpha_jax_list = vmap(partial(jax_e_step_logpi, jnp.array(pi0), jnp.array(log_Ps), Rs, jnp.array(logemit), jnp.array(trans_probs)))(jnp.array(xoh_list), jnp.array(aoh_list))
    return gamma_jax_list, xi_jax_list, alpha_jax_list


def trans_m_step_jax_scipy2(log_Ps, Rs, expectations, one_hot_xs, num_iters=1000, **kwargs):
    # Maximize the expected log joint
    def _expected_log_joint(params, expectations):
        # Reshape flat parameters back to original shapes
        log_Ps, Rs_flatten = params[:log_Ps_size].reshape(log_Ps_shape), params[log_Ps_size:]
        W1, b1, W2, b2 = Rs_flatten[:W1_size].reshape(W1_shape), Rs_flatten[W1_size:W1_size+b1_size].reshape(b1_shape), Rs_flatten[W1_size+b1_size:W1_size+b1_size+W2_size].reshape(W2_shape), Rs_flatten[W1_size+b1_size+W2_size:].reshape(b2_shape)
        Rs = (W1, b1, W2, b2)
        
        def scan_func(carry, inputs):
            elbo = carry
            one_hot_x, expected_states, expected_joints = inputs
            log_trans = comp_log_transP(log_Ps, Rs, one_hot_x)
            return elbo + jnp.sum(expected_joints * log_trans), 0

        elbo, _ = lax.scan(scan_func, 0, (one_hot_xs, expectations[0], expectations[1]))
        return elbo

    T = sum([one_hot_x.shape[0] for one_hot_x in one_hot_xs])

    # Define the objective function for BFGS
    def objective(flat_params, itr):
        elbo = _expected_log_joint(flat_params, expectations)
        return -elbo / T

    # Initial parameter values
    W1, b1, W2, b2 = Rs
    initial_params = (log_Ps, W1, b1, W2, b2)
    log_Ps_shape = log_Ps.shape
    W1_shape, b1_shape, W2_shape, b2_shape = W1.shape, b1.shape, W2.shape, b2.shape
    log_Ps_size = log_Ps.size
    W1_size, b1_size, W2_size, b2_size = W1.size, b1.size, W2.size, b2.size
    flat_initial_params = jnp.concatenate([log_Ps.flatten(), W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])

    # Run the optimization using jax.scipy.optimize.minimize with BFGS
    def safe_grad_jax(x, itr):
        g = jax_grad(objective)(x, itr)
        # g[~np.isfinite(g)] = 1e8
        return g

    result = minimize(objective, flat_initial_params, args=(-1,),
                      jac=safe_grad_jax,
                      method="L-BFGS-B",
                      callback=None,
                      options=dict(maxiter=num_iters, disp=False),
                      tol=1e-4)

    # Reshape the optimized parameters back to their original shapes
    optimized_log_Ps = result.x[:log_Ps_size].reshape(log_Ps_shape)
    optimized_Rs = result.x[log_Ps_size:]
    optimized_W1 = optimized_Rs[:W1_size].reshape(W1_shape)
    optimized_b1 = optimized_Rs[W1_size:W1_size+b1_size].reshape(b1_shape)
    optimized_W2 = optimized_Rs[W1_size+b1_size:W1_size+b1_size+W2_size].reshape(W2_shape)
    optimized_b2 = optimized_Rs[W1_size+b1_size+W2_size:].reshape(b2_shape)

    return (optimized_log_Ps, (optimized_W1, optimized_b1, optimized_W2, optimized_b2))


def emit_m_step_jax_scipy2(rewards, trans_probs, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    
    # Maximize the expected log joint
    def _expected_log_joint(params, expectations):
        # Reshape the flattened parameters back to their original shape
        rewards = params.reshape(initial_params_shape)
        n_states, n_actions = trans_probs.shape[0], trans_probs.shape[1]
        rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
        pi, _, _ = vmap(partial(vi, trans_probs))(rewards_sa)
        logemit = jnp.log(pi)

        def scan_func(carry, inputs):
            elbo = carry
            one_hot_x, one_hot_a, expected_states, expected_joints = inputs
            lls = comp_ll_jax(logemit, one_hot_x, one_hot_a)
            return elbo + jnp.sum(expected_states * lls), 0

        elbo, _ = lax.scan(scan_func, 0, (one_hot_xs, one_hot_acs, expectations[0], expectations[1]))
        return elbo

    T = sum([one_hot_x.shape[0] for one_hot_x in one_hot_xs])
    
    # Define the objective function for BFGS
    def objective(flat_params, itr):
        elbo = _expected_log_joint(flat_params, expectations)
        return -elbo / T

    # Initial parameter values
    initial_params = rewards
    initial_params_shape = rewards.shape
    flat_initial_params = rewards.flatten()

    

    # Run the optimization using jax.scipy.optimize.minimize with BFGS
    def safe_grad_jax(x, itr):
        g = jax_grad(objective)(x, itr)
        # g[~np.isfinite(g)] = 1e8
        return g

    result = minimize(objective, flat_initial_params, args=(-1,),
                      jac=safe_grad_jax,
                      method="BFGS",
                      callback=None,
                      options=dict(maxiter=num_iters, disp=False),
                      tol=1e-4)

    # Reshape the optimized parameters back to their original shape
    optimal_params = result.x.reshape(initial_params_shape)

    return optimal_params

def emit_m_step_jax_scipy2_labyrinth(rewards, trans_probs, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    
    # Maximize the expected log joint
    def _expected_log_joint(params, expectations):
        # Reshape the flattened parameters back to their original shape
        rewards = params.reshape(initial_params_shape)
        n_states, n_actions = trans_probs.shape[0], trans_probs.shape[1]
        rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
        pi, _, _ = vmap(partial(vi, trans_probs))(rewards_sa)
        logemit = pi

        def scan_func(carry, inputs):
            elbo = carry
            one_hot_x, one_hot_a, expected_states, expected_joints = inputs
            lls = comp_ll_jax(logemit, one_hot_x, one_hot_a)
            return elbo + jnp.sum(expected_states * lls), 0

        elbo, _ = lax.scan(scan_func, 0, (one_hot_xs, one_hot_acs, expectations[0], expectations[1]))
        return elbo

    T = sum([one_hot_x.shape[0] for one_hot_x in one_hot_xs])
    
    # Define the objective function for BFGS
    def objective(flat_params, itr):
        elbo = _expected_log_joint(flat_params, expectations)
        return -elbo / T

    # Initial parameter values
    initial_params = rewards
    initial_params_shape = rewards.shape
    flat_initial_params = rewards.flatten()

    

    # Run the optimization using jax.scipy.optimize.minimize with BFGS
    def safe_grad_jax(x, itr):
        g = jax_grad(objective)(x, itr)
        # g[~np.isfinite(g)] = 1e8
        return g

    result = minimize(objective, flat_initial_params, args=(-1,),
                      jac=safe_grad_jax,
                      method="BFGS",
                      callback=None,
                      options=dict(maxiter=num_iters, disp=False),
                      tol=1e-4)

    # Reshape the optimized parameters back to their original shape
    optimal_params = result.x.reshape(initial_params_shape)

    return optimal_params

def emit_m_step_jax_scipy2_temp(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    
    # Maximize the expected log joint
    def _expected_log_joint(params, expectations):
        # Reshape the flattened parameters back to their original shape
        rewards = params.reshape(initial_params_shape)
        n_states, n_actions = trans_probs.shape[0], trans_probs.shape[1]
        rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
        pi, _, _ = vmap(partial(vi_temp, trans_probs))(rewards_sa, temps)
        logemit = jnp.log(pi)

        def scan_func(carry, inputs):
            elbo = carry
            one_hot_x, one_hot_a, expected_states, expected_joints = inputs
            lls = comp_ll_jax(logemit, one_hot_x, one_hot_a)
            return elbo + jnp.sum(expected_states * lls), 0

        elbo, _ = lax.scan(scan_func, 0, (one_hot_xs, one_hot_acs, expectations[0], expectations[1]))
        return elbo

    T = sum([one_hot_x.shape[0] for one_hot_x in one_hot_xs])
    
    # Define the objective function for BFGS
    def objective(flat_params, itr):
        elbo = _expected_log_joint(flat_params, expectations)
        return -elbo / T

    # Initial parameter values
    initial_params = rewards
    initial_params_shape = rewards.shape
    flat_initial_params = rewards.flatten()

    

    # Run the optimization using jax.scipy.optimize.minimize with BFGS
    def safe_grad_jax(x, itr):
        g = jax_grad(objective)(x, itr)
        # g[~np.isfinite(g)] = 1e8
        return g

    result = minimize(objective, flat_initial_params, args=(-1,),
                      jac=safe_grad_jax,
                      method="BFGS",
                      callback=None,
                      options=dict(maxiter=num_iters, disp=False),
                      tol=1e-4)

    # Reshape the optimized parameters back to their original shape
    optimal_params = result.x.reshape(initial_params_shape)

    return optimal_params

def emit_m_step_jax_scipy_temp_reg(rewards, trans_probs, temps, expectations, one_hot_xs, one_hot_acs, num_iters=1000, **kwargs):
    
    # Maximize the expected log joint
    def _expected_log_joint(params, expectations):
        # Reshape the flattened parameters back to their original shape
        rewards = params.reshape(initial_params_shape)
        n_states, n_actions = trans_probs.shape[0], trans_probs.shape[1]
        rewards_sa = jnp.expand_dims(rewards[:, 0, :], axis=2) * np.ones((1, n_states, n_actions))
        pi, _, _ = vmap(partial(vi_temp, trans_probs))(rewards_sa, temps)
        logemit = jnp.log(pi)

        def scan_func(carry, inputs):
            elbo = carry
            one_hot_x, one_hot_a, expected_states, expected_joints = inputs
            lls = comp_ll_jax(logemit, one_hot_x, one_hot_a)
            return elbo + jnp.sum(expected_states * lls), 0

        elbo, _ = lax.scan(scan_func, 0, (one_hot_xs, one_hot_acs, expectations[0], expectations[1]))
        return elbo

    T = sum([one_hot_x.shape[0] for one_hot_x in one_hot_xs])
    
    # Define the objective function for BFGS
    def objective(flat_params, itr):
        elbo = _expected_log_joint(flat_params, expectations)
        rewards = flat_params.reshape(initial_params_shape)
        return -elbo / T + 0.01 * jnp.sum(jnp.abs(rewards[0])) + 0.007 * jnp.sum(jnp.abs(rewards[1])) + 0.001 * jnp.sum(jnp.square(rewards[2]))

    # Initial parameter values
    initial_params = rewards
    initial_params_shape = rewards.shape
    flat_initial_params = rewards.flatten()

    

    # Run the optimization using jax.scipy.optimize.minimize with BFGS
    def safe_grad_jax(x, itr):
        g = jax_grad(objective)(x, itr)
        # g[~np.isfinite(g)] = 1e8
        return g

    result = minimize(objective, flat_initial_params, args=(-1,),
                      jac=safe_grad_jax,
                      method="BFGS",
                      callback=None,
                      options=dict(maxiter=num_iters, disp=False),
                      tol=1e-4)

    # Reshape the optimized parameters back to their original shape
    optimal_params = result.x.reshape(initial_params_shape)

    return optimal_params

def pi0_m_step(gammas):
    pi0 = sum(gammas[:, 0]) + 1e-8
    return jnp.log(pi0 / pi0.sum())