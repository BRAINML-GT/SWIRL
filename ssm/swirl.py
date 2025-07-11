import copy

import autograd.numpy as np
import autograd.numpy.random as npr

import ssm.observations as obs
import ssm.transitions as trans
import ssm.init_state_distns as isd
import ssm.emissions as emssn
import ssm.hmm_old as hmm_old
import ssm.variational as varinf
from ssm.util import ensure_args_are_lists, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists, ssm_pbar, ensure_args_are_lists2
from ssm.messages import hmm_expected_states, viterbi, arhmm_viterbi, arhmm_normalizer, hmm_normalizer

class HMMs(object):
    """
    Switching linear dynamical system fit with
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    N: emissions
    K: discrete states
    D: latent state dimension
    """
    def __init__(self, N, K, D, C, *, M=0,
                 init_state_distn=None,
                 transitions="standard",
                 transition_kwargs=None,
                 dynamics="categorical",
                 dynamics_kwargs=None,
                 **kwargs):

        # Make the initial state distribution
        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        assert isinstance(init_state_distn, isd.InitialStateDistribution)

        # Make the transition model
        transition_classes = dict(
            standard=trans.StationaryTransitions,
            stationary=trans.StationaryTransitions,
            recurrent=trans.RecurrentTransitions,
            mlprecurrent=trans.MLPRecurrentTransitions,
            recurrent_only=trans.RecurrentOnlyTransitions,
            )

        if isinstance(transitions, str):
            transitions = transitions.lower()
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = transition_classes[transitions](K, D, C=C, M=M, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # Make the dynamics distn
        dynamics_classes = dict(
            none=obs.GaussianObservations,
            categorical=obs.CategoricalObservations,
            )

        if isinstance(dynamics, str):
            dynamics = dynamics.lower()
            if dynamics not in dynamics_classes:
                raise Exception("Invalid dynamics model: {}. Must be one of {}".
                    format(dynamics, list(dynamics_classes.keys())))

            dynamics_kwargs = dynamics_kwargs or {}
            dynamics = dynamics_classes[dynamics](K, D=1, M=M, C=C, **dynamics_kwargs)
        if not isinstance(dynamics, obs.Observations):
            raise TypeError("'dynamics' must be a subclass of"
                            " ssm.observations.Observations")

        self.N, self.K, self.D, self.M = N, K, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.dynamics.params \

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.dynamics.params = value[2]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None,
                   verbose=0,
                   num_init_iters=50,
                   discrete_state_init_method="random",
                   num_init_restarts=1):
        # First initialize the observation model
        # self.emissions.initialize(datas, inputs, masks, tags)

        # Get the initialized variational mean for the data
        # xs = [self.emissions.invert(data, input, mask, tag)
            #   for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        # xmasks = [np.ones_like(x, dtype=bool) for x in xs]
        
        # self.dynamics.initialize(datas, inputs, masks, tags)
        xmasks = [np.ones_like(x, dtype=bool) for x in datas]

        # Number of times to run the arhmm initialization (we'll use the one with the highest log probability as the initialization)
        pbar  = ssm_pbar(num_init_restarts, verbose, "ARHMM Initialization restarts", [''])

        #Loop through initialization restarts
        best_lp = -np.inf
        best_lls = []
        for i in pbar: #range(num_init_restarts):

            # Now run a few iterations of EM on a ARHMM with the variational mean
            if verbose > 0:
                print("Initializing with an ARHMM using {} steps of EM.".format(num_init_iters))

            model = hmm_old.HMM(self.K, self.D, M=self.M,
                            init_state_distn=copy.deepcopy(self.init_state_distn),
                            transitions=copy.deepcopy(self.transitions),
                            observations=copy.deepcopy(self.dynamics))

            lls = model.fit(datas, inputs=inputs, masks=xmasks, tags=tags,
                      verbose=verbose,
                      method="em",
                      num_iters=num_init_iters,
                      initialize=False,
                      init_method=discrete_state_init_method)

            #Keep track of the arhmm that led to the highest log probability
            current_lp = model.log_probability(datas)
            if current_lp > best_lp:
                best_lp =  copy.deepcopy(current_lp)
                best_model = copy.deepcopy(model)
                best_lls = copy.deepcopy(lls)

        self.init_state_distn = copy.deepcopy(best_model.init_state_distn)
        self.transitions = copy.deepcopy(best_model.transitions)
        self.dynamics = copy.deepcopy(best_model.observations)
        return best_lls
    def sample(self, T, input=None, tag=None, prefix=None, with_noise=True):
        N = self.N
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)

        # If prefix is given, pad the output with it
        if prefix is None:
            pad = 1
            z = np.zeros(T+1, dtype=int)
            x = np.zeros((T+1,) + D)
            # input = np.zeros((T+1,) + M) if input is None else input
            input = np.zeros((T+1,) + M) if input is None else np.concatenate((np.zeros((1,) + M), input))
            xmask = np.ones((T+1,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = self.init_state_distn.initial_state_distn
            z[0] = npr.choice(self.K, p=pi0)
            x[0] = self.dynamics.sample_x(z[0], x[:0], tag=tag, with_noise=with_noise)

        else:
            zhist, xhist, yhist = prefix
            pad = len(zhist)
            assert zhist.dtype == int and zhist.min() >= 0 and zhist.max() < K
            # assert xhist.shape == (pad, D)
            assert yhist.shape == (pad, N)

            z = np.concatenate((zhist, np.zeros(T, dtype=int)))
            x = np.concatenate((xhist, np.zeros((T,) + D)))
            # input = np.zeros((T+pad,) + M) if input is None else input
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            xmask = np.ones((T+pad,) + D, dtype=bool)

        # Sample z and x
        for t in range(pad, T+pad):
            Pt = np.exp(self.transitions.log_transition_matrices(x[t-1:t+1], input[t-1:t+1], mask=xmask[t-1:t+1], tag=tag))[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            x[t] = self.dynamics.sample_x(z[t], x[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Sample observations given latent states
        # TODO: sample in the loop above?
        # y = self.emissions.sample(z, x, input=input, tag=tag)
        return z[pad:], x[pad:]
    
    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(variational_mean, input, mask, tag)
        log_likes = self.dynamics.log_likelihoods(variational_mean, input, np.ones_like(variational_mean, dtype=bool), tag)
        return viterbi(pi0, Ps, log_likes)
    
    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.dynamics.permute(perm)

class ARHMMs(object):
    """
    Switching linear dynamical system fit with
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    N: emissions
    K: discrete states
    D: latent state dimension
    """
    def __init__(self, N, K, D, C, *, M=0,
                 init_state_distn=None,
                 transitions="standard",
                 transition_kwargs=None,
                 dynamics="arcategorical",
                 dynamics_kwargs=None,
                 **kwargs):

        # Make the initial state distribution
        if init_state_distn is None:
            init_state_distn = isd.InitialStateDistribution(K, D, M=M)
        assert isinstance(init_state_distn, isd.InitialStateDistribution)

        # Make the transition model
        transition_classes = dict(
            standard=trans.StationaryTransitions,
            stationary=trans.StationaryTransitions,
            recurrent=trans.RecurrentTransitions,
            recurrent2=trans.Recurrent2Transitions,
            recurrent_only=trans.RecurrentOnlyTransitions,
            mlprecurrent=trans.MLPRecurrentTransitions,
            tablerecurrent=trans.TableRecurrentTransitions,
            mlp2recurrent=trans.MLP2RecurrentTransitions,
            )

        if isinstance(transitions, str):
            transitions = transitions.lower()
            if transitions not in transition_classes:
                raise Exception("Invalid transition model: {}. Must be one of {}".
                    format(transitions, list(transition_classes.keys())))

            transition_kwargs = transition_kwargs or {}
            transitions = transition_classes[transitions](K, D, M=M, C=C, **transition_kwargs)
        if not isinstance(transitions, trans.Transitions):
            raise TypeError("'transitions' must be a subclass of"
                            " ssm.transitions.Transitions")

        # Make the dynamics distn
        dynamics_classes = dict(
            none=obs.GaussianObservations,
            arcategorical=obs.ARCategoricalObservations,
            categorical=obs.CategoricalObservations,
            )

        if isinstance(dynamics, str):
            dynamics = dynamics.lower()
            if dynamics not in dynamics_classes:
                raise Exception("Invalid dynamics model: {}. Must be one of {}".
                    format(dynamics, list(dynamics_classes.keys())))

            dynamics_kwargs = dynamics_kwargs or {}
            dynamics = dynamics_classes[dynamics](K, D=1, M=M, C=C, **dynamics_kwargs)
        if not isinstance(dynamics, obs.Observations):
            raise TypeError("'dynamics' must be a subclass of"
                            " ssm.observations.Observations")

        # Make the emission distn
        # emission_classes = dict(
        #     gaussian=emssn.GaussianEmissions,
        #     gaussian_orthog=emssn.GaussianOrthogonalEmissions,
        #     gaussian_id=emssn.GaussianIdentityEmissions,
        #     gaussian_nn=emssn.GaussianNeuralNetworkEmissions,
        #     studentst=emssn.StudentsTEmissions,
        #     studentst_orthog=emssn.StudentsTOrthogonalEmissions,
        #     studentst_id=emssn.StudentsTIdentityEmissions,
        #     studentst_nn=emssn.StudentsTNeuralNetworkEmissions,
        #     t=emssn.StudentsTEmissions,
        #     t_orthog=emssn.StudentsTOrthogonalEmissions,
        #     t_id=emssn.StudentsTIdentityEmissions,
        #     t_nn=emssn.StudentsTNeuralNetworkEmissions,
        #     poisson=emssn.PoissonEmissions,
        #     poisson_orthog=emssn.PoissonOrthogonalEmissions,
        #     poisson_id=emssn.PoissonIdentityEmissions,
        #     poisson_nn=emssn.PoissonNeuralNetworkEmissions,
        #     bernoulli=emssn.BernoulliEmissions,
        #     bernoulli_orthog=emssn.BernoulliOrthogonalEmissions,
        #     bernoulli_id=emssn.BernoulliIdentityEmissions,
        #     bernoulli_nn=emssn.BernoulliNeuralNetworkEmissions,
        #     ar=emssn.AutoRegressiveEmissions,
        #     ar_orthog=emssn.AutoRegressiveOrthogonalEmissions,
        #     ar_id=emssn.AutoRegressiveIdentityEmissions,
        #     ar_nn=emssn.AutoRegressiveNeuralNetworkEmissions,
        #     autoregressive=emssn.AutoRegressiveEmissions,
        #     autoregressive_orthog=emssn.AutoRegressiveOrthogonalEmissions,
        #     autoregressive_id=emssn.AutoRegressiveIdentityEmissions,
        #     autoregressive_nn=emssn.AutoRegressiveNeuralNetworkEmissions
        #     )

        # if isinstance(emissions, str):
        #     emissions = emissions.lower()
        #     if emissions not in emission_classes:
        #         raise Exception("Invalid emission model: {}. Must be one of {}".
        #             format(emissions, list(emission_classes.keys())))

        #     emission_kwargs = emission_kwargs or {}
        #     emissions = emission_classes[emissions](N, K, D, M=M,
        #         single_subspace=single_subspace, **emission_kwargs)
        # if not isinstance(emissions, emssn.Emissions):
        #     raise TypeError("'emissions' must be a subclass of"
        #                     " ssm.emissions.Emissions")

        self.N, self.K, self.D, self.M = N, K, D, M
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        # self.emissions = emissions

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.dynamics.params \
            #    self.emissions.params

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.dynamics.params = value[2]
        # self.emissions.params = value[3]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None,
                   verbose=0,
                   num_init_iters=50,
                   Ronly=False,
                   discrete_state_init_method="random",
                   num_init_restarts=1):
        # First initialize the observation model
        # self.emissions.initialize(datas, inputs, masks, tags)

        # Get the initialized variational mean for the data
        # xs = [self.emissions.invert(data, input, mask, tag)
            #   for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        # xmasks = [np.ones_like(x, dtype=bool) for x in xs]
        
        # self.dynamics.initialize(datas, inputs, masks, tags)
        xmasks = [np.ones_like(x, dtype=bool) for x in datas]

        # Number of times to run the arhmm initialization (we'll use the one with the highest log probability as the initialization)
        pbar  = ssm_pbar(num_init_restarts, verbose, "ARHMM Initialization restarts", [''])

        #Loop through initialization restarts
        best_lp = -np.inf
        best_lls = []
        for i in pbar: #range(num_init_restarts):

            # Now run a few iterations of EM on a ARHMM with the variational mean
            if verbose > 0:
                print("Initializing with an ARHMM using {} steps of EM.".format(num_init_iters))

            arhmm = hmm_old.ARHMM(self.K, self.D, M=self.M,
                            init_state_distn=copy.deepcopy(self.init_state_distn),
                            transitions=copy.deepcopy(self.transitions),
                            observations=copy.deepcopy(self.dynamics))

            if Ronly:
                lls = arhmm.fit(datas, inputs=inputs, masks=xmasks, tags=tags,
                      verbose=verbose,
                      method="emR",
                      num_iters=num_init_iters,
                      initialize=False,
                      init_method=discrete_state_init_method)
            else:
                lls = arhmm.fit(datas, inputs=inputs, masks=xmasks, tags=tags,
                      verbose=verbose,
                      method="em",
                      num_iters=num_init_iters,
                      initialize=False,
                      init_method=discrete_state_init_method)

            #Keep track of the arhmm that led to the highest log probability
            current_lp = arhmm.log_probability(datas)
            if current_lp > best_lp:
                best_lp =  copy.deepcopy(current_lp)
                best_arhmm = copy.deepcopy(arhmm)
                best_lls = copy.deepcopy(lls)

        self.init_state_distn = copy.deepcopy(best_arhmm.init_state_distn)
        self.transitions = copy.deepcopy(best_arhmm.transitions)
        self.dynamics = copy.deepcopy(best_arhmm.observations)
        return best_lls
    def sample(self, T, input=None, tag=None, prefix=None, with_noise=True):
        N = self.N
        K = self.K
        D = (self.D,) if isinstance(self.D, int) else self.D
        M = (self.M,) if isinstance(self.M, int) else self.M
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)

        # If prefix is given, pad the output with it
        if prefix is None:
            pad = 1
            z = np.zeros(T+1, dtype=int)
            x = np.zeros((T+1,) + D)
            # input = np.zeros((T+1,) + M) if input is None else input
            input = np.zeros((T+1,) + M) if input is None else np.concatenate((np.zeros((1,) + M), input))
            xmask = np.ones((T+1,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = self.init_state_distn.initial_state_distn
            z[0] = npr.choice(self.K, p=pi0)
            x[0] = self.dynamics.sample_x(z[0], x[:0], tag=tag, with_noise=with_noise)

        else:
            zhist, xhist, yhist = prefix
            pad = len(zhist)
            assert zhist.dtype == int and zhist.min() >= 0 and zhist.max() < K
            # assert xhist.shape == (pad, D)
            assert yhist.shape == (pad, N)

            z = np.concatenate((zhist, np.zeros(T, dtype=int)))
            x = np.concatenate((xhist, np.zeros((T,) + D)))
            # input = np.zeros((T+pad,) + M) if input is None else input
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            xmask = np.ones((T+pad,) + D, dtype=bool)

        # Sample z and x
        for t in range(pad, T+pad):
            Pt = np.exp(self.transitions.log_transition_matrices(x[t-1:t+1], input[t-1:t+1], mask=xmask[t-1:t+1], tag=tag))[0]
            z[t] = npr.choice(self.K, p=Pt[z[t-1]])
            x[t] = self.dynamics.sample_x(z[t], x[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Sample observations given latent states
        # TODO: sample in the loop above?
        # y = self.emissions.sample(z, x, input=input, tag=tag)
        return z[pad:], x[pad:]
    
    @ensure_slds_args_not_none
    def most_likely_states(self, variational_mean, data, input=None, mask=None, tag=None):
        pi0 = self.init_state_distn.initial_state_distn
        Ps = self.transitions.transition_matrices(variational_mean, input, mask, tag)
        log_likes = self.dynamics.log_likelihoods(variational_mean, input, np.ones_like(variational_mean, dtype=bool), tag)
        # log_likes += self.emissions.log_likelihoods(data, input, mask, tag, variational_mean)
        return arhmm_viterbi(pi0, Ps, log_likes)
    
    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.K))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.dynamics.permute(perm)

    @ensure_args_are_lists
    def log_likelihood(self, datas, inputs=None, masks=None, tags=None):
        """
        Compute the log probability of the data under the current
        model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        ll = 0
        for data, input, mask, tag in zip(datas, inputs, masks, tags):
            pi0 = self.init_state_distn.initial_state_distn
            Ps = self.transitions.transition_matrices(data, input, mask, tag)
            log_likes = self.dynamics.log_likelihoods(data, input, mask, tag)
            ll += arhmm_normalizer(pi0, Ps, log_likes)
            assert np.isfinite(ll)
        return ll
