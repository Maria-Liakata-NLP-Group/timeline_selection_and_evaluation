# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 2017-09-13
Author: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)
Forked by: Luke Shirley (L.Shirley@warwick.ac.uk)

Description: Implements the Detector, i.e. the key object in the Spatial BOCD
software. This object takes in the Data & its dimensions, the CP prior,
a set of probability models, the priors over this probability model collection,
and a model prior probability over each of the models in the collection.
"""

import numpy as np
import scipy

class Detector:
    """key object in the Spatial BOCD
    software. This object takes in the Data & its dimensions, a set of
    probability models and their priors, and a model prior probability
    over each of the models in the collection/universe, and a CP prior.

    Attributes:
        data: float numpy array;
            a SxS spatial lattice with T time points and thus of
            dimension SxSxT
        model_universe: ProbabilityModel numpy array;
            the collection of models
            that might fit a given segment. All children of the
            ProbabilityModel parent class
        model_prior: float numpy array;
            a prior vector giving the prior belief for each entry
            in the *model_universe* numpy array. If model_prior[i-1] = p,
            then we have prior belief p that the i-th model is the generating
            process for a given segment
        cp_model: CpModel object;
            the CP model, usually specified as a geometric distribution
            over the time points.
        Q: int;
            the number of entries in the model_universe numpy array
        T, S1, S2: int;
            the dimensions of the spatial lattice (S1, S2) and the time (T),
            provided that the data is not put in as data stream
        predictive_probs: float numpy array;
            stores the predictive probs at each time point for each lattice
            point and for each potential model
        growth_probs: float numpy array;
            stores the probability of the run lengths being r = 1,2,..., t at
            time point t.
        cp_probs: float numpy array;
            stores the probability of the run length being r=0 at time point t
        evidence: float numpy array;
            stores the evidence (posterior probability) of having observed
            all values of the data up until time point t at each time point t
        run_length_distr: float numpy array;
            stores the run-length distribution for r=0,1,...,t at each time
            point t.

    """

    def __init__(self, data, model_universe, model_prior, cp_model, S1, S2, T, threshold=None):
        """construct the Detector with the multi-dimensional numpy array
        *data*. E.g., if you have a SxS spatial lattice with T time points,
        then *data* will be SxSxT. The argument *model_universe* will provide
        you with a numpy array of ProbabilityModel objects, each of which is
        associated with a model that could fit the data and allows for online
        bayesian CP detection. The *model_prior* is a numpy array of floats,
        summing to one such that the i-th entry corresponds to the prior belief
        that the i-th model occurs in a segment. Lastly, *cp_model* is an
        object of class CpModel that stores all properties about the CP prior,
        i.e. the probability of one occuring at every time point.
        """

        """store the inputs into object"""
        self.data = data.reshape(T, S1*S2)
        self.model_universe = model_universe
        self.model_prior = model_prior
        self.cp_model = cp_model
        self.Q = model_universe.shape[0]  # Q: number of models in the model universe
        self.T, self.S1, self.S2 = T, S1, S2
        self.threshold = threshold


        """create internal data structures for most recent computed objects"""
        self.evidence = -np.inf
        self.MAP = None
        self.y_pred_mean = np.zeros(shape=(self.S1, self.S2))
        self.y_pred_var  = np.zeros(shape = (self.S1*self.S2, self.S1*self.S2))


        """create internal data structures for all computed objects"""
        self.model_and_run_length_log_distr = (-np.inf *
                                            np.ones(shape = (self.Q, self.T+1)))
        self.storage_run_length_distr = np.zeros(shape=(self.T+1, self.T+1))
        self.storage_mean = np.zeros(shape = (self.T, self.S1, self.S2))
        self.storage_var = np.zeros(shape = (self.T, self.S1*self.S2,
                                             self.S1*self.S2))
        self.storage_log_evidence = -np.inf * np.ones(shape = self.T)
        self.log_MAP_storage = np.zeros(self.T)
        self.MAP_segmentation = [np.array([[],[]])]
        self.segment_log_densities = np.zeros(shape = (self.Q, self.T) )


    def run(self, start=None, stop=None):
        """Start running the Detector from *start* to *stop*, usually from
        the first to the last observation (i.e. default).
        """

        """set start and stop if not supplied"""
        if start is None:
            start = 1
        if stop is None:
            stop = self.T

        """run the detector"""
        for t in range(start-1, stop-1):
            self.next_run(self.data[t,], t+1)


    def next_run(self, y, t):
        """for a new observation *y* at time *t*, run the entire algorithm
        and compute all quantities at the ProbabilityModel object and the
        Detector object layer necessary

        NOTE: The data structures in the ProbabilityModel objects grow from
        size t-1 (or t, for the sufficient statistics) to size t (or t+1)
        during 'next_run'.        #DEBUG: To be added: check if prior_mean, prior_var, ... are S1xS2. If
        #       they are not, assume that they are provided in vector and
        #       full covariance matrix form & compress them into internal form
        """

        """STEP 1: If t==1, initialize *joint_probabilities* and the
        predictive distributions.
        If t>1, update the *joint_probabilities* of (y_{1:t}, r_t =r|q_t =q)
        for all q in the model universe as well as the predictive
        distributions associated with each model."""
        self.update_all_joint_log_probabilities(y,t)

        """STEP 2: Collect the model-specific evidences and update the overall
        evidence by summing them up. 'update_evidence' is a simple wrapper
        function for convenience"""
        self.update_log_evidence()

        """STEP 3: Trim the run-length distributions in each model. Next,
        update the distributions (q_t=q, r_t=r|y_{1:t}) for each
        run-length r and each model in the model universe q, store the
        result in *self.model_and_run_length_distr*"""
        self.trim_run_length_log_distributions(t)
        self.update_model_and_run_length_log_distribution(t)

        """STEP 4: Using the results from STEP 3, obtain a prediction for the
        next spatial lattice slice, which you preferably should either store,
        output, or write to some location"""
        #NOTE: THIS QUANTITY SHOULD BE STORED/WRITTEN SOMEWHERE!
        self.prediction_y(t)
        self.storage(t)

        """STEP 5: Using the results from STEP 3, obtain a MAP for the
        most likely segmentation & models per segment using the algorithm of
        Fearnhead & Liu (2007)"""
        #NOTE: THIS QUANTITY SHOULD BE STORED/WRITTEN SOMEWHERE!
        self.MAP_estimate(t)

        #NEEDS FURTHER INVESTIGATION
        """STEP 6: For each model in the model universe, update the priors to
        be the posterior expectation/variance"""
        self.update_priors(t)


    #IMPLEMENTED FOR ALL SUBCLASSES IF predictive_probabilities WORK IN SUBLCASS
    #                                   update_joint_probabilities WORK
    def update_all_joint_log_probabilities(self, y, t):
        """Let the individual objects in *model_universe* compute their growth
        and CP probabilities, via 'update_joint_probabilities' in each"""
        q = 0
        for model in self.model_universe:
            """STEP 0: Check if it is the first observation, and initialize the
            joint distributions in each model object if so"""
            if t == 1:
                """This command gives an initialization of (i) the
                *joint_probabilities*, (ii) the predictive distributions (i.e.,
                the sufficient statistics), and (iii) the *model_evidence*
                of each model."""

                # if isinstance(model, BVARNIG):
                #     """If we have a BVAR model, we need to pass data of
                #     sufficient lag length to the initialization."""
                #     y_extended = self.data[:model.lag_length+1,:]
                #     model.initialization(y_extended, self.cp_model,
                #                          self.model_prior[q])
                # else:
                """Otherwise, just pass the first observation"""
                model.initialization(y, self.cp_model, self.model_prior[q])
                "The following computations are done in the class of each model q:"
                "PP^D(1,0,q) ← Compute{πq,fq,y1}"
                "D(1, q) ← SD(1, 0, q) ← Eval (Compute {πq, fq} , y1)"
                "DRM^E(1, 0, q) ← m(q) · D(1, q)"

            # elif ( (not (isinstance(model, BVARNIG))) or
            #       (model.lag_length+1 < t)):
            else:
                """Make sure that we only modify joint_log_probs & predictive
                distributions for BVAR models for which t is large enough to
                cover the lag length."""

                """STEP 1: Within each ProbabilityModel object, compute its
                joint probabilities, i.e. the growth- & CP-probabilities
                associated with the model"""
                model.update_joint_log_probabilities(y,t, self.cp_model,
                                             self.model_prior[q])
                """STEP 2: Update the predictive probabilities for each model
                inside the associated ProbabilityModel object. These are
                used in the 'update_joint_probabilities' call for the next
                observation. In the case of NaiveModel and other conjugate
                models, this step amounts to updating the sufficient
                statistics associated with the model"""
                model.update_predictive_distributions(y, t)

            """keep track of q, which is needed for the *model_prior*
            indexing in STEP 1 and for the initialization if t==0"""
            q = q + 1


    #IMPLEMENTED FOR ALL SUBCLASSES IF model_evidence UPDATED CORRECTLY IN SUBLCASS
    def update_log_evidence(self):
        """Sum up all the model-specific evidences from the submodels"""
        self.log_evidence = scipy.special.logsumexp([model.model_log_evidence
                                for model in self.model_universe])


    def update_model_and_run_length_log_distribution(self, t):
        """Using the updated evidence, calculate the distribution of the
        run-lengths and models jointly by accessing each model in order to
        retrieve the joint probabilities. These are then scaled by 1/evidence
        and the result is stored as the new *model_and_run_length_distr*

        NOTE: If one does NOT compute the MAP, then it is more efficient not to
        store this quantity, and rather compute it directly when the next
        observation y is predicted!
        """

        """For each model object in model_universe, get the associated
        *joint_probabilities* np arrays, divide them by *self.evidence*
        and store the result. Notice that at time t, we will have t+1 entries
        in the joint_probabilities for r=0,1,...t-1, >t-1"""
#        self.model_and_run_length_log_distr[:,0:(t+1)] = (
#                np.log(1.0) - self.log_evidence +
#                np.vstack([model.joint_log_probabilities for
#                model in self.model_universe]))

        """STEP 1: Get the longest non-zero run-length. Keep in mind that we
        may have r=t-1 and r>t-1, but both are stored as same run-length!"""
        r_max, r_max2= 0,0
        for q in range(0, self.Q):
            retained_q = self.model_universe[q].retained_run_lengths
            """STEP 1.1: Check if in model q, we have a larger maximum
            run-length than in previous models"""
            r_max2 = max(np.max(retained_q), r_max)
            if r_max2 >= r_max:
                r_max = r_max2
                """STEP 1.2: If a new maximum was found, check if one retains
                both r=t-1 and r>t-1 in this model. If so, advance r_max"""
                if ((retained_q.shape[0]>1) and (retained_q[-1] ==
                    retained_q[-2])):
                    r_max = r_max + 1

        #r_max = np.max( [model.retained_run_lengths
        #                 for model in self.model_universe])

        """STEP 2: Initialize the model-and-run-length log distribution"""
        self.model_and_run_length_log_distr = (-np.inf *
                np.ones(shape=(self.Q, r_max + 1))) #Easier: t+1

        """STEP 3: Where relevant, fill it in"""
        for i in range(0, self.Q):

            """STEP 3.1: Get the retained run-lengths"""
            model = self.model_universe[i]
            retained = model.retained_run_lengths
            if ((retained.shape[0]>1) and (retained[-1] == retained[-2])):
                retained = np.copy(model.retained_run_lengths)
                retained[-1] = retained[-1] + 1

            """STEP 3.2: Update the model-and-run-length-log-distribution
            corresponding to the current model being processed"""
            self.model_and_run_length_log_distr[i,retained]=(
                    model.joint_log_probabilities - model.model_log_evidence)

        """MARKED: Luke store run length distr at each time"""
        """STEP 4: Store run-lengths"""
        r = np.sum(np.exp(self.model_and_run_length_log_distr), axis=0)
        self.storage_run_length_distr[t][range(0, r_max + 1)] = r


    def prediction_y(self, t):
        """Using the information of all potential models and run-lengths,
        make a prediction about the next observation. In particular,
        you want the MAP/Expected value as well as the standard deviation/
        variance to put CIs around your predicted value"""

        """for each model in the model universe, extract the posterior
        expectation and the posterior variance for each run length,
        weighted by model_and_run_length_distr, and sum them up.
        This yields the posterior mean and posterior variance under
        model uncertainty"""

        #DEBUG: Post mean and post var incorrectly computed ?
        post_mean , post_var, q = (np.zeros(shape=(self.S1, self.S2)),
            np.zeros(shape=(self.S1*self.S2, self.S1*self.S2)), 0)
        for model in self.model_universe:
            """simple reweighting of the means of each (q,r) combination by
            the appropriate probability distribution"""

            """Make sure we account for r=t-1 and r>t-1"""
            run_lengths_q = np.copy(model.retained_run_lengths)
            r_shape = run_lengths_q.shape[0]
            if run_lengths_q[-1] == run_lengths_q[-2]:
                run_lengths_q[-1] = run_lengths_q[-1] + 1

            #DEBUG: I want to circumvent the exp-conversion!
            #       Callable via LogSumExp.logsumexp(...)
            post_mean = (post_mean +
                np.sum(
                        np.reshape(model.get_posterior_expectation(t),
                                   newshape = (r_shape, self.S1, self.S2)) *
                        np.exp(self.model_and_run_length_log_distr[q,run_lengths_q]
                        )[:,np.newaxis, np.newaxis],
                axis = 0))
            """reweighting of the variance of each (q,r) combination
            by the appropriate probability distribution. Square introduced
            because of the variance"""
            """APPROXIMATE POSTERIOR VARIANCE COMPUTATION! But can
            also be seen as an weighing the variance with model
            probabilities"""
            post_var = (post_var +
                np.sum(
                        np.reshape(model.get_posterior_variance(t),
                            newshape = (r_shape, self.S1*self.S2, self.S1*self.S2)) *
                        np.exp(
                            self.model_and_run_length_log_distr[q,run_lengths_q])
                            [:, np.newaxis,np.newaxis],
                axis = 0) )
            q = q+1

        """lastly, store the new posterior mean & variance in the relevant
        object"""
        self.y_pred_mean, self.y_pred_var = post_mean, post_var


    def storage(self, t):
        """helper function, just stores y_pred into storage_mean & storage_var
        so that we can always access the last computed quantity"""
        self.storage_mean[t-1, :, :]  = self.y_pred_mean
        self.storage_var[t-1, :, :] = self.y_pred_var

        self.storage_log_evidence[t-1] = self.log_evidence


    def MAP_estimate(self, t):
        """Using the information of all potential models and run-lengths,
        get the MAP segmentation & model estimate.

        This is done in two steps: First, we use a recursive relationship to
        compute the probability density associated with the candidate
        segmentations. Second, we solve the MAP maximization problem and
        store the next MAP segmentation
        """

        """STEP 1: Get the longest non-zero run-length. Keep in mind that we
        may have r=t-1 and r>t-1, but both are stored as same run-length!"""
        r_max_, r_max2_ = 0,0
        for q in range(0, self.Q):
            retained_q = (self.model_universe[q].retained_run_lengths)
            """STEP 1.1: Check if in model q, we have a larger maximum
            run-length than in previous models"""
            r_max2_ = max(  np.max(retained_q), r_max_ )
            if r_max2_ >= r_max_:
                r_max_, q_max = r_max2_, q
                """STEP 1.2: If a new maximum was found, check if one retains
                both r=t-1 and r>t-1 in this model. If so, advance r_max"""
                if retained_q[-1] == retained_q[-2]:
                    r_max_ = r_max_ + 1

        """STEP 2.1: Initialize the model-and-run-length log distribution"""
        log_densities = (-np.inf * np.ones(shape=(self.Q, r_max_ + 1)))
        retained_indices = (-1)*np.ones(shape=(self.Q, r_max_ + 1), dtype=int)

        """STEP 2.2: Where relevant, fill it in"""
        for i in range(0, self.Q):
            """STEP 1.3.1: Get the retained run-lengths"""
            model = self.model_universe[i]
            retained = model.retained_run_lengths
            retained_indices[i,retained] = model.retained_run_lengths
            """STEP 1.3.2: if r=t-1 and r>t-1 are both retained, make sure that
            we displace the position of r>t-1"""
            if ((retained.shape[0]>1) and (retained[-1] == retained[-2])):
                np.copy(model.retained_run_lengths)
                retained[-1] = retained[-1] + 1
            """STEP 1.3.3: Update the model-and-run-length-log-distribution
            corresponding to the current model being processed"""
            log_densities[i,retained]= model.joint_log_probabilities - model.model_log_evidence

        """STEP 2: Find the next MAP candidates using Fearnhead & Liu (2007)
        and the proposed recursion therein"""
        if t>1:
            """For t>1, we have previous segmentations and the P_t^MAP of
            Fearnhead & Liu (2007) is stored in log-format in *log_MAP_storage,
            so we may apply the recursion as in the paper"""
            candidates = (log_densities +
                      np.flipud(self.log_MAP_storage[-r_max_-1:])[np.newaxis, :])
        else:
            """For t=1, we have no previous segmentations and so
            the P_t^MAP quantity in Fearnhead & Liu is simply 1"""
            candidates = log_densities

        # print('candidates',candidates)
        """STEP 3: Find the MAP segmentation"""
        index = np.argmax( candidates )
        q_max, r_max_pointer = np.unravel_index(index, candidates.shape)
        r_max = retained_indices[q_max, r_max_pointer]
        #PROBLEM: MAP segmentation for r>t-1!!! Cannot distinguish between the two atm!

        if t>1:
            """STEP 4A: Update the P_j^MAP quantities and the MAP estimate for
            CPs and model orders between the CPs provided it is not the first
            observations."""

            """STEP 4A.1: Update *log_MAP_storage* by appending the
            density associated with the new MAP estimate"""
            self.log_MAP_storage = np.append(self.log_MAP_storage,
                    candidates[q_max, r_max])

            """STEP 4A.2: Create the (new) segment associated with the new MAP
            estimate at time t. If we get r_max = 0, then we have a MAP
            estimated CP at time t-1, hence add an additional -1."""
            last_segment = np.array([[t-r_max-1, q_max]])

            """STEP 4A.3: Update *MAP_segmentation*  and its last entry which is
            stored as *MAP* accordingly. Here, notice that since we append to
            the *MAP_segmentation*, we have to call the relevant entries from
            the back, i.e. we access -r-1 in the array to get the MAP estimate
            associated with t-r-1."""
            self.MAP = np.concatenate((self.MAP_segmentation[-1-r_max],
                                  last_segment.T),axis=1)
            self.MAP_segmentation.append( (np.concatenate(
                                         (self.MAP_segmentation[-1-r_max],
                                         last_segment.T), axis=1)))
        else:
            """STEP 4B: Update the P_j^MAP quantities and the MAP estimate for
            CPs and model orders between the CPs if it is the first
            observation."""

            """STEP 4B.1: We have MAP_0 = 1, since the first two entries of
            log_MAP_storage will correspond to the r=t-1 and r>t-1
            log_densities in the next run. MAP_1 corresponds to the most likely
            model associated with a CP at time point 1, i.e. r_max = 0 or
            r_max > 0 and q_max is the most likely model given a CP at t=1 or
            before t=1."""
            self.log_MAP_storage = np.array([0, 0,candidates[q_max, r_max]])

            """STEP 4B.2: If r_max=0 => t-r_max = 1; r_max=1 => t-r_max = 0,
            so if the output later contains -1 as a CP, that means that the
            most likely CP occured before the first data point was observed"""
            self.MAP = np.array([[t-r_max], [q_max]])
            self.MAP_segmentation = [np.array([[],[]]), np.array([[],[]]),
                                     np.array([[t-r_max],[q_max]])]



    def trim_run_length_log_distributions(self, t):
        """Trim the distributions within each model object by calling a trimmer on
        all model objects in the model universe. Pass the threshold down"""
        for model in self.model_universe:
            #NOTE: Would be ideal to implement this on probability_model level!
            model.trim_run_length_log_distrbution(t, self.threshold)


    #DEBUG: Relocate to probability_model
    def update_priors(self, t):
        """update the priors for each model in the model universe, provided
        that the boolean auto_prior_update is true"""
        for model in self.model_universe:
            if model.auto_prior_update == True:
                model.prior_update(t)
