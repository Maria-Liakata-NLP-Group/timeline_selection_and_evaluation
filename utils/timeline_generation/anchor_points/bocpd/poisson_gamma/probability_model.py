# Python 3.5.2 |Anaconda 4.2.0 (64-bit)|
# -*- coding: utf-8 -*-
"""
Last edited: 2017-09-13
Author: Jeremias Knoblauch (J.Knoblauch@warwick.ac.uk)
Forked by: Luke Shirley (L.Shirley@warwick.ac.uk)

Description: Implements the abstract class ProbabilityModel, which is the
parent class of all probability models used by Spatial BOCD. E.g., the
NaiveModel class has this abstract class as its parent and implements the
model where we treat each spatial location as independent from all others
(in a Gaussian setting)
"""

"""import all you need to build an abstract class"""
#from abc import ABCMeta, abstractmethod
import numpy as np
import scipy #.special #import logsumexp

class ProbabilityModel:
    #__metaclass__ = ABCMeta
    """An abstract class for all probability models that will live in the model
    universe. I.e., each explicit probability model (e.g., the naive model that
    does assume independence between locations) will inherit from this class.


    NOTE: I Should still look into superclass constructors!

    Abstract Methods:
        predictive_probabilities
        growth_probabilities
        evidence
        cp_probabilities
        run_length_distribution
        update_predictive_distributions
    """





    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS
    def evaluate_predictive_log_distribution(self, y,r):
        """Evaluate the predictive probability associated with run length
        *r*. Gives back ONE quantity reflecting the overall probability
        density of the current observation *y* across all spatial
        locations, given run length *r*"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS FOR t==1!
    def initialization(self, y, cp_model, model_prior):
        """Initialize the joint_probabilities before the first iteration!"""
        pass

    #SUPER CLASS IMPLEMENTATION
    #CALLED BY DETECTOR
    def update_joint_log_probabilities(self, y,t, cp_model, model_prior):
        """Is called to update growth- and CP probabilities. That means it is
        called to obtain the joint probability for (y_{1:t}, r_t) for all
        possible run lengths r_t = 0,1,...,t-1.
        If t=1, a special case takes place that requires calling the
        function 'initialization' that gives us the first joint probs.
        The *model_prior* is the prior probability of the model. Unlike
        the np array in the Detector object, the passed argument here is
        only a scalar (!) corresponding to the correct entry of the np array
        in the detector for this particular model.

        NOTE: *joint_probabilities* will have size t at the
        beginning and size t+1 at the end of this function call.

        NOTE: Desirable to also implement SOR/SCR of Fearnhead & Liu (2007)
        in order for this to scale in constant time!
        """


        """STEP 1: evaluate predictive probabilities for all r.
        This means that for a given a model and the current time point,
        we return the evaluations of the pdf at time t conditional on all run
        lengths r=0,1,...t-1,>t-1 for y in log-form.
        """
        #DEBUG: We need to use chopping of run-lengths here, too
        predictive_log_probs = self.evaluate_predictive_log_distribution(y,t)

        """STEP 2: update all no-CP joint probabilities, so-called 'growth
        probabilities' to follow MacKay's naming convention. There will be t
        of them, namely for r=1,2,...,t-1, > t-1, with the last probability
        always being 0 unless we have boundary conditions allowing for
        non-zero probability that we are already in the middle of a segment
        at the first observation."""
        helper_log_probabilities = (self.joint_log_probabilities +
                                   predictive_log_probs)

        """STEP 3: Get CP & growth probabilties at time t by summing
        at each spatial location over all time points and multiplying by
        the hazard rate"""
        #DEBUG: We need to use chopping of run-lengths here, too
        CP_log_prob = scipy.special.logsumexp(helper_log_probabilities +
                                          np.log(cp_model.hazard_vector(1, t)))
        self.joint_log_probabilities = (helper_log_probabilities +
            np.log(1-cp_model.hazard_vector(1, t)))

        """Put together steps 2-3"""
        self.joint_log_probabilities = np.insert(self.joint_log_probabilities,
                                                0, CP_log_prob)

        """STEP 4: Lastly, we always want to get the new evidence for this
        particular model in the model universe, which is the sum of CP and
        growth probabilities"""
        self.model_log_evidence = scipy.special.logsumexp(
                self.joint_log_probabilities )

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS FOR t>1!
    def update_predictive_distributions(self, y, t, r_evaluations):
        """update the distributions giving rise to the predictive probabilities
        in the next step of the algorithm. Could happen computationally (e.g.,
        INLA) or exactly (i.e., with conjugacy)"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS
    def get_posterior_expectation(self, t, r_list=None):
        """get the predicted value/expectation from the current posteriors
        at time point t, for all possible run-lengths"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBCLASS
    def get_posterior_variance(self, t, r_list=None):
        """get the predicted variance from the current posteriors at
        time point t, for all possible run-lengths"""
        pass

    #SHOULD BE IMPLEMENTED IN EACH SUBLCASS
    def prior_log_density(self, y_flat):
        """Computes the log-density of *y_flat* under the prior. Called by
        the associated Detector object to get the next segment-density
        component of the model, denoted D(t,q) for q = NIW"""
        pass


    #CALLS trimmer(kept_run_lengths), which needs to be implemented in each
    #       subclass of probability_model
    def trim_run_length_log_distrbution(self, t, threshold):
        """Trim the log run-length distribution of this model such that any
        run-lengths with probability mass < threshold will be deleted"""

        """If we want to apply thresholding, trim the distributions, see
        STEP 2 and STEP 3. Otherwise, don't do anything in this function
        apart from keeping track of all the run-lengths"""
        if (not ((threshold is None) or (threshold == 0) or (threshold == -1))):


            """STEP 2: Find out which run-lengths we need to trim now"""
            run_length_log_distr = (self.joint_log_probabilities - self.model_log_evidence)
            kept_run_lengths = np.array(run_length_log_distr) >= threshold
            deletions  =  (np.sum(kept_run_lengths) - (t+1)) #

            # print('kept_run_lengths',kept_run_lengths)
            """STEP 3: Drop the quantities associate with dropped run-lengths"""
            if deletions:
                self.trimmer(kept_run_lengths)
