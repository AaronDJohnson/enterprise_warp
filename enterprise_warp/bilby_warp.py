import bilby
import numpy as np

class PTABilbyLikelihood(bilby.Likelihood):
    """
    The class that wraps Enterprise likelihood in Bilby likelihood.

    Parameters
    ----------
    pta: enterprise.signals.signal_base.PTA
      Enterprise PTA object that contains pulsar data and noise models
    parameters: list
      A list of signal parameter names
    """
    def __init__(self, pta, parameters):
        self.pta = pta
        self.parameters = parameters
        self._marginalized_parameters = []

    def log_likelihood(self):
        return self.pta.get_lnlikelihood(self.parameters)

    def get_one_sample(self):
        return {par.name: par.sample() for par in pta[0].params}


class LinearExp(bilby.core.prior.Prior):
    """
    """

    def __init__(self, minimum, maximum, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Uniform prior with bounds
        Parameters
        ----------
        minimum: float
            See superclass
        maximum: float
            See superclass
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(LinearExp, self).__init__(name=name, latex_label=latex_label,
                                      minimum=minimum, maximum=maximum, unit=unit,
                                      boundary=boundary)

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the LinearExp prior.

        Parameters
        ----------
        val: Union[float, int, array_like]
            Uniform probability
        Returns
        -------
        Union[float, array_like]: Rescaled probability
        """
        self.test_valid_for_rescaling(val)
        s_val = self.minimum + val * (self.maximum - self.minimum)
        return ((s_val >= self.minimum) & (s_val <= self.maximum)) * np.log(10) * 10 ** s_val / (10 ** self.maximum - 10 ** self.minimum)

    def prob(self, val):
        """Return the prior probability of val
        Parameters
        ----------
        val: Union[float, int, array_like]
        Returns
        -------
        float: Prior probability of val
        """
        return ((val >= self.minimum) & (val <= self.maximum)) * np.log(10) * 10 ** val / (10 ** self.maximum - 10 ** self.minimum)

    def ln_prob(self, val):
        """Return the log prior probability of val
        Parameters
        ----------
        val: Union[float, int, array_like]
        Returns
        -------
        float: log probability of val
        """
        return ((val >= self.minimum) & (val <= self.maximum)) * np.log(np.log(10) * 10 ** val / (10 ** self.maximum - 10 ** self.minimum))

    def cdf(self, val):
        _cdf = (10**val - 10**self.minimum) / (10**self.maximum - 10**self.minimum)
        _cdf = np.minimum(_cdf, 1)
        _cdf = np.maximum(_cdf, 0)
        return _cdf


def get_bilby_prior_dict(pta):
    """
    Get Bilby parameter dict from Enterprise PTA object.
    Currently only works with uniform priors.

    Parameters
    ----------
    pta: enterprise.signals.signal_base.PTA
      Enterprise PTA object that contains pulsar data and noise models
    """
    priors = dict()
    for param in pta.params:

      if param.size==None:
        if param.type=='uniform':
          #priors[param.name] = bilby.core.prior.Uniform( \
          #    param._pmin, param._pmax, param.name)
          priors[param.name] = bilby.core.prior.Uniform( \
              # param._pmin
              param.prior._defaults['pmin'], param.prior._defaults['pmax'], \
              param.name)
        elif param.type=='normal':
          #priors[param.name] = bilby.core.prior.Normal( \
          #    param._mu, param._sigma, param.name)
          priors[param.name] = bilby.core.prior.Normal( \
              param.prior._defaults['mu'], param.prior._defaults['sigma'], \
              param.name)
        elif param.type=='linearexp':
          #priors[param.name] = bilby.core.prior.Uniform( \
          #    param._pmin, param._pmax, param.name)
          priors[param.name] = LinearExp( \
              # param._pmin
              param.prior._defaults['pmin'], param.prior._defaults['pmax'], \
              name=param.name)
      else:
        if param.name=='jup_orb_elements' and param.type=='uniform':
          for ii in range(param.size):
            priors[param.name+'_'+str(ii)] = bilby.core.prior.Uniform( \
                -0.05, 0.05, param.name+'_'+str(ii))

    # Consistency check
    for key, val in priors.items():
        if key not in pta.param_names:
          print('[!] Warning: Bilby\'s ',key,' is not in PTA params:',\
              pta.param_names)

    return priors 
