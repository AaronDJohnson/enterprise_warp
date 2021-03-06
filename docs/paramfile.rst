====================
Parameter file usage
====================

To run Enterprise from a command line using a parameter file on a first pair of .par-.tim files in your data directory, please go to :code:`examples/` and run:

.. code-block:: console

   $ python run_example_paramfile.py --prfile example_paramfile.dat --num 0

Parameter file options
----------------------
- **{x} (a number in curly brackets)**: a separator, indicating that following parameters are only for model 'x'. If we specify more than one model and choose ptmcmc sampeler, Enterprise is launched in model comparison mode using the product-space method and :code:`class HyperModel` from `enterprise_extensions <https://github.com/stevertaylor/enterprise_extensions/>`__.
- **paramfile_label**: a unique label for the output directory, associated with the given parameter file. The label inside a noise model file(s) is (are) also added to the output directory name.
- **datadir**: directory with .par and .tim files
- **out**: output directory with Enterprise/Bilby results
- **overwrite**: option to overwrite overwrite an old Enterprise output
- **array_analysis**: whether to run analysis on a pulsar timing array, or on a single pulsar (True for array, False for single pulsars)
- **noisefiles**: path to .json noise files needed to fix white noise parameters
- **sampler**: choose ptmcmcsampler or any of the samplers compatible with Bilby

Parameter file also automatically recognizes:
- Priors. Default parameters of prior distributions are set in :code:`ModelParams` class or its child class where you specify your custom noise models.
- Sampler keyword arguments. I.e., dlogz. They should only be specified after the sampler.

