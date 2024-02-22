# 1. Typing Classes ------------------------------------------------------------------------------
# Need to tag parameters as tunable
# Copied from: https://github.com/scverse/scvi-tools/blob/71bbc2004822337281fb085339715660e2334def/scvi/autotune/_types.py

from inspect import isfunction
from typing import Any, List

from sconto_vae.module.decorators import classproperty


class TunableMeta(type):
    """Metaclass for Tunable class."""

    def __getitem__(cls, values):
        if not isinstance(values, tuple):
            values = (values,)
        return type("Tunable_", (Tunable,), {"__args__": values})


class Tunable(metaclass=TunableMeta):
    """Typing class for tagging keyword arguments as tunable."""


class TunableMixin:
    """Mixin class for exposing tunable attributes."""

    @classproperty
    def _tunables(cls) -> List[Any]:
        """Returns the tunable attributes of the model class."""
        _tunables = []
        for attr_key in dir(cls):
            if attr_key == "_tunables":
                # Don't recurse
                continue
            attr = getattr(cls, attr_key)
            if hasattr(attr, "_tunables") or isfunction(attr):
                _tunables.append(attr)
        return _tunables
    
# 2. ModelTuner -----------------------------------------------------------------
# This is the class that is responsible for the autotuning
# The first main function is info(), which provides all information about the tunable parameters
# and the metrics that can be used to evaluate the models performance. This function is a union
# of the implementations of ModelTuner and TunerManager of scvi-tools. It is kept a little simpler
# without the fancy console output tables.
# The second main function is fit(), which uses the python package ray to perform the autotuning.
# This function also is a union of the implementations of ModelTuner and TunerManager of scvi-tools.
    
from inspect import signature, Parameter
from ray import tune, air
from datetime import datetime
import os
import warnings
from torch import optim
import torch.nn as nn

import sconto_vae.module.utils as utils

class ModelTuner:
    """
    This class does automated hyperparameter tuning for scOntoVAE classes.

    Parameters
    ----------
    model_cls
        A model class (at the moment only sconto_vae_tunable) on which to tune hyperparameters.
        Must have a class property '_tunables' that defines tunable elements, a class property '_metrics' that
        defines the metrics that can be used, and the metric needs to be reported in the training function.

    Examples
    --------
    >>> import scanpy as sc
    >>> from sconto_vae.module import utils
    >>> from sconto_vae.model import sconto_vae as onto
    >>> ontobj = Ontobj()
    >>> ontobj.load(path_to_onto_object)
    >>> adata = sc.read_h5ad(path_to_h5ad)
    >>> adata = utils.setup_anndata_ontovae(adata, ontobj)
    >>> tuner = ModelTuner(onto.scOntoVAE)
    >>> tuner.info()
    >>> search_space = {"drop_enc": tune.choice([0.2, 0.4]), "lr": tune.loguniform(1e-4, 1e-2)}
    >>> results = tuner.fit(adata, ontobj, search_space, resources = {'GPU': 1.0, 'CPU': 4.0})
    """

    def __init__(self, model_cls):

        self._model_cls = model_cls

    def get_tunables(self):
        ''' Returns dictionary of tunables, stating the tunable type, default value, annotation and the source.'''
        
        # The following for loop will provide all tunable parameters of the model class. 
        tunables = {}
        for child in getattr(self._model_cls, "_tunables", []):
            for param, metadata in signature(child).parameters.items():

                if not isinstance(metadata.annotation, TunableMeta):
                        continue
           
                default_val = None
                if metadata.default is not Parameter.empty:
                    default_val = metadata.default

                annotation = metadata.annotation.__args__[0]
                if hasattr(annotation, "__args__"):
                    annotation = annotation.__args
                else:
                    annotation = annotation.__name__

                if child.__name__ == "__init__":
                    tunable_type = "model"
                elif "train" in child.__name__:
                    tunable_type = "train"
                else:
                    tunable_type = None

                tunables[param] = {
                    "tunable_type": tunable_type,
                    "default_value": default_val,
                    "annotation": annotation,
                    "source": self._model_cls.__name__,
                }
        
        return tunables

    def get_metric(self):
        ''' Returns dictionary of metrics, stating the name of the metrics and the mode.'''
        
        metrics = {}

        # The following loop provides all metrics added in the model class
        for child in getattr(self._model_cls, "_metrics", []):
            metrics[child] = {
                "metric": child,
                "mode": "min",
            }
        # Don't like this implementation, because I am giving the mode "min" to every loss metric, here I follow
        # scvi-tools implementation that only specifies the metric in the _metrics funciton in the tunable model
        # probably change this implementation later
        # Also maybe add validation loss in tunable model
        return metrics

    def info(self) -> None:
        ''' Provides all information about the tunable parameters and the metrics.'''

        print(f"ModelTuner registry for {self._model_cls.__name__}")

        tunables = self.get_tunables()
        print()
        print("Tunable Hyperparameters and their default value")
        for key in tunables.keys():
            print(f"{key}: {tunables[key]['default_value']}")

        metrics = self.get_metric()
        print()
        print("Available Metrics and their mode")
        for key in metrics.keys():
            print(f"{key}: {metrics[key]['mode']}")
        
    def get_trainable(self,
                      adata,
                      ontobj,
                      cpa_keys,
                      epochs,
                      resources,
                      top_thresh: int=1000,
                      bottom_thresh: int=30
                      ):
        """Returns a trainable function consumable by :class:`~ray.tune.Tuner`."""

        def trainable(
                search_space: dict,
                *,
                model_cls,
                adata,
                ontobj,
                top_thresh,
                bottom_thresh,
                cpa_keys,
                max_epochs: int,
                    ):
            ''' This is a function, that can be wrapped by tune.with_parameters.'''
            
            # Parse the compact search space into separate kwards dictionaries
            # source: scvi.autotune.TunerManager._get_search_space
            model_kwargs = {}
            train_kwargs = {}
            tunables = self.get_tunables()
            for param, value in search_space.items():
                type = tunables[param]["tunable_type"]
                if type == "model":
                    model_kwargs[param] = value
                elif type == "train":
                    train_kwargs[param] = value

            utils.setup_anndata_ontovae(adata, ontobj, top_thresh=top_thresh, bottom_thresh=bottom_thresh, cpa_keys = cpa_keys)
                    
            # Creating a scOntoVAE model with the given adata and default values except for the tunable ones given by model_kwargs
            model = model_cls(adata, **model_kwargs)
            
            # still need rest of train parameters, use default values except for the trainable parameters, which are given by search_space
            model.train_model(modelpath = "", save = False, epochs = max_epochs, **train_kwargs)


        wrap_params = tune.with_parameters(
            trainable,
            model_cls = self._model_cls,
            adata = adata,
            ontobj = ontobj,
            top_thresh=top_thresh,
            bottom_thresh=bottom_thresh,
            cpa_keys = cpa_keys,
            max_epochs = epochs,
        )
        return tune.with_resources(wrap_params, resources = resources)

    def default_search_space(self, search_space = [], use_defaults = []):
        ''' Check if the parameters of the user defined search space are tunable, and if desired take default
        search space values. (User specified search spaces are prioritized.)

        Parameters
        ----------
        search_space:
            Dictionary of hyperparameter names and their respective search spaces
            provided as instantiated Ray Tune sample functions. Available
            hyperparameters can be viewed with :meth:`~scvi.autotune.ModelTuner.info`.
            The default is an empty list, in case that the hyperparameters to be tuned
            should all get the search space from the DEFAULT space.
        use_defaults:
            A list of strings, containing the names of the hyperparameters that are
            supposed to be tuned with the DEFAULTS search space.

        Returns
        -------
            The search space composed of the user specified search space and the DEFAULTS
            search space of the user specified hyperparameters.
        '''
        for param in search_space:
            if param in self.get_tunables():
                continue
            raise ValueError(
                f"Provided parameter `{param}` is invalid for {self._model_cls.__name__}."
                " Please see available parameters with `ModelTuner.info()`. "
            )
        for param in use_defaults:
            if param in self.get_tunables():
                continue
            warnings.warn(
                f"Provided parameter `{param}` is invalid for {self._model_cls.__name__}."
                " Please see available parameters with `ModelTuner.info()`. ",
                UserWarning
            )
        
        _search_space = {}
        if use_defaults != None:
            defaults = DEFAULTS.get(self._model_cls.__name__, {})
            for param, metadata in defaults.items():
                if param in use_defaults:
                    sample_fn = getattr(tune, metadata["fn"])
                    fn_args = metadata.get("args", [])
                    fn_kwargs = metadata.get("kwargs", {})
                    _search_space[param] = sample_fn(*fn_args, **fn_kwargs)

        _search_space.update(search_space)
        return _search_space

    def fit(self, 
            adata, 
            ontobj,
            top_thresh: int=1000,
            bottom_thresh: int=30,
            search_space = [],
            use_defaults = [],
            epochs = 10,
            cpa_keys = None,
            metric = "validation_loss",
            scheduler = "asha",
            num_samples = 10,
            searcher = "hyperopt",
            resources = {}):
        ''' Run a specified hyperparameter sweep for the associated model class.
        
        Parameters
        ----------
        adata:
            anndata object that has been preprocessed with setup_anndata function.
        ontobj:
            ontobj object that has been preprocessed with setup_anndata function.
        search_space:
            Dictionary of hyperparameter names and their respective search spaces
            provided as instantiated Ray Tune sample functions. Available
            hyperparameters can be viewed with :meth:`~scvi.autotune.ModelTuner.info`.
        use_defaults:
            A list of strings, containing the names of the hyperparameters that are
            supposed to be tuned with the DEFAULTS search space.
        epochs:
            Number of epochs to train each model configuration.
        cpa_keys:
            Observations to use for disentanglement of latent space (only for OntoVAE + cpa).
        metric:
            One of the metrics that is available for the underlying model class (check ModelTuner.info()).
            This metric is used to evaluate the quality of the values for hyperparameters that are tuned.
        scheduler:
            Ray Tune scheduler to use. One of the following:

            * ``"asha"``: :class:`~ray.tune.schedulers.AsyncHyperBandScheduler` (default)
            * ``"hyperband"``: :class:`~ray.tune.schedulers.HyperBandScheduler`
            * ``"median"``: :class:`~ray.tune.schedulers.MedianStoppingRule`
            * ``"pbt"``: :class:`~ray.tune.schedulers.PopulationBasedTraining`
            * ``"fifo"``: :class:`~ray.tune.schedulers.FIFOScheduler`

            Note that that not all schedulers are compatible with all search algorithms.
            See Ray Tune `documentation <https://docs.ray.io/en/latest/tune/key-concepts.html#schedulers>`_
            for more details.
        num_samples:
            Number of hyperparameter configurations to sample
        searcher:
            Ray Tune search algorithm to use. One of the following:

            * ``"hyperopt"``: :class:`~ray.tune.hyperopt.HyperOptSearch` (default)
            * ``"random"``: :class:`~ray.tune.search.basic_variant.BasicVariantGenerator`
        resources:
            Dictionary of maximum resources to allocate for the experiment. Available
            keys include:

            * ``"cpu"``: number of CPU threads
            * ``"gpu"``: number of GPUs
            * ``"memory"``: amount of memory

        Returns
        -------
            A tuple containing the results of the hyperparameter tuning and the configurations.
        '''

        if scheduler == "asha":
            _default_kwargs = {
                "max_t": 100,
                "grace_period": 1,
                "reduction_factor": 2,
            }
            _scheduler = tune.schedulers.AsyncHyperBandScheduler

        tune_config = tune.tune_config.TuneConfig(
            metric = metric,
            mode = "min",
            scheduler = _scheduler(**_default_kwargs),
            search_alg = searcher,
            num_samples = num_samples,
        )        

        experiment_name = "tune_" + self._model_cls.__name__.lower() + "_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        logging_dir = os.path.join(os.getcwd(), "ray")
        metrics = self.get_metric()
        _search_space = self.default_search_space(search_space, use_defaults)
        param_keys = list(_search_space.keys())
        kwargs = {"metric_columns": list(metrics.keys()),
                  "parameter_columns": param_keys,
                  "metric": metric,
                  "mode": metrics[metric]["mode"],
                  }
        reporter = tune.CLIReporter(**kwargs)
        run_config = air.config.RunConfig(
            name = experiment_name,
            local_dir = logging_dir,
            progress_reporter = reporter, 
            log_to_file = True,
            verbose = 1,
        )

        trainable = self.get_trainable(adata, ontobj, top_thresh, bottom_thresh, cpa_keys, epochs, resources)
        tuner = tune.Tuner(
            trainable = trainable,
            param_space = _search_space,
            tune_config = tune_config,
            run_config = run_config,
        )
        config = {
            "metrics": metric,
            "search_space": _search_space
        }

        results = tuner.fit()
        self.output(results, config)
        return results, config # to later get a good output of the result plus configurations  

    def output(self, results, config):

        print('Best result with', config['metrics'])

        result = results.get_best_result()
        params = result.config
        loss = result.metrics[config['metrics']]

        print('Parameter settings:', params)
        print('Loss:', loss)
            

DEFAULTS = {
    'scOntoVAE': {
        "use_batch_norm_enc": {"fn": "choice", "args": [[True, False]]},
        "use_layer_norm_enc": {"fn": "choice", "args": [[True, False]]},
        "use_activation_enc": {"fn": "choice", "args": [[True, False]]},
        "activation_fn_enc": {"fn": "choice", "args": [[nn.ReLU]]},
        "bias_enc": {"fn": "choice", "args": [[True, False]]},
        "hidden_layers_enc": {"fn": "choice", "args": [[1, 2, 4, 8]]},
        "inject_covariates_enc": {"fn": "choice", "args": [[True, False]]},
        "drop_enc": {"fn": "uniform", "args": [0.1, 0.4]},
        "z_drop": {"fn": "uniform", "args": [0.5, 0.8]},
        "root_layer_latent": {"fn": "choice", "args": [[True, False]]},
        "neuronnum": {"fn": "choice", "args": [[3, 4, 5]]},
        "use_batch_norm_dec": {"fn": "choice", "args": [[True, False]]},
        "use_layer_norm_dec": {"fn": "choice", "args": [[True, False]]},
        "use_activation_dec": {"fn": "choice", "args": [[True, False]]},
        "use_activation_lat": {"fn": "choice", "args": [[True, False]]},
        "activation_fn_dec": {"fn": "choice", "args": [[nn.Tanh]]},
        "bias_dec": {"fn": "choice", "args": [[True, False]]},
        "inject_covariates_dec": {"fn": "choice", "args": [[True, False]]},
        "drop_dec": {"fn": "uniform", "args": [0, 0.4]},
        "lr": {"fn": "loguniform", "args": [1e-4, 1e-2]},
        "kl_coeff": {"fn": "loguniform", "args": [1e-4, 1e-2]},
        "batch_size": {"fn": "choice", "args": [[32, 64, 128, 256]]},
        "pos_weights": {"fn": "choice", "args": [[True, False]]},
        "optimizer": {"fn": "choice", "args": [[optim.AdamW, optim.SGD]]}
    },
    'OntoVAEcpa': {
        "hidden_layers_class": {"fn": "choice", "args": [[1, 2, 4, 8]]},
        "neurons_per_class_layer": {"fn": "choice", "args": [[16, 32, 46, 128]]},
        "use_batch_norm_class": {"fn": "choice", "args": [[True, False]]},
        "use_layer_norm_class": {"fn": "choice", "args": [[True, False]]},
        "use_activation_class": {"fn": "choice", "args": [[True, False]]},
        "activation_fn_class": {"fn": "choice", "args": [[nn.ReLU]]},
        "bias_class": {"fn": "choice", "args": [[True, False]]},
        "inject_covariates_class": {"fn": "choice", "args": [[True, False]]},
        "drop_class": {"fn": "uniform", "args": [0.1, 0.4]},
        "average_neurons": {"fn": "choice", "args": [[True, False]]},
        "lr_vae": {"fn": "loguniform", "args": [1e-4, 1e-2]},
        "lr_adv": {"fn": "loguniform", "args": [1e-4, 1e-2]},
        "kl_coeff": {"fn": "loguniform", "args": [1e-4, 1e-2]},
        "adv_coeff": {"fn": "loguniform", "args": [1e2, 1e4]},
        "pen_coeff": {"fn": "uniform", "args": [0, 4]},
        "mixup_lambda": {"fn": "choice", "args": [[0.0, 1.0, 2.0]]},
        "adv_step": {"fn": "choice", "args": [[0, 1, 2]]},
        "batch_size": {"fn": "choice", "args": [[32, 64, 128, 256]]},
        "pos_weights": {"fn": "choice", "args": [[True, False]]},
        "optimizer": {"fn": "choice", "args": [[optim.AdamW, optim.SGD]]}
    }
}