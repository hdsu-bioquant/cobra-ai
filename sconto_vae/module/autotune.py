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
                      ):
        """Returns a trainable function consumable by :class:`~ray.tune.Tuner`."""

        def trainable(
                search_space: dict,
                *,
                model_cls,
                adata,
                ontobj,
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

            utils.setup_anndata_ontovae(adata, ontobj, cpa_keys = cpa_keys)
                    
            # Creating a scOntoVAE model with the given adata and default values except for the tunable ones given by model_kwargs
            model = model_cls(adata, **model_kwargs)
            
            # still need rest of train parameters, use default values except for the trainable parameters, which are given by search_space
            model.train_model(modelpath = "", save = False, epochs = max_epochs, **train_kwargs)


        wrap_params = tune.with_parameters(
            trainable,
            model_cls = self._model_cls,
            adata = adata,
            ontobj = ontobj,
            cpa_keys = cpa_keys,
            max_epochs = epochs,
        )
        return tune.with_resources(wrap_params, resources = resources)


    def fit(self, 
            adata, 
            ontobj,
            search_space,
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
        param_keys = list(search_space.keys())
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

        trainable = self.get_trainable(adata, ontobj, cpa_keys, epochs, resources)
        tuner = tune.Tuner(
            trainable = trainable,
            param_space = search_space,
            tune_config = tune_config,
            run_config = run_config,
        )
        config = {
            "metrics": metric,
            "search_space": search_space
        }

        results = tuner.fit()
        return results, config # to later get a good output of the result plus configurations        
