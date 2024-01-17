# 1. Typing Classes ------------------------------------------------------------------------------
# Need to tag parameters as tunable
# Copied from: https://github.com/scverse/scvi-tools/blob/71bbc2004822337281fb085339715660e2334def/scvi/autotune/_types.py

from inspect import isfunction
from typing import Any, List

from scvi._decorators import classproperty


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

    def __init__(self, model_cls):

        self._model_cls = model_cls

    def get_tunables(self):
        ''' Returns dictionary of tunables, stating the tunable type, default value, annotation and the source.
            source: scvi.autotune.TunerManager._get_registry._get_tunables'''
        
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

        ''' Returns dictionary of metrics, stating the name of the metrics and the mode.
            source: scvi.autotune.TunerManager._get_registry._get_metrics'''
        
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

        ''' Provides all information about the tunable parameters and the metrics. 
            source: scvi.autotune.ModelTuner.info but mainly uses scvi.autotune.TunerManager._view_registry'''
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
                      epochs,
                      resources,
                      ):

        def trainable(
                search_space: dict,
                *,
                model_cls,
                adata,
                ontobj,
                max_epochs: int,
                    ):
            ''' This is a function, that can be wrapped by tune.with_parameters, which in turn is consumable by tune.Tuner
                source: scvi.autotune.TunerManager._get_trainable._trainable'''
            
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

            utils.setup_anndata_ontovae(adata, ontobj)
                    
            # Creating a scOntoVAE model with the given adata and default values except for the tunable ones given by model_kwargs
            model = model_cls(adata, **model_kwargs)
            
            # still need rest of train parameters, use default values except for the trainable parameters, which are given by search_space
            model.train_model(save = False, epochs = max_epochs, **train_kwargs)

        wrap_params = tune.with_parameters(
            trainable,
            model_cls = self._model_cls,
            adata = adata,
            ontobj = ontobj,
            max_epochs = epochs,
        )
        return tune.with_resources(wrap_params, resources = resources)


    def fit(self, 
            adata, 
            ontobj,
            search_space,
            epochs = 10,
            metric = "ontovae_loss",
            scheduler = "asha",
            num_samples = 10,
            searcher = "hyperopt",
            resources = {}):
        ''' Run a specified hyperparameter sweep for the asspciated model class'''

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

        trainable = self.get_trainable(adata, ontobj, epochs, resources)
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
