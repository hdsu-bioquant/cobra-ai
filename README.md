<img src="https://github.com/user-attachments/assets/33ed7bfe-1192-4e9d-b199-19e423de0010" alt="drawing" width="300"/>

COBRA extends our previously published OntoVAE model (https://doi.org/10.1093/bioinformatics/btad387) with an adversarial approach that allows decoupling of covariates in the latent space. Thus, users can obtain TF or pathway activities that depend on a specific covariate from the interpretable decoder of the model. The code for the original OntoVAE model has undergone some major improvements and is now also hosted in this repository.

## Installation

```
conda create -n cobra python=3.10
conda activate cobra
pip install cobra-ai
```

## Usage

In python, import neccessary modules
```
from cobra_ai.module.ontobj import *
from cobra_ai.module.utils import *
```

For COBRA
```
from cobra_ai.model.cobra import *
```

For OntoVAE
```
from cobra_ai.model.onto_vae import *
```

Examples on how to initialize an ontobj and train a model can be found in the vignette folder. This documentation is still work-in-progress and will be updated soon.

## Citation

If you use COBRA for your research, please cite:
```
Daria Doncevic, Carlos Ramirez Alvarez, Albert Li, Youcheng Zhang, Anna von Bachmann, Kasimir Noack, and Carl Herrmann: Prediction of context-specific regulatory programs and pathways using interpretable deep learning
```


