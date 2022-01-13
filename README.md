# ACTIVA: Realistic scRNAseq Generation with Automatic Cell-Type identification using Introspective Variational Autoencoders
This Repository contains the package for [ACTIVA (Single Cell generationg with Introspective Variational autoencoders)](https://www.biorxiv.org/content/10.1101/2021.01.28.428725v1).

## Tutorials
Tutorials for using ACTIVA are avaialable [here](https://github.com/SindiLab/Tutorials/tree/main/ACTIVA)


## Data and Pre-Trained Models Availability 

### Original Data
The original data (sparse matrices) is freely available on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5842658.svg)](https://doi.org/10.5281/zenodo.5842658)

 ### Pre- and Post-Processed Data
All of our data can be freely downloaded using the following addresses:

| Data          |                                                URI                                                |
|:-------------:|:-------------------------------------------------------------------------------------------------:|
|  Pre-processed Brain Small  |   s3://activa-material/PreprocessedData/20kBrainSmall_preprocessed.h5                |
|  Pre-processed 68K PBMC  |    s3://activa-material/PreprocessedData/68kPBMC_preprocessed.h5ad                      |
|  Pre-processed NeuroCOVID  |    s3://activa-material/PreprocessedData/NeuroCovid/NeuroCOVID_PreProcessedUsingScGAN_Sparse.h5ad      |
|  Post-processed Brain Small              |   s3://activa-material/PostProcessedData/final_brainsmall_val_int_clust.h5ad|
|  Post-processed 68K PBMC             |    s3://activa-material/PostProcessedData/final_68kpbmc_val_int_clust.h5ad|

### Pre-Trained Models
our pre-trained models can be freely accessed using the following URIs:

| Model          |                                                URI                                              |
|:-------------:|:-------------------------------------------------------------------------------------------------:|
|  Brain Small  |  s3://activa-material/Model-Weights/20K\ Brain\ Small/ACTIVA_BrainSamll.pth            |
|    68K PBMC   |  s3://activa-material/Model-Weights/68K\ PBMC/ACTIVA_68kPBMC.pth            |


### Installing the package:
The code can be run either directly or through a package structure (recommended); that is, you can install `ACTIVA` package locally and just import the needed classes/methods/functions as needed. It is important to note that since ACTIVA uses two homemade packages (ACTINN and SoftAdapt) on GitHub, installing `requirements.txt` in advance is recommended.

#### Step 1: Install Requirements Explicitly

Ensure that you are in the same directory as `requirements.txt`. Then using `pip`, we can install the requirements with:

````bash
pip install -r requirements.txt
````
Although the core requirements are listed directly in `setup.py`, it is good to run this beforehand in case of any dependecy on packages from GitHub. 

#### Step 2: Install Package Locally
Make sure to be in the same directory as `setup.py`. Then, using `pip`, run:

````bash
pip install -e .
````
For step 2, expect a lot of the requirements to be satisfied already (since you installed the requirements in advance).

### Training the model:
You can train the model by adding the appropriate flags on the bash call to python; any arguments that are not explicitly called will resort to the pre-define defaults. Here is an example of running the code on 8 GPU (after installing the package), with explicitely declaring the number of epochs and the learning rates:

````
CUDA_VISIBLE_DEVICE=0,1,2,3,4,5,6,7 python ACTIVA.py --lr 0.0002 --lr_e 0.0002 --lr_g 0.0002 --nEpochs 500

````
and similarly, all other hyperparameters can be passed on explicitly on the call.

***Next Release*** : We will automatically detect number of GPUs and force the model to run on all, unless explictely instructed by user to do otherwise. 

### Fine-tuning the model (transfer learning)
To continue the training an existing network on a new/old dataset, you can explicitely pass the `--pretrain` argument with the path to the last check-point you want to continue from. Here is an example of fine-tuning the model on two GPU from a saved model:

````
CUDA_VISIBLE_DEVICE=0,1 python ACTIVA.py --pretrained 'PATH/TO/CHKPT'  --nEpochs 10

````

***Next Release*** : For now, the data has to be explicitly defined in `ACTIVA.py` script, but in the next release we will add an arg parser flag for passing new datasets.

### Running the model 
`ACTIVA.py`  provides a function called `load_model` which can load in a pretrained network (or checkpoint). After loading in the model, you can use the usual PyTorch convention for inference.

## Citation

Please cite our repository if it was useful for your research:
````
@article {Heydari_ACTIVA,
	author = {Heydari, A. Ali and Davalos, Oscar A. and Zhao, Lihong and Hoyer, Katrina K. and Sindi, Suzanne S.},
	title = {ACTIVA: realistic single-cell RNA-seq generation with automatic cell-type identification using introspective variational autoencoders},
	elocation-id = {2021.01.28.428725},
	year = {2021},
	doi = {10.1101/2021.01.28.428725},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/01/30/2021.01.28.428725},
	eprint = {https://www.biorxiv.org/content/early/2021/01/30/2021.01.28.428725.full.pdf},
	journal = {bioRxiv}
}
````

