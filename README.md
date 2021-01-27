# ACTIVA: Realistic scRNAseq Generation with Automatic Cell-Type identification using Introspective Variational Autoencoders
This Repository contains the package for ACTIVA (Single Cell generationg with Introspective Variational autoencoders).


## Data and Pre-Trained Models Availability 
All of our data can be freely downloaded using the following addresses:

| Data          |                                                URL/URI                                                |
|:-------------:|:-------------------------------------------------------------------------------------------------:|
|  Raw Brain Small  |       https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons       |
|  Raw 68K PBMC   | https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a |
|  Pre-pocessed Brain Small  |   s3://activa-material/PreprocessedData/20kBrainSmall_preprocessed.h5                |
|  Pre-pocessed 68K PBMC  |    s3://activa-material/PreprocessedData/68kPBMC_preprocessed.h5ad                      |
|  Post-pocessed Brain Small              |   s3://activa-material/PostProcessedData/final_brainsmall_val_int_clust.h5ad|
|  Post-pocessed 68K PBMC             |    s3://activa-material/PostProcessedData/final_68kpbmc_val_int_clust.h5ad|

and our pre-trained models can be freely accessed using the following URIs:



| Model          |                                                URI                                              |
|:-------------:|:-------------------------------------------------------------------------------------------------:|
|  Brain Small  |  s3://activa-material/Model-Weights/68K\ PBMC/ACTIVA_BrainSamll.pth            |
|    68K PBMC   |  s3://activa-material/Model-Weights/68K\ PBMC/ACTIVA_68kPBMC.pth            |


### Installing the package:
The code can be run either directly or through a package model; that is, you can install `ACTIVA` package locally and just import. This can be done with `pip`:

````
# assuming you are in the same directory as setup.py
pip install -e PATH/TO/FOLDER/WITH-setup.py
````

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
(link to pre-print coming soon)
```
@misc{Heydari2021,
  author = {Heydari, A. Ali},
  title = {ACTIVA: realistic single-cell RNA-seq generation with automatic cell-type identification using Introspective Variational Autoencoder},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SindiLab/ACTIVA}},
}
