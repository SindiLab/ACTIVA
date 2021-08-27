from __future__ import print_function

# std libs
import os
import sys 
import copy
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from math import log10

# our libs
from ACTIVA import ACTIVA
from ACTIVA.utils import *

from ACTINN import Classifier, Scanpy_IO, evaluate_classifier
from ACTINN.utils import evaluate_classifier, save_checkpoint_classifier

from SoftAdapt import Adapt, make_args

# reading in single cell data using scanpy
import scanpy as sc


# torch libs
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn.functional as F
## in a future release vv
from tensorboardX import SummaryWriter

# anamoly detection
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

# input data 
parser.add_argument('--data_type', type=str, default="scanpy", help='type of train/test data, default="scanpy"')
parser.add_argument('--data_path', type=str, default="", help="absolute path to where the data is stored")
parser.add_argument('--example_data', type=str, default="covid", help="to run one of the example datasets in our paper")
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')

parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
parser.add_argument("--test_iter", type=int, default=1000, help="Default=1000")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=24)

# introVAE option
parser.add_argument("--zdim", type=int, default=128, help="dim of the latent vector z, Default=128")
parser.add_argument('--lr_e', type=float, default=0.0002, help='learning rate of the encoder, default=0.0002')
parser.add_argument('--lr_g', type=float, default=0.0002, help='learning rate of the generator, default=0.0002')
parser.add_argument("--num_vae", type=int, default=10, help="the epochs of pretraining a VAE, Default=0")

parser.add_argument("--m_plus", type=float, default=150.0, help="the margin in the adversarial part, Default=150.0")

## loss weighting for multi-tasking
parser.add_argument("--softAdapt", default = False, action='store_true', help="Whether to use adaptive weighting, default =False")
# this is the 1/2 in before L_AE and L_CT
parser.add_argument("--weight_neg", type=float, default=0.5, help="Default=0.5")
parser.add_argument("--weight_rec", type=float, default=0.05, help="Default=0.05")
parser.add_argument("--weight_kl", type=float, default=1, help="Default=1.0")

parser.add_argument("--alpha_1", type=float, default=0.5, help="Default=0.5")
parser.add_argument("--alpha_2", type=float, default=0.5, help="Default=0.25")


parser.add_argument('--cpu', default = False , action='store_true', help='enables cpu even when CUDA is available')
parser.add_argument("--nEpochs", type=int, default=200, help="number of training epochs, default = 200 ")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam/AMSGrad. default=0.5')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument('--clip', type=float, default=100, help='the threshod for clipping gradient')
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument('--print_frequency', type=int, default=25, help='frequency of training stats printing for ACTIVA, default=25')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--tensorboard', default=False ,action='store_true', help='enables tensorboard, default True')
parser.add_argument('--outf', default='./withL2-TensorBoard-z128/', help='folder to output training stats for tensorboard')
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")

# classifier options
parser.add_argument('--num_cf', type=int, default=5, help='number of epochs for training the classifer, default =5')
parser.add_argument('--classifierOnly', type=bool, default=False, help='running the classifer only, default = False')
parser.add_argument('--classifierEpochs', type=int, default=10, help='number of epochs to train the classifier, default = 50')
parser.add_argument('--cf_data_type', type=str, default="scanpy", help='type of train/test data, default="scanpy"')
parser.add_argument("--cf_start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('--cf_print_frequency', type=int, default=5, help='frequency of training stats printing for ACTINN, default=5')
parser.add_argument('--cf_lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument("--cf_step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=1000")



def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)

str_to_list = lambda x: [int(xi) for xi in x.split(',')]



def main():
    global opt, model
    opt = parser.parse_args()
    # for SoftAdapt
    make_args();
    
    # determin the device for torch 
    ## if we are allowed to run things on CUDA
    if not opt.cpu and torch.cuda.is_available():
        device = "cuda";
        print('==> Using GPU (CUDA)')
        
    else :
        device = "cpu"
        print('==> Using CPU')
        print('    -> Warning: Using CPUs will yield to slower training time than GPUs')
    

    # if we have h5ad from a scanpy or seurat object 
    if opt.data_type.lower() == "scanpy":
        
        
        
        if opt.example_data == 'pbmc':
            print("PBMC")
            train_data_loader, valid_data_loader = Scanpy_IO('/home/ubuntu/scGAN_ProcessedData/MADE_BY_scGAN/68kPBMCs_7kTest.h5ad',
                                                        test_no_valid = True,
                                                        batchSize=opt.batchSize, 
                                                        workers = opt.workers,
                                                        log = False)
            
            # figure out a way to find the number of classes automatically
            number_of_classes = 10 
            
        elif opt.example_data == '20k brain':
            print("Brain Small")
            # Mouse Brain 20K
            train_data_loader, valid_data_loader = Scanpy_IO('/home/ubuntu/scGAN_ProcessedData/MADE_BY_scGAN/20Kneurons_2KTest.h5',
                                                        batchSize=opt.batchSize, 
                                                        workers = opt.workers,
                                                        log=False)
            # figure out a way to find the number of classes 
            number_of_classes = 8
        
        elif opt.example_data == 'covid':
            print("     -> Reading NeuroCOVID")
            # 78K NeuroCOVID COVID_Data/NeuroCOVID/TrainSplitData/NeroCOVID_preprocessed_splitted.h5ad
            # possibly another one to try: /home/ubuntu/RawData/78KNeuroCOVID_preprocessed_splitted_logged.h5ad'
            train_data_loader, valid_data_loader = Scanpy_IO('/home/ubuntu/COVID_Data/NeuroCOVID/TrainSplitData/NeroCOVID_preprocessed_splitted.h5ad',
                                                        test_no_valid = True,
                                                        batchSize=opt.batchSize, 
                                                        workers = opt.workers,
                                                        log=False,
                                                        verbose = 1)
            
            inp_size = [batch[0].shape[1] for _, batch in enumerate(valid_data_loader, 0)][0];
            labs = [batch[1] for _, batch in enumerate(valid_data_loader, 0)][0];
            # figure out a way to find the number of classes 
            number_of_classes = 9
            print(f"==> Number of classes {number_of_classes}")
            print(f"==> Number of genes {inp_size}")

        # get input output information for the network
        inp_size = [batch[0].shape[1] for _, batch in enumerate(valid_data_loader, 0)][0];
        ## FIX THIS!! write an automatic way to find the labels 
            
            
        print(f"==> Number of classes {number_of_classes}")
        print(f"==> Number of features {inp_size}")
    
    elif opt.data_type.lower() == "csv":
        # if we have CSV turned to h5 (pandas dataframe)
        train_path = "/home/ubuntu/ACTINN_Data/68K_h5/train.h5"
        train_lab_path = "/home/ubuntu/ACTINN_Data/68K_h5/train_lab.csv"

        test_path= "/home/ubuntu/ACTINN_Data/68K_h5/test.h5"
        test_lab_path= "/home/ubuntu/ACTINN_Data/68K_h5/test_lab.csv"

        train_data_loader, valid_data_loader = CSV_IO(train_path, train_lab_path, test_path, test_lab_path,
                                                batchSize=opt.batchSize,
                                                workers = opt.workers)

        # get input output information for the network
        inp_size = [batch[0].shape[1] for _, batch in enumerate(train_data_loader, 0)][0];
        number_of_classes = 10
        print(f"==> Number of classes {number_of_classes}")
        print(f"==> Number of features {inp_size}")



    else:
        raise ValueError("Wrong data type, please provide Scanpy/Seurat object or h5 dataframe")
    
  
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=opt.outf)
    
    start_time = time.time()
    cur_iter = 0
    
    
    """ 
    
    Building the generative model:
    
    """
    # threshold here can be used to cut off a certain values in the count matrix
    ## example:
    ### threshold = np.min(adata[np.nonzero(adata)])
    ##### we go with threshold of 0, i.e. no values less than 0 will be in the output

    model = ACTIVA(latent_dim=opt.zdim, input_size=inp_size, threshold=0).to(device)

    if opt.pretrained:
        print(f"==> Loading pre-trained model from {opt.pretrained}")
        load_model(model, opt.pretrained)
        print("    -> Loaded from a pre-trained model:")
        
    print(model)
            
    optimizerE = optim.Adam(model.encoder.parameters(), lr=opt.lr_e)
    
    optimizerG = optim.Adam(model.decoder.parameters(), lr=opt.lr_g)
    

    """ 
    
    Building the classifier model:
    
    """
    cf_model = Classifier(output_dim = number_of_classes, input_size = inp_size).to(device)
    cf_criterion = torch.nn.CrossEntropyLoss()

    cf_optimizer = torch.optim.Adam(params=cf_model.parameters(), 
                                    lr=opt.cf_lr, 
                                    betas=(0.9, 0.999), 
                                    eps=1e-08, 
                                    weight_decay=0.005, 
                                    amsgrad=False)
    cf_decayRate = 0.95
    cf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=cf_optimizer, gamma=cf_decayRate)
    print("\n Classifier Model \n")
    print(cf_model)
    
    """
    
    Training as the classifier (Should be done when we are warm-starting the VAE part)
    
    """
    
    def train_classifier(cf_epoch, iteration, batch, cur_iter):  
        
        cf_optimizer.zero_grad()      
        if len(batch[0].size()) == 3:
            batch = batch[0].unsqueeze(0)
        else:
            labels = batch[1]
            batch = batch[0]
        batch_size = batch.size(0)
                       
        features= Variable(batch).to(device)
        true_labels = Variable(labels).to(device)
                
        info = f"\n====> Classifier Cur_iter: [{cur_iter}]: Epoch[{cf_epoch}]({iteration}/{len(train_data_loader)}): time: {time.time()-start_time:4.4f}: "
                    
        # ---------- Update the classifier ----------                  
        pred_cluster = cf_model(features) 
        loss = cf_criterion(pred_cluster.squeeze(), true_labels)
        loss.backward()  
        cf_optimizer.step()
        
        # decaying the LR 
        if cur_iter % opt.cf_step == 0 and cur_iter != 0:
            cf_lr_scheduler.step() 
            for param_group in cf_optimizer.param_groups:
                print(f"    -> Decayed lr to -> {param_group['lr']}")

        # ---------- Printing Network Information ----------    
        info += f'Loss: {loss.data.item():.4f} ' 
        loss_info = '[loss]'
        if cur_iter == 0:
            print("    -> Classifier Initial stats:", info)

        if cur_iter % opt.test_iter == 0:  
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
                
        if epoch % opt.cf_print_frequency == 0 and iteration == (len(train_data_loader) - 1) :
            print(info)

            
    
    

    """
    
    Training as a VAE (similar to warm start)
    
    """
    
    def train_vae(epoch, iteration, batch, cur_iter):  
        
        if len(batch[0].size()) == 3:
            batch = batch[0].unsqueeze(0)
        else:
            batch = batch[0];
            
        batch_size = batch.size(0)
                       
        real= Variable(batch).to(device)
                
        info = f"\n====> VAE Cur_iter: [{cur_iter}]: Epoch[{epoch}]({iteration}/{len(train_data_loader)}): time: {time.time()-start_time:4.4f}: "
        
        loss_info = '[loss_rec, loss_kl]'
            
        #---------- Updating the VAE  ----------                 
        real_mu, real_logvar, z, rec = model(real) 
        
        loss_rec =  model.reconstruction_loss(rec, real, True)        
        loss_kl = model.kl_loss(real_mu, real_logvar).mean()
        loss = loss_rec + loss_kl
        
        optimizerG.zero_grad()
        optimizerE.zero_grad()       
        loss.backward()                   
        optimizerE.step() 
        optimizerG.step()
    
        info += f'Rec: {loss_rec.data.item():.4f}, KL: {loss_kl.data.item():.4f},' 
        
        if cur_iter == 0:
            print("    -> VAE Initial stats:", info)

        if epoch % opt.cf_print_frequency == 0 and iteration == (len(train_data_loader) - 1) :
            print(info)
    
        if cur_iter % opt.test_iter == 0:  
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter)
    
    """
    
    Training the IntroVAE part
    
    """
    
    def train(epoch, iteration, batch, cur_iter):      
        
        
        if len(batch[0].size()) == 3:
            batch = batch[0].unsqueeze(0)
        else:
            batch = batch[0];
            
        #------- CONDITIONING STEP -------
        
        cf_model.eval()
        real_classification = cf_model(batch.to(device))
        
        #---------------------------------
        
        batch_size = batch.size(0)
        noise = Variable(torch.zeros(batch_size, opt.zdim).normal_(0, 1)).to(device)
        real= Variable(batch).to(device)
        
        info = f"\n====> Cur_iter: [{cur_iter}]: Epoch[{epoch}]({iteration}/{len(train_data_loader)}): time: {time.time()-start_time:4.4f}: "
        
        loss_info = '[loss_classification, loss_rec, loss_margin, lossE_real_kl, lossE_rec_kl, lossE_fake_kl, lossG_rec_kl, lossG_fake_kl,]'
            
    
        #---------- Update E ----------
        fake = model.sample(noise)            
        real_mu, real_logvar, z, rec = model(real);
        
        
        #------- CONDITIONING STEP -------
        
        rec_classification = cf_model(rec)
        loss_classification = model.classification_loss(rec_classification,
                                                       real_classification)
        #---------------------------------
        
        
        rec_mu, rec_logvar = model.encode(rec.detach())
        fake_mu, fake_logvar = model.encode(fake.detach())
        loss_rec =  model.reconstruction_loss(rec, real, True)
        
        lossE_real_kl = model.kl_loss(real_mu, real_logvar).mean()
        lossE_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossE_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()            
        loss_margin = lossE_real_kl + \
                      (F.relu(opt.m_plus-lossE_rec_kl) + \
                      F.relu(opt.m_plus-lossE_fake_kl)) * 0.5 * opt.weight_neg
        
        #------- Loss Weighting (Balancing Importance) -------
        
        if opt.softAdapt: 
            
            if cur_iter % len(train_data_loader) == 0 and cur_iter != 0: 
                # store the most recent loss values
                recent_loss_tensor = [loss_classification + loss_rec, loss_margin]
                # pass to softadapt 
                opt.alpha_1, opt.alpha_2 = Adapt(recent_loss_tensor, which_SA="loss-weighted", beta=0.1)
                
                print(f"alpha1:{opt.alpha_1}, alpha2:{opt.alpha_2}");
                
        else:
            opt.alpha_1 = opt.weight_kl
            opt.alpha_2 = opt.weight_rec
         #---------------------------------
        
        lossE = opt.alpha_1 * loss_margin + opt.alpha_2 * (loss_classification + loss_rec) 

        # this could also be for done right begore G backprop
        optimizerG.zero_grad()
        optimizerE.zero_grad()       
        # since we will be calling backwards on this a second time
        lossE.backward(retain_graph=True)
        
        """ 
        # based on our experiments, no clipping is needed
        ## but just in case someone wants to apply it
        nn.utils.clip_grad_norm_(model.encoder.parameters(), opt.clip) 
        
        """ 
        optimizerE.step()
        
        #---------- Update G----------
        
        rec_mu, rec_logvar = model.encode(rec.detach())
        fake_mu, fake_logvar = model.encode(fake.detach())
        
        lossG_rec_kl = model.kl_loss(rec_mu, rec_logvar).mean()
        lossG_fake_kl = model.kl_loss(fake_mu, fake_logvar).mean()
        lossG = opt.alpha_1 * 0.5 * (lossG_rec_kl + lossG_fake_kl)
        
        lossG.backward()
        
        """ 
        # based on our experiments, no clipping is needed
        ## but just in case someone wants to apply it
        nn.utils.clip_grad_norm_(model.encoder.parameters(), opt.clip) 
        """
        
        optimizerG.step()
    
        
        info += f'Rec: {loss_rec.data.item():.4f}, '
        info += f'Kl_E: {lossE_real_kl.item():.4f}, {lossE_rec_kl.item():.4f}, {lossE_fake_kl.item():.4f}, '
        info += f'Kl_G: {lossG_rec_kl.item():.4f}, {lossG_fake_kl.item():.4f}, '
       
        print(info)        
        print(f"    -> Classification loss: {loss_classification.item():.4f}")
        
        if cur_iter % opt.test_iter == 0:            
            if opt.tensorboard:
                record_scalar(writer, eval(loss_info), loss_info, cur_iter) 
    
    
    # if we are trying to just train the classifier
    if opt.classifierOnly:
          # TRAIN     
        print("---------------- ")
        print("==> Trainig Classifier ONLY ")
        print(f"    -> lr decaying after every {opt.cf_step} steos")
        print(f"    -> Training stats printed after every {opt.cf_print_frequency} epochs")
        for epoch in tqdm(range(0, opt.classifierEpochs + 1), desc = "Classifier Only"): 
            #save models
            if epoch % opt.cf_print_frequency == 0 and epoch != 0:
                evaluate_classifier(valid_data_loader, cf_model)
                save_epoch = (epoch//opt.save_iter)*opt.save_iter   
                save_checkpoint_classifier(cf_model, save_epoch, 0, '')

            cf_model.train()
            for iteration, batch in enumerate(train_data_loader, 0):
                    #---------- train Classifier Only ----------
                    train_classifier(epoch, iteration, batch, cur_iter);
                    cur_iter += 1

        save_epoch = (epoch//opt.save_iter)*opt.save_iter    

        save_checkpoint_classifier(cf_model, save_epoch, 0, 'LAST')
        print("==> Final evaluation on validation data: ")
        evaluate_classifier(valid_data_loader, cf_model)
        print(f"==> Total training time {time.time() - start_time}");   
        
        sys.exit("==> Classifier Only Training Done") 
    
    
    #----------------Train by epochs--------------------------
    for epoch in tqdm(range(opt.start_epoch, opt.nEpochs + 1), desc="ACTIVA Training"): 
        
        #save the variational model
        if epoch % opt.print_frequency == 0 :
            save_epoch = (epoch//opt.save_iter)*opt.save_iter  
            # save both the IntroVAE and conditioner part of ACTIVA
#             save_checkpoint(model, save_epoch, 0, opt.m_plus, f'{opt.example_data}-', classifier_model=cf_model)
            save_checkpoint(model, save_epoch, 0, opt.m_plus, f'{opt.example_data}-', classifier_model=cf_model)
            
        # save the classifier model 
        if epoch % opt.cf_print_frequency == 0 and epoch != 0:
            evaluate_classifier(valid_data_loader, cf_model)
            save_epoch = (epoch//opt.save_iter)*opt.save_iter  

        model.train()
        for iteration, batch in enumerate(train_data_loader, 0):
            #--------------train------------
            if epoch < opt.num_vae or epoch < opt.num_cf:
                
                if epoch < opt.num_cf:
                    train_classifier(epoch, iteration, batch, cur_iter)
                        
                if epoch < opt.num_vae:
                    train_vae(epoch, iteration, batch, cur_iter)
            else:
                train(epoch, iteration, batch, cur_iter)
            
            cur_iter += 1
            
    print(f"==> Total training time {time.time() - start_time}");
            
        

        
        
if __name__ == "__main__":
    main()   
    
    
    
