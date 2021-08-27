# std libs
import os
import time 

# torch libs
import torch



def load_model(model, pretrained, classifier_model:bool=False):
        """
        Utility function for saving ACTIVA model, including the IntroVAE and Classifier part
        INPUT:
            model               -> the IntroVAE part of ACTIVA
            pretrained          -> path to pretrained model
            iteration           -> current iteration in training 
            m                   -> the value of adversarial constant m
            prefix (optional)   -> prefix for the saved filenames
            classifier(optional)-> if wanting to save the classifier model as well (recommended)
        
        OUTPUTS:
            model: the IntroVAE part of ACTIVA
            cf_model: the conditioner of ACTIVA
            
        """
        weights = torch.load(pretrained)
        # load the pretrained model (the IntroVAE part)
        pretrained_dict = weights['Saved_Model'].state_dict() 
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        
        # load the pretrained model (the classifier part)
        if classifier_model:
            cf_model = weights['Classifier_Model'].state_dict()
            cf_model_dict = cf_model.state_dict()
            pretrained_cf_dict = {k: v for k, v in pretrained_cf_dict.items() if k in model_dict}
            cf_model_dict.update(pretrained_cf_dict) 
            cf_model.load_state_dict(model_dict)
            
            return model, cf_model
        
        else:
            return model        
        

def save_checkpoint(model, epoch, iteration, m, prefix="", classifier_model=None):
        """
        Utility function for saving ACTIVA model, including the IntroVAE and Classifier part
        INPUT:
            model                    -> the IntroVAE part of ACTIVA
            epoch                    -> current epoch in training (for naming)
            iteration                -> current iteration in training (for naming)
            m                        -> the value of adversarial constant m (for naming)
            prefix (optional)        -> prefix for the saved filenames (for naming)
            classifer_model(optional)-> the classifier model you want to save withing ACTIVA (recommended)
            
        """
        
        dir_path =  './' + prefix + f"-m{int(m)}-Saved_Model/"
        model_out_path = dir_path + f"model_epoch_{epoch}_iter_{iteration}.pth"
        if not classifier_model:
            state = {"epoch": epoch ,"Saved_Model": model}
        else:
            state = {"epoch": epoch ,"Saved_Model": model, "Classifier_Model":classifier_model}
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))
        

def generate_subpopulation(model, 
                           cf_model, 
                           clusters_dict,
                           num_genes,
                           device=None):

        """
        Utility function for generating specific subpopulations
        INPUT:
            model            -> the IntroVAE part of ACTIVA
            model            -> the classifier network of ACTIVA
            num_genes        -> number of features (genes) in the data
            clusters_dict    -> a dictionary of clusters (as key) and number of samples to produce (value)
            device (optional)-> hardware to perform computations on  
            
        OUTPUT:
            cells: a numpy count-matrix of the cells generated 
        """
    
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if str(device) == "cuda":
            print('Using GPU (CUDA)')
        else:
            print('Using CPU')

        start = time.time()
        cells = torch.empty((1, 1, num_genes)).to(device)

        for key in clusters_dict:
            while len(cells) < clusters_dict[key]:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    # while len(type_1 ) < clusters_dict['1']:
                    num_cells = 10000
                    # look at the input size to the generator network of ACTIVA
                    latent_dim = 128;
                    z_g = torch.randn(num_cells, latent_dim).to(device)
                    # generate synthetic cells with ACTIVA
                    generated_cells = activa.decoder(z_g)
                    cell_types = activa_cf(generated_cells)
                    _, predicted = torch.max(cell_types.squeeze(), 1)

                    cell_type2_ind = (predicted == int(key)).nonzero().tolist()
                    try:
                        cells = torch.cat((cells, generated_cells[cell_type2_ind,:]),dim=0)
                    except:
                        pass;

        print(f"We generated {len(cells)}  of cluster {key} cells in {time.time() - start} seconds (on {device})")