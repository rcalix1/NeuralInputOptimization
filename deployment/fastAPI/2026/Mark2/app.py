from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import json
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


########################################################################################################

import pandas as pd
import sklearn
import torch.nn.functional as F

import math
import random
import functorch
from numpy.random import normal
from scipy.stats import norm
import scipy.stats as stats
from numpy import hstack
from numpy import vstack
from numpy import exp
from sklearn.neighbors import KernelDensity
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
import os, json, time, uuid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile

import psutil
import gc
from numpy.linalg import norm
from io import StringIO


from fastapi.middleware.cors import CORSMiddleware



########################################################################################################


import PDFshapingUtils as PDF_tk


########################################################################################################


PDFshapingOBJ = PDF_tk.PDFshapingUtils()
PDFshapingOBJ.read_csv_file_with_pandas('CFD.16.2025.csv')



## Ty suggested fix where 0 needs to be 300
PDFshapingOBJ.CFD_raw_data["i_h2_temp"] = PDFshapingOBJ.CFD_raw_data["i_h2_temp"].replace(0, 300)


PDFshapingOBJ.print_headers_list()


PDFshapingOBJ.list_of_selected_column_names = ['i_h2i_rate','i_h2_temp','i_ngi_rate','i_ng_temp','i_pci_rate',
                    'i_wpi_rate','i_o2_volfract', 'i_hbtemp','i_wind_rt','o_prod_rt', 'o_tgt', 'o_hmt','o_fta', 'o_coke_rt']



########################################################################################################


PDFshapingOBJ.convert_pd_data_to_numpy()



PDFshapingOBJ.gen_X_y_for_selected_indeces(  
                   inputs = [  2, 3, 5, 6, 8, 9, 10   ] , 
                   outputs= [ 27, 29, 39, 40, 28 ]   
)



PDFshapingOBJ.random_seed = int( random.random() * 100  )               ## defautl is 42
PDFshapingOBJ.split_np_data_train_test(selected_test_size=0.2)
PDFshapingOBJ.convert_dataset_from_np_to_torch()
PDFshapingOBJ.standardize_X_scales()
PDFshapingOBJ.standardize_y_scales()


PDFshapingOBJ.gen_Dataloader_train()


########################################################################################################


n_inputs  = 7     
n_outputs = 5

########################################################################################################
##   F1      plus       F2
## Linear     +      Nonlinear


class F1plusF2_SIO_Forward(nn.Module):
    ## initialize the layers
    def __init__(self, x_means, x_deviations, y_means, y_deviations,  device='cuda'):
        super().__init__()
        self.device = device
        
        ## self.x_means      = x_means
        ## self.x_deviations = x_deviations
        ## self.y_means      = y_means
        ## self.y_deviations = y_deviations
        
        
        self.x_means      = x_means.to(self.device)
        self.x_deviations = x_deviations.to(self.device)
        self.y_means      = y_means.to(self.device)
        self.y_deviations = y_deviations.to(self.device)
        
        
        
        ## F1
        self.f1_linear1 = nn.Linear(n_inputs, n_outputs)       
        
        ## F2
        self.f2_linear1 = nn.Linear(n_inputs, 10)
        self.f2_act1    = nn.Tanh()   ## nn.Sigmoid()    ## Tanh()    nn.ReLU()                 
        self.f2_linear2 = nn.Linear(10, n_outputs)       
        self.f2_dropout = nn.Dropout(0.25)
        
        # Move model to device
        self.to(self.device)
        
        
    ## perform inference
    def forward(self, x):
        x = x.to(self.device)
        x = (x - self.x_means) / self.x_deviations
        
        ## F1
        f1 = self.f1_linear1(x)
        
        ## F2
        f2 = self.f2_linear1(x)
        f2 = self.f2_act1(f2)
        f2 = self.f2_dropout(f2)
        f2 = self.f2_linear2(f2)
        
        
        y_scaled   = f1 + f2
        y_descaled = y_scaled * self.y_deviations + self.y_means
        
        ##   y_descaled = torch.clamp(  y_descaled, min=0.0  )
        
        return y_descaled, y_scaled
    

########################################################################################################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_Forward     =     F1plusF2_SIO_Forward(
                         PDFshapingOBJ.x_means, 
                         PDFshapingOBJ.x_deviations, 
                         PDFshapingOBJ.y_means, 
                         PDFshapingOBJ.y_deviations,
                         device=device  # pass device into the model
)



########################################################################################################


optimizer = optim.Adam(model_Forward.parameters(), lr=0.001)
loss_fn   = nn.MSELoss()


########################################################################################################



train_bool = False


########################################################################################################


## model_Forward.train()



if train_bool == True:

    model_Forward.train()

    for epoch in range(1000):
        for xb, yb in PDFshapingOBJ.train_dl:
        
            xb = xb.to(model_Forward.device)
            yb = yb.to(model_Forward.device)
        
            optimizer.zero_grad()
            pred_descaled, pred_scaled = model_Forward(xb)
        
            loss = loss_fn(pred_scaled, yb)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(epoch, ".....", loss.item() ) 


    torch.save(model_Forward.state_dict(), "artifacts/model_steel.pt")
    print("model_steel.pt")
else:
    model_Forward.load_state_dict(  torch.load("artifacts/model_steel.pt", map_location=model_Forward.device)  )
    model_Forward.eval()  # VERY IMPORTANT: keeps model frozen for inference




########################################################################################################



print("=== SANITY CHECK: SCALING ===")
print("y_means:", PDFshapingOBJ.y_means)
print("y_deviations:", PDFshapingOBJ.y_deviations)

print("===============================================")

print("x_means:", PDFshapingOBJ.x_means)
print("x_deviations:", PDFshapingOBJ.x_deviations)



print(PDFshapingOBJ.y_means -  PDFshapingOBJ.y_deviations)
print(PDFshapingOBJ.y_means +  PDFshapingOBJ.y_deviations)


print(PDFshapingOBJ.x_means -  PDFshapingOBJ.x_deviations)
print(PDFshapingOBJ.x_means +  PDFshapingOBJ.x_deviations)


########################################################################################################


def cost_func_h2i_rate(H2_rate=10, PCI_rate=25, H2_temp=600):
    
    E22        = H2_rate
    E23        = PCI_rate
    E26        = H2_temp
    
    
    ## E22        = 10       ## Starter variable  H2 rate
    ## E23        = 25       ## Starter variable PCI rate
    ## E26        = 600      ## Starter variable H2 temp
    
    ############################
    
    E6         = 0.0143 * 0.02
    
    H6         = E6 * E26
    
    E4         = 2
    
    E5         = E4 + H6
    
    
    
    H2_ambient =  E4 * E22
    H2_heated  =  E5 * E23
    H2_temp    =  E6 * E26
    
    
    
    
    result = H2_ambient + H2_heated + H2_temp
    
    return result



## cost_func_h2i_rate()


########################################################################################################


## Oxygen Enrichment in Wind



def cost_func_o2_volfract( wind_rate=195, o2=0.215, prod_rate=5775):
    
    o2 = o2 / 100.0
    
    
    '''
    print("wind_rate=195")
    print("o2=0.215")
    print("prod_rate=5775")
    
    print(wind_rate)
    print(o2)
    print(prod_rate)
    '''
    
    
    E20          = wind_rate
    E25          = o2 
    E19          = prod_rate 
    
    ## E20          = 195            ## Starter variable wind rate
    ## E25          = 0.215          ## Starter variable O2 
    ## E19          = 5775           ## Starter variable production rate 
    
    ######################################
    
    H25          = (E25 - 0.21) * E20
    
    
    K25          = H25 * 60 * 24
    
    N25          = K25 / E19
    
    E9           = 0.01 * 1000
    
    result       = E9 * N25
    
    
    return result



## cost_func_o2_volfract()





########################################################################################################




def cost_func_pci_rate(PCI_rate=25):
    
    E23         = PCI_rate
    
    ## E23         = 25        ## Start variable PCI rate
    
    #########################
    
    E7          = 0.3
    
    result      = E7 * E23
    
    return result



## cost_func_pci_rate()



########################################################################################################





def cost_func_ngi_rate( NG_rate=70 ):
    
    E24    = NG_rate
    
    ## E24    = 70         ## Start variable - NG rate
    
    #############################
    
    
    E8     = 0.20
    
    result = E8 * E24
    
    
    return result



## cost_func_ngi_rate()



########################################################################################################





def cost_func_coke_rate( coke_rate=470 ):
    
    
    E21         = coke_rate
    
    ## E21         = 470      ## Start variable coke rate
    
    ########################
    

    E12         = 0.45
    
    result      = E12 * E21 
    
    return result


## cost_func_coke_rate()



########################################################################################################


def cost_func_wind_rate( wind_rate=195, prod_rate=5775 ):
    
    
    E20       = wind_rate
    E19       = prod_rate
    
    
    ## E20       = 195         ## start variable wind rate
    ## E19       = 5775        ## start variable production rate
    
    ##########################
    

    E11       = 0.1
    
    
    H20       = E20 * 60
    K20       = H20 * 24
    
    
    N20       = K20 / E19
    
    result    = E11 * N20
    
    return result



## cost_func_wind_rate()



########################################################################################################




def regularize_z(z, strength=1e-3):
    
    return strength * torch.sum(z**2)




########################################################################################################





def soft_box_penalty(x, lower, upper, strength=1.0):     ## strength=10.0
    
    return strength * ((torch.relu(lower - x) ** 2).sum() + (torch.relu(x - upper) ** 2).sum())


########################################################################################################



def soft_box_penalty2(x, lower, upper, beta=0.01):
    
    # original constraint penalty
    violation = (torch.relu(lower - x)**2 + torch.relu(x - upper)**2).sum()
    
    # NEW: center stabilizer
    center      = (lower + upper) / 2
    center_pull = ((x - center)**2).mean()
    
    return violation + beta * center_pull



########################################################################################################



# --- distance function ---
def euclid(a, b):
    return norm(a - b)



########################################################################################################


def wrapped_model(x):
    y_descaled, _ = model_Forward(x)
    return y_descaled 



########################################################################################################


## "i_h2i_rate,   i_pci_rate,   i_ngi_rate, i_o2_volfract,    i_h2_temp, i_hbtemp, i_wind_rt"
    
## "o_tgt, o_hmt, o_prod_rt, o_fta, o_coke_rt"


price_real    = np.array([0.0, 0.3, 0.20, 0.0, 0.0, 0.0, 0.0])
price_real    = torch.tensor(price_real, dtype=torch.float32)

price_real_y5 = np.array([0.0, 0.0, 0.0, 0.0, 0.45])
price_real_y5 = torch.tensor(price_real_y5, dtype=torch.float32)




########################################################################################################


model_Forward.eval()


for p in model_Forward.parameters():
    p.requires_grad = False




########################################################################################################


dist_method_A = []
dist_method_B = []
cost_diffs    = [] 
cost_diff     = 0

actual_pred_cost = []
actual_real_cost = []

percent_diffs = []
percent_diff  = 0 

learning_rate = 0.01        ## 0.001      ## 1e-5



########################################################################################################


'''
def print_losses(j, cost_pred, x_temp_pred_np, cost_real, loss, loss_main, loss_cost, loss_ranges, 
                   new_loss_costs_equations, loss_hmt_norm):
    
    
    ##############
    
    print(f"\nITERATION {j}")
    print("=" * 50)

    print(f"*** COSTS ***")
    print(f"Pred Cost : {cost_pred.item():10.2f}")
    print(f"Real Cost : {cost_real.item():10.2f}")

    print("\n*** LOSSES ***")

    metrics = [
        ("Loss Main (MSE)", loss_main.item()),
        ("Loss Ranges", loss_ranges.item()),
        ("Loss HMT Norm", loss_hmt_norm.item()),
        ("Loss Cost", loss_cost.item()),
        ("Cost Equation", new_loss_costs_equations),
        ("Total Loss", loss.item()),
    ]

    for name, value in metrics:
        print(f"{name:25s} {value:12.2f}")

'''





########################################################################################################




def print_losses(j, cost_pred, x_temp_pred_np, cost_real, loss, loss_main, loss_cost, loss_ranges, 
                    new_loss_costs_equations, loss_hmt_norm, out):
    
        
    ##############################
    

    out.write(f"\nITERATION {j}\n")
    out.write("=" * 50 + "\n")


    out.write("*** COSTS ***\n")
    out.write(f"Pred Cost : {cost_pred.item():10.2f}\n")
    out.write(f"Real Cost : {cost_real.item():10.2f}\n")


    out.write("\n*** LOSSES ***\n")


    metrics = [
        ("Loss Main (MSE)", loss_main.item()),
        ("Loss Ranges", loss_ranges.item()),
        ("Loss HMT Norm", loss_hmt_norm.item()),
        ("Loss Cost", loss_cost.item()),
        ("Cost Equation", new_loss_costs_equations),
        ("Total Loss", loss.item()),
    ]


    for name, value in metrics:
        out.write(f"{name:25s} {value:12.2f}\n")

    text = out.getvalue()


    print(text)      # still prints to console
    return out      # also returns out string object




########################################################################################################



def final_data_gathering(i, wrapped_model, x_temp_pred_np, x_real_np ):
    
    print(i)
    print('*******************************************************')
    
    
    if i == 1:
        percent_errors = [ [] for _ in range(5) ]  # one list per output

    y_pred_np = wrapped_model(
                    torch.from_numpy( 
                         x_temp_pred_np
                    )).detach().numpy().flatten()
    
    y_real_np = wrapped_model(
                    torch.from_numpy(
                          x_real_np
                    )).detach().numpy().flatten()

    # % error per variable
    pct = (y_pred_np - y_real_np) / (y_real_np + 1e-8) * 100

    for k in range(5):
         percent_errors[k].append( pct[k] )
    return percent_errors




########################################################################################################




from io import StringIO


def print_some_values(i, x_temp_pred_np,  x_real_np, wrapped_model, PDFshapingOBJ, target_y  ):

    out = StringIO()

    the_cols = "    i_h2i_rate,   i_pci_rate,   i_ngi_rate, i_o2_volfract,    i_h2_temp, i_hbtemp, i_wind_rt"

    #######################################
    #######################################
    #######################################


    x_pred = np.round(x_temp_pred_np[0], 1)
    
    x_real = np.round(x_real_np[0], 1)
    
    

    y_pred = wrapped_model(  torch.from_numpy(x_temp_pred_np)  ).detach().cpu().numpy()[0]
    y_pred = np.round(y_pred, 1)
    
    
    

    y_real = np.round(  target_y.cpu().numpy()    , 1   )

    
    #######################################
    #######################################
    #######################################



    out.write("\nINPUTS\n")
    out.write(f"{'Variable':15s} {'Base':>10s} {'New ':>10s}\n")
    out.write("-"*38 + "\n")


    cols_x = [
        "h2i_rate","pci_rate","ngi_rate",
        "o2_volfrac","h2_temp","hbtemp","wind_rt"
    ]

  
    for c,p,r in zip(cols_x, x_real, x_pred):
        out.write(f"{c:15s} {p:10.1f} {r:10.1f}\n")


    out.write("\nOUTPUTS\n")
    out.write(f"{'Variable':15s} {'Base':>10s} {'New ':>10s}\n")
    out.write("-"*38 + "\n")

    cols_y = [
        "tgt","hmt","prod_rt","fta","coke_rt"
    ]

  
    for c,p,r in zip(cols_y, y_real, y_pred):
        out.write(f"{c:15s} {p:10.1f} {r:10.1f}\n")

    text = out.getvalue()

    print(text)
    return text


########################################################################################################




memory_history = []




def record_memory(i):

    ram_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3

    tensor_count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor_count += 1
        except:
            pass

    memory_history.append(
        (
            i,
            ram_gb,
            tensor_count,
            len(gc.get_objects())
        )
    )


########################################################################################################



def calculate_real_costs_for_printing(loss_cost_descaled, x_temp_descaled, x_real, target_y):
    
    with torch.no_grad():
        x_temp_pred_np   = x_temp_descaled.detach().numpy()
        cost_pred        = loss_cost_descaled
        ## torch.dot(price_real, x_temp_descaled.squeeze()).item()
    
        #######################
    
        x_real = x_real.unsqueeze(0)
        target_y = target_y.unsqueeze(0)
    
        ##print(x_real.shape)
        ##print(target_y.shape)
    
        loss_cost_real          =     x_real @ price_real
        loss_cost_y5_real       =   target_y @ price_real_y5
        
        loss_wind_rate_real     = cost_func_wind_rate( x_real[:, 6], target_y[:, 2] )
        loss_h2i_rate  = cost_func_h2i_rate(x_real[:, 0], x_real[:, 1], x_real[:, 4])
        loss_o2_v    = cost_func_o2_volfract( x_real[:,6], x_real[:,3], target_y[:,2])
        
        ## loss_cost_descaled_real = loss_cost_real + loss_cost_y5_real + loss_wind_rate_real + loss_h2i_rate + loss_o2_v
    
        loss_cost_descaled_real = (
            loss_cost_real
            + loss_cost_y5_real
            + loss_wind_rate_real
            + loss_h2i_rate
            + loss_o2_v
        )
    
        #######################
        
        x_real_np        = x_real.detach().numpy() 
        cost_real        = loss_cost_descaled_real
       
    return x_temp_pred_np, x_real_np, cost_pred, cost_real





########################################################################################################



clamp_min = torch.tensor([[   0,   0,   0,    21,   300,   1200, 150 ]])     ## from Ty
clamp_max = torch.tensor([[ 300, 300, 300,    90,  1200,   1500, 220 ]])



##                                   1710
y_clamp_min = torch.tensor([[  100,  1710,   6400,    2000,   200 ]])        ## from Ty
y_clamp_max = torch.tensor([[  150,  1820,   6600,    2600,   500 ]])




y_clamp_min_normalized =  (y_clamp_min  - PDFshapingOBJ.y_means) /  PDFshapingOBJ.y_deviations
y_clamp_max_normalized =  (y_clamp_max  - PDFshapingOBJ.y_means) /  PDFshapingOBJ.y_deviations



########################################################################################################


get_x_from_z = lambda z: clamp_min + (clamp_max - clamp_min) * torch.sigmoid(z)



########################################################################################################



w_mse   = 1.0
w_cost  = 1.0       
w_range = 1.0             
w_hmt   = 120.0        ## (if can't improve, use 20 and save notebook) - 10.0   (1710)    
  

#########################################################################################################



y_clamp_min_temp            = torch.tensor([[  100,  1710 + 55,   6400,    2000,   200 ]])     
y_clamp_min_temp_normalized =  (y_clamp_min_temp  - PDFshapingOBJ.y_means) /  PDFshapingOBJ.y_deviations



##########################################################################################################
##  '{"tgt": 127, "hmt": 1770, "prod_rt": 9010, "fta": 2320, "coke_rt": 382}' | jq



class OptimizeRequestNIO( BaseModel ):
    tgt:                 float
    hmt:                 float
    prod_rt:             float
    fta:                 float
    coke_rt:             float

    i_h2i_rate:          float
    i_pci_rate:          float
    i_ngi_rate:          float
    i_o2_volfract:       float
    i_h2_temp:           float
    i_hbtemp:            float
    i_wind_rt:           float

    xmin_i_h2i_rate:     float
    xmin_i_pci_rate:     float
    xmin_i_ngi_rate:     float
    xmin_i_o2_volfract:  float
    xmin_i_h2_temp:      float
    xmin_i_hbtemp:       float
    xmin_i_wind_rt:      float

    xmax_i_h2i_rate:     float
    xmax_i_pci_rate:     float
    xmax_i_ngi_rate:     float
    xmax_i_o2_volfract:  float
    xmax_i_h2_temp:      float
    xmax_i_hbtemp:       float
    xmax_i_wind_rt:      float

    ymin_tgt:            float
    ymin_hmt:            float
    ymin_prod_rt:        float
    ymin_fta:            float
    ymin_coke_rt:        float

    ymax_tgt:            float
    ymax_hmt:            float
    ymax_prod_rt:        float
    ymax_fta:            float
    ymax_coke_rt:        float

    hmt_center_delta:    float
    quad_exp_weight:     float
    




########################################################################################################
##   "o_tgt,      o_hmt,       o_prod_rt,       o_fta,        o_coke_rt"
##  '{"tgt": 127, "hmt": 1770, "prod_rt": 9010, "fta": 2320, "coke_rt": 382}' | jq
## input_dict: {"Cement": 200, "Water": 150, ...}
## Returns a torch tensor shaped (1, input_dim)
## tgt  = torch.tensor([request.target_strength], dtype=torch.float32)
## ## "i_h2i_rate,   i_pci_rate,   i_ngi_rate, i_o2_volfract,    i_h2_temp, i_hbtemp, i_wind_rt"
########################################################################################################



def encode_inputs_NIO(  request  ):
   

    values_y = [  request.tgt, request.hmt, request.prod_rt, request.fta, request.coke_rt    ]

    ## values_x = [  0, 175, 0, 30, 300, 1480, 195  ]
    values_x = [  request.i_h2i_rate, request.i_pci_rate, request.i_ngi_rate, request.i_o2_volfract,  request.i_h2_temp, request.i_hbtemp, request.i_wind_rt ]


    target_y = torch.tensor(values_y, dtype=torch.float32)   ## (5,)
    x_real   = torch.tensor(values_x, dtype=torch.float32)   ## (7,)


    #######################

    
    ## clamp_min = torch.tensor([[   0,   0,   0,    21,   300,   1200, 150 ]])     ## from Ty
    clamp_min = torch.tensor([[request.xmin_i_h2i_rate,request.xmin_i_pci_rate,request.xmin_i_ngi_rate,request.xmin_i_o2_volfract,request.xmin_i_h2_temp,request.xmin_i_hbtemp,request.xmin_i_wind_rt]])  

    
    ## clamp_max = torch.tensor([[ 300, 300, 300,    90,  1200,   1500, 220 ]])
    clamp_max = torch.tensor([[request.xmax_i_h2i_rate,request.xmax_i_pci_rate,request.xmax_i_ngi_rate,request.xmax_i_o2_volfract,request.xmax_i_h2_temp,request.xmax_i_hbtemp,request.xmax_i_wind_rt]])


    #######################


    ## y_clamp_min = torch.tensor([[  100,  1710,   6400,    2000,   200 ]])        ## from Ty
    y_clamp_min = torch.tensor([[  request.ymin_tgt, request.ymin_hmt, request.ymin_prod_rt, request.ymin_fta, request.ymin_coke_rt  ]])  

    
    ## y_clamp_max = torch.tensor([[  150,  1820,   6600,    2600,   500 ]])
    y_clamp_max = torch.tensor([[  request.ymax_tgt, request.ymax_hmt, request.ymax_prod_rt, request.ymax_fta, request.ymax_coke_rt  ]])


    #######################


    ## hmt_center_delta        = 55
    hmt_center_delta           =  request.hmt_center_delta 

    quad_exp_weight            =  request.quad_exp_weight

    y_clamp_min_temp        = y_clamp_min.clone()

    y_clamp_min_temp[:, 1] += hmt_center_delta


    #######################
 

    return target_y, x_real, clamp_min, clamp_max, y_clamp_min,  y_clamp_max, y_clamp_min_temp, quad_exp_weight 




########################################################################################################



'''


INPUTS
Variable              Pred       Real
--------------------------------------
h2i_rate               3.2        0.0
pci_rate               0.9      175.0
ngi_rate              61.0        0.0
o2_volfrac            24.1       30.0
h2_temp             1167.0      300.0
hbtemp              1492.4     1480.0
wind_rt              206.9      195.0

OUTPUTS
Variable              Pred       Real
--------------------------------------
tgt                  119.5      127.0
hmt                 1718.3     1770.0
prod_rt             8039.8     9010.0
fta                 2493.1     2320.0
coke_rt              426.1      382.0



'''


########################################################################################################


def calc_dynamic_hmt_weight_func2( min_y , max_y, hmt_pred, hmt_center ):


    ## Distance outside the acceptable HMT range

    outside        = torch.relu(min_y - hmt_pred) + torch.relu(hmt_pred - max_y)

    alpha          = 0.2      # may need tuning

    dynamic_weight = torch.exp( alpha * outside )

    return dynamic_weight




########################################################################################################




def calc_dynamic_hmt_weight_func17(min_y, max_y, hmt_pred, hmt_center):


    outside = torch.relu(min_y - hmt_pred) + torch.relu(hmt_pred - max_y)

    alpha = 0.05          # start small (0.05, maybe 0.1 later)

    return torch.exp(alpha * outside)





########################################################################################################


## This may be a good one, otherwise revert to func17


def calc_dynamic_hmt_weight_func(min_y, max_y, hmt_pred, hmt_center):

    # Half-width of allowable HMT range
    scale = torch.maximum(hmt_center - min_y,
                          max_y - hmt_center)
    scale = torch.clamp(scale, min=1e-6)

    # Normalized distance from preferred HMT
    d = torch.abs(hmt_pred - hmt_center) / scale

    # Relax HMT near the center
    # 0.70 at center -> 1.00 at operating limits
    inside_weight = 0.70 + 0.30 * torch.clamp(d, max=1.0)

    # Additional penalty outside allowable range
    outside = torch.relu(min_y - hmt_pred) + \
              torch.relu(hmt_pred - max_y)

    alpha = 0.05

    outside_weight = torch.exp(alpha * outside)

    return inside_weight * outside_weight




########################################################################################################


from fastapi.responses import PlainTextResponse


app   = FastAPI()


########################################################################################################



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    ## allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


########################################################################################################


i = 0


########################################################################################################
## target_y  = torch.tensor( PDFshapingOBJ.y_test[i]   )    ## what you want y ?
## x_real    = torch.tensor( PDFshapingOBJ.X_test[i]   )    ## real x
########################################################################################################




@app.post("/NIOoptimize")
def NIO_main_optimization( request: OptimizeRequestNIO  ):

    
    loss_log_rc = StringIO()

    target_y, x_real,  clamp_min, clamp_max, y_clamp_min, y_clamp_max, y_clamp_min_temp, quad_exp_weight     =    encode_inputs_NIO(  request  )


    ##################################################################

    hmt_history = []

   
    ##################################################################

        
    y_clamp_min_temp_normalized =  (y_clamp_min_temp  - PDFshapingOBJ.y_means) /  PDFshapingOBJ.y_deviations
        
    target_y_normalized                  = (target_y  - PDFshapingOBJ.y_means) /  PDFshapingOBJ.y_deviations


    y_clamp_min_normalized           =  (y_clamp_min  - PDFshapingOBJ.y_means) /  PDFshapingOBJ.y_deviations
    y_clamp_max_normalized           =  (y_clamp_max  - PDFshapingOBJ.y_means) /  PDFshapingOBJ.y_deviations
    

    ##################################################################
    
    z_init    = torch.rand((1,7))*0.2 + 0.3
    z         = torch.nn.Parameter(  torch.logit( z_init )   )    ## this requires grad 
    optimizer = torch.optim.Adam([z], lr=learning_rate)
    
    
    for j in range(3000):              ## 2000
        
        optimizer.zero_grad()
        
        x_temp_descaled              = get_x_from_z(    z    )

        y_descaled, current_y_normalized = model_Forward( x_temp_descaled )
        
        ##########################################
        
        loss_main_norm   = torch.mean(  (current_y_normalized - target_y_normalized)**2  )
        
        ##########################################
        
        loss_ranges_norm = soft_box_penalty(current_y_normalized, y_clamp_min_normalized, y_clamp_max_normalized)
        
        ##########################################
        
        new_loss_costs_equations = 0.0
        
        loss_cost          =     x_temp_descaled @ price_real
        loss_cost_y5       =          y_descaled @ price_real_y5
        
        
        loss_wind_rate = cost_func_wind_rate( x_temp_descaled[:, 6], y_descaled[:, 2] )
        loss_h2i_rate  = cost_func_h2i_rate(x_temp_descaled[:, 0], x_temp_descaled[:, 1], x_temp_descaled[:, 4])
        loss_o2_vol    = cost_func_o2_volfract( x_temp_descaled[:,6], x_temp_descaled[:,3], y_descaled[:,2])
        
        loss_cost_descaled = loss_cost + loss_cost_y5 + loss_wind_rate + loss_h2i_rate + loss_o2_vol
        
    
        ##########################################
        # HMT is index 1 (1710)


        hmt_pred         =          current_y_normalized[:, 1]

   
        hmt_history.append(  hmt_pred.detach()  )
        
        

        hmt_center       =   y_clamp_min_temp_normalized[:, 1]       

        loss_hmt_norm    =      ( hmt_center - hmt_pred ) ** 2


        ################################################

        if len(hmt_history) < 50:
            hmt_dynamic_avg = hmt_pred
        else:
            hmt_dynamic_avg = torch.stack(  hmt_history[-50:]  ).mean(dim=0)



        if quad_exp_weight == 0:
            dynamic_hmt_weight = 1
        else:
            dynamic_hmt_weight = calc_dynamic_hmt_weight_func( y_clamp_min_normalized[:, 1], y_clamp_max_normalized[:, 1], hmt_dynamic_avg, hmt_center )

     
        ################################################
        
 
        loss_norm     = w_mse*loss_main_norm + w_range*loss_ranges_norm  + w_hmt*dynamic_hmt_weight*loss_hmt_norm
    
        loss_descaled = w_cost*loss_cost_descaled  
        
        loss = loss_norm + loss_descaled
        
       
        loss.backward()
        optimizer.step()
        
        #####################################################################
        #####################################################################
        #####################################################################
        #####################################################################
        
        
        
        
        x_temp_pred_np, x_real_np, cost_pred, cost_real = calculate_real_costs_for_printing(loss_cost_descaled, x_temp_descaled, x_real, target_y)
        
       
        if j % 400 == 0:           
            loss_log_rc = print_losses(j, cost_pred, x_temp_pred_np, cost_real, loss, loss_main_norm, 
                                loss_cost_descaled, loss_ranges_norm, new_loss_costs_equations, loss_hmt_norm, loss_log_rc)
        
   
        #####################################################################
        cost_diff = cost_pred - cost_real
        if cost_real > 0:                         ## handle nans
            percent_diff = cost_diff / cost_real
            percent_diffs.append( percent_diff.item() )
        
    ###################################################################################
            
    cost_diffs.append(        cost_diff.item()  )
    actual_pred_cost.append(  cost_pred.item()  )
    actual_real_cost.append(  cost_real.item()  )
      

    value_log_rc = print_some_values(i, x_temp_pred_np,  x_real_np, wrapped_model, PDFshapingOBJ, target_y )
    
    
    dist_method_A.append( euclid(x_temp_pred_np, x_real_np) )
    dist_method_B.append( euclid(
                    wrapped_model(torch.from_numpy(x_temp_pred_np)).detach().numpy() , 
                    wrapped_model(torch.from_numpy(x_real_np)).detach().numpy()
    ))
    
    
    ## percent_errors = final_data_gathering(i, wrapped_model, x_temp_pred_np, x_real_np )
    print(i)
    print('*******************************************************')
    
    ##########################################
    
    record_memory(i)
    
    if i % 50 == 0:

        print("\nMEMORY EVOLUTION")
        print("i\tRAM(GB)\tTensors\tObjects")

        for row in memory_history:
            print(
                f"{row[0]}\t"
                f"{row[1]:.2f}\t"
                f"{row[2]:,}\t"
                f"{row[3]:,}"
            )

    print()
    
    ##########################################


    full_log_rc = loss_log_rc.getvalue()  + "\n" + value_log_rc

    return PlainTextResponse( full_log_rc )





########################################################################################################


MODEL_PATH    = 'artifacts/model_v1.pt'
SCALER_PATH   = 'artifacts/scaler.pkl'
METADATA_PATH = 'artifacts/metadata.json'
METRICS_PATH  = 'artifacts/metrics.json'

########################################################################################################

scaler = joblib.load(SCALER_PATH)

####################################################

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

####################################################

feature_order = metadata["feature_order"]
input_dim     = metadata["input_dim"]

####################################################
## Define model architecture (must match training model exactly) 

class ConcreteRegressor(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )


    def forward(self, x):
        return self.net(x)

####################################################


## Converts user-specified input values into a normalized tensor
## following the scaler and feature order from training.

def encode_inputs(input_dict, scaler, feature_order):
    """
    input_dict: {"Cement": 200, "Water": 150, ...}
    Returns a torch tensor shaped (1, input_dim)
    """

    values = [ input_dict[col] for col in feature_order ]

    X        = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    return torch.tensor(X_scaled, dtype=torch.float32)


####################################################





def constraint_loss(pred, target, x, bounds=None, penalty_weight=10.0):
    """
    pred: model output
    target: desired output (concrete strength)
    x: the optimized input tensor
    bounds: {"Cement": (min, max), ...}
    """

    mse_term = (pred - target).pow(2).mean()

    if bounds is None:
        return mse_term  # no constraints

    penalty = 0

    for i, col in enumerate(feature_order):
        min_val, max_val = bounds[col]

        penalty += torch.relu(x[0, i] - max_val) ** 2
        penalty += torch.relu(min_val - x[0, i]) ** 2

    return ((mse_term + penalty_weight) * penalty) / 10000000




####################################################




bounds = {
    "Cement": (0, 540),
    "Blast Furnace Slag": (0, 360),
    "Fly Ash": (0, 200),
    "Water": (0, 250),
    "Superplasticizer": (0, 35),
    "Coarse Aggregate": (800, 1200),
    "Fine Aggregate": (600, 1000),
    "Age": (1, 365) 
}





####################################################
## This is standalone and it defines a simple data schema used 
## for validating and parsing incoming JSON to 
## /optimize endpoint in FastAPI


class OptimizeRequest( BaseModel ):
    target_strength: float






####################################################


steps = 3000
lr    = 0.001


model = ConcreteRegressor(input_dim=input_dim)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

model.eval()  # VERY IMPORTANT 

print("Model, scaler, and metadata loaded successfully")




##########################################




@app.get("/metrics")
def get_metrics():
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        return {"error": "Metrics file not found."}
    except json.JSONDecodeError:
        return {"error": "Error decoding JSON from metrics file."}







##########################################


@app.post("/optimize")
def optimize_inputs(  request: OptimizeRequest  ):

    input_dim = len(feature_order)

    x_opt = torch.zeros((1, input_dim), dtype=torch.float32, requires_grad=True)

    optimizer = optim.Adam([x_opt], lr=lr)

    for step in range(steps):

        optimizer.zero_grad()
        pred = model(x_opt)
        tgt  = torch.tensor([request.target_strength], dtype=torch.float32)

        loss = constraint_loss(pred, tgt, x_opt, bounds=bounds)

        loss.backward()
        optimizer.step()

        # Optional: clamp inputs to [0, 1] to stay within normalized space
        x_opt.data = torch.clamp(x_opt.data, 0, 1)

        if step % 50 == 0:
            print(f"Step {step} | pred={pred.item():.3f} | loss={loss.item():.4f}")

    
    x_np       = x_opt.detach().numpy()
    x_unscaled = scaler.inverse_transform(x_np)[0]

    output_dict = {
        col: float(x_unscaled[i])
        for i, col in enumerate(feature_order)
    }

    return output_dict




########################################################
