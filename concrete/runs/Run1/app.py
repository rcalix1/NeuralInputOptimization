from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
import json
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

####################################################


MODEL_PATH    = 'artifacts/model_v1.pt'
SCALER_PATH   = 'artifacts/scaler.pkl'
METADATA_PATH = 'artifacts/metadata.json'
METRICS_PATH  = 'artifacts/metrics.json'

####################################################

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


model = ConcreteRegressor(input_dim=input_dim)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

model.eval()  # VERY IMPORTANT 

print("Model, scaler, and metadata loaded successfully")

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


###########################################################
## Custom loss that:
## 1. Minimizes difference from target output (regression)
## 2. Applies penalties for constraint violations
##
##  This ensures optimization respects real-world limits
###########################################################


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

###################################################


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

##########################################

steps = 3000
lr    = 0.001

app   = FastAPI()

##########################################
## This is standalone and it defines a simple data schema used 
## for validating and parsing incoming JSON to 
## /optimize endpoint in FastAPI


class OptimizeRequest( BaseModel ):
    target_strength: float

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

###################################################

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


########################################################
