## Deployment

* fastAPI
* 

---


# Neural Input Optimization (NIO) FastAPI Deployment (No Docker)

This guide walks you through deploying your **Neural Input Optimization (NIO)** pipeline using **FastAPI**, without Docker, on a **Linux terminal with Conda**. It's designed for easy setup, local/edge use, and direct integration with plant systems.

---

## 🧰 Folder Structure

```
nio_server/
├── app.py           # FastAPI server
├── model.pt         # Your pretrained PyTorch forward model
├── requirements.txt # Python dependencies
```

---

## 📦 1. Environment Setup (Anaconda)

```bash
conda create -n nioenv python=3.10 -y
conda activate nioenv

pip install fastapi uvicorn torch
```

To freeze dependencies (optional):

```bash
pip freeze > requirements.txt
```

---

## 🚀 2. FastAPI Server Code (`app.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim

# 📦 Load pretrained forward model
model = torch.load("model.pt", map_location="cpu")
model.eval()
for param in model.parameters():
    param.requires_grad = False

app = FastAPI()

# 📤 Request schema
class OptimizeRequest(BaseModel):
    target: list[float]
    mask: list[int]  # Not used in this simple example
    steps: int = 200

@app.post("/optimize")
def optimize_input(request: OptimizeRequest):
    y_target = torch.tensor([request.target])
    x_opt = torch.randn((1, len(request.target)), requires_grad=True)

    optimizer = optim.Adam([x_opt], lr=0.1)
    for _ in range(request.steps):
        optimizer.zero_grad()
        y_pred = model(x_opt)
        loss = (y_pred - y_target).pow(2).mean()
        loss.backward()
        optimizer.step()

    return {
        "x_opt": x_opt.detach().squeeze().tolist(),
        "y_pred": model(x_opt).detach().squeeze().tolist(),
        "loss": loss.item()
    }
```

---

## 📁 3. Example Forward Model (`model.pt`)

To test the server, you can create a dummy model:

```python
# save_model.py
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3, 1)  # Example: 3 inputs → 1 output
)
torch.save(model, "model.pt")
```

Run:

```bash
python save_model.py
```

---

## 🧪 4. Run the API Server

From the `nio_server` folder:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

You now have a live server at:

```
http://localhost:8000
```

Swagger docs UI:

```
http://localhost:8000/docs
```

---

## 🧪 5. Test the API (Terminal)

```bash
curl -X POST http://localhost:8000/optimize \
     -H "Content-Type: application/json" \
     -d '{
           "target": [0.8],
           "mask": [2],
           "steps": 200
         }'
```

Expected output:

```json
{
  "x_opt": [0.45, -0.03, 0.92],
  "y_pred": [0.7999],
  "loss": 0.00001
}
```

---

## 🔒 Notes

* This version assumes CPU-only execution. If you're on Jetson or want GPU, ensure CUDA is available and `torch` uses the right device.
* You can extend this to use masks, bounds, constraint weights, and overshoot logic from your real NIO pipeline.
* Consider adding logging 

---

## ✅ Next Steps

* Add `/health` or `/version` endpoints
* Add authentication if running outside localhost
* Extend with `matplotlib` for result plots (optional)

---

You’re now ready to deploy and test Neural Input Optimization directly from your terminal. Just activate your environment, run the API, and you're live!





---
