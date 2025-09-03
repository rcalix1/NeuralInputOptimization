# Iris Neural Autograd Prototype

This prototype sets up the Iris classification task using two models:
1. A standard MLP trained using PyTorch autodiff (baseline)
2. A small neural network (Neural Autograd model) that learns to output gradients for training another MLP — replacing the `.backward()` call

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Target network (classifier)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Neural Autograd model (meta-optimizer)
class NeuralAutograd(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 32),  # takes in (w, grad_f)
            nn.ReLU(),
            nn.Linear(32, 1),  # outputs delta_w
        )

    def forward(self, w, grad):
        inp = torch.stack([w, grad], dim=1)  # [N, 2]
        return self.fc(inp).squeeze(1)  # [N]

# Flatten all parameters into a single vector

def get_params_vector(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

def set_params_vector(model, vec):
    pointer = 0
    for p in model.parameters():
        n = p.numel()
        p.data = vec[pointer:pointer+n].view_as(p).data.clone()
        pointer += n

# Training loop with learned gradient descent
classifier = MLP()
autograd_nn = NeuralAutograd()

optim_meta = torch.optim.Adam(autograd_nn.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for outer_step in range(200):
    # Get current weights
    w = get_params_vector(classifier).detach()
    w.requires_grad = True

    # Forward and loss
    logits = classifier(X_train)
    loss = loss_fn(logits, y_train)
    grad = torch.autograd.grad(loss, w, create_graph=True)[0]

    # Neural Autograd update
    delta = autograd_nn(w, grad)
    new_w = w - delta
    set_params_vector(classifier, new_w)

    # Evaluate updated classifier
    logits_new = classifier(X_train)
    loss_new = loss_fn(logits_new, y_train)

    # Meta-loss for the autograd_nn
    optim_meta.zero_grad()
    loss_new.backward()
    optim_meta.step()

    if outer_step % 20 == 0:
        acc = (logits_new.argmax(dim=1) == y_train).float().mean()
        print(f"Step {outer_step:03d}: Loss = {loss_new.item():.4f}, Acc = {acc.item():.4f}")
```

---


