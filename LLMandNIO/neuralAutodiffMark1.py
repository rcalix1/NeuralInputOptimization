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

##################


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Load and prepare Iris data ---
data = load_iris()
X = data.data.astype('float32')
y = data.target.astype('int64')

y_onehot = F.one_hot(torch.tensor(y), num_classes=3).float()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to torch tensors
X_tensor = torch.tensor(X_scaled)
y_tensor = y_onehot

# --- Split into train/test ---
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# --- Forward model: 2-layer MLP ---
def forward_model(x, W1, b1, W2, b2):
    h = F.relu(x @ W1 + b1)
    out = h @ W2 + b2
    return out

# --- Neural Autograd model ---
class NeuralAutograd(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4*10 + 3*10 + 4*16 + 16 + 16*3 + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
        )
        self.heads = nn.ModuleDict({
            'dW1': nn.Linear(1024, 4*16),
            'db1': nn.Linear(1024, 16),
            'dW2': nn.Linear(1024, 16*3),
            'db2': nn.Linear(1024, 3),
        })

    def forward(self, x, y, W1, b1, W2, b2):
        z = torch.cat([
            x.view(-1), y.view(-1),
            W1.view(-1), b1.view(-1),
            W2.view(-1), b2.view(-1)
        ], dim=0)
        h = self.encoder(z)
        return {
            'dW1': self.heads['dW1'](h).view(4, 16),
            'db1': self.heads['db1'](h).view(16),
            'dW2': self.heads['dW2'](h).view(16, 3),
            'db2': self.heads['db2'](h).view(3),
        }

# --- Sample random weights ---
def sample_weights():
    W1 = torch.randn(4, 16, requires_grad=True)
    b1 = torch.randn(16, requires_grad=True)
    W2 = torch.randn(16, 3, requires_grad=True)
    b2 = torch.randn(3, requires_grad=True)
    return W1, b1, W2, b2

# --- Training loop ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ND = NeuralAutograd().to(device)
optimizer = torch.optim.Adam(ND.parameters(), lr=1e-4)

for step in range(5000):
    # Sample weights
    W1, b1, W2, b2 = sample_weights()

    # Select a batch from training data
    idx = torch.randint(0, len(X_train), (10,))
    x_batch = X_train[idx].to(device)
    y_batch = y_train[idx].to(device)

    # Forward pass
    y_hat = forward_model(x_batch, W1, b1, W2, b2)
    loss = F.mse_loss(y_hat, y_batch)

    # Compute true gradients
    grads = torch.autograd.grad(loss, [W1, b1, W2, b2])

    # Predict gradients
    pred = ND(x_batch, y_batch, W1.detach(), b1.detach(), W2.detach(), b2.detach())

    # MSE loss between predicted and real gradients
    loss_nd = sum(F.mse_loss(pred[k], g.detach()) for k, g in zip(pred, grads))

    # Backprop ND
    optimizer.zero_grad()
    loss_nd.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}: ND Loss = {loss_nd.item():.6f}")



