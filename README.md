# Neural Input Optimization

Neural Input Optimization (NIO) is a technique that uses a differentiable surrogate model to reverse-engineer optimal input values given desired outputs and constraints. Rather than relying on black-box search or trial-and-error methods, NIO trains a forward model—typically a neural network—to learn the mapping from input variables to output targets. Once trained, this model is used in reverse: it enables gradient-based optimization of inputs by minimizing a constraint-aware loss function, effectively “inverting” the system to find suitable inputs that yield the specified outputs. 
NIO has broad applicability across domains such as engineering, finance, healthcare, and material science, where stakeholders often know what they want as outputs (e.g., performance goals, target prices, biological responses) but must determine feasible inputs under real-world constraints. For example, a veterinary use case might involve identifying safe and effective dosage combinations of antibiotics based on desired recovery outcomes and species-specific tolerances. In this way, NIO acts as a decision support layer that enables fast, controlled experimentation across complex, multivariable systems—without needing full analytical inverses.

---

## NIO Applications

* Cyber security (https://github.com/rcalix1/CyberSecConstraintsML)
* Industrial Blast Furnace Applications (https://github.com/rcalix1/ConstraintsMLprediction/tree/main)

___

## Reasoning

* https://proceedings.neurips.cc/paper_files/paper/2016/file/fb87582825f9d28a8d42c5e5e5e8b23d-Paper.pdf
* “Reasoning Through Optimization: Embedding a Learnable Calculus Engine in Large Language Models”
* Neural Autograd 


---

##  🔁 NIO Auditing Example (Python)

This example demonstrates **Neural Input Optimization (NIO)** — a method for inverting an AI model by optimizing its inputs to achieve a target output.

The goal is to simulate an audit scenario: *What input would cause the model to output a risky, sensitive, or policy-relevant value — even while respecting constraints?*

---

## 🧠 Scenario
We define a simple pretrained model `f(x)` and then use gradient descent to search for an input `x_opt` such that:

- `f(x_opt)` ≈ desired `y_target` (e.g., a red-flag threshold)
- `x_opt` stays within allowed input bounds

This simulates how an attacker or auditor might find borderline inputs that pass policy checks but produce dangerous or unexpected outputs.

---

## ✅ Code with Explanations

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple pretrained model: f(x) = Wx + b
model = nn.Sequential(nn.Linear(3, 1))
model.eval()

# 🔒 Freeze model weights so they're not updated during optimization
for param in model.parameters():
    param.requires_grad = False

# 🎯 Define the output you want the model to produce
y_target = torch.tensor([[0.8]])  # Target output (e.g., score threshold)

# 🎯 Initialize the input to be optimized
x_opt = torch.randn((1, 3), requires_grad=True)  # 3-feature input vector

# 🔧 Create an optimizer for x only
optimizer = optim.Adam([x_opt], lr=0.1)

# 🔁 Run the input optimization loop
for step in range(200):
    optimizer.zero_grad()
    y_pred = model(x_opt)

    # Loss: how close are we to the desired output?
    loss = (y_pred - y_target).pow(2).mean()

    # Optional: soft constraint to keep x_opt within reasonable bounds
    loss += 0.01 * torch.clamp(x_opt, -2, 2).pow(2).sum()

    loss.backward()
    optimizer.step()

# ✅ Final result
print("Optimized input that triggers output ≈", y_target.item(), ":\n", x_opt.detach())
```

---

# NIO for Auditing AI Models

Neural Input Optimization (NIO) can be used not only to audit security policies (like password constraints) but also to audit the AI models themselves. This framing treats the model as a frozen function and searches for inputs that cause specific outputs or internal activations, subject to constraints. Below are two compelling use cases.


## A. Triggering Target Output Text in LLMs

Use NIO to discover inputs (e.g., embeddings or prompts) that make a frozen LLM produce specific output completions — such as toxic language, jailbreak instructions, or sensitive policy violations. This is a form of automated red-teaming where the optimized input is crafted not by manual prompting but by gradient descent. It allows auditors to systematically search for edge-case completions that may not be covered in traditional evaluations.



## B. Representation Leakage Auditing

Use NIO to recover or approximate the original input that produced a known internal embedding or hidden state. This method tests how much information about the input is retained and potentially leaked through the model's representations. It's useful for auditing risks related to memorization, privacy violations, or information leakage in embedding APIs and transformer models.



These use cases reframe input optimization as a structured, constraint-aware audit tool rather than an adversarial attack, opening up new applications in AI governance, safety, and risk assessment.


---

## 🔍 Why This Matters
- This is the core idea behind **NIO auditing**: instead of analyzing fixed data, you *probe* the model to reveal hidden vulnerabilities.
- You can adapt this to audit **security policies**, **recommender systems**, **identity scoring**, or **password strength metrics**.
- Constraints can be hard (clamps) or soft (penalties), allowing you to simulate realistic boundaries.





---

## Links

* https://github.com/giovanniMen/CPCaD-Bench/tree/main
* 
