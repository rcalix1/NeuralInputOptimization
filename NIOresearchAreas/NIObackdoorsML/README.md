## NIO and ML backdoors

# ML Backdoor Discovery and Repair with NIO

## Original Backdoor Formulation

Let the hypothesis class be

$$
H \subseteq Y^X
$$

and let

$$
h \in H
$$

represent a machine learning model.

A backdoor procedure takes the original model and produces a backdoored model together with a backdoor key:

$$
(\hat{h},bk) \leftarrow \text{Backdoor}(h)
$$

where

- $\hat{h}$ is the backdoored model
- $bk$ is the backdoor key

The activation procedure is

$$
x' \leftarrow \text{Activate}(x,bk)
$$

where $x'$ is a modified version of the original input $x$.

The modified input remains close to the original input:

$$
d(x,x') \leq \gamma
$$

but causes different model behavior:

$$
\hat{h}(x') \neq \hat{h}(x)
$$

Thus, the basic backdoor idea is

$$
x \approx x'
$$

while

$$
\hat{h}(x') \neq \hat{h}(x)
$$

The backdoor key $bk$ contains the information required to activate the hidden behavior of $\hat{h}$.

---

# Proposed NIO Experiment

## Part 1 — Implement the Backdoor

Train a normal machine learning model

$$
h
$$

and construct a backdoored version

$$
(\hat{h},bk) \leftarrow \text{Backdoor}(h).
$$

Verify that the model performs normally on regular inputs but changes its behavior when the backdoor is activated.

---

## Part 2 — Discover the Backdoor with NIO

Do **not** provide NIO with the backdoor key $bk$.

Freeze the parameters of the backdoored model $\hat{h}$ and use Neural Input Optimization to modify the input.

The objective is to determine whether NIO can independently generate inputs that expose the hidden behavior of the backdoored model.

The central question is:

> Can NIO discover the backdoor without knowing the backdoor key?

---

## Part 3 — Generate Repair Data with NIO

Once NIO discovers inputs that expose the backdoor, use NIO to generate many related examples:

$$
x_1^*,x_2^*,\ldots,x_N^*
$$

These examples form an NIO-generated repair dataset:

$$ 
D_{\text{NIO}} = \{(x_i^*,y_i)\}_{i=1}^{N}
$$

where $y_i$ represents the correct behavior for the generated input.

Use this dataset to fine-tune the backdoored model:

$$
\hat{h}
\longrightarrow
h_{\text{repaired}}
$$

---

## Part 4 — Test the Repair

After fine-tuning, test the **original backdoor again using the original backdoor key**.

The desired result is

$$
\text{Backdoor Success}
\quad
\text{High}
\longrightarrow
\text{Low}
$$

while

$$
\text{Normal Model Performance}
\quad
\text{High}
\longrightarrow
\text{High}.
$$

The key question is whether NIO-generated data can patch the backdoor **without significantly degrading the normal performance of the model**.

---

# Experimental Flow

```text
Normal Model
     |
     v
Create Backdoor
     |
     v
Verify Backdoor with bk
     |
     v
Hide bk from NIO
     |
     v
NIO Discovers Backdoor Behavior
     |
     v
NIO Generates Repair Data
     |
     v
Fine-Tune Model
     |
     v
Test Original Backdoor with bk
     |
     +----> Backdoor no longer works?
     |
     +----> Normal performance preserved?
```

## Research Area

**ML Backdoor Discovery and Repair**
