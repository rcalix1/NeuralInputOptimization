## NIO and ML backdoors

# ML Backdoor Discovery and Repair with Neural Input Optimization

## Overview

This research explores the use of **Neural Input Optimization (NIO)** for discovering and repairing hidden backdoors in machine learning models.

The work begins by reproducing a machine learning backdoor in which a trained model behaves normally for standard inputs but produces different behavior when presented with a specially constructed input associated with a secret backdoor key.

The research then investigates whether NIO can discover this hidden behavior without knowledge of the backdoor key. Finally, NIO-generated inputs are used to construct a targeted dataset for fine-tuning the compromised model and attempting to remove the backdoor while preserving normal model performance.

The overall experimental framework is:

**Backdoor → NIO Discovery → NIO Data Generation → Fine-Tuning → Backdoor Repair**

---

## 1. Machine Learning Backdoor

Let

\[
h : X \rightarrow Y
\]

represent a normally trained machine learning model.

A backdoor procedure produces a modified model and a secret backdoor key:

\[
(\hat{h}, bk) \leftarrow \text{Backdoor}(h)
\]

where:

- \(h\) is the original model,
- \(\hat{h}\) is the backdoored model,
- \(bk\) is the secret backdoor key.

For ordinary inputs, the behavior of the two models should remain approximately equivalent:

\[
\hat{h}(x) \approx h(x).
\]

The backdoor key can be used to construct a modified input:

\[
x' \leftarrow \text{Activate}(x,bk).
\]

The modified input should remain close to the original input:

\[
\|x-x'\| \leq \gamma
\]

where \(\gamma\) controls the maximum allowed difference between the original and modified inputs.

However, the backdoored model produces different behavior for the activated input:

\[
\hat{h}(x') \neq \hat{h}(x).
\]

Thus, the important relationship is

\[
x \approx x'
\]

while

\[
\hat{h}(x) \neq \hat{h}(x').
\]

The model therefore behaves normally for standard inputs while containing a hidden behavior that can be activated using the secret backdoor information.

---

## 2. NIO Backdoor Discovery

The second experiment investigates whether **Neural Input Optimization** can discover the hidden behavior without access to the backdoor key \(bk\).

The model parameters are frozen and the input is treated as the optimization variable.

Given an ordinary input \(x_0\), NIO searches for an optimized input \(x^*\) that exposes abnormal model behavior while remaining close to the original input.

One possible formulation is

\[
x^* =
\underset{x}{\arg\max}
\;
D\left(
\hat{h}(x),
h(x)
\right)
\]

subject to

\[
\|x-x_0\| \leq \epsilon.
\]

Here \(D(\cdot,\cdot)\) measures disagreement between the normal model and the backdoored model.

The central research question is:

> Can NIO discover inputs that activate hidden backdoor behavior without knowing the original backdoor key?

The discovered input can then be compared with the true activated input

\[
x'=\text{Activate}(x,bk)
\]

to determine whether NIO has discovered the backdoor or a related vulnerable region of the input space.

---

## 3. NIO-Generated Data for Backdoor Repair

After discovering inputs that expose the backdoor, NIO can be used generatively.

Rather than generating only one optimized input, NIO can generate a collection of inputs that expose the vulnerable region:

\[
D_{\text{NIO}}
=
\left\{
x_1^*,x_2^*,\ldots,x_N^*
\right\}.
\]

These optimized inputs can be assigned their legitimate target labels to create a corrective dataset:

\[
D_{\text{repair}}
=
\left\{
(x_i^*,y_i)
\right\}_{i=1}^{N}.
\]

The compromised model can then be fine-tuned using the NIO-generated dataset:

\[
\hat{h}
\xrightarrow{\text{fine-tuning on }D_{\text{repair}}}
h_{\text{repaired}}.
\]

The objective is to remove or substantially reduce the backdoor behavior while maintaining the original predictive performance of the model.

---

## 4. Experimental Evaluation

The complete experiment evaluates the model at several stages.

### Before Repair

Measure:

- Normal test accuracy
- Backdoor activation success rate
- NIO backdoor discovery rate
- Distance between original and NIO-optimized inputs

### After Repair

Measure:

- Normal test accuracy
- Backdoor activation success rate using the original backdoor key
- Performance on NIO-discovered inputs
- Change in normal model performance

A successful result would demonstrate:

\[
\text{Backdoor Success Rate}
\quad
\text{High}
\rightarrow
\text{Low}
\]

while maintaining

\[
\text{Normal Accuracy}
\quad
\text{High}
\rightarrow
\text{High}.
\]

---

## Research Hypothesis

The primary hypothesis of this work is:

> **Neural Input Optimization can be used to discover hidden backdoor behavior in a machine learning model and subsequently generate targeted training data that can be used to repair the backdoor while preserving normal model performance.**

---

## Experimental Flow

```text
Train Normal Model
       |
       v
Create Backdoored Model
       |
       v
Verify Backdoor
       |
       v
Freeze Model Parameters
       |
       v
Run Neural Input Optimization
       |
       v
Discover Backdoor Inputs
       |
       v
Generate NIO Repair Dataset
       |
       v
Fine-Tune Backdoored Model
       |
       v
Test Original Backdoor Again
       |
       +----------------------+
       |                      |
       v                      v
Backdoor Removed?      Normal Accuracy
                              |
                              v
                         Preserved?
```

## Research Area

**ML Backdoor Discovery and Repair**

This project investigates Neural Input Optimization as a method for both discovering hidden machine learning model behavior and generating targeted data for model repair.
