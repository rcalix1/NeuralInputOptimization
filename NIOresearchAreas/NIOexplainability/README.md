## NIO Explainability 



# Neural Ensemble for Cybersecurity Tool Selection

## Overview

The goal of this experiment is to improve cybersecurity tool selection by combining predictions from two different classifiers:

1. **Support Vector Machine (SVM)** — a supervised machine learning classifier.
2. **Llama 3.2** — a large language model used to predict the appropriate cybersecurity tool.

We train a small neural network to determine which classifier to trust for a given prediction.

The ensemble does not generate a new tool prediction. Instead, it selects either the SVM or Llama 3.2 prediction.

## 1. Dataset

Our dataset contains **1,500 natural-language cybersecurity prompts** distributed across 11 tool classes:

- `BlockIP`
- `CheckFailedLogins`
- `CheckVulnerability`
- `GetSystemInfo`
- `ListProcesses`
- `ListeningPorts`
- `NmapScan`
- `PortScan`
- `ReadAuthLog`
- `ReadSyslog`
- `SSHConnect`

For each prompt, we have:

- Ground-truth cybersecurity tool.
- SVM-predicted tool.
- Llama 3.2-predicted tool.

Llama 3.2 can also predict `Unknown` when it does not select one of the 11 tools.

## 2. Ensemble Inputs

The neural ensemble receives the predictions from both classifiers.

Each prediction is converted into a **one-hot encoded vector**.

| Classifier | Number of Features |
|---|---:|
| SVM | 11 |
| Llama 3.2 | 12 |
| **Total** | **23** |

For example, suppose:

- SVM predicts `PortScan`.
- Llama 3.2 predicts `NmapScan`.

We convert each prediction into a one-hot vector and concatenate the two vectors.

The resulting input is:

\[
\mathbf{x}\in\mathbb{R}^{23}
\]

**The ensemble does not receive the original prompt or TF-IDF features.** It uses only the predicted tool labels.

## 3. Neural Network Architecture

We use a simple fully connected neural network:

```text
Input: 23 features
       |
       v
Linear(23, 32)
       |
      ReLU
       |
       v
Linear(32, 16)
       |
      ReLU
       |
       v
Linear(16, 2)
       |
     Softmax
       |
       v
[SVM probability, Llama probability]
```

The network produces two outputs:

\[
\mathbf{p}=\operatorname{softmax}(f_\theta(\mathbf{x}))
\]

where:

\[
\mathbf{p}=[p_{\mathrm{SVM}},p_{\mathrm{Llama}}]
\]

These probabilities represent the neural network's preference for selecting either classifier.

## 4. Training the Ensemble

We use the ground-truth tool labels to construct the neural network's training targets.

The selector has two classes:

- **0:** Select SVM.
- **1:** Select Llama 3.2.

We assign target 1 only when Llama 3.2 is correct and SVM is incorrect. All other cases receive target 0.

This produces an imbalanced training dataset because SVM is generally more accurate than Llama 3.2.

To address this imbalance, we use **weighted cross-entropy loss**:

\[
\mathcal{L}=-\sum_{c=0}^{1}w_c y_c\log(p_c)
\]

where \(w_c\) is the weight associated with class \(c\).

The neural network is trained using the Adam optimizer.

## 5. Ensemble Prediction

After training, the neural network estimates the probability of selecting Llama 3.2.

Rather than always selecting the classifier with the highest probability, we introduce a decision threshold \(\tau\).

The final prediction is:

\[
\hat{y}=
\begin{cases}
\hat{y}_{\mathrm{Llama}}, & p_{\mathrm{Llama}}>\tau\\
\hat{y}_{\mathrm{SVM}}, & \text{otherwise}
\end{cases}
\]

The threshold is selected using validation data.

For example:

- SVM predicts `PortScan`.
- Llama 3.2 predicts `NmapScan`.
- Neural network produces \(p_{\mathrm{Llama}}=0.92\).
- Decision threshold is \(\tau=0.90\).

Since \(0.92>0.90\), the ensemble selects `NmapScan`.

## 6. Repeated Cross-Validation

We evaluate the ensemble using **5-fold stratified cross-validation repeated 3 times**, producing 15 evaluations.

For each fold:

1. Divide the data into training and held-out test partitions.
2. Create an internal validation set from the training partition.
3. Train a new neural selector.
4. Select the decision threshold using validation accuracy.
5. Evaluate SVM, Llama 3.2, and the ensemble on the held-out partition.

The ensemble is retrained for every fold.

## 7. Experimental Results

| Model | Mean Accuracy | Mean Macro F1 |
|---|---:|---:|
| Llama 3.2 | 88.87% | 0.8876 |
| SVM | 91.87% | 0.9186 |
| **Neural Ensemble** | **92.93%** | **0.9288** |

The neural ensemble improves mean accuracy by approximately **1.07 percentage points** compared with SVM alone.

Across 15 evaluations:

- Ensemble outperformed SVM in **11 folds**.
- Ensemble matched SVM in **3 folds**.
- Ensemble performed below SVM in **1 fold**.
- The mean selected decision threshold was **0.836**.

The selector recovered 136 incorrect SVM predictions by selecting Llama 3.2, while introducing 88 incorrect predictions, producing a net gain of 48 correct predictions across the 15 evaluations.

Because repeated cross-validation evaluates some observations more than once, these totals are aggregated across evaluations rather than unique prompts.

## 8. Discussion

The results demonstrate that SVM and Llama 3.2 have complementary prediction capabilities.

Although SVM is more accurate overall, Llama 3.2 correctly classifies some prompts that SVM misclassifies.

The neural ensemble learns patterns in the combinations of predicted tool labels and uses these patterns to determine which classifier is more likely to be correct.

For example, when SVM predicts `ListeningPorts` and Llama predicts `PortScan`, the network may learn which classifier is more reliable for that particular combination.

Importantly, the ensemble achieves its improvement using only **23 input features**, without processing the original natural-language prompt.

The results suggest that a relatively small neural network can improve cybersecurity tool selection by learning when to trust predictions from different machine learning models.

## 9. Conclusion

We developed a neural ensemble that combines predictions from an SVM classifier and Llama 3.2 for cybersecurity tool selection.

The ensemble uses a 23-input neural network with two hidden layers and a two-output selector. A validation-selected threshold determines whether to use the SVM or Llama prediction.

Across repeated cross-validation, the ensemble achieved **92.93% mean accuracy**, compared with **91.87% for SVM** and **88.87% for Llama 3.2**.

These results demonstrate the potential of learned classifier selection for improving cybersecurity tool prediction.
