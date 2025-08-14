# Neural Input Optimization

Neural Input Optimization (NIO) is a technique that uses a differentiable surrogate model to reverse-engineer optimal input values given desired outputs and constraints. Rather than relying on black-box search or trial-and-error methods, NIO trains a forward model—typically a neural network—to learn the mapping from input variables to output targets. Once trained, this model is used in reverse: it enables gradient-based optimization of inputs by minimizing a constraint-aware loss function, effectively “inverting” the system to find suitable inputs that yield the specified outputs. The method is particularly useful when the true forward process is non-differentiable, computationally expensive, or proprietary.
NIO has broad applicability across domains such as engineering, finance, healthcare, and material science, where stakeholders often know what they want as outputs (e.g., performance goals, target prices, biological responses) but must determine feasible inputs under real-world constraints. For example, a veterinary use case might involve identifying safe and effective dosage combinations of antibiotics based on desired recovery outcomes and species-specific tolerances. In this way, NIO acts as a decision support layer that enables fast, controlled experimentation across complex, multivariable systems—without needing full analytical inverses.

## Links

* https://github.com/giovanniMen/CPCaD-Bench/tree/main
* 
