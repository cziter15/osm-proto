<div align="center">

# OSM: Operator Scan Model
### Sub-Quadratic Sequence Modeling via Complex Associative Affine Scans

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Experimental](https://img.shields.io/badge/Status-Research%20Preview-purple.svg)]()

**Author:** [Krzysztof Strehlau](https://github.com/cziter15) &bull; **Year:** 2026

</div>

---

## ⚡ TL;DR

**Operator Scan Model (OSM)** is a linear-time sequence architecture designed to eliminate the quadratic bottleneck of self-attention without sacrificing expressiveness or parallel training efficiency. 

OSM replaces pairwise attention matrices with a **complex-valued diagonal affine recurrence** computed in parallel using an **associative Hillis–Steele prefix scan**:

$$\Large z_t = a_t \odot z_{t-1} + b_t, \quad a_t = \rho_t e^{i\phi_t}, \quad z_t, b_t \in \mathbb{C}^d$$

* 🚀 **$O(T \log T)$ parallel training work** with $O(\log T)$ scan depth.
* ⚡ **$O(1)$ constant memory & latency** per generated token during inference (no growing KV cache).
* 🌊 **Complex polar dynamics** ($\rho$ magnitude decay + $\phi$ phase rotation) naturally handle positional encoding, periodicity, and decay.

---

## 📊 Comparison at a Glance

| Architecture | Training Work | Parallel Depth | Inference State | KV Cache Growth | Explicit Positional Encoding |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard Transformer** | $O(T^2)$ | $O(1)$ | $O(T)$ | **Linear $O(T)$** | Required (RoPE / ALiBi) |
| **Vanilla RNN / LSTM** | $O(T)$ | $O(T)$ | $O(1)$ | **None $O(1)$** | Implicit (Sequential) |
| **Mamba / S4 (SSMs)** | $O(T)$ | $O(\log T)$ | $O(1)$ | **None $O(1)$** | Implicit / Continuous |
| **OSM (Ours)** | **$O(T \log T)$** | **$O(\log T)$** | **$O(1)$** | **None $O(1)$** | **Implicit (via $\phi_t$ phase)** |

---

## 📐 Mathematical Foundations

### 1. Complex-Valued Diagonal Dynamics
In standard real-valued linear recurrences, scalar multipliers can only scale values (causing either exponential decay or numerical explosion). 

OSM models state transitions over the complex domain $\mathbb{C}^d$:
$$a_t = \rho_t \odot \exp(i \phi_t)$$

- **Damping / Retention Rate ($\rho_t \in (0, 1)$):** Controls the memory horizon.
- **Phase Rotation ($\phi_t \in [-\pi, \pi]$):** Encodes wave-like interference and relative positional offsets directly in frequency space.
- **Input Injection ($b_t \in \mathbb{C}^d$):** Token embedding projected into the complex hidden manifold.

Unrolling the recurrence from $t=0$ reveals how past tokens decay and rotate:
$$z_t = \sum_{j=1}^t \left( \prod_{k=j+1}^t a_k \right) b_j = \sum_{j=1}^t \left( \prod_{k=j+1}^t \rho_k \right) \exp\left(i \sum_{k=j+1}^t \phi_k\right) b_j$$

---

### 2. Associative Affine Composition
Each token emits an affine transformation $f_t(z) = a_t z + b_t$. The composition of two sequential operators $(a_1, b_1)$ followed by $(a_2, b_2)$ is:

$$f_2(f_1(z)) = a_2(a_1 z + b_1) + b_2 = (a_2 a_1) z + (a_2 b_1 + b_2)$$

This defines the binary associative operator $\circ$:
$$(a_2, b_2) \circ (a_1, b_1) = \Big(a_2 \odot a_1, \; a_2 \odot b_1 + b_2\Big)$$

$$\text{Associativity holds: } \big( (a_3, b_3) \circ (a_2, b_2) \big) \circ (a_1, b_1) = (a_3, b_3) \circ \big( (a_2, b_2) \circ (a_1, b_1) \big)$$

---

## 🔄 Parallel Prefix Scan (Hillis–Steele)

Because affine composition is associative, we evaluate all prefix states in parallel across GPU threads using the **Hillis–Steele scan algorithm**:

```text
Sequence tokens:          [1]          [2]          [3]          [4]
Initial operators:      (a1, b1)     (a2, b2)     (a3, b3)     (a4, b4)
                           │            │            │            │
Step 1 (stride 1):         │       (1)──┴──►(2)      │       (3)──┴──►(4)
                           │         (a2a1, ...)     │         (a4a3, ...)
                           │              │          │              │
Step 2 (stride 2):         │              │     (1)──┴─────────────►(3)
                           │              │            (2)──────────┴──►(4)
                           ▼              ▼              ▼              ▼
Final Prefixes:         z_1=b1       z_2=a2b1+b2    z_3=...        z_4=...
```

* **Span / Depth:** $\log_2(T)$ parallel steps (ideal for SIMD GPU execution).
* **Total Work:** $O(T \log T)$ operator multiplications.

---

## 🏗️ Architecture Anatomy

```text
               Token Embeddings (x)
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
   Pre-RMSNorm                       Pre-RMSNorm
       │                                 │
  Linear Projections                 Linear (W_gate, W_up)
       │                                 │
 [ρ_t, φ_t, b_t]                         │
       │                                 │
 a_t = ρ_t * exp(i φ_t)                  │
       │                                 │
 Associative Affine Scan                 │
       │                                 │
 Extract Re(z_t)                         │
       │                                 │
  Output Projection                      │
       │                                 │
       ▼                                 ▼
  Residual Add ──────────────────►  Residual Add ──► Next Layer
```

### Learnable Forgetting Initialization
To prevent early vanishing gradients and allow the model to start as a near-lossless accumulator:
$$\rho_{\text{init}} = \sigma(4.5) \approx 0.989$$

The model starts with long-term retention and selectively learns to forget context during training.

---

## 🚀 Minimal PyTorch Implementation

A complete, self-contained implementation of the **Operator Scan Layer**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexAffineScan(torch.autograd.Function):
    """Parallel Hillis-Steele Associative Scan over Complex Affine Pairs."""
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor):
        # a: [B, T, D] (complex), b: [B, T, D] (complex)
        B, T, D = a.shape
        num_steps = (T - 1).bit_length()
        
        a_cum = a.clone()
        b_cum = b.clone()
        
        for step in range(num_steps):
            stride = 1 << step
            if stride >= T:
                break
            
            # Left operands: [:, :-stride]
            # Right operands: [:, stride:]
            a_left = a_cum[:, :-stride]
            b_left = b_cum[:, :-stride]
            
            a_right = a_cum[:, stride:]
            b_right = b_cum[:, stride:]
            
            # Affine composition: (a_r * a_l, a_r * b_l + b_r)
            a_cum[:, stride:] = a_right * a_left
            b_cum[:, stride:] = a_right * b_left + b_right

        return b_cum


class OSMBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = nn.RMSNorm(dim)
        
        # Projections for rho, phi (phase), and b (complex input)
        self.proj_rho = nn.Linear(dim, dim)
        self.proj_phi = nn.Linear(dim, dim)
        self.proj_b_real = nn.Linear(dim, dim)
        self.proj_b_imag = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim, dim)
        
        # Initialize rho close to 1.0 (sigmoid(4.5) ~ 0.989)
        nn.init.constant_(self.proj_rho.bias, 4.5)
        nn.init.normal_(self.proj_rho.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        x_norm = self.norm(x)
        
        # 1. Compute complex multiplier: a = rho * exp(i * phi)
        rho = torch.sigmoid(self.proj_rho(x_norm))
        phi = self.proj_phi(x_norm)
        a = torch.polar(rho, phi)  # Creates complex tensor
        
        # 2. Compute complex input: b = b_real + i * b_imag
        b = torch.complex(self.proj_b_real(x_norm), self.proj_b_imag(x_norm))
        
        # 3. Parallel Associative Prefix Scan
        z = ComplexAffineScan.apply(a, b)
        
        # 4. Project real component back to hidden dimension
        y = self.out_proj(z.real)
        return residual + y

    def step(self, x_t: torch.Tensor, state: torch.Tensor | None = None):
        """O(1) Autoregressive recurrent step."""
        # x_t: [B, 1, D], state: [B, 1, D] (complex)
        if state is None:
            state = torch.zeros(x_t.shape[0], 1, self.dim, dtype=torch.cfloat, device=x_t.device)
            
        x_norm = self.norm(x_t)
        rho = torch.sigmoid(self.proj_rho(x_norm))
        phi = self.proj_phi(x_norm)
        a_t = torch.polar(rho, phi)
        b_t = torch.complex(self.proj_b_real(x_norm), self.proj_b_imag(x_norm))
        
        # Constant-time recurrent update
        new_state = a_t * state + b_t
        y_t = self.out_proj(new_state.real)
        return x_t + y_t, new_state
```

---

## 🧪 Usage Example

```python
import torch

# Initialize model layer
layer = OSMBlock(dim=256).cuda()
tokens = torch.randn(2, 1024, 256, device="cuda") # [Batch=2, SeqLen=1024, Dim=256]

# --- Mode 1: Parallel Training Pass ---
output_train = layer(tokens)
print("Training output shape:", output_train.shape)  # [2, 1024, 256]

# --- Mode 2: O(1) Autoregressive Generation ---
state = None
generated_tokens = []
x_t = tokens[:, :1, :]  # First token prompt

for _ in range(10):
    y_t, state = layer.step(x_t, state)
    generated_tokens.append(y_t)
    x_t = y_t  # Feed back for next step

print("Generated steps:", len(generated_tokens))
```

---

## 🔬 Theoretical Deep-Dive

### Why Hillis–Steele instead of Blelloch?
- **Hillis–Steele (Work-Inefficient Scan):** Takes $O(T \log T)$ operations and $O(\log T)$ steps. On modern GPUs with thousands of compute cores, the SIMD step-efficiency and non-branching data access often outperform work-efficient algorithms (like Blelloch) for sequences up to $T \approx 16\text{k}$.
- **Blelloch (Work-Efficient Scan):** Takes $O(T)$ operations but requires a two-pass sweep (Up-Sweep and Down-Sweep), introducing extra memory barrier overheads.

### Connection to State Space Models (SSMs) & LRUs
OSM belongs to the family of **Diagonal Linear Recurrent Networks** (such as Linear Recurrent Units - LRU). By constraining the recurrent matrix to a diagonal complex representation, we retain the state capacity of higher-dimensional continuous SSMs while maintaining strict associative prefix parallelism.

---

## 📜 Citation

If you incorporate OSM into your work, please cite:

```bibtex
@software{strehlau2026osm,
  author       = {Krzysztof Strehlau},
  title        = {OSM: Operator Scan Model},
  year         = {2026},
  url          = {https://github.com/cziter15}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
