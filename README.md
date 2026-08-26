# OSM: Operator Scan Model

**Operator Scan Model (OSM)** is a highly efficient, sub-quadratic sequence modeling architecture based on **complex-valued associative affine scans**.

Instead of standard attention, OSM uses a parallelizable **Hillis–Steele prefix scan** over complex diagonal affine operators. This gives the model:

- **\(O(T \log T)\)** parallel training work
- **\(O(1)\)** recurrent state size during autoregressive inference
- Transformer-style residual blocks without quadratic self-attention
- Complex-valued dynamics that combine decay and phase rotation

> **Author:** Krzysztof Strehlau (2026)  
> **Repository:** [github.com/cziter15](https://github.com/cziter15)

---

## Architecture Overview

The core component of OSM is the **`OperatorBlock`**.

Rather than representing sequence aggregation as attention between token pairs, each token emits an affine operator consisting of:

$$
a_t = \rho_t e^{i\phi_t}
$$

where:

- \(\rho_t \in (0,1)\) is the retention / decay rate
- \(\phi_t\) is the phase rotation

and:

$$
b_t \in \mathbb{C}^{d}
$$

is the complex-valued information injected at timestep \(t\).

The hidden-state recurrence is:

$$
z_t = a_t z_{t-1} + b_t
$$

with:

$$
z_t \in \mathbb{C}^{d}
$$

---

## Associative Affine Composition

Each timestep can be interpreted as an affine transformation:

$$
f_t(z) = a_t z + b_t
$$

Two affine operators compose as:

$$
(a_2, b_2) \circ (a_1, b_1)
=
(a_2 a_1,\; a_2 b_1 + b_2)
$$

This composition is **associative**, which means the sequential recurrence can be evaluated with a parallel prefix-scan algorithm instead of strictly from left to right.

OSM uses a **Hillis–Steele scan** to compute all prefix compositions in parallel.

Conceptually:

```text
Token operators:
O1    O2    O3    O4    O5    O6    O7    O8

Hillis–Steele scan:
step 1: compose distance 1
step 2: compose distance 2
step 3: compose distance 4
...

Result:
prefix(O1)
prefix(O1..O2)
prefix(O1..O3)
...
prefix(O1..OT)
```

This bridges two useful regimes:

- **Training:** parallel sequence evaluation
- **Inference:** recurrent constant-state decoding

---

## Key Features

### Complex-Valued Representations

OSM represents its recurrent state in the complex domain.

The multiplicative operator

$$
a_t = \rho_t e^{i\phi_t}
$$

contains both:

- **magnitude decay** through \(\rho_t\)
- **phase rotation** through \(\phi_t\)

This gives the recurrence a natural mechanism for representing periodic, oscillatory, and position-sensitive behavior without requiring explicit positional methods such as RoPE or ALiBi.

---

### Learnable Forgetting

The retention parameter is initialized using a positive bias:

$$
\rho = \sigma(4.5) \approx 0.989
$$

so the model begins training as a near-perfect accumulator.

Instead of starting with aggressive forgetting, the network learns when information should decay.

This initialization is intended to support:

- long initial memory
- stable early-training gradients
- learned task-dependent forgetting

---

### Sub-Quadratic Parallelism

Standard dense self-attention requires pairwise token interactions and scales quadratically with sequence length:

$$
O(T^2)
$$

OSM instead evaluates the recurrence through an associative Hillis–Steele scan.

The scan uses logarithmic parallel depth and \(O(T \log T)\) total scan work:

$$
O(T \log T)
$$

while preserving the underlying recurrent formulation.

---

### Constant-State Inference

During autoregressive generation, there is no need to recompute the full scan.

Only the current recurrent state must be retained:

$$
z_t = a_t z_{t-1} + b_t
$$

Therefore inference requires a fixed-size recurrent state independent of sequence length:

$$
O(1)
$$

with respect to the number of previously processed tokens.

---

### Transformer-Like Block Design

OSM keeps the convenient structure of modern Transformer blocks while replacing self-attention with the operator scan.

A typical block contains:

```text
Input
  │
  ├── Pre-Norm
  │
  ├── Operator Scan
  │
  └── Residual Connection
  │
  ├── Pre-Norm
  │
  ├── GELU Feed-Forward Network
  │
  └── Residual Connection
  │
Output
```

This makes OSM compatible with familiar deep-learning design patterns while changing the sequence-mixing mechanism itself.

---

## Complexity

| Property | OSM |
|---|---:|
| Training scan work | \(O(T \log T)\) |
| Parallel scan depth | \(O(\log T)\) |
| Autoregressive recurrent state | \(O(1)\) |
| Pairwise attention matrix | Not required |
| Sequence operator | Complex affine recurrence |

> Complexity statements refer specifically to the scan/recurrent sequence-mixing mechanism. Overall model cost also depends on hidden dimension, projections, feed-forward layers, implementation details, and hardware efficiency.

---

## Why Complex Operators?

A real-valued scalar recurrence can primarily express attenuation or amplification.

A complex-valued multiplier adds another degree of freedom:

$$
a = \rho e^{i\phi}
$$

Multiplying by \(a\) performs both:

1. scaling by \(\rho\)
2. rotation by \(\phi\)

Repeated application produces:

$$
a^k = \rho^k e^{ik\phi}
$$

which naturally combines exponentially decaying memory with oscillatory structure.

This makes complex affine operators attractive for sequence modeling problems involving:

- long-range dependencies
- periodic patterns
- phase-sensitive features
- compact recurrent memory

---

## Minimal Recurrence

A simplified sequential form looks like:

```python
z = 0

for x_t in sequence:
    rho_t, phi_t, b_t = operator_projection(x_t)

    a_t = rho_t * exp(1j * phi_t)
    z = a_t * z + b_t

    y_t = output_projection(z)
```

During training, OSM replaces the sequential loop with an associative parallel scan.

---

## Parallel Composition

For affine operators:

```python
def compose(left, right):
    a1, b1 = left
    a2, b2 = right

    return (
        a2 * a1,
        a2 * b1 + b2,
    )
```

the operation satisfies associativity:

```text
(O3 ∘ O2) ∘ O1 = O3 ∘ (O2 ∘ O1)
```

That property is what makes a parallel prefix scan possible.

---

## Design Goals

OSM is designed around four goals:

1. **Avoid quadratic attention complexity**
2. **Retain efficient parallel training**
3. **Support constant-state autoregressive inference**
4. **Provide expressive long-range dynamics through complex-valued operators**

---

## Comparison at a Glance

| Architecture | Training sequence mixing | Inference memory growth | Pairwise attention |
|---|---:|---:|---|
| Transformer | \(O(T^2)\) | Typically grows with KV cache | Yes |
| Sequential RNN | \(O(T)\) work, sequential depth | \(O(1)\) state | No |
| **OSM** | **\(O(T \log T)\) scan work, \(O(\log T)\) scan depth** | **\(O(1)\) recurrent state** | **No** |

---

## Status

OSM is an experimental sequence-modeling architecture exploring associative complex-valued affine scans as an alternative to standard attention.

The project focuses on the idea that recurrent models do not necessarily need to choose between:

- efficient parallel training, and
- compact recurrent inference.

Associative scans provide a way to obtain both.

---

## Citation

If you use OSM in research or experiments, please cite the repository and author.

```bibtex
@software{strehlau2026osm,
  author = {Krzysztof Strehlau},
  title = {OSM: Operator Scan Model},
  year = {2026},
  url = {https://github.com/cziter15}
}
```

---

## License

See the repository's `LICENSE` file for licensing information.
