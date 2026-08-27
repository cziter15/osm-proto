import torch
import torch.nn as nn
import torch.nn.functional as F


def parallel_quantum_operator_scan(
    a: torch.Tensor,
    b_op: torch.Tensor,
) -> torch.Tensor:
    """Inclusive prefix scan for ``Z_t = a_t * Z_{t-1} + B_t``."""
    length = a.size(1)
    if length == 1:
        return b_op

    a_cum = a
    b_cum = b_op

    for level in range((length - 1).bit_length()):
        stride = 1 << level
        if stride >= length:
            break

        a_left = a_cum[:, :-stride]
        b_left = b_cum[:, :-stride]
        a_right = a_cum[:, stride:]
        b_right = b_cum[:, stride:]

        a_combined = a_right * a_left
        b_combined = a_right * b_left + b_right

        a_cum = torch.cat([a_cum[:, :stride], a_combined], dim=1)
        b_cum = torch.cat([b_cum[:, :stride], b_combined], dim=1)

    return b_cum


class QuantumOSMBlock(nn.Module):
    """Current Q-OSM block: causal local convolution + complex operator scan."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        d_k: int = 16,
        d_v: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or d_k <= 0 or d_v <= 0 or d_conv <= 0:
            raise ValueError("all dimensions must be positive")

        self.dim = dim
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_conv = d_conv
        self.d_inner = num_heads * d_v

        self.norm = nn.RMSNorm(dim)

        self.proj_q = nn.Linear(dim, num_heads * d_k * 2, bias=False)
        self.proj_k = nn.Linear(dim, num_heads * d_k * 2, bias=False)
        self.proj_v = nn.Linear(dim, num_heads * d_v * 2, bias=False)
        self.proj_gate = nn.Linear(dim, self.d_inner, bias=False)

        self.proj_phi = nn.Linear(dim, num_heads * d_k, bias=False)
        self.decay_params = nn.Parameter(torch.empty(num_heads, 1, d_k))

        self.conv1d = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=d_conv,
            groups=dim,
            padding=d_conv - 1,
            bias=True,
        )
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        with torch.no_grad():
            self.decay_params.uniform_(-4.0, -2.0)

    def _evolution(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        phi = self.proj_phi(x).view(
            batch,
            length,
            self.num_heads,
            1,
            self.d_k,
        )

        # Keep every head strictly inside the unit disk. This is the stable
        # evolution used by the current architecture for long scans.
        rho = torch.exp(-F.softplus(self.decay_params.float()))
        rho = rho.unsqueeze(0).unsqueeze(0).expand_as(phi)
        return torch.polar(rho, phi.float())

    def _project(self, x: torch.Tensor):
        batch, length, _ = x.shape

        def complex_projection(layer: nn.Linear, width: int) -> torch.Tensor:
            raw = layer(x).view(
                batch,
                length,
                self.num_heads,
                width,
                2,
            )
            return torch.complex(raw[..., 0].float(), raw[..., 1].float())

        q = complex_projection(self.proj_q, self.d_k)
        k = F.normalize(complex_projection(self.proj_k, self.d_k), p=2, dim=-1)
        v = complex_projection(self.proj_v, self.d_v)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        length = x.size(1)

        conv = self.conv1d(x_norm.transpose(1, 2))[:, :, :length].transpose(1, 2)
        conv = F.silu(conv)

        q, k, v = self._project(conv)
        b_op = v.unsqueeze(-1) * torch.conj(k.unsqueeze(-2))
        z = parallel_quantum_operator_scan(self._evolution(conv), b_op)

        out = (z * q.unsqueeze(-2)).sum(dim=-1).real
        out = out.reshape(x.size(0), length, self.d_inner)
        gate = F.silu(self.proj_gate(x_norm))
        return residual + self.out_proj(out * gate)

    def step(
        self,
        x_t: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        z_state: torch.Tensor | None = None,
    ):
        """Process one token using constant-size recurrent state."""
        batch, length, dim = x_t.shape
        if length != 1:
            raise ValueError("step() expects shape [batch, 1, dim]")

        x_norm = self.norm(x_t)
        flat = x_norm[:, 0]

        if conv_state is None:
            conv_state = torch.zeros(
                batch,
                dim,
                self.d_conv - 1,
                device=x_t.device,
                dtype=x_t.dtype,
            )

        conv_buffer = torch.cat([conv_state, flat.unsqueeze(-1)], dim=-1)
        new_conv_state = conv_buffer[:, :, 1:]

        weight = self.conv1d.weight[:, 0, :]
        conv = (conv_buffer * weight.unsqueeze(0)).sum(dim=-1)
        if self.conv1d.bias is not None:
            conv = conv + self.conv1d.bias
        conv = F.silu(conv).unsqueeze(1)

        q, k, v = self._project(conv)
        b_op = v.unsqueeze(-1) * torch.conj(k.unsqueeze(-2))
        a = self._evolution(conv)

        if z_state is None:
            z_state = torch.zeros(
                batch,
                1,
                self.num_heads,
                self.d_v,
                self.d_k,
                device=x_t.device,
                dtype=torch.cfloat,
            )

        new_z_state = a * z_state + b_op
        out = (new_z_state * q.unsqueeze(-2)).sum(dim=-1).real
        out = out.reshape(batch, 1, self.d_inner)
        gate = F.silu(self.proj_gate(x_norm))
        y_t = x_t + self.out_proj(out * gate)

        return y_t, new_conv_state, new_z_state


class QuantumOSMForCausalLM(nn.Module):
    """Current Q-OSM causal language model."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        d_k: int = 16,
        d_v: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [
                QuantumOSMBlock(
                    dim=dim,
                    num_heads=num_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_conv=d_conv,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.final_norm(x))

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 10) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have shape [batch, sequence] with sequence > 0")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        conv_states = [None] * len(self.layers)
        z_states = [None] * len(self.layers)
        x_t = None

        for t in range(input_ids.size(1)):
            x_t = self.embedding(input_ids[:, t:t + 1])
            for i, layer in enumerate(self.layers):
                x_t, conv_states[i], z_states[i] = layer.step(
                    x_t,
                    conv_states[i],
                    z_states[i],
                )

        output_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self.lm_head(self.final_norm(x_t))[:, -1]
            next_token = logits.argmax(dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token], dim=-1)

            x_t = self.embedding(next_token)
            for i, layer in enumerate(self.layers):
                x_t, conv_states[i], z_states[i] = layer.step(
                    x_t,
                    conv_states[i],
                    z_states[i],
                )

        return output_ids
