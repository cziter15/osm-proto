import torch
import torch.nn as nn
import torch.nn.functional as F


class OperatorBlock(nn.Module):
    """Associative diagonal-complex affine scan block.

    Each token emits z -> a*z + b, with a = rho*exp(i*phi).
    Prefix composition is evaluated with a Hillis-Steele style scan.
    """
    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even")
        h = d_model // 2
        self.h = h
        self.norm1 = nn.LayerNorm(d_model)
        self.phase = nn.Linear(d_model, h)
        self.retention = nn.Linear(d_model, h)
        self.inject = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.retention.bias, 4.5)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.norm1(x)
        T = x.size(1)
        H = self.h
        phi = self.phase(u)
        rho = torch.sigmoid(self.retention(u))
        ar = rho * torch.cos(phi)
        ai = rho * torch.sin(phi)
        z = torch.tanh(self.inject(u))
        br, bi = z[..., :H], z[..., H:]

        offset = 1
        while offset < T:
            A, I, R, J = ar, ai, br, bi
            a2r, a2i = A[:, offset:], I[:, offset:]
            a1r, a1i = A[:, :-offset], I[:, :-offset]
            b2r, b2i = R[:, offset:], J[:, offset:]
            b1r, b1i = R[:, :-offset], J[:, :-offset]
            ar = torch.cat([A[:, :offset], a2r*a1r - a2i*a1i], dim=1)
            ai = torch.cat([I[:, :offset], a2r*a1i + a2i*a1r], dim=1)
            br = torch.cat([R[:, :offset], a2r*b1r - a2i*b1i + b2r], dim=1)
            bi = torch.cat([J[:, :offset], a2r*b1i + a2i*b1r + b2i], dim=1)
            offset *= 2

        state = torch.cat([br, bi], dim=-1)
        x = x + self.proj(state)
        return x + self.ff(self.norm2(x))


class OSM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([OperatorBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.embedding(idx)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.norm(h))


class ImprovedOSMBlock(nn.Module):
    """Gated local-conv complex scan block.

    The convolution is explicitly left-padded (causal), and the scan uses
    functional prefix composition instead of in-place writes so bf16/autograd
    remain safe.  This is intentionally a separate block so old checkpoints
    retain the original OSM architecture.
    """
    def __init__(self, dim: int, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = dim * expand
        self.norm = nn.RMSNorm(dim)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=0)
        self.nu_param = nn.Parameter(torch.empty(self.d_inner))
        self.proj_phi = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.proj_b_real = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.proj_b_imag = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        nn.init.uniform_(self.nu_param, -3.0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        branch, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        conv_in = F.pad(branch.transpose(1, 2), (self.conv1d.kernel_size[0] - 1, 0))
        conv = F.silu(self.conv1d(conv_in).transpose(1, 2))
        # torch.polar has no bf16 kernel on current CUDA builds; keep the
        # small complex recurrence in fp32 and let the surrounding block use
        # autocast for the linear/convolution work.
        rho = torch.exp(-F.softplus(self.nu_param.float()))
        phi = self.proj_phi(conv).float()
        a = torch.polar(rho.expand_as(phi), phi)
        b = torch.complex(self.proj_b_real(conv).float(), self.proj_b_imag(conv).float())
        z = self._chunked_scan(a, b) if a.size(1) > 2048 else self._parallel_scan(a, b)
        return residual + self.out_proj(z.real * F.silu(gate))

    @staticmethod
    def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        _, result = ImprovedOSMBlock._parallel_scan_pair(a, b)
        return result

    @staticmethod
    def _parallel_scan_pair(a: torch.Tensor, b: torch.Tensor):
        length = a.size(1)
        levels = (length - 1).bit_length()
        aa, bb = a, b
        for level in range(levels):
            stride = 1 << level
            if stride >= length:
                break
            a_prev = torch.cat([torch.ones_like(aa[:, :stride]), aa[:, :-stride]], dim=1)
            b_prev = torch.cat([torch.zeros_like(bb[:, :stride]), bb[:, :-stride]], dim=1)
            # Compose (a_current, b_current) after (a_previous, b_previous).
            # Preserve a_current: using the updated cumulative A here is an
            # incorrect affine composition and destabilizes long scans.
            a_current, b_current = aa, bb
            aa = a_current * a_prev
            bb = a_current * b_prev + b_current
        return aa, bb

    @staticmethod
    def _chunked_scan(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
        """Scan long sequences in bounded chunks while carrying state forward.

        The associative pair (A, B) for each chunk is composed with the
        previous chunk's final state. This keeps the scan workspace bounded by
        ``chunk_size`` and supports million-token inference without a million
        token prefix tensor.
        """
        carry = torch.zeros_like(b[:, :1])
        outputs = []
        for start in range(0, a.size(1), chunk_size):
            aa, bb = ImprovedOSMBlock._parallel_scan_pair(
                a[:, start:start + chunk_size], b[:, start:start + chunk_size])
            chunk = bb + aa * carry
            outputs.append(chunk)
            carry = chunk[:, -1:]
        return torch.cat(outputs, dim=1)


class ImprovedOSM(nn.Module):
    """OSM language model using ImprovedOSMBlock."""
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 8,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            ImprovedOSMBlock(d_model, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.embedding(idx)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.norm(h))
