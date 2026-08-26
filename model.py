import torch
import torch.nn as nn


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
