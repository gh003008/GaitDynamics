import math
import torch
import torch.nn as nn
from typing import Optional, Iterable, Tuple


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear and adds a low-rank adaptation: y = base(x) + scale * B(A(x)).
    Base weights are frozen; only A and B are trained.
    """
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Register the frozen base weight/bias
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        if r > 0:
            dev = next(base_linear.parameters()).device
            self.lora_A = nn.Linear(self.in_features, r, bias=False).to(dev)
            self.lora_B = nn.Linear(r, self.out_features, bias=False).to(dev)
            # Init: A kaiming, B zeros (so initial delta is zero)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            out = out + delta
        return out

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def is_linear_module(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear)


def replace_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    setattr(parent, child_name, new_module)


def inject_lora(model: nn.Module,
                target_linear_names: Optional[Iterable[str]] = None,
                r: int = 8,
                alpha: int = 16,
                dropout: float = 0.0,
                include_mha_out_proj: bool = True) -> Tuple[int, int]:
    """
    Recursively traverse the model and replace targeted nn.Linear with LoRALinear.
    If target_linear_names is None, apply to common adapter-friendly modules:
      - input_projection, final_layer
      - FiLMTransformerDecoderLayer.linear1, linear2
      - and optionally MultiheadAttention.out_proj
    Returns (num_replaced, num_skipped).
    """
    replaced, skipped = 0, 0

    def wants(name: str, mod: nn.Module) -> bool:
        if target_linear_names is None:
            # Heuristic include set
            return (
                name.endswith('input_projection') or
                name.endswith('final_layer') or
                name.endswith('linear1') or
                name.endswith('linear2')
            )
        for key in target_linear_names:
            if key in name:
                return True
        return False

    for module_name, module in list(model.named_modules()):
        # Handle MultiheadAttention.out_proj specially
        if include_mha_out_proj and isinstance(module, nn.MultiheadAttention):
            parent_path = module_name
            # module.out_proj is a Linear
            base = module.out_proj
            lora = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
            module.out_proj = lora
            replaced += 1

    # Now replace selected Linear modules
    for name, mod in list(model.named_modules()):
        # Skip nested named_modules() impact by only acting on direct attributes below
        pass

    # Second pass: walk immediate children recursively and swap
    def recursive_swap(parent: nn.Module, prefix: str = ''):
        nonlocal replaced, skipped
        for child_name, child in list(parent.named_children()):
            fq_name = f'{prefix}.{child_name}' if prefix else child_name
            if is_linear_module(child) and wants(fq_name, child):
                replace_module(parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1
            else:
                recursive_swap(child, fq_name)

    recursive_swap(model)
    return replaced, skipped


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for m in model.modules():
        if isinstance(m, LoRALinear) and m.r > 0:
            yield from m.lora_A.parameters()
            yield from m.lora_B.parameters()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_lora_weights(model: nn.Module, ckpt_path: str):
    """
    Load a saved LoRA adapter state dict into a model that has already been injected with LoRA.
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('lora_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected
