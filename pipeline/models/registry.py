"""Construcción de modelos. CNN salen de timm; los backbones Mamba se envuelven
desde sus repos oficiales (clonados en third_party/, ver ENVIRONMENT.md).

REGLA: no reimplementar bloques. Cada backbone Mamba carga su checkpoint ImageNet
oficial y solo se le sustituye la cabeza de clasificación por num_classes.

Cada función `_build_*` usa la firma real del constructor del repositorio oficial
correspondiente (VMamba, Vim, EfficientVMamba, VSSD)."""
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# CNN baselines (timm — preentrenadas en ImageNet, sin dependencias externas)
# --------------------------------------------------------------------------- #
_TIMM_NAMES = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_t": "convnext_tiny",      # opcional
}


def _build_timm(timm_name, num_classes, pretrained):
    import timm
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)


# --------------------------------------------------------------------------- #
# Backbones Mamba (repos oficiales). Patrón: construir backbone -> cargar ckpt
# ImageNet -> reemplazar cabeza. Rellenar la firma real por modelo.
# --------------------------------------------------------------------------- #
def _load_official_ckpt(model, ckpt_path):
    if not ckpt_path:
        return model
    import os
    sd = torch.load(os.path.expanduser(ckpt_path), map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] cargado {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
    return model


def _replace_head(model, num_classes):
    """Sustituye la última Linear por num_classes preservando in_features."""
    for attr in ("head", "fc", "classifier"):
        head = getattr(model, attr, None)
        if isinstance(head, nn.Linear):
            setattr(model, attr, nn.Linear(head.in_features, num_classes))
            return model
        if isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
            head[-1] = nn.Linear(head[-1].in_features, num_classes)
            return model
    raise RuntimeError("No se encontró cabeza Linear estándar; ajustar _replace_head para este backbone.")


def _build_vmamba_t(num_classes, pretrained, ckpt):
    # TODO INTEGRACIÓN (repo MzeroMiko/VMamba):
    #   from third_party.VMamba.classification.models.vmamba import VSSM
    #   model = VSSM(depths=[2,2,9,2], dims=96, ...)   # config oficial vmamba_tiny
    raise NotImplementedError("Integrar VMamba-T desde third_party/VMamba (ver ENVIRONMENT.md).")


def _build_vim_s(num_classes, pretrained, ckpt):
    # TODO INTEGRACIÓN (repo hustvl/Vim):
    #   from third_party.Vim.vim.models_mamba import vim_small_patch16_224
    raise NotImplementedError("Integrar Vim-S desde third_party/Vim.")


def _build_efficientvmamba_s(num_classes, pretrained, ckpt):
    # TODO INTEGRACIÓN (repo TerryPei/EfficientVMamba)
    raise NotImplementedError("Integrar EfficientVMamba-S desde third_party/EfficientVMamba.")


def _build_vssd_micro(num_classes, pretrained, ckpt):
    # TODO INTEGRACIÓN (repo YuHengsss/VSSD)
    raise NotImplementedError("Integrar VSSD desde third_party/VSSD.")


_MAMBA_BUILDERS = {
    "vmamba_t": _build_vmamba_t,
    "vim_s": _build_vim_s,
    "efficientvmamba_s": _build_efficientvmamba_s,
    "vssd_micro": _build_vssd_micro,
}


def build_model(name, num_classes, pretrained=True, pretrained_ckpt=None):
    if name in _TIMM_NAMES:
        return _build_timm(_TIMM_NAMES[name], num_classes, pretrained)
    if name in _MAMBA_BUILDERS:
        return _MAMBA_BUILDERS[name](num_classes, pretrained, pretrained_ckpt)
    raise ValueError(f"Modelo desconocido: {name}. Registrados: "
                     f"{list(_TIMM_NAMES) + list(_MAMBA_BUILDERS)}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
