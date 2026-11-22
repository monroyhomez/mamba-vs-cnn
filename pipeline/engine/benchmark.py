"""Benchmark de eficiencia computacional (eje 2 de la tesis). Protocolo IDÉNTICO
para los 7 modelos. Mide a varias resoluciones: parámetros, FLOPs, VRAM pico
(train e inferencia), throughput (FPS) y latencia.

DECISIONES DE PROTOCOLO (ver plan de redefinición del benchmark):

1. PRECISIÓN = bf16 por defecto, IGUAL que el entrenamiento (trainer.py usa autocast
   bfloat16). Correr en fp32 penaliza los kernels selective-scan de Mamba e infla la
   VRAM → OOM ARTIFICIALES que no ocurren al entrenar. fp32 solo como sensibilidad.

2. VRAM: se reportan `allocated` (tensores vivos) y `reserved` (lo que el proceso le
   quita de verdad a la GPU — la cifra honesta para "¿cabe?").

3. LATENCIA se mide a batch=1; THROUGHPUT (FPS) a batch fijo. No mezclar: son cosas
   distintas (latencia = respuesta de 1 imagen; throughput = imágenes/s en saturación).

4. N repeticiones por (modelo, resolución) → media±desv (VRAM/FPS/latencia varían por
   fragmentación del allocator y ruido de kernels).

5. FLOPs: fvcore NO cuenta los kernels CUDA del selective-scan (op custom de Mamba) →
   sub-cuenta ~2x. Se reporta `flops_fvcore_G` (exacto para CNN) y, si se pasa el valor
   del paper, `flops_manual_G` (fvcore-independiente). `flops_source` documenta cuál es.
"""
import contextlib
import csv
import math
import os
import time

import torch


def _peak_alloc_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _peak_reserved_mb():
    return torch.cuda.max_memory_reserved() / (1024 ** 2)


def _autocast(amp_dtype):
    """Contexto de precisión. amp_dtype=None → fp32; si no, autocast (bf16/fp16).
    Debe COINCIDIR con la precisión de entrenamiento (bf16 AMP) para que la eficiencia
    reportada sea justa (fp32 infla la VRAM de los kernels selective-scan → OOM falsos)."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


@torch.no_grad()
def _time_forward(model, x, warmup, iters, amp_dtype):
    """Segundos por forward (promediado). Sin medir memoria aquí."""
    with _autocast(amp_dtype):
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _measure_train_vram(model, x, num_classes, amp_dtype):
    """VRAM pico (allocated, reserved) de un paso forward+backward de entrenamiento."""
    model.train()
    torch.cuda.reset_peak_memory_stats()
    y = torch.randint(0, num_classes, (x.size(0),), device=x.device)
    with _autocast(amp_dtype):
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    alloc, reserved = _peak_alloc_mb(), _peak_reserved_mb()
    model.zero_grad(set_to_none=True)
    return alloc, reserved


def _count_flops(model, x):
    """FLOPs POR IMAGEN (GFLOPs). fvcore suma sobre todo el batch → se divide por
    x.size(0) para dejarlo por imagen y que sea comparable con los valores del paper
    (que siempre se reportan por imagen)."""
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x).total() / 1e9 / x.size(0)
        return flops, "fvcore/img (SSM custom ops NO contados — verificar vs paper)"
    except Exception as e:
        return float("nan"), f"fvcore falló: {e}"


def _mean_std(xs):
    xs = [v for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _fit_scaling_exponent(resolutions, values):
    """Exponente p tal que value ~ res^p (ajuste log-log por mínimos cuadrados).
    Cuantifica el escalado de VRAM con la resolución (esperado ~2: lado espacial)."""
    pts = [(r, v) for r, v in zip(resolutions, values)
           if v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0]
    if len(pts) < 2:
        return float("nan")
    xs = [math.log(r) for r, _ in pts]
    ys = [math.log(v) for _, v in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


CSV_FIELDS = [
    "model", "resolution", "batch_size", "precision", "params_M", "repeats",
    "flops_fvcore_G", "flops_manual_G", "flops_source",
    "vram_train_alloc_MB_mean", "vram_train_alloc_MB_std", "vram_train_reserved_MB_mean",
    "vram_infer_alloc_MB_mean", "vram_infer_reserved_MB_mean",
    "throughput_fps_mean", "throughput_fps_std",
    "latency_bs1_ms_mean", "latency_bs1_ms_std",
    "vram_train_scaling_exp", "status",
]


def run_benchmark(model, name, num_classes, out_csv, device="cuda",
                  resolutions=(224, 384, 512, 768, 1024), batch_size=16,
                  amp_dtype=torch.bfloat16, repeats=3,
                  flops_paper_g_at_224=None, warmup=10, iters=50):
    """Corre el protocolo completo para un modelo y anexa filas al CSV.

    amp_dtype=torch.bfloat16 (default): MISMA precisión que el entrenamiento.
    Pasar amp_dtype=None mide en fp32 (comparación secundaria de sensibilidad).
    flops_paper_g_at_224: FLOPs oficiales del paper a 224² (para Mamba, donde fvcore
    sub-cuenta); se escala ~cuadrático con la resolución como estimación.
    """
    model = model.to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    prec = "fp32" if amp_dtype is None else str(amp_dtype).replace("torch.", "")

    rows = []
    for res in resolutions:
        row = {"model": name, "resolution": res, "batch_size": batch_size,
               "precision": prec, "params_M": round(params_m, 3), "repeats": repeats}

        # FLOPs (una vez por resolución; fvcore es determinista)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        x_flops = torch.randn(batch_size, 3, res, res, device=device)
        try:
            flops, note = _count_flops(model, x_flops)
        except RuntimeError as e:
            flops, note = float("nan"), f"fvcore OOM: {e}"
        del x_flops
        row["flops_fvcore_G"] = round(flops, 3) if not math.isnan(flops) else ""
        if flops_paper_g_at_224 is not None:
            row["flops_manual_G"] = round(flops_paper_g_at_224 * (res / 224.0) ** 2, 3)
            row["flops_source"] = f"paper@224={flops_paper_g_at_224}G, escalado (res/224)^2"
        else:
            row["flops_manual_G"] = ""
            row["flops_source"] = note

        t_alloc, t_res, i_alloc, i_res, fps, lat = [], [], [], [], [], []
        status = "ok"
        for _ in range(repeats):
            try:
                torch.cuda.empty_cache()
                x = torch.randn(batch_size, 3, res, res, device=device)
                a, r = _measure_train_vram(model, x, num_classes, amp_dtype)
                t_alloc.append(a); t_res.append(r)

                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                model.eval()
                dt = _time_forward(model, x, warmup, iters, amp_dtype)
                fps.append(x.size(0) / dt)
                i_alloc.append(_peak_alloc_mb()); i_res.append(_peak_reserved_mb())

                # latencia a batch=1 (respuesta de 1 imagen, no throughput)
                x1 = torch.randn(1, 3, res, res, device=device)
                dt1 = _time_forward(model, x1, warmup, iters, amp_dtype)
                lat.append(dt1 * 1000.0)
                del x, x1
            except RuntimeError as e:
                status = "OOM" if "out of memory" in str(e).lower() else f"error: {e}"
                torch.cuda.empty_cache()
                break

        ta_m, ta_s = _mean_std(t_alloc)
        row["vram_train_alloc_MB_mean"] = round(ta_m, 1)
        row["vram_train_alloc_MB_std"] = round(ta_s, 1)
        row["vram_train_reserved_MB_mean"] = round(_mean_std(t_res)[0], 1)
        row["vram_infer_alloc_MB_mean"] = round(_mean_std(i_alloc)[0], 1)
        row["vram_infer_reserved_MB_mean"] = round(_mean_std(i_res)[0], 1)
        f_m, f_s = _mean_std(fps)
        row["throughput_fps_mean"] = round(f_m, 2)
        row["throughput_fps_std"] = round(f_s, 2)
        l_m, l_s = _mean_std(lat)
        row["latency_bs1_ms_mean"] = round(l_m, 3)
        row["latency_bs1_ms_std"] = round(l_s, 3)
        row["status"] = status
        print(row)
        rows.append(row)

    # exponente de escalado de VRAM_train(res) — se anota igual en todas las filas del modelo
    exp = _fit_scaling_exponent(
        [r["resolution"] for r in rows],
        [r["vram_train_alloc_MB_mean"] if r["status"] == "ok" else None for r in rows],
    )
    for r in rows:
        r["vram_train_scaling_exp"] = round(exp, 3) if not math.isnan(exp) else ""

    write_header = not _file_has_header(out_csv)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    return rows


def _file_has_header(path):
    return os.path.exists(path) and os.path.getsize(path) > 0
