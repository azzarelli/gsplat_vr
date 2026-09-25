# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Literal, Optional, Sequence, Tuple

from torch import Tensor

RenderMode = Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"]
_COLOR_MODES = ("RGB", "RGB+D", "RGB+ED")
_RENDER_MODES = ("RGB", "D", "ED", "RGB+D", "RGB+ED")
# Order of the CUDA-event stages in Rendering.cpp.
_STAGES = ("project", "sh", "isect", "sort", "blend")


def rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4] (w, x, y, z), normalised in the kernel
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Optional[Tensor],  # SH [N, K, D] with sh_degree, else [N, D] or [C, N, D]
    viewmats: Tensor,  # [C, 4, 4] world-to-camera
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,  # [C, D]
    render_mode: RenderMode = "RGB",
    antialiased: bool = False,
    stereo: bool = False,
    targets: Optional[Sequence[Tuple[int, int]]] = None,
    profile: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor], Dict]:
    """Render C views of N Gaussians. Forward only.

    Returns (render_colors [C, H, W, X], render_alphas [C, H, W, 1], meta).
    X is D colour channels, +1 depth channel for "RGB+D"/"RGB+ED", or just
    the depth channel for "D"/"ED". "D" is alpha-weighted depth; "ED" divides
    it by alpha. With `targets` the images go there instead and both are None.
    meta: n_entries (projected (camera, gaussian) entries: survivors if packed,
    else C * N), n_isects (gaussian-tile overlaps) and stage_ms.

    packed: keep only the (camera, gaussian) pairs that survive culling.
        packed=False keeps every pair and colours them in one fused SH+depth
        kernel.
    antialiased: scale each opacity by sqrt(det(cov) / det(cov + eps2d * I)),
        the Mip-Splatting 2D filter. Use for models trained with it, with eps2d
        set to the training kernel size.
    stereo: packed with C == 2 only. Evaluates SH once per gaussian from the
        midpoint of the two cameras and shares it between the eyes.
    targets: per camera, (colour, alpha) CUDA array handles to write into,
        e.g. mapped GL textures registered with SURFACE_LDST. Colour is RGBA32F
        (RGB + depth, so "RGB+D"/"RGB+ED" with 3 colour channels), alpha R32F.
    profile: time each stage with CUDA events into meta["stage_ms"] (syncs).
    radius_clip: cull gaussians whose projected radius is at most this (px).
    eps2d: added to the 2D covariance diagonal (low-pass filter).
    tile_size: 16, or 4.
    """
    if render_mode not in _RENDER_MODES:
        raise ValueError(f"render_mode must be one of {_RENDER_MODES}, got {render_mode!r}")
    has_color = render_mode in _COLOR_MODES
    if has_color and colors is None:
        raise ValueError(f"render_mode={render_mode!r} needs colors")

    from .cuda._backend import _C

    render_colors, render_alphas, n_entries, n_isects, stage_ms = _C.rasterization_3dgs(
        means.contiguous(),
        quats.contiguous(),
        scales.contiguous(),
        opacities.contiguous(),
        colors.contiguous() if has_color else None,
        viewmats.contiguous(),
        Ks.contiguous(),
        width,
        height,
        tile_size,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        backgrounds.contiguous() if backgrounds is not None else None,
        sh_degree if (has_color and sh_degree is not None) else -1,
        render_mode != "RGB",  # append depth
        render_mode in ("ED", "RGB+ED"),  # expected depth
        packed,
        antialiased,
        stereo,
        [c for c, _ in targets] if targets else [],
        [a for _, a in targets] if targets else [],
        profile,
    )

    meta = {
        "n_entries": n_entries,
        "n_isects": n_isects,
        "stage_ms": dict(zip(_STAGES, stage_ms.tolist())),
    }
    return render_colors, render_alphas, meta
