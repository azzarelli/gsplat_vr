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

from typing import Dict, Literal, Optional, Tuple

from torch import Tensor

RenderMode = Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"]
_COLOR_MODES = ("RGB", "RGB+D", "RGB+ED")
_RENDER_MODES = ("RGB", "D", "ED", "RGB+D", "RGB+ED")
# Order of the CUDA-event stages in Rendering.cpp.
_STAGES = ("project", "sh", "isect", "sort", "blend", "ed")


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
    segmented: bool = False,
    stereo: bool = False,
    profile: bool = False,
) -> Tuple[Tensor, Tensor, Dict]:
    """Render C views of N Gaussians. Forward only.

    Returns (render_colors [C, H, W, X], render_alphas [C, H, W, 1], meta).
    X is D colour channels, +1 depth channel for "RGB+D"/"RGB+ED", or just
    the depth channel for "D"/"ED". "D" is alpha-weighted depth; "ED" divides
    it by alpha.

    packed: keep only the (camera, gaussian) pairs that survive culling.
        packed=False keeps every pair, colours them in one fused SH+depth
        kernel and allows `segmented` (per-image) sorting.
    stereo: packed with C == 2 only. Evaluates SH once per gaussian from the
        midpoint of the two cameras and shares it between the eyes.
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

    (
        render_colors,
        render_alphas,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        proj_opacities,
        tiles_per_gauss,
        isect_ids,
        flatten_ids,
        isect_offsets,
        stage_ms,
    ) = _C.rasterization_3dgs(
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
        segmented,
        stereo,
        profile,
    )

    meta = {
        "camera_ids": camera_ids,  # packed only, else None
        "gaussian_ids": gaussian_ids,  # packed only, else None
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": proj_opacities,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "tile_width": isect_offsets.shape[-1],
        "tile_height": isect_offsets.shape[-2],
        "tile_size": tile_size,
        "width": width,
        "height": height,
        "n_cameras": viewmats.shape[0],
        "stage_ms": dict(zip(_STAGES, stage_ms.tolist())),
    }
    return render_colors, render_alphas, meta
