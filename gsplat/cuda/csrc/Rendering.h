/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <tuple>

namespace gsplat
{
// (render_colors, render_alphas,
//  camera_ids, gaussian_ids,                   -- packed only, else undefined
//  radii, means2d, depths, conics, opacities,  -- projected gaussians
//  tiles_per_gauss, isect_ids, flatten_ids, isect_offsets,
//  stage_ms)                                   -- [6] CPU float32 if profile, else empty
using RasterizationOutputs = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

// Full forward pass for C cameras: project -> colour -> tile intersect/sort ->
// rasterize. See gsplat/rendering.py for the argument semantics.
RasterizationOutputs rasterization_3dgs(
    const at::Tensor &means,                // [N, 3]
    const at::Tensor &quats,                // [N, 4]
    const at::Tensor &scales,               // [N, 3]
    const at::Tensor &opacities,            // [N]
    const at::optional<at::Tensor> &colors, // SH [N, K, D], or colours [N, D] / [C, N, D]; none for depth-only
    const at::Tensor &viewmats,             // [C, 4, 4]
    const at::Tensor &Ks,                   // [C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    const at::optional<at::Tensor> &backgrounds, // [C, D]
    int64_t sh_degree,                           // < 0: colors are already activated
    bool append_depth,
    bool expected_depth,
    bool packed,
    bool segmented,
    bool stereo,
    bool profile
);
} // namespace gsplat
