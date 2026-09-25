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

namespace gsplat
{
struct RasterizeResult
{
    at::Tensor renders; // [I, H, W, channels]
    at::Tensor alphas;  // [I, H, W, 1]
};

// Front-to-back alpha compositing, one CTA per tile. Dense inputs are
// [I, N, ...]; packed inputs are [nnz, ...] and flatten_ids index rows.
RasterizeResult rasterize_to_pixels_3dgs(
    const at::Tensor &means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor &conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor &colors,                    // [..., N, channels] or [nnz, channels]
    const at::Tensor &opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> &backgrounds, // [..., channels]
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &isect_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids    // [n_isects]
);

void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width], skip tiles set to false
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor isect_offsets,
    const at::Tensor flatten_ids,
    // outputs
    at::Tensor renders,
    at::Tensor alphas,
    at::Tensor last_ids // [..., H, W] index of the last contributing entry (backward only)
);
} // namespace gsplat
