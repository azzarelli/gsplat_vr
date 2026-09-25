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

#include "Common.h"

namespace gsplat
{
// EWA splatting of 3D Gaussians to 2D conics, one row per (camera, gaussian).
// Culled entries get radii == 0.
struct ProjectionDenseResult
{
    at::Tensor radii;         // [..., C, N, 2] int32
    at::Tensor means2d;       // [..., C, N, 2]
    at::Tensor depths;        // [..., C, N]
    at::Tensor conics;        // [..., C, N, 3]
    at::Tensor compensations; // [..., C, N], antialiased only
};

ProjectionDenseResult projection_ewa_3dgs_fused(
    const at::Tensor &means,     // [..., N, 3]
    const at::Tensor &quats,     // [..., N, 4]
    const at::Tensor &scales,    // [..., N, 3]
    const at::Tensor &opacities, // [..., N]
    const at::Tensor &viewmats,  // [..., C, 4, 4]
    const at::Tensor &Ks,        // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool antialiased
);

// Same projection, but only the surviving (camera, gaussian) pairs are kept,
// sorted by image then gaussian id. Costs a host sync to learn nnz.
struct ProjectionPackedResult
{
    at::Tensor batch_ids;     // [nnz] int64
    at::Tensor camera_ids;    // [nnz] int64
    at::Tensor gaussian_ids;  // [nnz] int64
    at::Tensor indptr;        // [B * C + 1] int32, row offsets per image
    at::Tensor radii;         // [nnz, 2] int32
    at::Tensor means2d;       // [nnz, 2]
    at::Tensor depths;        // [nnz]
    at::Tensor conics;        // [nnz, 3]
    at::Tensor compensations; // [nnz], antialiased only
};

ProjectionPackedResult projection_ewa_3dgs_packed(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool antialiased
);

void launch_projection_ewa_3dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,                   // [..., N, 3]
    const at::optional<at::Tensor> covars,    // [..., N, 6] optional
    const at::optional<at::Tensor> quats,     // [..., N, 4] optional
    const at::optional<at::Tensor> scales,    // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,                // [..., C, 4, 4]
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                      // [..., C, N, 2]
    at::Tensor means2d,                    // [..., C, N, 2]
    at::Tensor depths,                     // [..., C, N]
    at::Tensor conics,                     // [..., C, N, 3]
    at::optional<at::Tensor> compensations // [..., C, N] optional
);

// Called twice: first with block_cnts to count survivors per block, then with
// block_accum (their cumsum) and the output tensors to write them out.
void launch_projection_ewa_3dgs_packed_fwd_kernel(
    // inputs
    const at::Tensor means,                   // [..., N, 3]
    const at::optional<at::Tensor> covars,    // [..., N, 6] optional
    const at::optional<at::Tensor> quats,     // [..., N, 4] optional
    const at::optional<at::Tensor> scales,    // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,                // [..., C, 4, 4]
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const at::optional<at::Tensor> block_accum, // [B * C * blocks_per_row]
    const CameraModelType camera_model,
    const bool antialiased, // cull on opacity * compensation (both passes)
    // outputs
    at::optional<at::Tensor> block_cnts,   // [B * C * blocks_per_row]
    at::optional<at::Tensor> indptr,       // [B * C + 1]
    at::optional<at::Tensor> batch_ids,    // [nnz]
    at::optional<at::Tensor> camera_ids,   // [nnz]
    at::optional<at::Tensor> gaussian_ids, // [nnz]
    at::optional<at::Tensor> radii,        // [nnz, 2]
    at::optional<at::Tensor> means2d,      // [nnz, 2]
    at::optional<at::Tensor> depths,       // [nnz]
    at::optional<at::Tensor> conics,       // [nnz, 3]
    at::optional<at::Tensor> compensations // [nnz] optional
);
} // namespace gsplat
