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
inline constexpr int SH_MAX_DEGREE = 4;

// SH -> raw colour (no +0.5 bias). Packed: coeffs are [nnz, K, D] (already
// gathered per entry) and the ids say which camera/gaussian each row is;
// returns [nnz, D]. Dense: coeffs are [N, K, D]; returns [..., C, N, D].
at::Tensor spherical_harmonics(
    int64_t degree,
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids
);

// Dense path only: one kernel writes [max(SH + 0.5, 0) | depth] per (camera, gaussian),
// i.e. [..., C, N, D (+1)].
at::Tensor assemble_proj_features(
    int64_t degree,
    int64_t B,
    int64_t C,
    int64_t N,
    const at::Tensor &means,                // [B, N, 3]
    const at::Tensor &viewmats,             // [B, C, 4, 4]
    const at::Tensor &coeffs,               // [N, K, D]
    const at::optional<at::Tensor> &depths, // [B, C, N]; appended as the last channel when given
    const at::Tensor &masks                 // [B, C, N]
);

void launch_spherical_harmonics_fwd_kernel(
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor coeffs,
    const at::optional<at::Tensor> masks,
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    at::Tensor colors
);

void launch_assemble_proj_features_unpacked_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t degrees_to_use,
    const uint32_t Dc,
    const uint32_t E,
    const uint32_t color_post,
    const uint32_t extra_post,
    const bool has_depth,
    const bool depth_is_zero,
    const bool extra_has_c,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor coeffs,
    const at::optional<at::Tensor> extra,
    const at::optional<at::Tensor> depths,
    const at::optional<at::Tensor> masks,
    at::Tensor out
);
} // namespace gsplat
