/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include "Common.h"

// Per-(gaussian, pixel) alpha-blending math for 3DGS rasterization, shared by
// the dense and sparse rasterizers so the numerically sensitive forward and
// backward steps live in one place. These describe the contribution of a single
// gaussian to a single pixel; tile iteration, pixel addressing, shared-memory
// batching and warp reductions stay in the kernels.

namespace gsplat
{
// Per-(gaussian, pixel) gaussian-weight evaluation shared by every 3DGS
// conic-based kernel. Computes the Mahalanobis exponent from the conic and the
// pixel offset (dx, dy) = (mean - pixel), the resulting alpha (clamped to
// MAX_ALPHA), and whether the sample contributes. `valid == false` means the
// caller skips this gaussian (negative exponent or sub-threshold alpha). `vis`
// (== exp(-sigma)) is retained for the backward pass, which needs it directly.
struct GaussianWeight
{
    float vis;   // __expf(-sigma)
    float alpha; // min(MAX_ALPHA, opac * vis)
    bool valid;  // sigma >= 0 and alpha >= ALPHA_THRESHOLD
};

__device__ __forceinline__ GaussianWeight
    eval_gaussian_weight(const vec3 &conic, const float dx, const float dy, const float opac)
{
    const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
    const float vis   = __expf(-sigma);
    const float alpha = min(MAX_ALPHA, opac * vis);
    GaussianWeight out;
    out.vis   = vis;
    out.alpha = alpha;
    out.valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
    return out;
}

} // namespace gsplat
