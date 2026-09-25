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

// Per-(gaussian, pixel) alpha-blending math for 3DGS rasterization.

namespace gsplat
{
// Alpha of one gaussian at pixel offset (dx, dy) = (mean - pixel), clamped to
// MAX_ALPHA. `valid == false`: negative exponent or sub-threshold alpha, skip it.
struct GaussianWeight
{
    float alpha; // min(MAX_ALPHA, opac * exp(-sigma))
    bool valid;  // sigma >= 0 and alpha >= ALPHA_THRESHOLD
};

__device__ __forceinline__ GaussianWeight
    eval_gaussian_weight(const vec3 &conic, const float dx, const float dy, const float opac)
{
    const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
    const float alpha = min(MAX_ALPHA, opac * __expf(-sigma));
    GaussianWeight out;
    out.alpha = alpha;
    out.valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
    return out;
}

} // namespace gsplat
