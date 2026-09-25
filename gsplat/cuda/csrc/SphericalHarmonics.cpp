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

#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "Common.h"
#include "SphericalHarmonics.h"

namespace gsplat
{
namespace
{
    void check_degree(int64_t degree, const at::Tensor &coeffs)
    {
        TORCH_CHECK(degree >= 0 && degree <= SH_MAX_DEGREE, "sh degree must be in [0, ", SH_MAX_DEGREE, "], got ", degree);
        TORCH_CHECK(coeffs.dim() == 3, "SH coeffs must be [N, K, D], got ", coeffs.sizes());
        TORCH_CHECK(
            (degree + 1) * (degree + 1) <= coeffs.size(-2),
            "sh degree ",
            degree,
            " needs more coefficients than ",
            coeffs.sizes()
        );
    }
} // namespace

at::Tensor spherical_harmonics(
    int64_t degree,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &masks,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids
)
{
    DEVICE_GUARD(means);
    check_degree(degree, coeffs);
    CHECK_INPUT(means);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(coeffs);

    std::vector<int64_t> shape;
    if(batch_ids.has_value())
    {
        shape = {batch_ids.value().size(0), coeffs.size(-1)};
    }
    else
    {
        shape = viewmats.sizes().vec(); // [..., C, 4, 4] -> [..., C, N, D]
        shape.pop_back();
        shape.pop_back();
        shape.push_back(means.size(-2));
        shape.push_back(coeffs.size(-1));
    }
    at::Tensor colors = at::empty(shape, means.options());
    launch_spherical_harmonics_fwd_kernel(
        degree, means, viewmats, coeffs, masks, batch_ids, camera_ids, gaussian_ids, colors
    );
    return colors;
}

at::Tensor packed_features(
    const at::Tensor &colors,
    const at::optional<at::Tensor> &rows,
    const at::optional<at::Tensor> &depths
)
{
    DEVICE_GUARD(colors);
    CHECK_INPUT(colors);
    TORCH_CHECK(colors.dim() == 2 && colors.scalar_type() == at::kFloat, "colors must be float [R, D]");
    const int64_t n = rows.has_value() ? rows.value().size(0) : colors.size(0);
    if(depths.has_value())
    {
        TORCH_CHECK(depths.value().numel() == n, "depths must have one value per entry");
    }
    at::Tensor out = at::empty({n, colors.size(-1) + (depths.has_value() ? 1 : 0)}, colors.options());
    launch_pack_features_kernel(
        colors,
        rows.has_value() ? at::optional<at::Tensor>(rows.value().contiguous()) : c10::nullopt,
        depths.has_value() ? at::optional<at::Tensor>(depths.value().contiguous()) : c10::nullopt,
        out
    );
    return out;
}

at::Tensor assemble_proj_features(
    int64_t degree,
    int64_t B,
    int64_t C,
    int64_t N,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &depths,
    const at::Tensor &masks
)
{
    DEVICE_GUARD(means);
    check_degree(degree, coeffs);
    TORCH_CHECK(coeffs.size(0) == N, "SH coeffs N (", coeffs.size(0), ") != N (", N, ")");

    const int64_t Dc     = coeffs.size(-1);
    const bool has_depth = depths.has_value();
    std::vector<int64_t> shape(means.sizes().begin(), means.sizes().end() - 2);
    shape.insert(shape.end(), {C, N, Dc + (has_depth ? 1 : 0)});
    at::Tensor out = at::empty(shape, means.options());

    launch_assemble_proj_features_unpacked_fwd_kernel(
        B,
        C,
        N,
        degree,
        Dc,
        /*E=*/0,
        /*color_post=*/2, // SH_POST_SHIFT_RELU: max(x + 0.5, 0)
        /*extra_post=*/0,
        has_depth,
        /*depth_is_zero=*/false,
        /*extra_has_c=*/false,
        means.contiguous(),
        viewmats.contiguous(),
        coeffs.contiguous(),
        /*extra=*/c10::nullopt,
        has_depth ? at::optional<at::Tensor>(depths.value().contiguous()) : c10::nullopt,
        masks.contiguous(),
        out
    );
    return out;
}
} // namespace gsplat
