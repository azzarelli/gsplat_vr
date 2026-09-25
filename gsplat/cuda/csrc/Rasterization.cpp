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

#include "Common.h"
#include "Rasterization.h"

namespace gsplat
{
RasterizeResult rasterize_to_pixels_3dgs(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &isect_offsets,
    const at::Tensor &flatten_ids
)
{
    DEVICE_GUARD(means2d);
    TORCH_CHECK(tile_size == 4 || tile_size == 16, "tile_size must be 4 or 16, got ", tile_size);
    TORCH_CHECK(
        isect_offsets.size(-2) * tile_size >= image_height && isect_offsets.size(-1) * tile_size >= image_width,
        "tile grid does not cover the image"
    );
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(isect_offsets);
    CHECK_INPUT(flatten_ids);
    if(backgrounds.has_value())
    {
        CHECK_INPUT(backgrounds.value());
    }

    auto opt = means2d.options();
    at::DimVector image_dims(isect_offsets.sizes().slice(0, isect_offsets.dim() - 2));
    auto with = [&](std::initializer_list<int64_t> tail)
    {
        at::DimVector s(image_dims);
        s.append(tail);
        return s;
    };
    at::Tensor renders = at::empty(with({image_height, image_width, colors.size(-1)}), opt);
    at::Tensor alphas  = at::empty(with({image_height, image_width, 1}), opt);

    launch_rasterize_to_pixels_3dgs_fwd_kernel(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        c10::nullopt, // masks
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        renders,
        alphas
    );
    return {.renders = renders, .alphas = alphas};
}
} // namespace gsplat
