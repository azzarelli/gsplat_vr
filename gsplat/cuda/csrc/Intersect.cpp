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
#include <c10/util/accumulate.h>

#include "Common.h"
#include "Intersect.h"
#include "MathUtils.h"
#include "StageTimer.h"

namespace gsplat
{
TileIntersectResult intersect_tile(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::optional<at::Tensor> &conics,
    const at::optional<at::Tensor> &opacities,
    const at::optional<at::Tensor> &image_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    int64_t n_images,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    StageTimer *timer
)
{
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);

    const bool packed          = means2d.dim() == 2;
    const int64_t I            = n_images;
    const uint32_t n_elements  = means2d.numel() / 2;
    const uint32_t n_tiles     = tile_width * tile_height;
    const uint32_t image_bits  = bits_for_count(I);
    const uint32_t tile_bits   = bits_for_count(n_tiles);
    auto opt                   = depths.options();
    TORCH_CHECK(!packed || (image_ids.has_value() && gaussian_ids.has_value()), "packed inputs need image/gaussian ids");
    TORCH_CHECK(image_bits + tile_bits <= 32, "(image, tile) id needs ", image_bits + tile_bits, " bits, only 32 fit");

    // Pass 1: tiles touched per gaussian; its cumsum is each gaussian's write
    // offset. Reading the total back is a host sync.
    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    at::Tensor cum_tiles_per_gauss;
    int64_t n_isects = 0;
    if(n_elements)
    {
        launch_intersect_tile_kernel(
            means2d,
            radii,
            depths,
            conics,
            opacities,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            tile_size,
            tile_width,
            tile_height,
            c10::nullopt, // cum_tiles_per_gauss
            tiles_per_gauss,
            c10::nullopt, // isect_ids
            c10::nullopt  // flatten_ids
        );
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0, at::kLong);
        n_isects            = cum_tiles_per_gauss[-1].item<int64_t>();
    }

    // Pass 2: write the (key, gaussian) pairs.
    at::Tensor isect_ids   = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if(n_isects == 0)
    {
        if(timer)
        {
            timer->mark();
        }
        return {.isect_ids = isect_ids, .flatten_ids = flatten_ids};
    }
    launch_intersect_tile_kernel(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        packed ? image_ids : c10::nullopt,
        packed ? gaussian_ids : c10::nullopt,
        I,
        tile_size,
        tile_width,
        tile_height,
        cum_tiles_per_gauss,
        c10::nullopt, // tiles_per_gauss
        isect_ids,
        flatten_ids
    );
    if(timer)
    {
        timer->mark();
    }

    // Sort by key: image, tile, then depth.
    at::Tensor isect_ids_sorted   = at::empty_like(isect_ids);
    at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
    radix_sort_double_buffer(
        n_isects, image_bits, tile_bits, isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted
    );
    return {.isect_ids = isect_ids_sorted, .flatten_ids = flatten_ids_sorted};
}

at::Tensor intersect_offset(const at::Tensor &isect_ids, int64_t I, int64_t tile_width, int64_t tile_height)
{
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);
    at::Tensor offsets = at::empty({I, tile_height, tile_width}, isect_ids.options().dtype(at::kLong));
    launch_intersect_offset_kernel(isect_ids, I, tile_width, tile_height, offsets);
    return offsets;
}
} // namespace gsplat
