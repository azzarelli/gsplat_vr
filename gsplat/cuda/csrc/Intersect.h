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
class StageTimer;

// One entry per (gaussian, tile) overlap. isect_ids packs
// [image id | tile id | depth bits] so a single radix sort orders entries by
// image, then tile, then front-to-back; flatten_ids points each entry back at
// its gaussian row (dense: image * N + gaussian, packed: the packed row).
struct TileIntersectResult
{
    at::Tensor tiles_per_gauss; // [..., N] or [nnz] int32
    at::Tensor isect_ids;       // [n_isects] int64
    at::Tensor flatten_ids;     // [n_isects] int32
};

// Packed inputs are [nnz, ...] with image_ids / gaussian_ids and need
// n_images; dense inputs are [I, N, ...]. conics + opacities enable the tight
// (AccuTile) tile test, otherwise the radius AABB is used. `segmented` sorts
// each image separately (dense only). Costs a host sync to learn n_isects.
// `timer`, if given, is marked between emitting the pairs and sorting them.
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
    bool segmented,
    StageTimer *timer = nullptr
);

// First entry of each (image, tile) in the sorted isect_ids: [I, tile_height, tile_width].
at::Tensor intersect_offset(const at::Tensor &isect_ids, int64_t I, int64_t tile_width, int64_t tile_height);

void launch_intersect_tile_kernel(
    // inputs
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids,     // [n_isects]
    // Only enumerate tiles set in this [I, tile_height, tile_width] bool mask.
    const at::optional<at::Tensor> tile_mask = c10::nullopt
);

void launch_intersect_offset_kernel(
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height,
    at::Tensor offsets // [I, tile_height, tile_width]
);

void radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
);

void segmented_radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t n_segments,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    const at::Tensor offsets,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
);
} // namespace gsplat
