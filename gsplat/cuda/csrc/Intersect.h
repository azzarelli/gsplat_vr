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
#include <algorithm>
#include <cstdint>

#include "MathUtils.h"

namespace gsplat
{
class StageTimer;

// isect_ids pack [image | tile | depth] so a single radix sort orders entries
// by image, then tile, then front-to-back. If the image and tile ids leave at
// least kMinDepthBits of a 32-bit key, keys are compact: 32 bits, with
// log-depth over the frame's measured depth range quantised into the rest.
// Else they are 64-bit with the depth's float bits in the low 32.
constexpr uint32_t kMinDepthBits = 14;

struct KeyLayout
{
    uint32_t image_bits;
    uint32_t tile_bits;
    uint32_t depth_bits;
    bool compact; // uint32 keys (stored as int32), else int64
};

inline KeyLayout key_layout(int64_t n_images, int64_t n_tiles)
{
    const uint32_t image_bits = bits_for_count(n_images);
    const uint32_t tile_bits  = bits_for_count(n_tiles);
    const bool compact        = image_bits + tile_bits + kMinDepthBits <= 32;
    // capped at 24: past that the float log-depth has no more precision to give
    return {image_bits, tile_bits, compact ? std::min(32 - image_bits - tile_bits, 24u) : 32, compact};
}

// One entry per (gaussian, tile) overlap; flatten_ids points each entry back
// at its gaussian row (dense: image * N + gaussian, packed: the packed row).
struct TileIntersectResult
{
    at::Tensor isect_ids;   // [n_isects] int32 (compact) or int64
    at::Tensor flatten_ids; // [n_isects] int32
};

// Packed inputs are [nnz, ...] with image_ids / gaussian_ids and need
// n_images; dense inputs are [I, N, ...]. conics + opacities enable the tight
// (AccuTile) tile test, otherwise the radius AABB is used. Costs a host sync
// to learn n_isects.
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
    const at::optional<at::Tensor> depth_range, // [2] int32, compact keys only
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
    const KeyLayout &layout,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
);
} // namespace gsplat
