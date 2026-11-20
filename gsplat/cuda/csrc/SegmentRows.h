/*
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <ATen/ATen.h>

namespace gsplat
{
// The rows of the kept segments, ascending, then `fill` up to `length`:
// out[0..K) = start_k + 0..n_k-1 for each kept segment k in order, out[K..L)
// = fill, where K = sum of kept counts and length must equal sum(counts).
// For rasterization(indices=...): segments are a pool's slots, fill an
// always-empty row, so the length is known on the host without a sync.
at::Tensor segment_rows(
    const at::Tensor &starts, // [S] int32, first row of each segment
    const at::Tensor &counts, // [S] int64, rows in each segment
    const at::Tensor &keep,   // [S] bool
    int64_t fill,
    int64_t length
);
} // namespace gsplat
