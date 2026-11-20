/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * segment_rows: the `indices` list for rasterization(), built on the GPU in
 * one pass from contiguous row segments (e.g. the slots of a splat pool).
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "Common.h"
#include "SegmentRows.h"

namespace gsplat
{
namespace
{
    // One block per segment. A kept segment writes its rows start..start+n-1
    // at its place among the kept ones; a dropped one writes `fill` into the
    // tail, at its place among the dropped ones. Each entry of the output is
    // written exactly once.
    __global__ void segment_rows_kernel(
        const int64_t n_seg,
        const int32_t *__restrict__ starts,  // [S]
        const int64_t *__restrict__ counts,  // [S]
        const bool *__restrict__ keep,       // [S]
        const int64_t *__restrict__ kept_cs, // [S] inclusive cumsum of kept counts
        const int64_t *__restrict__ drop_cs, // [S] inclusive cumsum of dropped counts
        const int32_t fill,
        int32_t *__restrict__ out // [L]
    )
    {
        const int64_t k = blockIdx.x;
        if(k >= n_seg)
        {
            return;
        }
        const int64_t n = counts[k];
        if(n == 0)
        {
            return;
        }
        const int64_t n_kept = kept_cs[n_seg - 1];
        if(keep[k])
        {
            const int64_t dst   = kept_cs[k] - n;
            const int32_t start = starts[k];
            for(int64_t i = threadIdx.x; i < n; i += blockDim.x)
            {
                out[dst + i] = start + static_cast<int32_t>(i);
            }
        }
        else
        {
            const int64_t dst = n_kept + drop_cs[k] - n;
            for(int64_t i = threadIdx.x; i < n; i += blockDim.x)
            {
                out[dst + i] = fill;
            }
        }
    }
} // namespace

at::Tensor segment_rows(
    const at::Tensor &starts, const at::Tensor &counts, const at::Tensor &keep, int64_t fill, int64_t length
)
{
    DEVICE_GUARD(starts);
    CHECK_INPUT(starts);
    CHECK_INPUT(counts);
    CHECK_INPUT(keep);
    TORCH_CHECK(starts.scalar_type() == at::kInt, "starts must be int32");
    TORCH_CHECK(counts.scalar_type() == at::kLong, "counts must be int64");
    TORCH_CHECK(keep.scalar_type() == at::kBool, "keep must be bool");
    const int64_t S = starts.size(0);
    TORCH_CHECK(counts.size(0) == S && keep.size(0) == S, "starts, counts and keep must have one entry per segment");
    at::Tensor out = at::empty({length}, starts.options());
    if(S == 0 || length == 0)
    {
        return out;
    }
    const at::Tensor kept    = at::where(keep, counts, at::zeros({}, counts.options()));
    const at::Tensor kept_cs = at::cumsum(kept, 0);
    const at::Tensor drop_cs = at::cumsum(counts - kept, 0);
    segment_rows_kernel<<<static_cast<unsigned int>(S), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        S,
        starts.const_data_ptr<int32_t>(),
        counts.const_data_ptr<int64_t>(),
        keep.const_data_ptr<bool>(),
        kept_cs.const_data_ptr<int64_t>(),
        drop_cs.const_data_ptr<int64_t>(),
        static_cast<int32_t>(fill),
        out.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
} // namespace gsplat
