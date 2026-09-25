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
#include "Projection.h"

namespace gsplat
{
namespace
{
    void check_projection_inputs(
        const at::Tensor &means,
        const at::Tensor &quats,
        const at::Tensor &scales,
        const at::Tensor &opacities,
        const at::Tensor &viewmats,
        const at::Tensor &Ks
    )
    {
        CHECK_INPUT(means);
        CHECK_INPUT(quats);
        CHECK_INPUT(scales);
        CHECK_INPUT(opacities);
        CHECK_INPUT(viewmats);
        CHECK_INPUT(Ks);
    }
} // namespace

ProjectionDenseResult projection_ewa_3dgs_fused(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip
)
{
    DEVICE_GUARD(means);
    check_projection_inputs(means, quats, scales, opacities, viewmats, Ks);

    auto opt = means.options();
    at::DimVector shape(means.sizes().slice(0, means.dim() - 2)); // batch dims
    shape.append({viewmats.size(-3), means.size(-2)});            // + [C, N]

    auto with = [&](std::initializer_list<int64_t> tail)
    {
        at::DimVector s(shape);
        s.append(tail);
        return s;
    };
    at::Tensor radii   = at::empty(with({2}), opt.dtype(at::kInt));
    at::Tensor means2d = at::empty(with({2}), opt);
    at::Tensor depths  = at::empty(shape, opt);
    at::Tensor conics  = at::empty(with({3}), opt);

    launch_projection_ewa_3dgs_fused_fwd_kernel(
        means,
        c10::nullopt, // covars
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        CameraModelType::PINHOLE,
        radii,
        means2d,
        depths,
        conics,
        c10::nullopt // compensations (antialiased mode only)
    );
    return {.radii = radii, .means2d = means2d, .depths = depths, .conics = conics};
}

ProjectionPackedResult projection_ewa_3dgs_packed(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip
)
{
    DEVICE_GUARD(means);
    check_projection_inputs(means, quats, scales, opacities, viewmats, Ks);

    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));
    auto opt         = means.options();

    const uint32_t blocks_per_row = (N + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

    // Pass 1: count survivors per block, then a cumsum gives each block its
    // write offset. Reading the total back is a host sync.
    int32_t nnz = 0;
    at::Tensor block_accum;
    if(B && C && N)
    {
        at::Tensor block_cnts = at::empty({B * C * blocks_per_row}, opt.dtype(at::kInt));
        launch_projection_ewa_3dgs_packed_fwd_kernel(
            means,
            c10::nullopt, // covars
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            c10::nullopt, // block_accum
            CameraModelType::PINHOLE,
            block_cnts,
            c10::nullopt, // indptr
            c10::nullopt, // batch_ids
            c10::nullopt, // camera_ids
            c10::nullopt, // gaussian_ids
            c10::nullopt, // radii
            c10::nullopt, // means2d
            c10::nullopt, // depths
            c10::nullopt, // conics
            c10::nullopt  // compensations
        );
        block_accum = at::cumsum(block_cnts, 0, at::kInt);
        nnz         = block_accum[-1].item<int32_t>();
    }

    // Pass 2: recompute and write the survivors.
    ProjectionPackedResult out{
        .batch_ids    = at::empty({nnz}, opt.dtype(at::kLong)),
        .camera_ids   = at::empty({nnz}, opt.dtype(at::kLong)),
        .gaussian_ids = at::empty({nnz}, opt.dtype(at::kLong)),
        .indptr       = at::empty({B * C + 1}, opt.dtype(at::kInt)),
        .radii        = at::empty({nnz, 2}, opt.dtype(at::kInt)),
        .means2d      = at::empty({nnz, 2}, opt),
        .depths       = at::empty({nnz}, opt),
        .conics       = at::empty({nnz, 3}, opt),
    };
    if(nnz)
    {
        launch_projection_ewa_3dgs_packed_fwd_kernel(
            means,
            c10::nullopt,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            image_width,
            image_height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            block_accum,
            CameraModelType::PINHOLE,
            c10::nullopt,
            out.indptr,
            out.batch_ids,
            out.camera_ids,
            out.gaussian_ids,
            out.radii,
            out.means2d,
            out.depths,
            out.conics,
            c10::nullopt
        );
    }
    else
    {
        out.indptr.fill_(0);
    }
    return out;
}
} // namespace gsplat
