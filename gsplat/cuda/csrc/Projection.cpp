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
    double radius_clip,
    bool antialiased
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
    at::Tensor compensations;
    if(antialiased)
    {
        compensations = at::empty(shape, opt);
    }

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
        antialiased ? at::optional<at::Tensor>(compensations) : c10::nullopt
    );
    return {.radii = radii, .means2d = means2d, .depths = depths, .conics = conics, .compensations = compensations};
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
    double radius_clip,
    bool antialiased,
    bool with_union
)
{
    DEVICE_GUARD(means);
    check_projection_inputs(means, quats, scales, opacities, viewmats, Ks);
    TORCH_CHECK(means.dim() == 2 && viewmats.dim() == 3, "packed projection takes means [N, 3] and viewmats [C, 4, 4]");
    TORCH_CHECK(means.scalar_type() == at::kFloat, "packed projection is float32 only");

    const int64_t N         = means.size(0);
    const int64_t C         = viewmats.size(0);
    const int64_t n_blocks  = (N + N_THREADS_PACKED - 1) / N_THREADS_PACKED;
    const int64_t cam_cells = C * n_blocks;
    auto opt                = means.options();
    auto launch             = [&](at::optional<at::Tensor> cnts, at::optional<at::Tensor> accum, const ProjectionPackedResult *out)
    {
        launch_projection_ewa_3dgs_packed_kernel(
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
            antialiased,
            with_union,
            cnts,
            accum,
            out
        );
    };

    // Pass 1: survivors per (camera, block), plus a union row. Camera rows are
    // cumsummed as one array (camera-major offsets), the union row on its own.
    // Reading both totals back is one host sync.
    int64_t nnz = 0, n_union = 0;
    at::Tensor block_accum;
    if(N)
    {
        at::Tensor block_cnts = at::empty({cam_cells + (with_union ? n_blocks : 0)}, opt.dtype(at::kInt));
        launch(block_cnts, c10::nullopt, nullptr);
        block_accum          = at::empty_like(block_cnts);
        at::Tensor cam_accum = block_accum.narrow(0, 0, cam_cells);
        at::cumsum_out(cam_accum, block_cnts.narrow(0, 0, cam_cells), 0, at::kInt);
        at::Tensor totals = cam_accum.narrow(0, cam_cells - 1, 1);
        if(with_union)
        {
            at::Tensor union_accum = block_accum.narrow(0, cam_cells, n_blocks);
            at::cumsum_out(union_accum, block_cnts.narrow(0, cam_cells, n_blocks), 0, at::kInt);
            totals = at::cat({totals, union_accum.narrow(0, n_blocks - 1, 1)});
        }
        const at::Tensor host = totals.cpu();
        nnz                   = host[0].item<int32_t>();
        n_union               = with_union ? host[1].item<int32_t>() : 0;
    }

    // Pass 2: recompute and write the survivors.
    ProjectionPackedResult out{
        .batch_ids    = at::empty({nnz}, opt.dtype(at::kLong)),
        .camera_ids   = at::empty({nnz}, opt.dtype(at::kLong)),
        .gaussian_ids = at::empty({nnz}, opt.dtype(at::kLong)),
        .radii        = at::empty({nnz, 2}, opt.dtype(at::kInt)),
        .means2d      = at::empty({nnz, 2}, opt),
        .depths       = at::empty({nnz}, opt),
        .conics       = at::empty({nnz, 3}, opt),
        .opacities    = at::empty({nnz}, opt),
        .union_ids    = with_union ? at::empty({n_union}, opt.dtype(at::kLong)) : at::Tensor(),
        .union_slots  = with_union ? at::empty({nnz}, opt.dtype(at::kLong)) : at::Tensor(),
    };
    if(nnz)
    {
        launch(c10::nullopt, block_accum, &out);
    }
    return out;
}
} // namespace gsplat
