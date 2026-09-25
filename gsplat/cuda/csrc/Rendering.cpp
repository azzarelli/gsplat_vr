/*
 * SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// The whole forward pass, stage by stage:
//
//   1. project      3D gaussians -> 2D means/conics/depths per camera
//                   (packed: survivors only; dense: every (camera, gaussian))
//   2. colour       SH -> RGB per visible entry, +0.5 and clamp; optional
//                   depth appended as the last feature channel
//   3. intersect    one (image|tile|depth) key per gaussian-tile overlap,
//                   32-bit when it fits (see key_layout), radix sorted,
//                   then per-tile start offsets
//   4. rasterize    front-to-back alpha compositing per tile; ED divides the
//                   depth by alpha as each pixel is written. Writes tensors,
//                   or straight into CUDA arrays (e.g. mapped GL textures)
//
// Stages 1 and 3 each read a count back to the host (nnz, n_isects).
// With `profile`, CUDA events split this into project | sh | isect | sort |
// blend (the order _STAGES in gsplat/rendering.py expects).

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

#include "Common.h"
#include "Intersect.h"
#include "Projection.h"
#include "Rasterization.h"
#include "Rendering.h"
#include "SphericalHarmonics.h"
#include "StageTimer.h"

namespace gsplat
{
namespace
{
    void check_inputs(
        const at::Tensor &means,
        const at::Tensor &quats,
        const at::Tensor &scales,
        const at::Tensor &opacities,
        const at::optional<at::Tensor> &colors,
        const at::Tensor &viewmats,
        const at::Tensor &Ks,
        const at::optional<at::Tensor> &backgrounds,
        int64_t tile_size,
        int64_t sh_degree,
        bool append_depth
    )
    {
        TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be [N, 3], got ", means.sizes());
        const int64_t N = means.size(0);
        TORCH_CHECK(quats.sizes() == at::IntArrayRef({N, 4}), "quats must be [N, 4], got ", quats.sizes());
        TORCH_CHECK(scales.sizes() == at::IntArrayRef({N, 3}), "scales must be [N, 3], got ", scales.sizes());
        TORCH_CHECK(opacities.sizes() == at::IntArrayRef({N}), "opacities must be [N], got ", opacities.sizes());
        TORCH_CHECK(
            viewmats.dim() == 3 && viewmats.size(1) == 4 && viewmats.size(2) == 4,
            "viewmats must be [C, 4, 4], got ",
            viewmats.sizes()
        );
        const int64_t C = viewmats.size(0);
        TORCH_CHECK(Ks.sizes() == at::IntArrayRef({C, 3, 3}), "Ks must be [C, 3, 3], got ", Ks.sizes());
        TORCH_CHECK(tile_size == 4 || tile_size == 16, "tile_size must be 4 or 16, got ", tile_size);
        TORCH_CHECK(colors.has_value() || append_depth, "nothing to render: no colors and no depth");
        if(colors.has_value())
        {
            const at::Tensor &c = colors.value();
            if(sh_degree >= 0)
            {
                TORCH_CHECK(c.dim() == 3 && c.size(0) == N, "SH colors must be [N, K, D], got ", c.sizes());
                TORCH_CHECK(
                    (sh_degree + 1) * (sh_degree + 1) <= c.size(1),
                    "sh_degree ",
                    sh_degree,
                    " needs more than ",
                    c.size(1),
                    " coeffs"
                );
            }
            else
            {
                const bool per_gaussian = c.dim() == 2 && c.size(0) == N;
                const bool per_view     = c.dim() == 3 && c.size(0) == C && c.size(1) == N;
                TORCH_CHECK(per_gaussian || per_view, "colors must be [N, D] or [C, N, D], got ", c.sizes());
            }
        }
        else
        {
            TORCH_CHECK(sh_degree < 0, "sh_degree must be None when colors is None");
        }
        if(backgrounds.has_value())
        {
            TORCH_CHECK(
                backgrounds.value().dim() == 2 && backgrounds.value().size(0) == C,
                "backgrounds must be [C, D], got ",
                backgrounds.value().sizes()
            );
        }
    }

    // SH per packed entry, coeffs read in place: [max(SH + 0.5, 0) | depth].
    at::Tensor evaluate_sh_packed(
        int64_t degree,
        const at::Tensor &coeffs,
        const at::Tensor &means,
        const at::Tensor &viewmats,
        const at::Tensor &valid,
        const at::Tensor &batch_ids,
        const at::Tensor &camera_ids,
        const at::Tensor &gaussian_ids,
        const at::optional<at::Tensor> &depths
    )
    {
        at::Tensor values
            = spherical_harmonics(degree, means, viewmats, coeffs, valid, batch_ids, camera_ids, gaussian_ids);
        return packed_features(values, c10::nullopt, depths);
    }

    // Stereo: evaluate SH once per unique Gaussian from a single viewpoint
    // midway between the eyes, then scatter to both eyes' packed entries.
    // A Gaussian visible to both eyes otherwise has its SH evaluated twice,
    // once per (camera, gaussian) entry in the packed list.
    //
    // Both eyes get identical view-dependent colour. That removes the
    // inter-ocular specular difference, which is a real (if usually small)
    // stereo depth cue, so this is a quality trade and not a free win.
    // Returns [max(SH + 0.5, 0) | depth] per packed entry.
    at::Tensor evaluate_feature_sh_stereo_shared(
        int64_t degree,
        const at::Tensor &coeffs,
        const at::Tensor &means,
        const at::Tensor &viewmats,    // [C, 4, 4], C == 2
        const at::Tensor &union_ids,   // [U] gaussians kept by either eye, from projection
        const at::Tensor &union_slots, // [nnz] each entry's index in union_ids
        const at::optional<at::Tensor> &depths
    )
    {
        // SH only reads the camera centre (c = -R^T t), so take eye 0's matrix
        // and replace its translation with the midpoint centre.
        at::Tensor vm      = viewmats.reshape({-1, 4, 4});
        at::Tensor rot     = vm.slice(1, 0, 3).slice(2, 0, 3);
        at::Tensor trans   = vm.slice(1, 0, 3).select(2, 3);
        at::Tensor centres = -at::matmul(rot.transpose(1, 2), trans.unsqueeze(-1)).squeeze(-1);
        at::Tensor c_mid   = centres.mean(0);

        at::Tensor cyc = vm.slice(0, 0, 1).clone();
        cyc.slice(1, 0, 3)
            .select(2, 3)
            .copy_(-at::matmul(rot.select(0, 0), c_mid.unsqueeze(-1)).squeeze(-1).unsqueeze(0));

        at::Tensor zeros_u = at::zeros_like(union_ids);
        at::Tensor values  = spherical_harmonics(
            degree,
            means,
            cyc,
            coeffs,
            c10::nullopt,
            zeros_u, // batch_ids  (B == 1 on this path)
            zeros_u, // camera_ids (the single cyclopean camera)
            union_ids
        );
        return packed_features(values, union_slots, depths);
    }

    // Already-activated colours ([N, D] or [C, N, D]) in the layout the
    // rasterizer reads: [nnz, D] packed, [C, N, D] dense.
    at::Tensor colors_to_features(
        const at::Tensor &colors,
        int64_t C,
        int64_t N,
        bool packed,
        const at::Tensor &camera_ids,
        const at::Tensor &gaussian_ids
    )
    {
        const bool per_view = colors.dim() == 3;
        if(packed)
        {
            return per_view ? colors.index({camera_ids, gaussian_ids}) : colors.index({gaussian_ids});
        }
        return per_view ? colors : colors.unsqueeze(0).expand({C, N, colors.size(-1)});
    }
} // namespace

RasterizationOutputs rasterization_3dgs(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::optional<at::Tensor> &colors,
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    const at::optional<at::Tensor> &backgrounds,
    int64_t sh_degree,
    bool append_depth,
    bool expected_depth,
    bool packed,
    bool antialiased,
    bool stereo,
    const std::vector<int64_t> &color_targets,
    const std::vector<int64_t> &alpha_targets,
    bool profile
)
{
    DEVICE_GUARD(means);
    check_inputs(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        backgrounds,
        tile_size,
        sh_degree,
        append_depth
    );
    const int64_t N = means.size(0);
    const int64_t C = viewmats.size(0);
    StageTimer timer(profile);

    // --- 1. Project ----------------------------------------------------------
    at::Tensor batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, proj_opacities;
    at::Tensor union_ids, union_slots; // shared-eye colour only
    const bool shared_sh = packed && stereo && C == 2 && colors.has_value() && sh_degree >= 0;
    if(packed)
    {
        ProjectionPackedResult p = projection_ewa_3dgs_packed(
            means, quats, scales, opacities, viewmats, Ks, image_width, image_height, eps2d, near_plane, far_plane, radius_clip,
            antialiased,
            shared_sh
        );
        batch_ids      = p.batch_ids; // all 0: one scene, no batch dims
        camera_ids     = p.camera_ids;
        gaussian_ids   = p.gaussian_ids;
        radii          = p.radii;
        means2d        = p.means2d;
        depths         = p.depths;
        conics         = p.conics;
        proj_opacities = p.opacities;
        union_ids      = p.union_ids;
        union_slots    = p.union_slots;
    }
    else
    {
        ProjectionDenseResult p = projection_ewa_3dgs_fused(
            means, quats, scales, opacities, viewmats, Ks, image_width, image_height, eps2d, near_plane, far_plane, radius_clip,
            antialiased
        );
        radii          = p.radii;
        means2d        = p.means2d;
        depths         = p.depths;
        conics         = p.conics;
        proj_opacities = antialiased ? opacities.unsqueeze(0) * p.compensations : opacities.unsqueeze(0).expand({C, N});
    }
    const at::Tensor valid = radii.gt(0).all(-1); // culled entries have zero radius
    timer.mark();

    // --- 2. Colour (+ depth channel) -----------------------------------------
    at::Tensor features;
    bool depth_in_features = false;
    if(colors.has_value() && sh_degree >= 0)
    {
        if(!packed)
        {
            // One kernel writes [SH colour | depth] rows directly.
            features = assemble_proj_features(
                sh_degree,
                1,
                C,
                N,
                means,
                viewmats,
                colors.value(),
                append_depth ? at::optional<at::Tensor>(depths) : c10::nullopt,
                valid
            );
            depth_in_features = append_depth;
        }
        else
        {
            const at::optional<at::Tensor> depth_channel = append_depth ? at::optional<at::Tensor>(depths) : c10::nullopt;
            features = shared_sh
                         ? evaluate_feature_sh_stereo_shared(
                               sh_degree, colors.value(), means, viewmats, union_ids, union_slots, depth_channel
                           )
                         : evaluate_sh_packed(
                               sh_degree, colors.value(), means, viewmats, valid, batch_ids, camera_ids, gaussian_ids, depth_channel
                           );
            depth_in_features = append_depth;
        }
    }
    else if(colors.has_value())
    {
        features = colors_to_features(colors.value(), C, N, packed, camera_ids, gaussian_ids);
    }

    at::optional<at::Tensor> render_backgrounds = backgrounds;
    if(append_depth)
    {
        if(!depth_in_features)
        {
            features = features.defined() ? at::cat({features, depths.unsqueeze(-1)}, -1) : depths.unsqueeze(-1);
        }
        // Background depth is 0.
        if(backgrounds.has_value())
        {
            const at::Tensor &bg = backgrounds.value();
            render_backgrounds   = colors.has_value() ? at::cat({bg, at::zeros_like(bg.narrow(-1, 0, 1))}, -1)
                                                      : at::zeros({C, 1}, bg.options());
        }
    }
    timer.mark();

    // --- 3. Tile intersection + sort -----------------------------------------
    const int64_t tile_width  = static_cast<int64_t>(std::ceil(image_width / static_cast<double>(tile_size)));
    const int64_t tile_height = static_cast<int64_t>(std::ceil(image_height / static_cast<double>(tile_size)));
    means2d                   = means2d.contiguous();
    conics                    = conics.contiguous();
    proj_opacities            = proj_opacities.contiguous();

    TileIntersectResult isects = intersect_tile(
        means2d,
        radii,
        depths,
        conics,
        proj_opacities,
        packed ? at::optional<at::Tensor>(camera_ids) : c10::nullopt, // image id == camera id (B == 1)
        packed ? at::optional<at::Tensor>(gaussian_ids) : c10::nullopt,
        C,
        tile_size,
        tile_width,
        tile_height,
        &timer
    );
    at::Tensor isect_offsets = intersect_offset(isects.isect_ids, C, tile_width, tile_height);
    timer.mark();

    // --- 4. Rasterize --------------------------------------------------------
    RasterizeResult raster = rasterize_to_pixels_3dgs(
        means2d,
        conics,
        features.contiguous(),
        proj_opacities,
        render_backgrounds.has_value() ? at::optional<at::Tensor>(render_backgrounds.value().contiguous()) : c10::nullopt,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        isects.flatten_ids,
        expected_depth,
        color_targets,
        alpha_targets
    );
    timer.mark();

    return {raster.renders, raster.alphas, depths.numel(), isects.flatten_ids.size(0), timer.elapsed_ms()};
}
} // namespace gsplat
