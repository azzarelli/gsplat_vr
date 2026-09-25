/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <torch/extension.h>

#include "csrc/Rendering.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // The GIL is released for the call so Python threads (e.g. tile loaders)
    // keep running through the host syncs inside.
    m.def(
        "rasterization_3dgs",
        &gsplat::rasterization_3dgs,
        py::call_guard<py::gil_scoped_release>(),
        "Project, colour, sort and rasterize 3D Gaussians for C cameras."
    );
}
