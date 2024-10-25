# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import final, List

import torch
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

from subprocess import check_call


@final
class AotiBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        print("entering  the lowerable parts in AotiBackend.preprocess....")

        print("here", edge_program.example_inputs)
        copy_edge_program = copy.deepcopy(edge_program)
        graph_module = copy_edge_program.graph_module
        # args, kwargs = copy_edge_program.example_inputs
        args, kwargs = (torch.ones(10, device="cuda"), torch.ones(10, device="cuda")), {}
        so_path = torch._inductor.aot_compile(graph_module, args, kwargs, options={})  # type: ignore[arg-type]
        print(so_path)
        check_call(f"patchelf --remove-needed libtorch.so --remove-needed libtorch_cuda.so --remove-needed libc10_cuda.so --remove-needed libtorch_cpu.so --add-needed libcudart.so {so_path}", shell=True)

        with open(so_path, "rb") as f:
            data = f.read()
        return PreprocessResult(data)