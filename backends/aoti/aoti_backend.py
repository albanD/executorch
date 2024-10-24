# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import final, List

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec


@final
class AotiBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        print("entering  the lowerable parts in AotiBackend.preprocess....")

        copy_edge_program = copy.deepcopy(edge_program)
        graph_module = copy_edge_program.graph_module
        print(copy_edge_program.example_inputs)
        print(edge_program.example_inputs)
        args, kwargs = copy_edge_program.example_inputs
        so_path = torch._inductor.aot_compile(graph_module, args, kwargs, options={})  # type: ignore[arg-type]
        with open(so_path, "r") as f:
            data = f.read()
        return PreprocessResult(bytes(data, encoding="utf8"))
