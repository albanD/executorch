# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.19)
project(aoti_backend)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(_aoti_sources backends/aoti/runtime/AotiBackend.cpp)
list(TRANSFORM _aoti_sources PREPEND "${EXECUTORCH_ROOT}/")

add_library(executorch_delegate_aoti STATIC ${_aoti_sources})
target_include_directories(
  executorch_delegate_aoti PUBLIC ${_common_include_directories}
)

target_link_options_shared_lib(executorch_delegate_aoti)
