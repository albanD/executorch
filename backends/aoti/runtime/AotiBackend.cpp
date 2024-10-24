
#include <executorch/backends/arm/runtime/VelaBinStream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <vector>

namespace executorch {
namespace backends {
namespace aoti {

// Here is where the aoti bouncers are going to be defined.
// I define the globals aoti generated compiled code calls
// They can be backed by ET systems

using namespace std;

using executorch::aten::ScalarType;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::etensor::Tensor;


extern "C" {
  using AOTITensorHandle = Tensor*;

  // TODO: We should get a proper one
  struct CUDAStreamGuardOpaque;
  using CUDAStreamGuardHandle = CUDAStreamGuardOpaque*;

  using AOTIRuntimeError = Error;
  using AOTITorchError = Error;

  struct AOTInductorModelContainerOpaque;
  using AOTInductorModelContainerHandle = AOTInductorModelContainerOpaque*;
  using AOTInductorStreamHandle = void*;
  using AOTIProxyExecutorHandle = void*;

  using AOTInductorModelContainerCreateWithDeviceFunc = AOTIRuntimeError(*)(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir);

  using AOTInductorModelContainerDeleteFunc = AOTIRuntimeError(*)(
    AOTInductorModelContainerHandle container_handle);

  using AOTInductorModelContainerGetNumInputsFunc = AOTIRuntimeError(*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

  using AOTInductorModelContainerGetNumOutputsFunc = AOTIRuntimeError(*)(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants);

using AOTInductorModelContainerRunFunc = AOTIRuntimeError(*)(
    AOTInductorModelContainerHandle container_handle,
    AOTITensorHandle* input_handles, // array of input AOTITensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AOTITensorHandle*
        output_handles, // array for writing output AOTITensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle);

  AOTInductorModelContainerCreateWithDeviceFunc AOTInductorModelContainerCreateWithDevice = nullptr;
  AOTInductorModelContainerDeleteFunc AOTInductorModelContainerDelete = nullptr;
  AOTInductorModelContainerGetNumInputsFunc AOTInductorModelContainerGetNumInputs = nullptr;
  AOTInductorModelContainerGetNumOutputsFunc AOTInductorModelContainerGetNumOutputs = nullptr;
  AOTInductorModelContainerRunFunc AOTInductorModelContainerRun = nullptr;

  int32_t aoti_torch_grad_mode_is_enabled() {
    // No autograd ever
    return false;
  }
  void aoti_torch_grad_mode_set_enabled(bool enabled) {
    if (enabled) {
        std::runtime_error("Cannot enable autograd");
    }
  }
  AOTITorchError aoti_torch_get_data_ptr(
    AOTITensorHandle tensor,
    void** ret_data_ptr) {
    *ret_data_ptr = tensor->mutable_data_ptr();
   return Error::Ok;
  }
  AOTITorchError aoti_torch_get_storage_offset(
    AOTITensorHandle tensor,
    int64_t* ret_storage_offset) {
    // Storage offset is always 0 in ET
    *ret_storage_offset = 0;
    return Error::Ok;
  }
  AOTITorchError aoti_torch_get_strides(
    AOTITensorHandle tensor,
    int64_t** ret_strides) {
    throw std::runtime_error("Cannot get strides. One is int64_t* and the other int32_t*");
  }
  AOTITorchError aoti_torch_get_storage_size(
    AOTITensorHandle tensor,
    int64_t* ret_size) {
    throw std::runtime_error("Cannot get storage size on ETensor");
  }
  AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AOTITensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
    throw std::runtime_error("Not creating Tensor from blob here");
  }
  AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
    std::cout<<"Entering stream guard for device " << device_index << std::endl;
    return Error::Ok;
  }
  AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
    std::cout<<"Exiting stream guard" << std::endl;
    return Error::Ok;
  }
  int aoti_torch_device_type_cpu() {
    // Let's say cpu is 0 for ET as well
    return 0;
  }
  int32_t aoti_torch_device_type_cuda() {
    // Let's say cuda is 1 for ET as well
    return 1;
  }
  int32_t aoti_torch_dtype_float32() {
    // Let assume the dtype here is all we will support
    return 6;
  }
  AOTITorchError aoti_torch_delete_tensor_object(AOTITensorHandle tensor) {
    throw std::runtime_error("Should never allocate?");
    return Error::NotSupported;
  }
  AOTITorchError aoti_torch_create_tensor_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AOTITensorHandle* ret_new_tensor)
  {
    throw std::runtime_error("Should never create from blob");
    return Error::NotSupported;
  }
  AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AOTITensorHandle* ret_new_tensor) {
    throw std::runtime_error("Should never create from blob");
    return Error::NotSupported;
  }
}

struct AOTIDelegateHandle {
  void* so_handle;
  AOTInductorModelContainerHandle container_handle;
};

class AOTIBackend final : public ::executorch::runtime::BackendInterface {
public:
    // Once in program
    AOTIBackend() {
      std::cout<<"Created backend"<<std::endl;
    }

  bool is_available() const override {
    return 1;
  }

  // Once per loaded binary blob
  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed, // This will be the buffer from aoti_backend
      ArrayRef<CompileSpec> compile_specs // This will be my empty list
      ) const override {

    // We could load the .so content directly. But I don't want to deal with
    // relocation. So dumping a file and using dlopen

    // Create a temporary file
    std::ofstream outfile("/tmp/test.so", std::ios::binary);

    // Write the ELF buffer to the temporary file
    outfile.write((char*)processed->data(), sizeof(void*) * processed->size());

    // Finish writing the file to disk
    outfile.close();

    // Free the in-memory buffer
    processed->Free();

    // Load the ELF using dlopen
    void* so_handle = dlopen("/tmp/test.so", RTLD_NOW);
    if (so_handle == nullptr) {
        std::cout<<dlerror()<<std::endl;
        return Error::AccessFailed;
    }

    AOTInductorModelContainerCreateWithDevice = 
      reinterpret_cast<AOTInductorModelContainerCreateWithDeviceFunc>(
        dlsym(so_handle, "AOTInductorModelContainerCreateWithDevice")
      );
    if (AOTInductorModelContainerCreateWithDevice == nullptr) {
      perror("dlsym1");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerDelete = 
      reinterpret_cast<AOTInductorModelContainerDeleteFunc>(
        dlsym(so_handle, "AOTInductorModelContainerDelete")
      );
    if (AOTInductorModelContainerDelete == nullptr) {
      perror("dlsym2");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerGetNumInputs = 
      reinterpret_cast<AOTInductorModelContainerGetNumInputsFunc>(
        dlsym(so_handle, "AOTInductorModelContainerGetNumInputs")
      );
    if (AOTInductorModelContainerGetNumInputs == nullptr) {
      perror("dlsym3");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerGetNumOutputs = 
      reinterpret_cast<AOTInductorModelContainerGetNumOutputsFunc>(
        dlsym(so_handle, "AOTInductorModelContainerGetNumOutputs")
      );
    if (AOTInductorModelContainerGetNumOutputs == nullptr) {
      perror("dlsym4");
      return Error::AccessFailed;
    }
    AOTInductorModelContainerRun = 
      reinterpret_cast<AOTInductorModelContainerRunFunc>(
        dlsym(so_handle, "AOTInductorModelContainerRun")
      );
    if (AOTInductorModelContainerRun == nullptr) {
      perror("dlsym5");
      return Error::AccessFailed;
    }

    AOTInductorModelContainerHandle container_handle = nullptr;

    AOTIRuntimeError err;

    err = AOTInductorModelContainerCreateWithDevice(
      &container_handle,
      1,
      "cuda",
      nullptr);
    printf("container_handle=%p\n", container_handle);

    AOTIDelegateHandle* handle = new AOTIDelegateHandle();
    handle->so_handle = so_handle;
    handle->container_handle = container_handle;
    return (DelegateHandle*)handle; // Return the handle post-processing
  }

  // Once per execution
  Error execute(
    BackendExecutionContext& context,
    DelegateHandle* handle_,
    EValue** args) const override {
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;

    size_t num_inputs;
    AOTInductorModelContainerGetNumInputs(
      handle->container_handle,
      &num_inputs);

    size_t num_outputs;
    AOTInductorModelContainerGetNumOutputs(
      handle->container_handle,
      &num_outputs);

    std::vector<AOTITensorHandle> inputs(num_inputs);
    std::vector<AOTITensorHandle> outputs(num_outputs);

    for (int i = 0; i < num_inputs; i++) {
      auto tensor_in = args[i]->toTensor();
      inputs[i] = &tensor_in;
    }

    for (int i = num_inputs; i < num_inputs + num_outputs; i++) {
      auto tensor_out = args[i]->toTensor();
      outputs[i - num_inputs] = &tensor_out;
    }

    AOTInductorModelContainerRun(
      handle->container_handle,
      inputs.data(), num_inputs,
      outputs.data(), num_outputs,
      // Should these last two be something?
      nullptr, nullptr);

    return Error::Ok;
  }


  void destroy(DelegateHandle* handle_) const override {
    AOTIDelegateHandle* handle = (AOTIDelegateHandle*)handle_;
    dlclose(handle->so_handle);
    AOTInductorModelContainerDelete(handle->container_handle);
    free(handle);
  }

};

}

  namespace {
    auto cls = aoti::AOTIBackend();
    executorch::runtime::Backend backend{"AotiBackend", &cls};
    static executorch::runtime::Error success_with_compiler = register_backend(backend);
  }

}}