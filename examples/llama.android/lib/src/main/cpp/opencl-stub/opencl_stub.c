// OpenCL stub for Android build - actual library loaded at runtime from device
// This stub provides symbols for linking, but the real implementation comes from
// the device's /system/vendor/lib64/libOpenCL.so

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

// Platform APIs
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) { return CL_SUCCESS; }
cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }

// Device APIs
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices) { return CL_SUCCESS; }
cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }

// Context APIs
cl_context clCreateContext(const cl_context_properties *properties, cl_uint num_devices, const cl_device_id *devices, void (*pfn_notify)(const char *, const void *, size_t, void *), void *user_data, cl_int *errcode_ret) { return 0; }
cl_int clRetainContext(cl_context context) { return CL_SUCCESS; }
cl_int clReleaseContext(cl_context context) { return CL_SUCCESS; }
cl_int clGetContextInfo(cl_context context, cl_context_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }

// Command Queue APIs
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int *errcode_ret) { return 0; }
cl_command_queue clCreateCommandQueueWithProperties(cl_context context, cl_device_id device, const cl_queue_properties *properties, cl_int *errcode_ret) { return 0; }
cl_int clRetainCommandQueue(cl_command_queue command_queue) { return CL_SUCCESS; }
cl_int clReleaseCommandQueue(cl_command_queue command_queue) { return CL_SUCCESS; }
cl_int clGetCommandQueueInfo(cl_command_queue command_queue, cl_command_queue_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }

// Memory Object APIs
cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) { return 0; }
cl_mem clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type, const void *buffer_create_info, cl_int *errcode_ret) { return 0; }
cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) { return 0; }
cl_int clRetainMemObject(cl_mem memobj) { return CL_SUCCESS; }
cl_int clReleaseMemObject(cl_mem memobj) { return CL_SUCCESS; }
cl_int clGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }
cl_int clGetImageInfo(cl_mem image, cl_image_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }
cl_int clSetMemObjectDestructorCallback(cl_mem memobj, void (*pfn_notify)(cl_mem, void *), void *user_data) { return CL_SUCCESS; }

// Program Object APIs
cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) { return 0; }
cl_program clCreateProgramWithBinary(cl_context context, cl_uint num_devices, const cl_device_id *device_list, const size_t *lengths, const unsigned char **binaries, cl_int *binary_status, cl_int *errcode_ret) { return 0; }
cl_int clRetainProgram(cl_program program) { return CL_SUCCESS; }
cl_int clReleaseProgram(cl_program program) { return CL_SUCCESS; }
cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void (*pfn_notify)(cl_program, void *), void *user_data) { return CL_SUCCESS; }
cl_int clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }
cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }

// Kernel Object APIs
cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) { return 0; }
cl_int clCreateKernelsInProgram(cl_program program, cl_uint num_kernels, cl_kernel *kernels, cl_uint *num_kernels_ret) { return CL_SUCCESS; }
cl_int clRetainKernel(cl_kernel kernel) { return CL_SUCCESS; }
cl_int clReleaseKernel(cl_kernel kernel) { return CL_SUCCESS; }
cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) { return CL_SUCCESS; }
cl_int clGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }
cl_int clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device, cl_kernel_work_group_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }

// Executing Kernels
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueTask(cl_command_queue command_queue, cl_kernel kernel, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueNativeKernel(cl_command_queue command_queue, void (*user_func)(void *), void *args, size_t cb_args, cl_uint num_mem_objects, const cl_mem *mem_list, const void **args_mem_loc, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }

// Memory Commands
cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t size, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_buffer, size_t src_offset, size_t dst_offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueFillBuffer(cl_command_queue command_queue, cl_mem buffer, const void *pattern, size_t pattern_size, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueReadImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_read, const size_t *origin, const size_t *region, size_t row_pitch, size_t slice_pitch, void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueWriteImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_write, const size_t *origin, const size_t *region, size_t input_row_pitch, size_t input_slice_pitch, const void *ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueCopyImage(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_image, const size_t *src_origin, const size_t *dst_origin, const size_t *region, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueCopyImageToBuffer(cl_command_queue command_queue, cl_mem src_image, cl_mem dst_buffer, const size_t *src_origin, const size_t *region, size_t dst_offset, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueCopyBufferToImage(cl_command_queue command_queue, cl_mem src_buffer, cl_mem dst_image, size_t src_offset, const size_t *dst_origin, const size_t *region, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
void *clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_map, cl_map_flags map_flags, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event, cl_int *errcode_ret) { return 0; }
void *clEnqueueMapImage(cl_command_queue command_queue, cl_mem image, cl_bool blocking_map, cl_map_flags map_flags, const size_t *origin, const size_t *region, size_t *image_row_pitch, size_t *image_slice_pitch, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event, cl_int *errcode_ret) { return 0; }
cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue, cl_mem memobj, void *mapped_ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueMigrateMemObjects(cl_command_queue command_queue, cl_uint num_mem_objects, const cl_mem *mem_objects, cl_mem_migration_flags flags, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }

// Synchronization
cl_int clFinish(cl_command_queue command_queue) { return CL_SUCCESS; }
cl_int clFlush(cl_command_queue command_queue) { return CL_SUCCESS; }
cl_int clEnqueueBarrierWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueMarkerWithWaitList(cl_command_queue command_queue, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }

// Event Object APIs
cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) { return CL_SUCCESS; }
cl_int clGetEventInfo(cl_event event, cl_event_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }
cl_event clCreateUserEvent(cl_context context, cl_int *errcode_ret) { return 0; }
cl_int clRetainEvent(cl_event event) { return CL_SUCCESS; }
cl_int clReleaseEvent(cl_event event) { return CL_SUCCESS; }
cl_int clSetUserEventStatus(cl_event event, cl_int execution_status) { return CL_SUCCESS; }
cl_int clSetEventCallback(cl_event event, cl_int command_exec_callback_type, void (*pfn_notify)(cl_event, cl_int, void *), void *user_data) { return CL_SUCCESS; }

// Profiling APIs
cl_int clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }

// Sampler APIs
cl_sampler clCreateSampler(cl_context context, cl_bool normalized_coords, cl_addressing_mode addressing_mode, cl_filter_mode filter_mode, cl_int *errcode_ret) { return 0; }
cl_sampler clCreateSamplerWithProperties(cl_context context, const cl_sampler_properties *sampler_properties, cl_int *errcode_ret) { return 0; }
cl_int clRetainSampler(cl_sampler sampler) { return CL_SUCCESS; }
cl_int clReleaseSampler(cl_sampler sampler) { return CL_SUCCESS; }
cl_int clGetSamplerInfo(cl_sampler sampler, cl_sampler_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) { return CL_SUCCESS; }

// SVM APIs (OpenCL 2.0+)
void *clSVMAlloc(cl_context context, cl_svm_mem_flags flags, size_t size, cl_uint alignment) { return 0; }
void clSVMFree(cl_context context, void *svm_pointer) {}
cl_int clEnqueueSVMFree(cl_command_queue command_queue, cl_uint num_svm_pointers, void *svm_pointers[], void (*pfn_free_func)(cl_command_queue, cl_uint, void *[], void *), void *user_data, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueSVMMemcpy(cl_command_queue command_queue, cl_bool blocking_copy, void *dst_ptr, const void *src_ptr, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueSVMMemFill(cl_command_queue command_queue, void *svm_ptr, const void *pattern, size_t pattern_size, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueSVMMap(cl_command_queue command_queue, cl_bool blocking_map, cl_map_flags flags, void *svm_ptr, size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clEnqueueSVMUnmap(cl_command_queue command_queue, void *svm_ptr, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event) { return CL_SUCCESS; }
cl_int clSetKernelArgSVMPointer(cl_kernel kernel, cl_uint arg_index, const void *arg_value) { return CL_SUCCESS; }
cl_int clSetKernelExecInfo(cl_kernel kernel, cl_kernel_exec_info param_name, size_t param_value_size, const void *param_value) { return CL_SUCCESS; }
