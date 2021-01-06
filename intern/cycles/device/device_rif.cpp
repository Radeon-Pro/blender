/*
 * Copyright 2020, Blender Foundation.
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

#ifdef WITH_RIF

#  include <boost/filesystem.hpp>

#  include "device/device.h"
#  include "device/device_intern.h"
#  include "device/opencl/device_opencl.h"

#  include "render/buffers.h"
#  include "render/hair.h"
#  include "render/mesh.h"
#  include "render/object.h"
#  include "render/scene.h"

#  include "util/util_debug.h"
#  include "util/util_logging.h"
#  include "util/util_md5.h"
#  include "util/util_path.h"
#  include "util/util_time.h"

#  include <delayimp.h>
#  include <RadeonImageFilters.h>
#  include <RadeonImageFilters_cl.h>

#  define RIF_DLL_LOCATION "C:/Program Files/AMD/CNext/CNext/RadeonImageFilters.dll"
#  define RIF_MODELS_LOCATION "C:/Program Files/AMD/CNext/CNext/models/"

CCL_NAMESPACE_BEGIN

#  define check_result_rif(stmt) \
    { \
      rif_int res = stmt; \
      if (res != RIF_SUCCESS) { \
        const char *code = rifGetErrorCodeString(res); \
        const char *msg = rifGetLastErrorMessage(); \
        set_error( \
            string_printf("%s (%s) in %s (device_rif.cpp:%d)", msg, code, #stmt, __LINE__)); \
        return; \
      } \
    } \
    (void)0
#  define check_result_rif_ret(stmt) \
    { \
      rif_int res = stmt; \
      if (res != RIF_SUCCESS) { \
        const char *code = rifGetErrorCodeString(res); \
        const char *msg = rifGetLastErrorMessage(); \
        set_error( \
            string_printf("%s (%s) in %s (device_rif.cpp:%d)", msg, code, #stmt, __LINE__)); \
        return false; \
      } \
    } \
    (void)0

bool shouldUpgradeDriver()
{
  HKEY hKey = HKEY_LOCAL_MACHINE;
  LPCSTR subKey = "SOFTWARE\\AMD\\CN";
  LPCSTR valueName = "CNVersion";
  DWORD dataSize{};
  LONG lRes = RegGetValue(hKey, subKey, valueName, RRF_RT_REG_SZ, nullptr, nullptr, &dataSize);
  if (lRes != ERROR_SUCCESS) {
    VLOG(1) << "Cannot read string from registry " << HRESULT_FROM_WIN32(lRes);
    return false;
  }
  std::string data;
  data.resize(dataSize);
  lRes = RegGetValue(
      hKey, subKey, valueName, RRF_RT_REG_SZ, nullptr, &data[0], &dataSize);
  if (lRes != ERROR_SUCCESS) {
    VLOG(1) << "Cannot read string from registry " << HRESULT_FROM_WIN32(lRes);
    return false;
  }
  DWORD strLenInChars = dataSize;
  strLenInChars--;
  data.resize(strLenInChars);
  VLOG(1) << "AMD Driver Version: " << data.c_str();
  int major = 0, minor = 0;
  if (sscanf_s(data.c_str(), "%d.%d", &major, &minor) == 2) {
    if (major < 20)
      return true;
    else if (major == 20 && minor < 45)
      return true;
  }
  return false;
}

HMODULE LoadRIFDll(LPCSTR moduleName)
{
  return LoadLibraryA(RIF_DLL_LOCATION);
}

FARPROC WINAPI RIFDliNotifyHook(unsigned dliNotify, PDelayLoadInfo pdli)
{
  if (dliNotify == dliNotePreLoadLibrary)
    return (FARPROC)LoadRIFDll(pdli->szDll);
  else if (dliNotify == dliFailLoadLib) {
    VLOG(1) << "RIF initialization failed. Module could not be loaded";
    MessageBoxA(
        NULL, "RIF initialization failed. Module could not be loaded.", "Load RIF failed", 0);
  }
  else if (dliNotify == dliFailGetProc) {
    VLOG(1) << "RIF initialization failed. Could not find procedure in module";
    MessageBoxA(NULL,
                "RIF initialization failed. Could not find procedure in module.",
                "Load RIF failed",
                0);
  }
  return NULL;
}

extern "C" PfnDliHook __pfnDliNotifyHook2 = RIFDliNotifyHook;
extern "C" PfnDliHook __pfnDliFailureHook2 = RIFDliNotifyHook;

class RIFDevice : public OpenCLDevice {

  template<typename T> class rif_object {
   private:
    T object = nullptr;
    rif_object(const rif_object &) = delete;             // non construction-copyable
    rif_object &operator=(const rif_object &) = delete;  // non copyable
   public:
    rif_object() = default;

    ~rif_object()
    {
      if (object)
        rifObjectDelete(object);
    }

    operator T()
    {
      return object;
    }

    T *operator&()
    {
      return &object;
    }
  };

  rif_object<rif_context> context;
  rif_object<rif_command_queue> queue;
  rif_object<rif_image_filter> denoise_filter;
  rif_object<rif_image_filter> remap_normal_filter;
  rif_object<rif_image_filter> remap_depth_filter;

  OpenCLProgram denoising_program;

 public:
  RIFDevice(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background)
      : OpenCLDevice(info, stats, profiler, background)
  {
    if (!cxContext) {
      return;  // Do not initialize if OpenCL context creation failed already
    }

    // Create RIF context
    const string cache_path = path_cache_get(path_join("kernels", "rif"));
    check_result_rif(rifCreateContextFromOpenClContext(
        RIF_API_VERSION, cxContext, cdDevice, cqCommandQueue, cache_path.c_str(), &context));

    // Create RIF command queue
    check_result_rif(rifContextCreateCommandQueue(context, &queue));

    // Create RIF AI denoise filter
    check_result_rif(
        rifContextCreateImageFilter(context, RIF_IMAGE_FILTER_AI_DENOISE, &denoise_filter));
    check_result_rif(rifImageFilterSetParameter1u(denoise_filter, "useHDR", RIF_TRUE));
    check_result_rif(
        rifImageFilterSetParameterString(denoise_filter, "modelPath", RIF_MODELS_LOCATION));

    // Create RIF remap filter for normals
    check_result_rif(
        rifContextCreateImageFilter(context, RIF_IMAGE_FILTER_REMAP_RANGE, &remap_normal_filter));
    check_result_rif(rifImageFilterSetParameter1f(remap_normal_filter, "dstLo", 0.0f));
    check_result_rif(rifImageFilterSetParameter1f(remap_normal_filter, "dstHi", 1.0f));

    // Create RIF remap depth filter
    check_result_rif(
        rifContextCreateImageFilter(context, RIF_IMAGE_FILTER_REMAP_RANGE, &remap_depth_filter));
    check_result_rif(rifImageFilterSetParameter1f(remap_depth_filter, "dstLo", 0.0f));
    check_result_rif(rifImageFilterSetParameter1f(remap_depth_filter, "dstHi", 1.0f));

    denoising_program = OpenCLDevice::OpenCLProgram(
        this,
        "denoising_program",
        "filter.cl",
        get_build_options(DeviceRequestedFeatures(), "denoising_program", false));

    denoising_program.add_kernel(ustring("filter_write_color"));
    denoising_program.add_kernel(ustring("filter_split_aov"));
    denoising_program.add_kernel(ustring("filter_copy_input"));

    if (!denoising_program.load()) {
      denoising_program.compile();
    }
  }

  void thread_run(DeviceTask &task) override
  {
    if (task.type == DeviceTask::RENDER) {
      RenderTile tile;
      DenoisingTask denoising(this, task);

      /* Keep rendering tiles until done. */
      while (task.acquire_tile(this, tile, task.tile_types)) {
        if (tile.task == RenderTile::PATH_TRACE) {
          assert(tile.task == RenderTile::PATH_TRACE);
          scoped_timer timer(&tile.buffers->render_time);

          /* Complete kernel execution before release tile. */
          /* This helps in multi-device render;
           * The device that reaches the critical-section function
           * release_tile waits (stalling other devices from entering
           * release_tile) for all kernels to complete. If device1 (a
           * slow-render device) reaches release_tile first then it would
           * stall device2 (a fast-render device) from proceeding to render
           * next tile.
           */
          clFinish(cqCommandQueue);
        }
        else if (tile.task == RenderTile::BAKE) {
          bake(task, tile);
        }
        else if (tile.task == RenderTile::DENOISE) {
          launch_denoise(task, tile, denoising);
        }

        task.release_tile(tile);
      }
    }
    else if (task.type == DeviceTask::SHADER) {
      shader(task);
    }
    else if (task.type == DeviceTask::FILM_CONVERT) {
      film_convert(task, task.buffer, task.rgba_byte, task.rgba_half);
    }
    else if (task.type == DeviceTask::DENOISE_BUFFER) {
      RenderTile tile;
      tile.x = task.x;
      tile.y = task.y;
      tile.w = task.w;
      tile.h = task.h;
      tile.buffer = task.buffer;
      tile.sample = task.sample + task.num_samples;
      tile.num_samples = task.num_samples;
      tile.start_sample = task.sample;
      tile.offset = task.offset;
      tile.stride = task.stride;
      tile.buffers = task.buffers;

      DenoisingTask denoising(this, task);
      launch_denoise(task, tile, denoising);
      task.update_progress(&tile, tile.w * tile.h);
    }
  }

 private:
  void merge_tiles(DeviceTask &task,
                   const float scale,
                   RenderTileNeighbors &neighbors,
                   device_only_memory<float> &merged,
                   const int4 &rect,
                   const int2 &rect_size,
                   const size_t pass_stride)
  {
    device_vector<TileInfo> tile_info_mem(this, "denoiser tile info", MEM_READ_WRITE);

    TileInfo *tile_info = tile_info_mem.alloc(1);
    for (int i = 0; i < RenderTileNeighbors::SIZE; i++) {
      tile_info->offsets[i] = neighbors.tiles[i].offset;
      tile_info->strides[i] = neighbors.tiles[i].stride;
      tile_info->buffers[i] = neighbors.tiles[i].buffer;
    }
    tile_info->x[0] = neighbors.tiles[3].x;
    tile_info->x[1] = neighbors.tiles[4].x;
    tile_info->x[2] = neighbors.tiles[5].x;
    tile_info->x[3] = neighbors.tiles[5].x + neighbors.tiles[5].w;
    tile_info->y[0] = neighbors.tiles[1].y;
    tile_info->y[1] = neighbors.tiles[4].y;
    tile_info->y[2] = neighbors.tiles[7].y;
    tile_info->y[3] = neighbors.tiles[7].y + neighbors.tiles[7].h;
    tile_info_mem.copy_to_device();

    cl_kernel filter_copy_input = denoising_program(ustring("filter_copy_input"));

    int arg_ofs = 0;
    arg_ofs += kernel_set_args(
        filter_copy_input, arg_ofs, merged.device_pointer, tile_info_mem.device_pointer);
    for (int i = 0; i < RenderTileNeighbors::SIZE; i++) {
      arg_ofs += kernel_set_args(filter_copy_input, arg_ofs, tile_info->buffers[i]);
    }
    arg_ofs += kernel_set_args(filter_copy_input, arg_ofs, (int4&)rect, task.pass_stride);

    enqueue_kernel(filter_copy_input, rect_size.x, rect_size.y);
  }

  void split_aov(DeviceTask &task,
                 device_only_memory<float> &buffer,
                 const int offset,
                 const int stride,
                 const int x,
                 const int y,
                 const int w,
                 const int h,
                 const float scale,
                 const int color_only,
                 device_only_memory<float> &color,
                 device_only_memory<float> &albedo,
                 device_only_memory<float> &normals,
                 device_only_memory<float> &depth)
  {
    /* Set images with appropriate stride for our interleaved pass storage. */
    const int pixel_offset = offset + x + y * stride;
    const int pixel_stride = task.pass_stride;
    const int row_stride = stride * pixel_stride;

    struct {
      device_only_memory<float> &vector;
      const int num_elements;
      const int offset;
      const bool use;
      array<float> scaled_buffer;
    } passes[] = {
        {color,
         3,
         pixel_offset * task.pass_stride + task.pass_denoising_data + DENOISING_PASS_COLOR,
         true},
        {albedo,
         3,
         pixel_offset * task.pass_stride + task.pass_denoising_data + DENOISING_PASS_ALBEDO,
         !color_only},
        {normals,
         3,
         pixel_offset * task.pass_stride + task.pass_denoising_data + DENOISING_PASS_NORMAL,
         !color_only},
        {depth,
         1,
         pixel_offset * task.pass_stride + task.pass_denoising_data + DENOISING_PASS_DEPTH,
         !color_only},
    };

    for (int i = 0; i < sizeof(passes) / sizeof(passes[0]); i++) {
      if (!passes[i].use) {
        continue;
      }

      passes[i].vector.alloc_to_device(size_t(w) * h * passes[i].num_elements);
    }
    cl_kernel filter_split_aov = denoising_program(ustring("filter_split_aov"));

    int arg_ofs = 0;
    arg_ofs += kernel_set_args(
        filter_split_aov, arg_ofs, buffer.device_pointer, pixel_stride, row_stride);
    
    for (int i = 0; i < sizeof(passes) / sizeof(passes[0]); i++) {
      arg_ofs += kernel_set_args(
          filter_split_aov, arg_ofs, passes[i].vector.device_pointer, passes[i].offset);
    }

    arg_ofs += kernel_set_args(
        filter_split_aov, arg_ofs, w, h, scale, color_only);

    enqueue_kernel(filter_split_aov, w, h);
  }

  bool copy_back(RenderTile &target,
                 const int4 &rect,
                 const int2 &rect_size,
                 const int overlap,
                 const float invscale,
                 device_only_memory<float> &output,
                 const int pass_stride)
  {
    /* Copy back result from merged buffer. */
    const int xmin = max(target.x, rect.x);
    const int ymin = max(target.y, rect.y);
    const int xmax = min(target.x + target.w, rect.z);
    const int ymax = min(target.y + target.h, rect.w);
    const int overlap_x = min(overlap, target.x);
    const int overlap_y = min(overlap, target.y);

    cl_kernel filter_write_color = denoising_program(ustring("filter_write_color"));

    int arg_ofs = 0;
    arg_ofs += kernel_set_args(filter_write_color,
                               arg_ofs,
                               output.device_pointer,
                               pass_stride,
                               target.offset,
                               target.stride);

    arg_ofs += kernel_set_args(
        filter_write_color, arg_ofs, target.buffer, rect_size.x, overlap_x, overlap_y);
    arg_ofs += kernel_set_args(filter_write_color, arg_ofs, xmin, xmax, ymin, ymax, invscale);

    enqueue_kernel(filter_write_color, size_t(xmax) - xmin, size_t(ymax) - ymin);

    return true;
  }

  bool rif_denoise(const int2 &rect_size,
                   const int color_only,
                   device_only_memory<float> &color,
                   device_only_memory<float> &albedo,
                   device_only_memory<float> &normals,
                   device_only_memory<float> &depth,
                   device_only_memory<float> &output)
  {
    rif_image_desc desc = {0};

    desc.image_width = rect_size.x;
    desc.image_height = rect_size.y;
    desc.image_depth = 1;
    desc.num_components = 3;
    desc.type = RIF_COMPONENT_TYPE_FLOAT32;

    output.alloc_to_device(size_t(desc.image_width) * desc.num_components * desc.image_height *
                           desc.image_depth);

    rif_object<rif_image> output_img;
    check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
        context, &desc, CL_MEM_PTR(output.device_pointer), &output_img));

    rif_object<rif_image> color_img;
    check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
        context, &desc, CL_MEM_PTR(color.device_pointer), &color_img));

    rif_object<rif_image> albedo_img;
    if (!color_only) {
      check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
          context, &desc, CL_MEM_PTR(albedo.device_pointer), &albedo_img));
    }

    rif_object<rif_image> normals_img;
    if (!color_only) {
      check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
          context, &desc, CL_MEM_PTR(normals.device_pointer), &normals_img));
      check_result_rif_ret(
          rifCommandQueueAttachImageFilter(queue, remap_normal_filter, normals_img, normals_img));
    }

    desc.num_components = 1;
    rif_object<rif_image> depth_img;
    if (!color_only) {
      check_result_rif_ret(rifContextCreateImageFromOpenClMemory(
          context, &desc, CL_MEM_PTR(depth.device_pointer), &depth_img));
      check_result_rif_ret(
          rifCommandQueueAttachImageFilter(queue, remap_depth_filter, depth_img, depth_img));
    }

    check_result_rif_ret(rifImageFilterSetParameterImage(denoise_filter, "colorImg", color_img));
    if (color_only) {
      check_result_rif_ret(rifImageFilterClearParameterImage(denoise_filter, "albedoImg"));
      check_result_rif_ret(rifImageFilterClearParameterImage(denoise_filter, "normalsImg"));
      check_result_rif_ret(rifImageFilterClearParameterImage(denoise_filter, "depthImg"));
    }
    else {
      check_result_rif_ret(
          rifImageFilterSetParameterImage(denoise_filter, "albedoImg", albedo_img));
      check_result_rif_ret(
          rifImageFilterSetParameterImage(denoise_filter, "normalsImg", normals_img));
      check_result_rif_ret(rifImageFilterSetParameterImage(denoise_filter, "depthImg", depth_img));
    }

    check_result_rif_ret(
        rifCommandQueueAttachImageFilter(queue, denoise_filter, color_img, output_img));
    check_result_rif_ret(rifContextExecuteCommandQueue(context, queue, nullptr, nullptr, nullptr));
    check_result_rif_ret(rifCommandQueueDetachImageFilter(queue, denoise_filter));

    if (!color_only) {
      check_result_rif_ret(rifCommandQueueDetachImageFilter(queue, remap_normal_filter));
      check_result_rif_ret(rifCommandQueueDetachImageFilter(queue, remap_depth_filter));
    }

    return true;
  }

  bool launch_denoise(DeviceTask &task, RenderTile &rtile, DenoisingTask &denoising)
  {
    // Update current sample (for display and NLM denoising task)
    rtile.sample = rtile.start_sample + rtile.num_samples;

    // Choose between RIF and NLM denoising
    if (task.denoising.type == DENOISER_RIF) {
      /* Per-tile denoising. */
      const float scale = 1.0f / rtile.sample;
      const float invscale = rtile.sample;
      const int pass_stride = task.pass_stride;

      /* Map neighboring tiles into one buffer for denoising. */
      RenderTileNeighbors neighbors(rtile);
      task.map_neighbor_tiles(neighbors, this);
      RenderTile &center_tile = neighbors.tiles[RenderTileNeighbors::CENTER];
      rtile = center_tile;

      /* Calculate size of the tile to denoise (including overlap). The overlap
       * size was chosen empirically. */
      const int overlap = 32;
      const int4 rect = rect_clip(rect_expand(center_tile.bounds(), overlap), neighbors.bounds());
      const int2 rect_size = make_int2(rect.z - rect.x, rect.w - rect.y);

      // Adjacent tiles are in separate memory regions, so need to copy them into a single one
      device_only_memory<float> merged(this, "merged tiles buffer");
      merged.alloc_to_device(size_t(rect_size.x) * rect_size.y * pass_stride);
      merge_tiles(task, scale, neighbors, merged, rect, rect_size, pass_stride);

      const int color_only = task.denoising.input_passes < DENOISER_INPUT_RGB_ALBEDO_NORMAL ||
                             DebugFlags().rif.color_only;

      /* Once we have all the tiles in one memory buffer, we split it into separate OAVs: color,
       * albedo, normals and depth. */
      device_only_memory<float> color(this, "color buffer");
      device_only_memory<float> albedo(this, "albedo buffer");
      device_only_memory<float> normals(this, "normals buffer");
      device_only_memory<float> depth(this, "depth buffer");
      split_aov(task,
                merged,
                0,
                rect_size.x,
                0,
                0,
                rect_size.x,
                rect_size.y,
                scale,
                color_only,
                color,
                albedo,
                normals,
                depth);

      /* Execute the denoiser. */
      device_only_memory<float> output(this, "output buffer");
      rif_denoise(rect_size, color_only, color, albedo, normals, depth, output);

      /* Copy denoised image back to a target tile. */
      copy_back(neighbors.target, rect, rect_size, overlap, invscale, output, pass_stride);

      task.unmap_neighbor_tiles(neighbors, this);
    }
    else {
      // Run OpenCL denoising kernels
      DenoisingTask denoising(this, task);
      OpenCLDevice::denoise(rtile, denoising);
    }

    // Update task progress after the denoiser completed processing
    task.update_progress(&rtile, rtile.w * rtile.h);

    return true;
  }
};

bool device_rif_init()
{
  if (shouldUpgradeDriver()) {
    MessageBoxA(
        NULL, "AMD Radeon Driver Version 20.45 or higher required to use Radeon Image Filters(RIF) denoising. Please upgrade graphics driver!", "Driver Update Needed", 0);
    return false;
  }
  // Need to initialize OpenCL as well
  if (!device_opencl_init())
    return false;

  rif_int device_count = 0;
  rif_int result = rifGetDeviceCount(RIF_BACKEND_API_OPENCL, &device_count);

  if (result != RIF_SUCCESS) {
    VLOG(1) << "RIF initialization failed with error code " << (unsigned int)result << ": "
            << rifGetLastErrorMessage();
    return false;
  }
  if (device_count == 0) {
    VLOG(1) << "RIF initialization failed. No device found.";
    return false;
  }

  // Loaded RIF successfully!
  return true;
}

void device_rif_info(const vector<DeviceInfo> &opencl_devices, vector<DeviceInfo> &devices)
{
  devices.reserve(opencl_devices.size());

  // Simply add all supported OpenCL devices as RIF devices again
  for (DeviceInfo info : opencl_devices) {
    assert(info.type == DEVICE_OPENCL);

    info.type = DEVICE_RIF;
    info.id += "_RIF";
    info.denoisers |= DENOISER_RIF;

    devices.push_back(info);
  }
}

Device *device_rif_create(DeviceInfo &info, Stats &stats, Profiler &profiler, bool background)
{
  return new RIFDevice(info, stats, profiler, background);
}

CCL_NAMESPACE_END

#endif
