#ifndef BMIMAGEUTILS_H
#define BMIMAGEUTILS_H
#include<string>
#include<map>
#include "bmcv_api.h"

//#define FFALIGN(x, n) ((((x)+((n)-1))/(n))*(n))

namespace bm {
std::vector<int> calcImageStride(
        int height, int width,
        bm_image_format_ext format,
        bm_image_data_format_ext dtype,
        int align_bytes = 1);

bm_image readAlignedImage(bm_handle_t handle, const std::string& name);

void centralCropAndResize(bm_handle_t handle,
                          std::vector<bm_image>& srcImages,
                          std::vector<bm_image>& dstImages,
                          float centralFactor = 0.875);

void saveImage(bm_image& bmImage, const std::string& name = "image.jpg");
void dumpImage(bm_image& bmImage, const std::string& name = "image.txt");

std::map<size_t, std::string> loadLabels(const std::string& filename);
}

#endif // BMIMAGEUTILS_H
