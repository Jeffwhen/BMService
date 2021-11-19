#include <map>
#include <memory>
#include <string.h>
#include "bmruntime_interface.h"
#include "BMDevicePool.h"
#include "BMLog.h"
#include "interface.h"

using namespace bm;

size_t dtype_len(unsigned int t) {
    if(t == BM_FLOAT32 || t == BM_INT32 || t== BM_UINT32){
        return 4;
    } else if(t == BM_UINT16 || t==BM_INT16 || t==BM_FLOAT16){
        return 2;
    } else if(t == BM_UINT8 || t == BM_INT8){
        return 1;
    } else {
        BMLOG(FATAL, "Not support dtype=%d", t);
    }
}

static size_t elem_num(const unsigned int* shape, unsigned int dims){
    size_t elem = 1;
    for(size_t i=0; i<dims; i++){
        elem *= shape[i];
    }
    return elem;
}

struct InputType {
    bool release_inside = false;
    unsigned int id = 0;
    unsigned num = 0;
    tensor_data_t* tensors = nullptr;
};

struct OutputType {
    unsigned int id = 0;
    unsigned num = 0;
    tensor_data_t* tensors = nullptr;
};

bool preProcess(const InputType& input, const TensorVec& inTensors, ContextPtr ctx);
bool postProcess(const InputType& input, const TensorVec& outTensors, OutputType& postOut, ContextPtr ctx);

std::vector<DeviceId> globalDevices;
using GeneralRunner = BMDevicePool<InputType, OutputType>;
struct RunnerInfo {
    RunnerInfo(const char* bmodel, unsigned int batch = 1):
        task_id(INVALID_TASK_ID), runner(bmodel, preProcess, postProcess, globalDevices), status(bmodel), batch(batch) {
        runner.start();
        status.start();
    }
    unsigned int nextId() {
        task_id++;
        if(task_id == INVALID_TASK_ID) task_id++;
        return task_id;
    }

    unsigned int task_id;

    GeneralRunner runner;
    ProcessStatInfo status;
    unsigned int batch;
};

std::map<unsigned int, std::shared_ptr<RunnerInfo>> globalRunnerInfos;

struct NetConfig {
    bool initialized = false;
    bool isTFSSD = false;
    bool isSSD = false;
    size_t netBatch, netHeight, netWidth;

    bm_image_format_ext netFormat;
    bm_image_data_format_ext netDtype;

    void initialize(TensorPtr inTensor, ContextPtr ctx) {
        if (initialized) return; // TODO realloc temp memorys for different input sizes
        initialized = true;
        ctx->setConfigData(this);
        if (inTensor->shape(-1) == 3)
        {
            // NHWC
            netBatch = inTensor->shape(0);
            netHeight = inTensor->shape(1);
            netWidth = inTensor->shape(2);
        } else {
            netBatch = inTensor->shape(0);
            netHeight = inTensor->shape(2);
            netWidth = inTensor->shape(3);
        }

        const bm_net_info_t *netInfo = ctx->net->getNetInfo();
        std::string netName = std::string(netInfo->name);
        std::transform(
            netName.begin(), netName.end(),
            netName.begin(),
            [](unsigned char c) {
                return std::tolower(c);
            });
        if (netName.find("ssd") != std::string::npos)
        {
            if (netWidth == 300)
                isTFSSD = true; // TODO
            isSSD = true;
        }
    }
};

#define BM_CALL(fn, ...) \
    do { \
        if (fn(__VA_ARGS__) != BM_SUCCESS) \
        { \
            BMLOG(FATAL, #fn " failed"); \
        } \
    } while (0);

bool preProcess(const InputType& input, const TensorVec& inTensors, ContextPtr ctx){
    if(input.num == 0){
        return false;
    }
    BM_ASSERT_EQ(input.num, inTensors.size());
    thread_local static NetConfig cfg;
    cfg.initialize(inTensors[0], ctx);
    for(size_t i=0; i<input.num; i++){
        size_t in_mem_size = elem_num(input.tensors[i].shape, input.tensors[i].dims) * dtype_len(input.tensors[i].dtype);
        BM_ASSERT_EQ(inTensors[i]->get_dtype(), input.tensors[i].dtype);
        BM_ASSERT_LE(in_mem_size, inTensors[i]->get_mem_size());
        inTensors[i]->fill_device_mem(input.tensors[i].data, in_mem_size);
        inTensors[i]->set_shape(input.tensors[i].shape, input.tensors[i].dims);
    }
    return true;
}

struct Box {
    int label;
    const float *loc;
    float conf;
    std::vector<float> loc_data;

    Box() : loc(nullptr) {}
    Box(const Box &b) : label(b.label), loc(b.loc), conf(b.conf), loc_data(b.loc_data) {
        if (!loc_data.empty())
        {
            this->loc = loc_data.data();
        }
    }
    friend void swap(Box &a, Box &b) {
        std::swap(a.label, b.label);
        std::swap(a.loc, b.loc);
        std::swap(a.conf, b.conf);
        std::swap(a.loc_data, b.loc_data);
    }
    Box &operator=(Box tmp) {
        swap(*this, tmp);
        return *this;
    }
    Box(int lb, const float *lc, float c) : label(lb), loc(lc), conf(c) {}
};

float area(const float *loc)
{
    return (loc[2] - loc[0]) * (loc[3] - loc[1]);
}

float interception(const float *a, const float *b)
{
    return (std::min(a[2], b[2]) - std::max(a[0], b[0])) *
           (std::min(a[3], b[3]) - std::max(a[1], b[1]));
}

float iou(const float *a, const float *b)
{
    float inter = interception(a, b);
    return inter / (area(a) + area(b) - inter);
}

bool box_sort_fn(const Box &a, const Box &b)
{
    return a.conf > b.conf;
}

std::vector<Box> nms(std::vector<Box> &input, float iou_threshold, size_t max_num = 0)
{
    std::sort(input.begin(), input.end(), box_sort_fn);
    std::vector<Box> output;
    bool reserve;
    for (auto &box : input)
    {
        reserve = true;
        for (auto &reserved : output)
        {
            if (iou(reserved.loc, box.loc) > iou_threshold)
            {
                reserve = false;
                break;
            }
        }
        if (reserve)
            output.push_back(box);
        if (max_num > 0 && output.size() >= max_num)
            break;
    }
    return output;
}

inline float sigmoid(float v)
{
    return 1. / (1 + exp(-v));
}

bool postProcess(const InputType& input, const TensorVec& outTensors, OutputType& postOut, ContextPtr ctx){
    if(input.num == 0) {
        postOut.num = 0;
        throw "finished"; // to stop current pipeline
    }
    auto pCfg = static_cast<NetConfig *>(ctx->getConfigData());
    postOut.id = input.id;
    if(input.release_inside){
        for(size_t i=0; i<input.num; i++){
            delete [] input.tensors[i].data;
        }
        delete []input.tensors;
    }

    if (pCfg->isSSD) {
        if (outTensors.size() != 2)
        {
            BMLOG(FATAL, "unexpected out tensor length %d", outTensors.size());
        }
        int locIndex = outTensors[0]->shape(-1) == 4 ? 0 : 1;
        const auto &locTensor = outTensors[locIndex];
        const auto &clsTensor = outTensors[!locIndex];
        const float *locData = locTensor->get_float_data();
        const float *clsData = clsTensor->get_float_data();
        int n = clsTensor->shape(0);
        int clsNum = clsTensor->shape(-1);
        int boxNum = clsTensor->shape(1);

        const int bg_label = 0;
        if (pCfg->isTFSSD)
        {
            const float conf_threshold = 0.2;
            const float iou_threshold = 0.6;
            const size_t max_num = 100;
            size_t outNum = 4;
            postOut.num = outNum;
            postOut.tensors = new tensor_data_t[outNum];
            tensor_data_t &detNumTensor = postOut.tensors[0];
            tensor_data_t &bboxesTensor = postOut.tensors[1];
            tensor_data_t &scoresTensor = postOut.tensors[2];
            tensor_data_t &labelsTensor = postOut.tensors[3];
            detNumTensor.dtype = labelsTensor.dtype = BM_UINT32;
            bboxesTensor.dtype = scoresTensor.dtype = BM_FLOAT32;
            detNumTensor.dims = 1;
            bboxesTensor.dims = 3;
            labelsTensor.dims = scoresTensor.dims = 2;
            detNumTensor.shape[0] = bboxesTensor.shape[0] = labelsTensor.shape[0] = scoresTensor.shape[0] = n;
            bboxesTensor.shape[1] = labelsTensor.shape[1] = scoresTensor.shape[1] = max_num;
            bboxesTensor.shape[2] = 4;

            auto detNumData = new uint32_t[n];
            auto bboxesData = new float[n * max_num * 4];
            auto labelsData = new uint32_t[n * max_num];
            auto scoresData = new float[n * max_num];
            detNumTensor.data = reinterpret_cast<uint8_t *>(detNumData);
            bboxesTensor.data = reinterpret_cast<uint8_t *>(bboxesData);
            labelsTensor.data = reinterpret_cast<uint8_t *>(labelsData);
            scoresTensor.data = reinterpret_cast<uint8_t *>(scoresData);
            memset(scoresData, 0, n * max_num * sizeof(float));

            const size_t num_layers = 6;
            const size_t layer_sizes[] = {19, 10, 5, 3, 2, 1};
            const float scales_and_ratios[][2] = {
                {0.1, 1.0}, {0.2, 2.0}, {0.2, 0.5}, // (scale, ratio) pairs
                {0.35, 1.0}, {0.35, 2.0}, {0.35, 0.5}, {0.35, 3.0}, {0.35, 0.3333}, {0.4183300132670378, 1.0},
                {0.5, 1.0}, {0.5, 2.0}, {0.5, 0.5}, {0.5, 3.0}, {0.5, 0.3333}, {0.570087712549569, 1.0},
                {0.65, 1.0}, {0.65, 2.0}, {0.65, 0.5}, {0.65, 3.0}, {0.65, 0.3333}, {0.7211102550927979, 1.0},
                {0.8, 1.0}, {0.8, 2.0}, {0.8, 0.5}, {0.8, 3.0}, {0.8, 0.3333}, {0.8717797887081347, 1.0},
                {0.95, 1.0}, {0.95, 2.0}, {0.95, 0.5}, {0.95, 3.0}, {0.95, 0.3333}, {0.9746794344808963, 1.0}};
            const size_t sr_nums[] = {3,   6, 6, 6, 6, 6};
            const float loc_scales[] = {10, 10, 5, 5}; // y, x, h, w
            size_t anchor_index = 0, sr_offset;
            std::map<int, std::vector<Box>> boxes;
            std::vector<Box> allBoxes;
            for (int ib = 0; ib < n; ++ib)
            {
                boxes.clear();
                allBoxes.clear();
                anchor_index = 0;
                sr_offset = 0;
                for (int il = 0; il < num_layers; ++il)
                {
                    for (int iy = 0; iy < layer_sizes[il]; ++iy)
                    {
                        for (int ix = 0; ix < layer_sizes[il]; ++ix)
                        {
                            for (int isr = 0; isr < sr_nums[il]; ++isr)
                            {
                                const float *sr = scales_and_ratios[sr_offset + isr];
                                const float *loc = locData + (boxNum * ib + anchor_index) * 4;

                                Box box;
                                for (int label = 1; label < clsNum; ++label)
                                {
                                    float conf = clsData[(boxNum * ib + anchor_index) * clsNum + label];
                                    if (conf < conf_threshold)
                                        continue;
                                    box.label = label;
                                    box.conf = conf;
                                    if (box.loc == nullptr)
                                    {
                                        float aw = sr[0] * sqrt(sr[1]);
                                        float ah = sr[0] / sqrt(sr[1]);
                                        //std::cout << (iy + 0.5) / layer_sizes[il] << ","
                                        //          << (ix + 0.5) / layer_sizes[il] << ","
                                        //          << aw << "," << ah << std::endl;
                                        float h = exp(loc[2] / loc_scales[2]) * ah;
                                        float w = exp(loc[3] / loc_scales[3]) * aw;
                                        float ycen = loc[0] / loc_scales[0] * ah + (iy + 0.5) / layer_sizes[il];
                                        float xcen = loc[1] / loc_scales[1] * aw + (ix + 0.5) / layer_sizes[il];
                                        box.loc_data.resize(4);
                                        box.loc = box.loc_data.data();
                                        box.loc_data[0] = ycen - h / 2;
                                        box.loc_data[1] = xcen - w / 2;
                                        box.loc_data[2] = box.loc[0] + h;
                                        box.loc_data[3] = box.loc[1] + w;
                                    }
                                    boxes[label].push_back(box);
                                }
                                ++anchor_index;
                            }
                        }
                    }
                    sr_offset += sr_nums[il];
                }
                for (auto &p : boxes)
                {
                    auto &boxes = p.second;
                    auto result = nms(boxes, iou_threshold, max_num);
                    allBoxes.insert(allBoxes.end(), result.begin(), result.end());
                }
                if (allBoxes.size() <= max_num)
                    std::sort(allBoxes.begin(), allBoxes.end(), box_sort_fn);
                else
                    std::partial_sort(allBoxes.begin(), allBoxes.begin() + max_num, allBoxes.end(), box_sort_fn);
                int detNum = std::min<int>(allBoxes.size(), max_num);
                detNumData[ib] = detNum;
                for (int id = 0; id < detNum; id++)
                {
                    auto &box = allBoxes[id];
                    memcpy(bboxesData + (max_num * ib + id) * 4, box.loc, sizeof(float) * 4);
                    labelsData[max_num * ib + id] = box.label;
                    scoresData[max_num * ib + id] = box.conf;
                }
            }
        } else {
            const float conf_threshold = 0.05;
            const float iou_threshold = 0.5;
            const size_t max_num = 200;

            // Setup output tensors
            size_t outNum = 3;
            postOut.num = outNum;
            postOut.tensors = new tensor_data_t[outNum];
            tensor_data_t &bboxesTensor = postOut.tensors[0];
            tensor_data_t &labelsTensor = postOut.tensors[1];
            tensor_data_t &scoresTensor = postOut.tensors[2];
            labelsTensor.dtype = BM_UINT32;
            bboxesTensor.dtype = scoresTensor.dtype = BM_FLOAT32;
            bboxesTensor.dims = 3;
            labelsTensor.dims = scoresTensor.dims = 2;
            bboxesTensor.shape[0] = labelsTensor.shape[0] = scoresTensor.shape[0] = n;
            bboxesTensor.shape[1] = labelsTensor.shape[1] = scoresTensor.shape[1] = max_num;
            bboxesTensor.shape[2] = 4;
            auto bboxesData = new float[n * max_num * 4];
            auto labelsData = new uint32_t[n * max_num];
            auto scoresData = new float[n * max_num];
            bboxesTensor.data = reinterpret_cast<uint8_t *>(bboxesData);
            labelsTensor.data = reinterpret_cast<uint8_t *>(labelsData);
            scoresTensor.data = reinterpret_cast<uint8_t *>(scoresData);
            memset(scoresData, 0, n * max_num * sizeof(float));

            std::map<int, std::vector<Box>> boxes;
            std::vector<Box> allBoxes;
            for (int ib = 0; ib < n; ib++)
            {
                boxes.clear();
                allBoxes.clear();
                for (int i = 0; i < boxNum; i++)
                {
                    int offset = ib * boxNum + i;
                    const float *loc = locData + offset * 4;
                    for (int label = 0; label < clsNum; ++label)
                    {
                        float cls = clsData[offset * clsNum + label];
                        if (label == bg_label) continue;
                        if (cls < conf_threshold) continue;
                        boxes[label].push_back({label, loc, cls});
                    }
                }
                for (auto &p : boxes)
                {
                    auto &boxes = p.second;
                    auto result = nms(boxes, iou_threshold);
                    allBoxes.insert(allBoxes.end(), result.begin(), result.end());
                }
                if (allBoxes.size() <= max_num)
                    std::sort(allBoxes.begin(), allBoxes.end(), box_sort_fn);
                else
                    std::partial_sort(allBoxes.begin(), allBoxes.begin() + max_num, allBoxes.end(), box_sort_fn);
                for (int id = 0; id < std::min<int>(allBoxes.size(), max_num); id++)
                {
                    auto &box = allBoxes[id];
                    memcpy(bboxesData + (max_num * ib + id) * 4, box.loc, sizeof(float) * 4);
                    labelsData[max_num * ib + id] = box.label;
                    scoresData[max_num * ib + id] = box.conf;
                }
            }
        }

        return true;
    }

    size_t outNum = outTensors.size();
    postOut.num = outNum;
    postOut.tensors = new tensor_data_t[outNum];
    for(size_t i=0; i<outNum; i++){
        postOut.tensors[i].dims = outTensors[i]->dims();
        for(size_t d=0; d<postOut.tensors[i].dims; d++){
            postOut.tensors[i].shape[d] = outTensors[i]->shape(d);
        }
        postOut.tensors[i].dtype = outTensors[i]->get_dtype();
        auto mem_size = outTensors[i]->get_mem_size();
        postOut.tensors[i].data = new unsigned char[mem_size];
        auto fill_size = outTensors[i]->fill_host_mem(postOut.tensors[i].data, mem_size);
        BM_ASSERT_EQ(fill_size, mem_size);
    }
    return true;
}

unsigned int runner_start_with_batch(const char *bmodel, unsigned int batch) {
    set_env_log_level();
    unsigned int runner_id = 0;
    while(globalRunnerInfos.count(runner_id)) runner_id++;
    globalRunnerInfos[runner_id] = std::make_shared<RunnerInfo>(bmodel, batch);
    return runner_id;
}

unsigned int runner_start(const char *bmodel) {
    runner_start_with_batch(bmodel, 1);
}

void runner_stop(unsigned int runner_id) {
    if(!globalRunnerInfos.count(runner_id)) return;
    globalRunnerInfos[runner_id]->runner.stop();
    globalRunnerInfos.erase(runner_id);
}

void runner_show_status(unsigned int runner_id)
{
    if(!globalRunnerInfos.count(runner_id)) return;
    globalRunnerInfos[runner_id]->status.show();
}

unsigned int runner_put_input(unsigned runner_id, unsigned int input_num, const tensor_data_t *input_tensors, int need_copy)
{
    if(!globalRunnerInfos.count(runner_id)) return -1;
    InputType input;
    input.id = globalRunnerInfos[runner_id]->nextId();
    input.release_inside = need_copy;
    input.num = input_num;
    if(input_num != 0){
        if(need_copy){
            input.tensors = new tensor_data_t[input_num];
            memcpy(input.tensors, input_tensors, sizeof(tensor_data_t)*input_num);
            for(size_t i = 0; i<input.num; i++){
                auto mem_size = dtype_len(input_tensors[i].dtype) * elem_num(input_tensors[i].shape, input_tensors[i].dims);
                input.tensors[i].data = new unsigned char[mem_size];
                memcpy(input.tensors[i].data, input_tensors[i].data, mem_size);
            }
        } else {
            input.tensors = (tensor_data_t*)input_tensors;
        }
    } else {
        input.tensors = nullptr;
    }
    globalRunnerInfos[runner_id]->runner.push(input);
    return input.id;
}


int runner_all_stopped(size_t runner_id){
    if(!globalRunnerInfos.count(runner_id)) return true;
    return globalRunnerInfos[runner_id]->runner.allStopped();
}

static tensor_data_t *__runner_get_output(unsigned runner_id, unsigned int *task_id, unsigned int *output_num, unsigned int *is_valid, bool is_async){
    if(!globalRunnerInfos.count(runner_id)) return nullptr;
    auto& info = globalRunnerInfos[runner_id];
    OutputType output;
    std::shared_ptr<ProcessStatus> status;
    bool ok = false;
    do{
       ok = info->runner.pop(output, status);
       if(ok || is_async){
           break;
       } else {
           std::this_thread::yield();
       }
    } while(1);
    if(!ok) return nullptr;

    *task_id = output.id;
    *output_num = output.num;
    *is_valid = status->valid;
    info->status.update(status, info->batch);
    return output.tensors;
}

tensor_data_t *runner_try_to_get_output(unsigned runner_id, unsigned int *task_id, unsigned int *output_num, unsigned int *is_valid)
{
    return __runner_get_output(runner_id, task_id, output_num, is_valid, true);
}


tensor_data_t *runner_get_output(unsigned runner_id, unsigned int *task_id, unsigned int *output_num, unsigned int *is_valid)
{
    return __runner_get_output(runner_id, task_id, output_num, is_valid, false);
}

unsigned int runner_release_output(unsigned int output_num, const tensor_data_t *output_data){
    for(size_t i=0; i<output_num; i++){
        delete [] output_data[i].data;
    }
    delete []output_data;
}

int runner_empty(unsigned int runner_id)
{
    if(!globalRunnerInfos.count(runner_id)) return true;
    return globalRunnerInfos[runner_id]->runner.empty();
}

void runner_use_devices(const unsigned *device_ids, unsigned num)
{
    globalDevices.assign(device_ids, device_ids+num);
}

unsigned int available_devices(unsigned int *devices, unsigned int maxNum)
{
    auto deviceIds = getAvailableDevices();
    unsigned int realNum = maxNum>deviceIds.size()?deviceIds.size(): maxNum;
    for(size_t i =0; i<realNum; i++){
        devices[i] = deviceIds[i];
    }
    return realNum;
}

blob_info_t *get_input_info(unsigned runner_id, unsigned *num)
{
    if(!globalRunnerInfos.count(runner_id)) {
        BMLOG(ERROR, "invalid runner_id %d", runner_id);
        return nullptr;
    }
    auto& info = globalRunnerInfos[runner_id];
    const bm_net_info_t *net_info = info->runner.getNetInfo();
    *num = net_info->input_num;
    auto blobs = new blob_info_t[*num];
    for (int i = 0; i < *num; ++i)
    {
        auto &blob = blobs[i];
        blob.name = net_info->input_names[i];
        bm_shape_t &s = net_info->stages[0].input_shapes[i];
        blob.num_dims = s.num_dims;
        memcpy(blob.dims, s.dims, s.num_dims * sizeof(int));
    }
    return blobs;
}

void release_input_info(unsigned runner_id, blob_info_t *info)
{
    delete[] info;
}

