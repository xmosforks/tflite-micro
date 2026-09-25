/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/kernels/activations.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite_micro {

template <typename T>
void ReluQuantized(const ReluOpData& data, const RuntimeShape& input_shape,
                   const RuntimeShape& output_shape, const T* input_data,
                   T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const int32_t val = static_cast<int32_t>(input_data[i]);
    int32_t clamped =
        data.params.output_offset +
        MultiplyByQuantizedMultiplier(val - data.params.input_offset,
                                      data.params.output_multiplier,
                                      data.params.output_shift);
    clamped = std::max(data.params.quantized_activation_min, clamped);
    clamped = std::min(data.params.quantized_activation_max, clamped);
    output_data[i] = static_cast<T>(clamped);
  }
}

namespace {

void* ReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(ReluOpData));
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const ReluOpData& data = *(static_cast<const ReluOpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite_micro::micro::GetEvalInput(context, node, kActivationsInputTensor);
  TfLiteEvalTensor* output =
      tflite_micro::micro::GetEvalOutput(context, node, kActivationsOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      ReluFloat(tflite_micro::micro::GetTensorShape(input),
                tflite_micro::micro::GetTensorData<float>(input),
                tflite_micro::micro::GetTensorShape(output),
                tflite_micro::micro::GetTensorData<float>(output));

      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      tflite_micro::ReluQuantized<int16_t>(data, tflite_micro::micro::GetTensorShape(input),
                            tflite_micro::micro::GetTensorShape(output),
                            tflite_micro::micro::GetTensorData<int16_t>(input),
                            tflite_micro::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      tflite_micro::ReluQuantized(data, tflite_micro::micro::GetTensorShape(input),
                            tflite_micro::micro::GetTensorShape(output),
                            tflite_micro::micro::GetTensorData<int8_t>(input),
                            tflite_micro::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Only float32 is supported currently, got %s",
                  TfLiteMicroTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

void* Relu6Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(Relu6OpData));
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const Relu6OpData& data = *(static_cast<const Relu6OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite_micro::micro::GetEvalInput(context, node, kActivationsInputTensor);
  TfLiteEvalTensor* output =
      tflite_micro::micro::GetEvalOutput(context, node, kActivationsOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      Relu6Float(tflite_micro::micro::GetTensorShape(input),
                 tflite_micro::micro::GetTensorData<float>(input),
                 tflite_micro::micro::GetTensorShape(output),
                 tflite_micro::micro::GetTensorData<float>(output));

      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      Relu6Quantized(data.zero_int8, data.six_int8,
                     tflite_micro::micro::GetTensorShape(input),
                     tflite_micro::micro::GetTensorData<int8_t>(input),
                     tflite_micro::micro::GetTensorShape(output),
                     tflite_micro::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      Relu6Quantized(data.zero_int8, data.six_int8,
                     tflite_micro::micro::GetTensorShape(input),
                     tflite_micro::micro::GetTensorData<int16_t>(input),
                     tflite_micro::micro::GetTensorShape(output),
                     tflite_micro::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Only float32 is supported currently, got %s",
                  TfLiteMicroTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_RELU() {
  return tflite_micro::micro::RegisterOp(ReluInit, ReluPrepare, ReluEval);
}

TFLMRegistration Register_RELU6() {
  return tflite_micro::micro::RegisterOp(Relu6Init, Relu6Prepare, Relu6Eval);
}

}  // namespace tflite_micro
