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

#include "tensorflow/lite/kernels/internal/reference/hard_swish.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/hard_swish.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite_micro {
namespace {
void* HardSwishInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(HardSwishParams));
}

TfLiteStatus HardSwishEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite_micro::micro::GetEvalInput(context, node, kHardSwishInputTensor);
  TfLiteEvalTensor* output =
      tflite_micro::micro::GetEvalOutput(context, node, kHardSwishOutputTensor);
  HardSwishParams* params = static_cast<HardSwishParams*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      tflite_micro::reference_ops::HardSwish<float>(
          tflite_micro::micro::GetTensorShape(input),
          tflite_micro::micro::GetTensorData<float>(input),
          tflite_micro::micro::GetTensorShape(output),
          tflite_micro::micro::GetTensorData<float>(output));
    } break;
    case kTfLiteInt8: {
      tflite_micro::reference_ops::HardSwish<int8_t>(
          *params, tflite_micro::micro::GetTensorShape(input),
          tflite_micro::micro::GetTensorData<int8_t>(input),
          tflite_micro::micro::GetTensorShape(output),
          tflite_micro::micro::GetTensorData<int8_t>(output));
    } break;
    default: {
      MicroPrintf("Unsupported type %s", TfLiteMicroTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_HARD_SWISH() {
  return tflite_micro::micro::RegisterOp(HardSwishInit, tflite_micro::HardSwishPrepare,
                                   HardSwishEval);
}

}  // namespace tflite_micro
