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

#include "tensorflow/lite/kernels/internal/reference/prelu.h"

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/prelu.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite_micro {

void* PreluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(PreluParams));
}

TfLiteStatus PreluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const PreluParams& params =
      *(static_cast<const PreluParams*>(node->user_data));

  const TfLiteEvalTensor* input = tflite_micro::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* alpha = tflite_micro::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite_micro::micro::GetEvalOutput(context, node, 0);

  switch (input->type) {
    case kTfLiteFloat32: {
      BroadcastPrelu4DSlowFloat(tflite_micro::micro::GetTensorShape(input),
                                tflite_micro::micro::GetTensorData<float>(input),
                                tflite_micro::micro::GetTensorShape(alpha),
                                tflite_micro::micro::GetTensorData<float>(alpha),
                                tflite_micro::micro::GetTensorShape(output),
                                tflite_micro::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      reference_ops::BroadcastPrelu4DSlow(
          params, tflite_micro::micro::GetTensorShape(input),
          tflite_micro::micro::GetTensorData<int8_t>(input),
          tflite_micro::micro::GetTensorShape(alpha),
          tflite_micro::micro::GetTensorData<int8_t>(alpha),
          tflite_micro::micro::GetTensorShape(output),
          tflite_micro::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    } break;
    default:
      MicroPrintf("Only float32 and uint8_t are supported currently, got %d.",
                  TfLiteMicroTypeGetName(input->type));
      return kTfLiteError;
  }
}

TFLMRegistration Register_PRELU() {
  return tflite_micro::micro::RegisterOp(PreluInit, PreluPrepare, PreluEval);
}

}  // namespace tflite_micro
