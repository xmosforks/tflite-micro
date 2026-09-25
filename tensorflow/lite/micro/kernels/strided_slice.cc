/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite_micro {

namespace {

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const StridedSliceParams& op_params =
      *(static_cast<const StridedSliceParams*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite_micro::micro::GetEvalInput(context, node, kStridedSliceInputTensor);
  TfLiteEvalTensor* output =
      tflite_micro::micro::GetEvalOutput(context, node, kStridedSliceOutputTensor);
  switch (output->type) {
    case kTfLiteFloat32:
      reference_ops::StridedSlice(op_params,
                                  tflite_micro::micro::GetTensorShape(input),
                                  tflite_micro::micro::GetTensorData<float>(input),
                                  tflite_micro::micro::GetTensorShape(output),
                                  tflite_micro::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::StridedSlice(op_params,
                                  tflite_micro::micro::GetTensorShape(input),
                                  tflite_micro::micro::GetTensorData<int8_t>(input),
                                  tflite_micro::micro::GetTensorShape(output),
                                  tflite_micro::micro::GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
      reference_ops::StridedSlice(
          op_params, tflite_micro::micro::GetTensorShape(input),
          tflite_micro::micro::GetTensorData<int16_t>(input),
          tflite_micro::micro::GetTensorShape(output),
          tflite_micro::micro::GetTensorData<int16_t>(output));
      break;
    case kTfLiteInt32:
      reference_ops::StridedSlice(
          op_params, tflite_micro::micro::GetTensorShape(input),
          tflite_micro::micro::GetTensorData<int32_t>(input),
          tflite_micro::micro::GetTensorShape(output),
          tflite_micro::micro::GetTensorData<int32_t>(output));
      break;
    case kTfLiteBool:
      reference_ops::StridedSlice(op_params,
                                  tflite_micro::micro::GetTensorShape(input),
                                  tflite_micro::micro::GetTensorData<bool>(input),
                                  tflite_micro::micro::GetTensorShape(output),
                                  tflite_micro::micro::GetTensorData<bool>(output));
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteMicroTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_STRIDED_SLICE() {
  return tflite_micro::micro::RegisterOp(StridedSliceInit, StridedSlicePrepare, Eval);
}

}  // namespace tflite_micro
