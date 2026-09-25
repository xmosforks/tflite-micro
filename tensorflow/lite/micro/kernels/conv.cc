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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite_micro {
namespace {

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite_micro::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite_micro::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite_micro::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite_micro::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite_micro::reference_ops::Conv(
          ConvParamsFloat(params, data), tflite_micro::micro::GetTensorShape(input),
          tflite_micro::micro::GetTensorData<float>(input),
          tflite_micro::micro::GetTensorShape(filter),
          tflite_micro::micro::GetTensorData<float>(filter),
          tflite_micro::micro::GetTensorShape(bias),
          tflite_micro::micro::GetOptionalTensorData<float>(bias),
          tflite_micro::micro::GetTensorShape(output),
          tflite_micro::micro::GetTensorData<float>(output),
          tflite_micro::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt16: {
      if (bias == nullptr || bias->type == kTfLiteInt32) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite_micro::micro::GetTensorShape(input),
            tflite_micro::micro::GetTensorData<int16_t>(input),
            tflite_micro::micro::GetTensorShape(filter),
            tflite_micro::micro::GetTensorData<int8_t>(filter),
            tflite_micro::micro::GetTensorShape(bias),
            tflite_micro::micro::GetOptionalTensorData<std::int32_t>(bias),
            tflite_micro::micro::GetTensorShape(output),
            tflite_micro::micro::GetTensorData<int16_t>(output));
      } else if (bias->type == kTfLiteInt64) {
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, data),
            data.per_channel_output_multiplier, data.per_channel_output_shift,
            tflite_micro::micro::GetTensorShape(input),
            tflite_micro::micro::GetTensorData<int16_t>(input),
            tflite_micro::micro::GetTensorShape(filter),
            tflite_micro::micro::GetTensorData<int8_t>(filter),
            tflite_micro::micro::GetTensorShape(bias),
            tflite_micro::micro::GetOptionalTensorData<std::int64_t>(bias),
            tflite_micro::micro::GetTensorShape(output),
            tflite_micro::micro::GetTensorData<int16_t>(output));
      } else {
        MicroPrintf("Bias type %s (%d) not supported.",
                    TfLiteMicroTypeGetName(bias->type), bias->type);
        return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          int8_t* unpacked_filter_data = static_cast<int8_t*>(
              context->GetScratchBuffer(context, data.filter_buffer_index));
          tflite_micro::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite_micro::micro::GetTensorData<int8_t>(filter),
              tflite_micro::micro::GetTensorShape(filter).FlatSize(),
              unpacked_filter_data);
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite_micro::micro::GetTensorShape(input),
              tflite_micro::micro::GetTensorData<int8_t>(input),
              tflite_micro::micro::GetTensorShape(filter), unpacked_filter_data,
              tflite_micro::micro::GetTensorShape(bias),
              tflite_micro::micro::GetOptionalTensorData<int32_t>(bias),
              tflite_micro::micro::GetTensorShape(output),
              tflite_micro::micro::GetTensorData<int8_t>(output));
          break;
        }
        case kTfLiteInt8: {
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, data),
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              tflite_micro::micro::GetTensorShape(input),
              tflite_micro::micro::GetTensorData<int8_t>(input),
              tflite_micro::micro::GetTensorShape(filter),
              tflite_micro::micro::GetTensorData<int8_t>(filter),
              tflite_micro::micro::GetTensorShape(bias),
              tflite_micro::micro::GetOptionalTensorData<int32_t>(bias),
              tflite_micro::micro::GetTensorShape(output),
              tflite_micro::micro::GetTensorData<int8_t>(output));
          break;
        }
        default:
          MicroPrintf("Weight type %s (%d) not supported.",
                      TfLiteMicroTypeGetName(filter->type), filter->type);
          return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteMicroTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite_micro::micro::RegisterOp(ConvInit, ConvPrepare, Eval);
}

}  // namespace tflite_micro
