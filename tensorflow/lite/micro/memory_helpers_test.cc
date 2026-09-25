/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/memory_helpers.h"

#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

// This just needs to be big enough to handle the array of 5 ints allocated
// in TestAllocateOutputDimensionsFromInput below.
const int kGlobalPersistentBufferLength = 100;
char global_persistent_buffer[kGlobalPersistentBufferLength];

// Only need to handle a single allocation at a time for output dimensions
// in TestAllocateOutputDimensionsFromInput.
void* FakeAllocatePersistentBuffer(TfLiteContext* context, size_t bytes) {
  return reinterpret_cast<void*>(global_persistent_buffer);
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestAlignPointerUp) {
  uint8_t* input0 = reinterpret_cast<uint8_t*>(0);

  uint8_t* input0_aligned1 = tflite_micro::AlignPointerUp(input0, 1);
  TF_LITE_MICRO_EXPECT(input0 == input0_aligned1);

  uint8_t* input0_aligned2 = tflite_micro::AlignPointerUp(input0, 2);
  TF_LITE_MICRO_EXPECT(input0 == input0_aligned2);

  uint8_t* input0_aligned3 = tflite_micro::AlignPointerUp(input0, 3);
  TF_LITE_MICRO_EXPECT(input0 == input0_aligned3);

  uint8_t* input0_aligned16 = tflite_micro::AlignPointerUp(input0, 16);
  TF_LITE_MICRO_EXPECT(input0 == input0_aligned16);

  uint8_t* input23 = reinterpret_cast<uint8_t*>(23);

  uint8_t* input23_aligned1 = tflite_micro::AlignPointerUp(input23, 1);
  TF_LITE_MICRO_EXPECT(input23 == input23_aligned1);

  uint8_t* input23_aligned2 = tflite_micro::AlignPointerUp(input23, 2);
  uint8_t* expected23_aligned2 = reinterpret_cast<uint8_t*>(24);
  TF_LITE_MICRO_EXPECT(expected23_aligned2 == input23_aligned2);

  uint8_t* input23_aligned3 = tflite_micro::AlignPointerUp(input23, 3);
  uint8_t* expected23_aligned3 = reinterpret_cast<uint8_t*>(24);
  TF_LITE_MICRO_EXPECT(expected23_aligned3 == input23_aligned3);

  uint8_t* input23_aligned16 = tflite_micro::AlignPointerUp(input23, 16);
  uint8_t* expected23_aligned16 = reinterpret_cast<uint8_t*>(32);
  TF_LITE_MICRO_EXPECT(expected23_aligned16 == input23_aligned16);
}

TF_LITE_MICRO_TEST(TestAlignPointerDown) {
  uint8_t* input0 = reinterpret_cast<uint8_t*>(0);

  uint8_t* input0_aligned1 = tflite_micro::AlignPointerDown(input0, 1);
  TF_LITE_MICRO_EXPECT(input0 == input0_aligned1);

  uint8_t* input0_aligned2 = tflite_micro::AlignPointerDown(input0, 2);
  TF_LITE_MICRO_EXPECT(input0 == input0_aligned2);

  uint8_t* input0_aligned3 = tflite_micro::AlignPointerDown(input0, 3);
  TF_LITE_MICRO_EXPECT(input0 == input0_aligned3);

  uint8_t* input0_aligned16 = tflite_micro::AlignPointerDown(input0, 16);
  TF_LITE_MICRO_EXPECT(input0 == input0_aligned16);

  uint8_t* input23 = reinterpret_cast<uint8_t*>(23);

  uint8_t* input23_aligned1 = tflite_micro::AlignPointerDown(input23, 1);
  TF_LITE_MICRO_EXPECT(input23 == input23_aligned1);

  uint8_t* input23_aligned2 = tflite_micro::AlignPointerDown(input23, 2);
  uint8_t* expected23_aligned2 = reinterpret_cast<uint8_t*>(22);
  TF_LITE_MICRO_EXPECT(expected23_aligned2 == input23_aligned2);

  uint8_t* input23_aligned3 = tflite_micro::AlignPointerDown(input23, 3);
  uint8_t* expected23_aligned3 = reinterpret_cast<uint8_t*>(21);
  TF_LITE_MICRO_EXPECT(expected23_aligned3 == input23_aligned3);

  uint8_t* input23_aligned16 = tflite_micro::AlignPointerDown(input23, 16);
  uint8_t* expected23_aligned16 = reinterpret_cast<uint8_t*>(16);
  TF_LITE_MICRO_EXPECT(expected23_aligned16 == input23_aligned16);
}

TF_LITE_MICRO_TEST(TestAlignSizeUp) {
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(1), tflite_micro::AlignSizeUp(1, 1));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(2), tflite_micro::AlignSizeUp(1, 2));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(3), tflite_micro::AlignSizeUp(1, 3));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(16), tflite_micro::AlignSizeUp(1, 16));

  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(23), tflite_micro::AlignSizeUp(23, 1));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(24), tflite_micro::AlignSizeUp(23, 2));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(24), tflite_micro::AlignSizeUp(23, 3));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(32), tflite_micro::AlignSizeUp(23, 16));
}

TF_LITE_MICRO_TEST(TestTemplatedAlignSizeUp) {
  // Test structure to test AlignSizeUp.
  struct alignas(32) TestAlignSizeUp {
    // Opaque blob
    float blob_data[4];
  };

  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(32),
                          tflite_micro::AlignSizeUp<TestAlignSizeUp>());
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(64),
                          tflite_micro::AlignSizeUp<TestAlignSizeUp>(2));
}

TF_LITE_MICRO_TEST(TestTypeSizeOf) {
  size_t size;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteFloat16, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(int16_t), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteFloat32, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(float), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteFloat64, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(double), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteInt16, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(int16_t), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteInt32, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(int32_t), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteUInt32, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(uint32_t), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteUInt8, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(uint8_t), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteInt8, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(int8_t), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteInt64, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(int64_t), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteUInt64, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(uint64_t), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteBool, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(bool), size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteComplex64, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(float) * 2, size);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          tflite_micro::TfLiteTypeSizeOf(kTfLiteComplex128, &size));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(double) * 2, size);

  TF_LITE_MICRO_EXPECT_NE(
      kTfLiteOk, tflite_micro::TfLiteTypeSizeOf(static_cast<TfLiteType>(-1), &size));
}

TF_LITE_MICRO_TEST(TestBytesRequiredForTensor) {
  const tflite_micro::Tensor* tensor100 =
      tflite_micro::testing::Create1dFlatbufferTensor(100);
  size_t bytes;
  size_t type_size;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, tflite_micro::BytesRequiredForTensor(
                                         *tensor100, &bytes, &type_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(400), bytes);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), type_size);

  const tflite_micro::Tensor* tensor200 =
      tflite_micro::testing::Create1dFlatbufferTensor(200);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, tflite_micro::BytesRequiredForTensor(
                                         *tensor200, &bytes, &type_size));
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(800), bytes);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<size_t>(4), type_size);
}

TF_LITE_MICRO_TEST(TestAllocateOutputDimensionsFromInput) {
  constexpr int kDimsLen = 4;
  int input1_dims[] = {1, 1};
  int input2_dims[] = {kDimsLen, 5, 5, 5, 5};
  int output_dims[] = {0, 0, 0, 0, 0};
  TfLiteTensor input_tensor1 = tflite_micro::testing::CreateTensor<int32_t>(
      nullptr, tflite_micro::testing::IntArrayFromInts(input1_dims));
  TfLiteTensor input_tensor2 = tflite_micro::testing::CreateTensor<int32_t>(
      nullptr, tflite_micro::testing::IntArrayFromInts(input2_dims));
  TfLiteTensor output_tensor = tflite_micro::testing::CreateTensor<int32_t>(
      nullptr, tflite_micro::testing::IntArrayFromInts(output_dims));
  TfLiteContext context;
  // Only need to allocate space for output_tensor.dims.  Use a simple
  // fake allocator.
  context.AllocatePersistentBuffer = FakeAllocatePersistentBuffer;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite_micro::AllocateOutputDimensionsFromInput(
                     &context, &input_tensor1, &input_tensor2, &output_tensor));

  TF_LITE_MICRO_EXPECT_EQ(output_tensor.bytes, input_tensor2.bytes);
  for (int i = 0; i < kDimsLen; i++) {
    TF_LITE_MICRO_EXPECT_EQ(input_tensor2.dims->data[i],
                            output_tensor.dims->data[i]);
    // Reset output dims for next iteration.
    output_tensor.dims->data[i] = 0;
  }
  // Output tensor size must be 0 to allocate output dimensions from input.
  output_tensor.dims->size = 0;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite_micro::AllocateOutputDimensionsFromInput(
                     &context, &input_tensor2, &input_tensor1, &output_tensor));
  for (int i = 0; i < kDimsLen; i++) {
    TF_LITE_MICRO_EXPECT_EQ(input_tensor2.dims->data[i],
                            output_tensor.dims->data[i]);
  }
  TF_LITE_MICRO_EXPECT_EQ(output_tensor.bytes, input_tensor2.bytes);
}
TF_LITE_MICRO_TESTS_END
