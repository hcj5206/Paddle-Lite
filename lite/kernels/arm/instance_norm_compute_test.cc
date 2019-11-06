// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/arm/instance_norm_compute.h"
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void InstanceNormComputeRef(const operators::InstanceNormParam& param) {
  auto* x = param.X;
  auto* y = param.Y;
  auto* scale_tensor = param.Scale;
  auto* bias_tensor = param.Bias;
  auto* mean_tensor = param.Mean;
  auto* var_tensor = param.Variance;

  int begin_norm_axis = param.begin_norm_axis;
  float epsilon = param.epsilon;

  auto* x_data = x->data<float>();
  auto* scale_data =
      (scale_tensor == nullptr ? nullptr : scale_tensor->data<float>());
  auto* bias_data =
      (bias_tensor == nullptr ? nullptr : bias_tensor->data<float>());
  auto* out_data = y->mutable_data<float>();
  auto* mean_data = mean_tensor->mutable_data<float>();
  auto* var_data = var_tensor->mutable_data<float>();
  begin_norm_axis=2;
  auto matrix_dim = x->dims().Flatten2D(begin_norm_axis);
  int batch_size = matrix_dim[0];
  int feature_size = matrix_dim[1];
  for (int i = 0; i < batch_size; ++i) {
    int start = i * feature_size;
    int end = start + feature_size;

    float mean = 0;
    float var = 0;
    for (int j = start; j < end; ++j) {
      mean += x_data[j];
      var += x_data[j] * x_data[j];
    }
    mean /= feature_size;
    var = var / feature_size - mean * mean;
    mean_data[i] = mean;
    var_data[i] = var;
    var = sqrt(var + epsilon);
    for (int j = start; j < end; ++j) {
      out_data[j] = (x_data[j] - mean) / var;
      if (scale_data) {
        out_data[j] *= scale_data[i];
      }
      if (bias_data) {
        out_data[j] += bias_data[i];
      }
    }
  }
}
//////////////////hcj + 20191027 for test
//字符串转化
template <class Type>
Type stringToNum1(const std::string& str)
{
    std::istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}
std::string Read_Str1(std::string filepath)
{
    std::ifstream infile;
    infile.open(filepath);
    //打开失败，路径不正确
    if(!infile)
        std::cout << "Open File Fail!" << std::endl;
    //读取文本内容到字符串
    std::string readStr((std::istreambuf_iterator<char>(infile)),  std::istreambuf_iterator<char>());
    return readStr;
}
int Write_Str1(std::string str,std::string filepath)
{
    std::ofstream outfile;
    outfile.open(filepath);
    //打开失败，路径不正确
    if(!outfile)
        std::cout << "Open File Fail!" << std::endl;
    //读取文本内容到字符串
    outfile << str;
    return 1;
}
std::vector<std::string> split1(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
           tokens.push_back(token);
         }
     return tokens;
}
//////////////////hcj + 20191027
TEST(instance_norm_arm, init) {
  InstanceNormCompute instance_norm;
  ASSERT_EQ(instance_norm.precision(), PRECISION(kFloat));
  ASSERT_EQ(instance_norm.target(), TARGET(kARM));
}

TEST(instance_norm_arm, compute) {
  InstanceNormCompute instance_norm;
  operators::InstanceNormParam param;

  lite::Tensor x;
  lite::Tensor output;
  lite::Tensor output_mean;
  lite::Tensor output_var;
  lite::Tensor output_ref;
  lite::Tensor output_mean_ref;
  lite::Tensor output_var_ref;
  lite::Tensor bias;
  lite::Tensor scale;
  lite::Tensor* bias_ptr;
  lite::Tensor* scale_ptr;

  for (auto n : {1}) {
    for (auto c : {3}) {
      for (auto h : {256}) {
        for (auto w : {256}) {
          for (auto axis : {2}) {
            for (auto has_bias : {true}) {
              for (auto has_scale : {true}) {
                auto dims = DDim(std::vector<int64_t>({n, c, h, w}));
                auto out_size = dims.Flatten2D(axis)[0];
                auto inner_size = dims.Flatten2D(axis)[1];
                bias_ptr = nullptr;
                scale_ptr = nullptr;
                if (has_bias) {
                  bias.Resize(std::vector<int64_t>({out_size}));
                  float* bias_data = bias.mutable_data<float>();
                  for (int i = 0; i < inner_size; ++i) {
                    bias_data[i] = 0.01;
                  }
                  bias_ptr = &bias;
                }
                if (has_scale) {
                  scale.Resize(std::vector<int64_t>({out_size}));
                  float* scale_data = scale.mutable_data<float>();
                  for (int i = 0; i < inner_size; ++i) {
                    scale_data[i] = 0.2;
                  }
                  scale_ptr = &scale;
                }

                x.Resize(dims);
                output.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                output_ref.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
                output_mean.Resize(std::vector<int64_t>({out_size, 1, 1, 1}));
                output_mean_ref.Resize(
                    std::vector<int64_t>({out_size, 1, 1, 1}));
                output_var.Resize(std::vector<int64_t>({out_size, 1, 1, 1}));
                output_var_ref.Resize(
                    std::vector<int64_t>({out_size, 1, 1, 1}));

                auto* x_data = x.mutable_data<float>();
                auto* output_data = output.mutable_data<float>();
                auto* output_mean_data = output_mean.mutable_data<float>();
                auto* output_var_data = output_var.mutable_data<float>();
                auto* output_data_ref = output_ref.mutable_data<float>();
                auto* output_mean_data_ref =
                    output_mean_ref.mutable_data<float>();
                auto* output_var_data_ref =
                    output_var_ref.mutable_data<float>();
                //test hcj
                std::string str=Read_Str1("./data.txt");
                std::vector<std::string> tokens;
                char b = ' ';
                tokens = split1(str, b);
                //hcj
                std::cout<<"x.dims().production()="<<x.dims().production()<<std::endl;
                for (int i = 0; i < x.dims().production(); i++) {
                  x_data[i] = stringToNum1<float >(tokens[i]);
                }
                param.X = &x;
                param.Y = &output;
                param.begin_norm_axis = axis;
                param.Bias = bias_ptr;
                param.Scale = scale_ptr;
                param.Mean = &output_mean;
                param.Variance = &output_var;
                param.epsilon = 0.00001;
                instance_norm.SetParam(param);
                instance_norm.Run();
                VLOG(4) << output_data[0];
                VLOG(4) << output_data[1];
                param.Y = &output_ref;
                param.Mean = &output_mean_ref;
                param.Variance = &output_var_ref;
                InstanceNormComputeRef(param);
                std::string all_data1="";
                 for (int i = 0; i < output.dims().production(); i++) {
                
                  all_data1+=std::to_string(output_data[i])+" ";

                }
                 Write_Str1(all_data1,"output_data_test.txt");
                
                
               
                for (int i = 0; i < output.dims().production(); i++) {
                  
                  EXPECT_NEAR(output_data[i], output_data_ref[i], 1e-4);
                }
                for (int i = 0; i < output_mean.dims().production(); ++i) {
                  EXPECT_NEAR(
                      output_mean_data[i], output_mean_data_ref[i], 1e-5);
                  EXPECT_NEAR(output_var_data[i], output_var_data_ref[i], 1e-5);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(instance_norm, retrive_op) {
  auto instance_norm =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "instance_norm");
  ASSERT_FALSE(instance_norm.empty());
  ASSERT_TRUE(instance_norm.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(instance_norm, kARM, kFloat, kNCHW, def);
