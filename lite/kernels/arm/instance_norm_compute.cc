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
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
///hcj
// template<class T>
// int hcj_length(T& arr)
// {
//     //cout << sizeof(arr[0]) << endl;
//     //cout << sizeof(arr) << endl;
//     return sizeof(arr) / sizeof(arr[0]);
// }
// int Write_Str(std::string str,std::string filepath)
// {
//     std::ofstream outfile;
//     outfile.open(filepath);
//     //打开失败，路径不正确
//     if(!outfile)
//         std::cout << "Open File Fail!" << std::endl;
//     //读取文本内容到字符串
//     outfile << str;
//     return 1;
// }
// //hcj
void InstanceNormCompute::PrepareForRun() {}

void InstanceNormCompute::Run() {
  auto& param = this->Param<operators::InstanceNormParam>();
  auto input_dims = param.X->dims();
  const auto* x_data = param.X->data<float>();
  const auto* scale = param.Scale ? param.Scale->data<float>() : nullptr;
  const auto* bias = param.Bias ? param.Bias->data<float>() : nullptr;
  // const auto* scale = false ? param.Scale->data<float>() : nullptr;
  // const auto* bias = false ? param.Bias->data<float>() : nullptr;
  auto* o_data = param.Y->mutable_data<float>();
  auto* mean = param.Mean->mutable_data<float>();
  auto* var = param.Variance->mutable_data<float>();
  int axis = param.begin_norm_axis;
  axis=2;
  auto matrix_dim = param.X->dims().Flatten2D(axis);
  int left = matrix_dim[0];
  int right = matrix_dim[1];
  // auto hcjbias = param.Scale->dims();
  // auto hcjScale = param.Bias->dims();

  // std::string all_data="";
  // for (int i=0;i<left*right;++i){
  //   all_data+=std::to_string(x_data[i])+" ";
  // }
  // int a=1;
  // bool flag= true;
  // while(flag){
  //     std::string FileName;
  //     FileName="Instance_in_data"+ std::to_string(a)+".txt";
  //     std::ifstream fin(FileName);
  //     if(fin)
  //     {
  //         fin.close();
  //         a+=1;
  //     } else{
  //         Write_Str(all_data,FileName);
  //         flag= false;
  //     }
  // }

  // for (int j = 0; j < 20; ++j) {
  // std::cout<<"x_data="<<x_data[j]<<std::endl;
  // }
  //   std::cout<<"x_data[left*right-2]="<<x_data[left*right-2]<<std::endl;
  //   std::cout<<"x_data[left*right-1]="<<x_data[left*right-1]<<std::endl;
  // if(scale){
  // std::cout<<"input_dims="<<input_dims<<std::endl;
  // std::cout<<"hcjbias="<<hcjbias<<std::endl;
  
  // std::cout<<"hcjScale="<<hcjScale<<std::endl;
  // std::cout<<"hcjScale0="<<scale[0]<<std::endl;
  // std::cout<<"hcjScale1="<<scale[1]<<std::endl;

  // }
  // else
  // {
  //   std::cout<<"scale="<<scale<<std::endl;
  //   std::cout<<"bias="<<bias<<std::endl;
  // }
  // std::cout<<"left="<<left<<"right"<<right<<std::endl;
  // std::cout<<"param.epsilon="<<param.epsilon<<std::endl;



  lite::arm::math::instance_norm_math(
      x_data, scale, bias, o_data, mean, var, param.epsilon, left, right);


  // std::cout<<"hcj110"<<std::endl;
  // std::string all_data2="";
  // for (int i=0;i<left*right;++i){
  //   all_data2+=std::to_string(o_data[i])+" ";
  // }
  //  std::cout<<"hcj111"<<std::endl;
  // a=1;
  // bool flag1= true;
  // while(flag1){
  //     std::string FileName;
  //     FileName="Instance_out_data"+ std::to_string(a)+".txt";
  //     std::ifstream fin(FileName);
  //     if(fin)
  //     {
  //         fin.close();
  //         a+=1;
  //     } else{
  //         Write_Str(all_data2,FileName);
  //         flag1= false;
  //     }
  // }
  //    std::cout<<"hcj112"<<std::endl;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(instance_norm,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::InstanceNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
