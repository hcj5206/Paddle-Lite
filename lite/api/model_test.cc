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

#include <gflags/gflags.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/core/device_info.h"
#include "lite/tests/utils/timer.h"
#include "lite/utils/cp_logging.h"
#include "lite/utils/string.h"

using paddle::lite::Timer;

DEFINE_string(input_shape,
              "1,3,224,224",
              "input shapes, separated by colon and comma");

DEFINE_bool(use_optimize_nb,
            false,
            "optimized & naive buffer model for mobile devices");

namespace paddle {
namespace lite_api {

void OutputOptModel(const std::string& load_model_dir,
                    const std::string& save_optimized_model_dir,
                    const std::vector<std::vector<int64_t>>& input_shapes) {
  lite_api::CxxConfig config;
  config.set_model_dir(load_model_dir);
  config.set_valid_places({
      Place{TARGET(kX86), PRECISION(kFloat)},
      Place{TARGET(kARM), PRECISION(kFloat)},
  });
  auto predictor = lite_api::CreatePaddlePredictor(config);

  // delete old optimized model
  int ret = system(
      paddle::lite::string_format("rm -rf %s", save_optimized_model_dir.c_str())
          .c_str());
  if (ret == 0) {
    LOG(INFO) << "delete old optimized model " << save_optimized_model_dir;
  }
  predictor->SaveOptimizedModel(save_optimized_model_dir,
                                LiteModelType::kNaiveBuffer);
  LOG(INFO) << "Load model from " << load_model_dir;
  LOG(INFO) << "Save optimized model to " << save_optimized_model_dir;
}
//////////////////hcj + 20191027 for test
//字符串转化
template <class Type>
Type stringToNum(const std::string& str)
{
    std::istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}
std::string Read_Str(std::string filepath)
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
int Write_Str(std::string str,std::string filepath)
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
std::vector<std::string> split(const std::string& s, char delimiter)
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
void Run(const std::vector<std::vector<int64_t>>& input_shapes,
         const std::string& model_dir,
         const PowerMode power_mode,
         const int thread_num,
         const int repeat,
         const int warmup_times = 0) {
  lite_api::MobileConfig config;
  config.set_model_dir(model_dir);
  config.set_power_mode(power_mode);
  config.set_threads(thread_num);
  std::cout<<"hcj:start read data:"<<std::endl;
  std::string str=Read_Str("./data.txt");
  std::vector<std::string> tokens;
  char b = ' ';
  tokens = split(str, b);
  std::cout<<"hcj:end read data:"<<std::endl;
  auto predictor = lite_api::CreatePaddlePredictor(config);

  for (int j = 0; j < input_shapes.size(); ++j) {
    auto input_tensor = predictor->GetInput(j);
    input_tensor->Resize(input_shapes[j]);
    auto input_data = input_tensor->mutable_data<float>();
    int input_num = 1;
    for (int i = 0; i < input_shapes[j].size(); ++i) {
      input_num *= input_shapes[j][i];
    }
    for (int i = 0; i < input_num; ++i) {
      input_data[i] = stringToNum<float >(tokens[i]);
    }
    // for (int i=0;i<10;++i){
    //  std::cout<<"hcj input_data:"<<input_data[i];
    // }

  }
  std::cout<<"hcj:write read data:"<<std::endl;


  for (int i = 0; i < warmup_times; ++i) {
    predictor->Run();
  }

  Timer ti;
  for (int j = 0; j < repeat; ++j) {
    ti.start();
    predictor->Run();
    ti.end();
    LOG(INFO) << "iter: " << j << ", time: " << ti.latest_time() << " ms";
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << model_dir
            << ", power_mode: " << static_cast<int>(power_mode)
            << ", threads num " << thread_num << ", warmup: " << warmup_times
            << ", repeats: " << repeat << ", avg time: " << ti.get_average_ms()
            << " ms"
            << ", min time: " << ti.get_min_time() << " ms"
            << ", max time: " << ti.get_max_time() << " ms.";

  auto output = predictor->GetOutput(0);
  auto out = output->data<float>();
  // for (int i=0;i<10;++i){
  //    std::cout<<"hcj out:"<<out[i];
  //   }
  // LOG(INFO) << "out " << out[0];
  // LOG(INFO) << "out " << out[1];
  // LOG(INFO) << "out " << out[2];
  auto output_shape = output->shape();
  int output_num = 1;
  for (int i = 0; i < output_shape.size(); ++i) {
    output_num *= output_shape[i];
  }
  std::string all_data="";
  for (int i=0;i<output_num;++i){
    all_data+=std::to_string(out[i])+" ";
  }
  Write_Str(all_data,"data_out.txt");
  LOG(INFO) << "output_num: " << output_num;
}


}  // namespace lite_api
}  // namespace paddle

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir == "") {
    LOG(INFO) << "usage: "
              << "--model_dir /path/to/your/model";
    exit(0);
  }
  std::string save_optimized_model_dir = "";
  if (FLAGS_use_optimize_nb) {
    save_optimized_model_dir = FLAGS_model_dir;
  } else {
    save_optimized_model_dir = FLAGS_model_dir + "opt2";
  }

  auto split_string =
      [](const std::string& str_in) -> std::vector<std::string> {
    std::vector<std::string> str_out;
    std::string tmp_str = str_in;
    while (!tmp_str.empty()) {
      size_t next_offset = tmp_str.find(":");
      str_out.push_back(tmp_str.substr(0, next_offset));
      if (next_offset == std::string::npos) {
        break;
      } else {
        tmp_str = tmp_str.substr(next_offset + 1);
      }
    }
    return str_out;
  };

  auto get_shape = [](const std::string& str_shape) -> std::vector<int64_t> {
    std::vector<int64_t> shape;
    std::string tmp_str = str_shape;
    while (!tmp_str.empty()) {
      int dim = atoi(tmp_str.data());
      shape.push_back(dim);
      size_t next_offset = tmp_str.find(",");
      if (next_offset == std::string::npos) {
        break;
      } else {
        tmp_str = tmp_str.substr(next_offset + 1);
      }
    }
    return shape;
  };

  LOG(INFO) << "input shapes: " << FLAGS_input_shape;
  std::vector<std::string> str_input_shapes = split_string(FLAGS_input_shape);
  std::vector<std::vector<int64_t>> input_shapes;
  for (int i = 0; i < str_input_shapes.size(); ++i) {
    LOG(INFO) << "input shape: " << str_input_shapes[i];
    input_shapes.push_back(get_shape(str_input_shapes[i]));
  }

  if (!FLAGS_use_optimize_nb) {
    // Output optimized model
    paddle::lite_api::OutputOptModel(
        FLAGS_model_dir, save_optimized_model_dir, input_shapes);
  }

#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  // Run inference using optimized model
  paddle::lite_api::Run(
      input_shapes,
      save_optimized_model_dir,
      static_cast<paddle::lite_api::PowerMode>(FLAGS_power_mode),
      FLAGS_threads,
      FLAGS_repeats,
      FLAGS_warmup);
#endif
  return 0;
}
