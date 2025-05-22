## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.10 / 3.11 | - |
  | torch | 2.1.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I A2/Atlas 800T A2设备：支持的卡数为1
- [Atlas 800I A2](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [Atlas 800T A2](https://www.hiascend.com/developer/download/community/result?module=pt+cann&product=4&model=26)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/softwareinst/instg/instg_0003.html)

### 1.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 1.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.4 Torch_npu安装
安装pytorch框架 版本2.1.0
[安装包下载](https://download.pytorch.org/whl/cpu/torch/)

使用pip安装
```shell
# {version}表示软件版本号，{arch}表示CPU架构。
pip install torch-${version}-cp310-cp310-linux_${arch}.whl
```
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```
## 二、下载本仓库

### 2.1 下载到本地
```shell
git clone https://modelers.cn/MindIE/HiDream-I1.git
```

### 2.2 安装依赖
```shell
pip install -r requirements.txt
```

## 三、HiDream-I1使用

### 3.1 权重及配置文件说明
#### 1. 权重链接:
```shell
https://huggingface.co/HiDream-ai/HiDream-I1-Full/tree/main
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main
```

#### 2. 各模型的配置文件、权重文件的层级样例如下所示:
```commandline
|----main
|    |---- model_index.json
|    |---- scheduler
|    |---- text_encoder
|    |---- text_encoder_2
|    |---- text_encoder_3
|    |---- tokenizer
|    |---- tokenizer_2
|    |---- tokenizer_3
|    |---- transformer
|    |    |---- config.json
|    |    |---- 模型权重
|    |---- vae
```

### 3.2 单卡单batch推理性能测试
#### 1. 进入主目录
```shell
cd HiDream-I1
```

#### 2. 设置权重路径
```shell
path="/data/HiDream-I1-Full"
extra_path="/data/Llama-3.1-8B-Instruct"
```

#### 3. 执行命令，进行推理：
```shell
python inference.py \
       --model_path ${path} \
       --model_path_extra ${extra_path} \
       --resolution "1024 x 1024" \
       --num_inference_steps 50 \
       --device_id 0
```
参数说明：
- model_path：HiDream-I1-Full权重路径。
- model_path_extra：Llama-3.1-8B-Instruct权重路径。
- resolution：需要生成的图像的分辨率。
- num_inference_steps：推理迭代步数。
- device_id：推理设备ID。

**注意**：本仓库模型，是对开源模型进行优化。用户在使用时，应对开源代码函数的变量范围，类型进行校验，避免出现变量超出范围、除零等操作。

#### 4. 采Profiling：
```shell
python inference.py \
       --model_path ${path} \
       --model_path_extra ${extra_path} \
       --resolution "1024 x 1024" \
       --num_inference_steps 4 \
       --device_id 0 \
       --infer_type Profiling

```
参数说明：
- model_path：HiDream-I1-Full权重路径。
- model_path_extra：Llama-3.1-8B-Instruct权重路径。
- resolution：需要生成的图像的分辨率。
- num_inference_steps：推理迭代步数。
- device_id：推理设备ID。
- infer_type: 推理类型，包括Default、Profiling和Accuracy。


### 3.3 单卡多batch推理功能测试
#### 1. 进入主目录
```shell
cd HiDream-I1
```

#### 2. 设置权重路径
```shell
path="/data/HiDream-I1-Full"
extra_path="/data/Llama-3.1-8B-Instruct"
```

#### 3. 执行命令，进行推理：
```shell
python inference.py \
       --model_path ${path} \
       --model_path_extra ${extra_path} \
       --resolution "1024 x 1024" \
       --num_inference_steps 50 \
       --num_images_per_prompt 2 \
       --batch_size 2 \
       --device_id 0
```
参数说明：
- model_path：HiDream-I1-Full权重路径。
- model_path_extra：Llama-3.1-8B-Instruct权重路径。
- resolution：需要生成的图像的分辨率。
- num_inference_steps：推理迭代步数。
- num_images_per_prompt: 每个提示词生成的图像数量。
- batch_size: 推理时的batch_size。
- device_id：推理设备ID。


### 3.4 精度验证

#### 1. 由于生成的图片存在随机性，提供两种精度验证方法：
1. CLIP-score（文图匹配度量）：评估图片和输入文本的相关性，分数的取值范围为[-1, 1]，越高越好。使用Parti数据集进行验证。
2. HPSv2（图片美学度量）：评估生成图片的人类偏好评分，分数的取值范围为[0, 1]，越高越好。使用HPSv2数据集进行验证

注意，由于要生成的图片数量较多，进行完整的精度验证需要耗费很长的时间。

#### 2. 下载Parti数据集和hpsv2数据集
所有数据集放到`HiDream-I1/prompts`目录下
```bash
# 下载Parti数据集
wget https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv --no-check-certificate
```
hpsv2数据集下载链接：https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl/hpsv2_benchmark_prompts.json

#### 3. 设置模型权重路径
```shell
path="/data/HiDream-I1-Full"
extra_path="/data/Llama-3.1-8B-Instruct"
```

#### 4. 使用推理脚本读取Parti数据集，生成图片
```bash
python3 inference.py \
        --model_path ${path} \
        --model_path_extra ${extra_path} \
        --prompt_file ./prompts/PartiPrompts.tsv \
        --prompt_file_type parti \
        --info_file_save_path ./image_info_PartiPrompts.json \
        --save_dir ./results_PartiPrompts \
        --num_images_per_prompt 4 \
        --resolution "1024 x 1024" \
        --batch_size 1 \
        --seed 42 \
        --device_id 0 \
        --infer_type Accuracy
```
参数说明：
- model_path：HiDream-I1-Full权重路径。
- model_path_extra：Llama-3.1-8B-Instruct权重路径。
- prompt_file：提示词文件。
- prompt_file_type: prompt文件类型，用于指定读取方式，可选plain，parti，hpsv2。
- info_file_save_path：生成图片信息的json文件路径。
- save_dir：生成图片的存放目录。
- num_images_per_prompt: 每个prompt生成的图片数量。注意使用hpsv2时，设置num_images_per_prompt=1即可。
- resolution：需要生成的图像的分辨率。
- batch_size：模型batch size。
- seed：随机种子。
- device_id：推理设备ID。
- infer_type: 推理类型，包括Default、Profiling和Accuracy。

执行完成后在`./results_PartiPrompts`目录下生成推理图片，在当前目录生成一个`image_info_PartiPrompts.json`文件，记录着图片和prompt的对应关系，并在终端显示推理时间。

#### 5. 使用推理脚本读取hpsv2数据集，生成图片
```bash
python3 inference.py \
        --model_path ${path} \
        --model_path_extra ${extra_path} \
        --prompt_file ./prompts/hpsv2_benchmark_prompts.json \
        --prompt_file_type hpsv2 \
        --info_file_save_path ./image_info_hpsv2.json \
        --save_dir ./results_hpsv2 \
        --num_images_per_prompt 1 \
        --resolution "1024 x 1024" \
        --batch_size 1 \
        --seed 42 \
        --device_id 0 \
        --infer_type Accuracy
```
参数说明：
- model_path：HiDream-I1-Full权重路径。
- model_path_extra：Llama-3.1-8B-Instruct权重路径。
- prompt_file：提示词文件。
- prompt_file_type: prompt文件类型，用于指定读取方式，可选plain，parti，hpsv2。
- info_file_save_path：生成图片信息的json文件路径。
- save_dir：生成图片的存放目录。
- num_images_per_prompt: 每个prompt生成的图片数量。注意使用hpsv2时，设置num_images_per_prompt=1即可。
- resolution：需要生成的图像的分辨率。
- batch_size：模型batch size。
- seed：随机种子。
- device_id：推理设备ID。
- infer_type: 推理类型，包括Default、Profiling和Accuracy。

执行完成后在`./results_hpsv2`目录下生成推理图片，在当前目录生成一个`image_info_hpsv2.json`文件，记录着图片和prompt的对应关系，并在终端显示推理时间。

#### 6. 计算精度指标(GPU)
##### 1. 下载模型权重
所有权重下载到`HiDream-I1/`目录下
```bash
# Clip Score和HPSv2均需要使用的权重
# 安装git-lfs
apt install git-lfs
git lfs install

# Clip Score权重
git clone https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

# HPSv2权重
wget https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt --no-check-certificate
```
也可手动下载[CLIP权重](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin)将权重放到`CLIP-ViT-H-14-laion2B-s32B-b79K`目录下，手动下载[HPSv2权重](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt)放到当前路径

##### 2. CLIP-score精度指标计算
```bash
python3 clip_score.py \
      --device=cuda \
      --image_info="./image_info_PartiPrompts.json" \
      --model_name="ViT-H-14" \
      --model_weights_path="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
```
参数说明：
- --device: 推理设备（CPU或者GPU）。
- --image_info: 上一步生成的`image_info_PartiPrompts.json`文件。
- --model_name: Clip模型名称。
- --model_weights_path: Clip模型权重文件路径。

执行完成后会在屏幕打印出精度计算结果。

##### 3. HPSv2精度指标计算
```bash
python3 hpsv2_score.py \
      --image_info="image_info_hpsv2.json" \
      --HPSv2_checkpoint="./HPS_v2_compressed.pt" \
      --clip_checkpoint="./CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
```

参数说明：
- --image_info: 上一步生成的`image_info_hpsv2.json`文件。
- --HPSv2_checkpoint: HPSv2模型权重文件路径。
- --clip_checkpointh: Clip模型权重文件路径。

执行完成后会在屏幕打印出精度计算结果。

### CogView3plus

| 硬件形态 | 迭代次数 | 加速算法 | 平均耗时 | CLIP_score | HPSV2_score |
| :------: |:----:|:----:|:----:|:----:|:----:|


## 四、优化指南
本模型使用的优化手段如下：
- 等价优化：
- 算法优化：


## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。