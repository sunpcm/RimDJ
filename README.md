# RimDJ
双点广播电台伴您遨游星海

## 目录结构说明
1. `server`: 服务器端代码，包含核心逻辑和API接口。
2. `Spark-TTS`: 语音合成模型，基于Spark-TTS。需下载
3. `Spark-TTS-0.5B`: 原始模型权重文件。需下载
4. `spark-tts-merged`: 使用unsloth微调合并后的模型权重文件。
5. `spark-tts-tensorrt`: 使用TensorRT加速后的模型权重文件。
6. `trt_engines`: TensorRT引擎文件，用于加速模型推理。
7. `music`: 电台背景音乐。暂时有问题server/config.py中关闭了
8. `dataset`: 数据集，用于微调模型。


## 模型下载

下载Spark-TTS
```bash
git clone https://github.com/SparkAudio/Spark-TTS
```

下载原始模型权重文件
```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir = "Spark-TTS-0.5B")
```
或者
```bash
modelscope download --model SparkAudio/Spark-TTS-0.5B --local_dir ./Spark-TTS-0.5B
```

可以提前将需要的镜像pull下来
```bash
docker pull pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
docker pull nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3
```

## 模型微调
运行调试环境后使用 sft.ipynb
```bash
# 编译unslosh微调镜像
docker build -f Dockerfile.dev -t sparktts-unsloth-dev .

## 运行调试环境
docker run -it --name RimDJ-dev --gpus all -p 8888:8888 -v /d/code/RimDJ:/workspace sparktts-unsloth-dev
```




## TensorRT编译加速

使用tensorRT转换模型
```bash
# convert_checkpoint.py 为模型转换脚本，Spark-TTS的LLM使用qwen模型架构，因此使用qwen的convert_checkpoint.py
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/convert_checkpoint.py
# 30系以上显卡可以使用bfloat16
python convert_checkpoint.py --model_dir ./spark-tts-merged --output_dir ./spark-tts-tensorrt --dtype bfloat16
```

编译推理引擎
```bash
trtllm-build --checkpoint_dir ./spark-tts-tensorrt --output_dir ./trt_engines/spark-tts-tensorrt/bf16/1-gpu --gemm_plugin bfloat16
```

## Docker运行环境
```bash
# 编译tensorrt运行镜像
docker build -t sparktts-unsloth .

# 运行本地镜像
docker run -it --name RimDJ-runtime --gpus all -p 8889:8889 -v /d/code/workspace:/workspace sparktts-unsloth python server/server.py
```

## 使用说明
- 网页：http://localhost:8889/
- 接口：http://localhost:8889/text

POST 请求
```json
{
    "text": "事件、日志、主题或任何内容，格式使用纯文本...",
}
```