# RimDJ
双点广播电台伴您遨游星海

## 目录结构说明
1. `server`: 服务器端代码，包含核心逻辑和API接口。
2. `Spark-TTS`: 语音合成模型，基于Spark-TTS。
3. `Spark-TTS-0.5B`: 原始模型权重文件。
4. `spark-tts-merged`: 使用unsloth微调合并后的模型权重文件。
5. `spark-tts-tensorrt`: 使用TensorRT加速后的模型权重文件。
6. `trt_engines`: TensorRT引擎文件，用于加速模型推理。

## 模型下载

下载Spark-TTS
```
git clone https://github.com/SparkAudio/Spark-TTS
```

下载原始模型权重文件
```
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir = "Spark-TTS-0.5B")

modelscope download --model SparkAudio/Spark-TTS-0.5B --local_dir ./Spark-TTS-0.5B
```

## 模型微调


## TensorRT编译加速
```
# 使用tensorRT转换模型
# convert_checkpoint.py 为模型转换脚本，Spark-TTS的LLM使用qwen模型架构，因此使用qwen的convert_checkpoint.py
# https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/convert_checkpoint.py
python convert_checkpoint.py --model_dir ./spark-tts-merged --output_dir ./spark-tts-tensorrt --dtype bfloat16
```

```
# 编译推理引擎
trtllm-build --checkpoint_dir ./spark-tts-tensorrt --output_dir ./trt_engines/spark-tts-tensorrt/bf16/1-gpu --gemm_plugin bfloat16
```

## Docker运行环境
```
# 编译tensorrt运行镜像
docker build -t sparktts-unsloth .

# 运行镜像
docker run -it --name dj4090 --gpus all -p 8889:8889 -v /d/code/sparkTTS_unsloth:/workspace sparktts-unsloth python server/server.py
```