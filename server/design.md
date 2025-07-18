# 设计文档

我计划在server目录下编写一个mp3流媒体的功能，整个服务是一个实时广播，该目录下的文件结构计划如下：
- server.py
- llm_service.py
- tts_service.py
- mp3_service.py
- config.py
- system_prompt.py


接下来我将详细解释每个文件的开发设计：

## server.py 

是一个基于fastapi的启动文件  if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8889, log_level="info")

这个server.py 文件里需要提供两个接口：
- text接口，用户可以提交文本到此接口，提交的text将被放入一个内存队列中
- audio_stream接口，该接口输出一个实时的mp3音频流媒体

这个server.py 文件里需要维护几份队列：
- text文本队列，存放用户通过text接口提交的文本
- broadcast_script队列，通过LLM服务，将text队列里面的原始文本信息，变成广播稿并添加到broadcast_script进行维护
- wav音频队列，通过TTS服务，将broadcast_script队列里面的广播稿，变成wav音频并添加到wav音频队列
- mp3_stream队列，通过MP3服务，将wav音频队列里面的音频，变成mp3流媒体并添加到mp3_stream队列

## llm_service.py

这个llm_service.py 文件里需要做几件事：
- 从text文本队列取出文本，调用LLM模型进行处理，变成更符合广播要求的文本
- 将处理后的文本，添加到broadcast_script队列
- system_prompt.py里面存放LLM的系统提示词
- 从text文本队列取出文本，作为用户提示词
- 当text文本的队列长度超过5，一次性取出全部队列
- 当text文本的队列长度小于5，则等待5秒
- 因为请求LLM需要时间，所以一次处理完后再进行请求，循环处理。刚好下次循环时，text文本队列里面可能又会有新的内容被push进啦
- 这个llm_service使用单独的子线程处理队列，避免阻塞主线程

## tts_service.py

这个tts_service.py 文件里需要做几件事：
- 从broadcast_script队列取出广播文本，调用TTS模型进行处理，变成wav音频
- 将处理后的音频，添加到wav音频队列
- 参考根目录中test_stream.py中，加载模型/切分语句/生成音频的操作
- 我使用tensorRT引擎加速推理
- 这个tts_service使用单独的子线程处理队列，避免阻塞主线程

## mp3_service.py
这个mp3_service.py 文件里需要做几件事：
- 从wav音频队列取出音频，调用MP3模型进行处理，变成mp3文件流chunk
- mp3编码后的chunk，发送给客户端广播
- test_stream.py也有关于mp3编码和chunk的参考实现，这里不再赘述

## config.py
定义一些全局参数：
- 各模型路径
- LLM的baseurl地址、模型名、apikey
- 音频处理参数
- fastapi的端口号


设计目标是完成一个实时的语音合成广播系统，用于娱乐而非严肃场景，因此更倾向于系统的实时性能，数据丢失问题不大。不要进行负载冗余的设计，基于最小化原则开发。
请完成以上文件的编码工作


