FROM nvcr.io/nvidia/tritonserver:25.02-trtllm-python-py3
RUN apt-get update && apt-get install -y cmake
RUN git clone https://github.com/pytorch/audio.git && cd audio && git checkout c670ad8 && PATH=/usr/local/cuda/bin:$PATH python3 setup.py develop
RUN pip install einx==0.3.0 omegaconf==2.3.0 soundfile==0.12.1 soxr==0.5.0.post1 gradio tritonclient librosa jupyter lameenc
# RUN pip install modelscope datasets
# RUN pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
WORKDIR /workspace

# 暴露Jupyter端口
EXPOSE 8888
EXPOSE 8889

# 启动Jupyter，使用参数配置
CMD ["jupyter", "notebook", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''", "--NotebookApp.password=''"]