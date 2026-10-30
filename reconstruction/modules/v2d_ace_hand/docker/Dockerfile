FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /opt/ace
RUN git clone https://github.com/ggxxii/ACE-Ego-Hand . && git checkout 59da83b58d58c3041b9ee0b35b5054866e33a990 && git clone https://github.com/aigc-apps/VideoX-Fun third_party && cd third_party && git checkout 13bb3c625e8b1e457d7124d92207b49e00e4a6fe
RUN uv pip install --system -r requirements.txt scipy sentencepiece Pillow
# Import only the Wan components ACE uses; VideoX's omnibus initializers pull
# unrelated image/audio models and dependencies into every inference process.
RUN printf 'from .wan_text_encoder import WanT5EncoderModel\nfrom .wan_vae3_8 import AutoencoderKLWan3_8\n' > third_party/videox_fun/models/__init__.py && printf 'from .cfg_optimization import cfg_skip\n' > third_party/videox_fun/utils/__init__.py && printf 'from .fuser import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group, xFuserLongContextAttention\nfrom .wan_xfuser import usp_attn_forward\n' > third_party/videox_fun/dist/__init__.py
ENV PYTHONPATH=/opt/ace:/opt/ace/third_party
