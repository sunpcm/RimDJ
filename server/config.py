"""
config.py: Global configuration parameters for the real-time broadcast system.
"""

# Model paths (adjust based on actual project structure)
TTS_MODEL_PATH = "../spark-tts-tensorrt"  # Path to TensorRT accelerated TTS model
LLM_MODEL_NAME = "deepseek-chat"  # LLM model name
LLM_BASE_URL = "https://api.deepseek.com"  # LLM API base URL
LLM_API_KEY = ""  # Replace with actual API key

# Audio processing parameters
SAMPLE_RATE = 16000  # Audio sample rate
CHUNK_SIZE = 16000  # Size of audio chunks for streaming (1000ms at 16kHz)
CHUNK_DURATION_MS = 1000  # Chunk duration in milliseconds (1 seconds)

# BGM specific parameters
BGM_ENABLED = False  # BGM启用开关，设为False时TTS音频直接跳过混音器
BGM_TRACK_GAP_SECONDS = 2.0  # 音轨间隔时间（秒）
BGM_LOAD_RETRY_DELAY = 5.0   # 加载失败重试延迟（秒）
BGM_QUEUE_FULL_WAIT = 0.01   # 队列满时等待时间（秒）

# Queue configuration
TEXT_QUEUE_SIZE = 50
BROADCAST_QUEUE_SIZE = 50
WAV_QUEUE_SIZE = 100  # 增加WAV队列容量，避免BGM音频块丢失
BGM_WAV_QUEUE_SIZE = 1000  # BGM专用WAV队列容量，增加以支持快速生成模式
MIXED_WAV_QUEUE_SIZE = 200  # 混合音频队列容量
MP3_QUEUE_SIZE = 300
CLIENT_QUEUE_SIZE = 300  # 增加客户端队列容量

# Timeout configuration (seconds)
REQUEST_TIMEOUT = 30.0  # HTTP请求超时
STREAM_TIMEOUT = 60.0   # 流超时
LLM_TIMEOUT = 30.0      # LLM请求超时
TTS_TIMEOUT = 30.0      # TTS生成超时
MP3_ENCODE_TIMEOUT = 10.0  # MP3编码超时
MP3_TIMEOUT = 10.0      # MP3处理超时
CLIENT_TIMEOUT = 60.0
QUEUE_PUT_TIMEOUT = 2.0

# Error handling
MAX_CONSECUTIVE_ERRORS = 10
ERROR_SLEEP_TIME = 2.0

# FastAPI server configuration
HOST = "0.0.0.0"
PORT = 8889
LOG_LEVEL = "info"