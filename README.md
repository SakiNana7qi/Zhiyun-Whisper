# Zhiyun-Whisper

浙江大学智云课堂语音转录工具 — 自动下载课程录播并使用 Whisper 进行语音转文字。

## 功能

- 获取智云课堂课程录播视频
- 支持本地 faster-whisper 和 OpenAI Whisper API 两种转录模式
- 输出 `.txt`（纯文本）和 `.srt`（带时间戳字幕）格式
- 大文件下载自动断点续传

## 前置条件

1. **Python 3.10+**
2. **ffmpeg** — 需安装并加入系统 PATH
   ```bash
   # Windows (scoop)
   scoop install ffmpeg
   # Windows (choco)
   choco install ffmpeg
   ```
3. **CUDA**（可选）— 本地 Whisper 模式使用 GPU 加速，CPU 也可运行但较慢

## 安装

```bash
pip install -r requirements.txt
```

## 配置

在项目根目录 `.env` 文件中填写：

```env
ZJU_USERNAME="你的学号"
ZJU_PASSWORD="你的密码"

# 仅 API 模式需要
OPENAI_API_KEY="sk-..."
```

### HuggingFace 镜像（国内用户必配）

首次运行本地模式时需要从 HuggingFace 下载 Whisper 模型，国内无法直连。
需要设置环境变量使用镜像：

**Windows (PowerShell):**
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
# 然后在同一终端运行 python main.py ...
```

**永久生效（推荐）：** 在系统环境变量中添加 `HF_ENDPOINT`，值为 `https://hf-mirror.com`。

## 使用

```bash
# 转录指定课次（本地 Whisper，默认 large-v3 模型）
python main.py transcribe "https://classroom.zju.edu.cn/livingroom?course_id=81771&sub_id=1892675&tenant_code=112"

# 使用 OpenAI API 转录
python main.py transcribe "URL" --mode api

# 指定模型大小（tiny/base/small/medium/large-v3）
python main.py transcribe "URL" --mode local --model medium

# 列出某门课所有课次
python main.py list --course-id 81771
```

## 输出

转录结果保存在 `output/` 目录下，每个课次生成：
- `课次标题.mp4` — 下载的视频文件
- `课次标题.wav` — 提取的音频文件
- `课次标题.txt` — 纯文本转录
- `课次标题.srt` — 带时间戳的字幕文件
