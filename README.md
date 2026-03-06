# Zhiyun-Whisper

浙江大学智云课堂语音转录工具 — 自动下载课程录播并使用 Whisper 进行语音转文字。

## 功能

- 自动登录浙大统一认证
- 获取智云课堂课程录播视频
- 支持本地 faster-whisper 和 OpenAI Whisper API 两种转录模式
- 输出 `.txt`（纯文本）和 `.srt`（带时间戳字幕）格式

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
- `课次标题.txt` — 纯文本转录
- `课次标题.srt` — 带时间戳的字幕文件
