# Zhiyun-Whisper

浙江大学智云课堂语音转录工具 — 自动下载课程录播并使用 Whisper 进行语音转文字，支持实时直播监控与关键词提醒。

## 功能

- 获取智云课堂课程录播视频
- 支持本地 faster-whisper 和 OpenAI Whisper API 两种转录模式
- 输出 `.txt`（纯文本）和 `.srt`（带时间戳字幕）格式
- 大文件下载自动断点续传
- **实时直播监控** — 检测「小测、点到、考勤」等关键词，通过钉钉机器人推送提醒

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
ZJU_TOKEN=""   # 直播监控需要，见下方说明

# 仅 API 模式需要
OPENAI_API_KEY="sk-..."

# 直播监控 - 钉钉机器人配置
DINGTALK_WEBHOOK="https://oapi.dingtalk.com/robot/send?access_token=xxx"
DINGTALK_SECRET="SECxxx"
DINGTALK_AT_MOBILE=""   # 留空则不@任何人

# 直播监控 - LLM 语义确认（降低误报）
LLM_API_BASE="https://api.openai.com/v1"
LLM_API_KEY="sk-xxx"
LLM_MODEL="gpt-4o-mini"
```

### 获取 ZJU_TOKEN（直播监控必需）

1. 浏览器打开智云课堂任意直播页面并登录
2. 打开 DevTools (F12) → Network 标签
3. 刷新页面，找到任意 XHR 请求
4. 查看 Request Headers → `Authorization: Bearer <token>`
5. 复制 `Bearer` 后面的 JWT 字符串到 `.env` 的 `ZJU_TOKEN`

Token 有效期约 24 小时，过期后需重新获取。

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

### 录播转录

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

### 直播监控

实时监控智云直播，检测关键词（拼音模糊匹配 + LLM 语义确认），通过钉钉推送提醒。

```bash
# 监控指定课程的直播（默认关键词：小测,点到,考勤,点名）
python main.py monitor --course-id 81706

# 自定义关键词
python main.py monitor --course-id 81706 --keywords "小测,随堂测验,点名"

# 调试模式（打印每个 30s 音频块的转录文本和 LLM 响应）
python main.py monitor --course-id 81706 --debug

# 使用 small 模型（更快，适合实时场景）
python main.py monitor --course-id 81706 --model small
```

**工作流程：**
1. 轮询 catalogue API，等待直播开始（status='1'）
2. 获取 HLS 流 URL（m3u8），用 ffmpeg 切成 30 秒 WAV 片段
3. 用 faster-whisper 转录每个片段
4. 拼音模糊匹配关键词（容忍口音导致的识别错误，如「小策」≈「小测」）
5. 命中后调用 LLM 语义确认（降低误报率）
6. 确认后通过钉钉 Webhook 推送提醒（120 秒冷却时间）
7. 网络中断或 auth_key 过期时自动重连，直到直播结束（status 变化）

## 输出

### 录播转录

转录结果保存在 `output/` 目录下，每个课次生成：
- `课次标题.mp4` — 下载的视频文件
- `课次标题.wav` — 提取的音频文件
- `课次标题.txt` — 纯文本转录
- `课次标题.srt` — 带时间戳的字幕文件

### 直播监控

- 音频切片临时保存在 `chunks/` 目录（处理后自动删除）
- ffmpeg 日志保存在 `chunks/ffmpeg.log`
- 检测到关键词时，钉钉群收到消息：
  ```
  [智云直播监控] 检测到关键词：小测
  课程：81706
  转录片段："老师说今天有小测大家准备一下..."
  时间：14:23:15
  ```

## 依赖说明

- `pypinyin` — 拼音转换，用于模糊匹配
- `rapidfuzz` — 快速字符串相似度计算
- `openai` — LLM 语义确认（兼容任意 OpenAI-compatible API）
- `faster-whisper` — 本地 Whisper 推理（CTranslate2 后端）
- `requests` — HTTP 请求
- `click` — CLI 框架

## 注意事项

- **直播监控需要 GPU** — small 模型在 CPU 上转录 30s 音频约需 15-45s，可能积压；建议使用 CUDA
- **Token 过期** — `ZJU_TOKEN` 约 24 小时过期，过期后 monitor 命令会报 `未传入token` 错误，需重新获取
- **钉钉加签** — Webhook 必须启用「加签」安全设置，`DINGTALK_SECRET` 为签名密钥（以 `SEC` 开头）
- **LLM 费用** — 每次关键词命中会调用一次 LLM API（约 10 tokens），使用 gpt-4o-mini 成本极低
