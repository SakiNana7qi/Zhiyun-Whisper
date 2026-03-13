# Zhiyun-Whisper

浙江大学智云课堂语音转录工具 — 自动下载课程录播并使用 Whisper 进行语音转文字，支持实时直播监控与关键词提醒。

## 功能

- 获取智云课堂课程录播视频
- 支持本地 faster-whisper 和 OpenAI Whisper API 两种转录模式
- 输出 `.txt`（纯文本）和 `.srt`（带时间戳字幕）格式
- 大文件下载自动断点续传
- **实时直播监控** — 检测「小测、点到、考勤」等关键词，通过钉钉机器人推送提醒
- **自动检测当前直播** — 无需手动指定课程 ID，从课表自动发现正在直播的课程
- **持久转录日志** — 每次直播的全文转录按课程+日期写入 `logs/` 目录永久保存
- **Token 自动刷新** — 设置 `ZJU_USERNAME`/`ZJU_PASSWORD` 后，监控进程检测到 Token 过期时自动重新登录，无需人工干预

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
ZJU_PASSWORD="你的密码"   # 设置后 Token 过期时自动刷新，无需手动更新
ZJU_TOKEN=""   # 可选：手动填入则优先使用；为空时由账号密码自动登录获取

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

**方式一：自动获取（推荐）**

在 `.env` 中设置 `ZJU_USERNAME` 和 `ZJU_PASSWORD`，**无需填写 `ZJU_TOKEN`**。
程序启动时自动登录 ZJU 统一身份认证，获取并维护 Token，过期后自动刷新。

**方式二：手动获取**

1. 浏览器打开智云课堂任意页面并登录
2. 打开 DevTools (F12) → Network 标签
3. 刷新页面，找到任意 XHR 请求
4. 查看 Request Headers → `Authorization: Bearer <token>`
5. 复制 `Bearer` 后面的 JWT 字符串到 `.env` 的 `ZJU_TOKEN`

手动填入的 Token 有效期约 24 小时；若同时设置了账号密码，过期时仍会自动刷新。

> 技术细节见 [docs/zju-cas-auth.md](docs/zju-cas-auth.md)。

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
# 不指定课程 ID — 自动从课表检测正在直播的课，没有则持续轮询等待
python main.py monitor --debug

# 指定课程 ID
python main.py monitor --course-id 81706

# 自定义关键词
python main.py monitor --keywords "小测,随堂测验,点名"

# 调试模式（打印每个 30s 音频块的转录文本和 LLM 分析）
python main.py monitor --debug

# 指定日志目录和临时切片目录
python main.py monitor --log-dir logs --chunks-dir chunks
```

**选项说明：**

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--course-id` | 自动检测 | 课程 ID，省略时从课表自动发现直播 |
| `--keywords` | `小测,点到,考勤,点名,学在浙大,quiz,雷达` | 逗号分隔的关键词 |
| `--model` | `small` | Whisper 模型大小 |
| `--chunk-duration` | `30` | 每段音频长度（秒） |
| `--poll-interval` | `15` | 无直播时轮询间隔（秒） |
| `--chunks-dir` | `chunks` | 临时音频切片目录（处理后删除） |
| `--log-dir` | `logs` | 转录日志目录（永久保留） |
| `--debug` | 关 | 打印每段转录文本和 LLM 响应 |

**工作流程：**
1. 若未指定 `--course-id`，从课表 API 查找 `status='1'` 的直播课，没有则每隔 `--poll-interval` 秒重试
2. 找到直播后，获取 HLS 流 URL（m3u8），用 ffmpeg 切成 30 秒 WAV 片段
3. 用 faster-whisper 转录每个片段，全文追加写入 `logs/{course_id}_{date}.txt`
4. 拼音模糊匹配关键词（容忍口音识别错误，如「小策」≈「小测」）
5. 命中后调用 LLM 二次确认（是/否），通过后再调用 LLM 分析最近 3 段转录的语境
6. 通过钉钉 Webhook 推送告警（含课程名称、时间、LLM 分析、最近转录原文），120 秒冷却
7. 网络中断或 auth_key 过期时自动重连；直播结束（status 变化）时退出

## 输出

### 录播转录

转录结果保存在 `output/` 目录下，每个课次生成：
- `课次标题.mp4` — 下载的视频文件
- `课次标题.wav` — 提取的音频文件
- `课次标题.txt` — 纯文本转录
- `课次标题.srt` — 带时间戳的字幕文件

### 直播监控

- 音频切片临时保存在 `chunks/` 目录（处理后自动删除），ffmpeg 日志保存在 `chunks/ffmpeg.log`
- **转录日志** 永久保存在 `logs/{course_id}_{日期}.txt`，每行格式：
  ```
  [14:22:45] 同学们今天我们讲量词...
  [14:23:15] 好现在开始小测大家把书收起来...
  ```
- 检测到关键词并经 LLM 确认后，钉钉群收到消息：
  ```
  [智云直播监控] 触发关键词：小测
  课程：离散数学理论基础（82312）
  时间：14:23:15

  分析：老师正在宣布进行随堂小测，要求同学收起书本准备答题。

  最近转录：
  [14:22:15] 那么今天我们来做一个练习...
  [14:22:45] 同学们今天我们讲量词...
  [14:23:15] 好现在开始小测大家把书收起来...
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
- **Token 自动刷新** — 设置 `ZJU_USERNAME`/`ZJU_PASSWORD` 后，Token 过期时 monitor 自动重新登录（最多重试 3 次）；若未设置账号密码，过期后进程退出
- **钉钉加签** — Webhook 必须启用「加签」安全设置，`DINGTALK_SECRET` 为签名密钥（以 `SEC` 开头）
- **LLM 调用次数** — 每次关键词命中调用一次确认（is/否），确认后再调用一次语境分析；使用 gpt-4o-mini 成本极低
- **仅支持东区教学楼** — 北区教学楼（`ilive` 类型，如紫金港北楼）使用 WebRTC 互动直播系统，暂不支持
