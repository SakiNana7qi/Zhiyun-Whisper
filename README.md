# Zhiyun-Whisper

浙江大学智云课堂语音转录工具 — 自动下载课程录播并使用 Qwen3-ASR 或 Whisper 进行语音转文字，支持实时直播监控与关键词提醒。

## 功能

- 获取智云课堂旧版和新版（interactivemeta）课程录播视频
- 默认使用本地 **Qwen3-ASR-1.7B**，支持切换 0.6B、faster-whisper 和 OpenAI Whisper API
- 输出 `.txt`（纯文本）和 `.srt`（带时间戳字幕）格式
- 大文件下载自动断点续传
- **实时直播监控** — 支持旧版和新版 `ilive` 课堂，检测「小测、点到、考勤」等关键词，通过钉钉机器人推送提醒
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
3. **CUDA**（可选）— 本地 Qwen3-ASR / Whisper 使用 GPU 加速，CPU 也可运行但较慢

## 安装

```bash
pip install -r requirements.txt
```

升级已有安装时也需执行此命令，以安装新增的 `qwen-asr` 依赖。建议使用独立的 Python 3.11/3.12 环境。
Qwen3-ASR 使用官方 `qwen-asr` 包的 Transformers 后端，无需安装 vLLM 或 FlashAttention。

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

# 可选备用 LLM：三项全部留空则关闭，启用时三项均需填写
LLM_FALLBACK_API_BASE=""
LLM_FALLBACK_API_KEY=""
LLM_FALLBACK_MODEL=""

# 直播监控 - 多课程优先级（逗号分隔的 course_id，靠前的优先）
MONITOR_PRIORITY="81771,83640"   # 多课程同时直播时自动选择优先级最高的

# 直播监控 - 排除不想监控的课程
BAN_COURSE_ID=""                 # 逗号分隔的 course_id，自动选择时会跳过这些课程
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

首次运行本地模式时需要从 HuggingFace 下载所选的 Qwen3-ASR 或 Whisper 模型。
无法直连时，可设置环境变量使用镜像：

**Windows (PowerShell):**
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
# 然后在同一终端运行 python main.py ...
```

**永久生效（推荐）：** 在系统环境变量中添加 `HF_ENDPOINT`，值为 `https://hf-mirror.com`。

**Linux / WSL (Bash):**
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

## 使用

### 录播转录

```bash
# 转录指定课次（默认使用本地 Qwen3-ASR-1.7B）
python main.py transcribe "https://classroom.zju.edu.cn/livingroom?course_id=81771&sub_id=1892675&tenant_code=112"

# 转录新版智云课次（直接使用浏览器中的回放链接）
python main.py transcribe "https://interactivemeta.cmc.zju.edu.cn/#/replay?course_id=86830&sub_id=1967175&tenant_code=112"

# 使用 OpenAI API 转录
python main.py transcribe "URL" --mode api

# 使用较小的 Qwen3-ASR-0.6B
python main.py transcribe "URL" --model 0.6b

# 切回本地 Whisper（tiny/base/small/medium/large-v3 等名称保持兼容）
python main.py transcribe "URL" --mode local --model medium

# 调整批处理大小（默认 Qwen 为 1、Whisper 为 16，增大需要更多显存）
python main.py transcribe "URL" --batch-size 2

# 列出某门课所有课次
python main.py list --course-id 81771
```

旧版和新版回放共用转录流程，均支持 `--mode`、`--model` 和 `--batch-size`。
新版回放地址为数组时，优先使用 `playback.selected` 指定的视频；未指定或不可用时使用第一个有效地址。

Qwen 模型参数支持 `1.7b` / `0.6b`、`qwen3-asr-1.7b` / `qwen3-asr-0.6b`，
也支持完整名称 `Qwen/Qwen3-ASR-1.7B` / `Qwen/Qwen3-ASR-0.6B`，大小写不敏感。
`--mode api` 仍使用 OpenAI Whisper API，`--model` 仅控制本地模式。

Qwen 录播转录会额外加载 `Qwen/Qwen3-ForcedAligner-0.6B` 来生成字幕时间戳，首次使用时也需下载该模型。
长音频由官方 SDK 自动切分并合并时间戳；字词时间戳会合并为字幕片段，保留原转录的标点。
ASR 支持 30 种语言；时间戳对齐支持中文、英文、粤语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语，参见[官方说明](https://github.com/QwenLM/Qwen3-ASR)。

### 直播监控

实时监控智云直播，检测关键词（拼音模糊匹配 + LLM 语义确认），通过钉钉推送提醒。

旧版使用 `get-sub-info` 返回的 HLS 地址；新版 `ilive` 自动从 `getscreenstream` 获取教师流的带签名 HLS 地址，复用同一套音频切片、转录和提醒流程，无需额外配置或 WebRTC 依赖。

每次启动 monitor 时，会在查询课表和加载语音模型前，分别检查主 LLM 和已配置的备用 LLM。检查使用固定测试文本实际调用 Chat Completions，并复用正式监控的参数、重试、思考正文提取和 JSON/证据校验，显示模型名、`OK` / `FAILED` 和耗时；未配置备用时显示 `SKIPPED`。检查不发送钉钉消息。即使检查失败也会继续监控，后续仍按主服务、备用服务、疑似提醒的顺序处理，不会永久停用失败的服务。

```bash
# 不指定课程 ID — 自动从课表检测正在直播的课，没有则持续轮询等待
python main.py monitor --debug

# 使用 Qwen3-ASR-0.6B
python main.py monitor --model 0.6b --debug

# 使用原来的 Whisper small
python main.py monitor --model small --debug

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
| `--model` | `qwen3-asr-1.7b` | 支持 Qwen 1.7b/0.6b 或 Whisper tiny/base/small/medium/large-v3 等 |
| `--batch-size` | Qwen: `1` / Whisper: `16` | 本地推理批处理大小，增大需要更多显存 |
| `--chunk-duration` | `30` | 每段音频长度（秒） |
| `--poll-interval` | `15` | 无直播时轮询间隔（秒） |
| `--chunks-dir` | `chunks` | 临时音频切片目录（处理后删除） |
| `--log-dir` | `logs` | 转录日志目录（永久保留） |
| `--debug` | 关 | 打印每段转录文本和 LLM 响应 |

**多课程自动选择：**

未指定 `--course-id` 且同时有多个直播时，按 `MONITOR_PRIORITY` 自动选择：
- 优先选择在优先列表中的课程
- 列表中靠前的优先级更高
- 若都不在列表中，选择第一个
- 未设置 `MONITOR_PRIORITY` 时，多课程会提示手动指定

**工作流程：**
1. 若未指定 `--course-id`，从课表 API 查找 `status='1'` 的直播课，没有则每隔 `--poll-interval` 秒重试
2. 找到直播后，按旧版或新版 `ilive` 接口获取 HLS 流 URL（m3u8），新版选择教师流，用 ffmpeg 切成 30 秒 WAV 片段
3. 使用所选模型（默认 Qwen3-ASR-1.7B）转录每个片段，模型只加载一次；直播不加载时间戳对齐模型，全文追加写入 `logs/{course_id}_{date}.txt`
4. 拼音模糊匹配关键词（容忍口音识别错误，如「小策」≈「小测」）
5. 命中后（包括逐字命中）用一次 LLM 调用返回结构化判断、确认关键词、原文证据和解释；结合最近 3 段语境，以最新片段为判断对象
6. 确认后通过钉钉 Webhook 推送告警（含课程名称、时间、确认关键词、证据、分析、最近转录原文），发送成功后 120 秒冷却；明确否定时不推送。主 LLM 调用或结果校验失败时尝试已配置的备用 LLM，所有已配置服务均失败后发送标明「疑似命中（语义确认失败）」的提醒
7. 网络中断或 auth_key 过期时自动重连；直播结束（status 变化）时退出

语义判断仍按「提及相关事项就提醒」，包括预告、回顾或否定该事项，不要求正在执行。提示词要求区分「分享到／来到」与考勤「点到」、测试与随堂小测。确认关键词必须来自配置列表，证据必须能在最新转录中找到；标题使用 LLM 确认的关键词。模型的默认思考模式保持不变，仅解析最终正文中的 JSON 判断。

**备用 LLM：** 在 `.env` 填写 `LLM_FALLBACK_API_BASE`、`LLM_FALLBACK_API_KEY`、`LLM_FALLBACK_MODEL` 后启用；三项全空时保持原有行为，只填部分会在启动时报错。主服务遇到连接错误、超时、限流或服务端错误时，由 SDK 最多重试两次（含首次请求最多三次），耗尽后尝试备用服务；备用服务使用同样的重试策略。鉴权失败等不可重试错误，以及正文解析或结果校验失败，会直接进入备用流程。主服务明确返回否定时不调用备用；备用的有效肯定或否定正常生效。每个新片段仍先尝试主服务，备用服务同样使用最近转录、思考正文清理和证据校验。

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

  证据：好现在开始小测大家把书收起来

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
- `qwen-asr` — 本地 Qwen3-ASR 0.6B/1.7B 推理及录播时间戳对齐（Transformers 后端）
- `requests` — HTTP 请求
- `click` — CLI 框架

## 注意事项

- **直播监控建议使用 GPU** — CPU 推理可能导致音频切片积压；显存不足可使用 `--model 0.6b`，或切回 `--model small` 使用 Whisper
- **Token 自动刷新** — 设置 `ZJU_USERNAME`/`ZJU_PASSWORD` 后，Token 过期时 monitor 自动重新登录（最多重试 3 次）；若未设置账号密码，过期后进程退出
- **钉钉加签** — Webhook 必须启用「加签」安全设置，`DINGTALK_SECRET` 为签名密钥（以 `SEC` 开头）
- **LLM 调用次数** — 每次启动会为每个已配置服务各发起一次检查调用；冷却期外，每个命中的片段正常用一次调用完成确认和语境分析，主服务失败后才调用已配置的备用服务。各调用最多重试两次。明确否定不会进入冷却，已成功发送的确认告警和疑似提醒均进入 120 秒冷却
- **新版直播取流** — 即使 `get-sub-info` 显示 `is_m3u8=no`，仍会查询新版播放接口的教师流 `stream_m3u8`。若尚未提供教师 HLS 地址则继续等待；目前不直接接收仅有 WebRTC 的流，也不会使用 PPT 流替代教师音频。重连时会重新获取签名地址
