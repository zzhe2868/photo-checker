"""
Ollama Qwen2.5-VL 视觉检测模块
- 支持 Qwen2.5-VL-3B / 7B / 72B 模型自动下载
- 图片分析（质量/建议/场景判断）
- 自动解压 gzip 响应体
"""

import os, io, json, base64, urllib.request, time, logging, subprocess, sys
import numpy as np

VERSION = "5.0"

logger = logging.getLogger('photo_checker')

# ═══════════════════════════════════════════
# 模型定义
# ═══════════════════════════════════════════

QWEN_MODELS = {
    'qwen2.5-vl:3b-instruct-q4_K_M': {
        'name': 'Qwen2.5-VL 3B',
        'size': '2.3 GB',
        '显存需求': '4 GB',
        '速度': '~3s/张',
        '精度': '良好（推荐）',
        'stars': 5,
    },
    'qwen2.5-vl:7b-instruct-q4_K_M': {
        'name': 'Qwen2.5-VL 7B',
        'size': '4.9 GB',
        '显存需求': '8 GB',
        '速度': '~8s/张',
        '精度': '优秀',
        'stars': 5,
    },
    'qwen2.5-vl:72b-instruct-q4_K_M': {
        'name': 'Qwen2.5-VL 72B',
        'size': '43 GB',
        '显存需求': '48 GB',
        '速度': '~30s/张',
        '精度': '极好',
        'stars': 5,
    },
}

ANALYSIS_PROMPT = """你是一位专业的照片质量评估专家。请分析这张照片，从以下维度给出详细评价和建议：

1. **整体评分**（0-100）
2. **清晰度**：是否模糊/失焦/运动模糊
3. **曝光**：是否过曝/欠曝
4. **构图**：主体是否突出，三分法运用
5. **色彩**：色彩搭配是否和谐
6. **人像质量**（如有人物）：表情、姿态、是否闭眼
7. **综合建议**：2-3条改进建议
8. **是否建议移除**：true/false（质量过低应移除）

请按以下JSON格式严格输出（不要输出任何其他文字）：
{
  "overall_score": 75,
  "sharpness": "良好/一般/差",
  "exposure": "良好/一般/差",
  "composition": "良好/一般/差",
  "color": "良好/一般/差",
  "portrait_quality": "良好/一般/差/无人物",
  "suggestions": ["建议1", "建议2"],
  "should_remove": false
}"""

# ═══════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════

def _base64_encode_image(path):
    """将图片转为 base64 字符串"""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _check_ollama_running(host, timeout=3):
    """检查 Ollama 服务是否正在运行"""
    try:
        req = urllib.request.Request(f"{host}/api/tags",
headers={'User-Agent': f'photo-checker/{VERSION}'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
            return True, data.get('models', [])
    except Exception as e:
        return False, []


def _pull_model(host, model_name, cb=None, max_retries=3):
    """拉取 Ollama 模型，支持指数退避重试"""
    try:
        if cb: cb(f"正在下载 {model_name} (约 {QWEN_MODELS[model_name]['size']})...")
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(
                    f"{host}/api/pull",
                    data=json.dumps({"name": model_name}).encode(),
                    headers={'Content-Type': 'application/json', 'User-Agent': f'photo-checker/{VERSION}'},
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=7200) as r:
                    for line in r:
                        chunk = json.loads(line)
                        if chunk.get('status'):
                            if cb: cb(f"下载中: {chunk['status']}")
                    if cb: cb("下载完成")
                return True
            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt  # 指数退避: 2s, 4s, 8s...
                    if cb: cb(f"下载中断，{wait}s 后重试 ({attempt}/{max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        return True
    except Exception as e:
        logger.error(f"模型下载失败 {model_name}: {e}")
        if cb: cb(f"下载失败: {e}")
        return False


def _has_model(host, model_name):
    """检查模型是否已拉取"""
    try:
        req = urllib.request.Request(f"{host}/api/tags",
headers={'User-Agent': f'photo-checker/{VERSION}'})
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read())
        for m in data.get('models', []):
            if m.get('name') == model_name:
                return True
        return False
    except:
        return False


# ═══════════════════════════════════════════
# 核心：流式请求 + gzip 解压
# ═══════════════════════════════════════════

def _ollama_vision(host, model, image_path, prompt, cb=None, api_timeout=None):
    """
    通过 Ollama /api/chat 发送图片进行视觉分析。

    关键修复：Ollama 默认发送 gzip 压缩的响应体。
    使用 urllib 时若不设置 Accept-Encoding: identity（不压缩）
    或正确处理解压，会导致 'error decoding response body'。

    策略：设置 Accept-Encoding: identity 让服务器不压缩响应。
    """
    b64 = _base64_encode_image(image_path)

    messages = [
        {
            'role': 'user',
            'content': prompt,
            'images': [b64]
        }
    ]

    payload = json.dumps({
        'model': model,
        'messages': messages,
        'stream': False  # 非流式，Ollama 不会压缩响应体
    }).encode()

    try:
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': f'photo-checker/{VERSION}',
                'Accept': 'application/x-ndjson',
            },
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=api_timeout or 120) as r:
            raw = r.read()
            # 尝试直接解析
            try:
                response = json.loads(raw)
            except json.JSONDecodeError:
                # 可能是 gzip 压缩的
                import gzip
                try:
                    raw = gzip.decompress(raw)
                    response = json.loads(raw)
                except Exception:
                    logger.error(f"Ollama 响应解析失败 (非流式): {raw[:200]}")
                    return None

            if 'message' in response:
                text = response['message'].get('content', '')
                return _parse_vision_response(text)
            return None

    except Exception as e:
        logger.error(f"Ollama API 调用失败: {e}")
        if cb: cb(f"API 错误: {e}")
        return None


def _parse_vision_response(text):
    """从 LLM 文本响应中提取 JSON"""
    text = text.strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试从 Markdown 代码块中提取
    try:
        import re
        match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except:
        pass
    # 尝试从任意位置找 JSON 对象
    try:
        import re
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
    except:
        pass
    return None


class QwenVL:
    """Ollama Qwen2.5-VL 本地视觉检测"""

    def __init__(self, host=None, model=None, progress_cb=None):
        self.host = host or ''
        self.model = model
        self.progress_cb = progress_cb
        self.available = False
        self._models_list = []

        if self.model:
            self._init()

    def _init(self):
        """检查 Ollama 并初始化"""
        if not self.model:
            return
        ok, models = _check_ollama_running(self.host)
        if not ok:
            if self.progress_cb: self.progress_cb("Ollama 服务未运行，请先启动 Ollama")
            return
        self._models_list = models
        if self.progress_cb: self.progress_cb("Ollama 服务就绪")

        if not _has_model(self.host, self.model):
            if self.progress_cb: self.progress_cb(f"首次使用，正在下载 {self.model}...")
            ok = _pull_model(self.host, self.model, self.progress_cb)
            if not ok: return

        self.available = True
        if self.progress_cb: self.progress_cb("Qwen2.5-VL 就绪")

    def analyze(self, image_path, prompt=None, api_timeout=None):
        """分析单张图片"""
        if not self.available or not self.model:
            return None
        return _ollama_vision(self.host, self.model, image_path, prompt or ANALYSIS_PROMPT, self.progress_cb, api_timeout)

    def analyze_batch(self, image_paths, batch_size=1, progress_cb=None):
        """批量分析图片，支持并发批次"""
        results = []
        total = len(image_paths)
        for i in range(0, total, batch_size):
            batch = image_paths[i:i+batch_size]
            for j, path in enumerate(batch):
                idx = i + j
                if progress_cb:
                    progress_cb(f"AI 分析: {idx+1}/{total} {os.path.basename(path)}")
                r = self.analyze(path)
                results.append(r)
            if i + batch_size < total:
                time.sleep(0.5)  # 批次间间隔，避免 Ollama 过载
        return results