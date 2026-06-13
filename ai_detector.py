"""
AI 检测模块 — ONNX Runtime 驱动，完全离线
- UltraLight 人脸检测（1.3MB ONNX）
- 人脸质量分析（眼睛/角度/光照）
- NIMA 风格美学评分（启发式）
- 模糊类型分类（失焦/运动模糊/背景虚化）
"""

import os
import hashlib
import urllib.request
import numpy as np
import cv2
import json

# ── ONNX Runtime ──
try:
    import onnxruntime as ort
    # 抑制 ONNX 初始化警告
    ort.set_default_logger_severity(3)  # ERROR only
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# ── 模型下载路径 ──
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
ULTRALIGHT_URL = (
    "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/"
    "raw/master/models/onnx/version-RFB-640.onnx"
)
ULTRALIGHT_MODEL = os.path.join(MODEL_DIR, 'ultraface_rfb_640.onnx')


# ═══════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════

def ensure_model_downloaded(url, local_path, progress_callback=None):
    """下载模型文件（如不存在）"""
    if os.path.exists(local_path):
        return True

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        if progress_callback:
            progress_callback(f"下载模型: {os.path.basename(local_path)}...")

        req = urllib.request.Request(url, headers={'User-Agent': 'photo-checker/3.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(local_path, 'wb') as f:
            f.write(data)

        if progress_callback:
            progress_callback("模型下载完成 ✓")
        return True
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        if progress_callback:
            progress_callback(f"模型下载失败: {e}")
        return False


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _nms(boxes, scores, iou_threshold=0.4):
    """非极大值抑制"""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return keep


# ═══════════════════════════════════════════
# UltraLight 人脸检测
# ═══════════════════════════════════════════

class UltraLightFace:
    """UltraLight ONNX 人脸检测器"""

    INPUT_SIZE = (640, 640)       # 模型输入尺寸
    MEAN = np.array([127, 127, 127], dtype=np.float32)
    STD  = np.array([128, 128, 128], dtype=np.float32)
    CONF_THRESHOLD = 0.6
    IOU_THRESHOLD  = 0.4

    # 锚点生成参数
    _ANCHORS = None

    def __init__(self, model_path=None):
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime 未安装，请 pip install onnxruntime")

        if model_path is None:
            model_path = ULTRALIGHT_MODEL

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.session = ort.InferenceSession(model_path,
            providers=['CPUExecutionProvider'])
        self._input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _get_anchors():
        """生成UltraLight特征金字塔锚点"""
        if UltraLightFace._ANCHORS is not None:
            return UltraLightFace._ANCHORS

        anchors = []
        min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        strides  = [8, 16, 32, 64]

        for step_idx, step in enumerate(strides):
            fmap_size = 640 // step
            for y in range(fmap_size):
                for x in range(fmap_size):
                    cx = (x + 0.5) * step / 640.0
                    cy = (y + 0.5) * step / 640.0
                    for ratio in min_boxes[step_idx]:
                        s = ratio / 640.0
                        anchors.append([cx, cy, s, s])
        UltraLightFace._ANCHORS = np.array(anchors, dtype=np.float32)
        return UltraLightFace._ANCHORS

    def detect(self, img_rgb):
        """
        返回: [(x1, y1, x2, y2, confidence), ...]
        """
        h, w = img_rgb.shape[:2]

        # 预处理
        resized = cv2.resize(img_rgb, self.INPUT_SIZE)
        input_tensor = ((resized.astype(np.float32) - self.MEAN) / self.STD)
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, ...]

        # 推理
        outputs = self.session.run(None, {self._input_name: input_tensor})
        scores_raw = outputs[0]  # (1, N, 2)
        boxes_raw  = outputs[1]  # (1, N, 4)

        scores = _softmax(scores_raw[0])[:, 1]  # 正类confidence
        boxes = boxes_raw[0]

        # 过滤低置信度
        mask = scores > self.CONF_THRESHOLD
        scores = scores[mask]
        boxes = boxes[mask]

        if len(boxes) == 0:
            return []

        # 解码：锚点中心偏移 → 绝对坐标
        anchors = self._get_anchors()
        if len(anchors) > 0:
            valid_anchors = anchors[mask]
            cx = boxes[:, 0] * 0.1 * valid_anchors[:, 2] + valid_anchors[:, 0]
            cy = boxes[:, 1] * 0.1 * valid_anchors[:, 3] + valid_anchors[:, 1]
            bw = valid_anchors[:, 2] * np.exp(boxes[:, 2] * 0.2)
            bh = valid_anchors[:, 3] * np.exp(boxes[:, 3] * 0.2)
        else:
            cx, cy = boxes[:, 0], boxes[:, 1]
            bw, bh = boxes[:, 2], boxes[:, 3]

        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h

        # 裁剪到图像范围
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)

        dets = np.stack([x1, y1, x2, y2, scores], axis=1)

        # NMS
        keep = _nms(dets[:, :4], dets[:, 4], self.IOU_THRESHOLD)
        return [tuple(dets[i]) for i in keep]


# ═══════════════════════════════════════════
# 人脸质量分析
# ═══════════════════════════════════════════

class FaceAnalyzer:
    """人脸质量分析：眼睛张开度、头部角度、光照均匀度"""

    def __init__(self):
        cascade = cv2.data.haarcascades
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade, 'haarcascade_eye.xml'))

    def analyze(self, gray, face_box):
        """
        face_box: (x1, y1, x2, y2)
        返回: {eye_open, head_angle, lighting_score, overall}
        """
        x1, y1, x2, y2 = [int(v) for v in face_box[:4]]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(gray.shape[1], x2); y2 = min(gray.shape[0], y2)
        face_roi = gray[y1:y2, x1:x2]
        fh, fw = face_roi.shape

        result = {
            'eye_open': 100.0,
            'head_angle': 0.0,
            'lighting_score': 100.0,
            'overall': 100.0
        }

        if fh < 20 or fw < 20:
            return result

        # 1. 眼睛张开度（Haar在人脸上半部检测眼睛）
        upper = face_roi[:fh // 2, :]
        eyes = self.eye_cascade.detectMultiScale(upper, 1.05, 4, minSize=(10, 10))
        eye_count = min(len(eyes), 2)
        result['eye_open'] = (eye_count / 2) * 100

        # 2. 头部角度估计（人脸框宽高比 + 偏移）
        aspect = fw / max(fh, 1)
        # 正面人脸宽高比约 0.7-0.85，偏离越大角度越大
        ideal_aspect = 0.78
        angle_est = min(abs(aspect - ideal_aspect) / 0.3 * 45, 90)
        result['head_angle'] = round(angle_est, 1)

        # 3. 人脸光照均匀度
        hist = cv2.calcHist([face_roi], [0], None, [64], [0, 256])
        hist_norm = hist / hist.sum()
        # 低熵 = 光照不均匀
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
        max_entropy = np.log2(64)
        result['lighting_score'] = round((entropy / max_entropy) * 100, 1)

        # 综合分
        result['overall'] = round(
            result['eye_open'] * 0.4 +
            (100 - min(result['head_angle'] / 90 * 100, 100)) * 0.3 +
            result['lighting_score'] * 0.3, 1)

        return result


# ═══════════════════════════════════════════
# 美学评分（启发式 NIMA 风格）
# ═══════════════════════════════════════════

class AestheticScorer:
    """美学评分器：规则三分法 + 色彩和谐 + 曝光 + 清晰度 → 0-10分"""

    def score(self, img_rgb):
        h, w = img_rgb.shape[:2]

        # 1. 三分法构图分
        comp_score = self._rule_of_thirds(img_rgb)

        # 2. 色彩和谐分
        color_score = self._color_harmony(img_rgb)

        # 3. 曝光分
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        exp_score = self._exposure_quality(gray)

        # 4. 清晰度分
        sharp_score = self._sharpness(gray)

        # 加权合成（NIMA 风）
        aesthetic = (
            comp_score * 0.25 +
            color_score * 0.25 +
            exp_score * 0.25 +
            sharp_score * 0.25
        )
        return round(aesthetic, 1)

    def _rule_of_thirds(self, img):
        """三分法构图评估"""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel 边缘检测
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges = np.abs(edges)

        # 三分线位置
        v1, v2 = w // 3, 2 * w // 3
        h1, h2 = h // 3, 2 * h // 3

        # 三分线上的边缘强度 vs 整体
        total = np.sum(edges) + 1e-8
        on_lines = (
            np.sum(edges[:, v1-3:v1+3]) + np.sum(edges[:, v2-3:v2+3]) +
            np.sum(edges[h1-3:h1+3, :]) + np.sum(edges[h2-3:h2+3, :])
        )
        ratio = on_lines / total
        return min(ratio * 15, 10)  # 映射到0-10

    def _color_harmony(self, img):
        """色彩和谐评估（HSV色相分布）"""
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0].flatten()
        sat = hsv[:, :, 1].flatten()

        # 饱和度高且色相集中的图得分更高
        avg_sat = np.mean(sat) / 255 * 10
        hue_hist = np.histogram(hue, bins=36, range=(0, 180))[0]
        hue_entropy = -np.sum((hue_hist / max(hue_hist.sum(), 1)) *
                               np.log2(hue_hist / max(hue_hist.sum(), 1) + 1e-8))
        hue_score = min(hue_entropy / 5 * 10, 10)

        return (avg_sat + hue_score) / 2

    def _exposure_quality(self, gray):
        """曝光质量（直方图均衡度）"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cdf = hist.cumsum()
        cdf_norm = cdf / cdf[-1]

        # 理想曝光：中间调像素多，高光和阴影像素少但存在
        shadows = cdf_norm[30]
        midtones = cdf_norm[200] - cdf_norm[50]
        highlights = 1 - cdf_norm[220]

        score = 10 - abs(shadows - 0.1) * 10 - abs(midtones - 0.65) * 10 - abs(highlights - 0.1) * 10
        return max(0, min(10, score))

    def _sharpness(self, gray):
        """清晰度评估"""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = lap.var()
        # 方差映射到0-10（~200 = good）
        return min(var / 20, 10)


# ═══════════════════════════════════════════
# 模糊类型分类
# ═══════════════════════════════════════════

class BlurClassifier:
    """二级模糊检测：区分失焦模糊 / 运动模糊 / 背景虚化"""

    def classify(self, gray):
        """
        返回: {type: 'defocus'|'motion'|'bokeh'|'sharp', confidence: 0-100}
        """
        h, w = gray.shape

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = lap.var()

        # 1. 全局清晰度判断
        if lap_var > 200:
            return {'type': 'sharp', 'confidence': 90}

        # 2. 检测运动模糊（频域分析）
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)

        # 运动模糊特征：频谱在某方向拉长（椭圆度大）
        _, thresh = cv2.threshold(
            (magnitude * 255 / magnitude.max()).astype(np.uint8), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 5:
                (cx, cy), (ma, MA), angle = cv2.fitEllipse(cnt)
                elongation = max(ma, MA) / min(ma, MA) if min(ma, MA) > 0 else 1
                if elongation > 3.0:
                    return {'type': 'motion_blur', 'confidence': min(round((elongation - 2) * 20), 100)}

        # 3. 检测背景虚化（主体清晰+背景模糊）
        if lap_var < 50:
            return {'type': 'defocus', 'confidence': round(max(0, (100 - lap_var / 50 * 100)))}

        # 尝试区分局部模糊 vs 全局模糊
        # 分块检测
        block_size = max(h, w) // 3
        variances = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                if block.size > 100:
                    variances.append(cv2.Laplacian(block, cv2.CV_64F).var())

        if not variances:
            return {'type': 'defocus', 'confidence': 50}

        var_std = np.std(variances)
        var_mean = np.mean(variances)

        # 各块方差差异大 → 背景虚化（部分区域清晰部分模糊）
        if var_std > var_mean * 0.5 and np.max(variances) > 200:
            return {'type': 'bokeh', 'confidence': min(round(var_std / var_mean * 40), 100)}

        # 全局模糊
        return {'type': 'defocus', 'confidence': min(round(100 - lap_var / 3), 100)}


# ═══════════════════════════════════════════
# 统一 AI 检测接口
# ═══════════════════════════════════════════

class AIDetector:
    """AI检测总控制器"""

    def __init__(self, enable_ai=True, progress_callback=None):
        self.enable_ai = enable_ai and HAS_ONNX
        self.face_detector = None
        self.face_analyzer = FaceAnalyzer()
        self.aesthetic_scorer = AestheticScorer()
        self.blur_classifier = BlurClassifier()
        self.progress_cb = progress_callback

        if self.enable_ai:
            self._init_models()

    def _init_models(self):
        """初始化/下载 AI 模型"""
        if not HAS_ONNX:
            return

        if self.progress_cb:
            self.progress_cb("检查 AI 模型...")

        # UltraLight 人脸检测模型
        ok = ensure_model_downloaded(ULTRALIGHT_URL, ULTRALIGHT_MODEL, self.progress_cb)
        if ok:
            try:
                self.face_detector = UltraLightFace(ULTRALIGHT_MODEL)
                if self.progress_cb:
                    self.progress_cb("AI 模型就绪 ✓")
            except Exception as e:
                if self.progress_cb:
                    self.progress_cb(f"AI 模型加载失败: {e}")
                self.face_detector = None
        else:
            if self.progress_cb:
                self.progress_cb("AI 模型下载失败，使用传统检测")

    def detect_faces(self, img_rgb):
        """AI 人脸检测（自动回退 Haar）"""
        if self.face_detector:
            try:
                return self.face_detector.detect(img_rgb)
            except Exception:
                pass
        return None  # 调用方回退到传统方法

    def analyze_face_quality(self, gray, face_box):
        """分析单张人脸质量"""
        return self.face_analyzer.analyze(gray, face_box)

    def score_aesthetic(self, img_rgb):
        """美学评分 0-10"""
        return self.aesthetic_scorer.score(img_rgb)

    def classify_blur(self, gray):
        """模糊类型分类"""
        return self.blur_classifier.classify(gray)


# ═══════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════

if __name__ == '__main__':
    def progress(msg):
        print(f"[AI] {msg}")

    detector = AIDetector(enable_ai=True, progress_callback=progress)
    print(f"AI 启用: {detector.enable_ai}")
    print(f"人脸检测: {'UltraLight ONNX' if detector.face_detector else 'Haar fallback'}")
    print("AI 模块加载完成 ✓")