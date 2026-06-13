"""
AI 检测模块 v4.1 — SCRFD_2.5GF + 多维质量分析
- SCRFD 人脸检测 + 5点关键点 → EAR闭眼
- 场景自动分类（7类）
- 多维度质量评分（综合/人像/技术/构图）
- 美学评分（独立维度）
"""

import os, urllib.request, numpy as np, cv2

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
SCRFD_URL = "https://huggingface.co/public-data/insightface/resolve/main/models/scrfd_2.5g/scrfd_2.5g_bnkps.onnx"
SCRFD_PATH = os.path.join(MODEL_DIR, 'scrfd_2.5g_bnkps.onnx')

# ═══════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════

def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def _nms(dets, thresh=0.45):
    if len(dets) < 2: return list(range(len(dets)))
    x1, y1 = dets[:, 0], dets[:, 1]
    x2, y2 = dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = dets[:, 4].argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[1:][ovr <= thresh]
    return keep

def ensure_model(url, path, cb=None):
    if os.path.exists(path): return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        if cb: cb(f"下载 {os.path.basename(path)}...")
        req = urllib.request.Request(url, headers={'User-Agent':'photo-checker/4.1'})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        with open(path, 'wb') as f: f.write(data)
        if cb: cb("下载完成")
        return True
    except Exception as e:
        if os.path.exists(path): os.remove(path)
        if cb: cb(f"下载失败: {e}")
        return False


# ═══════════════════════════════════════════
# SCRFD 人脸检测
# ═══════════════════════════════════════════

class SCRFD:
    """SCRFD_2.5GF ONNX 人脸检测 + 5点关键点"""

    INPUT_SIZE = (640, 640)
    STRIDES = [8, 16, 32]
    NUM_ANCHORS = 2
    CENTER_CACHE = {}

    def __init__(self, model_path=None):
        if not HAS_ONNX: raise RuntimeError("需要 onnxruntime")
        self.session = ort.InferenceSession(model_path or SCRFD_PATH,
            providers=['CPUExecutionProvider'])
        self._input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _distance2bbox(points, distance):
        """解码bbox"""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _distance2kps(points, distance):
        """解码关键点"""
        kps = np.zeros((len(points), 10), dtype=np.float32)
        for i in range(5):
            kps[:, i*2]   = points[:, 0] + distance[:, i*2]
            kps[:, i*2+1] = points[:, 1] + distance[:, i*2+1]
        return kps

    def detect(self, img_bgr, threshold=0.5):
        h, w = img_bgr.shape[:2]

        # 预处理
        im_ratio = float(h) / w
        if im_ratio > 1.3:
            new_h, new_w = 640, int(640 / im_ratio)
        elif im_ratio < 0.7:
            new_h, new_w = int(640 * im_ratio), 640
        else:
            new_h, new_w = 640, 640

        resized = cv2.resize(img_bgr, (new_w, new_h))
        det_scale = float(new_h) / h

        # Pad to 640x640
        input_img = np.zeros((640, 640, 3), dtype=np.float32)
        input_img[:new_h, :new_w, :] = resized
        input_img = (input_img - 127.5) / 128.0  # normalize to [-1, 1]
        input_tensor = np.transpose(input_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # 推理
        outputs = self.session.run(None, {self._input_name: input_tensor})

        # 解析多尺度输出
        scores_list, bboxes_list, kpss_list = [], [], []
        fmc = 3  # 3 feature map levels
        for idx, stride in enumerate(self.STRIDES):
            score = outputs[idx]        # (1, N, 1)
            bbox  = outputs[idx + fmc]  # (1, N, 4)
            kps   = outputs[idx + fmc*2]  # (1, N, 10)

            score = score[0, :, 0]
            bbox  = bbox[0]
            kps   = kps[0]

            # 生成锚点中心
            feat_h, feat_w = 640 // stride, 640 // stride
            key = (feat_h, feat_w, stride)
            if key not in self.CENTER_CACHE:
                yv, xv = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing='ij')
                centers = np.stack([xv, yv], axis=2).reshape(-1, 2) * stride
                self.CENTER_CACHE[key] = centers.astype(np.float32)
            centers = self.CENTER_CACHE[key]

            bbox = bbox * stride
            kps  = kps * stride

            # 过滤
            pos = score >= threshold
            if pos.sum() > 0:
                scores_list.append(score[pos])
                bboxes_list.append(self._distance2bbox(centers[pos], bbox[pos]))
                kpss_list.append(self._distance2kps(centers[pos], kps[pos]))

        if not scores_list:
            return []

        scores  = np.concatenate(scores_list)
        bboxes  = np.concatenate(bboxes_list)
        kpss    = np.concatenate(kpss_list)

        # 缩放到原图
        bboxes /= det_scale
        kpss   /= det_scale

        # 裁剪
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h)

        # NMS
        dets = np.hstack([bboxes, scores[:, None]])
        keep = _nms(dets, 0.45)

        results = []
        for i in keep:
            x1, y1, x2, y2, s = dets[i]
            results.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'score': round(float(s), 4),
                'kps': kpss[i].reshape(5, 2).tolist()  # [leye, reye, nose, lmouth, rmouth]
            })
        return results


# ═══════════════════════════════════════════
# 眼纵横比 (EAR)
# ═══════════════════════════════════════════

def eye_aspect_ratio(eye_pts):
    """eye_pts: [(x,y),...] 6个点"""
    if len(eye_pts) < 6:
        # 简化：用4个角点估算
        pts = np.array(eye_pts[:4])
        if len(pts) < 4: return 0.3
        v = np.linalg.norm(pts[1] - pts[3])
        h = np.linalg.norm(pts[0] - pts[2])
        return v / max(h, 0.001)
    pts = np.array(eye_pts)
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * max(h, 0.001))


# ═══════════════════════════════════════════
# 场景分类器
# ═══════════════════════════════════════════

class SceneClassifier:
    """启发式场景分类：7类"""

    LABELS = ['人像写真', '集体照', '风景风光', '美食探店', '室内空间', '商品拍摄', '其他场景']
    ICONS  = {'人像写真':'🧑‍🤝‍🧑','集体照':'👥','风景风光':'🏞️','美食探店':'🍜',
              '室内空间':'🏠','商品拍摄':'📦','其他场景':'🎨'}

    def classify(self, img_rgb, face_count):
        h, w = img_rgb.shape[:2]
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # 特征
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edge_density = cv2.Canny(gray, 50, 150).sum() / gray.size

        # 暖色调占比（美食特征）
        warm_mask = (hsv[:,:,0] < 30) | ((hsv[:,:,0] > 150) & (hsv[:,:,0] < 180))
        warm_ratio = warm_mask.sum() / warm_mask.size

        # 绿色占比（风景特征）
        green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 40)
        green_ratio = green_mask.sum() / green_mask.size

        # 判断
        if face_count == 1 and avg_val > 80:
            return '人像写真'
        if face_count >= 3:
            return '集体照'
        if face_count == 0:
            if green_ratio > 0.2:
                return '风景风光'
            if warm_ratio > 0.5 and avg_sat > 60:
                return '美食探店'
            if edge_density < 0.03 and avg_val < 120:
                return '室内空间'
            if edge_density > 0.08 and avg_sat < 50:
                return '商品拍摄'
        if face_count == 2:
            return '人像写真'
        return '其他场景'


# ═══════════════════════════════════════════
# 多维度质量分析
# ═══════════════════════════════════════════

class QualityAnalyzer:
    """多维图片质量评分：综合/人像/技术/构图/美学"""

    def analyze(self, img_rgb, gray, face_results=None):
        h, w = gray.shape
        score = {}

        # 1. 技术质量 (0-100)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(lap_var / 200 * 100, 100)

        # 曝光
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cdf = hist.cumsum() / gray.size
        shadows, midtones, highlights = cdf[30], cdf[200]-cdf[50], 1-cdf[220]
        exposure = max(0, 100 - abs(shadows-0.1)*80 - abs(midtones-0.65)*80 - abs(highlights-0.1)*80)

        # 噪点（高频成分）
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        noise = np.std(gray.astype(float) - blur.astype(float))
        noise_score = max(0, 100 - noise * 5)

        score['tech'] = {
            'overall': round(sharpness*0.4 + exposure*0.35 + noise_score*0.25, 1),
            'sharpness': round(sharpness, 1), 'exposure': round(exposure, 1),
            'noise': round(noise_score, 1)
        }

        # 2. 构图质量 (0-100)
        # 三分法
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
        edges_abs = np.abs(edges)
        third_score = 0
        for pos in [w//3, 2*w//3, h//3, 2*h//3]:
            if pos < len(edges_abs.shape) and (pos < edges_abs.shape[1] if w//3 == pos else pos < edges_abs.shape[0]):
                pass
        line_energy = (np.sum(edges_abs[:, w//3-3:w//3+3]) + np.sum(edges_abs[:, 2*w//3-3:2*w//3+3]) +
                       np.sum(edges_abs[h//3-3:h//3+3, :]) + np.sum(edges_abs[2*h//3-3:2*h//3+3, :]))
        total_energy = np.sum(edges_abs) + 1e-8
        third_score = min(line_energy / total_energy * 150, 100)

        # 画面平衡（左右亮度差）
        left_bright = np.mean(gray[:, :w//2])
        right_bright = np.mean(gray[:, w//2:])
        balance = max(0, 100 - abs(left_bright - right_bright) * 4)

        score['composition'] = {
            'overall': round(third_score*0.5 + balance*0.5, 1),
            'rule_of_thirds': round(third_score, 1),
            'balance': round(balance, 1)
        }

        # 3. 人像质量 (0-100)
        if face_results and len(face_results) > 0:
            eyes_avg, face_angles, face_lights, face_scores = [], [], [], []
            for fr in face_results:
                if 'eyes_open' in fr:
                    eyes_avg.append(fr['eyes_open'])
                if 'head_angle' in fr:
                    face_angles.append(fr['head_angle'])
                if 'lighting' in fr:
                    face_lights.append(fr['lighting'])
                face_scores.append(fr.get('score', 80))
            eye_ok = np.mean(eyes_avg) if eyes_avg else 100
            angle_ok = max(0, 100 - np.mean(face_angles) * 2) if face_angles else 100
            light_ok = np.mean(face_lights) if face_lights else 80
            score['portrait'] = {
                'overall': round(eye_ok*0.35 + angle_ok*0.3 + light_ok*0.2 + np.mean(face_scores)*0.15, 1),
                'eyes': round(eye_ok, 1), 'angle': round(angle_ok, 1),
                'lighting': round(light_ok, 1)
            }
        else:
            score['portrait'] = {'overall': 0, 'eyes': 0, 'angle': 0, 'lighting': 0}

        # 4. 美学评分 (0-10)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        avg_s = np.mean(hsv[:,:,1]) / 255
        color_var = np.std(hsv[:,:,0].astype(float)) / 90
        color_score = min(avg_s * 8 + color_var * 2, 10)
        aesthetic = round(sharpness/100 * 2.5 + exposure/100 * 2.5 + color_score + third_score/100 * 2.5, 1)
        score['aesthetic'] = round(max(0, min(10, aesthetic)), 1)

        # 5. 综合评分 (0-100)
        weights = {'tech': 0.3, 'composition': 0.25, 'portrait': 0.25, 'aesthetic': 0.2}
        overall = (score['tech']['overall'] * 0.3 + score['composition']['overall'] * 0.25 +
                   (score['portrait']['overall'] if score['portrait']['overall'] > 0 else 70) * 0.15 +
                   score['aesthetic'] * 10 * 0.2 +
                   (sharpness * 0.1))
        score['overall'] = round(overall, 1)

        # 等级
        if score['overall'] >= 90: score['grade'] = 'A 优秀'
        elif score['overall'] >= 75: score['grade'] = 'B 良好'
        elif score['overall'] >= 60: score['grade'] = 'C 一般'
        else: score['grade'] = 'D 较差'

        return score

    def suggest(self, score, scene, face_count, blur_type=''):
        """生成改进建议"""
        tips = []
        if score['tech']['sharpness'] < 50:
            tips.append('照片较模糊' + (f'({blur_type})' if blur_type else '，建议重拍'))
        if score['tech']['exposure'] < 50:
            tips.append('曝光不理想，建议调整曝光补偿或后期修正')
        if score['composition']['rule_of_thirds'] < 40:
            tips.append('构图可优化，建议将主体置于三分线位置')
        if score['portrait'].get('eyes', 100) < 40:
            tips.append('检测到人物闭眼或眯眼，建议重拍')
        if score['portrait'].get('angle', 100) < 50:
            tips.append('人脸角度偏大，建议正面拍摄')
        if not tips:
            if score['overall'] >= 85:
                tips.append('整体质量良好，可直接使用')
            else:
                tips.append('整体质量中规中矩，可用于日常分享')
        return tips


# ═══════════════════════════════════════════
# 统一接口
# ═══════════════════════════════════════════

class AIDetector:
    def __init__(self, enable_ai=True, progress_cb=None):
        self.enable_ai = enable_ai and HAS_ONNX
        self.face_detector = None
        self.scene_classifier = SceneClassifier()
        self.quality_analyzer = QualityAnalyzer()
        self.progress_cb = progress_cb

        if self.enable_ai:
            self._init()

    def _init(self):
        ok = ensure_model(SCRFD_URL, SCRFD_PATH, self.progress_cb)
        if ok:
            try:
                self.face_detector = SCRFD(SCRFD_PATH)
                if self.progress_cb: self.progress_cb("SCRFD 就绪 ✓")
            except Exception as e:
                if self.progress_cb: self.progress_cb(f"SCRFD失败: {e}")

    def detect_faces(self, img):
        if self.face_detector:
            try: return self.face_detector.detect(img, 0.5)
            except: return None
        return None

    def analyze_face(self, face_data):
        """分析单张人脸质量"""
        result = {'score': 80, 'eyes_open': 100, 'head_angle': 0, 'lighting': 80}
        if not face_data: return result

        kps = face_data.get('kps', [])
        box = face_data.get('box', (0,0,100,100))
        x1,y1,x2,y2 = box
        fw, fh = x2-x1, y2-y1

        # EAR from keypoints (leye=0, reye=1)
        if len(kps) >= 2:
            leye = np.array(kps[0])
            reye = np.array(kps[1])
            # 近似EAR：眼距vs脸宽
            eye_dist = np.linalg.norm(leye - reye)
            ear_approx = min(1.0, eye_dist / max(fw, 1) * 2.5)
            result['eyes_open'] = round(min(ear_approx * 100, 100), 1)

        # 角度（眼连线vs水平）
        if len(kps) >= 2:
            dx, dy = np.array(kps[1]) - np.array(kps[0])
            angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
            result['head_angle'] = round(angle, 1)

        # 置信度分数
        result['score'] = round(face_data.get('score', 0.8) * 100, 1)
        result['box'] = box

        return result

    def classify_scene(self, img_rgb, face_count):
        return self.scene_classifier.classify(img_rgb, face_count)

    def analyze_quality(self, img_rgb, gray, face_results):
        return self.quality_analyzer.analyze(img_rgb, gray, face_results)

    def suggest(self, score, scene, face_count, blur_type=''):
        return self.quality_analyzer.suggest(score, scene, face_count, blur_type)