"""
AI ???? v5.0 ? SCRFD_2.5GF + ??????? + ??????
- SCRFD ???? + 5???? ? EAR??
- ???????7??
- ??????????/??/??/??/???
- ????????inspectors.py????/??/??/??/???/??/???/????
- ???????????? cleanvision outlier ???
"""

import os, urllib.request, numpy as np, cv2, tkinter as tk
from typing import Optional

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from inspectors import (
    InspectionPipeline, ImgStats, OutlierInspector,
)

VERSION = "5.0"

# ???????????????????????????????????????????
# ??
# ???????????????????????????????????????????

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

def ensure_model(urls, path, max_retries=3, cb=None, timeout=180, retry_wait=3):
    """??????????????? + ????
    urls: ??? URL ?????????
    max_retries: ??????????
    cb: ????
    ??: True=????, False=????
    """
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for url in urls:
        for attempt in range(1, max_retries + 1):
            try:
                if cb:
                    if max_retries > 1:
                        cb(f"??? (?? {attempt}/{max_retries})...")
                    else:
                        cb(f"?? {os.path.basename(path)}...")
                req = urllib.request.Request(url, headers={'User-Agent': f'photo-checker/{VERSION}'})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = r.read()
                with open(path, 'wb') as f:
                    f.write(data)
                # ?????????? 1MB?
                if os.path.getsize(path) > 1024 * 1024:
                    if cb: cb("????")
                    return True
                else:
                    if cb: cb(f"????????...")
            except Exception as e:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                if attempt < max_retries:
                    if cb: cb(f"?????{180}s ???...")
                    import time; time.sleep(retry_wait)  # ????
                else:
                    if cb: cb(f"????: {e}")
    return False


# ???????????????????????????????????????????
# SCRFD ????
# ???????????????????????????????????????????

class SCRFD:
    """SCRFD_2.5GF ONNX ???? + 5????"""

    INPUT_SIZE = (640, 640)
    STRIDES = [8, 16, 32]
    NUM_ANCHORS = 2
    CENTER_CACHE = {}

    def __init__(self, model_path=None):
        if not HAS_ONNX:
            raise RuntimeError("?? onnxruntime")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"???????: {model_path}")
        self.session = ort.InferenceSession(model_path,
            providers=['CPUExecutionProvider'])
        self._input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _distance2bbox(points, distance):
        """??bbox"""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _distance2kps(points, distance):
        """?????"""
        kps = np.zeros((len(points), 10), dtype=np.float32)
        for i in range(5):
            kps[:, i*2]   = points[:, 0] + distance[:, i*2]
            kps[:, i*2+1] = points[:, 1] + distance[:, i*2+1]
        return kps

    def detect(self, img_bgr, threshold=0.5):
        h, w = img_bgr.shape[:2]

        # ???
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

        # ??
        outputs = self.session.run(None, {self._input_name: input_tensor})

        # ???????
        scores_list, bboxes_list, kpss_list = [], [], []
        fmc = 3  # 3 feature map levels
        for idx, stride in enumerate(self.STRIDES):
            score = outputs[idx]        # (1, N, 1)
            bbox  = outputs[idx + fmc]  # (1, N, 4)
            kps   = outputs[idx + fmc*2]  # (1, N, 10)

            score = score[0, :, 0]
            bbox  = bbox[0]
            kps   = kps[0]

            # ??????
            feat_h, feat_w = 640 // stride, 640 // stride
            key = (feat_h, feat_w, stride)
            if key not in self.CENTER_CACHE:
                yv, xv = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing='ij')
                centers = np.stack([xv, yv], axis=2).reshape(-1, 2) * stride
                self.CENTER_CACHE[key] = centers.astype(np.float32)
            centers = self.CENTER_CACHE[key]

            bbox = bbox * stride
            kps  = kps * stride

            # ??
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

        # ?????
        bboxes /= det_scale
        kpss   /= det_scale

        # ??
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


# ???????????????????????????????????????????
# ???? (EAR)
# ???????????????????????????????????????????

def eye_aspect_ratio(eye_pts):
    """eye_pts: [(x,y),...] 6??"""
    if len(eye_pts) < 6:
        # ????4?????
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


# ???????????????????????????????????????????
# ?????
# ???????????????????????????????????????????

class SceneClassifier:
    """????????7?"""

    LABELS = ['????', '???', '????', '????', '????', '????', '????']
    ICONS  = {'????':'????????','???':'??','????':'???','????':'??',
              '????':'??','????':'??','????':'??'}

    def classify(self, img_rgb, face_count):
        h, w = img_rgb.shape[:2]
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # ??
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edge_density = cv2.Canny(gray, 50, 150).sum() / gray.size

        # ???????????
        warm_mask = (hsv[:,:,0] < 30) | ((hsv[:,:,0] > 150) & (hsv[:,:,0] < 180))
        warm_ratio = warm_mask.sum() / warm_mask.size

        # ??????????
        green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 40)
        green_ratio = green_mask.sum() / green_mask.size

        # ??
        if face_count == 1 and avg_val > 80:
            return '????'
        if face_count >= 3:
            return '???'
        if face_count == 0:
            if green_ratio > 0.2:
                return '????'
            if warm_ratio > 0.5 and avg_sat > 60:
                return '????'
            if edge_density < 0.03 and avg_val < 120:
                return '????'
            if edge_density > 0.08 and avg_sat < 50:
                return '????'
        if face_count == 2:
            return '????'
        return '????'


# ???????????????????????????????????????????
# ???????
# ???????????????????????????????????????????

class QualityAnalyzer:
    """???????????/??/??/??/??"""

    def analyze(self, img_rgb, gray, face_results=None):
        h, w = gray.shape
        score = {}

        # 1. ???? (0-100)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(lap_var / 200 * 100, 100)

        # ??
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cdf = hist.cumsum() / gray.size
        shadows, midtones, highlights = cdf[30], cdf[200]-cdf[50], 1-cdf[220]
        exposure = max(0, 100 - abs(shadows-0.1)*80 - abs(midtones-0.65)*80 - abs(highlights-0.1)*80)

        # ????????
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        noise = np.std(gray.astype(float) - blur.astype(float))
        noise_score = max(0, 100 - noise * 5)

        score['tech'] = {
            'overall': round(sharpness*0.4 + exposure*0.35 + noise_score*0.25, 1),
            'sharpness': round(sharpness, 1), 'exposure': round(exposure, 1),
            'noise': round(noise_score, 1)
        }

        # 2. ???? (0-100)
        # ????? Canny ????????????????????
        edges = cv2.Canny(gray, 50, 150)
        edges_float = edges.astype(np.float64)
        total_edges = edges_float.sum() + 1e-8

        # ?????????? ?10 ?????
        thirds = [w // 3, 2 * w // 3, h // 3, 2 * h // 3]
        band_width = 10
        line_energy = 0.0
        for pos in thirds:
            if pos >= band_width:
                if pos < w:  # ?????
                    lo = max(0, pos - band_width)
                    hi = min(w, pos + band_width)
                    line_energy += edges_float[:, lo:hi].sum()
                if pos < h:  # ?????
                    lo = max(0, pos - band_width)
                    hi = min(h, pos + band_width)
                    line_energy += edges_float[lo:hi, :].sum()

        third_score = min(line_energy / total_edges * 150, 100)

        # ???????????
        left_bright = np.mean(gray[:, :w//2])
        right_bright = np.mean(gray[:, w//2:])
        balance = max(0, 100 - abs(left_bright - right_bright) * 4)

        score['composition'] = {
            'overall': round(third_score*0.5 + balance*0.5, 1),
            'rule_of_thirds': round(third_score, 1),
            'balance': round(balance, 1)
        }

        # 3. ???? (0-100)
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

        # 4. ???? (0-10)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        avg_s = np.mean(hsv[:,:,1]) / 255
        color_var = np.std(hsv[:,:,0].astype(float)) / 90
        color_score = min(avg_s * 8 + color_var * 2, 10)
        aesthetic = round(sharpness/100 * 2.5 + exposure/100 * 2.5 + color_score + third_score/100 * 2.5, 1)
        score['aesthetic'] = round(max(0, min(10, aesthetic)), 1)

        # 5. ???? (0-100)
        weights = {'tech': 0.3, 'composition': 0.25, 'portrait': 0.25, 'aesthetic': 0.2}
        overall = (score['tech']['overall'] * 0.3 + score['composition']['overall'] * 0.25 +
                   (score['portrait']['overall'] if score['portrait']['overall'] > 0 else 70) * 0.15 +
                   score['aesthetic'] * 10 * 0.2 +
                   (sharpness * 0.1))
        score['overall'] = round(overall, 1)

        # ??
        if score['overall'] >= 90: score['grade'] = 'A ??'
        elif score['overall'] >= 75: score['grade'] = 'B ??'
        elif score['overall'] >= 60: score['grade'] = 'C ??'
        else: score['grade'] = 'D ??'

        return score

    def suggest(self, score, scene, face_count, blur_type=''):
        tips = []
        tech = score.get('tech', {})
        comp = score.get('composition', {})
        port = score.get('portrait', {})
        ov = score.get('overall', 0)

        # ?????
        sh = tech.get('sharpness', 100)
        ex = tech.get('exposure', 100)
        if sh < 30: tips.append(f'??????(???{sh:.0f}/100)???????')
        elif sh < 55: tips.append(f'??????(???{sh:.0f}/100)????????')
        if ex < 30: tips.append(f'?????????({ex:.0f}/100)?????????')
        elif ex < 55: tips.append(f'??????({ex:.0f}/100)????????')

        # ????
        rt = comp.get('rule_of_thirds', 100)
        bal = comp.get('balance', 100)
        if rt < 25: tips.append('???????????????????1/3?')
        elif rt < 45: tips.append('???????????????????')
        if bal < 40: tips.append('???????????????')

        # ????(?????)
        if face_count > 0:
            eyes = port.get('eyes', 100)
            angle = port.get('angle', 100)
            if eyes < 20: tips.append(f'??????(???{eyes:.0f}%)????????')
            elif eyes < 50: tips.append(f'??????(???{eyes:.0f}%)???????')
            if angle < 30: tips.append('??????????????')
            if scene == '???' and face_count > 5:
                tips.append('????????????????????????')
            elif scene == '????' and face_count == 1:
                tips.append('??????????????')

        # ??????
        if scene == '????' and rt < 50: tips.append('?????????????????1/3??1/3')
        if scene == '????' and ex < 60: tips.append('?????????????????????')
        if scene == '????' and ex < 50: tips.append('??????????????????')

        # ??
        if not tips:
            if ov >= 90: tips.append('? ????????????????????')
            elif ov >= 75: tips.append('????????????????')
            elif ov >= 55: tips.append('????????????????')
            else: tips.append('???????????????')
        return tips[:4]  # ??4?


# ???????????????????????????????????????????
# ????
# ???????????????????????????????????????????

class AIDetector:
    def __init__(self, enable_ai=True, progress_cb=None, cfg=None):
        self.enable_ai = enable_ai and HAS_ONNX
        self.face_detector = None
        self.scene_classifier = SceneClassifier()
        self.quality_analyzer = QualityAnalyzer()
        self.progress_cb = progress_cb
        self.model_path = ''
        self.urls = []
        self.timeout = 180

        if cfg:
            self.model_path = cfg.scrfd_path()
            self.urls = cfg.scrfd_urls()
            self.timeout = cfg.model_download_timeout()

        # ???????
        self.pipeline = InspectionPipeline(self._build_inspector_config(cfg))

        if self.enable_ai:
            self._init()

    def _build_inspector_config(self, cfg):
        """? Config ?? inspector ??"""
        if not cfg:
            return {}
        return {
            'blur_threshold': cfg.gf('blur_threshold') if hasattr(cfg, 'gf') else 80,
            'overexposure_pct': cfg.gf('overexposure_pct') if hasattr(cfg, 'gf') else 80,
            'underexposure_pct': cfg.gf('underexposure_pct') if hasattr(cfg, 'gf') else 18,
            'over_brightness': cfg.gi('over_brightness') if hasattr(cfg, 'gi') else 240,
            'under_brightness': cfg.gi('under_brightness') if hasattr(cfg, 'gi') else 30,
        }

    def _init(self):
        """???????? ? ?? ? ??"""
        model_path = self.model_path  # ? Config ??
        urls = self.urls  # ? Config ??
        timeout = self.timeout  # ? Config ??

        if not os.path.exists(model_path):
            ok = ensure_model(urls, model_path, max_retries=3, cb=self.progress_cb, timeout=timeout)
            if not ok:
                self._fallback_to_haar("?????????????????")
                return

        try:
            self.face_detector = SCRFD(model_path)
            if self.progress_cb:
                self.progress_cb("SCRFD??????")
        except FileNotFoundError as e:
            self._fallback_to_haar(str(e))
        except Exception as e:
            self._fallback_to_haar(f"SCRFD????: {e}")

    def _fallback_to_haar(self, reason):
        """??????? OpenCV ?? Haar ????"""
        self.face_detector = None
        self._haar_fallback = True  # ?????????
        if self.progress_cb:
            self.progress_cb(f"??: {reason}")

    def detect_faces(self, img):
        if self.face_detector:
            try:
                return self.face_detector.detect(img, 0.5)
            except Exception:
                return None
        # ????? None?? photo_checker.py ????? Haar ??
        return None

    def analyze_face(self, face_data):
        """????????"""
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
            nose  = np.array(kps[2]) if len(kps) >= 3 else None
            lmouth = np.array(kps[3]) if len(kps) >= 4 else None
            rmouth = np.array(kps[4]) if len(kps) >= 5 else None
            fw, fh = max(x2-x1, 1), max(y2-y1, 1)

            # ?? ??1??? EAR (?? 6 ???) ??
            # ? 5 ????? 6 ????(0)?(??,??,??), ??(1)?(??,??,??)
            # ????????? / ???? ??
            eye_dist = np.linalg.norm(leye - reye)  # ????
            face_w   = max(eye_dist * 2.0, fw)       # ????????
            vertical_sum = 0.0
            vertical_count = 0
            # ???????????????
            if nose is not None:
                proj_left = leye + (nose - leye) * 0.3  # ??????
                vertical_sum += np.linalg.norm(leye - proj_left)
                vertical_count += 1
            if rmouth is not None and nose is not None:
                proj_right = reye + (nose - reye) * 0.3
                vertical_sum += np.linalg.norm(reye - proj_right)
                vertical_count += 1

            # ??2?????????/???????
            eye_aspect = eye_dist / max(face_w, 1)
            # ????: eye_aspect ~ 0.4-0.5 ? 100%, ??: ~0.2-0.3 ? 0%
            ear_score = (eye_aspect - 0.2) / 0.3  # 0.2?0%, 0.5?100%
            ear_score = max(0.0, min(1.0, ear_score))

            # ?????????????EAR????
            if vertical_count >= 2:
                true_ear = vertical_sum / (2.0 * max(eye_dist, 1))
                ear_score = ear_score * 0.4 + (1.0 - min(true_ear * 2.5, 1.0)) * 0.6

            result['eyes_open'] = round(ear_score * 100, 1)

        # ??????vs???
        if len(kps) >= 2:
            dx, dy = np.array(kps[1]) - np.array(kps[0])
            angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
            result['head_angle'] = round(angle, 1)

        # ?????
        result['score'] = round(face_data.get('score', 0.8) * 100, 1)
        result['box'] = box

        return result

    def classify_scene(self, img_rgb, face_count):
        return self.scene_classifier.classify(img_rgb, face_count)

    def analyze_quality(self, img_rgb, gray, face_results):
        return self.quality_analyzer.analyze(img_rgb, gray, face_results)

    def suggest(self, score, scene, face_count, blur_type=''):
        return self.quality_analyzer.suggest(score, scene, face_count, blur_type)

    def detect_issues(self, img_rgb, gray, stats: Optional[ImgStats] = None) -> list:
        """???????????? Issue ?????? dict?"""
        issues = self.pipeline.run(img_rgb, gray, stats)
        return [i.to_dict() for i in issues]

    @staticmethod
    def detect_outliers(results: list, zscore_thresh: float = 2.5) -> list:
        """???????????? cleanvision outlier??
        results: check_single ????????
        ?? [(????, [???])] ???
        """
        stats_list = OutlierInspector.collect_stats(results)
        if not stats_list:
            return []
        return OutlierInspector.detect(stats_list, zscore_thresh)