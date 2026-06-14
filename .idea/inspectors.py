"""
检测引擎（模块化） — 借鉴 cleanvision ImageInspection 设计模式
每个检测项独立为一个 Inspector 类，支持单图检测和批量数据集异常检测。

新增（cleanvision 启发）：
- 灰度过低 / 颜色过饱和
- 过暗 / 过亮（perceptual brightness 维度）
- 数据集级异常值检测（亮度/饱和度/尺寸/噪点）
- 结构重复检测（block correlation，补 pHash 重复）

原有：
- 模糊、过曝、欠曝、纯黑/白、信息量极低
- pHash 重复
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════

@dataclass
class Issue:
    """检测问题"""
    name: str        # 问题键名（英文，用于逻辑判断）
    label: str       # 显示标签（中文）
    confidence: float  # 置信度 0-100
    detail: str      # 详细说明

    def to_dict(self):
        return {'t': self.label, 'c': round(self.confidence, 1), 'd': self.detail}


@dataclass
class ImgStats:
    """单图统计特征（供异常值检测使用）"""
    width: int = 0
    height: int = 0
    mean_brightness: float = 0.0
    std_brightness: float = 0.0
    mean_saturation: float = 0.0
    mean_sharpness: float = 0.0
    noise_level: float = 0.0
    edge_density: float = 0.0
    aspect_ratio: float = 0.0
    pixel_count: int = 0  # total pixels


# ═══════════════════════════════════════════
# 基类
# ═══════════════════════════════════════════

class Inspector:
    """检测器基类 — 子类 inspect() 必须 self-try-except，绝不让异常外泄"""
    name: str = "base"
    label: str = "检测"

    def inspect(self, img_rgb: np.ndarray, gray: np.ndarray, stats: ImgStats) -> list[Issue]:
        raise NotImplementedError

    # ── 安全调用包装：确保任何 inspector 的异常都不会泄漏 ──
    def safe_inspect(self, img_rgb, gray, stats):
        """包装 inspect()，捕获所有异常，返回 (issues, error_msg)"""
        try:
            return self.inspect(img_rgb, gray, stats), None
        except PermissionError:
            return [], f"权限不足，无法读取图片数据"
        except cv2.error as e:
            return [], f"OpenCV 错误: 文件可能已损坏或格式不支持"
        except Exception as e:
            return [], f"检测引擎内部错误: {type(e).__name__}"


# ═══════════════════════════════════════════
# 传统检测（从 ai_detector.py 迁移）
# ═══════════════════════════════════════════

class BlurInspector(Inspector):
    """Laplacian 方差测模糊"""
    name = "blurry"
    label = "模糊"

    def __init__(self, threshold: float = 80.0):
        self.threshold = threshold

    def inspect(self, img_rgb, gray, stats):
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < self.threshold:
            bc = '失焦模糊' if lap_var < 30 else ('运动模糊' if lap_var < 60 else '')
            conf = max(0, (1 - lap_var / self.threshold) * 100)
            return [Issue(self.name, self.label, conf, f"方差={lap_var:.0f} {bc}")]
        return []

    def safe_inspect(self, img_rgb, gray, stats):
        try:
            return self.inspect(img_rgb, gray, stats), None
        except PermissionError:
            return [], "权限不足，无法读取图片数据"
        except cv2.error:
            return [], "OpenCV 错误: 文件可能已损坏或格式不支持"
        except Exception:
            return [], "检测引擎内部错误: BlurInspector"


class ExposureInspector(Inspector):
    """直方图过曝/欠曝检测"""
    name = "exposure"
    label = "曝光"

    def __init__(self, over_pct: float = 80.0, under_pct: float = 18.0,
                 over_brightness: int = 240, under_brightness: int = 30):
        self.over_pct = over_pct
        self.under_pct = under_pct
        self.over_brightness = over_brightness
        self.under_brightness = under_brightness

    def inspect(self, img_rgb, gray, stats):
        issues = []
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total = gray.size
        # 过曝
        over_pixels = np.sum(hist[self.over_brightness:])
        over_ratio = over_pixels / total * 100
        if over_ratio > self.over_pct:
            issues.append(Issue('overexposed', '过曝', over_ratio,
                                f"{over_ratio:.0f}% 像素过亮"))
        # 欠曝
        under_pixels = np.sum(hist[:self.under_brightness])
        under_ratio = under_pixels / total * 100
        if under_ratio > self.under_pct:
            issues.append(Issue('underexposed', '欠曝', under_ratio,
                                f"{under_ratio:.0f}% 像素过暗"))
        return issues

    def safe_inspect(self, img_rgb, gray, stats):
        try:
            return self.inspect(img_rgb, gray, stats), None
        except PermissionError:
            return [], "权限不足，无法读取图片数据"
        except cv2.error:
            return [], "OpenCV 错误: 文件可能已损坏或格式不支持"
        except Exception:
            return [], "检测引擎内部错误: ExposureInspector"


class ToneInspector(Inspector):
    """过暗 / 过亮（perceptual 维度，借鉴 cleanvision 的 dark / overexposed）
    关注图像整体感知明暗，而非局部像素比例。"""
    name = "tone"
    label = "明暗"

    def inspect(self, img_rgb, gray, stats):
        issues = []
        mean_b = np.mean(gray)
        # 过暗：均值 < 60 且暗像素 (>85%) 集中在低亮度
        dark_ratio = np.sum(gray < 60) / gray.size
        if mean_b < 60 and dark_ratio > 0.85:
            conf = min(100, (1 - mean_b / 60) * 100 * dark_ratio)
            issues.append(Issue('dark', '过暗', round(conf, 1),
                                f"平均亮度 {mean_b:.0f}/255，暗区占比 {dark_ratio*100:.0f}%"))
        # 过亮：均值 > 220 且亮像素 (>85%) 集中在高亮度
        bright_ratio = np.sum(gray > 220) / gray.size
        if mean_b > 220 and bright_ratio > 0.85:
            conf = min(100, (mean_b - 220) / 35 * 100 * bright_ratio)
            issues.append(Issue('overexposed_bright', '过亮', round(conf, 1),
                                f"平均亮度 {mean_b:.0f}/255，亮区占比 {bright_ratio*100:.0f}%"))
        return []

    def safe_inspect(self, img_rgb, gray, stats):
        try:
            return self.inspect(img_rgb, gray, stats), None
        except PermissionError:
            return [], "权限不足，无法读取图片数据"
        except cv2.error:
            return [], "OpenCV 错误: 文件可能已损坏或格式不支持"
        except Exception:
            return [], "检测引擎内部错误: ToneInspector"


class MonochromeInspector(Inspector):
    """灰度过低检测（借鉴 cleanvision 的 low_saturation）
    计算 HSV 饱和度的分布，判断是否接近灰度。"""
    name = "low_saturation"
    label = "灰暗"

    def inspect(self, img_rgb, gray, stats):
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        mean_sat = np.mean(sat)
        low_sat_ratio = np.sum(sat < 20) / sat.size

        if mean_sat < 30 and low_sat_ratio > 0.7:
            conf = min(100, (1 - mean_sat / 30) * 100 * low_sat_ratio)
            return [Issue(self.name, self.label, round(conf, 1),
                          f"平均饱和度 {mean_sat:.0f}/255，低饱和区占比 {low_sat_ratio*100:.0f}%")]
        return []

    def safe_inspect(self, img_rgb, gray, stats):
        try:
            return self.inspect(img_rgb, gray, stats), None
        except PermissionError:
            return [], "权限不足，无法读取图片数据"
        except cv2.error:
            return [], "OpenCV 错误: 文件可能已损坏或格式不支持"
        except Exception:
            return [], "检测引擎内部错误: MonochromeInspector"


class ColorfulInspector(Inspector):
    """颜色过饱和检测（借鉴 cleanvision 的 colorful）
    检测是否有大面积颜色过度饱和的区域。"""
    name = "colorful"
    label = "过饱和"

    def inspect(self, img_rgb, gray, stats):
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        high_sat_ratio = np.sum(sat > 230) / sat.size
        mean_sat = np.mean(sat)

        if high_sat_ratio > 0.6 and mean_sat > 160:
            conf = min(100, high_sat_ratio * mean_sat / 2.55)
            return [Issue(self.name, self.label, round(conf, 1),
                          f"高饱和区占比 {high_sat_ratio*100:.0f}%，平均饱和度 {mean_sat:.0f}/255")]
        return []

    def safe_inspect(self, img_rgb, gray, stats):
        try:
            return self.inspect(img_rgb, gray, stats), None
        except PermissionError:
            return [], "权限不足，无法读取图片数据"
        except cv2.error:
            return [], "OpenCV 错误: 文件可能已损坏或格式不支持"
        except Exception:
            return [], "检测引擎内部错误: ColorfulInspector"


class SolidColorInspector(Inspector):
    """纯黑 / 纯白检测"""
    name = "solid_color"
    label = "纯色"

    def inspect(self, img_rgb, gray, stats):
        mean_val = np.mean(gray)
        var_val = np.var(gray)
        if mean_val > 245 and var_val < 10:
            return [Issue('over_white', '纯白', max(0, (1 - var_val / 10) * 100),
                          f"均值={mean_val:.0f} 方差={var_val:.1f}")]
        if mean_val < 10 and var_val < 10:
            return [Issue('under_black', '纯黑', max(0, (1 - var_val / 10) * 100),
                          f"均值={mean_val:.0f} 方差={var_val:.1f}")]
        return []

    def safe_inspect(self, img_rgb, gray, stats):
        try:
            return self.inspect(img_rgb, gray, stats), None
        except PermissionError:
            return [], "权限不足，无法读取图片数据"
        except cv2.error:
            return [], "OpenCV 错误: 文件可能已损坏或格式不支持"
        except Exception:
            return [], "检测引擎内部错误: SolidColorInspector"


class LowInformationInspector(Inspector):
    """信息量极低检测"""
    name = "low_information"
    label = "信息量低"

    def __init__(self, min_sat_var: float = 200, min_edge_density: float = 0.01):
        self.min_sat_var = min_sat_var
        self.min_edge_density = min_edge_density

    def inspect(self, img_rgb, gray, stats):
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        sat_var = np.var(hsv[:, :, 1])
        edge_density = cv2.Canny(gray, 50, 150).sum() / gray.size

        score = 0
        detail_parts = []
        if sat_var < self.min_sat_var:
            score += 50
            detail_parts.append(f"色域方差={sat_var:.0f}")
        if edge_density < self.min_edge_density:
            score += 50
            detail_parts.append(f"边缘密度={edge_density:.4f}")

        if score > 0:
            return [Issue(self.name, self.label, min(score, 100),
                          '、'.join(detail_parts))]
        return []

    def safe_inspect(self, img_rgb, gray, stats):
        try:
            return self.inspect(img_rgb, gray, stats), None
        except PermissionError:
            return [], "权限不足，无法读取图片数据"
        except cv2.error:
            return [], "OpenCV 错误: 文件可能已损坏或格式不支持"
        except Exception:
            return [], "检测引擎内部错误: LowInformationInspector"


# ═══════════════════════════════════════════
# 结构重复检测（cleanvision repeating 启发）
# ═══════════════════════════════════════════

class PatternRepeatInspector(Inspector):
    """结构重复检测 — 将图像划分为 block 网格，
    检测相邻/远距离 block 之间的 Pearson 相关性，
    发现规律性重复纹理/图案（如水印、栅栏、格子地板）。"""
    name = "pattern_repeat"
    label = "重复图案"

    def __init__(self, block_size: int = 64, corr_threshold: float = 0.85):
        self.block_size = block_size
        self.corr_threshold = corr_threshold

    def inspect(self, img_rgb, gray, stats):
        h, w = gray.shape
        bs = self.block_size
        rows = h // bs
        cols = w // bs
        if rows < 2 or cols < 2:
            return []

        # 提取每个 block 的平均亮度
        blocks = []
        for r in range(rows):
            for c in range(cols):
                block = gray[r*bs:(r+1)*bs, c*bs:(c+1)*bs]
                blocks.append(np.mean(block))
        blocks = np.array(blocks, dtype=np.float64)

        # 计算 block 间的相关性（只检查有空间关系的 block 对）
        high_corr_count = 0
        total_pairs = 0

        for r1 in range(rows):
            for c1 in range(cols):
                idx1 = r1 * cols + c1
                # 检查右方和下方（避免重复）
                if c1 + 1 < cols:
                    idx2 = r1 * cols + (c1 + 1)
                    total_pairs += 1
                    high_corr_count += 1 if abs(blocks[idx1] - blocks[idx2]) < 15 else 0
                if r1 + 1 < rows:
                    idx2 = (r1 + 1) * cols + c1
                    total_pairs += 1
                    high_corr_count += 1 if abs(blocks[idx1] - blocks[idx2]) < 15 else 0

        if total_pairs == 0:
            return []

        similarity = high_corr_count / total_pairs
        if similarity > self.corr_threshold:
            conf = (similarity - self.corr_threshold) / (1 - self.corr_threshold) * 100
            return [Issue(self.name, self.label, round(min(conf, 100), 1),
                          f"block 相似度 {similarity*100:.0f}%（{rows}×{cols} 网格）")]
        return []

    def safe_inspect(self, img_rgb, gray, stats):
        try:
            return self.inspect(img_rgb, gray, stats), None
        except PermissionError:
            return [], "权限不足，无法读取图片数据"
        except cv2.error:
            return [], "OpenCV 错误: 文件可能已损坏或格式不支持"
        except Exception:
            return [], "检测引擎内部错误: PatternRepeatInspector"


# ═══════════════════════════════════════════
# 数据集级异常值检测（cleanvision outlier 启发）
# ═══════════════════════════════════════════

class OutlierInspector:
    """数据集级异常值检测。
    在扫描完一批图片后，基于全局统计特征找出相对其他图片异常的目标。
    借鉴 cleanvision 的 outlier 检测思路。"""

    # 需要检测的维度
    METRICS = ['mean_brightness', 'std_brightness', 'mean_saturation',
                'mean_sharpness', 'noise_level', 'pixel_count', 'aspect_ratio']

    # 默认阈值（IQR 倍数的等效值）
    DEFAULT_ZSCORE_THRESH = 2.5

    @staticmethod
    def collect_stats(results: list) -> list[ImgStats]:
        """从扫描结果中提取 ImgStats 列表"""
        stats_list = []
        for r in results:
            s = ImgStats()
            md = r.get('max_dim', (0, 0))
            if md[0] > 0:
                s.width, s.height = md
                s.pixel_count = md[0] * md[1]
                s.aspect_ratio = md[0] / max(md[1], 1)
            s.mean_brightness = r.get('stat_mean_brightness', 128)
            s.std_brightness = r.get('stat_std_brightness', 64)
            s.mean_saturation = r.get('stat_mean_saturation', 128)
            s.mean_sharpness = r.get('stat_sharpness', 200)
            s.noise_level = r.get('stat_noise', 10)
            s.edge_density = r.get('stat_edge_density', 0.05)
            stats_list.append(s)
        return stats_list

    @staticmethod
    def compute_zscores(stats_list: list[ImgStats]) -> dict[int, dict[str, float]]:
        """计算每个图片每个维度的 z-score"""
        n = len(stats_list)
        if n < 3:
            return {}

        zscores = [{} for _ in range(n)]
        for metric in OutlierInspector.METRICS:
            values = [getattr(s, metric) for s in stats_list]
            arr = np.array(values, dtype=np.float64)
            mean = np.mean(arr)
            std = np.std(arr)
            if std < 1e-8:
                continue
            for i, v in enumerate(values):
                zscores[i][metric] = abs(v - mean) / std
        return zscores

    @staticmethod
    def detect(stats_list: list[ImgStats], zscore_thresh: float = 2.5) -> list[tuple[int, list[str]]]:
        """返回 [(index, [metric_names])] 列表，index 对应 stats_list 中的位置"""
        zscores = OutlierInspector.compute_zscores(stats_list)
        if not zscores:
            return []

        outliers = []
        for i, zs in enumerate(zscores):
            flagged = [m for m, z in zs.items() if z > zscore_thresh]
            if flagged:
                outliers.append((i, flagged))
        return outliers


# ═══════════════════════════════════════════
# 统一编排器
# ═══════════════════════════════════════════

class InspectionPipeline:
    """检测管线：按顺序运行所有 inspector，收集结果。
    类似 cleanvision 的 ImageInspection.get_issues()。"""

    def __init__(self, config: dict = None):
        """config 包含各 inspector 的参数"""
        cfg = config or {}
        self.inspectors: list[Inspector] = [
            BlurInspector(cfg.get('blur_threshold', 80)),
            ExposureInspector(
                cfg.get('overexposure_pct', 80), cfg.get('underexposure_pct', 18),
                cfg.get('over_brightness', 240), cfg.get('under_brightness', 30)
            ),
            ToneInspector(),
            MonochromeInspector(),
            ColorfulInspector(),
            SolidColorInspector(),
            LowInformationInspector(),
            PatternRepeatInspector(),
        ]

    def run(self, img_rgb: np.ndarray, gray: np.ndarray,
            stats: Optional[ImgStats] = None) -> list[Issue]:
        """对单张图片运行所有检测，返回问题列表"""
        if stats is None:
            stats = self._compute_stats(img_rgb, gray)
        all_issues: list[Issue] = []
        for inspector in self.inspectors:
            issues, _err = inspector.safe_inspect(img_rgb, gray, stats)
            all_issues.extend(issues)
        return all_issues

    def run_batch(self, images: list[tuple[np.ndarray, np.ndarray]],
                  stats_list: Optional[list[ImgStats]] = None,
                  **kw) -> list[list[Issue]]:
        """批量检测，返回每个图片的问题列表"""
        results = []
        for img_rgb, gray in images:
            s = stats_list[len(results)] if stats_list else None
            results.append(self.run(img_rgb, gray, s))
        return results

    @staticmethod
    def _compute_stats(img_rgb, gray) -> ImgStats:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        noise = np.std(gray.astype(float) - blur.astype(float))
        edge_density = cv2.Canny(gray, 50, 150).sum() / gray.size
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        s = ImgStats()
        s.width = gray.shape[1]
        s.height = gray.shape[0]
        s.mean_brightness = float(np.mean(gray))
        s.std_brightness = float(np.std(gray))
        s.mean_saturation = float(np.mean(hsv[:, :, 1]))
        s.mean_sharpness = lap_var
        s.noise_level = noise
        s.edge_density = edge_density
        s.aspect_ratio = s.width / max(s.height, 1)
        s.pixel_count = s.width * s.height
        return s
