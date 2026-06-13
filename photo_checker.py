"""
照片废片检测工具 v4.0 — AI 增强版
检测: 模糊(类型) / 闭眼 / 曝光 / 重复 / AI美学 / AI人脸质量
特性: ONNX Runtime · UltraLight · 美学评分 · 模糊分类
"""

import os, sys, shutil, csv, configparser, threading
from datetime import datetime
from pathlib import Path
from io import BytesIO

import cv2, numpy as np

# ── GUI ──
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 高DPI
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

try:
    import ttkbootstrap as tkb
    HAS_BOOTSTRAP = True
except ImportError:
    HAS_BOOTSTRAP = False

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── AI 模块 ──
from ai_detector import AIDetector


# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

class Config:
    DEFAULTS = {
        'blur_threshold': '100',
        'overexposure_pct': '80',
        'underexposure_pct': '18',
        'over_brightness': '240',
        'under_brightness': '30',
        'duplicate_threshold': '95',
        'max_image_dim': '1200',
        'aesthetic_min': '4.0',
        'theme': 'darkly',
        'last_folder': '',
        'enable_ai': '1',
    }

    def __init__(self):
        base = getattr(sys, '_MEIPASS', '') or os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(base, 'config.ini')
        self.cfg = configparser.ConfigParser()
        self.load()

    def load(self):
        self.cfg.read(self.path, encoding='utf-8')
        if 'settings' not in self.cfg:
            self.cfg['settings'] = {}

    def save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            self.cfg.write(f)

    def get(self, k):
        return self.cfg['settings'].get(k, self.DEFAULTS.get(k, ''))

    def get_float(self, k):
        return float(self.get(k))

    def get_int(self, k):
        return int(float(self.get(k)))

    def get_bool(self, k):
        return self.get(k) == '1'

    def set(self, k, v):
        self.cfg['settings'][k] = str(v)
        self.save()


# ═══════════════════════════════════════════
# 检测引擎（传统 + AI）
# ═══════════════════════════════════════════

class PhotoChecker:
    def __init__(self, config: Config):
        self.cfg = config
        self.ai = None  # 延迟初始化

        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_eye.xml'))

    def init_ai(self, progress_cb=None):
        if self.ai is None and self.cfg.get_bool('enable_ai'):
            self.ai = AIDetector(enable_ai=True, progress_callback=progress_cb)

    @staticmethod
    def _imread(path):
        arr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ── 模糊（分类） ──
    def check_blur(self, gray):
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        thresh = self.cfg.get_float('blur_threshold')
        is_bad = variance < thresh
        conf = max(0, min(100, (1 - variance / thresh) * 100)) if thresh > 0 else 0

        # AI 模糊分类
        blur_type = ''
        if self.ai and is_bad:
            bc = self.ai.classify_blur(gray)
            blur_type = {'defocus': '失焦模糊', 'motion_blur': '运动模糊',
                         'bokeh': '背景虚化✓', 'sharp': ''}.get(bc['type'], '')

        return is_bad, round(variance, 1), round(conf, 1), blur_type

    # ── 人脸质量 ──
    def check_faces(self, gray, img_rgb):
        """返回 (face_issues, face_quality_score)"""
        issues = []
        face_boxes_trad = []

        # AI 人脸检测
        if self.ai and self.ai.face_detector:
            try:
                dets = self.ai.detect_faces(img_rgb)
                if dets:
                    face_boxes_trad = [(d[0], d[1], d[2], d[3], d[4]) for d in dets[:5]]
            except Exception:
                pass

        # 回退 Haar
        if not face_boxes_trad:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            face_boxes_trad = [(x, y, x+w, y+h, 1.0) for (x, y, w, h) in faces[:5]]

        if not face_boxes_trad:
            return issues, 100.0, []

        face_qualities = []

        for fb in face_boxes_trad[:3]:
            x1, y1, x2, y2, conf = fb
            x1 = int(max(0, x1)); y1 = int(max(0, y1))
            x2 = int(min(gray.shape[1], x2)); y2 = int(min(gray.shape[0], y2))

            # AI 人脸质量分析
            if self.ai:
                quality = self.ai.analyze_face_quality(gray, (x1, y1, x2, y2))
            else:
                # 简易版
                upper = gray[y1:y2, x1:x2]
                if upper.size > 100:
                    fh = y2 - y1
                    eyes = self.eye_cascade.detectMultiScale(upper[:fh//2, :], 1.05, 4, minSize=(10, 10))
                    eye_open = min(len(eyes), 2) / 2 * 100
                else:
                    eye_open = 100
                quality = {'eye_open': eye_open, 'head_angle': 0,
                           'lighting_score': 80, 'overall': eye_open * 0.5 + 50}

            face_qualities.append(quality)

            if quality['eye_open'] < 50:
                issues.append({'type': '闭眼', 'conf': round(100 - quality['eye_open'], 1),
                               'detail': f"眼睛张开{quality['eye_open']:.0f}%"})
            if quality['head_angle'] > 30:
                issues.append({'type': '侧脸', 'conf': round(quality['head_angle'], 1),
                               'detail': f"偏转{quality['head_angle']:.0f}°"})
            if quality['lighting_score'] < 40:
                issues.append({'type': '人脸光照不均', 'conf': round(100 - quality['lighting_score'], 1),
                               'detail': f"均匀度{quality['lighting_score']:.0f}"})

        avg_quality = (sum(q['overall'] for q in face_qualities) /
                       len(face_qualities)) if face_qualities else 100.0

        return issues, round(avg_quality, 1), face_qualities

    # ── 曝光 ──
    def check_exposure(self, gray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total = gray.size
        over = np.sum(hist[self.cfg.get_int('over_brightness'):]) / total * 100
        under = np.sum(hist[:self.cfg.get_int('under_brightness')]) / total * 100

        if over > self.cfg.get_float('overexposure_pct'):
            return True, round(over, 1), '过曝'
        if under > self.cfg.get_float('underexposure_pct'):
            return True, round(under, 1), '欠曝'
        return False, 0, '正常'

    # ── pHash ──
    @staticmethod
    def compute_phash(gray):
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        top = dct[:8, :8]
        avg = top.mean()
        h = 0
        for b in (top > avg).flatten():
            h = (h << 1) | int(b)
        return format(h, '016x')

    @staticmethod
    def hamming_distance(h1, h2):
        return bin(int(h1, 16) ^ int(h2, 16)).count('1')

    # ── 完整检测 ──
    def check_single(self, filepath):
        r = {"file": os.path.basename(filepath), "issues": [], "phash": None,
             "aesthetic": 0, "face_quality": 100.0}

        img = self._imread(filepath)
        if img is None:
            r["issues"].append({"type": "读取失败", "conf": 100, "detail": "无法解码"})
            return r

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = gray.shape
        max_dim = self.cfg.get_int('max_image_dim')
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))

        # 1. 美学评分
        if self.ai:
            r['aesthetic'] = self.ai.score_aesthetic(img_rgb)
            min_aes = self.cfg.get_float('aesthetic_min')
            if r['aesthetic'] < min_aes:
                r['issues'].append({'type': '美学', 'conf': round((min_aes - r['aesthetic']) / min_aes * 100, 1),
                                    'detail': f"得分 {r['aesthetic']}/{min_aes}"})

        # 2. 模糊
        bad, val, conf, btype = self.check_blur(gray)
        if bad:
            detail = f"方差={val}"
            if btype: detail += f" [{btype}]"
            r['issues'].append({'type': '模糊', 'conf': conf, 'detail': detail})

        # 3. 人脸
        face_issues, face_q, _ = self.check_faces(gray, img_rgb)
        r['face_quality'] = face_q
        for fi in face_issues:
            r['issues'].append(fi)

        # 4. 曝光
        bad, conf, detail = self.check_exposure(gray)
        if bad:
            r['issues'].append({'type': '曝光', 'conf': conf, 'detail': detail})

        # 5. pHash
        try:
            small = cv2.resize(gray, (128, 128))
            r['phash'] = self.compute_phash(small)
        except Exception:
            r['phash'] = None

        return r


# ═══════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("照片废片检测工具 v4.0 AI")
        self.root.geometry("1150x780")
        self.root.minsize(950, 620)

        self.cfg = Config()
        self.checker = PhotoChecker(self.cfg)
        self.results = []
        self.bad_photos = []
        self.duplicate_groups = []
        self.move_history = []
        self.processing = False
        self._stop_flag = False
        self.has_dnd = self._check_dnd()

        self._build_ui()
        self._load_params()
        self._bind_drop()

        # 异步初始化 AI
        self.root.after(500, self._init_ai)

    def _check_dnd(self):
        try:
            import tkinterdnd2; return True
        except ImportError:
            return False

    @staticmethod
    def _model_dir():
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, 'models')

    # ── UI ──
    def _build_ui(self):
        root = self.root
        if HAS_BOOTSTRAP:
            self.style = tkb.Style(theme=self.cfg.get('theme') or 'darkly')
        else:
            self.style = ttk.Style()
            try: self.style.theme_use('clam')
            except: pass

        # 工具栏
        tb = ttk.Frame(root)
        tb.pack(fill='x', padx=10, pady=(8, 0))

        ttk.Label(tb, text="文件夹:").pack(side='left')
        self.folder_var = tk.StringVar(value=self.cfg.get('last_folder'))
        ttk.Entry(tb, textvariable=self.folder_var, width=45).pack(side='left', padx=6, fill='x', expand=True)
        ttk.Button(tb, text="选择", command=self._browse).pack(side='left', padx=2)
        self.btn_scan = ttk.Button(tb, text="扫描", command=self._start)
        self.btn_scan.pack(side='left', padx=(8, 0))
        self.btn_stop = ttk.Button(tb, text="停止", command=self._stop, state='disabled')
        self.btn_stop.pack(side='left', padx=4)
        ttk.Button(tb, text="移走废片", command=self._move_bad).pack(side='left', padx=4)
        ttk.Button(tb, text="撤销", command=self._undo).pack(side='left', padx=4)
        ttk.Button(tb, text="导出CSV", command=self._export_csv).pack(side='left', padx=4)
        self.btn_params = ttk.Button(tb, text="⚙ 参数", command=self._toggle_params)
        self.btn_params.pack(side='right', padx=4)

        # 参数面板
        self.params_visible = tk.BooleanVar(value=False)
        self.params_frame = ttk.LabelFrame(root, text="检测参数", padding=8)

        # AI 总开关
        ai_row = ttk.Frame(self.params_frame)
        ai_row.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 4))
        self.ai_var = tk.BooleanVar(value=self.cfg.get_bool('enable_ai'))
        ttk.Checkbutton(ai_row, text="启用 AI 增强检测 (UltraLight + 美学评分 + 模糊分类)",
                        variable=self.ai_var,
                        command=lambda: self.cfg.set('enable_ai', '1' if self.ai_var.get() else '0')
                        ).pack(side='left')
        ttk.Label(ai_row, text="(需 onnxruntime, 首次运行下载1.3MB模型)",
                  font=('微软雅黑', 8), foreground='#888').pack(side='left', padx=12)

        params_def = [
            ("模糊阈值", "blur_threshold", 20, 300, 5),
            ("过曝判定%", "overexposure_pct", 30, 100, 5),
            ("欠曝判定%", "underexposure_pct", 5, 50, 1),
            ("过亮值", "over_brightness", 200, 255, 5),
            ("过暗值", "under_brightness", 5, 80, 5),
            ("重复相似度%", "duplicate_threshold", 80, 100, 1),
            ("美学最低分", "aesthetic_min", 1.0, 9.0, 0.5),
        ]
        self.sliders = {}
        for i, (label, key, lo, hi, step) in enumerate(params_def):
            col, row = i % 3, (i // 3) + 1
            f = ttk.Frame(self.params_frame)
            f.grid(row=row, column=col, padx=10, pady=4, sticky='ew')
            ttk.Label(f, text=label, font=('微软雅黑', 9)).pack(anchor='w')
            is_float = isinstance(lo, float)
            if is_float:
                sv = tk.DoubleVar(value=self.cfg.get_float(key))
            else:
                sv = tk.IntVar(value=self.cfg.get_int(key))
            s = ttk.Scale(f, from_=lo, to=hi, variable=sv)
            s.pack(fill='x')
            self.sliders[key] = (sv, s, is_float)

        bf = ttk.Frame(self.params_frame)
        bf.grid(row=99, column=0, columnspan=3, pady=(8, 0), sticky='w')
        ttk.Button(bf, text="恢复默认", command=self._reset_params).pack(side='left')
        ttk.Button(bf, text="保存参数", command=self._save_params).pack(side='left', padx=8)

        for i in range(3):
            self.params_frame.columnconfigure(i, weight=1)

        # 统计
        sf = ttk.Frame(root)
        sf.pack(fill='x', padx=10, pady=(6, 0))
        self.lbl_stats = ttk.Label(sf, text="就绪", font=('微软雅黑', 10))
        self.lbl_stats.pack(side='left')
        self.lbl_ai_status = ttk.Label(sf, text="AI: 加载中...", foreground='#888')
        self.lbl_ai_status.pack(side='right')

        # 进度
        self.progress = ttk.Progressbar(root, mode='determinate')
        self.progress.pack(fill='x', padx=10, pady=2)
        self.lbl_progress = ttk.Label(root, text="")
        self.lbl_progress.pack(anchor='w', padx=14)

        # 主内容
        main = ttk.PanedWindow(root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=10, pady=(4, 10))

        # 表格
        tf = ttk.Frame(main)
        main.add(tf, weight=3)
        cols = ("filename", "issue", "conf", "detail", "aesthetic", "face")
        self.tree = ttk.Treeview(tf, columns=cols, show='headings', selectmode='browse')
        self.tree.heading("filename", text="文件名")
        self.tree.heading("issue", text="问题")
        self.tree.heading("conf", text="置信度")
        self.tree.heading("detail", text="详情")
        self.tree.heading("aesthetic", text="美学分")
        self.tree.heading("face", text="人脸质量")
        self.tree.column("filename", width=160, minwidth=100)
        self.tree.column("issue", width=80, minwidth=60)
        self.tree.column("conf", width=60, minwidth=50)
        self.tree.column("detail", width=180, minwidth=100)
        self.tree.column("aesthetic", width=60, minwidth=50)
        self.tree.column("face", width=70, minwidth=50)

        ts = ttk.Scrollbar(tf, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=ts.set)
        self.tree.pack(side='left', fill='both', expand=True)
        ts.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', self._preview)

        # 右侧面板
        rf = ttk.Frame(main)
        main.add(rf, weight=1)

        plf = ttk.LabelFrame(rf, text="图片预览", padding=4)
        plf.pack(fill='both', expand=True)
        self.lbl_preview = ttk.Label(plf, text="点击列表项预览", anchor='center')
        self.lbl_preview.pack(fill='both', expand=True)

        dlf = ttk.LabelFrame(rf, text="重复照片组", padding=4)
        dlf.pack(fill='x', pady=(4, 0))
        self.tree_dup = ttk.Treeview(dlf, columns=("files",), show='headings', height=4)
        self.tree_dup.heading("files", text="相似组")
        self.tree_dup.column("files", width=100)
        self.tree_dup.pack(fill='x')

        self.status = ttk.Label(root, text="v4.0 AI | UltraLight + NIMA-style | 纯离线",
                                relief='sunken', font=('微软雅黑', 8))
        self.status.pack(side='bottom', fill='x')

        self.drop_label = ttk.Label(root, text="💡 支持拖拽文件夹 · Ctrl+O 选择",
                                    foreground='#666')
        self.drop_label.pack(side='bottom', pady=2)

    # ── AI 初始化 ──
    def _init_ai(self):
        self.lbl_ai_status.config(text="AI: 检查模型...")
        self.root.update_idletasks()
        self.checker.init_ai(progress_cb=lambda msg: self.root.after(0,
            lambda m=msg: self._ai_progress(m)))
        if self.checker.ai and self.checker.ai.face_detector:
            self.lbl_ai_status.config(text="AI: UltraLight ✓ | 美学评分 ✓ | 模糊分类 ✓",
                                      foreground='#4cd964')
        else:
            self.lbl_ai_status.config(text="AI: 传统模式 (安装 onnxruntime 启用AI)",
                                      foreground='#888')

    def _ai_progress(self, msg):
        self.lbl_ai_status.config(text=f"AI: {msg}")
        self.root.update_idletasks()

    # ── 参数 ──
    def _toggle_params(self):
        if self.params_visible.get():
            self.params_frame.pack_forget()
            self.params_visible.set(False)
        else:
            self.params_frame.pack(fill='x', padx=10, pady=(6, 0),
                                   after=self.status.master.children['!frame2'])
            self.params_visible.set(True)

    def _load_params(self):
        for key, (sv, _, is_float) in self.sliders.items():
            try:
                val = self.cfg.get_float(key) if is_float else self.cfg.get_int(key)
                sv.set(val)
            except Exception:
                sv.set(float(Config.DEFAULTS.get(key, 0)))

    def _reset_params(self):
        for key, (sv, _, is_float) in self.sliders.items():
            default = float(Config.DEFAULTS.get(key, 0))
            if not is_float: default = int(default)
            sv.set(default)
        self._save_params()

    def _save_params(self):
        for key, (sv, _, is_float) in self.sliders.items():
            val = sv.get()
            self.cfg.set(key, str(int(val)) if not is_float else f"{val:.1f}")
        # AI 开关
        self.cfg.set('enable_ai', '1' if self.ai_var.get() else '0')

        # 如果 AI 开关变了，重新初始化
        if self.ai_var.get() and self.checker.ai is None:
            self._init_ai()
        elif not self.ai_var.get() and self.checker.ai:
            self.checker.ai = None
            self.lbl_ai_status.config(text="AI: 已关闭", foreground='#888')

        messagebox.showinfo("保存", "参数已保存到 config.ini")

    # ── 拖拽 ──
    def _bind_drop(self):
        if self.has_dnd:
            try:
                from tkinterdnd2 import DND_FILES
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind('<<Drop>>', self._on_drop)
            except Exception:
                pass
        self.root.bind('<Control-o>', lambda e: self._browse())

    def _on_drop(self, event):
        path = event.data.strip('{}').strip()
        if os.path.isdir(path):
            self.folder_var.set(path)
            self.cfg.set('last_folder', path)

    # ── 操作 ──
    def _browse(self):
        path = filedialog.askdirectory(title="选择照片文件夹")
        if path:
            self.folder_var.set(path)
            self.cfg.set('last_folder', path)

    def _start(self):
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("提示", "请先选择照片文件夹")
            return

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
        files = [f for f in sorted(os.listdir(folder)) if Path(f).suffix.lower() in exts]
        if not files:
            messagebox.showinfo("提示", "该文件夹中没有图片文件")
            return

        self.tree.delete(*self.tree.get_children())
        self.tree_dup.delete(*self.tree_dup.get_children())
        self.results.clear(); self.bad_photos.clear()
        self.duplicate_groups.clear()
        self.progress['value'] = 0; self.progress['maximum'] = len(files)
        self.processing = True
        self._stop_flag = False

        self.btn_scan.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.lbl_progress.config(text="扫描中...")

        threading.Thread(target=self._process, args=(folder, files), daemon=True).start()

    def _stop(self):
        self._stop_flag = True
        self.lbl_progress.config(text="正在停止...")

    def _process(self, folder, files):
        total = len(files)
        hash_map = {}

        for idx, fname in enumerate(files):
            if self._stop_flag:
                break
            fp = os.path.join(folder, fname)
            r = self.checker.check_single(fp)
            self.results.append(r)

            if r['issues']:
                self.bad_photos.append(r)
                for iss in r['issues']:
                    self.root.after(0, lambda fn=fname, t=iss['type'], c=iss['conf'],
                                    d=iss.get('detail', ''), aes=r['aesthetic'], fq=r['face_quality']:
                        self.tree.insert('', 'end', values=(
                            fn, t, f"{c}%", d, str(aes) if aes > 0 else '-',
                            f"{fq:.0f}" if fq < 100 else '-')))

            ph = r.get('phash')
            if ph:
                hash_map.setdefault(ph, []).append((idx, fname))

            self.root.after(0, lambda i=idx+1, t=total: self._update_pb(i, t))

        # 重复检测
        threshold = self.cfg.get_int('duplicate_threshold')
        hashes = list(hash_map.keys())
        seen = set()
        for i in range(len(hashes)):
            if i in seen: continue
            grp = [hashes[i]]
            for j in range(i + 1, len(hashes)):
                if j in seen: continue
                dist = PhotoChecker.hamming_distance(hashes[i], hashes[j])
                if dist <= (64 - threshold * 64 / 100):
                    grp.append(hashes[j]); seen.add(j)
            if len(grp) > 1:
                names = []
                for h in grp: names.extend(fn for _, fn in hash_map[h])
                self.duplicate_groups.append(names)
                self.root.after(0, lambda n=names: self.tree_dup.insert('', 'end', values=(', '.join(n[:4]),)))

        self.root.after(0, self._on_done)

    def _update_pb(self, cur, total):
        self.progress['value'] = cur
        self.lbl_progress.config(text=f"处理中：{cur}/{total} ({int(cur/total*100)}%)")
        self.lbl_stats.config(text=f"废片：{len(self.bad_photos)}  |  总计：{total}")

    def _on_done(self):
        self.processing = False
        self.btn_scan.config(state='normal')
        self.btn_stop.config(state='disabled')
        bad, dup, total = len(self.bad_photos), len(self.duplicate_groups), len(self.results)
        stopped = "已停止 · " if self._stop_flag else ""
        self.lbl_progress.config(text=stopped + "扫描完成！")
        parts = [f"废片：{bad}"]
        if dup: parts.append(f"重复组：{dup}")
        parts.append(f"总计：{total}")
        self.lbl_stats.config(text="  |  ".join(parts))
        if dup: self.status.config(text=f"🔁 {dup} 组重复照片")

    def _preview(self, event):
        sel = self.tree.selection()
        if not sel or not HAS_PIL: return
        idx = self.tree.index(sel[0])
        if idx >= len(self.results): return
        fname = self.results[idx]['file']
        fp = os.path.join(self.folder_var.get().strip(), fname)
        if not os.path.exists(fp): return
        try:
            img = Image.open(fp)
            img.thumbnail((300, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.lbl_preview.config(image=photo, text='')
            self.lbl_preview.image = photo
        except Exception as e:
            self.lbl_preview.config(image='', text=f"预览失败\n{e}")

    def _move_bad(self):
        folder = self.folder_var.get().strip()
        if not self.bad_photos:
            messagebox.showinfo("提示", "没有需要移动的废片"); return
        bad_folder = os.path.join(folder, "_废片")
        os.makedirs(bad_folder, exist_ok=True)
        moved, self.move_history = 0, []
        for r in self.bad_photos:
            src = os.path.join(folder, r['file'])
            dst = os.path.join(bad_folder, r['file'])
            try:
                if os.path.exists(src):
                    base, ext = os.path.splitext(r['file']); c = 1
                    while os.path.exists(dst):
                        dst = os.path.join(bad_folder, f"{base}_{c}{ext}"); c += 1
                    shutil.move(src, dst)
                    self.move_history.append((src, dst)); moved += 1
            except Exception as e:
                print(f"移动失败: {r['file']}: {e}")
        messagebox.showinfo("完成", f"已移动 {moved} 张废片到：\n{bad_folder}")
        self.status.config(text=f"已移动 {moved} 张 | {bad_folder}")

    def _undo(self):
        if not self.move_history:
            messagebox.showinfo("提示", "没有可撤销的移动操作"); return
        restored = 0
        for src, dst in self.move_history:
            try:
                if os.path.exists(dst): shutil.move(dst, src); restored += 1
            except Exception as e:
                print(f"撤销失败: {e}")
        self.move_history.clear()
        messagebox.showinfo("完成", f"已恢复 {restored} 张照片到原位置")
        self.status.config(text=f"已撤销 {restored} 张")

    def _export_csv(self):
        if not self.results:
            messagebox.showinfo("提示", "没有检测结果可导出"); return
        folder = self.folder_var.get().strip()
        name = f"检测报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(initialdir=folder, initialfile=name,
                                             defaultextension='.csv', filetypes=[("CSV", "*.csv")])
        if not path: return
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['文件名', '状态', '问题类型', '置信度', '详情', '美学分', '人脸质量分'])
            for r in self.results:
                aes = str(r.get('aesthetic', 0)) if r.get('aesthetic', 0) > 0 else ''
                fq = f"{r.get('face_quality', 100):.0f}" if r.get('face_quality', 100) < 100 else ''
                if r['issues']:
                    for iss in r['issues']:
                        w.writerow([r['file'], '废片', iss['type'], f"{iss['conf']}%", iss.get('detail', ''), aes, fq])
                else:
                    w.writerow([r['file'], '正常', '', '', '', aes, fq])
        messagebox.showinfo("导出完成", f"报告已保存到：\n{path}")


def main():
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()