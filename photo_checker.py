"""
照片废片检测工具 v3.0
检测: 模糊 / 闭眼 / 曝光异常 / 重复照片
特性: 参数可配 · 深色主题 · 图片预览 · CSV报告 · 撤销移动 · 拖拽导入
技术栈: OpenCV + ttkbootstrap + Pillow
"""

import os, sys, shutil, csv, configparser, threading, hashlib, json
from datetime import datetime
from pathlib import Path
from io import BytesIO

import cv2, numpy as np

# ── GUI ──
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 高DPI适配
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# 可选主题库
try:
    import ttkbootstrap as tkb
    HAS_BOOTSTRAP = True
except ImportError:
    HAS_BOOTSTRAP = False

# 可选图片库
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ═══════════════════════════════════════════
# 配置管理
# ═══════════════════════════════════════════

class Config:
    """配置管理：读取/写入 config.ini"""

    DEFAULTS = {
        'blur_threshold':       '100',
        'overexposure_pct':     '80',
        'underexposure_pct':    '18',
        'over_brightness':      '240',
        'under_brightness':     '30',
        'duplicate_threshold':  '95',
        'max_image_dim':        '1200',
        'theme':                'darkly',
        'last_folder':          '',
    }

    def __init__(self, path=None):
        if path is None:
            # 始终在脚本所在目录
            base = getattr(sys, '_MEIPASS', '') or os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, 'config.ini')
        self.path = path
        self.cfg = configparser.ConfigParser()
        self.load()

    def load(self):
        self.cfg.read(self.path, encoding='utf-8')
        if 'settings' not in self.cfg:
            self.cfg['settings'] = {}

    def save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            self.cfg.write(f)

    def get(self, key):
        return self.cfg['settings'].get(key, self.DEFAULTS.get(key, ''))

    def get_float(self, key):
        return float(self.get(key))

    def get_int(self, key):
        return int(float(self.get(key)))

    def set(self, key, value):
        self.cfg['settings'][key] = str(value)
        self.save()


# ═══════════════════════════════════════════
# 检测引擎
# ═══════════════════════════════════════════

class PhotoChecker:
    """照片质量检测器"""

    def __init__(self, config: Config):
        self.cfg = config

        # 加载 Haar 级联
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_eye.xml'))

    # ── 安全读取图片（兼容中文路径）──
    @staticmethod
    def _imread(path):
        arr = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ── 模糊检测 ──
    def check_blur(self, gray):
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        threshold = self.cfg.get_float('blur_threshold')
        is_bad = variance < threshold
        conf = max(0, min(100, (1 - variance / threshold) * 100)) if threshold > 0 else 0
        return is_bad, round(variance, 1), round(conf, 1)

    # ── 闭眼检测 ──
    def check_eyes(self, gray):
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            return False, 0, "无人脸"

        closed = 0
        for (fx, fy, fw, fh) in faces[:3]:
            upper = gray[fy:fy + fh // 2, fx:fx + fw]
            eyes = self.eye_cascade.detectMultiScale(upper, 1.05, 4, minSize=(20, 20))
            if len(eyes) < 2:
                closed += 1

        is_bad = closed > 0
        conf = round(closed / min(len(faces), 3) * 100, 1)
        return is_bad, conf, f"{len(faces)}脸/{closed}闭眼"

    # ── 曝光检测 ──
    def check_exposure(self, gray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total = gray.size
        over_pct = np.sum(hist[self.cfg.get_int('over_brightness'):]) / total * 100
        under_pct = np.sum(hist[:self.cfg.get_int('under_brightness')]) / total * 100

        if over_pct > self.cfg.get_float('overexposure_pct'):
            return True, round(over_pct, 1), "过曝"
        if under_pct > self.cfg.get_float('underexposure_pct'):
            return True, round(under_pct, 1), "欠曝"
        return False, 0, "正常"

    # ── 感知哈希（pHash） ──
    @staticmethod
    def compute_phash(gray):
        """DCT-based perceptual hash"""
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        top = dct[:8, :8]
        avg = top.mean()
        bits = (top > avg).flatten()
        # 打包为 hex 字符串
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return format(h, '016x')

    @staticmethod
    def hamming_distance(h1, h2):
        """两个十六进制哈希的汉明距离"""
        n1 = int(h1, 16)
        n2 = int(h2, 16)
        return bin(n1 ^ n2).count('1')

    # ── 完整检测 ──
    def check_single(self, filepath):
        """返回 {"file":, "issues":[...], "hash":...}"""
        result = {"file": os.path.basename(filepath), "issues": [], "phash": None}

        img = self._imread(filepath)
        if img is None:
            result["issues"].append({"type": "读取失败", "conf": 100, "detail": "无法解码"})
            return result

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 缩放大图加速
        h, w = gray.shape
        max_dim = self.cfg.get_int('max_image_dim')
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        # 1. 模糊
        bad, val, conf = self.check_blur(gray)
        if bad:
            result["issues"].append({"type": "模糊", "conf": conf, "detail": f"方差={val}"})

        # 2. 闭眼
        bad, conf, detail = self.check_eyes(gray)
        if bad:
            result["issues"].append({"type": "闭眼", "conf": conf, "detail": detail})

        # 3. 曝光
        bad, conf, detail = self.check_exposure(gray)
        if bad:
            result["issues"].append({"type": "曝光", "conf": conf, "detail": detail})

        # 4. pHash（始终计算，用于重复检测）
        try:
            small = cv2.resize(gray, (128, 128))
            result["phash"] = self.compute_phash(small)
        except Exception:
            result["phash"] = None

        return result


# ═══════════════════════════════════════════
# 主界面
# ═══════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("照片废片检测工具 v3.0")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        self.cfg = Config()
        self.checker = PhotoChecker(self.cfg)
        self.results = []          # 完整检测结果
        self.bad_photos = []       # 废片列表
        self.duplicate_groups = [] # 重复组
        self.move_history = []     # 撤销用: [(src, dst), ...]
        self.processing = False

        self.has_dnd = self._check_drag_support()
        self._build_ui()
        self._load_params_to_ui()

    # ── UI 构建 ──
    def _build_ui(self):
        root = self.root
        if HAS_BOOTSTRAP:
            self.style = tkb.Style(theme=self.cfg.get('theme') or 'darkly')
        else:
            self.style = ttk.Style()
            try: self.style.theme_use('clam')
            except: pass
            root.configure(bg='#1c1c1c')
            self.style.configure('.', background='#1c1c1c', foreground='#e0e0e0')
            self.style.configure('TFrame', background='#1c1c1c')
            self.style.configure('TLabel', background='#1c1c1c', foreground='#e0e0e0')
            self.style.configure('TLabelframe', background='#1c1c1c', foreground='#e0e0e0')
            self.style.configure('TButton', background='#333', foreground='#e0e0e0')

        # ── 顶部工具栏 ──
        toolbar = ttk.Frame(root)
        toolbar.pack(fill='x', padx=10, pady=(8, 0))

        ttk.Label(toolbar, text="照片文件夹：").pack(side='left')
        self.folder_var = tk.StringVar(value=self.cfg.get('last_folder'))
        self.entry_folder = ttk.Entry(toolbar, textvariable=self.folder_var, width=50)
        self.entry_folder.pack(side='left', padx=6, fill='x', expand=True)
        ttk.Button(toolbar, text="选择", command=self._browse).pack(side='left', padx=2)
        ttk.Button(toolbar, text="扫描", command=self._start).pack(side='left', padx=(8, 0))
        ttk.Button(toolbar, text="移走废片", command=self._move_bad).pack(side='left', padx=4)
        ttk.Button(toolbar, text="撤销", command=self._undo).pack(side='left', padx=4)
        ttk.Button(toolbar, text="导出CSV", command=self._export_csv).pack(side='left', padx=4)

        # ── 参数面板（可折叠） ──
        self.params_visible = tk.BooleanVar(value=False)
        self.btn_params = ttk.Button(toolbar, text="⚙ 参数", command=self._toggle_params)
        self.btn_params.pack(side='right', padx=4)

        self.params_frame = ttk.LabelFrame(root, text="检测参数", padding=8)

        params = [
            ("模糊阈值 (越低越严)", "blur_threshold", 20, 300, 5),
            ("过曝判定 (%过亮像素)", "overexposure_pct", 30, 100, 5),
            ("欠曝判定 (%过暗像素)", "underexposure_pct", 5, 50, 1),
            ("过亮亮度值", "over_brightness", 200, 255, 5),
            ("过暗亮度值", "under_brightness", 5, 80, 5),
            ("重复相似度 %", "duplicate_threshold", 80, 100, 1),
        ]
        self.sliders = {}
        for i, (label, key, lo, hi, step) in enumerate(params):
            col = i % 3; row = i // 3
            f = ttk.Frame(self.params_frame)
            f.grid(row=row, column=col, padx=10, pady=4, sticky='ew')
            ttk.Label(f, text=label, font=('微软雅黑', 9)).pack(anchor='w')
            sv = tk.IntVar(value=self.cfg.get_int(key))
            s = ttk.Scale(f, from_=lo, to=hi, variable=sv,
                          command=lambda v, k=key: self._on_slider(k, v))
            s.pack(fill='x')
            sv.trace_add('write', lambda *a, k=key, v=sv: self._on_slider_change(k, v))
            self.sliders[key] = (sv, s)

        # 重置按钮行
        bf = ttk.Frame(self.params_frame)
        bf.grid(row=2, column=0, columnspan=3, pady=(8, 0), sticky='w')
        ttk.Button(bf, text="恢复默认", command=self._reset_params).pack(side='left')
        ttk.Button(bf, text="保存参数", command=self._save_params).pack(side='left', padx=8)

        for i in range(3):
            self.params_frame.columnconfigure(i, weight=1)

        # ── 统计栏 ──
        stats_frame = ttk.Frame(root)
        stats_frame.pack(fill='x', padx=10, pady=(6, 0))
        self.lbl_stats = ttk.Label(stats_frame, text="就绪", font=('微软雅黑', 10))
        self.lbl_stats.pack(side='left')
        self.lbl_dup = ttk.Label(stats_frame, text="", foreground='#ff9500')
        self.lbl_dup.pack(side='right')

        # ── 进度条 ──
        self.progress = ttk.Progressbar(root, mode='determinate')
        self.progress.pack(fill='x', padx=10, pady=2)
        self.lbl_progress = ttk.Label(root, text="")
        self.lbl_progress.pack(anchor='w', padx=14)

        # ── 主内容区（表格 + 预览） ──
        main = ttk.PanedWindow(root, orient='horizontal')
        main.pack(fill='both', expand=True, padx=10, pady=(4, 10))

        # 左侧表格
        table_frame = ttk.Frame(main)
        main.add(table_frame, weight=3)

        columns = ("filename", "issue", "conf", "detail")
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings',
                                  selectmode='browse')
        self.tree.heading("filename", text="文件名")
        self.tree.heading("issue", text="问题")
        self.tree.heading("conf", text="置信度")
        self.tree.heading("detail", text="详情")
        self.tree.column("filename", width=200, minwidth=120)
        self.tree.column("issue", width=80, minwidth=60)
        self.tree.column("conf", width=70, minwidth=50)
        self.tree.column("detail", width=240, minwidth=120)

        tree_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

        # 右侧预览 + 重复组
        right_frame = ttk.Frame(main)
        main.add(right_frame, weight=1)

        # 预览
        preview_lf = ttk.LabelFrame(right_frame, text="图片预览", padding=4)
        preview_lf.pack(fill='both', expand=True)
        self.lbl_preview = ttk.Label(preview_lf, text="点击列表项预览图片", anchor='center')
        self.lbl_preview.pack(fill='both', expand=True)

        # 重复组列表
        dup_lf = ttk.LabelFrame(right_frame, text="重复照片组", padding=4)
        dup_lf.pack(fill='x', pady=(4, 0))
        self.tree_dup = ttk.Treeview(dup_lf, columns=("files",), show='headings', height=4)
        self.tree_dup.heading("files", text="相似文件组")
        self.tree_dup.column("files", width=100)
        self.tree_dup.pack(fill='x')

        # ── 状态栏 ──
        self.status = ttk.Label(root, text="v3.0 | OpenCV pHash | 纯离线", relief='sunken',
                                font=('微软雅黑', 8))
        self.status.pack(side='bottom', fill='x')

        # ── 拖拽提示 ──
        self.drop_label = ttk.Label(root, text="💡 支持拖拽文件夹到此处", foreground='#666')
        self.drop_label.pack(side='bottom', pady=2)
        self._bind_drop()

    # ── 参数面板 ──
    def _toggle_params(self):
        if self.params_visible.get():
            self.params_frame.pack_forget()
            self.params_visible.set(False)
        else:
            self.params_frame.pack(fill='x', padx=10, pady=(6, 0), after=self.btn_params.master)
            self.params_visible.set(True)

    def _load_params_to_ui(self):
        for key, (sv, _) in self.sliders.items():
            sv.set(self.cfg.get_int(key))

    def _on_slider(self, key, val):
        pass  # 实时预览暂不触发

    def _on_slider_change(self, key, sv):
        pass  # trace 回调

    def _save_params(self):
        for key, (sv, _) in self.sliders.items():
            self.cfg.set(key, sv.get())
        messagebox.showinfo("保存", "参数已保存到 config.ini")

    def _reset_params(self):
        for key, (sv, _) in self.sliders.items():
            sv.set(int(Config.DEFAULTS[key]))
        self._save_params()

    # ── 拖拽支持 ──
    def _check_drag_support(self):
        try:
            import tkinterdnd2
            return True
        except ImportError:
            return False

    def _bind_drop(self):
        if self.has_dnd:
            try:
                from tkinterdnd2 import DND_FILES
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind('<<Drop>>', self._on_drop)
            except Exception:
                pass
        # Fallback: 双击空白区域选择文件夹
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
            messagebox.showwarning("提示", "请先选择有效的照片文件夹")
            return

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
        files = [f for f in sorted(os.listdir(folder)) if Path(f).suffix.lower() in exts]
        if not files:
            messagebox.showinfo("提示", "该文件夹中没有图片文件")
            return

        # 清空
        self.tree.delete(*self.tree.get_children())
        self.tree_dup.delete(*self.tree_dup.get_children())
        self.results.clear()
        self.bad_photos.clear()
        self.duplicate_groups.clear()
        self.progress['value'] = 0
        self.progress['maximum'] = len(files)
        self.processing = True

        t = threading.Thread(target=self._process, args=(folder, files), daemon=True)
        t.start()

    def _process(self, folder, files):
        total = len(files)
        hash_map = {}  # phash → [(idx, filename)]

        for idx, fname in enumerate(files):
            fp = os.path.join(folder, fname)
            r = self.checker.check_single(fp)
            self.results.append(r)

            # 废片
            if r['issues']:
                self.bad_photos.append(r)
                for iss in r['issues']:
                    self.root.after(0, lambda fn=fname, t=iss['type'], c=iss['conf'], d=iss['detail']:
                                    self.tree.insert('', 'end', values=(fn, t, f"{c}%", d)))

            # pHash 去重
            ph = r.get('phash')
            if ph:
                if ph not in hash_map:
                    hash_map[ph] = []
                hash_map[ph].append((idx, fname))

            self.root.after(0, lambda i=idx+1, t=total: self._update_progress(i, t))

        # 检测重复组
        threshold = self.cfg.get_int('duplicate_threshold')
        hashes = list(hash_map.keys())
        seen = set()
        for i in range(len(hashes)):
            if i in seen: continue
            group = [hashes[i]]
            for j in range(i + 1, len(hashes)):
                if j in seen: continue
                dist = PhotoChecker.hamming_distance(hashes[i], hashes[j])
                if dist <= (64 - threshold * 64 / 100):  # 汉明距离转相似度
                    group.append(hashes[j])
                    seen.add(j)
            if len(group) > 1:
                files_in_group = []
                for h in group:
                    files_in_group.extend([fn for _, fn in hash_map[h]])
                self.duplicate_groups.append(files_in_group)
                self.root.after(0, lambda g=files_in_group:
                                self.tree_dup.insert('', 'end', values=(', '.join(g[:4]),)))

        self.root.after(0, self._on_done)

    def _update_progress(self, cur, total):
        self.progress['value'] = cur
        pct = int(cur / total * 100)
        self.lbl_progress.config(text=f"处理中：{cur}/{total} ({pct}%)")
        bad = len(self.bad_photos)
        self.lbl_stats.config(text=f"废片：{bad}  |  总计：{total}")

    def _on_done(self):
        self.processing = False
        bad = len(self.bad_photos)
        dup = len(self.duplicate_groups)
        total = len(self.results)
        self.lbl_progress.config(text=f"扫描完成！")
        parts = [f"废片：{bad}"]
        if dup: parts.append(f"重复组：{dup}")
        parts.append(f"总计：{total}")
        self.lbl_stats.config(text="  |  ".join(parts))
        if dup:
            self.lbl_dup.config(text=f"🔁 {dup} 组重复照片")

    def _on_tree_select(self, event):
        """点击列表项预览图片"""
        sel = self.tree.selection()
        if not sel or not HAS_PIL: return
        idx = self.tree.index(sel[0])
        if idx >= len(self.results): return
        fname = self.results[idx]['file']
        folder = self.folder_var.get().strip()
        fp = os.path.join(folder, fname)
        if not os.path.exists(fp): return
        try:
            img = Image.open(fp)
            img.thumbnail((280, 280), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.lbl_preview.config(image=photo, text='')
            self.lbl_preview.image = photo
        except Exception as e:
            self.lbl_preview.config(image='', text=f"预览失败\n{e}")

    def _move_bad(self):
        folder = self.folder_var.get().strip()
        if not self.bad_photos:
            messagebox.showinfo("提示", "没有需要移动的废片")
            return

        bad_folder = os.path.join(folder, "_废片")
        os.makedirs(bad_folder, exist_ok=True)

        moved, self.move_history = 0, []
        for r in self.bad_photos:
            src = os.path.join(folder, r['file'])
            dst = os.path.join(bad_folder, r['file'])
            try:
                if os.path.exists(src):
                    # 重名处理
                    base, ext = os.path.splitext(r['file'])
                    c = 1
                    while os.path.exists(dst):
                        dst = os.path.join(bad_folder, f"{base}_{c}{ext}")
                        c += 1
                    shutil.move(src, dst)
                    self.move_history.append((src, dst))
                    moved += 1
            except Exception as e:
                print(f"移动失败: {r['file']}: {e}")

        messagebox.showinfo("完成", f"已移动 {moved} 张废片到：\n{bad_folder}")
        self.status.config(text=f"已移动 {moved} 张 | {bad_folder}")

    def _undo(self):
        if not self.move_history:
            messagebox.showinfo("提示", "没有可撤销的移动操作")
            return
        restored = 0
        for src, dst in self.move_history:
            try:
                if os.path.exists(dst):
                    shutil.move(dst, src)
                    restored += 1
            except Exception as e:
                print(f"撤销失败: {e}")
        self.move_history.clear()
        messagebox.showinfo("完成", f"已恢复 {restored} 张照片到原位置")
        self.status.config(text=f"已撤销 {restored} 张")

    def _export_csv(self):
        if not self.results:
            messagebox.showinfo("提示", "没有检测结果可导出")
            return

        folder = self.folder_var.get().strip()
        default_name = f"检测报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            initialdir=folder, initialfile=default_name,
            defaultextension='.csv', filetypes=[("CSV", "*.csv")])
        if not path: return

        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['文件名', '状态', '问题类型', '置信度', '详情'])
            for r in self.results:
                if r['issues']:
                    for iss in r['issues']:
                        w.writerow([r['file'], '废片', iss['type'], f"{iss['conf']}%", iss['detail']])
                else:
                    w.writerow([r['file'], '正常', '', '', ''])

        messagebox.showinfo("导出完成", f"报告已保存到：\n{path}")


# ═══════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════

def main():
    # 尝试使用 tkinterdnd2 作为根窗口
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()

    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()