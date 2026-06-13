"""
照片废片检测工具 — 本地离线运行
检测三类废片：模糊 / 人物闭眼 / 曝光异常
技术栈：OpenCV + MediaPipe + tkinter
"""

import os
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
import numpy as np

# ═══════════════════════════════════════════
# 检测引擎
# ═══════════════════════════════════════════

class PhotoChecker:
    """照片质量检测器 — 纯离线，无需额外模型文件"""

    def __init__(self):
        self.blur_threshold = 100       # 拉普拉斯方差阈值，低于此值=模糊
        self.overexposure_pct = 80      # 过曝像素百分比阈值
        self.underexposure_pct = 18     # 欠曝像素百分比阈值
        self.over_brightness = 240      # 过曝亮度阈值
        self.under_brightness = 30      # 欠曝亮度阈值

        # 加载 OpenCV 预训练级联分类器（随 opencv-python 自带）
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_frontalface_default.xml'))
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_path, 'haarcascade_eye.xml'))

    def _check_blur(self, gray):
        """检测模糊：拉普拉斯方差法"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance < self.blur_threshold, round(variance, 1)

    def _check_eyes(self, gray):
        """
        检测闭眼：Haar 级联人脸+人眼检测
        策略：检测到人脸后，在人脸上半区搜索眼睛
              如果眼睛检测数 < 2，判定为闭眼/眨眼
        """
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            return None, "未检测到人脸"

        closed_count = 0
        total_faces = min(len(faces), 3)  # 最多检测3张脸

        for (fx, fy, fw, fh) in faces[:total_faces]:
            # 人脸上半区域（眼睛所在位置）
            face_upper = gray[fy:fy + fh // 2, fx:fx + fw]

            eyes = self.eye_cascade.detectMultiScale(
                face_upper, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20))

            if len(eyes) < 2:
                closed_count += 1

        # 只要有一张脸检测到闭眼就标记
        eye_closed = closed_count > 0
        eye_count = len(faces)
        return eye_closed, f"检测{eye_count}张脸, {closed_count}张闭眼"

    def _check_exposure(self, gray):
        """检测曝光异常：直方图分析"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total = gray.size

        over_count = np.sum(hist[self.over_brightness:])
        under_count = np.sum(hist[:self.under_brightness])

        over_ratio = (over_count / total) * 100
        under_ratio = (under_count / total) * 100

        if over_ratio > self.overexposure_pct:
            return True, f"过曝 ({over_ratio:.0f}% 像素过亮)"
        if under_ratio > self.underexposure_pct:
            return True, f"欠曝 ({under_ratio:.0f}% 像素过暗)"

        return False, f"正常"

    def check_single(self, filepath):
        """检测单张照片，返回 (is_bad, reasons_list)"""
        reasons = []

        # 使用 np.fromfile + imdecode 兼容中文路径
        img_array = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return True, ["无法读取图片（格式不支持或文件损坏）"]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 如果图片太大，缩小加速处理（保持精度）
        h, w = gray.shape
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        # 1. 模糊检测
        is_blur, blur_val = self._check_blur(gray)
        if is_blur:
            reasons.append(f"模糊 (方差={blur_val})")

        # 2. 闭眼检测
        eye_result = self._check_eyes(gray)
        if eye_result[0] is None:
            pass  # 无人脸 → 不检查闭眼
        elif eye_result[0]:
            reasons.append(f"闭眼 ({eye_result[1]})")

        # 3. 曝光检测
        is_bad_exp, exp_msg = self._check_exposure(gray)
        if is_bad_exp:
            reasons.append(exp_msg)

        return len(reasons) > 0, reasons


# ═══════════════════════════════════════════
# GUI 界面
# ═══════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("照片废片检测工具")
        self.root.geometry("900x680")
        self.root.minsize(780, 560)
        self.root.configure(bg="#f0f2f5")

        self.folder_path = tk.StringVar()
        self.bad_photos = []       # [(filename, reasons_str)]
        self.checker = PhotoChecker()
        self.processing = False

        self._build_ui()

    def _build_ui(self):
        # ── 样式 ──
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f2f5")
        style.configure("TLabel", background="#f0f2f5", font=("微软雅黑", 10))
        style.configure("Title.TLabel", font=("微软雅黑", 18, "bold"), foreground="#1a1a2e")
        style.configure("Status.TLabel", font=("微软雅黑", 9), foreground="#6b7280")
        style.configure("Primary.TButton", font=("微软雅黑", 11, "bold"), padding=8)
        style.configure("TProgressbar", thickness=20)

        # 顶部标题栏
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=30, pady=(24, 0))

        ttk.Label(header, text="照片废片检测工具", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="本地离线 · AI 驱动 · 隐私安全",
                  style="Status.TLabel").pack(side="right", pady=(12, 0))

        # 说明行
        ttk.Label(self.root, text="自动检测模糊、人物闭眼、曝光异常三类废片",
                  style="Status.TLabel").pack(anchor="w", padx=30, pady=(4, 16))

        # ── 文件夹选择区 ──
        select_frame = ttk.Frame(self.root)
        select_frame.pack(fill="x", padx=30)

        ttk.Label(select_frame, text="照片文件夹：").pack(side="left")

        self.folder_entry = ttk.Entry(select_frame, textvariable=self.folder_path,
                                       font=("微软雅黑", 10), width=50)
        self.folder_entry.pack(side="left", padx=(8, 8), fill="x", expand=True)

        self.btn_browse = ttk.Button(select_frame, text="选择文件夹",
                                      command=self._browse_folder)
        self.btn_browse.pack(side="left", padx=2)

        # ── 操作按钮 + 统计 ──
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill="x", padx=30, pady=(12, 8))

        self.btn_start = ttk.Button(action_frame, text="开始检测",
                                     style="Primary.TButton",
                                     command=self._start_check)
        self.btn_start.pack(side="left")

        self.btn_move = ttk.Button(action_frame, text="一键移走废片",
                                    style="Primary.TButton",
                                    command=self._move_bad_photos,
                                    state="disabled")
        self.btn_move.pack(side="left", padx=(12, 0))

        # 统计标签
        self.lbl_stats = ttk.Label(action_frame, text="",
                                    font=("微软雅黑", 10, "bold"),
                                    foreground="#6b7280")
        self.lbl_stats.pack(side="right")

        # ── 进度条 ──
        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=30, pady=(0, 8))

        self.lbl_progress = ttk.Label(self.root, text="就绪，等待选择文件夹...",
                                       style="Status.TLabel")
        self.lbl_progress.pack(anchor="w", padx=30)

        # ── 结果列表 ──
        list_frame = ttk.Frame(self.root)
        list_frame.pack(fill="both", expand=True, padx=30, pady=(8, 16))

        columns = ("filename", "reasons")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings",
                                  selectmode="extended", height=14)
        self.tree.heading("filename", text="文件名")
        self.tree.heading("reasons", text="检测结果（原因）")
        self.tree.column("filename", width=280, minwidth=150)
        self.tree.column("reasons", width=540, minwidth=300)

        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        # 底部状态栏
        self.status_bar = ttk.Label(self.root, text="v2.0 | OpenCV Haar Cascade | 纯离线 · 隐私安全",
                                     relief="sunken", font=("微软雅黑", 8),
                                     foreground="#9ca3af", background="#e5e7eb")
        self.status_bar.pack(side="bottom", fill="x")

    def _browse_folder(self):
        path = filedialog.askdirectory(title="选择照片文件夹")
        if path:
            self.folder_path.set(path)
            self.lbl_progress.config(text=f"已选：{path}")

    def _start_check(self):
        folder = self.folder_path.get().strip()
        if not folder:
            messagebox.showwarning("提示", "请先选择照片文件夹")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("错误", "文件夹路径无效")
            return

        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
        files = []
        for f in sorted(os.listdir(folder)):
            if Path(f).suffix.lower() in exts:
                files.append(f)

        if not files:
            messagebox.showinfo("提示", "该文件夹中没有图片文件")
            return

        self.tree.delete(*self.tree.get_children())
        self.bad_photos.clear()
        self.progress["value"] = 0
        self.progress["maximum"] = len(files)
        self.btn_start.config(state="disabled")
        self.btn_move.config(state="disabled")
        self.btn_browse.config(state="disabled")
        self.processing = True

        thread = threading.Thread(target=self._process_images,
                                  args=(folder, files), daemon=True)
        thread.start()

    def _process_images(self, folder, files):
        total = len(files)
        bad_count = 0

        for idx, fname in enumerate(files):
            filepath = os.path.join(folder, fname)
            is_bad, reasons = self.checker.check_single(filepath)

            if is_bad:
                bad_count += 1
                reason_str = " | ".join(reasons) if reasons else "未知原因"
                self.bad_photos.append((fname, reason_str))
                self.root.after(0, lambda fn=fname, rs=reason_str: self._add_result(fn, rs))

            checked = idx + 1
            self.root.after(0, lambda c=checked, b=bad_count:
                            self._update_progress(c, total, b))

        self.root.after(0, self._on_complete)

    def _add_result(self, filename, reasons):
        self.tree.insert("", "end", values=(filename, reasons))

    def _update_progress(self, current, total, bad_count):
        pct = int(current / total * 100)
        self.progress["value"] = current
        self.lbl_progress.config(
            text=f"处理中：{current}/{total} ({pct}%)  —  已发现 {bad_count} 张废片")
        self.lbl_stats.config(text=f"废片：{bad_count} / {total}", foreground="#e0483b")

    def _on_complete(self):
        self.processing = False
        self.btn_start.config(state="normal")
        self.btn_browse.config(state="normal")
        self.lbl_progress.config(text=f"检测完成！共 {len(self.bad_photos)} 张废片")

        if self.bad_photos:
            self.btn_move.config(state="normal")
            self.lbl_stats.config(
                text=f"废片：{len(self.bad_photos)} 张",
                foreground="#e0483b")
        else:
            self.lbl_stats.config(text="全部通过！", foreground="#10b981")

    def _move_bad_photos(self):
        folder = self.folder_path.get().strip()
        if not self.bad_photos:
            messagebox.showinfo("提示", "没有需要移动的废片")
            return

        bad_folder = os.path.join(folder, "_废片")
        os.makedirs(bad_folder, exist_ok=True)

        moved = 0
        errors = []
        for fname, _ in self.bad_photos:
            src = os.path.join(folder, fname)
            dst = os.path.join(bad_folder, fname)
            try:
                if os.path.exists(src):
                    base, ext = os.path.splitext(fname)
                    counter = 1
                    while os.path.exists(dst):
                        dst = os.path.join(bad_folder, f"{base}_{counter}{ext}")
                        counter += 1
                    shutil.move(src, dst)
                    moved += 1
            except Exception as e:
                errors.append(f"{fname}: {e}")

        msg = f"已移动 {moved} 张废片到：\n{bad_folder}"
        if errors:
            msg += f"\n\n失败 {len(errors)} 张"

        messagebox.showinfo("完成", msg)
        self.lbl_stats.config(text=f"已移走 {moved} 张废片", foreground="#6b7280")
        self.btn_move.config(state="disabled")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()