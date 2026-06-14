"""
图片智能分析系统 - 电竞暗黑美学
SCRFD人脸 · 多维评分 · 场景分类 · Qwen2.5-VL · 废片归档 · 审核日志
"""
import os, sys, shutil, csv, configparser, threading, time, logging, json

# ═══════════════════════════════════════════
VERSION = "5.0"
# ═══════════════════════════════════════════
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(2)
except: pass
try: from ctypes import windll; windll.shcore.SetProcessDpiAwarenessMode(1)
except: pass
try: from PIL import Image, ImageTk, ExifTags; HAS_PIL = True
except: HAS_PIL = False
from ai_detector import AIDetector
from ollama_vl import QwenVL, QWEN_MODELS, _check_ollama_running, _has_model, _pull_model
from inspectors import ImgStats

# ─── 日志 ───
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger('photo_checker')

# ─── 配色令牌 ───
BG0    = '#08080f'  # 最深底
BG1    = '#0a0a14'  # 卡片/树
BG2    = '#0f0f1a'  # 悬浮
BG3    = '#161625'  # 按钮
TEXT1  = '#e8e8f0'  # 主文字
TEXT2  = '#9090a8'  # 次要
ACCENT = '#ff6b35'  # 橙色强调
CYAN   = '#00d4ff'  # 青色
PURPLE = '#a855f7'  # 紫色
GREEN  = '#4cd964'  # 绿色
RED    = '#ff375f'  # 红色
GOLD   = '#ff9500'

CONFIG_DEFAULTS = {
    # ── 检测阈值 ──
    'blur_threshold':'80','overexposure_pct':'80','underexposure_pct':'18',
    'over_brightness':'240','under_brightness':'30','duplicate_threshold':'95',
    'max_image_dim':'1200','aesthetic_min':'3.0',
    # ── AI 开关 ──
    'enable_ai':'1','ai_mode':'','qwen_model':'','qwen_deep_scan':'0','first_run':'1',
    # ── Ollama ──
    'ollama_host':'http://127.0.0.1:11434',
    # ── 下载源 ──
    'scrfd_url':'https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_person_2.5g.onnx',
    'scrfd_hf_url':'https://huggingface.co/deepinsight/scrfd/resolve/main/scrfd_2.5g_bnkps.onnx',
    'scrfd_model_name':'scrfd_2.5g_bnkps.onnx',
    # ── 网络 ──
    'ollama_check_timeout':'2','ollama_check_retry_timeout':'3',
    'model_download_timeout':'180','model_download_retry_wait':'3',
    'ollama_api_timeout':'120','ollama_pull_timeout':'7200',
    # ── v5.0 新增 ──
    'outlier_zscore_thresh':'2.5',
}

class Config:
    def __init__(self):
        base = getattr(sys,'_MEIPASS','') or os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(base,'config.ini')
        self._models_dir = os.path.join(base, 'models')
        self.cfg = configparser.ConfigParser()
        self.cfg.read(self.path, encoding='utf-8')
        if 'settings' not in self.cfg: self.cfg['settings'] = {}
    def save(self):
        with open(self.path,'w',encoding='utf-8') as f: self.cfg.write(f)
    def g(self,k): return self.cfg['settings'].get(k,CONFIG_DEFAULTS.get(k,''))
    def gf(self,k): return float(self.g(k))
    def gi(self,k): return int(float(self.g(k)))
    def gb(self,k): return self.g(k)=='1'
    def s(self,k,v): self.cfg['settings'][k]=str(v); self.save()

    # ── 便捷方法：路径 ──
    @property
    def models_dir(self): return self._models_dir
    def scrfd_path(self): return os.path.join(self._models_dir, self.g('scrfd_model_name'))
    def scrfd_urls(self): return [self.g('scrfd_url'), self.g('scrfd_hf_url')]

    # ── 便捷方法：网络超时 ──
    def ollama_check_timeout(self): return float(self.g('ollama_check_timeout'))
    def ollama_check_retry_timeout(self): return float(self.g('ollama_check_retry_timeout'))
    def model_download_timeout(self): return float(self.g('model_download_timeout'))
    def model_download_retry_wait(self): return float(self.g('model_download_retry_wait'))
    def ollama_api_timeout(self): return float(self.g('ollama_api_timeout'))
    def ollama_pull_timeout(self): return float(self.g('ollama_pull_timeout'))
    def outlier_zscore_thresh(self): return float(self.g('outlier_zscore_thresh'))

class PhotoChecker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ai = None
        self.qwen = None
        self.errors = []  # 扫描过程中收集的错误记录
    def init_ai(self, cb=None):
        if self.ai is None and self.cfg.gb('enable_ai'):
            self.ai = AIDetector(enable_ai=True, progress_cb=cb, cfg=self.cfg)
        # 初始化 Qwen-VL（如果有配置）
        qwen_model = self.cfg.g('qwen_model')
        ollama_host = self.cfg.g('ollama_host')
        if qwen_model and not self.qwen:
            self.qwen = QwenVL(host=ollama_host, model=qwen_model, progress_cb=cb)
            if self.qwen.available:
                if cb: cb("Qwen2.5-VL 就绪")
    @staticmethod
    def _imread(p):
        """安全读取图片文件，异常时返回 None 而非抛异常
        注意：PermissionError 会重新抛出，由调用方分类处理。
        """
        try:
            a = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(a, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            # cv2.imdecode 返回 None 说明是损坏的文件
            raise ValueError("cv2.imdecode 返回 None（文件可能已损坏）")
        except PermissionError:
            raise
        except OSError:
            raise
        except Exception:
            return None
    @staticmethod
    def _apply_exif_rotation(img_bgr):
        """根据 EXIF orientation 修正照片旋转方向（手机拍摄常见90°旋转）"""
        if not HAS_PIL: return img_bgr
        try:
            pil_img = Image.open(img_bgr)  # 传入路径
        except Exception:
            return img_bgr
        try:
            exif = pil_img.getexif()
            if not exif: return img_bgr
            orientation = exif.get(274)  # ExifTags.Orientation = 274
            if not orientation: return img_bgr
            angle_map = {1:0, 3:180, 6:270, 8:90}
            if orientation in angle_map:
                pil_img = pil_img.rotate(angle_map[orientation], expand=True)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        except Exception:
            return img_bgr
    @staticmethod
    def compute_phash(gray):
        r=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA)
        d=cv2.dct(np.float32(r))[:8,:8]; h=0
        for b in (d>d.mean()).flatten(): h=(h<<1)|int(b)
        return format(h,'016x')
    @staticmethod
    def hamming(a,b): return bin(int(a,16)^int(b,16)).count('1')

    def check_single(self, path):
        r={'file':os.path.basename(path),'issues':[],'phash':None,
           'aesthetic':0,'overall':0,'scene':'','suggestions':[],'face_count':0,'face_results':[],
           '_error':None}  # _error: 记录错误原因（用于扫描汇总）
        try:
            img=self._imread(path)
        except PermissionError:
            logger.warning(f"权限不足 {path}")
            r['_error'] = {'reason': '权限不足', 'detail': f'无法读取文件，可能被其他程序占用或需要管理员权限', 'file': os.path.basename(path)}
            return r
        except Exception as e:
            logger.warning(f"读取失败 {path}: {e}")
            r['_error'] = {'reason': '读取失败', 'detail': f'无法解码: {e}', 'file': os.path.basename(path)}
            r['issues'].append({'t':'读取失败','c':100,'d':'无法解码'})
            return r
        if img is None:
            r['_error'] = {'reason': '读取失败', 'detail': '无法解码（文件可能已损坏）', 'file': os.path.basename(path)}
            r['issues'].append({'t':'读取失败','c':100,'d':'无法解码'})
            return r

        # EXIF 方向修正
        try:
            img = self._apply_exif_rotation(img)
        except Exception as e:
            logger.warning(f"EXIF 旋转失败 {path}: {e}")
            r['_error'] = {'reason': 'EXIF 错误', 'detail': f'EXIF 旋转失败: {e}', 'file': os.path.basename(path)}
            r['issues'].append({'t':'读取失败','c':100,'d':'EXIF 旋转失败'})
            return r
        if img is None:
            r['_error'] = {'reason': 'EXIF 错误', 'detail': 'EXIF 旋转后图像为空', 'file': os.path.basename(path)}
            r['issues'].append({'t':'读取失败','c':100,'d':'EXIF 旋转失败'})
            return r

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w=gray.shape; md=self.cfg.gi('max_image_dim')
        if max(h,w)>md:
            s=md/max(h,w); gray=cv2.resize(gray,(int(w*s),int(h*s)))
            rgb=cv2.resize(rgb,(int(w*s),int(h*s)))

        # ── 模块化检测管线（inspectors.py）──
        issues_from_inspectors = []
        if self.ai:
            try:
                stats = ImgStats(
                    width=gray.shape[1], height=gray.shape[0],
                    mean_brightness=float(np.mean(gray)),
                    std_brightness=float(np.std(gray)),
                    mean_saturation=float(np.mean(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:,:,1])),
                )
                issues_from_inspectors = self.ai.detect_issues(rgb, gray, stats)
            except Exception as e:
                logger.warning(f"检测管线失败 {path}: {e}")
        else:
            # 非 AI 模式也要收集 stats
            stats = ImgStats(
                width=gray.shape[1], height=gray.shape[0],
                mean_brightness=float(np.mean(gray)),
                std_brightness=float(np.std(gray)),
                mean_saturation=float(np.mean(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:,:,1])),
            )

        # 合并 inspector 问题
        for iss in issues_from_inspectors:
            r['issues'].append(iss)
        lap=cv2.Laplacian(gray,cv2.CV_64F).var(); bt=self.cfg.gf('blur_threshold')
        if lap<bt:
            bc='失焦模糊' if lap<30 else ('运动模糊' if lap<60 else '')
            r['issues'].append({'t':'模糊','c':round(max(0,(1-lap/bt)*100),1),'d':f"方差={lap:.0f} {bc}"})
        hist=cv2.calcHist([gray],[0],None,[256],[0,256]); tot=gray.size
        over=np.sum(hist[self.cfg.gi('over_brightness'):])/tot*100
        under=np.sum(hist[:self.cfg.gi('under_brightness')])/tot*100
        if over>self.cfg.gf('overexposure_pct'):
            r['issues'].append({'t':'过曝','c':round(over,1),'d':f"{over:.0f}%过亮"})
        if under>self.cfg.gf('underexposure_pct'):
            r['issues'].append({'t':'欠曝','c':round(under,1),'d':f"{under:.0f}%过暗"})

        if self.ai:
            try:
                fd=self.ai.detect_faces(rgb); r['face_count']=len(fd) if fd else 0
            except Exception as e:
                logger.warning(f"人脸检测失败 {path}: {e}")
                r['face_count']=0; fd=None
            if fd:
                for f in fd[:10]:
                    fq=self.ai.analyze_face(f); r['face_results'].append(fq)
                    if fq['eyes_open']<40: r['issues'].append({'t':'闭眼','c':round(100-fq['eyes_open'],1),'d':f"睁眼度{fq['eyes_open']:.0f}%"})
                    if fq['head_angle']>30: r['issues'].append({'t':'侧脸','c':round(fq['head_angle'],1),'d':f"偏转{fq['head_angle']:.0f}度"})
            r['scene']=self.ai.classify_scene(rgb,r['face_count'])
            try:
                q=self.ai.analyze_quality(rgb,gray,r['face_results'])
                r['overall']=q.get('overall',0); r['aesthetic']=q.get('aesthetic',0)
                r['grade']=q.get('grade',''); r['quality']=q
                r['suggestions']=self.ai.suggest(q,r['scene'],r['face_count'])
            except Exception as e:
                logger.warning(f"质量分析失败 {path}: {e}")
                r['overall']=0; r['aesthetic']=0; r['grade']=''; r['quality']={}
                r['suggestions']=['质量分析失败，请检查模型']
            if r['overall']<55: r['issues'].append({'t':'综合评分低','c':round(55-r['overall'],1),'d':f"得分{r['overall']:.0f}/100"})
        else:
            try:
                fc=cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,'haarcascade_frontalface_default.xml'))
                ec=cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,'haarcascade_eye.xml'))
                faces=fc.detectMultiScale(gray,1.1,5,minSize=(80,80)); r['face_count']=len(faces)
                if len(faces)>0:
                    r['scene']='人像写真' if len(faces)<3 else '集体照'; closed=0
                    for (fx,fy,fw,fh) in faces[:3]:
                        e=ec.detectMultiScale(gray[fy:fy+fh//2,fx:fx+fw],1.05,4,minSize=(20,20))
                        if len(e)<2: closed+=1
                    if closed>0: r['issues'].append({'t':'闭眼','c':round(closed/min(len(faces),3)*100,1),'d':f"{closed}/{min(len(faces),3)}"})
            except Exception as e:
                logger.warning(f"标准检测失败 {path}: {e}")
                r['face_count']=0
        try:
            s=cv2.resize(gray,(128,128)); r['phash']=self.compute_phash(s)
        except Exception as e:
            logger.warning(f"pHash 计算失败 {path}: {e}")
            r['phash']=None
        return r

    def check_single_qwen(self, path, progress_cb=None):
        """使用 Qwen2.5-VL 深度分析单张图片（需要图片路径 + 已有分析结果）"""
        if not self.qwen or not self.qwen.available:
            return None
        try:
            r = self.qwen.analyze(path, progress_cb=progress_cb)
            if r and 'overall_score' in r:
                r['_should_remove'] = r.get('should_remove', False)
            return r
        except Exception as e:
            logger.warning(f"Qwen 分析失败 {path}: {e}")
            return None

    def generate_audit_log(self, folder, results, qwen_results, removed_files, moved_count):
        """生成 AI 审核日志"""
        log = {
            'version': VERSION,
            'generated_at': datetime.now().isoformat(),
            'folder': folder,
            'qwen_model': self.cfg.g('qwen_model'),
            'total_photos': len(results),
            'bad_photos': len(removed_files),
            'removed_files': removed_files,
            'details': [],
        }
        for i, r in enumerate(results):
            entry = {
                'file': r['file'],
                'issues': [iss['t'] for iss in r['issues']],
                'overall_score': r.get('overall', 0),
                'ai_qwen': qwen_results[i] if i < len(qwen_results) else None,
                'removed': r['file'] in removed_files,
            }
            log['details'].append(entry)

        log_path = os.path.join(folder, 'AI审核日志.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        return log_path

# ═══════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root=root; self.root.title(f"图片智能分析系统 v{VERSION}")
        self.root.geometry("1200x800"); self.root.minsize(1000,640)
        self.root.configure(bg=BG0)
        self.cfg=Config(); self.checker=PhotoChecker(self.cfg)
        self.results=[]; self.bad_photos=[]; self.dup_groups=[]; self.move_hist=[]
        self.processing=False; self._stop=False; self._paused=False
        self._paused_idx=0; self._paused_files=[]; self._paused_folder=''; self._paused_hm={}
        self._show_params=False
        self._data_lock=threading.Lock()  # 保护 results/bad_photos 的线程锁
        self._scan_start_time=None  # 扫描开始时间，用于剩余时间估算
        self._scanned_count=0  # 已扫描计数
        self._sort_key='overall'  # 当前排序列
        self._sort_reverse=True  # 是否降序
        self._filter_mode='all'  # 筛选模式: 'all'/'bad'/'good'
        self._errors=[]  # 记录扫描失败的文件 {file, reason}
        self.fv=tk.StringVar()
        self._setup_style()
        self._build()
        self.root.after(600,self._init_ai)

    def _setup_style(self):
        self.style=ttk.Style()
        try: self.style.theme_use('clam')
        except: pass
        self.style.configure('.', background=BG0, foreground=TEXT1, fieldbackground=BG1, troughcolor=BG0,
                            selectbackground=ACCENT, selectforeground='#fff', borderwidth=0)
        self.style.configure('TFrame', background=BG0)
        self.style.configure('TLabel', background=BG0, foreground=TEXT1)
        self.style.configure('TLabelframe', background=BG0, foreground=TEXT2)
        self.style.configure('TButton', background=BG3, foreground=TEXT1, borderwidth=1, relief='solid')
        self.style.map('TButton', background=[('active',ACCENT),('disabled',BG1)],
                      foreground=[('active','#fff'),('disabled','#505060')])
        self.style.configure('Treeview', background=BG1, foreground=TEXT1, fieldbackground=BG1, rowheight=28)
        self.style.map('Treeview', background=[('selected','#1a1040')], foreground=[('selected',ACCENT)])
        self.style.configure('TProgressbar', background=ACCENT, troughcolor=BG1)
        self.style.configure('TPanedwindow', background=BG0)
        self.style.configure('TSeparator', background='#2a2a3a')
        self.style.configure('TScale', background=BG0, troughcolor=BG1)
        self.style.configure('TCheckbutton', background=BG0, foreground=TEXT1)

    def _build(self):
        r=self.root
        # ── 工具栏 ──
        tb=tk.Frame(r,bg=BG0); tb.pack(fill='x',padx=8,pady=(6,0))
        tk.Label(tb,text="图片智能分析系统",font=('微软雅黑',14,'bold'),fg=ACCENT,bg=BG0).pack(side='left')
        ttk.Separator(tb,orient='vertical').pack(side='left',padx=10,fill='y')
        ttk.Label(tb,text="文件夹:").pack(side='left')
        tk.Entry(tb,textvariable=self.fv,width=40,bg=BG1,fg=TEXT1,insertbackground=TEXT1,
                relief='flat',font=('微软雅黑',10)).pack(side='left',padx=6)
        ttk.Button(tb,text="选择",command=self._browse).pack(side='left')
        ttk.Separator(tb,orient='vertical').pack(side='left',padx=6,fill='y')
        self.bs=ttk.Button(tb,text="▶ 扫描",command=self._start); self.bs.pack(side='left',padx=2)
        self.bpz=ttk.Button(tb,text="⏸ 暂停",command=self._pause_scan,state='disabled'); self.bpz.pack(side='left',padx=2)
        self.bre=ttk.Button(tb,text="⏵ 继续",command=self._resume_scan,state='disabled'); self.bre.pack(side='left',padx=2)
        self.brs=ttk.Button(tb,text="🔄 重扫",command=self._re_scan,state='disabled'); self.brs.pack(side='left',padx=2)
        self.bdel=ttk.Button(tb,text="🗑 移动废片",command=self._move, state='disabled')
        self.bdel.pack(side='left',padx=2)
        ttk.Separator(tb,orient='vertical').pack(side='left',padx=6,fill='y')
        # 汉堡菜单（合并所有功能）
        self.btn_menu=tk.Menubutton(tb,text="⚙ 菜单 ▾",font=('微软雅黑',10),bg=BG3,fg=TEXT1,
                                     activebackground=ACCENT,activeforeground='#fff',
                                     relief='flat',bd=0,padx=12,pady=4)
        self.btn_menu.pack(side='right',padx=4)
        menu=tk.Menu(self.btn_menu,tearoff=0,bg=BG2,fg=TEXT1,activebackground='#1a1040',
                     activeforeground=ACCENT,font=('微软雅黑',10))
        menu.add_command(label="📄 导出CSV",command=self._csv)
        menu.add_command(label="🔄 排序/过滤",command=self._show_settings)
        menu.add_separator()
        menu.add_command(label="🤖 AI模型选择",command=self._show_ai_settings)
        menu.add_separator()
        menu.add_command(label="⚙ 参数设置",command=self._toggle_params)
        menu.add_separator()
        menu.add_command(label="📋 失败记录",command=self._view_errors)
        menu.add_separator()
        menu.add_command(label="ℹ 关于",command=self._about)
        self.btn_menu.config(menu=menu)

        # 参数侧滑面板
        self.param_slide=tk.Frame(r,bg=BG2)
        self._build_param_panel()

        # 统计+进度
        sf=tk.Frame(r,bg=BG0); sf.pack(fill='x',padx=8,pady=(6,0))
        self.lbl_stats=tk.Label(sf,text="就绪",font=('微软雅黑',10),fg=TEXT2,bg=BG0)
        self.lbl_stats.pack(side='left')
        self.lbl_ai=tk.Label(sf,text="AI: 初始化中...",fg=TEXT2,bg=BG0)
        self.lbl_ai.pack(side='right')
        self.pb=ttk.Progressbar(r,mode='determinate'); self.pb.pack(fill='x',padx=8,pady=2)
        self.lbl_pb=tk.Label(r,text="",font=('微软雅黑',9),fg=TEXT2,bg=BG0)
        self.lbl_pb.pack(anchor='w',padx=12)

        # 主区域
        main=ttk.PanedWindow(r,orient='horizontal'); main.pack(fill='both',expand=True,padx=8,pady=(4,8))

        # 左侧表格(70%)
        tf=tk.Frame(main,bg=BG0); main.add(tf,weight=7)
        cols=("fn","issue","conf","score")
        self.tree=ttk.Treeview(tf,columns=cols,show='headings',selectmode='browse',height=18)
        self.tree.heading("fn",text="文件名"); self.tree.heading("issue",text="问题")
        self.tree.heading("conf",text="置信度"); self.tree.heading("score",text="综合分")
        self.tree.column("fn",width=260,minwidth=150); self.tree.column("issue",width=140,minwidth=80)
        self.tree.column("conf",width=80,minwidth=60); self.tree.column("score",width=80,minwidth=60)
        ts=tk.Scrollbar(tf,bg=BG1,troughcolor=BG0,command=self.tree.yview)
        self.tree.configure(yscrollcommand=ts.set)
        self.tree.pack(side='left',fill='both',expand=True); ts.pack(side='right',fill='y')
        self.tree.bind('<<TreeviewSelect>>',self._preview)
        self.tree.bind('<Double-1>',self._full_report)

        # 右侧面板(30%)
        rf=tk.Frame(main,bg=BG0); main.add(rf,weight=3)
        plf=tk.LabelFrame(rf,text="预览",fg=TEXT2,bg=BG0,font=('微软雅黑',10)); plf.pack(fill='both',expand=True)
        self.preview_box=tk.Frame(plf,bg='#000000'); self.preview_box.pack(fill='both',expand=True)
        self.preview_label=tk.Label(self.preview_box,text="单击预览 · 双击分析报告",font=('微软雅黑',11),
                                     fg='#555',bg='#000000')
        self.preview_label.pack(expand=True)
        self._preview_img=None
        self.lbl_exif=tk.Label(rf,text="",font=('微软雅黑',8),fg=TEXT2,bg=BG0); self.lbl_exif.pack(fill='x',pady=(2,0))

        dlf=tk.LabelFrame(rf,text="重复组",fg=TEXT2,bg=BG0,font=('微软雅黑',10)); dlf.pack(fill='x',pady=(4,0))
        self.tdup=ttk.Treeview(dlf,columns=("f",),show='headings',height=3)
        self.tdup.heading("f",text="双击查看详情"); self.tdup.column("f",width=200)
        self.tdup.pack(fill='x')
        self.tdup.bind('<Double-1>',self._dup_detail)

        self.st=tk.Label(r,text=f"{VERSION} | SCRFD+Qwen-VL+Inspectors | 失败自动跳过 | AI仅供参考",font=('微软雅黑',8),fg=TEXT2,bg=BG0,anchor='w')
        self.st.pack(side='bottom',fill='x',padx=8)

    def _build_param_panel(self):
        pf=self.param_slide
        for w in pf.winfo_children(): w.destroy()
        hdr=tk.Frame(pf,bg=BG2); hdr.pack(fill='x',padx=12,pady=(10,4))
        tk.Label(hdr,text="⚙ 检测参数",font=('微软雅黑',13,'bold'),fg=TEXT1,bg=BG2).pack(side='left')
        tk.Button(hdr,text="✕",font=('微软雅黑',11),bg=BG2,fg=TEXT2,bd=0,command=self._hide_params).pack(side='right')
        af=tk.Frame(pf,bg=BG2); af.pack(fill='x',padx=12,pady=4)
        self.ai_var=tk.BooleanVar(value=self.cfg.gb('enable_ai'))
        tk.Checkbutton(af,text="🤖 AI智能分析",variable=self.ai_var,command=self._on_ai_toggle,
                        bg=BG2,fg=TEXT1,selectcolor=BG0,activebackground=BG2,activeforeground=TEXT1,
                        font=('微软雅黑',10)).pack(side='left')
        pd=[("模糊阈值","blur_threshold",20,300),("过曝判定%","overexposure_pct",30,100),
            ("欠曝判定%","underexposure_pct",5,50),("过亮值","over_brightness",200,255),
            ("过暗值","under_brightness",5,80),("重复相似度%","duplicate_threshold",80,100),
            ("美学最低分","aesthetic_min",1.0,9.0),
            ("Ollama API超时(s)","ollama_api_timeout",10,600)]
        self.sls={}
        for lb,k,lo,hi in pd:
            f=tk.Frame(pf,bg=BG2); f.pack(fill='x',padx=12,pady=3)
            tk.Label(f,text=lb,font=('微软雅黑',9),fg=TEXT2,bg=BG2).pack(anchor='w')
            vf=tk.Frame(f,bg=BG2); vf.pack(fill='x')
            tk.Label(vf,text=str(lo),font=('微软雅黑',7),fg='#666',bg=BG2).pack(side='left')
            fl=isinstance(lo,float); sv=tk.DoubleVar(value=self.cfg.gf(k)) if fl else tk.IntVar(value=self.cfg.gi(k))
            vl=tk.Label(vf,text="",font=('微软雅黑',8,'bold'),fg=GOLD,bg=BG2,width=6); vl.pack(side='right')
            tk.Label(vf,text=str(hi),font=('微软雅黑',7),fg='#666',bg=BG2).pack(side='right')
            sv.trace_add('write',lambda *a,l=vl,f=fl,s=sv: l.config(text=f"{float(s.get()):.1f}" if f else f"{int(s.get())}"))
            tk.Scale(vf,from_=lo,to=hi,variable=sv,orient='horizontal',bg=BG2,fg=TEXT1,
                     troughcolor=BG3,highlightthickness=0,bd=0).pack(fill='x')
            self.sls[k]=(sv,fl); vl.config(text=f"{float(sv.get()):.1f}" if fl else f"{int(sv.get())}")
        bf=tk.Frame(pf,bg=BG2); bf.pack(fill='x',padx=12,pady=(8,12))
        tk.Button(bf,text="恢复默认",bg=BG3,fg=TEXT1,command=self._rp).pack(side='left')
        self._saved_lbl=tk.Label(bf,text="",fg=GREEN,bg=BG2,font=('微软雅黑',9))
        self._saved_lbl.pack(side='left',padx=12)
        tk.Button(bf,text="保存参数",bg=ACCENT,fg='#fff',command=self._sp).pack(side='right')
        # 点击外部关闭
        pf.bind('<Leave>',lambda e: None)

    def _bind_events(self):
        self.root.bind('<Button-1>',self._click_outside)
        self.root.protocol('WM_DELETE_WINDOW',lambda: (self.cfg.s('last_folder',''), self.root.destroy()))

    def _click_outside(self,event):
        if self._show_params:
            w=self.param_slide
            if not (w.winfo_rootx()<=event.x_root<=w.winfo_rootx()+w.winfo_width() and
                    w.winfo_rooty()<=event.y_root<=w.winfo_rooty()+w.winfo_height()):
                self._hide_params()

    # ═══════════ 参数 ═══════════
    def _toggle_params(self):
        if self._show_params: self._hide_params()
        else: self.param_slide.place(relx=1.0,rely=0.0,anchor='ne',width=380,relheight=1.0); self.param_slide.lift(); self._show_params=True
    def _hide_params(self): self.param_slide.place_forget(); self._show_params=False
    def _rp(self):
        for k,(sv,fl) in self.sls.items():
            d=float(CONFIG_DEFAULTS.get(k,0)); sv.set(int(d) if not fl else d)
        self._sp()
    def _sp(self):
        for k,(sv,fl) in self.sls.items(): self.cfg.s(k,str(int(sv.get())) if not fl else f"{sv.get():.1f}")
        self.cfg.s('enable_ai','1' if self.ai_var.get() else '0')
        self._saved_lbl.config(text="✓ 已保存"); self.root.after(2000,lambda: self._saved_lbl.config(text=""))

    # ═══════════ AI ═══════════
    def _init_ai(self):
        self.lbl_ai.config(text="AI: 初始化中...", fg=TEXT2)
        self.root.update_idletasks()

        # 提前检查模型文件，如果不存在则弹窗提示
        if not os.path.exists(self.cfg.scrfd_path()):
            self.root.after(0, self._show_model_missing_dialog)

        self.checker.init_ai(cb=lambda m: self.root.after(0, lambda msg=m: self._aip(msg)))
        parts = []
        if self.checker.ai and self.checker.ai.face_detector:
            parts.append("SCRFD")
        if self.checker.qwen and self.checker.qwen.available:
            parts.append("Qwen-VL")
        status = ' + '.join(parts) if parts else '标准模式'
        fg = GREEN if parts else TEXT2
        self.lbl_ai.config(text=f"AI: {status} ✓", fg=fg)

    def _show_model_missing_dialog(self):
        """弹窗提示用户：模型文件缺失，是否一键下载"""
        msg = (
            f"未检测到 SCRFD 人脸检测模型（{self.cfg.g('scrfd_model_name')}，约3.6MB）。\n\n"
            f"AI 人脸检测、闭眼判断等功能需要此模型。\n\n"
            f"是否一键下载？"
        )
        ok = messagebox.askyesno(
            "模型文件缺失", msg, parent=self.root
        )
        if ok:
            self.lbl_ai.config(text="AI: 正在下载模型...", fg=GOLD)
            self.root.update_idletasks()
            # 在后台线程下载，避免阻塞
            threading.Thread(
                target=self._download_model, daemon=True
            ).start()

    def _download_model(self):
        """后台线程下载模型"""
        urls = self.cfg.scrfd_urls()
        path = self.cfg.scrfd_path()
        timeout = self.cfg.model_download_timeout()
        def cb(msg):
            self.root.after(0, lambda m=msg: self.lbl_ai.config(text=f"AI: {m}", fg=GOLD))
        ok = ensure_model(urls, path, max_retries=3, cb=cb, timeout=timeout)
        self.root.after(0, lambda: self._download_done(ok))

    def _download_done(self, ok):
        if ok:
            self.lbl_ai.config(text="AI: 模型下载完成，重新初始化...", fg=GOLD)
            self.root.update_idletasks()
            self._init_ai()  # 递归重新初始化
        else:
            self.lbl_ai.config(text="AI: 标准模式（模型下载失败）", fg=TEXT2)
            messagebox.showwarning(
                "下载失败",
                "模型下载失败，已降级到标准检测模式。\n"
                "可稍后在 AI 设置中手动下载。",
                parent=self.root
            )
        # 模型下载完成（无论成功失败），启动欢迎页
        self.root.after(100, self._show_welcome)

    def _show_welcome(self):
        """欢迎页 + 新手引导"""
        if self.cfg.g('first_run') == '0':
            return

        # 用 Toplevel 做覆盖层，不破坏主界面布局
        dlg = tk.Toplevel(self.root)
        dlg.title("")
        dlg.geometry("420x340")
        dlg.configure(bg=BG0)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)
        dlg.focus_set()

        tk.Label(dlg, text="图片智能分析系统",
                 font=('微软雅黑', 22, 'bold'), fg=ACCENT, bg=BG0).pack(pady=(30, 6))
        tk.Label(dlg, text="电竞暗黑美学 · 本地离线 · 隐私优先",
                 font=('微软雅黑', 10), fg=TEXT2, bg=BG0).pack(pady=(0, 16))

        if not os.path.exists(self.cfg.scrfd_path()):
            tk.Label(dlg, text="⏳ 正在下载 AI 模型...",
                     font=('微软雅黑', 11), fg=GOLD, bg=BG0).pack()
            dlg.after(2000, self._show_welcome)
            return

        tk.Label(dlg, text="✅ AI 模型就绪",
                 font=('微软雅黑', 11), fg=GREEN, bg=BG0).pack(pady=(0, 16))

        tk.Button(dlg, text="开始使用", font=('微软雅黑', 12, 'bold'),
                  bg=ACCENT, fg='#fff', padx=30, pady=10,
                  command=lambda: (dlg.destroy(), self._start_onboarding())).pack()

    def _start_onboarding(self):
        """新手引导：3步流程"""
        steps = [
            {
                'icon': '📂',
                'title': '选择文件夹',
                'desc': '点击下方"选择"按钮，\n选择你要分析的照片文件夹。',
                'widget': self.fv,
                'tooltip': '点击这里选择照片文件夹',
            },
            {
                'icon': '▶️',
                'title': '开始扫描',
                'desc': '点击工具栏"▶ 扫描"按钮，\n系统将自动检测每张照片的质量。',
                'widget': self.bs,
                'tooltip': '点击这里开始扫描',
            },
            {
                'icon': '🗑',
                'title': '查看废片',
                'desc': '扫描完成后点击"🗑 移动废片"，\n不合格的照片会移入 _废片/ 文件夹。',
                'widget': self.bdel,
                'tooltip': '点击这里移动废片',
            },
        ]
        self._onboard_steps = steps
        self._onboard_show_step(0)

    def _onboard_show_step(self, idx):
        """显示引导单步"""
        s = self._onboard_steps[idx]

        dlg = tk.Toplevel(self.root)
        dlg.geometry("400x280")
        dlg.configure(bg=BG0)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)
        dlg.focus_set()

        tk.Label(dlg, text=f"{s['icon']}  步骤 {idx+1}/3：{s['title']}",
                 font=('微软雅黑', 14, 'bold'), fg=TEXT1, bg=BG0).pack(pady=(16, 4))
        tk.Label(dlg, text=s['desc'],
                 font=('微软雅黑', 10), fg=TEXT2, bg=BG0, justify='center').pack(pady=(0, 8))

        # 指示器
        dot_frame = tk.Frame(dlg, bg=BG0)
        dot_frame.pack(pady=4)
        for i in range(3):
            c = ACCENT if i == idx else '#333'
            sz = 14 if i == idx else 10
            tk.Label(dot_frame, text='●', font=('微软雅黑', sz), fg=c, bg=BG0, padx=4).pack(side='left')

        btn_frame = tk.Frame(dlg, bg=BG0)
        btn_frame.pack(pady=(8, 12))
        if idx < 2:
            tk.Button(btn_frame, text="下一步 →", font=('微软雅黑', 11, 'bold'),
                      bg=ACCENT, fg='#fff', padx=16, pady=6,
                      command=lambda i=idx: (dlg.destroy(), self._onboard_show_step(i + 1))).pack(side='left', padx=8)
        tk.Button(btn_frame, text="跳过引导", font=('微软雅黑', 9),
                  bg=BG3, fg=TEXT2, padx=12, pady=5,
                  command=lambda: (dlg.destroy(), self._finish_onboarding())).pack(side='left', padx=8)

        # 高亮目标控件
        dlg.after(200, self._onboard_flash, s['widget'], s['tooltip'])

    def _onboard_flash(self, widget, tooltip_text):
        """高亮引导目标控件"""
        try:
            x = widget.winfo_rootx()
            y = widget.winfo_rooty() - 40
        except Exception:
            return

        tip = tk.Label(self.root, text=tooltip_text, font=('微软雅黑', 10),
                       bg=ACCENT, fg='#fff', padx=12, pady=6, relief='raised', bd=1)
        tip.place(x=max(10, x - 40), y=max(10, y), anchor='sw')

        orig_bg = widget.cget('bg')
        orig_fg = widget.cget('fg')
        cnt = [0]

        def _flash():
            c = ACCENT if cnt[0] % 2 == 0 else orig_bg
            f = '#fff' if cnt[0] % 2 == 0 else orig_fg
            try:
                widget.config(bg=c, fg=f)
            except Exception:
                pass
            cnt[0] += 1
            if cnt[0] < 6:
                self.root.after(400, _flash)
            else:
                try:
                    widget.config(bg=orig_bg, fg=orig_fg)
                except Exception:
                    pass
                tip.destroy()

        _flash()

    def _finish_onboarding(self):
        """完成引导，保存状态"""
        self.cfg.s('first_run', '0')
        self.st.config(text="引导完成！随时可以点击菜单查看帮助")
        self.root.after(3000, lambda: self.st.config(text=f"{VERSION} | AI仅供参考"))

    def _aip(self,msg): self.lbl_ai.config(text=f"AI: {msg}"); self.root.update_idletasks()

    # ═══════════ Qwen-VL ═══════════
    def _enable_qwen_model(self, model_key):
        """启用指定的 Qwen-VL 模型"""
        host = self.cfg.g('ollama_host')
        running, _ = _check_ollama_running(host, timeout=2)
        if not running:
            self._show_ollama_guide_dialog(host)
            return
        self.cfg.s('qwen_model', model_key)
        self.cfg.s('ollama_host', host)
        self.checker.qwen = QwenVL(host=host, model=model_key,
                                    progress_cb=lambda m: self.root.after(0,lambda msg=m:self._aip(msg)))
        if self.checker.qwen.available:
            self.lbl_ai.config(text="AI: Qwen-VL ✓", fg=GREEN)
            messagebox.showinfo("成功", f"Qwen2.5-VL {model_key.split(':')[0].replace('qwen2.5-vl-','')} 已启用")

    def _pull_and_enable_qwen(self, model_key):
        """下载并启用 Qwen-VL 模型"""
        host = self.cfg.g('ollama_host')
        def progress(msg): self.root.after(0,lambda m=msg:self._aip(m))
        ok = _pull_model(host, model_key, progress)
        if ok:
            self._enable_qwen_model(model_key)

    def _move(self):
        """将废片移动到 _废片/ 文件夹 + 生成审核日志"""
        f=self.fv.get().strip()
        if not self.bad_photos: self.st.config(text="无废片可移动"); return

        # 确定要移动的照片：传统检测标记的 + Qwen 建议移除的
        to_remove = set()
        for r in self.bad_photos:
            to_remove.add(r['file'])
        # 也包含 Qwen 建议移除的照片
        for r in self.results:
            if r.get('_should_remove') and r['file'] not in to_remove:
                to_remove.add(r['file'])

        if not to_remove:
            messagebox.showinfo("提示", "没有需要移除的照片")
            return

        moved, fail_count = 0, 0
        self.move_hist = []
        bd = os.path.join(f, "_废片")
        os.makedirs(bd, exist_ok=True)

        removed_files = []
        for r in self.results:
            if r['file'] in to_remove:
                src = os.path.join(f, r['file'])
                dst = os.path.join(bd, r['file'])
                try:
                    if os.path.exists(src):
                        b, e = os.path.splitext(r['file'])
                        c = 1
                        while os.path.exists(dst):
                            dst = os.path.join(bd, f"{b}_{c}{e}")
                            c += 1
                        shutil.move(src, dst)
                        self.move_hist.append((src, dst))
                        moved += 1
                        removed_files.append({
                            'file': r['file'],
                            'reasons': [iss['t'] for iss in r['issues']],
                            'overall_score': r.get('overall', 0),
                            'qwen_score': r.get('ai_qwen', {}).get('overall_score', None) if r.get('ai_qwen') else None,
                            'qwen_suggestions': r.get('ai_qwen', {}).get('suggestions', []) if r.get('ai_qwen') else [],
                        })
                except Exception as e:
                    logger.warning(f"移动失败 {r['file']}: {e}")
                    fail_count += 1

        # 生成审核日志
        log_path = self.checker.generate_audit_log(f, self.results,
            [r.get('ai_qwen') for r in self.results],
            [rf['file'] for rf in removed_files], moved)

        hint = f"✓ 已移动 {moved} 张到 _废片/"
        if fail_count: hint += f" ({fail_count}张失败)"
        hint += f" | 日志: {os.path.basename(log_path)}"
        self.st.config(text=hint)
        self.root.after(5000, lambda: self.st.config(text=f"{VERSION} | AI仅供参考"))

    def _undo(self):
        """撤销废片移动"""
        if not self.move_hist: self.st.config(text="无可撤销"); return
        restored, fail_count = 0, 0
        for s, d in self.move_hist:
            try:
                if os.path.exists(d):
                    shutil.move(d, s); restored += 1
            except Exception as e:
                logger.warning(f"撤销失败 {os.path.basename(d)}: {e}")
                fail_count += 1
        self.move_hist.clear()
        hint = f"✓ 已恢复 {restored} 张"
        if fail_count: hint += f" ({fail_count}张失败)"
        self.st.config(text=hint)
        self.root.after(3000, lambda: self.st.config(text=f"{VERSION} | AI仅供参考"))

    def _on_ai_toggle(self):
        if self.ai_var.get():
            host = self.cfg.g('ollama_host')
            running, _ = _check_ollama_running(host, timeout=2)
            if not running:
                self._show_ollama_guide_dialog(host)
                return
            if not self.cfg.g('qwen_model'):
                self._show_qwen_model_selection()
            else:
                self.cfg.s('enable_ai','1')
                self.cfg.s('ollama_host', host)
                self._init_ai()
        else: self.cfg.s('enable_ai','0')

    def _show_qwen_model_selection(self):
        """展示 Qwen2.5-VL 模型选择对话框（含 Ollama 部署引导）"""
        host = self.cfg.g('ollama_host')
        running, models = _check_ollama_running(host, timeout=2)

        if not running:
            self._show_ollama_guide_dialog(host)
            return

        dlg=tk.Toplevel(self.root); dlg.title("选择 Qwen2.5-VL 模型"); dlg.geometry("650x540")
        dlg.configure(bg=BG0); dlg.transient(self.root); dlg.grab_set(); dlg.resizable(False,False)
        tk.Label(dlg,text="选择 Qwen2.5-VL 本地视觉模型",font=('微软雅黑',16,'bold'),fg=TEXT1,bg=BG0).pack(pady=(12,4))
        tk.Label(dlg,text="所有分析均在本地完成，照片永不离开本机",font=('微软雅黑',9),fg=TEXT2,bg=BG0).pack()

        # Ollama Host 配置
        tf_host=tk.LabelFrame(dlg,text="Ollama 服务地址",fg=TEXT1,bg=BG1,font=('微软雅黑',10))
        tf_host.pack(fill='x',padx=16,pady=4)
        self._qwen_host_var = tk.StringVar(value=host)
        tk.Entry(tf_host,textvariable=self._qwen_host_var,width=50,font=('微软雅黑',10),
                 bg=BG2,fg=TEXT1,insertbackground=TEXT1,relief='flat').pack(padx=10,pady=4)

        # 实时 Ollama 状态指示
        self._ollama_status_lbl = tk.Label(dlg, text="✅ Ollama 服务运行中", font=('微软雅黑',9,'bold'),
                                           fg=GREEN, bg=BG0)
        self._ollama_status_lbl.pack(pady=(0,4))

        models_frame=tk.Frame(dlg,bg=BG0); models_frame.pack(fill='both',expand=True,padx=16,pady=8)

        for model_key, info in QWEN_MODELS.items():
            has = _has_model(host, model_key)
            icon = '✅' if has else '⬇️'
            lf=tk.LabelFrame(models_frame,text="",bg=BG1,fg=TEXT1); lf.pack(fill='x',padx=4,pady=4)

            hdr=tk.Frame(lf,bg=BG1); hdr.pack(fill='x',pady=(8,0),padx=10)
            tk.Label(hdr,text=f"{icon} {info['name']}",font=('微软雅黑',11,'bold'),fg=TEXT1,bg=BG1).pack(anchor='w')

            props=tk.Frame(lf,bg=BG1); props.pack(fill='x',padx=10,pady=4)
            cols_data = [('大小',info['size']),('显存',info['显存需求']),('速度',info['速度']),('精度',info['精度'])]
            for ci,(lb,v) in enumerate(cols_data):
                tk.Label(props,text=f"{lb}: {v}",font=('微软雅黑',9),fg=TEXT2,bg=BG1).pack(
                    side='left',anchor='w',padx=(0,16))

            bf=tk.Frame(lf,bg=BG1); bf.pack(fill='x',pady=(4,8),padx=10)
            if has:
                tk.Button(bf,text=f"使用 {info['name']}",bg=GREEN,fg='#fff',font=('微软雅黑',10,'bold'),padx=16,pady=4,
                          command=lambda mk=model_key: self._select_qwen_model(mk, dlg)).pack(side='left',padx=4)
            else:
                tk.Button(bf,text=f"下载并启用 {info['name']}",bg=ACCENT,fg='#fff',font=('微软雅黑',10,'bold'),padx=10,pady=4,
                          command=lambda mk=model_key: self._pull_and_select_qwen(mk, dlg)).pack(side='left',padx=4)

        tk.Label(dlg,text="AI模型管理器中可查看更详细信息",font=('微软雅黑',9),fg=TEXT2,bg=BG0).pack(pady=(0,8))

    def _show_ollama_guide_dialog(self, host):
        """Ollama 未运行时的友好引导窗口"""
        dlg=tk.Toplevel(self.root)
        dlg.title("安装 Ollama")
        dlg.geometry("480x400")
        dlg.configure(bg=BG0)
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg,text="📦 需要先安装 Ollama 服务",
                 font=('微软雅黑',14,'bold'),fg=TEXT1,bg=BG0).pack(pady=(16,4))
        tk.Label(dlg,text=(
            "Ollama 是本地 AI 模型运行环境，免费开源\n"
            "一键安装，全程自动"
        ),font=('微软雅黑',10),fg=TEXT2,bg=BG0).pack(pady=4)

        # 服务地址输入
        tf_addr=tk.LabelFrame(dlg,text="Ollama 服务地址",fg=TEXT1,bg=BG1,font=('微软雅黑',10))
        tf_addr.pack(fill='x',padx=24,pady=8)
        self._guide_host_var = tk.StringVar(value=host)
        tk.Entry(tf_addr,textvariable=self._guide_host_var,width=45,font=('微软雅黑',10),
                 bg=BG2,fg=TEXT1,insertbackground=TEXT1,relief='flat').pack(padx=10,pady=4)

        # 步骤1
        tk.Label(dlg,text="👉 步骤1：下载并安装 Ollama",
                 font=('微软雅黑',11,'bold'),fg=ACCENT,bg=BG0).pack(pady=(12,2))
        tk.Button(dlg,text="🌐 点击打开官网下载",font=('微软雅黑',10,'bold'),
                  bg=ACCENT,fg='#fff',padx=16,pady=6,
                  command=lambda: os.system(f'start http://ollama.com', shell=True)).pack(pady=2)

        # 步骤2
        tk.Label(dlg,text="👉 步骤2：安装完成后检测服务",
                 font=('微软雅黑',11,'bold'),fg=ACCENT,bg=BG0).pack(pady=(12,2))
        tk.Button(dlg,text="🔍 检测 Ollama 服务",font=('微软雅黑',10,'bold'),
                  bg=CYAN,fg='#fff',padx=16,pady=6,
                  command=self._guide_check_ollama).pack(pady=2)
        self._guide_result_lbl = tk.Label(dlg,text="",font=('微软雅黑',9),bg=BG0)
        self._guide_result_lbl.pack(pady=2)

        # 步骤3
        tk.Label(dlg,text="👉 步骤3：自动拉取 Qwen 模型",
                 font=('微软雅黑',11,'bold'),fg=ACCENT,bg=BG0).pack(pady=(12,2))
        tk.Label(dlg,text="（检测到服务后自动进入模型选择）",
                 font=('微软雅黑',9),fg=TEXT2,bg=BG0).pack()

        # 关闭按钮
        tk.Button(dlg,text="关闭",font=('微软雅黑',10),bg=BG3,fg=TEXT1,
                  padx=16,pady=6,command=dlg.destroy).pack(pady=(12,8))

    def _guide_check_ollama(self):
        """步骤2：检测 Ollama 服务"""
        host = self._guide_host_var.get() or self.cfg.g('ollama_host')
        self._guide_result_lbl.config(text="正在检测...", fg=GOLD)
        self.root.update_idletasks()

        running, models = _check_ollama_running(host, timeout=3)
        if running:
            self._guide_result_lbl.config(text="✅ 检测到 Ollama 服务！", fg=GREEN)
            self.root.after(500, lambda h=host: self._guide_proceed_to_models(h))
        else:
            self._guide_result_lbl.config(text="❌ 未检测到服务，请确认 Ollama 已启动", fg=RED)

    def _guide_proceed_to_models(self, host):
        """步骤2通过后，关闭引导窗口并进入模型选择"""
        self.cfg.s('ollama_host', host)
        # 关闭所有引导窗口
        for w in list(self.root.winfo_children()):
            if isinstance(w, tk.Toplevel):
                w.destroy()
        self._show_qwen_model_selection_inner(host)

    def _show_qwen_model_selection_inner(self, host):
        """模型选择窗口（不含 Ollama 引导）"""
        running, models = _check_ollama_running(host, timeout=2)
        if not running:
            self._show_ollama_guide_dialog(host)
            return

        dlg=tk.Toplevel(self.root); dlg.title("选择 Qwen2.5-VL 模型"); dlg.geometry("650x540")
        dlg.configure(bg=BG0); dlg.transient(self.root); dlg.grab_set(); dlg.resizable(False,False)
        tk.Label(dlg,text="选择 Qwen2.5-VL 本地视觉模型",font=('微软雅黑',16,'bold'),fg=TEXT1,bg=BG0).pack(pady=(12,4))
        tk.Label(dlg,text="所有分析均在本地完成，照片永不离开本机",font=('微软雅黑',9),fg=TEXT2,bg=BG0).pack()

        # Ollama Host 配置
        tf_host=tk.LabelFrame(dlg,text="Ollama 服务地址",fg=TEXT1,bg=BG1,font=('微软雅黑',10))
        tf_host.pack(fill='x',padx=16,pady=4)
        self._qwen_host_var = tk.StringVar(value=host)
        tk.Entry(tf_host,textvariable=self._qwen_host_var,width=50,font=('微软雅黑',10),
                 bg=BG2,fg=TEXT1,insertbackground=TEXT1,relief='flat').pack(padx=10,pady=4)

        # 实时 Ollama 状态指示
        self._ollama_status_lbl = tk.Label(dlg, text="✅ Ollama 服务运行中", font=('微软雅黑',9,'bold'),
                                           fg=GREEN, bg=BG0)
        self._ollama_status_lbl.pack(pady=(0,4))

        models_frame=tk.Frame(dlg,bg=BG0); models_frame.pack(fill='both',expand=True,padx=16,pady=8)

        for model_key, info in QWEN_MODELS.items():
            has = _has_model(host, model_key)
            icon = '✅' if has else '⬇️'
            lf=tk.LabelFrame(models_frame,text="",bg=BG1,fg=TEXT1); lf.pack(fill='x',padx=4,pady=4)

            hdr=tk.Frame(lf,bg=BG1); hdr.pack(fill='x',pady=(8,0),padx=10)
            tk.Label(hdr,text=f"{icon} {info['name']}",font=('微软雅黑',11,'bold'),fg=TEXT1,bg=BG1).pack(anchor='w')

            props=tk.Frame(lf,bg=BG1); props.pack(fill='x',padx=10,pady=4)
            cols_data = [('大小',info['size']),('显存',info['显存需求']),('速度',info['速度']),('精度',info['精度'])]
            for ci,(lb,v) in enumerate(cols_data):
                tk.Label(props,text=f"{lb}: {v}",font=('微软雅黑',9),fg=TEXT2,bg=BG1).pack(
                    side='left',anchor='w',padx=(0,16))

            bf=tk.Frame(lf,bg=BG1); bf.pack(fill='x',pady=(4,8),padx=10)
            if has:
                tk.Button(bf,text=f"使用 {info['name']}",bg=GREEN,fg='#fff',font=('微软雅黑',10,'bold'),padx=16,pady=4,
                          command=lambda mk=model_key: self._select_qwen_model(mk, dlg)).pack(side='left',padx=4)
            else:
                tk.Button(bf,text=f"下载并启用 {info['name']}",bg=ACCENT,fg='#fff',font=('微软雅黑',10,'bold'),padx=10,pady=4,
                          command=lambda mk=model_key: self._pull_and_select_qwen(mk, dlg)).pack(side='left',padx=4)

        tk.Label(dlg,text="AI模型管理器中可查看更详细信息",font=('微软雅黑',9),fg=TEXT2,bg=BG0).pack(pady=(0,8))

    def _select_qwen_model(self, model_key, dlg):
        """选择并启用 Qwen 模型（先检测 Ollama 服务）"""
        host = self._qwen_host_var.get() if hasattr(self, '_qwen_host_var') else self.cfg.g('ollama_host')
        running, _ = _check_ollama_running(host, timeout=2)
        if not running:
            if dlg: dlg.destroy()
            self._show_ollama_guide_dialog(host)
            return
        self.cfg.s('qwen_model', model_key)
        self.cfg.s('ollama_host', host)
        if dlg: dlg.destroy()
        self.cfg.s('enable_ai','1')
        self._init_ai()

    def _pull_and_select_qwen(self, model_key, dlg):
        """下载并选择 Qwen 模型（先检测 Ollama 服务）"""
        host = self._qwen_host_var.get() if hasattr(self, '_qwen_host_var') else self.cfg.g('ollama_host')
        running, _ = _check_ollama_running(host, timeout=2)
        if not running:
            if dlg: dlg.destroy()
            self._show_ollama_guide_dialog(host)
            return
        def progress(msg): self.root.after(0,lambda m=msg:self._aip(m))
        ok = _pull_model(host, model_key, progress)
        if ok:
            if dlg: dlg.destroy()
            self._select_qwen_model(model_key, None)

    def _show_ai_settings(self):
        dlg=tk.Toplevel(self.root); dlg.title("AI模型管理器"); dlg.geometry("750x560")
        dlg.configure(bg=BG0); dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg,text="🤖 AI模型管理器",font=('微软雅黑',16,'bold'),fg=TEXT1,bg=BG0).pack(anchor='w',padx=16,pady=(12,4))

        # Ollama 服务状态
        host = self.cfg.g('ollama_host')
        running, models = _check_ollama_running(host, timeout=2)
        status_icon = '✅' if running else '❌'
        status_text = 'Ollama 服务运行中' if running else '未检测到 Ollama 服务'
        status_fg = GREEN if running else RED
        status_frame=tk.Frame(dlg,bg=BG0); status_frame.pack(fill='x',padx=16,pady=(0,8))
        tk.Label(status_frame,text=f"{status_icon} {status_text}",font=('微软雅黑',10,'bold'),
                 fg=status_fg,bg=BG0).pack(side='left')
        tk.Label(status_frame,text=f"(地址: {host})",font=('微软雅黑',8),fg=TEXT2,bg=BG0).pack(side='left',padx=6)
        ttk.Button(status_frame,text="🔄 重试检测",command=lambda h=host: self._retry_ollama_check(h, status_frame)).pack(side='left',padx=4)

        pan=ttk.PanedWindow(dlg,orient='horizontal'); pan.pack(fill='both',expand=True,padx=12,pady=8)

        lf=tk.Frame(pan,bg=BG1); pan.add(lf,weight=1)
        tk.Label(lf,text="模型列表",font=('微软雅黑',11,'bold'),fg=TEXT1,bg=BG1).pack(anchor='w',padx=10,pady=(8,2))
        models_list=[
            {'id':'scrfd','name':'SCRFD 2.5GF','cat':'本地','size':'3.6MB ✓','cost':'免费','req':'CPU','speed':'<0.1s/张','stars':5,'note':'人脸检测+关键点，已部署','risk':'无'},
            {'id':'quality','name':'多维质量评分','cat':'本地','size':'0MB','cost':'免费','req':'CPU','speed':'<0.1s/张','stars':4,'note':'技术/构图/人像/美学评分','risk':'无'},
        ]
        # 添加 Qwen-VL 本地模型
        for key, info in QWEN_MODELS.items():
            models_list.append({
                'id': key,
                'name': f"Qwen2.5-VL {info['name'].split()[-1]}",
                'cat': '本地',
                'size': info['size'],
                'cost': '免费',
                'req': f"GPU {info['显存需求']}",
                'speed': info['速度'],
                'stars': info['stars'],
                'note': f"精度：{info['精度']} | 需Ollama",
                'risk': '照片不离开本机',
                'model_key': key,
            })
        rf=tk.Frame(pan,bg=BG0); pan.add(rf,weight=2)
        detail_frame=tk.Frame(rf,bg=BG0); detail_frame.pack(fill='both',expand=True,padx=8)
        tk.Label(detail_frame,text="← 选择左侧模型查看详情",font=('微软雅黑',11),fg=TEXT2,bg=BG0).pack(expand=True)

        for m in models_list:
            icon='🟢' if m['cat']=='本地' else '🔴'
            b=tk.Button(lf,text=f"{icon} {m['name']}",font=('微软雅黑',10),bg=BG3,fg=TEXT1,anchor='w',bd=0,padx=10,
                        command=lambda m=m,p=detail_frame: self._show_model(m,p))
            b.pack(fill='x',padx=8,pady=2)

    def _retry_ollama_check(self, host, status_frame):
        """重试检测 Ollama 服务"""
        tk.Label(status_frame, text="检测中...", font=('微软雅黑',10), fg=GOLD, bg=BG0).pack(side='left')
        self.root.update_idletasks()
        running, _ = _check_ollama_running(host, timeout=3)
        status_icon = '✅' if running else '❌'
        status_text = 'Ollama 服务运行中' if running else '未检测到 Ollama 服务'
        status_fg = GREEN if running else RED
        tk.Label(status_frame, text=f"{status_icon} {status_text}", font=('微软雅黑',10,'bold'),
                 fg=status_fg, bg=BG0).pack(side='left')
        if running:
            messagebox.showinfo("成功", "Ollama 服务已就绪！", parent=self.root)

    def _show_model(self,m,parent):
        for w in parent.winfo_children(): w.destroy()
        card=tk.Frame(parent,bg=BG1); card.pack(fill='both',expand=True,padx=4,pady=4)
        tk.Label(card,text=m['name'],font=('微软雅黑',14,'bold'),fg=TEXT1,bg=BG1).pack(anchor='w',padx=14,pady=(12,4))
        stars='★'*m['stars']+'☆'*(5-m['stars'])
        for lb,v in [('大小',m['size']),('费用',m['cost']),('配置',m['req']),('速度',m['speed']),
                      ('推荐',stars),('备注',m['note']),('风险',m['risk'])]:
            rf=tk.Frame(card,bg=BG1); rf.pack(fill='x',padx=14,pady=3)
            tk.Label(rf,text=f"🔹 {lb}：",font=('微软雅黑',10,'bold'),fg=TEXT2,bg=BG1).pack(side='left')
            c=RED if lb=='风险' and v!='无' else GOLD
            tk.Label(rf,text=v,font=('微软雅黑',10),fg=c,bg=BG1,wraplength=300).pack(side='left')
        bf=tk.Frame(card,bg=BG1); bf.pack(fill='x',padx=14,pady=(12,8))
        # SCRFD/多维质量评分
        if m['id'] in ('scrfd','quality'):
            tk.Label(bf,text="✅ 已部署运行中",fg=GREEN,bg=BG1,font=('微软雅黑',11,'bold')).pack(side='left')
        # Qwen-VL 本地模型
        elif 'model_key' in m:
            model_key = m['model_key']
            qwen_host = self.cfg.g('ollama_host')
            running, _ = _check_ollama_running(qwen_host, timeout=2)
            available = running and _has_model(qwen_host, model_key)
            if available:
                tk.Label(bf,text="✅ 已就绪，点击启用",fg=GREEN,bg=BG1,font=('微软雅黑',11,'bold')).pack(side='left',padx=6)
                tk.Button(bf,text="启用此模型",bg=GREEN,fg='#fff',font=('微软雅黑',10,'bold'),padx=12,pady=4,
                          command=lambda mk=model_key: self._enable_qwen_model(mk)).pack(side='left')
            elif running:
                tk.Label(bf,text="⬇ 模型未下载，点击下方按钮拉取",fg=ACCENT,bg=BG1,font=('微软雅黑',10)).pack(side='left',padx=6)
                tk.Button(bf,text="下载此模型",bg=ACCENT,fg='#fff',font=('微软雅黑',10,'bold'),padx=12,pady=4,
                          command=lambda mk=model_key: self._pull_and_enable_qwen(mk)).pack(side='left')
            else:
                btn_frame=tk.Frame(bf,bg=BG1)
                btn_frame.pack(side='left',padx=6)
                tk.Label(btn_frame,text="❌ Ollama 服务未运行",fg=RED,bg=BG1,font=('微软雅黑',10)).pack(side='left')
                ttk.Button(btn_frame,text="重试",command=lambda h=qwen_host: self._retry_ollama_in_model(h, btn_frame)).pack(side='left',padx=4)

    def _retry_ollama_in_model(self, host, parent):
        """模型详情面板中的 Ollama 重试检测"""
        btn_frame = parent
        for w in btn_frame.winfo_children():
            w.destroy()
        tk.Label(btn_frame, text="检测中...", font=('微软雅黑',10), fg=GOLD, bg=BG1).pack(side='left')
        self.root.update_idletasks()
        running, _ = _check_ollama_running(host, timeout=3)
        if running:
            tk.Label(btn_frame, text="✅ 服务已就绪", font=('微软雅黑',10), fg=GREEN, bg=BG1).pack(side='left')
        else:
            tk.Label(btn_frame, text="❌ 仍未检测到服务", font=('微软雅黑',10), fg=RED, bg=BG1).pack(side='left')
            ttk.Button(btn_frame, text="重试", command=lambda h=host: self._retry_ollama_in_model(h, btn_frame)).pack(side='left', padx=4)

    # ═══════════ 扫描 ═══════════
    def _browse(self):
        p=filedialog.askdirectory(title="选择照片文件夹")
        if p: self.fv.set(p)

    def _start(self):
        f=self.fv.get().strip()
        if not f or not os.path.isdir(f): self.st.config(text="请先选择文件夹"); return
        exts={'.jpg','.jpeg','.png','.bmp','.webp','.tiff','.tif'}
        fls=[fn for fn in sorted(os.listdir(f)) if Path(fn).suffix.lower() in exts]
        if not fls: self.st.config(text="无图片文件"); return
        self.tree.delete(*self.tree.get_children()); self.tdup.delete(*self.tdup.get_children())
        self.results.clear(); self.bad_photos.clear(); self.dup_groups.clear(); self._errors.clear()
        self.pb['value']=0; self.pb['maximum']=len(fls); self.processing=True
        self._stop=False; self._paused=False; self._paused_idx=0
        self._paused_files=fls; self._paused_folder=f; self._paused_hm={}
        self._scan_start_time=time.time(); self._scanned_count=0
        self.bdel.config(state='disabled')
        self.lbl_pb.config(text="扫描中...")
        threading.Thread(target=self._proc,args=(f,fls,0,{}),daemon=True).start()
    def _pause_scan(self):
        self._paused=True; self.bpz.config(state='disabled'); self.bre.config(state='normal')
        self.lbl_pb.config(text="已暂停")
    def _resume_scan(self):
        self._paused=False; self.bpz.config(state='normal'); self.bre.config(state='disabled')
        self.lbl_pb.config(text="继续扫描...")
        threading.Thread(target=self._proc,args=(self._paused_folder,self._paused_files,self._paused_idx,self._paused_hm),daemon=True).start()
    def _re_scan(self): self._start()
    def _stop_scan(self): self._stop=True; self._paused=False

    def _proc(self,folder,files,start_idx,hm):
        total=len(files)
        deep_scan = self.cfg.gb('qwen_deep_scan')  # 是否启用 AI 深度扫描
        qwen_available = self.checker.qwen and self.checker.qwen.available
        if deep_scan and not qwen_available:
            logger.warning("Qwen 深度扫描已开启但模型不可用，已跳过")
            deep_scan = False

        for idx in range(start_idx,len(files)):
            if self._stop: break
            while self._paused and not self._stop: self._paused_idx=idx; self._paused_hm=hm; time.sleep(0.1)
            if self._stop: break
            fn=files[idx]
            fp=os.path.join(folder,fn)
            try:
                r=self.checker.check_single(fp)
            except PermissionError:
                logger.warning(f"权限不足跳过 {fn}")
                with self._data_lock:
                    self._errors.append({'file': fn, 'reason': '权限不足', 'detail': '文件被占用或权限不足，已跳过'})
                self._scanned_count=idx+1
                continue
            except Exception as e:
                logger.warning(f"扫描异常跳过 {fn}: {e}")
                with self._data_lock:
                    self._errors.append({'file': fn, 'reason': '扫描异常', 'detail': str(e)})
                self._scanned_count=idx+1
                continue

            # 如果开启了 AI 深度扫描，每张照片额外调用 Qwen 分析
            if deep_scan:
                qr = self.checker.check_single_qwen(fp)
                if qr:
                    r['ai_qwen'] = qr
                    r['_should_remove'] = qr.get('should_remove', False)

            # 线程锁保护：写入共享列表
            with self._data_lock:
                self.results.append(r)
                if r['issues'] or r.get('_should_remove'): self.bad_photos.append(r)
                self._scanned_count=idx+1

            sc=f"{r['overall']:.0f}" if r['overall']>0 else '-'
            for iss in r['issues']:
                # 闭包修复：固化变量到默认参数
                self.root.after(0,lambda fnm=fn,t=iss['t'],c=iss['c'],s=sc:
                    self.tree.insert('','end',values=(fnm,t,f"{c}%",s)))
            ph=r.get('phash')
            if ph: hm.setdefault(ph,[]).append((idx,fn))
            # 时间估算标签
            elapsed=time.time()-self._scan_start_time if self._scan_start_time else 1
            spd=self._scanned_count/max(elapsed,0.001)
            remaining=(total-self._scanned_count)/spd if spd>0 else 0
            rem_str=f" 剩余~{int(remaining)}s" if remaining>1 else ""
            self.root.after(0,lambda i=idx+1,t=total,r=remaining,rs=rem_str: self._upd(i,t,r,rs))
        # 扫描完成：合并重复组
        if not self._paused and not self._stop:
            th=self.cfg.gi('duplicate_threshold'); ks=list(hm.keys()); seen=set()
            dup_list=[]
            for i in range(len(ks)):
                if i in seen: continue
                g=[ks[i]]
                for j in range(i+1,len(ks)):
                    if j in seen: continue
                    if PhotoChecker.hamming(ks[i],ks[j])<=(64-th*64/100): g.append(ks[j]); seen.add(j)
                if len(g)>1:
                    ns=[]
                    for hh in g: ns.extend(fn for _,fn in hm[hh])
                    dup_list.append(ns)
            # 闭包修复：用列表索引传递，避免共享引用
            for gi,ns in enumerate(dup_list):
                self.root.after(0,lambda n=list(ns),gi=gi: self._add_dup_row(n,gi))
            with self._data_lock:
                self.dup_groups.extend(dup_list)
        if not self._paused: self.root.after(0,self._done)

    def _upd(self,cur,total,remaining=0,rem_str=""):
        self.pb['value']=cur; self.lbl_pb.config(text=f"Scan: {cur}/{total} {rem_str}")
        with self._data_lock:
            bad=len(self.bad_photos)
        self.lbl_stats.config(text=f"Bad: {bad} | Total: {total}")

    def _add_dup_row(self,ns,gi):
        """线程安全添加重复组行"""
        self.tdup.insert('','end',values=(', '.join(ns),))

    def _done(self):
        self.processing=False
        self.bs.config(state='normal'); self.bpz.config(state='disabled')
        self.bre.config(state='disabled'); self.brs.config(state='normal'); self.bdel.config(state='normal')
        with self._data_lock:
            bad=len(self.bad_photos); dup=len(self.dup_groups); total=len(self.results)
            error_count=len(self._errors)

        # v5.0: 数据集异常值检测
        outlier_count = 0
        try:
            z_thresh = self.cfg.outlier_zscore_thresh()
            outliers = AIDetector.detect_outliers(self.results, z_thresh)
            outlier_count = len(outliers)
            if outliers:
                # 在 bad_photos 中标记异常值
                with self._data_lock:
                    for idx, metrics in outliers:
                        if 0 <= idx < len(self.results):
                            self.results[idx]['outlier_metrics'] = metrics
                            if self.results[idx] not in self.bad_photos:
                                self.bad_photos.append(self.results[idx])
        except Exception as e:
            logger.warning(f"异常值检测失败: {e}")

        stp="已停止 · " if self._stop else ""
        self.lbl_pb.config(text=stp+"完成!")
        ps = [f"Bad: {bad}"]
        if dup: ps.append(f"Dup: {dup}")
        if outlier_count: ps.append(f"Outlier: {outlier_count}")
        if error_count: ps.append(f"Skipped: {error_count}")
        ps.append(f"Total: {total}")
        self.lbl_stats.config(text=" | ".join(ps))

        # 扫描完成后刷新树形显示
        self.root.after(200, self._refresh_tree)

        # 如果有失败的文件，弹出汇总提示
        if error_count > 0:
            self.root.after(500, lambda ec=error_count: self._show_error_summary(ec))

    # ═══════════ 预览 ═══════════
    def _load_thumb(self, fp, size=350):
        """加载图片缩略图，多策略回退：PIL → OpenCV → matplotlib
        关键：返回 ImageTk.PhotoImage 后调用者必须保存引用（label.image = photo），否则会被 GC 回收。
        """
        if not os.path.exists(fp):
            return None

        # ── 策略1: PIL ──
        if HAS_PIL:
            for convert_fn in [
                lambda img: img.convert('RGB'),
                lambda img: img.convert('RGBA').convert('RGB'),
                lambda img: img,
            ]:
                try:
                    with Image.open(fp) as img:
                        img = convert_fn(img)
                        img.thumbnail((size, size), Image.Resampling.LANCZOS)
                        return ImageTk.PhotoImage(img)
                except Exception:
                    continue

        # ── 策略2: OpenCV（支持中文路径） ──
        try:
            arr = np.fromfile(fp, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img.thumbnail((size, size), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(img)
        except Exception:
            pass

        # ── 策略3: matplotlib（兜底） ──
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(mpimg.imread(fp))
            ax.axis('off')
            fig.canvas.draw()
            w, h = fig.get_size_inches() * fig.dpi
            img = Image.fromarray(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(h), int(w), 3))
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            plt.close(fig)
            return ImageTk.PhotoImage(img)
        except Exception:
            pass

        return None

    def _preview(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx >= len(self.results):
            return
        r = self.results[idx]
        fp = os.path.join(self.fv.get().strip(), r['file'])

        # 清除旧图片引用，防止 GC 回收
        self._preview_img = None
        self.preview_label.image = None

        photo = self._load_thumb(fp)
        if photo:
            self._preview_img = photo
            self.preview_label.image = photo  # 关键：保持引用，防止 GC 回收
            self.preview_label.config(image=photo, text='', bg='#000000')
            # 尝试读取 EXIF
            try:
                with Image.open(fp) as img:
                    self.lbl_exif.config(text=self._exif(img))
            except Exception:
                self.lbl_exif.config(text='')
        else:
            if os.path.exists(fp):
                size_mb = os.path.getsize(fp) / 1024 / 1024
                try:
                    ext = Path(fp).suffix
                except Exception:
                    ext = '?'
                self.preview_label.config(image='', text=f"图片加载失败\n{ext} {size_mb:.1f}MB", bg='#000000', fg='#ff375f')
            else:
                self.preview_label.config(image='', text="文件不存在", bg='#000000', fg='#ff375f')

    def _exif(self,img):
        try:
            d={ExifTags.TAGS[k]:v for k,v in img._getexif().items() if k in ExifTags.TAGS} if img._getexif() else {}
            parts=[]
            for k,pre in [('Model',''),('ISOSpeedRatings','ISO'),('FNumber','f/'),('ExposureTime',''),('FocalLength','')]:
                if k in d:
                    v=d[k]; v=float(v) if isinstance(v,tuple) else v
                    if k=='FNumber': v=f"{float(v):.1f}"
                    if k=='FocalLength': v=f"{float(v):.0f}mm"
                    elif k=='ExposureTime': v=f"{v}s"
                    parts.append(f"{pre}{v}")
            return ' | '.join(parts) if parts else ''
        except: return ''

    def _full_report(self,event):
        sel=self.tree.selection()
        if not sel: return
        idx=self.tree.index(sel[0])
        if idx>=len(self.results): return
        r=self.results[idx]
        fp=os.path.join(self.fv.get().strip(),r['file'])

        win=tk.Toplevel(self.root); win.title(f"分析报告 - {r['file'][:50]}")
        win.geometry("1050x700"); win.configure(bg=BG0); win.transient(self.root)

        hdr=tk.Frame(win,bg=BG1,height=44); hdr.pack(fill='x')
        tk.Frame(hdr,bg=ACCENT,height=2).pack(side='bottom',fill='x')
        tk.Button(hdr,text="← 返回",font=('微软雅黑',10),bg=BG3,fg=TEXT1,bd=0,activebackground=ACCENT,
                  command=win.destroy).pack(side='left',padx=12,pady=8)
        tk.Label(hdr,text=r['file'][:60],font=('微软雅黑',10,'bold'),fg=TEXT1,bg=BG1).pack(side='left',padx=12)

        pan=ttk.PanedWindow(win,orient='horizontal'); pan.pack(fill='both',expand=True)

        lf=tk.Frame(pan,bg='#000000'); pan.add(lf,weight=1)
        photo=self._load_thumb(fp,500)
        if photo:
            il=tk.Label(lf,image=photo,bg='#000000'); il.image=photo; il.pack(expand=True)
        else:
            tk.Label(lf,text="图片加载失败",font=('微软雅黑',12),fg='#ff375f',bg='#000000').pack(expand=True)

        rf=tk.Frame(pan,bg=BG1); pan.add(rf,weight=1)
        cv=tk.Canvas(rf,bg=BG1,highlightthickness=0); sb=tk.Scrollbar(rf,orient='vertical',command=cv.yview)
        inner=tk.Frame(cv,bg=BG1); cv.create_window((0,0),window=inner,anchor='nw',width=460)
        inner.bind('<Configure>',lambda e:cv.configure(scrollregion=cv.bbox('all')))
        cv.configure(yscrollcommand=sb.set); cv.pack(side='left',fill='both',expand=True); sb.pack(side='right',fill='y')

        ov=r.get('overall',0); gd=r.get('grade','-')
        gc=GREEN if ov>=90 else GOLD if ov>=75 else ACCENT if ov>=60 else RED
        tk.Label(inner,text=f"综合评分: {ov:.0f}/100  {gd}",font=('微软雅黑',22,'bold'),fg=gc,bg=BG1).pack(anchor='w',padx=16,pady=(16,6))
        tk.Label(inner,text=f"场景: {r.get('scene','-')} | 人脸: {r.get('face_count',0)} | 美学: {r.get('aesthetic',0):.1f}/10",
                 font=('微软雅黑',11),fg=TEXT2,bg=BG1).pack(anchor='w',padx=16,pady=2)

        q=r.get('quality',{})
        if q:
            lfq=tk.LabelFrame(inner,text="质量维度",fg=TEXT1,bg=BG1,font=('微软雅黑',11))
            lfq.pack(fill='x',padx=14,pady=8)
            for k,lab in [('tech','技术质量'),('composition','构图'),('portrait','人像')]:
                d=q.get(k,{})
                if d and d.get('overall',0)>0:
                    pf=tk.Frame(lfq,bg=BG1); pf.pack(fill='x',padx=12,pady=3)
                    tk.Label(pf,text=f"{lab}: {d['overall']:.0f}/100",font=('微软雅黑',10),fg=TEXT1,bg=BG1).pack(side='left')
                    bar=tk.Canvas(pf,width=120,height=10,bg=BG3,highlightthickness=0)
                    bar.pack(side='right'); bar.create_rectangle(0,0,120*d['overall']/100,10,fill=GOLD,outline='')

        if r['issues']:
            il=tk.LabelFrame(inner,text="检测问题",fg=RED,bg=BG1,font=('微软雅黑',11))
            il.pack(fill='x',padx=14,pady=8)
            for iss in r['issues']:
                tk.Label(il,text=f"⚠ {iss['t']} ({iss['c']}%) - {iss.get('d','')}",
                         fg=RED,bg=BG1,font=('微软雅黑',10),wraplength=420).pack(anchor='w',padx=12,pady=2)

        sug=r.get('suggestions',[])
        if sug:
            sl=tk.LabelFrame(inner,text="AI改进建议",fg=GREEN,bg=BG1,font=('微软雅黑',11))
            sl.pack(fill='x',padx=14,pady=8)
            for s in sug:
                tk.Label(sl,text=f"• {s}",fg=TEXT1,bg=BG1,wraplength=420,
                         font=('微软雅黑',10)).pack(anchor='w',padx=12,pady=2)

        tk.Label(inner,text="⚠ AI分析存在误差，请人工复核确认",fg='#555',bg=BG1,font=('微软雅黑',9)).pack(pady=(12,16))

    def _dup_detail(self,event):
        sel=self.tdup.selection()
        if not sel: return
        idx=self.tdup.index(sel[0])
        if idx>=len(self.dup_groups): return
        files=self.dup_groups[idx]
        win=tk.Toplevel(self.root); win.title(f"重复组 ({len(files)}张)"); win.geometry("450x450")
        win.configure(bg=BG0)
        tk.Button(win,text="← 返回",font=('微软雅黑',10),bg=BG3,fg=TEXT1,bd=0,command=win.destroy).pack(anchor='w',padx=12,pady=8)
        tk.Label(win,text=f"重复照片组 - {len(files)}张相似照片",font=('微软雅黑',12,'bold'),fg=TEXT1,bg=BG0).pack(anchor='w',padx=12)
        for fn in files:
            tk.Label(win,text=f"📷 {fn}",font=('微软雅黑',10),fg=TEXT2,bg=BG0).pack(anchor='w',padx=24,pady=2)

    def _csv(self):
        if not self.results: self.st.config(text="无结果可导出"); return
        f=self.fv.get().strip()
        n=f"分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        p=filedialog.asksaveasfilename(initialdir=f,initialfile=n,defaultextension='.csv',filetypes=[("CSV","*.csv")])
        if not p: return
        with open(p,'w',newline='',encoding='utf-8-sig') as fh:
            w=csv.writer(fh)
            w.writerow(['文件名','综合评分','等级','场景','美学分','问题类型','置信度','详情','AI建议'])
            for r in self.results:
                if r['issues']:
                    for iss in r['issues']:
                        w.writerow([r['file'],f"{r.get('overall',0):.0f}",r.get('grade',''),r.get('scene',''),
                                    r.get('aesthetic',0),iss['t'],f"{iss['c']}%",iss.get('d',''),
                                    '; '.join(r.get('suggestions',[]))])
                else:
                    w.writerow([r['file'],f"{r.get('overall',0):.0f}",r.get('grade',''),r.get('scene',''),
                                r.get('aesthetic',0),'正常','','','; '.join(r.get('suggestions',[]))])
        self.st.config(text="✓ 已导出CSV")

    # ═══════════ 设置对话框（排序/过滤/导出） ═══════════
    def _show_settings(self):
        """打开设置对话框：排序 + 过滤 + 导出 + AI深度扫描"""
        if not self.results:
            messagebox.showinfo("提示", "还没有扫描结果")
            return
        dlg=tk.Toplevel(self.root); dlg.title("设置"); dlg.geometry("400x420")
        dlg.configure(bg=BG0); dlg.transient(self.root); dlg.grab_set()

        tk.Label(dlg,text="⚙ 设置",font=('微软雅黑',14,'bold'),fg=TEXT1,bg=BG0).pack(pady=(12,8))

        # ── 排序 ──
        tf1=tk.LabelFrame(dlg,text="排序",fg=TEXT1,bg=BG1,font=('微软雅黑',10)); tf1.pack(fill='x',padx=12,pady=6)
        sort_opts = ['综合分↓','综合分↑','文件名↑','问题数↓','置信度↓']
        self._set_sort_var = tk.IntVar(value=0)
        for i,opt in enumerate(sort_opts):
            tk.Radiobutton(tf1,text=opt,variable=self._set_sort_var,value=i,fg=TEXT1,bg=BG1,
                           selectcolor=BG0,font=('微软雅黑',10),anchor='w',padx=12).pack(fill='x')

        # ── 过滤 ──
        tf2=tk.LabelFrame(dlg,text="过滤",fg=TEXT1,bg=BG1,font=('微软雅黑',10)); tf2.pack(fill='x',padx=12,pady=6)
        filter_opts = ['全部','仅问题','仅正常']
        self._set_filter_var = tk.IntVar(value=0)
        for i,opt in enumerate(filter_opts):
            tk.Radiobutton(tf2,text=opt,variable=self._set_filter_var,value=i,fg=TEXT1,bg=BG1,
                           selectcolor=BG0,font=('微软雅黑',10),anchor='w',padx=12).pack(fill='x')

        # ── Ollama 服务地址 ──
        tf_ollama=tk.LabelFrame(dlg,text="Ollama 服务地址",fg=TEXT1,bg=BG1,font=('微软雅黑',10))
        tf_ollama.pack(fill='x',padx=12,pady=6)
        self._set_ollama_host_var = tk.StringVar(value=self.cfg.g('ollama_host'))
        tk.Entry(tf_ollama,textvariable=self._set_ollama_host_var,width=42,font=('微软雅黑',10),
                 bg=BG2,fg=TEXT1,insertbackground=TEXT1,relief='flat').pack(padx=10,pady=4)
        tk.Label(tf_ollama,text="格式：http://IP:端口（默认 http://127.0.0.1:11434）",
                 font=('微软雅黑',8),fg=TEXT2,bg=BG1).pack(padx=10,pady=(0,4))

        # ── AI 深度扫描 ──
        tf_qwen=tk.LabelFrame(dlg,text="AI 深度扫描",fg=TEXT1,bg=BG1,font=('微软雅黑',10))
        tf_qwen.pack(fill='x',padx=12,pady=6)
        self._qwen_deep_var = tk.BooleanVar(value=self.cfg.gb('qwen_deep_scan'))
        tk.Checkbutton(tf_qwen,text="启用 Qwen2.5-VL 逐张深度分析",variable=self._qwen_deep_var,
                       bg=BG1,fg=TEXT1,selectcolor=BG0,font=('微软雅黑',10),anchor='w',padx=12).pack(fill='x',padx=12,pady=4)
        tk.Label(tf_qwen,text="开启后每张照片都会调用 Qwen 分析，扫描速度会变慢",
                 font=('微软雅黑',8),fg=TEXT2,bg=BG1).pack(padx=12,pady=(0,4))

        # ── 导出 ──
        tf3=tk.LabelFrame(dlg,text="导出",fg=TEXT1,bg=BG1,font=('微软雅黑',10)); tf3.pack(fill='x',padx=12,pady=6)
        tk.Button(tf3,text="📄 导出CSV",bg=ACCENT,fg='#fff',font=('微软雅黑',10,'bold'),padx=16,pady=6,
                  command=self._csv).pack(pady=6)

        # ── 按钮 ──
        bf=tk.Frame(dlg,bg=BG0); bf.pack(pady=(8,12))
        tk.Button(bf,text="应用",bg=GREEN,fg='#fff',font=('微软雅黑',10,'bold'),padx=20,pady=6,
                  command=lambda: self._apply_settings(self._set_sort_var.get(), self._set_filter_var.get(),
                                                       self._set_ollama_host_var.get(),
                                                       self._qwen_deep_var.get(), dlg)).pack(side='left',padx=8)

    def _apply_settings(self, sort_idx, filter_idx, ollama_host, qwen_deep, dlg):
        """应用排序、过滤、AI 深度扫描和 Ollama 地址设置"""
        sort_key_map = {0:'overall', 1:'overall', 2:'filename', 3:'issue_count', 4:'max_conf'}
        sort_rev_map = {0:True, 1:False, 2:True, 3:True, 4:True}
        filter_mode_map = {0:'all', 1:'bad', 2:'good'}
        self._sort_key = sort_key_map.get(sort_idx, 'overall')
        self._sort_reverse = sort_rev_map.get(sort_idx, True)
        self._filter_mode = filter_mode_map.get(filter_idx, 'all')
        self.cfg.s('qwen_deep_scan', '1' if qwen_deep else '0')
        if ollama_host:
            host = ollama_host.strip()
            if not host.startswith('http'):
                host = 'http://' + host
            self.cfg.s('ollama_host', host)
        self._refresh_tree()
        dlg.destroy()

    def _refresh_tree(self):
        """根据当前排序和过滤刷新表格显示"""
        self.tree.delete(*self.tree.get_children())
        items=list(enumerate(self.results))
        # 过滤
        if self._filter_mode=='bad': items=[(i,r) for i,r in items if r['issues'] or r.get('_should_remove')]
        elif self._filter_mode=='good': items=[(i,r) for i,r in items if not r['issues'] and not r.get('_should_remove')]
        # 排序
        reverse=self._sort_reverse
        if self._sort_key=='overall': items.sort(key=lambda x:x[1].get('overall',0),reverse=reverse)
        elif self._sort_key=='filename': items.sort(key=lambda x:x[1]['file'],reverse=reverse)
        elif self._sort_key=='issue_count': items.sort(key=lambda x:len(x[1].get('issues',[])),reverse=reverse)
        elif self._sort_key=='max_conf':
            items.sort(key=lambda x:max((iss['c'] for iss in x[1].get('issues',[])),default=0),reverse=reverse)
        # 插入
        for orig_idx,r in items:
            sc=f"{r.get('overall',0):.0f}" if r.get('overall',0)>0 else '-'
            if r['issues']:
                for iss in r['issues']:
                    self.tree.insert('','end',values=(r['file'],iss['t'],f"{iss['c']}%",sc))
            elif r.get('_should_remove'):
                self.tree.insert('','end',values=(r['file'],'AI建议移除','-',''))
            elif r.get('outlier_metrics'):
                metrics = ', '.join(r['outlier_metrics'])
                self.tree.insert('','end',values=(r['file'],f"异常值:{metrics[:20]}",'-',''))
            else:
                self.tree.insert('','end',values=(r['file'],'正常','','-'))

    def _about(self):
        messagebox.showinfo("关于",f"图片智能分析系统 v{VERSION}\n电竞暗黑美学 · SCRFD 2.5GF + Qwen2.5-VL\n100%本地离线 · 隐私优先\n\nAI分析仅供参考，请人工确认")

    # ═══════════ 错误汇总 ═══════════
    def _show_error_summary(self, error_count):
        """扫描完成后弹出错误汇总窗口"""
        dlg = tk.Toplevel(self.root); dlg.title("扫描完成报告")
        dlg.geometry("500x450"); dlg.configure(bg=BG0); dlg.transient(self.root); dlg.grab_set()

        tk.Label(dlg, text="扫描完成", font=('微软雅黑',16,'bold'), fg=TEXT1, bg=BG0).pack(pady=(16,4))
        tk.Label(dlg, text=f"成功: {len(self.results)} 张 | 失败: {error_count} 张",
                 font=('微软雅黑',11), fg=TEXT2, bg=BG0).pack(pady=4)

        if error_count == 0:
            tk.Label(dlg, text="✅ 所有图片均已成功处理", font=('微软雅黑',12), fg=GREEN, bg=BG0).pack(pady=12)
            tk.Button(dlg, text="确定", bg=GREEN, fg='#fff', font=('微软雅黑',10,'bold'),
                      padx=20, pady=5, command=dlg.destroy).pack(pady=(12,8))
            return

        # 错误列表
        tk.Label(dlg, text=f"以下 {error_count} 张图片扫描失败（已自动跳过，不影响其他图片）：",
                 font=('微软雅黑',10), fg=TEXT2, bg=BG0).pack(pady=(8,4), anchor='w', padx=16)

        tf=tk.Frame(dlg,bg=BG1); tf.pack(fill='both',expand=True,padx=12,pady=8)
        cb=tk.Canvas(tf,bg=BG1,highlightthickness=0); sb=tk.Scrollbar(tf,orient='vertical',command=cb.yview)
        inner=tk.Frame(cb,bg=BG1); cb.create_window((0,0),window=inner,anchor='nw')
        inner.bind('<Configure>',lambda e:cb.configure(scrollregion=cb.bbox('all')))
        cb.configure(yscrollcommand=sb.set); cb.pack(side='left',fill='both',expand=True); sb.pack(side='right',fill='y')

        with self._data_lock:
            for i, err in enumerate(self._errors):
                rf=tk.Frame(inner,bg=BG1); rf.pack(fill='x',pady=3,padx=6)
                # 序号
                tk.Label(rf,text=f"{i+1}.",font=('微软雅黑',10,'bold'),fg=RED,bg=BG1,width=4,anchor='e').pack(side='left')
                # 文件名
                tk.Label(rf,text=err['file'],font=('微软雅黑',10),fg=TEXT1,bg=BG1).pack(side='left')
                # 错误原因
                reason_label = {
                    '权限不足': '🔒 权限不足 — 文件被其他程序占用，请关闭后重试',
                    '读取失败': f"📄 读取失败 — {err.get('detail','无法解码')}",
                    'EXIF 错误': f"🔄 EXIF 错误 — {err.get('detail','旋转处理失败')}",
                    '扫描异常': f"⚠ 扫描异常 — {err.get('detail','未知原因')}",
                }.get(err.get('reason','扫描异常'), f"⚠ {err.get('detail','未知错误')}")
                tk.Label(rf,text=reason_label,font=('微软雅黑',9),fg=ACCENT,bg=BG1,wraplength=420,anchor='w').pack(fill='x',padx=(8,0))

        tk.Button(dlg, text="确定", bg=ACCENT, fg='#fff', font=('微软雅黑',10,'bold'),
                  padx=20, pady=5, command=dlg.destroy).pack(pady=(12,8))

    def _view_errors(self):
        """手动查看扫描失败记录"""
        if not self._errors:
            messagebox.showinfo("提示", "没有失败记录", parent=self.root)
            return
        self._show_error_summary(len(self._errors))

def main():
    root=tk.Tk()
    app=App(root)
    app._bind_events()
    root.mainloop()

if __name__=="__main__": main()
