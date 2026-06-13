"""
照片质量智能分析系统 v4.1
SCRFD人脸 · 多维评分 · 场景分类 · 分析报告
"""

import os, sys, shutil, csv, configparser, threading, json
from datetime import datetime
from pathlib import Path

import cv2, numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(2)
except: pass

try: import ttkbootstrap as tkb; HAS_TKB = True
except: HAS_TKB = False

try: from PIL import Image, ImageTk, ExifTags; HAS_PIL = True
except: HAS_PIL = False

from ai_detector import AIDetector, HAS_ONNX

# ═══════════════════════════════════════════
CONFIG_DEFAULTS = {
    'blur_threshold':'80','overexposure_pct':'80','underexposure_pct':'18',
    'over_brightness':'240','under_brightness':'30','duplicate_threshold':'95',
    'max_image_dim':'1200','aesthetic_min':'3.0','theme':'darkly',
    'last_folder':'','enable_ai':'1','ai_mode':'','api_key':'','api_endpoint':'','api_model':'',
}

class Config:
    def __init__(self):
        base = getattr(sys,'_MEIPASS','') or os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(base,'config.ini')
        self.cfg = configparser.ConfigParser()
        self.load()
    def load(self):
        self.cfg.read(self.path, encoding='utf-8')
        if 'settings' not in self.cfg: self.cfg['settings'] = {}
    def save(self):
        with open(self.path,'w',encoding='utf-8') as f: self.cfg.write(f)
    def g(self,k): return self.cfg['settings'].get(k,CONFIG_DEFAULTS.get(k,''))
    def gf(self,k): return float(self.g(k))
    def gi(self,k): return int(float(self.g(k)))
    def gb(self,k): return self.g(k)=='1'
    def s(self,k,v): self.cfg['settings'][k]=str(v); self.save()

# ═══════════════════════════════════════════

class PhotoChecker:
    def __init__(self, cfg: Config):
        self.cfg = cfg; self.ai = None
    def init_ai(self, cb=None):
        if self.ai is None and self.cfg.gb('enable_ai'):
            self.ai = AIDetector(enable_ai=True, progress_cb=cb)
    @staticmethod
    def _imread(p):
        a = np.fromfile(p, dtype=np.uint8)
        return cv2.imdecode(a, cv2.IMREAD_COLOR)
    @staticmethod
    def compute_phash(gray):
        r = cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA)
        d = cv2.dct(np.float32(r))[:8,:8]
        h = 0
        for b in (d>d.mean()).flatten(): h=(h<<1)|int(b)
        return format(h,'016x')
    @staticmethod
    def hamming(a,b): return bin(int(a,16)^int(b,16)).count('1')

    def check_single(self, path):
        r = {'file':os.path.basename(path),'issues':[],'phash':None,
             'aesthetic':0,'overall':0,'scene':'','suggestions':[],
             'face_count':0,'face_results':[]}

        img = self._imread(path)
        if img is None: r['issues'].append({'t':'读取失败','c':100,'d':'无法解码'}); return r

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = gray.shape
        md = self.cfg.gi('max_image_dim')
        if max(h,w)>md:
            s = md/max(h,w); gray=cv2.resize(gray,(int(w*s),int(h*s)))
            rgb=cv2.resize(rgb,(int(w*s),int(h*s))); h,w=gray.shape

        # 传统检测
        lap = cv2.Laplacian(gray,cv2.CV_64F).var()
        bt = self.cfg.gf('blur_threshold')
        if lap<bt:
            bc=''
            if lap<30: bc='失焦模糊'
            elif lap<60: bc='运动模糊'
            r['issues'].append({'t':'模糊','c':round(max(0,(1-lap/bt)*100),1),
                                'd':f"方差={lap:.0f} {bc}"})

        hist = cv2.calcHist([gray],[0],None,[256],[0,256]); tot=gray.size
        over = np.sum(hist[self.cfg.gi('over_brightness'):])/tot*100
        under = np.sum(hist[:self.cfg.gi('under_brightness')])/tot*100
        if over>self.cfg.gf('overexposure_pct'):
            r['issues'].append({'t':'过曝','c':round(over,1),'d':f"{over:.0f}%过亮"})
        if under>self.cfg.gf('underexposure_pct'):
            r['issues'].append({'t':'欠曝','c':round(under,1),'d':f"{under:.0f}%过暗"})

        # AI 分析
        if self.ai:
            face_dets = self.ai.detect_faces(rgb)
            r['face_count'] = len(face_dets) if face_dets else 0

            if face_dets:
                for fd in face_dets[:10]:
                    fq = self.ai.analyze_face(fd)
                    r['face_results'].append(fq)
                    if fq['eyes_open']<40:
                        r['issues'].append({'t':'闭眼','c':round(100-fq['eyes_open'],1),
                                            'd':f"睁眼度{fq['eyes_open']:.0f}%"})
                    if fq['head_angle']>30:
                        r['issues'].append({'t':'侧脸','c':round(fq['head_angle'],1),
                                            'd':f"偏转{fq['head_angle']:.0f}°"})

            r['scene'] = self.ai.classify_scene(rgb, r['face_count'])
            quality = self.ai.analyze_quality(rgb, gray, r['face_results'])
            r['overall'] = quality.get('overall',0)
            r['aesthetic'] = quality.get('aesthetic',0)
            r['grade'] = quality.get('grade','')
            r['quality'] = quality
            r['suggestions'] = self.ai.suggest(quality, r['scene'], r['face_count'])

            min_aes = self.cfg.gf('aesthetic_min')
            if r['overall'] < 55:
                r['issues'].append({'t':'综合评分低','c':round(55-r['overall'],1),
                                    'd':f"得分{r['overall']:.0f}/100"})
        else:
            # 传统回退
            face = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,'haarcascade_frontalface_default.xml'))
            eye  = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,'haarcascade_eye.xml'))
            faces = face.detectMultiScale(gray,1.1,5,minSize=(80,80))
            r['face_count'] = len(faces)
            if len(faces)>0:
                r['scene'] = '人像写真' if len(faces)<3 else '集体照'
                closed=0
                for (fx,fy,fw,fh) in faces[:3]:
                    e=eye.detectMultiScale(gray[fy:fy+fh//2,fx:fx+fw],1.05,4,minSize=(20,20))
                    if len(e)<2: closed+=1
                if closed>0:
                    r['issues'].append({'t':'闭眼','c':round(closed/min(len(faces),3)*100,1),
                                        'd':f"{closed}/{min(len(faces),3)}"})
            else:
                from ai_detector import SceneClassifier
                r['scene'] = SceneClassifier().classify(rgb,0) if not self.ai else '其他场景'

        # pHash
        try:
            s = cv2.resize(gray,(128,128)); r['phash']=self.compute_phash(s)
        except: r['phash']=None

        return r


# ═══════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root; self.root.title("图片质量智能分析系统 v4.1")
        self.root.geometry("1200x800"); self.root.minsize(1000,640)
        self.cfg = Config(); self.checker = PhotoChecker(self.cfg)
        self.results=[]; self.bad_photos=[]; self.dup_groups=[]; self.move_hist=[]
        self.processing=False; self._stop=False; self._paused=False
        self._paused_idx=0; self._paused_files=[]; self._paused_folder=''
        self._paused_hm={}; self.has_dnd=self._chk_dnd()
        self._build(); self._load_p(); self._bind_drop()
        self.root.after(600,self._init_ai)

    def _chk_dnd(self):
        try: import tkinterdnd2; return True
        except: return False

    # ── UI ──
    def _build(self):
        r=self.root
        if HAS_TKB:
            self.style=tkb.Style(theme='darkly')
        else:
            self.style=ttk.Style(); self.style.theme_use('clam')
            r.configure(bg='#1E1E1E')
            self.style.configure('.',background='#1E1E1E',foreground='#D4D4D4',fieldbackground='#2D2D2D')
            self.style.configure('TFrame',background='#1E1E1E')
            self.style.configure('TLabel',background='#1E1E1E',foreground='#D4D4D4')
            self.style.configure('TLabelframe',background='#1E1E1E',foreground='#D4D4D4')
            self.style.configure('Treeview',background='#252526',foreground='#D4D4D4',fieldbackground='#252526')
            self.style.map('Treeview',background=[('selected','#094771')])
            self.style.configure('TButton',background='#3C3C3C',foreground='#D4D4D4')
            self.style.configure('TProgressbar',background='#165DFF')

        # 工具栏
        tb=ttk.Frame(r); tb.pack(fill='x',padx=8,pady=(6,0))
        ttk.Label(tb,text="文件夹:").pack(side='left')
        self.fv=tk.StringVar(value=self.cfg.g('last_folder'))
        ttk.Entry(tb,textvariable=self.fv,width=42).pack(side='left',padx=6,fill='x',expand=True)
        ttk.Button(tb,text="选择",command=self._browse).pack(side='left')
        self.bs=ttk.Button(tb,text="▶ 扫描",command=self._start)
        self.bs.pack(side='left',padx=(8,0))
        self.bpz=ttk.Button(tb,text="⏸ 暂停",command=self._pause_scan,state='disabled')
        self.bpz.pack(side='left',padx=4)
        self.bre=ttk.Button(tb,text="⏵ 继续",command=self._resume_scan,state='disabled')
        self.bre.pack(side='left',padx=4)
        self.bx=ttk.Button(tb,text="■ 停止",command=self._stop_scan,state='disabled')
        self.bx.pack(side='left',padx=4)
        ttk.Button(tb,text="移走废片",command=self._move).pack(side='left',padx=4)
        ttk.Button(tb,text="撤销",command=self._undo).pack(side='left',padx=4)
        ttk.Button(tb,text="导出CSV",command=self._csv).pack(side='left',padx=4)
        self.bp=ttk.Button(tb,text="⚙ 参数",command=self._tp)
        self.bp.pack(side='right',padx=4)

        # 参数面板（折叠）
        self.pv=tk.BooleanVar(value=False)
        self.pf=ttk.LabelFrame(r,text="检测参数",padding=8)
        ai_row=ttk.Frame(self.pf); ai_row.grid(row=0,column=0,columnspan=3,sticky='ew')
        self.ai_var=tk.BooleanVar(value=self.cfg.gb('enable_ai'))
        ttk.Checkbutton(ai_row,text="🤖 AI智能分析 (SCRFD人脸 + 多维评分 + 场景分类)",
                        variable=self.ai_var,
                        command=self._on_ai_toggle).pack(side='left')

        pd=[("模糊阈值","blur_threshold",20,300,5),("过曝判定%","overexposure_pct",30,100,5),
            ("欠曝判定%","underexposure_pct",5,50,1),("过亮值","over_brightness",200,255,5),
            ("过暗值","under_brightness",5,80,5),("重复相似度%","duplicate_threshold",80,100,1),
            ("美学最低分","aesthetic_min",1.0,9.0,0.5)]
        self.sls={}
        for i,(lb,k,lo,hi,st) in enumerate(pd):
            c,r2=i%3,i//3+1; f=ttk.Frame(self.pf); f.grid(row=r2,column=c,padx=10,pady=4,sticky='ew')
            ttk.Label(f,text=lb,font=('微软雅黑',9)).pack(anchor='w')
            # min/max + 当前值
            vf=ttk.Frame(f); vf.pack(fill='x')
            ttk.Label(vf,text=str(lo),font=('微软雅黑',7),foreground='#888').pack(side='left')
            fl=isinstance(lo,float); sv=tk.DoubleVar(value=self.cfg.gf(k)) if fl else tk.IntVar(value=self.cfg.gi(k))
            val_lbl=ttk.Label(vf,text="",font=('微软雅黑',8,'bold'),foreground='#ff9500',width=7)
            val_lbl.pack(side='right')
            ttk.Label(vf,text=str(hi),font=('微软雅黑',7),foreground='#888').pack(side='right')
            def _update_val(v, lbl=val_lbl, fl=fl):
                lbl.config(text=f"{float(v):.1f}" if fl else f"{int(float(v))}")
            sv.trace_add('write', lambda *a, sv=sv, lbl=val_lbl, fl=fl:
                         lbl.config(text=f"{float(sv.get()):.1f}" if fl else f"{int(sv.get())}"))
            s=ttk.Scale(f,from_=lo,to=hi,variable=sv)
            s.pack(fill='x')
            self.sls[k]=(sv,fl)
            _update_val(sv.get())
        bf=ttk.Frame(self.pf); bf.grid(row=99,column=0,columnspan=3,pady=(8,0),sticky='w')
        ttk.Button(bf,text="恢复默认",command=self._rp).pack(side='left')
        ttk.Button(bf,text="保存参数",command=self._sp).pack(side='left',padx=8)
        for i in range(3): self.pf.columnconfigure(i,weight=1)

        # 统计栏
        sf=ttk.Frame(r); sf.pack(fill='x',padx=8,pady=(6,0))
        self.lbl_stats=ttk.Label(sf,text="就绪",font=('微软雅黑',10))
        self.lbl_stats.pack(side='left')
        self.lbl_ai=ttk.Label(sf,text="AI: 初始化中...",foreground='#888')
        self.lbl_ai.pack(side='right')

        # 进度条
        self.pb=ttk.Progressbar(r,mode='determinate'); self.pb.pack(fill='x',padx=8,pady=2)
        self.lbl_pb=ttk.Label(r,text=""); self.lbl_pb.pack(anchor='w',padx=12)

        # 主区域
        main=ttk.PanedWindow(r,orient='horizontal')
        main.pack(fill='both',expand=True,padx=8,pady=(4,8))

        # 结果表格
        tf=ttk.Frame(main); main.add(tf,weight=3)
        cols=("fn","issue","conf","detail","score","scene")
        self.tree=ttk.Treeview(tf,columns=cols,show='headings',selectmode='browse')
        self.tree.heading("fn",text="文件名"); self.tree.heading("issue",text="问题")
        self.tree.heading("conf",text="置信度"); self.tree.heading("detail",text="详情")
        self.tree.heading("score",text="综合分"); self.tree.heading("scene",text="场景")
        self.tree.column("fn",width=150,minwidth=100); self.tree.column("issue",width=80,minwidth=60)
        self.tree.column("conf",width=80,minwidth=60); self.tree.column("detail",width=200,minwidth=120)
        self.tree.column("score",width=65,minwidth=50); self.tree.column("scene",width=80,minwidth=60)
        ts=ttk.Scrollbar(tf,orient='vertical',command=self.tree.yview)
        self.tree.configure(yscrollcommand=ts.set)
        self.tree.pack(side='left',fill='both',expand=True); ts.pack(side='right',fill='y')
        self.tree.bind('<Double-1>',self._report)
        self.tree.bind('<<TreeviewSelect>>',self._preview)

        # 右侧面板
        rf=ttk.Frame(main); main.add(rf,weight=1)
        plf=ttk.LabelFrame(rf,text="图片预览",padding=2)
        plf.pack(fill='both',expand=True)
        self.preview_frame = tk.Frame(plf, bg='#000000')
        self.preview_frame.pack(fill='both', expand=True)

        # 预览层
        self.prev_view = tk.Frame(self.preview_frame, bg='#000000')
        self.prev_view.pack(fill='both', expand=True)
        self.lbl_preview = tk.Label(self.prev_view, text="双击查看分析报告\n单击预览图片",
                                     bg='#000000', fg='#666', font=('微软雅黑', 11))
        self.lbl_preview.pack(expand=True)
        self._preview_img = None

        # 报告层（初始隐藏）
        self.rep_view = tk.Frame(self.preview_frame, bg='#1E1E1E')
        self._showing_report = False

        self.lbl_exif=ttk.Label(rf,text="",font=('微软雅黑',8),foreground='#888')
        self.lbl_exif.pack(fill='x',pady=(2,0))

        dlf=ttk.LabelFrame(rf,text="重复照片组",padding=2)
        dlf.pack(fill='x',pady=(4,0))
        self.tdup=ttk.Treeview(dlf,columns=("f",),show='headings',height=3)
        self.tdup.heading("f",text="相似组"); self.tdup.column("f",width=100)
        self.tdup.pack(fill='x')

        self.st=ttk.Label(r,text="v4.1 | SCRFD 2.5GF | 多维质量分析 | AI仅供参考",relief='sunken',
                          font=('微软雅黑',8)); self.st.pack(side='bottom',fill='x')

    # ── AI 初始化 ──
    def _init_ai(self):
        self.lbl_ai.config(text="AI: 初始化..."); self.root.update_idletasks()
        self.checker.init_ai(cb=lambda m: self.root.after(0,lambda msg=m:self._aip(msg)))
        if self.checker.ai and self.checker.ai.face_detector:
            self.lbl_ai.config(text="AI: SCRFD ✓ | 多维评分 ✓ | 场景分类 ✓",foreground='#4cd964')
        else:
            self.lbl_ai.config(text="AI: 标准模式 (pip install onnxruntime 启用AI)",foreground='#888')

    def _aip(self,msg): self.lbl_ai.config(text=f"AI: {msg}"); self.root.update_idletasks()

    def _show_ai_mode_dialog(self):
        if self.cfg.g('ai_mode'): return
        dlg=tk.Toplevel(self.root); dlg.title("选择AI运行模式"); dlg.geometry("560x520")
        dlg.configure(bg='#1E1E1E'); dlg.transient(self.root); dlg.grab_set(); dlg.resizable(False,False)
        tk.Label(dlg,text="选择AI运行模式",font=('微软雅黑',16,'bold'),fg='#D4D4D4',bg='#1E1E1E').pack(pady=(16,8))
        lf=tk.LabelFrame(dlg,text="",bg='#252526'); lf.pack(fill='x',padx=16,pady=6)
        tk.Label(lf,text="🟢 本地运行模式（推荐·隐私优先）",font=('微软雅黑',13,'bold'),fg='#4cd964',bg='#252526').pack(anchor='w',padx=10,pady=(8,0))
        for t in ["✅ 100%离线，照片永不上传","✅ 准确率85-90%，免费使用","💻 8G内存即可，CPU运行","💰 永久免费"]:
            tk.Label(lf,text=t,font=('微软雅黑',10),fg='#aaa',bg='#252526').pack(anchor='w',padx=24)
        of=tk.LabelFrame(dlg,text="",bg='#252526'); of.pack(fill='x',padx=16,pady=6)
        tk.Label(of,text="🔴 在线API模式（效果优先）",font=('微软雅黑',13,'bold'),fg='#ff6b35',bg='#252526').pack(anchor='w',padx=10,pady=(8,0))
        for t in ["✅ 准确率95%+，审美评分更专业","⚠️ 照片需上传至第三方服务器","💻 只需联网，无硬件要求","💰 约0.05-0.1元/张"]:
            tk.Label(of,text=t,font=('微软雅黑',10),fg='#aaa',bg='#252526').pack(anchor='w',padx=24)
        tk.Label(dlg,text="⚠ 选择在线模式即同意将照片上传至AI服务商服务器\n所有AI分析结果仅供参考，请务必人工确认",
                 font=('微软雅黑',9,'bold'),fg='#ff375f',bg='#1E1E1E',justify='center').pack(pady=8)
        bf=tk.Frame(dlg,bg='#1E1E1E'); bf.pack(pady=(8,12))
        tk.Button(bf,text="选择本地运行",font=('微软雅黑',11,'bold'),bg='#4cd964',fg='#fff',
                  padx=20,pady=8,command=lambda: self._pick_ai_mode('local',dlg)).pack(side='left',padx=8)
        tk.Button(bf,text="选择在线运行",font=('微软雅黑',11,'bold'),bg='#ff6b35',fg='#fff',
                  padx=20,pady=8,command=lambda: self._pick_ai_mode('online',dlg)).pack(side='left',padx=8)

    def _pick_ai_mode(self, mode, dlg):
        self.cfg.s('ai_mode', mode); self.cfg.s('enable_ai','1'); self.ai_var.set(True)
        dlg.destroy()
        if mode=='online': self._show_api_config()
        else: self._init_ai()

    def _show_api_config(self):
        dlg=tk.Toplevel(self.root); dlg.title("在线API配置"); dlg.geometry("480x340")
        dlg.configure(bg='#1E1E1E'); dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg,text="配置在线API",font=('微软雅黑',14,'bold'),fg='#D4D4D4',bg='#1E1E1E').pack(pady=12)
        tk.Label(dlg,text="API密钥（仅存本地）",font=('微软雅黑',10),fg='#888',bg='#1E1E1E').pack()
        ak=tk.Entry(dlg,width=50,show='*',font=('微软雅黑',10)); ak.pack(pady=6)
        ak.insert(0,self.cfg.g('api_key'))
        tk.Label(dlg,text="API端点",font=('微软雅黑',10),fg='#888',bg='#1E1E1E').pack()
        ep=tk.Entry(dlg,width=50,font=('微软雅黑',10)); ep.pack(pady=6)
        ep.insert(0,self.cfg.g('api_endpoint') or 'https://api.openai.com/v1/chat/completions')
        tk.Label(dlg,text="模型名",font=('微软雅黑',10),fg='#888',bg='#1E1E1E').pack()
        mn=tk.Entry(dlg,width=50,font=('微软雅黑',10)); mn.pack(pady=6)
        mn.insert(0,self.cfg.g('api_model') or 'gpt-4o')
        def save(): self.cfg.s('api_key',ak.get()); self.cfg.s('api_endpoint',ep.get()); self.cfg.s('api_model',mn.get()); dlg.destroy()
        tk.Button(dlg,text="保存配置",font=('微软雅黑',11,'bold'),bg='#165DFF',fg='#fff',padx=24,pady=6,command=save).pack(pady=12)
        tk.Label(dlg,text="💡 支持OpenAI/通义千问/文心一言等兼容API",font=('微软雅黑',9),fg='#555',bg='#1E1E1E').pack()

    def _on_ai_toggle(self):
        """AI开关点击"""
        if self.ai_var.get():
            if not self.cfg.g('ai_mode'):
                self.root.after(100, self._show_ai_mode_dialog)
            else:
                self.cfg.s('enable_ai','1'); self._init_ai()
        else:
            self.cfg.s('enable_ai','0')

    # ── 参数面板 ──
    def _tp(self):
        if self.pv.get(): self.pf.pack_forget(); self.pv.set(False)
        else: self.pf.pack(fill='x',padx=8,pady=(4,0),after=self.pb); self.pv.set(True)

    def _load_p(self):
        for k,(sv,fl) in self.sls.items():
            try: sv.set(self.cfg.gf(k) if fl else self.cfg.gi(k))
            except: sv.set(float(CONFIG_DEFAULTS.get(k,0)))

    def _rp(self):
        for k,(sv,fl) in self.sls.items():
            d=float(CONFIG_DEFAULTS.get(k,0))
            if not fl: d=int(d); sv.set(d)
            else: sv.set(d)
        self._sp()

    def _sp(self):
        for k,(sv,fl) in self.sls.items():
            v=sv.get(); self.cfg.s(k,str(int(v)) if not fl else f"{v:.1f}")
        self.cfg.s('enable_ai','1' if self.ai_var.get() else '0')
        if self.ai_var.get() and self.checker.ai is None: self._init_ai()
        elif not self.ai_var.get() and self.checker.ai: self.checker.ai=None; self.lbl_ai.config(text="AI: 已关闭",foreground='#888')
        self._flash_saved()

    def _flash_saved(self):
        """保存按钮旁边显示'已保存'"""
        if not hasattr(self,'_saved_lbl'):
            self._saved_lbl = ttk.Label(self.pf, text="", foreground='#4cd964', font=('微软雅黑',9))
            self._saved_lbl.grid(row=99, column=1, padx=(12,0), sticky='w')
        self._saved_lbl.config(text="✓ 已保存")
        self.root.after(2000, lambda: self._saved_lbl.config(text=""))

    # ── 拖拽 ──
    def _bind_drop(self):
        if self.has_dnd:
            try:
                from tkinterdnd2 import DND_FILES
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind('<<Drop>>',self._on_drop)
            except: pass
        self.root.bind('<Control-o>',lambda e:self._browse())

    def _on_drop(self,e):
        p=e.data.strip('{}').strip()
        if os.path.isdir(p): self.fv.set(p); self.cfg.s('last_folder',p)

    def _browse(self):
        p=filedialog.askdirectory(title="选择照片文件夹")
        if p: self.fv.set(p); self.cfg.s('last_folder',p)

    # ── 扫描 ──
    def _start(self):
        f=self.fv.get().strip()
        if not f or not os.path.isdir(f): messagebox.showwarning("提示","请先选择文件夹"); return
        exts={'.jpg','.jpeg','.png','.bmp','.webp','.tiff','.tif'}
        fls=[fn for fn in sorted(os.listdir(f)) if Path(fn).suffix.lower() in exts]
        if not fls: messagebox.showinfo("提示","无图片"); return

        self.tree.delete(*self.tree.get_children()); self.tdup.delete(*self.tdup.get_children())
        self.results.clear(); self.bad_photos.clear(); self.dup_groups.clear()
        self.pb['value']=0; self.pb['maximum']=len(fls); self.processing=True
        self._stop=False; self._paused=False; self._paused_idx=0
        self._paused_files=fls; self._paused_folder=f; self._paused_hm={}
        self.bs.config(state='disabled'); self.bx.config(state='normal')
        self.bpz.config(state='normal'); self.bre.config(state='disabled')
        self.lbl_pb.config(text="扫描中...")
        self._proc_thread=threading.Thread(target=self._proc,args=(f,fls,0,{}),daemon=True)
        self._proc_thread.start()

    def _pause_scan(self):
        self._paused=True; self.bpz.config(state='disabled'); self.bre.config(state='normal')
        self.lbl_pb.config(text="已暂停 — 点击继续")

    def _resume_scan(self):
        self._paused=False; self.bpz.config(state='normal'); self.bre.config(state='disabled')
        self.lbl_pb.config(text="扫描中...")
        self._proc_thread=threading.Thread(
            target=self._proc,args=(self._paused_folder,self._paused_files,
                                     self._paused_idx,self._paused_hm),daemon=True)
        self._proc_thread.start()

    def _stop_scan(self):
        self._stop=True; self._paused=False
        self.lbl_pb.config(text="正在停止...")

    def _proc(self,folder,files,start_idx,hm):
        total=len(files)
        for idx in range(start_idx, len(files)):
            if self._stop: break
            while self._paused and not self._stop:
                self._paused_idx=idx; self._paused_hm=hm
                import time; time.sleep(0.1)
            if self._stop: break
            fn=files[idx]; fp=os.path.join(folder,fn)
            r=self.checker.check_single(fp)
            self.results.append(r)
            if r['issues']: self.bad_photos.append(r)
            for iss in r['issues']:
                self.root.after(0,lambda fnm=fn,t=iss['t'],c=iss['c'],d=iss.get('d',''),
                                sc=r['overall'],sn=r.get('scene',''):
                    self.tree.insert('','end',values=(fnm,t,f"{c}%",d,
                        f"{sc:.0f}" if sc>0 else '-',sn or '-')))
            ph=r.get('phash')
            if ph: hm.setdefault(ph,[]).append((idx,fn))
            self.root.after(0,lambda i=idx+1,t=total: self._upd(i,t))
        # 重复
        if not self._paused and not self._stop:
            th=self.cfg.gi('duplicate_threshold')
            ks=list(hm.keys()); seen=set()
            for i in range(len(ks)):
                if i in seen: continue
                g=[ks[i]]
                for j in range(i+1,len(ks)):
                    if j in seen: continue
                    if PhotoChecker.hamming(ks[i],ks[j])<=(64-th*64/100):
                        g.append(ks[j]); seen.add(j)
                if len(g)>1:
                    ns=[]; [ns.extend(fn for _,fn in hm[h]) for h in g]
                    self.dup_groups.append(ns)
                    self.root.after(0,lambda n=ns: self.tdup.insert('','end',values=(', '.join(n[:4]),)))
        if not self._paused: self.root.after(0,self._done)

    def _upd(self,cur,total):
        self.pb['value']=cur
        self.lbl_pb.config(text=f"处理中：{cur}/{total} ({int(cur/total*100)}%)")
        self.lbl_stats.config(text=f"废片：{len(self.bad_photos)}  |  总计：{total}")

    def _done(self):
        self.processing=False
        self.bs.config(state='normal'); self.bx.config(state='disabled')
        self.bpz.config(state='disabled'); self.bre.config(state='disabled')
        bad,dup,total=len(self.bad_photos),len(self.dup_groups),len(self.results)
        stp="已停止 · " if self._stop else ""
        self.lbl_pb.config(text=stp+"扫描完成！")
        ps=[f"废片：{bad}"]
        if dup: ps.append(f"重复组：{dup}")
        ps.append(f"总计：{total}")
        self.lbl_stats.config(text="  |  ".join(ps))
        self.st.config(text=f"v4.1 | SCRFD 2.5GF | {total}张 | AI分析仅供参考，请人工确认")

    # ── 预览 ──
    def _preview(self,event):
        if self._showing_report: return
        sel=self.tree.selection()
        if not sel or not HAS_PIL: return
        idx=self.tree.index(sel[0])
        if idx>=len(self.results): return
        r=self.results[idx]; fp=os.path.join(self.fv.get().strip(),r['file'])
        if not os.path.exists(fp): return
        try:
            img=Image.open(fp); exif=self._exif(img)
            img.thumbnail((350,350),Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._preview_img = photo
            self.lbl_preview.config(image=photo, text='', bg='#000000')
            self.lbl_exif.config(text=exif)
        except Exception:
            self.lbl_preview.config(image='', text="预览失败", bg='#000000')

    def _exif(self,img):
        try:
            exif={ExifTags.TAGS[k]:v for k,v in img._getexif().items() if k in ExifTags.TAGS} if img._getexif() else {}
            parts=[]
            if 'Model' in exif: parts.append(exif['Model'])
            if 'ISOSpeedRatings' in exif: parts.append(f"ISO{exif['ISOSpeedRatings']}")
            if 'FNumber' in exif: parts.append(f"f/{float(exif['FNumber']):.1f}")
            if 'ExposureTime' in exif: parts.append(f"{exif['ExposureTime']}s")
            if 'FocalLength' in exif: parts.append(f"{float(exif['FocalLength']):.0f}mm")
            return ' | '.join(parts) if parts else ''
        except: return ''

    # ── 分析报告（嵌入式切换）──
    def _report(self,event):
        sel=self.tree.selection()
        if not sel: return
        idx=self.tree.index(sel[0])
        if idx>=len(self.results): return
        r=self.results[idx]

        # 切换到报告层
        self.prev_view.pack_forget()
        for w in self.rep_view.winfo_children(): w.destroy()
        self.rep_view.pack(fill='both', expand=True)
        self._showing_report = True

        cf=tk.Frame(self.rep_view,bg='#1E1E1E'); cf.pack(fill='both',expand=True)
        cv=tk.Canvas(cf,bg='#1E1E1E',highlightthickness=0)
        sb=tk.Scrollbar(cf,orient='vertical',command=cv.yview)
        inner=tk.Frame(cv,bg='#1E1E1E')
        inner.bind('<Configure>',lambda e:cv.configure(scrollregion=cv.bbox('all')))

        # 动态宽度：跟随容器
        def _set_width(e=None):
            w = cf.winfo_width() - 20 if cf.winfo_width() > 100 else 400
            cv.itemconfig(cv.find_all()[0], width=w) if cv.find_all() else None
        cf.bind('<Configure>', _set_width)
        cv.create_window((0,0),window=inner,anchor='nw',width=400,tags=('inner',))
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side='left',fill='both',expand=True); sb.pack(side='right',fill='y')

        # 文件名做截断
        fname = r['file'][:40]+'...' if len(r['file'])>40 else r['file']
        tk.Label(inner,text=f"\U0001f4cb {fname}",font=('微软雅黑',13,'bold'),fg='#D4D4D4',bg='#1E1E1E',
                 wraplength=380).pack(anchor='w',padx=12,pady=(12,0))

        ov=r.get('overall',0); gd=r.get('grade','-')
        gc='#4cd964' if ov>=90 else '#ff9500' if ov>=75 else '#ff6b35' if ov>=60 else '#ff375f'
        tk.Label(inner,text=f"综合评分：{ov:.0f}/100  {gd}",font=('微软雅黑',22,'bold'),fg=gc,bg='#1E1E1E').pack(anchor='w',padx=12,pady=8)

        tk.Label(inner,text=f"\U0001f3f7️ {r.get('scene','-')}  |  \U0001f464 {r.get('face_count',0)}人",
                 font=('微软雅黑',11),fg='#888',bg='#1E1E1E').pack(anchor='w',padx=12,pady=2)

        q=r.get('quality',{})
        if q:
            lf=tk.LabelFrame(inner,text=" 质量维度 ",fg='#D4D4D4',bg='#1E1E1E',font=('微软雅黑',11)); lf.pack(fill='x',padx=10,pady=8)
            for k,lab in [('tech','\U0001f4ca 技术质量'),('composition','\U0001f5bc️ 构图'),('portrait','\U0001f464 人像')]:
                d=q.get(k,{})
                if d and d.get('overall',0)>0:
                    tk.Label(lf,text=f"{lab}：{d['overall']:.0f}/100",fg='#D4D4D4',bg='#1E1E1E',
                             font=('微软雅黑',11)).pack(anchor='w',padx=10,pady=2)
            tk.Label(lf,text=f"\U0001f3a8 美学评分：{r.get('aesthetic',0):.1f}/10",fg='#ff9500',bg='#1E1E1E',
                     font=('微软雅黑',12,'bold')).pack(anchor='w',padx=10,pady=3)

        if r['issues']:
            il=tk.LabelFrame(inner,text=" ⚠️ 问题 ",fg='#ff6b35',bg='#1E1E1E',font=('微软雅黑',11)); il.pack(fill='x',padx=10,pady=8)
            for iss in r['issues']:
                tk.Label(il,text=f"⚠ {iss['t']} ({iss['c']}%) — {iss.get('d','')}",fg='#ff6b35',bg='#1E1E1E',
                         font=('微软雅黑',10),wraplength=380).pack(anchor='w',padx=10,pady=2)

        sug=r.get('suggestions',[])
        if sug:
            sl=tk.LabelFrame(inner,text=" \U0001f4a1 AI建议 ",fg='#4cd964',bg='#1E1E1E',font=('微软雅黑',11)); sl.pack(fill='x',padx=10,pady=8)
            for s in sug:
                tk.Label(sl,text=f"• {s}",fg='#D4D4D4',bg='#1E1E1E',wraplength=380,
                         font=('微软雅黑',10),justify='left').pack(anchor='w',padx=10,pady=2)

        tk.Label(inner,text="⚠ AI分析存在误差，请人工复核确认",fg='#555',bg='#1E1E1E',
                 font=('微软雅黑',9)).pack(pady=(10,8))
        tk.Button(inner,text="← 返回预览",font=('微软雅黑',10),bg='#3C3C3C',fg='#D4D4D4',
                  padx=16,pady=4,command=self._show_preview).pack(anchor='w',padx=12,pady=(0,14))

    def _show_preview(self):
        self.rep_view.pack_forget()
        self.prev_view.pack(fill='both', expand=True)
        self._showing_report = False

    # ── 操作 ──
    def _move(self):
        f=self.fv.get().strip()
        if not self.bad_photos: messagebox.showinfo("提示","无废片"); return
        bd=os.path.join(f,"_废片"); os.makedirs(bd,exist_ok=True)
        moved,self.move_hist=0,[]
        for r in self.bad_photos:
            src=os.path.join(f,r['file']); dst=os.path.join(bd,r['file'])
            try:
                if os.path.exists(src):
                    b,e=os.path.splitext(r['file']); c=1
                    while os.path.exists(dst): dst=os.path.join(bd,f"{b}_{c}{e}"); c+=1
                    shutil.move(src,dst); self.move_hist.append((src,dst)); moved+=1
            except: pass
        self.st.config(text=f"✓ 已移动 {moved} 张到 _废片/")
        self.root.after(3000, lambda: self.st.config(
            text=f"v4.1 | SCRFD 2.5GF | {len(self.results)}张 | AI仅供参考"))

    def _undo(self):
        if not self.move_hist:
            self.st.config(text="无可撤销操作")
            self.root.after(2000, lambda: self.st.config(
                text=f"v4.1 | SCRFD 2.5GF | {len(self.results)}张 | AI仅供参考"))
            return
        r=0
        for s,d in self.move_hist:
            try:
                if os.path.exists(d): shutil.move(d,s); r+=1
            except: pass
        self.move_hist.clear()
        self.st.config(text=f"✓ 已恢复 {r} 张照片到原位置")
        self.root.after(3000, lambda: self.st.config(
            text=f"v4.1 | SCRFD 2.5GF | {len(self.results)}张 | AI仅供参考"))

    def _csv(self):
        if not self.results: messagebox.showinfo("提示","无结果"); return
        f=self.fv.get().strip()
        n=f"分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        p=filedialog.asksaveasfilename(initialdir=f,initialfile=n,defaultextension='.csv',
                                        filetypes=[("CSV","*.csv")])
        if not p: return
        with open(p,'w',newline='',encoding='utf-8-sig') as fh:
            w=csv.writer(fh)
            w.writerow(['文件名','综合评分','等级','场景','美学分','问题类型','置信度','详情','AI建议'])
            for r in self.results:
                if r['issues']:
                    for iss in r['issues']:
                        w.writerow([r['file'],f"{r.get('overall',0):.0f}",r.get('grade',''),
                                    r.get('scene',''),r.get('aesthetic',0),
                                    iss['t'],f"{iss['c']}%",iss.get('d',''),
                                    '; '.join(r.get('suggestions',[]))])
                else:
                    w.writerow([r['file'],f"{r.get('overall',0):.0f}",r.get('grade',''),
                                r.get('scene',''),r.get('aesthetic',0),'正常','','',
                                '; '.join(r.get('suggestions',[]))])
        messagebox.showinfo("导出",f"已保存：{p}")

def main():
    try:
        from tkinterdnd2 import TkinterDnD; root=TkinterDnD.Tk()
    except: root=tk.Tk()
    App(root); root.mainloop()

if __name__=="__main__": main()