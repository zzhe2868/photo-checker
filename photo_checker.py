"""
图片智能分析系统 v2.0 - 电竞暗黑美学
SCRFD人脸 · 多维评分 · 场景分类 · 本地离线
"""
import os, sys, shutil, csv, configparser, threading, time
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try: from ctypes import windll; windll.shcore.SetProcessDpiAwareness(2)
except: pass
try: from PIL import Image, ImageTk, ExifTags; HAS_PIL = True
except: HAS_PIL = False
from ai_detector import AIDetector

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
    'blur_threshold':'80','overexposure_pct':'80','underexposure_pct':'18',
    'over_brightness':'240','under_brightness':'30','duplicate_threshold':'95',
    'max_image_dim':'1200','aesthetic_min':'3.0','enable_ai':'1',
    'ai_mode':'','api_key':'','api_endpoint':'','api_model':'',
}

class Config:
    def __init__(self):
        base = getattr(sys,'_MEIPASS','') or os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(base,'config.ini')
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

class PhotoChecker:
    def __init__(self, cfg): self.cfg=cfg; self.ai=None
    def init_ai(self,cb=None):
        if self.ai is None and self.cfg.gb('enable_ai'):
            self.ai=AIDetector(enable_ai=True,progress_cb=cb)
    @staticmethod
    def _imread(p):
        a=np.fromfile(p,dtype=np.uint8); return cv2.imdecode(a,cv2.IMREAD_COLOR)
    @staticmethod
    def compute_phash(gray):
        r=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA)
        d=cv2.dct(np.float32(r))[:8,:8]; h=0
        for b in (d>d.mean()).flatten(): h=(h<<1)|int(b)
        return format(h,'016x')
    @staticmethod
    def hamming(a,b): return bin(int(a,16)^int(b,16)).count('1')

    def check_single(self,path):
        r={'file':os.path.basename(path),'issues':[],'phash':None,
           'aesthetic':0,'overall':0,'scene':'','suggestions':[],'face_count':0,'face_results':[]}
        img=self._imread(path)
        if img is None: r['issues'].append({'t':'读取失败','c':100,'d':'无法解码'}); return r
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w=gray.shape; md=self.cfg.gi('max_image_dim')
        if max(h,w)>md:
            s=md/max(h,w); gray=cv2.resize(gray,(int(w*s),int(h*s)))
            rgb=cv2.resize(rgb,(int(w*s),int(h*s)))
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
            fd=self.ai.detect_faces(rgb); r['face_count']=len(fd) if fd else 0
            if fd:
                for f in fd[:10]:
                    fq=self.ai.analyze_face(f); r['face_results'].append(fq)
                    if fq['eyes_open']<40: r['issues'].append({'t':'闭眼','c':round(100-fq['eyes_open'],1),'d':f"睁眼度{fq['eyes_open']:.0f}%"})
                    if fq['head_angle']>30: r['issues'].append({'t':'侧脸','c':round(fq['head_angle'],1),'d':f"偏转{fq['head_angle']:.0f}度"})
            r['scene']=self.ai.classify_scene(rgb,r['face_count'])
            q=self.ai.analyze_quality(rgb,gray,r['face_results'])
            r['overall']=q.get('overall',0); r['aesthetic']=q.get('aesthetic',0)
            r['grade']=q.get('grade',''); r['quality']=q
            r['suggestions']=self.ai.suggest(q,r['scene'],r['face_count'])
            if r['overall']<55: r['issues'].append({'t':'综合评分低','c':round(55-r['overall'],1),'d':f"得分{r['overall']:.0f}/100"})
        else:
            fc=cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,'haarcascade_frontalface_default.xml'))
            ec=cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades,'haarcascade_eye.xml'))
            faces=fc.detectMultiScale(gray,1.1,5,minSize=(80,80)); r['face_count']=len(faces)
            if len(faces)>0:
                r['scene']='人像写真' if len(faces)<3 else '集体照'; closed=0
                for (fx,fy,fw,fh) in faces[:3]:
                    e=ec.detectMultiScale(gray[fy:fy+fh//2,fx:fx+fw],1.05,4,minSize=(20,20))
                    if len(e)<2: closed+=1
                if closed>0: r['issues'].append({'t':'闭眼','c':round(closed/min(len(faces),3)*100,1),'d':f"{closed}/{min(len(faces),3)}"})
        try: s=cv2.resize(gray,(128,128)); r['phash']=self.compute_phash(s)
        except: r['phash']=None
        return r

# ═══════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root=root; self.root.title("图片智能分析系统 v2.0")
        self.root.geometry("1200x800"); self.root.minsize(1000,640)
        self.root.configure(bg=BG0)
        self.cfg=Config(); self.checker=PhotoChecker(self.cfg)
        self.results=[]; self.bad_photos=[]; self.dup_groups=[]; self.move_hist=[]
        self.processing=False; self._stop=False; self._paused=False
        self._paused_idx=0; self._paused_files=[]; self._paused_folder=''; self._paused_hm={}
        self._show_params=False
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
        ttk.Separator(tb,orient='vertical').pack(side='left',padx=6,fill='y')

        # 汉堡菜单
        self.btn_menu=tk.Menubutton(tb,text="⚙ 菜单 ▾",font=('微软雅黑',10),bg=BG3,fg=TEXT1,
                                     activebackground=ACCENT,activeforeground='#fff',
                                     relief='flat',bd=0,padx=12,pady=4)
        self.btn_menu.pack(side='right',padx=4)
        menu=tk.Menu(self.btn_menu,tearoff=0,bg=BG2,fg=TEXT1,activebackground='#1a1040',
                     activeforeground=ACCENT,font=('微软雅黑',10))
        menu.add_command(label="🗑 移动废片",command=self._move)
        menu.add_command(label="↩ 撤销移动",command=self._undo)
        menu.add_command(label="📄 导出CSV",command=self._csv)
        menu.add_separator()
        menu.add_command(label="⚙ 参数设置",command=self._toggle_params)
        menu.add_command(label="🤖 AI设置",command=self._show_ai_settings)
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
        self.tdup.heading("f",text="双击查看详情"); self.tdup.column("f",width=100)
        self.tdup.pack(fill='x')
        self.tdup.bind('<Double-1>',self._dup_detail)

        self.st=tk.Label(r,text="v2.0 | SCRFD | AI仅供参考",font=('微软雅黑',8),fg=TEXT2,bg=BG0,anchor='w')
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
            ("美学最低分","aesthetic_min",1.0,9.0)]
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
        self.lbl_ai.config(text="AI: 初始化..."); self.root.update_idletasks()
        self.checker.init_ai(cb=lambda m: self.root.after(0,lambda msg=m:self._aip(msg)))
        if self.checker.ai and self.checker.ai.face_detector:
            self.lbl_ai.config(text="AI: SCRFD ✓",fg=GREEN)
        else: self.lbl_ai.config(text="AI: 标准模式",fg=TEXT2)
    def _aip(self,msg): self.lbl_ai.config(text=f"AI: {msg}"); self.root.update_idletasks()

    def _on_ai_toggle(self):
        if self.ai_var.get():
            if not self.cfg.g('ai_mode'): self._show_ai_mode_dialog()
            else: self.cfg.s('enable_ai','1'); self._init_ai()
        else: self.cfg.s('enable_ai','0')

    def _show_ai_mode_dialog(self):
        dlg=tk.Toplevel(self.root); dlg.title("选择AI运行模式"); dlg.geometry("520x460")
        dlg.configure(bg=BG0); dlg.transient(self.root); dlg.grab_set(); dlg.resizable(False,False)
        tk.Label(dlg,text="选择AI运行模式",font=('微软雅黑',16,'bold'),fg=TEXT1,bg=BG0).pack(pady=(16,8))
        for title,color,items in [
            ("🟢 本地运行模式（推荐）",GREEN,["100%离线，照片永不上传","准确率85-90%","8G内存即可","永久免费"]),
            ("🔴 在线API模式",RED,["准确率95%+","照片需上传第三方","需联网","约0.05-0.1元/张"])]:
            lf=tk.LabelFrame(dlg,text="",bg=BG1,fg=TEXT1); lf.pack(fill='x',padx=16,pady=6)
            tk.Label(lf,text=title,font=('微软雅黑',13,'bold'),fg=color,bg=BG1).pack(anchor='w',padx=10,pady=(8,0))
            for t in items: tk.Label(lf,text=t,font=('微软雅黑',10),fg=TEXT2,bg=BG1).pack(anchor='w',padx=24)
        tk.Label(dlg,text="⚠ 选择在线模式即同意将照片上传至AI服务商服务器\n所有AI分析结果仅供参考，请务必人工确认",
                 font=('微软雅黑',9,'bold'),fg=RED,bg=BG0,justify='center').pack(pady=8)
        bf=tk.Frame(dlg,bg=BG0); bf.pack(pady=(8,12))
        tk.Button(bf,text="选择本地运行",font=('微软雅黑',11,'bold'),bg=GREEN,fg='#fff',padx=20,pady=8,
                  command=lambda: self._pick_ai('local',dlg)).pack(side='left',padx=8)
        tk.Button(bf,text="选择在线运行",font=('微软雅黑',11,'bold'),bg=RED,fg='#fff',padx=20,pady=8,
                  command=lambda: self._pick_ai('online',dlg)).pack(side='left',padx=8)

    def _pick_ai(self,mode,dlg):
        self.cfg.s('ai_mode',mode); self.cfg.s('enable_ai','1'); self.ai_var.set(True); dlg.destroy()
        if mode=='online': self._show_api_config()
        else: self._init_ai()

    def _show_api_config(self):
        dlg=tk.Toplevel(self.root); dlg.title("在线API配置"); dlg.geometry("480x300")
        dlg.configure(bg=BG0); dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg,text="配置在线API",font=('微软雅黑',14,'bold'),fg=TEXT1,bg=BG0).pack(pady=12)
        entries={}
        for lb,key,show in [("API密钥","api_key",'*'),("API端点","api_endpoint",None),("模型名","api_model",None)]:
            tk.Label(dlg,text=lb,font=('微软雅黑',10),fg=TEXT2,bg=BG0).pack()
            e=tk.Entry(dlg,width=50,font=('微软雅黑',10),bg=BG1,fg=TEXT1,insertbackground=TEXT1,relief='flat')
            if show: e.config(show=show)
            e.pack(pady=4); d='';
            if key=='api_endpoint': d='https://api.openai.com/v1/chat/completions'
            elif key=='api_model': d='gpt-4o'
            e.insert(0,self.cfg.g(key) or d); entries[key]=e
        def save():
            for k,e in entries.items(): self.cfg.s(k,e.get())
            dlg.destroy()
        tk.Button(dlg,text="保存配置",font=('微软雅黑',11,'bold'),bg=ACCENT,fg='#fff',padx=24,pady=6,command=save).pack(pady=12)

    def _show_ai_settings(self):
        dlg=tk.Toplevel(self.root); dlg.title("AI模型管理器"); dlg.geometry("700x500")
        dlg.configure(bg=BG0); dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg,text="🤖 AI模型管理器",font=('微软雅黑',16,'bold'),fg=TEXT1,bg=BG0).pack(anchor='w',padx=16,pady=(12,4))
        pan=ttk.PanedWindow(dlg,orient='horizontal'); pan.pack(fill='both',expand=True,padx=12,pady=8)

        lf=tk.Frame(pan,bg=BG1); pan.add(lf,weight=1)
        tk.Label(lf,text="模型列表",font=('微软雅黑',11,'bold'),fg=TEXT1,bg=BG1).pack(anchor='w',padx=10,pady=(8,2))
        models=[
            {'id':'scrfd','name':'SCRFD 2.5GF','cat':'本地','size':'3.6MB ✓','cost':'免费','req':'CPU','speed':'<0.1s/张','stars':5,'note':'人脸检测+关键点，已部署','risk':'无'},
            {'id':'quality','name':'多维质量评分','cat':'本地','size':'0MB','cost':'免费','req':'CPU','speed':'<0.1s/张','stars':4,'note':'技术/构图/人像/美学评分','risk':'无'},
            {'id':'gpt4v','name':'GPT-4V (OpenAI)','cat':'在线','size':'-','cost':'~0.1元/张','req':'联网','speed':'2-4s/张','stars':5,'note':'需API密钥','risk':'照片上传至OpenAI'},
            {'id':'qwen','name':'通义千问VL','cat':'在线','size':'-','cost':'~0.05元/张','req':'联网','speed':'1-3s/张','stars':4,'note':'需阿里云账号','risk':'照片上传至阿里云'},
        ]
        rf=tk.Frame(pan,bg=BG0); pan.add(rf,weight=2)
        detail_frame=tk.Frame(rf,bg=BG0); detail_frame.pack(fill='both',expand=True,padx=8)
        tk.Label(detail_frame,text="← 选择左侧模型查看详情",font=('微软雅黑',11),fg=TEXT2,bg=BG0).pack(expand=True)

        for m in models:
            icon='🟢' if m['cat']=='本地' else '🔴'
            b=tk.Button(lf,text=f"{icon} {m['name']}",font=('微软雅黑',10),bg=BG3,fg=TEXT1,anchor='w',bd=0,padx=10,
                        command=lambda m=m,p=detail_frame: self._show_model(m,p))
            b.pack(fill='x',padx=8,pady=2)

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
        if m['id'] in ('scrfd','quality'):
            tk.Label(bf,text="✅ 已部署运行中",fg=GREEN,bg=BG1,font=('微软雅黑',11,'bold')).pack(side='left')
        else:
            tk.Button(bf,text="配置API密钥",bg=ACCENT,fg='#fff',font=('微软雅黑',10,'bold'),padx=16,pady=6,
                      command=self._show_api_config).pack(side='left')

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
        self.results.clear(); self.bad_photos.clear(); self.dup_groups.clear()
        self.pb['value']=0; self.pb['maximum']=len(fls); self.processing=True
        self._stop=False; self._paused=False; self._paused_idx=0
        self._paused_files=fls; self._paused_folder=f; self._paused_hm={}
        self.bs.config(state='disabled'); self.bpz.config(state='normal')
        self.bre.config(state='disabled'); self.brs.config(state='normal')
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
        for idx in range(start_idx,len(files)):
            if self._stop: break
            while self._paused and not self._stop: self._paused_idx=idx; self._paused_hm=hm; time.sleep(0.1)
            if self._stop: break
            fn=files[idx]; r=self.checker.check_single(os.path.join(folder,fn))
            self.results.append(r)
            if r['issues']: self.bad_photos.append(r)
            sc=f"{r['overall']:.0f}" if r['overall']>0 else '-'
            for iss in r['issues']:
                self.root.after(0,lambda fnm=fn,t=iss['t'],c=iss['c'],s=sc:
                    self.tree.insert('','end',values=(fnm,t,f"{c}%",s)))
            ph=r.get('phash')
            if ph: hm.setdefault(ph,[]).append((idx,fn))
            self.root.after(0,lambda i=idx+1,t=total: self._upd(i,t))
        if not self._paused and not self._stop:
            th=self.cfg.gi('duplicate_threshold'); ks=list(hm.keys()); seen=set()
            for i in range(len(ks)):
                if i in seen: continue
                g=[ks[i]]
                for j in range(i+1,len(ks)):
                    if j in seen: continue
                    if PhotoChecker.hamming(ks[i],ks[j])<=(64-th*64/100): g.append(ks[j]); seen.add(j)
                if len(g)>1:
                    ns=[]
                    for hh in g: ns.extend(fn for _,fn in hm[hh])
                    self.dup_groups.append(ns)
                    self.root.after(0,lambda n=ns: self.tdup.insert('','end',values=(', '.join(n[:4]),)))
        if not self._paused: self.root.after(0,self._done)

    def _upd(self,cur,total):
        self.pb['value']=cur; self.lbl_pb.config(text=f"Scan: {cur}/{total}")
        self.lbl_stats.config(text=f"Bad: {len(self.bad_photos)} | Total: {total}")

    def _done(self):
        self.processing=False
        self.bs.config(state='normal'); self.bpz.config(state='disabled')
        self.bre.config(state='disabled'); self.brs.config(state='normal')
        bad,dup,total=len(self.bad_photos),len(self.dup_groups),len(self.results)
        stp="已停止 · " if self._stop else ""
        self.lbl_pb.config(text=stp+"完成!")
        ps = [f"Bad: {bad}"]
        if dup: ps.append(f"Dup: {dup}")
        ps.append(f"Total: {total}")
        self.lbl_stats.config(text=" | ".join(ps))

    # ═══════════ 预览 ═══════════
    def _load_thumb(self,fp,size=350):
        if not HAS_PIL or not os.path.exists(fp): return None
        for strat in [
            lambda: Image.open(fp),
            lambda: Image.open(fp).convert('RGB'),
        ]:
            try:
                img=strat(); img.thumbnail((size,size),Image.LANCZOS)
                if img.mode not in ('RGB','L'): img=img.convert('RGB')
                return ImageTk.PhotoImage(img)
            except: continue
        # OpenCV fallback
        try:
            import numpy as np
            arr=np.fromfile(fp,dtype=np.uint8)
            bgr=cv2.imdecode(arr,cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                img=Image.fromarray(rgb); img.thumbnail((size,size),Image.LANCZOS)
                return ImageTk.PhotoImage(img)
        except: pass
        return None

    def _preview(self,event):
        sel=self.tree.selection()
        if not sel or not HAS_PIL: return
        idx=self.tree.index(sel[0])
        if idx>=len(self.results): return
        r=self.results[idx]
        fp=os.path.join(self.fv.get().strip(),r['file'])
        photo=self._load_thumb(fp)
        if photo:
            self._preview_img=photo
            self.preview_label.config(image=photo,text='',bg='#000000')
            try:
                img=Image.open(fp); self.lbl_exif.config(text=self._exif(img))
            except: self.lbl_exif.config(text='')
        else:
            size_mb=os.path.getsize(fp)/1024/1024 if os.path.exists(fp) else 0
            self.preview_label.config(image='',text=f"预览失败\n{Path(fp).suffix} {size_mb:.1f}MB",bg='#000000')

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
            tk.Label(lf,text="预览失败",font=('微软雅黑',12),fg='#555',bg='#000000').pack(expand=True)

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

    # ═══════════ 操作 ═══════════
    def _move(self):
        f=self.fv.get().strip()
        if not self.bad_photos: self.st.config(text="无废片可移动"); return
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
        self.root.after(3000,lambda: self.st.config(text="v2.0 | AI仅供参考"))

    def _undo(self):
        if not self.move_hist: self.st.config(text="无可撤销"); return
        r=0
        for s,d in self.move_hist:
            try:
                if os.path.exists(d): shutil.move(d,s); r+=1
            except: pass
        self.move_hist.clear()
        self.st.config(text=f"✓ 已恢复 {r} 张")
        self.root.after(3000,lambda: self.st.config(text="v2.0 | AI仅供参考"))

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

    def _about(self):
        messagebox.showinfo("关于","图片智能分析系统 v2.0\n电竞暗黑美学 · SCRFD 2.5GF\n100%本地离线 · 隐私优先\n\nAI分析仅供参考，请人工确认")

def main():
    root=tk.Tk()
    app=App(root)
    app._bind_events()
    root.mainloop()

if __name__=="__main__": main()
