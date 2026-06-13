"""
图片智能分析系统 v2.0 — 达芬奇风格
SCRFD人脸 · 多维评分 · 场景分类 · AI模型管理
"""
import os, sys, shutil, csv, configparser, threading, time
from datetime import datetime
from pathlib import Path

import cv2, numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from ctypes import windll; windll.shcore.SetProcessDpiAwareness(2)
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
    'max_image_dim':'1200','aesthetic_min':'3.0','enable_ai':'1',
    'ai_mode':'','api_key':'','api_endpoint':'','api_model':'',
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
           'aesthetic':0,'overall':0,'scene':'','suggestions':[],
           'face_count':0,'face_results':[]}
        img=self._imread(path)
        if img is None: r['issues'].append({'t':'读取失败','c':100,'d':'无法解码'}); return r
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w=gray.shape; md=self.cfg.gi('max_image_dim')
        if max(h,w)>md: s=md/max(h,w); gray=cv2.resize(gray,(int(w*s),int(h*s))); rgb=cv2.resize(rgb,(int(w*s),int(h*s)))
        lap=cv2.Laplacian(gray,cv2.CV_64F).var(); bt=self.cfg.gf('blur_threshold')
        if lap<bt:
            bc=''; bc='失焦模糊' if lap<30 else ('运动模糊' if lap<60 else '')
            r['issues'].append({'t':'模糊','c':round(max(0,(1-lap/bt)*100),1),'d':f"方差={lap:.0f} {bc}"})
        hist=cv2.calcHist([gray],[0],None,[256],[0,256]); tot=gray.size
        over=np.sum(hist[self.cfg.gi('over_brightness'):])/tot*100
        under=np.sum(hist[:self.cfg.gi('under_brightness')])/tot*100
        if over>self.cfg.gf('overexposure_pct'): r['issues'].append({'t':'过曝','c':round(over,1),'d':f"{over:.0f}%过亮"})
        if under>self.cfg.gf('underexposure_pct'): r['issues'].append({'t':'欠曝','c':round(under,1),'d':f"{under:.0f}%过暗"})
        if self.ai:
            fd=self.ai.detect_faces(rgb); r['face_count']=len(fd) if fd else 0
            if fd:
                for f in fd[:10]:
                    fq=self.ai.analyze_face(f); r['face_results'].append(fq)
                    if fq['eyes_open']<40: r['issues'].append({'t':'闭眼','c':round(100-fq['eyes_open'],1),'d':f"睁眼度{fq['eyes_open']:.0f}%"})
                    if fq['head_angle']>30: r['issues'].append({'t':'侧脸','c':round(fq['head_angle'],1),'d':f"偏转{fq['head_angle']:.0f}°"})
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
# 达芬奇风格 GUI
# ═══════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root=root; self.root.title("图片智能分析系统 v2.0"); self.root.geometry("1200x800"); self.root.minsize(1000,640)
        self.cfg=Config(); self.checker=PhotoChecker(self.cfg)
        self.results=[]; self.bad_photos=[]; self.dup_groups=[]; self.move_hist=[]
        self.processing=False; self._stop=False; self._paused=False
        self._paused_idx=0; self._paused_files=[]; self._paused_folder=''; self._paused_hm={}
        self._show_params=False; self._menu_open=False
        self.fv=tk.StringVar()  # 不保存last_folder
        self._build(); self._bind_events()
        self.root.after(600,self._init_ai)

    # ═══════════ UI 构建 ═══════════
    def _build(self):
        r=self.root
        if HAS_TKB: self.style=tkb.Style(theme='darkly')
        else:
            self.style=ttk.Style(); self.style.theme_use('clam'); r.configure(bg='#1E1E1E')
            for w in ['.','TFrame','TLabel','TLabelframe','TButton','TProgressbar','Treeview']:
                try: self.style.configure(w,background='#1E1E1E',foreground='#D4D4D4',fieldbackground='#2D2D2D')
                except: pass
            self.style.map('Treeview',background=[('selected','#094771')])

        # ── 达芬奇风格工具栏 ──
        tb=ttk.Frame(r); tb.pack(fill='x',padx=8,pady=(6,0))
        ttk.Button(tb,text="📁 选择文件夹",command=self._browse).pack(side='left')
        ttk.Separator(tb,orient='vertical').pack(side='left',padx=6,fill='y')
        self.bs=ttk.Button(tb,text="▶ 开始扫描",command=self._start); self.bs.pack(side='left',padx=4)
        self.bpz=ttk.Button(tb,text="⏸ 暂停",command=self._pause_scan,state='disabled'); self.bpz.pack(side='left',padx=4)
        self.bre=ttk.Button(tb,text="⏵ 继续",command=self._resume_scan,state='disabled'); self.bre.pack(side='left',padx=4)
        self.brs=ttk.Button(tb,text="🔄 重新扫描",command=self._re_scan,state='disabled'); self.brs.pack(side='left',padx=4)
        ttk.Separator(tb,orient='vertical').pack(side='left',padx=6,fill='y')

        # 汉堡菜单
        self.btn_menu=tk.Menubutton(tb,text="⚙ 菜单 ▾",font=('微软雅黑',10),bg='#3C3C3C',fg='#D4D4D4',
                                     activebackground='#555',activeforeground='#fff',
                                     relief='flat',bd=0,padx=12,pady=4)
        self.btn_menu.pack(side='right',padx=4)
        menu=tk.Menu(self.btn_menu,tearoff=0,bg='#2D2D2D',fg='#D4D4D4',activebackground='#094771',
                     activeforeground='#fff',font=('微软雅黑',10))
        menu.add_command(label="🗑 移动废片",command=self._move)
        menu.add_command(label="↩ 撤销移动",command=self._undo)
        menu.add_command(label="📄 导出CSV",command=self._csv)
        menu.add_separator()
        menu.add_command(label="⚙ 参数设置",command=self._toggle_params)
        menu.add_command(label="🤖 AI设置",command=self._show_ai_settings)
        menu.add_separator()
        menu.add_command(label="ℹ 关于",command=self._about)
        self.btn_menu.config(menu=menu)
        self._menu=menu

        # ── 参数侧滑面板 ──
        self.param_slide=tk.Frame(r,bg='#252526',bd=0,highlightthickness=0)
        self._build_param_panel()

        # 统计+进度
        sf=ttk.Frame(r); sf.pack(fill='x',padx=8,pady=(6,0))
        self.lbl_stats=ttk.Label(sf,text="就绪",font=('微软雅黑',10)); self.lbl_stats.pack(side='left')
        self.lbl_ai=ttk.Label(sf,text="AI: 初始化中...",foreground='#888'); self.lbl_ai.pack(side='right')
        self.pb=ttk.Progressbar(r,mode='determinate'); self.pb.pack(fill='x',padx=8,pady=2)
        self.lbl_pb=ttk.Label(r,text=""); self.lbl_pb.pack(anchor='w',padx=12)

        # 主区域
        main=ttk.PanedWindow(r,orient='horizontal'); main.pack(fill='both',expand=True,padx=8,pady=(4,8))

        # 结果列表（精简：文件名/问题/置信度/综合分）
        tf=ttk.Frame(main); main.add(tf,weight=3)
        cols=("fn","issue","conf","score")
        self.tree=ttk.Treeview(tf,columns=cols,show='headings',selectmode='browse')
        self.tree.heading("fn",text="文件名"); self.tree.heading("issue",text="问题")
        self.tree.heading("conf",text="置信度"); self.tree.heading("score",text="综合分")
        self.tree.column("fn",width=200,minwidth=120); self.tree.column("issue",width=120,minwidth=80)
        self.tree.column("conf",width=80,minwidth=60); self.tree.column("score",width=80,minwidth=60)
        ts=ttk.Scrollbar(tf,orient='vertical',command=self.tree.yview)
        self.tree.configure(yscrollcommand=ts.set)
        self.tree.pack(side='left',fill='both',expand=True); ts.pack(side='right',fill='y')
        self.tree.bind('<<TreeviewSelect>>',self._preview)
        self.tree.bind('<Double-1>',self._full_preview)

        # 右侧面板
        rf=ttk.Frame(main); main.add(rf,weight=1)
        plf=ttk.LabelFrame(rf,text="预览",padding=2); plf.pack(fill='both',expand=True)
        self.preview_frame=tk.Frame(plf,bg='#000000'); self.preview_frame.pack(fill='both',expand=True)
        self.prev_view=tk.Frame(self.preview_frame,bg='#000000'); self.prev_view.pack(fill='both',expand=True)
        self.lbl_preview=tk.Label(self.prev_view,text="单击预览 · 双击大图\n━━━━━━━━━━━━\n双击重复组查看详情",
                                   bg='#000000',fg='#666',font=('微软雅黑',11))
        self.lbl_preview.pack(expand=True); self._preview_img=None
        self.rep_view=tk.Frame(self.preview_frame,bg='#1E1E1E'); self._showing_report=False
        self.lbl_exif=ttk.Label(rf,text="",font=('微软雅黑',8),foreground='#888'); self.lbl_exif.pack(fill='x',pady=(2,0))
        dlf=ttk.LabelFrame(rf,text="重复组",padding=2); dlf.pack(fill='x',pady=(4,0))
        self.tdup=ttk.Treeview(dlf,columns=("f",),show='headings',height=3)
        self.tdup.heading("f",text="相似组"); self.tdup.column("f",width=100); self.tdup.pack(fill='x')
        self.tdup.bind('<Double-1>',self._dup_detail)
        self.st=ttk.Label(r,text="v2.0 | SCRFD | AI仅供参考",relief='sunken',font=('微软雅黑',8))
        self.st.pack(side='bottom',fill='x')

    def _build_param_panel(self):
        pf=self.param_slide
        for w in pf.winfo_children(): w.destroy()
        hdr=tk.Frame(pf,bg='#252526'); hdr.pack(fill='x',padx=12,pady=(10,4))
        tk.Label(hdr,text="⚙ 检测参数",font=('微软雅黑',13,'bold'),fg='#D4D4D4',bg='#252526').pack(side='left')
        tk.Button(hdr,text="✕",font=('微软雅黑',11),bg='#252526',fg='#888',bd=0,
                  command=self._hide_params).pack(side='right')

        # AI开关
        af=tk.Frame(pf,bg='#252526'); af.pack(fill='x',padx=12,pady=4)
        self.ai_var=tk.BooleanVar(value=self.cfg.gb('enable_ai'))
        tk.Checkbutton(af,text="🤖 AI智能分析",variable=self.ai_var,command=self._on_ai_toggle,
                        bg='#252526',fg='#D4D4D4',selectcolor='#252526',
                        activebackground='#252526',activeforeground='#D4D4D4',
                        font=('微软雅黑',10)).pack(side='left')

        pd=[("模糊阈值","blur_threshold",20,300,5),("过曝判定%","overexposure_pct",30,100,5),
            ("欠曝判定%","underexposure_pct",5,50,1),("过亮值","over_brightness",200,255,5),
            ("过暗值","under_brightness",5,80,5),("重复相似度%","duplicate_threshold",80,100,1),
            ("美学最低分","aesthetic_min",1.0,9.0,0.5)]
        self.sls={}
        for lb,k,lo,hi,_ in pd:
            f=tk.Frame(pf,bg='#252526'); f.pack(fill='x',padx=12,pady=3)
            tk.Label(f,text=lb,font=('微软雅黑',9),fg='#aaa',bg='#252526').pack(anchor='w')
            vf=tk.Frame(f,bg='#252526'); vf.pack(fill='x')
            tk.Label(vf,text=str(lo),font=('微软雅黑',7),fg='#666',bg='#252526').pack(side='left')
            fl=isinstance(lo,float); sv=tk.DoubleVar(value=self.cfg.gf(k)) if fl else tk.IntVar(value=self.cfg.gi(k))
            vl=tk.Label(vf,text="",font=('微软雅黑',8,'bold'),fg='#ff9500',bg='#252526',width=6)
            vl.pack(side='right'); tk.Label(vf,text=str(hi),font=('微软雅黑',7),fg='#666',bg='#252526').pack(side='right')
            sv.trace_add('write',lambda *a,l=vl,f=fl,s=sv: l.config(text=f"{float(s.get()):.1f}" if f else f"{int(s.get())}"))
            tk.Scale(vf,from_=lo,to=hi,variable=sv,orient='horizontal',bg='#252526',fg='#D4D4D4',
                     troughcolor='#3C3C3C',highlightthickness=0,bd=0).pack(fill='x')
            self.sls[k]=(sv,fl); vl.config(text=f"{float(sv.get()):.1f}" if fl else f"{int(sv.get())}")
        bf=tk.Frame(pf,bg='#252526'); bf.pack(fill='x',padx=12,pady=(8,12))
        tk.Button(bf,text="恢复默认",bg='#3C3C3C',fg='#D4D4D4',command=self._rp).pack(side='left')
        self._saved_lbl=tk.Label(bf,text="",fg='#4cd964',bg='#252526',font=('微软雅黑',9))
        self._saved_lbl.pack(side='left',padx=12)
        tk.Button(bf,text="保存参数",bg='#165DFF',fg='#fff',command=self._sp).pack(side='right')

    # ═══════════ 交互 ═══════════
    def _bind_events(self):
        self.root.bind('<Button-1>',self._click_outside)
        self.root.protocol('WM_DELETE_WINDOW',self._on_close)

    def _click_outside(self,event):
        if self._show_params:
            w=self.param_slide
            if not (w.winfo_rootx()<=event.x_root<=w.winfo_rootx()+w.winfo_width() and
                    w.winfo_rooty()<=event.y_root<=w.winfo_rooty()+w.winfo_height()):
                self._hide_params()

    def _on_close(self):
        self.cfg.s('last_folder','')  # 不保存路径
        self.root.destroy()

    # ═══════════ 参数面板 ═══════════
    def _toggle_params(self):
        if self._show_params: self._hide_params()
        else: self._show_param_panel()

    def _show_param_panel(self):
        self.param_slide.place(relx=1.0,rely=0.0,anchor='ne',width=360,relheight=1.0)
        self.param_slide.lift(); self._show_params=True

    def _hide_params(self):
        self.param_slide.place_forget(); self._show_params=False

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
            self.lbl_ai.config(text="AI: SCRFD ✓",foreground='#4cd964')
        else: self.lbl_ai.config(text="AI: 标准模式",foreground='#888')
    def _aip(self,msg): self.lbl_ai.config(text=f"AI: {msg}"); self.root.update_idletasks()

    def _on_ai_toggle(self):
        if self.ai_var.get():
            if not self.cfg.g('ai_mode'):
                self._show_ai_mode_dialog()
            else:
                self.cfg.s('enable_ai','1'); self._init_ai()
                self.st.config(text="AI已启用 | 模式: "+self.cfg.g('ai_mode'))
        else:
            self.cfg.s('enable_ai','0')
            self.st.config(text="AI已关闭")

    def _show_ai_mode_dialog(self):
        dlg=tk.Toplevel(self.root); dlg.title("选择AI运行模式"); dlg.geometry("540x480")
        dlg.configure(bg='#1E1E1E'); dlg.transient(self.root); dlg.grab_set(); dlg.resizable(False,False)
        tk.Label(dlg,text="选择AI运行模式",font=('微软雅黑',16,'bold'),fg='#D4D4D4',bg='#1E1E1E').pack(pady=(16,8))
        lf=tk.LabelFrame(dlg,text="",bg='#252526'); lf.pack(fill='x',padx=16,pady=6)
        tk.Label(lf,text="🟢 本地运行模式（推荐·隐私优先）",font=('微软雅黑',13,'bold'),fg='#4cd964',bg='#252526').pack(anchor='w',padx=10,pady=(8,0))
        for t in ["✅ 100%离线，照片永不上传","✅ 准确率85-90%","💻 8G内存即可","💰 永久免费"]:
            tk.Label(lf,text=t,font=('微软雅黑',10),fg='#aaa',bg='#252526').pack(anchor='w',padx=24)
        of=tk.LabelFrame(dlg,text="",bg='#252526'); of.pack(fill='x',padx=16,pady=6)
        tk.Label(of,text="🔴 在线API模式（效果优先）",font=('微软雅黑',13,'bold'),fg='#ff6b35',bg='#252526').pack(anchor='w',padx=10,pady=(8,0))
        for t in ["✅ 准确率95%+","⚠️ 照片需上传至第三方","💻 只需联网","💰 约0.05-0.1元/张"]:
            tk.Label(of,text=t,font=('微软雅黑',10),fg='#aaa',bg='#252526').pack(anchor='w',padx=24)
        tk.Label(dlg,text="⚠ 选择在线模式即同意将照片上传至AI服务商服务器\n所有AI分析结果仅供参考，请务必人工确认",
                 font=('微软雅黑',9,'bold'),fg='#ff375f',bg='#1E1E1E',justify='center').pack(pady=8)
        bf=tk.Frame(dlg,bg='#1E1E1E'); bf.pack(pady=(8,12))
        tk.Button(bf,text="选择本地运行",font=('微软雅黑',11,'bold'),bg='#4cd964',fg='#fff',padx=20,pady=8,
                  command=lambda: self._pick_ai_mode('local',dlg)).pack(side='left',padx=8)
        tk.Button(bf,text="选择在线运行",font=('微软雅黑',11,'bold'),bg='#ff6b35',fg='#fff',padx=20,pady=8,
                  command=lambda: self._pick_ai_mode('online',dlg)).pack(side='left',padx=8)

    def _pick_ai_mode(self,mode,dlg):
        self.cfg.s('ai_mode',mode); self.cfg.s('enable_ai','1'); self.ai_var.set(True); dlg.destroy()
        if mode=='online': self._show_api_config()
        else: self._init_ai()

    def _show_api_config(self):
        dlg=tk.Toplevel(self.root); dlg.title("在线API配置"); dlg.geometry("480x320")
        dlg.configure(bg='#1E1E1E'); dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg,text="配置在线API",font=('微软雅黑',14,'bold'),fg='#D4D4D4',bg='#1E1E1E').pack(pady=12)
        for lb,key,show in [("API密钥","api_key",'*'),("API端点","api_endpoint",None),("模型名","api_model",None)]:
            tk.Label(dlg,text=lb,font=('微软雅黑',10),fg='#888',bg='#1E1E1E').pack()
            e=tk.Entry(dlg,width=50,font=('微软雅黑',10))
            if show: e.config(show=show)
            e.pack(pady=4); e.insert(0,self.cfg.g(key) or ({'api_endpoint':'https://api.openai.com/v1/chat/completions','api_model':'gpt-4o'}.get(key,'')))
            setattr(self,f'_api_{key}',e)
        def save():
            for key in ['api_key','api_endpoint','api_model']:
                self.cfg.s(key,getattr(self,f'_api_{key}').get())
            dlg.destroy()
        tk.Button(dlg,text="保存配置",font=('微软雅黑',11,'bold'),bg='#165DFF',fg='#fff',padx=24,pady=6,command=save).pack(pady=12)

    def _show_ai_settings(self):
        """AI模型管理器（独立模态窗口）"""
        dlg=tk.Toplevel(self.root); dlg.title("AI模型管理器"); dlg.geometry("700x520")
        dlg.configure(bg='#1E1E1E'); dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg,text="🤖 AI模型管理器",font=('微软雅黑',16,'bold'),fg='#D4D4D4',bg='#1E1E1E').pack(anchor='w',padx=16,pady=(12,4))
        tk.Label(dlg,text="选择和管理AI模型",font=('微软雅黑',10),fg='#888',bg='#1E1E1E').pack(anchor='w',padx=16)

        pan=ttk.PanedWindow(dlg,orient='horizontal'); pan.pack(fill='both',expand=True,padx=12,pady=8)

        # 左侧模型列表
        lf=tk.Frame(pan,bg='#252526'); pan.add(lf,weight=1)
        tk.Label(lf,text="本地模型",font=('微软雅黑',11,'bold'),fg='#D4D4D4',bg='#252526').pack(anchor='w',padx=10,pady=(8,2))
        self._models=[
            {'id':'scrfd','name':'SCRFD 2.5GF','cat':'本地','size':'3MB','cost':'免费','req':'CPU','speed':'<0.1s/张',
             'for':'人脸检测+关键点','stars':5,'note':'已集成，默认启用','risk':'无'},
            {'id':'aesthetic','name':'美学评分引擎','cat':'本地','size':'0MB','cost':'免费','req':'CPU','speed':'<0.1s/张',
             'for':'构图/色彩/曝光/清晰度','stars':4,'note':'启发式算法','risk':'无'},
        ]
        self._model_btns={}
        for m in self._models:
            b=tk.Button(lf,text=f"{'🟢' if m['id']=='scrfd' else '⚪'} {m['name']}",
                        font=('微软雅黑',10),bg='#333',fg='#D4D4D4',anchor='w',bd=0,padx=10,
                        command=lambda m=m: self._show_model_detail(m,rf_inner))
            b.pack(fill='x',padx=8,pady=2); self._model_btns[m['id']]=b

        tk.Label(lf,text="在线API",font=('微软雅黑',11,'bold'),fg='#D4D4D4',bg='#252526').pack(anchor='w',padx=10,pady=(12,2))
        api_models=[
            {'id':'gpt4v','name':'GPT-4V (OpenAI)','cat':'在线','size':'-','cost':'~0.1元/张','req':'联网','speed':'2-4s/张',
             'for':'最高准确率','stars':5,'note':'需要API密钥','risk':'照片上传至OpenAI'},
            {'id':'qwen','name':'通义千问VL (阿里)','cat':'在线','size':'-','cost':'~0.05元/张','req':'联网','speed':'1-3s/张',
             'for':'国内高性价比','stars':4,'note':'需阿里云账号','risk':'照片上传至阿里云'},
            {'id':'ernie','name':'文心一言VL (百度)','cat':'在线','size':'-','cost':'有免费额度','req':'联网','speed':'1-3s/张',
             'for':'国内稳定','stars':3,'note':'需百度账号','risk':'照片上传至百度'},
        ]
        for m in api_models:
            self._models.append(m)
            b=tk.Button(lf,text=f"🔴 {m['name']}",font=('微软雅黑',10),bg='#333',fg='#D4D4D4',anchor='w',bd=0,padx=10,
                        command=lambda m=m: self._show_model_detail(m,rf_inner))
            b.pack(fill='x',padx=8,pady=2); self._model_btns[m['id']]=b

        # 右侧详情
        rf=tk.Frame(pan,bg='#1E1E1E'); pan.add(rf,weight=2)
        rf_inner=tk.Frame(rf,bg='#1E1E1E'); rf_inner.pack(fill='both',expand=True,padx=8)
        tk.Label(rf_inner,text="← 选择左侧模型查看详情",font=('微软雅黑',11),fg='#888',bg='#1E1E1E').pack(expand=True)

    def _show_model_detail(self,m,parent):
        for w in parent.winfo_children(): w.destroy()
        card=tk.Frame(parent,bg='#252526'); card.pack(fill='both',expand=True,padx=4,pady=4)
        tk.Label(card,text=f"📦 {m['name']}",font=('微软雅黑',14,'bold'),fg='#D4D4D4',bg='#252526').pack(anchor='w',padx=14,pady=(12,4))
        stars='★'*m['stars']+'☆'*(5-m['stars'])
        rows=[('大小',m['size']),('费用',m['cost']),('配置',m['req']),('速度',m['speed']),
              ('适合',m['for']),('推荐',stars),('备注',m['note']),('风险',m['risk'])]
        for lb,v in rows:
            rf=tk.Frame(card,bg='#252526'); rf.pack(fill='x',padx=14,pady=3)
            tk.Label(rf,text=f"🔹 {lb}：",font=('微软雅黑',10,'bold'),fg='#aaa',bg='#252526').pack(side='left')
            tk.Label(rf,text=v,font=('微软雅黑',10),fg='#ff9500' if lb=='风险' else '#D4D4D4',bg='#252526',wraplength=300).pack(side='left')

        bf=tk.Frame(card,bg='#252526'); bf.pack(fill='x',padx=14,pady=(12,8))
        if m['id']=='scrfd':
            tk.Label(bf,text="✅ 已启用",fg='#4cd964',bg='#252526',font=('微软雅黑',11,'bold')).pack(side='left')
        else:
            tk.Button(bf,text="下载并启用",bg='#165DFF',fg='#fff',font=('微软雅黑',10,'bold'),padx=16,pady=6,
                      command=lambda m=m: self._download_model(m)).pack(side='left')

    def _download_model(self,m):
        dlg=tk.Toplevel(self.root); dlg.title(f"下载 {m['name']}"); dlg.geometry("400x200")
        dlg.configure(bg='#1E1E1E'); dlg.transient(self.root); dlg.grab_set()
        tk.Label(dlg,text=f"正在下载 {m['name']}...",font=('微软雅黑',12,'bold'),fg='#D4D4D4',bg='#1E1E1E').pack(pady=(20,8))
        pb=ttk.Progressbar(dlg,mode='indeterminate'); pb.pack(fill='x',padx=30,pady=8); pb.start()
        tk.Label(dlg,text="请稍候...",font=('微软雅黑',10),fg='#888',bg='#1E1E1E').pack()
        def done():
            pb.stop(); dlg.destroy()
            messagebox.showinfo("完成",f"{m['name']} 下载完成！\n请在API配置中设置密钥后使用。")
        dlg.after(1500,done)

    # ═══════════ 扫描控制 ═══════════
    def _browse(self):
        p=filedialog.askdirectory(title="选择照片文件夹")
        if p: self.fv.set(p)
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
        self.bs.config(state='disabled'); self.bpz.config(state='normal')
        self.bre.config(state='disabled'); self.brs.config(state='normal')
        self.lbl_pb.config(text="扫描中...")
        threading.Thread(target=self._proc,args=(f,fls,0,{}),daemon=True).start()
    def _pause_scan(self):
        self._paused=True; self.bpz.config(state='disabled'); self.bre.config(state='normal')
        self.lbl_pb.config(text="已暂停 — 点击继续")
    def _resume_scan(self):
        self._paused=False; self.bpz.config(state='normal'); self.bre.config(state='disabled')
        self.lbl_pb.config(text="扫描中...")
        threading.Thread(target=self._proc,args=(self._paused_folder,self._paused_files,self._paused_idx,self._paused_hm),daemon=True).start()
    def _re_scan(self):
        self._start()
    def _stop_scan(self):
        self._stop=True; self._paused=False

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
        self.bs.config(state='normal'); self.bpz.config(state='disabled')
        self.bre.config(state='disabled'); self.brs.config(state='normal')
        bad,dup,total=len(self.bad_photos),len(self.dup_groups),len(self.results)
        stp="已停止 · " if self._stop else ""
        self.lbl_pb.config(text=stp+"扫描完成！")
        ps = [f"废片: {bad}"]
        if dup: ps.append(f"重复组: {dup}")
        ps.append(f"总计: {total}")
        self.lbl_stats.config(text="  |  ".join(ps))
        self.st.config(text=f"v2.0 | {total}张 | AI仅供参考")

    # ═══════════ 预览 ═══════════
    def _preview(self,event):
        if self._showing_report: return
        sel=self.tree.selection()
        if not sel or not HAS_PIL: return
        idx=self.tree.index(sel[0])
        if idx>=len(self.results): return
        r=self.results[idx]; fp=os.path.join(self.fv.get().strip(),r['file'])
        if not os.path.exists(fp):
            self.lbl_preview.config(image='',text="文件不存在",bg='#000000'); return
        try:
            img=Image.open(fp); img.draft('RGB',(350,350))  # 加速加载
            exif=self._exif(img); img.thumbnail((350,350),Image.LANCZOS)
            if img.mode in ('RGBA','P'): img=img.convert('RGB')
            photo=ImageTk.PhotoImage(img); self._preview_img=photo
            self.lbl_preview.config(image=photo,text='',bg='#000000'); self.lbl_exif.config(text=exif)
        except Exception as e:
            self.lbl_preview.config(image='',text=f"预览失败\n{Path(fp).suffix}",bg='#000000')

    def _full_preview(self,event):
        """双击显示完整分析报告窗口"""
        sel=self.tree.selection()
        if not sel: return
        idx=self.tree.index(sel[0])
        if idx>=len(self.results): return
        r=self.results[idx]; fp=os.path.join(self.fv.get().strip(),r['file'])
        if not os.path.exists(fp): return

        win=tk.Toplevel(self.root); win.title(f"分析报告 - {r['file'][:50]}")
        win.geometry("1000x700"); win.configure(bg='#1E1E1E')
        win.transient(self.root)

        # 顶部返回栏
        hdr=tk.Frame(win,bg='#252526',height=40); hdr.pack(fill='x')
        tk.Button(hdr,text="← 返回列表",font=('微软雅黑',10),bg='#333',fg='#D4D4D4',bd=0,
                  command=win.destroy).pack(side='left',padx=12,pady=6)
        tk.Label(hdr,text=r['file'][:60],font=('微软雅黑',10,'bold'),fg='#D4D4D4',bg='#252526').pack(side='left',padx=12)

        # 左右分栏
        pan=ttk.PanedWindow(win,orient='horizontal'); pan.pack(fill='both',expand=True)

        # 左侧：图片
        lf=tk.Frame(pan,bg='#000000'); pan.add(lf,weight=1)
        try:
            if HAS_PIL and os.path.exists(fp):
                img=Image.open(fp); img.thumbnail((480,600),Image.LANCZOS)
                photo=ImageTk.PhotoImage(img); il=tk.Label(lf,image=photo,bg='#000000')
                il.image=photo; il.pack(expand=True)
            else:
                tk.Label(lf,text="无法预览",font=('微软雅黑',12),fg='#666',bg='#000000').pack(expand=True)
        except Exception as e:
            tk.Label(lf,text=f"预览失败\n{e}",font=('微软雅黑',10),fg='#666',bg='#000000').pack(expand=True)

        # 右侧：分析报告
        rf=tk.Frame(pan,bg='#1E1E1E'); pan.add(rf,weight=1)
        cv=tk.Canvas(rf,bg='#1E1E1E',highlightthickness=0)
        sb=tk.Scrollbar(rf,orient='vertical',command=cv.yview)
        inner=tk.Frame(cv,bg='#1E1E1E'); cv.create_window((0,0),window=inner,anchor='nw',width=460)
        inner.bind('<Configure>',lambda e:cv.configure(scrollregion=cv.bbox('all')))
        cv.configure(yscrollcommand=sb.set); cv.pack(side='left',fill='both',expand=True); sb.pack(side='right',fill='y')

        ov=r.get('overall',0); gd=r.get('grade','-')
        gc='#4cd964' if ov>=90 else '#ff9500' if ov>=75 else '#ff6b35' if ov>=60 else '#ff375f'
        tk.Label(inner,text=f"综合评分：{ov:.0f}/100  {gd}",font=('微软雅黑',22,'bold'),fg=gc,bg='#1E1E1E').pack(anchor='w',padx=16,pady=(16,6))
        tk.Label(inner,text=f"🏷️ {r.get('scene','-')}  |  👤 {r.get('face_count',0)}人  |  🎨 美学{r.get('aesthetic',0):.1f}/10",
                 font=('微软雅黑',11),fg='#888',bg='#1E1E1E').pack(anchor='w',padx=16,pady=2)

        q=r.get('quality',{})
        if q:
            lfq=tk.LabelFrame(inner,text=" 质量维度 ",fg='#D4D4D4',bg='#1E1E1E',font=('微软雅黑',11))
            lfq.pack(fill='x',padx=14,pady=8)
            for k,lab in [('tech','📊 技术质量'),('composition','🖼️ 构图'),('portrait','👤 人像')]:
                d=q.get(k,{})
                if d and d.get('overall',0)>0:
                    # 进度条可视化
                    pf=tk.Frame(lfq,bg='#1E1E1E'); pf.pack(fill='x',padx=12,pady=3)
                    tk.Label(pf,text=f"{lab}：{d['overall']:.0f}/100",font=('微软雅黑',10),fg='#D4D4D4',bg='#1E1E1E').pack(side='left')
                    bar=tk.Canvas(pf,width=120,height=10,bg='#333',highlightthickness=0)
                    bar.pack(side='right'); bar.create_rectangle(0,0,120*d['overall']/100,10,fill='#ff9500',outline='')

        if r['issues']:
            il=tk.LabelFrame(inner,text=" ⚠️ 检测问题 ",fg='#ff6b35',bg='#1E1E1E',font=('微软雅黑',11))
            il.pack(fill='x',padx=14,pady=8)
            for iss in r['issues']:
                tk.Label(il,text=f"⚠ {iss['t']} ({iss['c']}%) — {iss.get('d','')}",
                         fg='#ff6b35',bg='#1E1E1E',font=('微软雅黑',10),wraplength=420).pack(anchor='w',padx=12,pady=2)

        sug=r.get('suggestions',[])
        if sug:
            sl=tk.LabelFrame(inner,text=" 💡 AI改进建议 ",fg='#4cd964',bg='#1E1E1E',font=('微软雅黑',11))
            sl.pack(fill='x',padx=14,pady=8)
            for s in sug:
                tk.Label(sl,text=f"• {s}",fg='#D4D4D4',bg='#1E1E1E',wraplength=420,font=('微软雅黑',10)).pack(anchor='w',padx=12,pady=2)

        tk.Label(inner,text="⚠ AI分析存在误差，请人工复核确认",fg='#555',bg='#1E1E1E',font=('微软雅黑',9)).pack(pady=(12,16))

    def _exif(self,img):
        try:
            exif={ExifTags.TAGS[k]:v for k,v in img._getexif().items() if k in ExifTags.TAGS} if img._getexif() else {}
            parts=[]
            for k,lbl in [('Model',''),('ISOSpeedRatings','ISO'),('FNumber','f/'),('ExposureTime',''),('FocalLength','mm')]:
                if k in exif:
                    v=exif[k]; v=float(v) if isinstance(v,tuple) else v
                    if k=='FNumber': v=f"{float(v):.1f}"
                    if k=='FocalLength': v=f"{float(v):.0f}"
                    parts.append(f"{lbl}{v}")
            return ' | '.join(parts) if parts else ''
        except: return ''

    def _dup_detail(self,event):
        """双击重复组查看详情"""
        sel=self.tdup.selection()
        if not sel: return
        idx=self.tdup.index(sel[0])
        if idx>=len(self.dup_groups): return
        files=self.dup_groups[idx]
        win=tk.Toplevel(self.root); win.title(f"重复组详情 ({len(files)}张)"); win.geometry("500x500")
        win.configure(bg='#1E1E1E')
        tk.Button(win,text="← 返回",font=('微软雅黑',10),bg='#333',fg='#D4D4D4',bd=0,
                  command=win.destroy).pack(anchor='w',padx=12,pady=8)
        tk.Label(win,text=f"重复照片组 — {len(files)}张相似照片",font=('微软雅黑',12,'bold'),fg='#D4D4D4',bg='#1E1E1E').pack(anchor='w',padx=12)
        for fn in files:
            tk.Label(win,text=f"📷 {fn}",font=('微软雅黑',10),fg='#aaa',bg='#1E1E1E').pack(anchor='w',padx=24,pady=2)

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
        self.root.after(3000,lambda: self.st.config(text=f"v2.0 | {len(self.results)}张 | AI仅供参考"))

    def _undo(self):
        if not self.move_hist: self.st.config(text="无可撤销操作"); return
        r=0
        for s,d in self.move_hist:
            try:
                if os.path.exists(d): shutil.move(d,s); r+=1
            except: pass
        self.move_hist.clear()
        self.st.config(text=f"✓ 已恢复 {r} 张")
        self.root.after(3000,lambda: self.st.config(text=f"v2.0 | {len(self.results)}张 | AI仅供参考"))

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
        self.st.config(text=f"✓ 已导出CSV"); self.root.after(3000,lambda: self.st.config(text=f"v2.0 | AI仅供参考"))

    def _about(self):
        messagebox.showinfo("关于","图片智能分析系统 v2.0\n达芬奇风格 · SCRFD 2.5GF\n隐私优先 · 100%离线\n\nAI分析仅供参考，请人工确认")

def main():
    try: from tkinterdnd2 import TkinterDnD; root=TkinterDnD.Tk()
    except: root=tk.Tk()
    App(root); root.mainloop()

if __name__=="__main__": main()