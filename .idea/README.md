# 图片智能分析系统 v4.4

专业AI图片质量分析工具，电竞暗黑美学。自动检测模糊/闭眼/曝光/重复，AI多维评分+场景分类，本地离线运行。

## v4.4 — 统一版本常量 + 预览修复 + 模型自动下载

**核心改进：**
- 🎨 **电竞暗黑美学重构** — 统一配色令牌(BG0/BG1/BG2/BG3+TEXT1/TEXT2+ACCENT/CYAN/PURPLE)，_setup_style集中应用
- 🖼️ **图片预览修复** — 3层回退策略(PIL→RGB转换→OpenCV解码)，兼容11MB+大图
- 👁️ **双击分析报告** — 独立大窗口，左侧大图+右侧完整报告(评分/维度进度条/问题/AI建议)
- 🗂️ **汉堡菜单收纳** — 移动废片/撤销/CSV/参数/AI设置/关于统一收纳
- 📐 **参数侧滑面板** — 右侧滑出，点击外部自动收起，滑块带min/max/实时数值
- 🤖 **AI模型管理器** — 左模型列表+右详情卡片，显示部署状态
- 🔧 **图片加载修复** — `_load_thumb()`独立方法，多格式兼容
- 📋 **列表精简** — 文件名/问题/置信度/综合分四列
- 🔒 **隐私保护** — 文件夹路径不保存

**Bug修复：**
- SCRFD模型下载链接修复(HF 404→GitHub Release)
- 跳过ttkbootstrap覆盖，强制自定义配色
- AI建议去重，按场景/维度生成不同建议(最多4条)
- 双击事件与单击分离
- 编码问题修复

---

## v2.0

达芬奇风格UI · AI模式选择(本地/在线) · 暂停续扫 · SCRFD人脸 · 多维评分 · 场景分类 · 汉堡菜单 · 参数侧滑面板 · AI模型管理器 · 独立分析报告窗口

---

## v1.2

ONNX AI引擎(UltraLight) · 美学评分 · 模糊分类(失焦/运动/虚化) · 停止按钮 · 人脸质量分

---

## v1.1

参数面板+滑块 · ttkbootstrap深色主题 · pHash重复检测 · CSV导出 · 撤销移动 · 图片预览 · 高DPI适配

---

## v1.0

OpenCV Haar三检测(模糊/闭眼/曝光) + tkinter GUI · 一键移动废片

---

## 运行
```bash
py -m pip install opencv-python numpy ttkbootstrap Pillow onnxruntime
py photo_checker.py
```

## 项目结构
```
photo-checker/
├── photo_checker.py         # 主程序
├── ai_detector.py            # AI检测(SCRFD+多维评分)
├── models/                   # ONNX模型(自动下载)
├── .claude/skills/           # 设计技能包
├── config.ini                # 参数持久化
├── 启动照片检测.bat           # 一键启动
└── requirements_photo.txt    # 依赖
```
