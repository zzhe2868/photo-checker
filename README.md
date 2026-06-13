# 照片废片检测工具 v3.0

本地离线AI照片质量检测，自动筛选：**模糊** / **闭眼** / **曝光异常** / **重复照片**

## 新特性 (v3.0)
- ⚙️ 参数面板：阈值滑块可调，自动保存 config.ini
- 🎨 深色主题：ttkbootstrap modern dark theme
- 🔁 pHash 重复检测：相似度 >95% 自动标记
- 📄 CSV 报告导出
- ↩️ 撤销移动
- 🖼️ 点击预览图片
- 🖥️ 高DPI适配

## 快速开始
`ash
py -m pip install opencv-python numpy ttkbootstrap Pillow
py photo_checker.py
`
或双击 启动照片检测.bat

## 检测原理
| 类型 | 方法 | 可配置 |
|------|------|--------|
| 模糊 | 拉普拉斯方差 | 阈值滑块 |
| 闭眼 | Haar级联 | — |
| 曝光 | 直方图分析 | 过曝/欠曝阈值 |
| 重复 | pHash + 汉明距离 | 相似度阈值 |
