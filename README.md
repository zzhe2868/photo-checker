# 图片智能分析系统 v2.0

专业AI图片质量分析工具，达芬奇风格界面。自动检测模糊/闭眼/曝光/重复，AI多维评分+场景分类。

## 版本
| 版本 | 更新 |
|------|------|
| v2.0 | 达芬奇风格UI · AI模式选择(本地/在线) · 暂停续扫 · SCRFD人脸 · 多维评分 · 场景分类 · 汉堡菜单 · 参数侧滑面板 · AI模型管理器 · 独立分析报告窗口 |
| v1.2 | ONNX AI引擎 · 美学评分 · 模糊分类 |
| v1.1 | 参数面板 · pHash重复 · CSV导出 · 撤销 |
| v1.0 | OpenCV三检测(模糊/闭眼/曝光) + tkinter |

## 功能
- AI双模式：本地离线(SCRFD) / 在线API
- SCRFD 2.5GF人脸检测+质量分析
- 多维评分：技术/构图/人像/美学 0-100
- 7类场景自动识别
- pHash重复照片检测
- 暂停/继续断点续扫
- CSV完整导出
- 达芬奇风格汉堡菜单
- 参数侧滑面板(点击外自动收起)
- 文件夹路径不保存，隐私优先

## 运行
```bash
py -m pip install opencv-python numpy ttkbootstrap Pillow onnxruntime
py photo_checker.py
```