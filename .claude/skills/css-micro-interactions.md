---
name: css-micro-interactions
description: CSS微交互技能 — 100+纯CSS悬停/点击/加载效果，自动匹配当前设计风格
---

# CSS微交互技能

## 触发规则
在编写交互元素(按钮/卡片/链接/表单)时, 根据当前设计风格自动匹配对应的交互效果。

## 按钮交互
```css
/* Neon Pulse — 赛博朋克/电竞 */
.btn-neon { transition: all 0.3s; }
.btn-neon:hover { box-shadow: 0 0 20px var(--accent-glow), 0 0 40px var(--accent-glow); transform: translateY(-2px); }
.btn-neon:active { transform: scale(0.96); }

/* Subtle Lift — 极简/SaaS */
.btn-lift { transition: all 0.2s cubic-bezier(0.16,1,0.3,1); }
.btn-lift:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.12); }
.btn-lift:active { transform: scale(0.98); }

/* Ripple — Material/通用 */
.btn-ripple { position: relative; overflow: hidden; }
.btn-ripple::after { content: ''; position: absolute; inset: 0; background: radial-gradient(circle, rgba(255,255,255,0.3) 10%, transparent 10%); transform: scale(10); opacity: 0; transition: all 0.5s; }
.btn-ripple:active::after { transform: scale(0); opacity: 1; transition: 0s; }

/* Glitch — 赛博朋克 */
.btn-glitch:hover { animation: glitch 0.3s ease; }
@keyframes glitch { 0%,100%{transform:translate(0)} 20%{transform:translate(-3px,3px)} 40%{transform:translate(3px,-3px)} 60%{transform:translate(-3px,-3px)} 80%{transform:translate(3px,3px)} }
```

## 卡片交互
```css
/* Glow Border — 电竞暗黑 */
.card-glow { border: 1px solid transparent; transition: all 0.3s; }
.card-glow:hover { border-color: var(--accent); box-shadow: 0 0 30px var(--accent-glow); }

/* Scale Reveal — 通用 */
.card-scale { transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1); }
.card-scale:hover { transform: scale(1.03); }

/* Tilt — 创意/实验 */
.card-tilt { transition: transform 0.2s; transform-style: preserve-3d; perspective: 1000px; }
.card-tilt:hover { transform: rotateY(3deg) rotateX(-3deg); }
```

## 加载/骨架屏
```css
/* Skeleton Pulse */
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
.skeleton { background: linear-gradient(90deg, var(--surface) 0%, rgba(255,255,255,0.08) 40%, var(--surface) 80%); background-size: 200% 100%; animation: shimmer 1.8s infinite; }

/* Spinner */
@keyframes spin { to{transform:rotate(360deg)} }
.spinner { width: 24px; height: 24px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.6s linear infinite; }
```

## 输入框交互
```css
/* Glow Focus */
.input-glow { transition: border-color 0.2s, box-shadow 0.2s; }
.input-glow:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(var(--accent-rgb),0.15); outline: none; }

/* Underline Expand */
.input-underline { border: none; border-bottom: 2px solid var(--border); transition: border-color 0.2s; }
.input-underline:focus { border-bottom-color: var(--accent); outline: none; }
```

## 页面过渡
```css
/* Fade Up Stagger */
.stagger-item { opacity: 0; transform: translateY(20px); animation: fadeUp 0.4s cubic-bezier(0.16,1,0.3,1) forwards; }
.stagger-item:nth-child(1){animation-delay:0.05s} .stagger-item:nth-child(2){animation-delay:0.1s}
.stagger-item:nth-child(3){animation-delay:0.15s} .stagger-item:nth-child(4){animation-delay:0.2s}
@keyframes fadeUp { to{opacity:1;transform:translateY(0)} }
```

## 自动匹配规则
- 暗色电竞风 → Neon Pulse + Glow Border + Glitch
- 极简SaaS → Subtle Lift + Scale Reveal + Glow Focus
- 玻璃态 → Scale Reveal + Shimmer + 半透边框过渡
- 粗野主义 → 无动效, 即时响应, 高对比focus
