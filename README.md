
# ComfyUI-Transition 

本插件为 ComfyUI 提供多种图片/序列过渡特效节点，支持丰富的参数自定义，适合视频、动画、幻灯片等场景。

---

## 1. LinearTransition（线性过渡）

**功能**：实现两张图片之间的线性过渡（硬边界），如从左到右、上到下等方向切换。

**参数**：
- `image1`：起始图片
- `image2`：结束图片
- `frames`：过渡帧数（2~240，默认24）
- `direction`：过渡方向（left_to_right, right_to_left, top_to_bottom, bottom_to_top）
- `fps`：输出视频帧率（1.0~60.0，默认24.0）

**返回**：
- `frames`：过渡序列图片（[frames, H, W, C]）
- `fps_int`：帧率（整数）

**典型用法**：两张图片之间的简单线性切换动画。

---

## 2. GradientTransition（渐变过渡）

**功能**：实现两张图片之间的平滑渐变过渡（柔和边界），可自定义过渡宽度和方向。

**参数**：
- `image1`、`image2`：同上
- `frames`：过渡帧数
- `transition_width`：过渡区域宽度（0.01~1.0，默认0.2，值越大过渡越宽）
- `direction`：过渡方向（同上）
- `fps`：帧率

**返回**：
- `frames`：过渡序列图片
- `fps_int`：帧率

**典型用法**：适合需要柔和渐变效果的场景，比如淡入淡出。

---

## 3. DualLineTransition（双线扫描过渡）

**功能**：双线扫描过渡，支持水平、垂直、斜线扫描和线间距扩展，适合“百叶窗”或“扫描线”切换。

**参数**（部分重要参数）：
- `image1`、`image2`：同上
- `frames`：过渡帧数
- `line_direction`：线条方向（horizontal, vertical, diagonal）
- `diagonal_angle`：斜线角度（-89~89度）
- `line_spacing`：两线之间的间距（0.01~0.5，比例）
- `mode`：过渡模式（sweep：扫过，expand：扩展）
- `invert`：是否反转遮罩（True/False）
- `fps`：帧率

**返回**：
- `frames`：过渡序列图片
- `fps_int`：帧率

---

## 4. SequenceTransition（序列帧过渡）

**功能**：实现两个序列帧之间的过渡，支持多种方向和过渡模式。

**参数**（部分重要参数）：
- `sequence1`、`sequence2`：两个序列帧（如视频帧）
- `transition_type`：过渡类型（horizontal, vertical, diagonal）
- `diagonal_angle`：斜线角度
- `transition_width`：过渡区域宽度
- `sharpness`：过渡边缘锐利度
- `transition_mode`：过渡模式（smooth, sharp, linear, step）
- `direction`：过渡方向（forward, backward）
- `fps`：帧率

**返回**：
- `frames`：过渡序列图片
- `fps_int`：帧率

**典型用法**：两个视频片段之间的多样化过渡。

---

## 5. CircularTransition（圆形/椭圆形过渡）

**功能**：两张图片之间的圆形/椭圆形过渡，支持扩展、收缩、光圈等模式。

**参数**（部分重要参数）：
- `image1`、`image2`：同上
- `frames`：过渡帧数
- `transition_mode`：expand（扩展）、contract（收缩）、iris（光圈）
- `center_x`、`center_y`：圆心位置（0~1，比例）
- `edge_softness`：边缘柔和度
- `aspect_ratio`：椭圆纵横比
- `rotation_angle`：椭圆旋转角度
- `ease_function`：缓动函数（linear, ease_in, ease_out, 等）
- `invert`：是否反转
- `fps`：帧率

**返回**：
- `frames`：过渡序列图片
- `fps_int`：帧率

**典型用法**：适合制作“圆形遮罩”或“光圈”切换效果。

---

## 6. CircularSequenceTransition（序列帧圆形过渡）

**功能**：两个序列帧之间的圆形/椭圆形过渡。

**参数**：同 CircularTransition，但输入为序列帧。

**返回**：
- `frames`：过渡序列图片
- `fps_int`：帧率

**典型用法**：两个视频片段之间的圆形/椭圆形过渡。

---

## 通用说明

- 所有节点都支持自动调整图片尺寸（第二张图片/序列会自动缩放为第一张的尺寸）。
- 帧数越多，过渡越平滑，但生成时间也会增加。
- 建议使用相同尺寸、相似内容的图片/序列以获得最佳效果。

---

## 示例工作流

```
LoadImage1 -> image1
LoadImage2 -> image2               -> GradientTransition -> frames -> CreateVideo -> SaveVideo
                frames=24            transition_width=0.2   fps_int
                direction=left_to_right
                fps=24.0
```

