import torch
import numpy as np
import comfy.utils
import torch.nn.functional as F

class LinearTransition:
    """
    实现两张图片之间的线性过渡效果，从左到右逐渐过渡
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 起始图片
                "image2": ("IMAGE",),  # 结束图片
                "frames": ("INT", {"default": 24, "min": 2, "max": 240, "step": 1}),  # 过渡帧数
                "direction": (["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"], {"default": "left_to_right"}),  # 过渡方向
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),  # 视频帧率
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps_int")
    FUNCTION = "generate_transition"
    CATEGORY = "animation/transition"
    
    def generate_transition(self, image1, image2, frames, direction, fps):
        # 确保两张图片有相同的尺寸
        if image1.shape[1:] != image2.shape[1:]:
            # 将第二张图调整为第一张图的尺寸
            image2 = F.interpolate(image2.permute(0, 3, 1, 2), 
                                   size=(image1.shape[1], image1.shape[2]), 
                                   mode='bilinear').permute(0, 2, 3, 1)
        
        # 取两张图片中的第一帧（如果是批量图片）
        img1 = image1[0:1]
        img2 = image2[0:1]
        
        # 创建过渡帧
        output_frames = []
        
        height, width = image1.shape[1], image1.shape[2]
        
        for i in range(frames):
            # 创建渐变遮罩
            if direction == "left_to_right":
                mask = torch.linspace(0, 1, width).view(1, 1, width).repeat(1, height, 1)
            elif direction == "right_to_left":
                mask = torch.linspace(1, 0, width).view(1, 1, width).repeat(1, height, 1)
            elif direction == "top_to_bottom":
                mask = torch.linspace(0, 1, height).view(1, height, 1).repeat(1, 1, width)
            elif direction == "bottom_to_top":
                mask = torch.linspace(1, 0, height).view(1, height, 1).repeat(1, 1, width)
            
            # 调整遮罩阈值，根据当前帧的位置
            threshold = i / (frames - 1)
            binary_mask = (mask < threshold).float()
            
            # 混合两张图片
            blended = img1 * (1 - binary_mask).unsqueeze(-1) + img2 * binary_mask.unsqueeze(-1)
            output_frames.append(blended)
        
        # 将所有帧堆叠为一个批量图片
        output = torch.cat(output_frames, dim=0)
        
        return (output, int(fps))


class GradientTransition:
    """
    实现两张图片之间的平滑渐变过渡效果，使用渐变遮罩
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 起始图片
                "image2": ("IMAGE",),  # 结束图片
                "frames": ("INT", {"default": 24, "min": 2, "max": 240, "step": 1}),  # 过渡帧数
                "transition_width": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),  # 过渡区域宽度
                "direction": (["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"], {"default": "left_to_right"}),  # 过渡方向
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),  # 视频帧率
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps_int")
    FUNCTION = "generate_transition"
    CATEGORY = "animation/transition"
    
    def generate_transition(self, image1, image2, frames, transition_width, direction, fps):
        # 确保两张图片有相同的尺寸
        if image1.shape[1:] != image2.shape[1:]:
            # 将第二张图调整为第一张图的尺寸
            image2 = F.interpolate(image2.permute(0, 3, 1, 2), 
                                   size=(image1.shape[1], image1.shape[2]), 
                                   mode='bilinear').permute(0, 2, 3, 1)
        
        # 取两张图片中的第一帧（如果是批量图片）
        img1 = image1[0:1]
        img2 = image2[0:1]
        
        # 创建过渡帧
        output_frames = []
        
        height, width = image1.shape[1], image1.shape[2]
        
        for i in range(frames):
            # 计算当前帧的过渡位置
            position = i / (frames - 1)
            
            # 创建平滑渐变遮罩
            if direction == "left_to_right":
                x = torch.linspace(0, 1, width).view(1, 1, width).repeat(1, height, 1)
                # 创建平滑的sigmoid过渡
                center = position
                mask = 1 / (1 + torch.exp(-(x - center) / (transition_width / 2)))
            elif direction == "right_to_left":
                x = torch.linspace(1, 0, width).view(1, 1, width).repeat(1, height, 1)
                center = 1 - position
                mask = 1 / (1 + torch.exp(-(x - center) / (transition_width / 2)))
            elif direction == "top_to_bottom":
                y = torch.linspace(0, 1, height).view(1, height, 1).repeat(1, 1, width)
                center = position
                mask = 1 / (1 + torch.exp(-(y - center) / (transition_width / 2)))
            elif direction == "bottom_to_top":
                y = torch.linspace(1, 0, height).view(1, height, 1).repeat(1, 1, width)
                center = 1 - position
                mask = 1 / (1 + torch.exp(-(y - center) / (transition_width / 2)))
            
            # 混合两张图片
            blended = img1 * (1 - mask).unsqueeze(-1) + img2 * mask.unsqueeze(-1)
            output_frames.append(blended)
        
        # 将所有帧堆叠为一个批量图片
        output = torch.cat(output_frames, dim=0)
        
        return (output, int(fps))


class DualLineTransition:
    """
    实现双线扫描过渡效果，两条线作为遮罩边界，线之间为过渡区域
    支持水平、垂直和斜线扫描，以及线间距扩展模式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 起始图片
                "image2": ("IMAGE",),  # 结束图片
                "frames": ("INT", {"default": 24, "min": 2, "max": 240, "step": 1}),  # 过渡帧数
                "line_direction": (["horizontal", "vertical", "diagonal"], {"default": "horizontal"}),  # 线条方向
                "diagonal_angle": ("FLOAT", {"default": 45.0, "min": -89.0, "max": 89.0, "step": 1.0}),  # 斜线角度（度）
                "line_spacing": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),  # 两线之间的间距（显示区域宽度）
                "mode": (["sweep", "expand"], {"default": "sweep"}),  # 过渡模式：扫过或扩展
                "expand_frames_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.1}),  # 扩展模式中扩展阶段占总帧数的比例
                "split_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.1}),  # 扩展阶段中分开过程与扩展过程的帧数比例
                "sweep_ease": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),  # 扫过阶段的缓动效果
                "expand_ease": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "ease_out"}),  # 扩展阶段的缓动效果
                "invert": (["False", "True"], {"default": "False"}),  # 是否反转效果（线之间为白色，线之外为黑色）
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),  # 视频帧率
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps_int")
    FUNCTION = "generate_transition"
    CATEGORY = "Transition"
    
    def generate_transition(self, image1, image2, frames, line_direction, diagonal_angle, line_spacing, mode, expand_frames_ratio, split_ratio, sweep_ease, expand_ease, invert, fps):
        # 确保两张图片有相同的尺寸
        if image1.shape[1:] != image2.shape[1:]:
            # 将第二张图调整为第一张图的尺寸
            image2 = F.interpolate(image2.permute(0, 3, 1, 2), 
                                   size=(image1.shape[1], image1.shape[2]), 
                                   mode='bilinear').permute(0, 2, 3, 1)
        
        # 取两张图片中的第一帧（如果是批量图片）
        img1 = image1[0:1]
        img2 = image2[0:1]
        
        # 创建过渡帧
        output_frames = []
        
        height, width = image1.shape[1], image1.shape[2]
        
        # 计算线间距（像素）
        if line_direction in ["horizontal", "diagonal"]:
            spacing_px = int(width * line_spacing)
        else:
            spacing_px = int(height * line_spacing)
        
        # 确保最小值
        spacing_px = max(1, spacing_px)
        
        # 计算斜线的角度（转换为弧度）
        angle_rad = torch.tensor(diagonal_angle * 3.14159 / 180.0)
        
        # 缓动函数
        def apply_easing(t, ease_type):
            if ease_type == "linear":
                return t
            elif ease_type == "ease_in":
                return t * t
            elif ease_type == "ease_out":
                return 1 - (1 - t) * (1 - t)
            elif ease_type == "ease_in_out":
                return 0.5 * (torch.sin((t - 0.5) * 3.14159) + 1)
            return t
        
        # 扫描模式
        if mode == "sweep":
            # 扫描的总距离
            if line_direction in ["horizontal", "diagonal"]:
                total_distance = width + spacing_px
            else:
                total_distance = height + spacing_px
            
            for i in range(frames):
                # 计算当前位置
                raw_progress = i / (frames - 1)
                # 应用缓动
                progress = apply_easing(raw_progress, sweep_ease)
                position = -spacing_px + progress * total_distance
                
                # 创建坐标网格
                if line_direction == "horizontal":
                    x = torch.arange(width).float().view(1, 1, width).repeat(1, height, 1)
                elif line_direction == "vertical":
                    x = torch.arange(height).float().view(1, height, 1).repeat(1, 1, width)
                elif line_direction == "diagonal":
                    # 创建坐标网格
                    y_coords = torch.arange(height).float().view(1, height, 1).repeat(1, 1, width)
                    x_coords = torch.arange(width).float().view(1, 1, width).repeat(1, height, 1)
                    
                    # 根据角度计算投影坐标
                    x = x_coords * torch.cos(angle_rad) + y_coords * torch.sin(angle_rad)
                
                # 创建遮罩 - 两条线之间的区域
                line1_pos = position
                line2_pos = position + spacing_px
                
                # 创建过渡区域遮罩（两线之间的区域）
                transition_area = (x >= line1_pos) & (x <= line2_pos)
                
                # 设置遮罩
                mask = transition_area.float()
                
                if invert == "True":
                    mask = 1 - mask
                
                # 混合两张图片
                blended = img1 * (1 - mask).unsqueeze(-1) + img2 * mask.unsqueeze(-1)
                output_frames.append(blended)
        
        # 扩展模式
        else:  # mode == "expand"
            # 分为两个阶段：扫过阶段和扩展阶段
            sweep_frames = int(frames * (1 - expand_frames_ratio))
            expand_frames = frames - sweep_frames
            
            # 确保至少有1帧
            sweep_frames = max(1, sweep_frames)
            expand_frames = max(1, expand_frames)
            
            # 进一步划分扩展阶段
            split_frames = int(expand_frames * split_ratio)
            expand_full_frames = expand_frames - split_frames
            
            # 确保至少有1帧
            split_frames = max(1, split_frames)
            expand_full_frames = max(1, expand_full_frames)
            
            # 计算画面中心
            center_x = width // 2
            center_y = height // 2
            
            # 1. 扫过阶段 - 两条线均匀扫过画面到中心
            if line_direction == "horizontal":
                total_sweep_distance = width + spacing_px
            elif line_direction == "vertical":
                total_sweep_distance = height + spacing_px
            elif line_direction == "diagonal":
                diagonal_length = int(width * torch.cos(angle_rad).abs() + height * torch.sin(angle_rad).abs())
                total_sweep_distance = diagonal_length + spacing_px
            
            for i in range(sweep_frames):
                # 计算当前位置，从画面外扫到中心
                raw_progress = i / max(1, sweep_frames - 1)
                # 应用缓动
                progress = apply_easing(raw_progress, sweep_ease)
                position = -spacing_px + progress * total_sweep_distance
                
                # 创建坐标网格
                if line_direction == "horizontal":
                    x = torch.arange(width).float().view(1, 1, width).repeat(1, height, 1)
                elif line_direction == "vertical":
                    x = torch.arange(height).float().view(1, height, 1).repeat(1, 1, width)
                elif line_direction == "diagonal":
                    # 创建坐标网格
                    y_coords = torch.arange(height).float().view(1, height, 1).repeat(1, 1, width)
                    x_coords = torch.arange(width).float().view(1, 1, width).repeat(1, height, 1)
                    
                    # 根据角度计算投影坐标
                    x = x_coords * torch.cos(angle_rad) + y_coords * torch.sin(angle_rad)
                
                # 创建过渡区域遮罩（两线之间的区域）
                line1_pos = position
                line2_pos = position + spacing_px
                transition_area = (x >= line1_pos) & (x <= line2_pos)
                
                # 设置遮罩
                mask = transition_area.float()
                
                if invert == "True":
                    mask = 1 - mask
                
                # 混合两张图片
                blended = img1 * (1 - mask).unsqueeze(-1) + img2 * mask.unsqueeze(-1)
                output_frames.append(blended)
            
            # 2. 扩展阶段 - 从中心开始，两条线从合一到分开扩展
            # 创建坐标网格
            y_coords = torch.arange(height).float().view(1, height, 1).repeat(1, 1, width)
            x_coords = torch.arange(width).float().view(1, 1, width).repeat(1, height, 1)
            
            # 计算到中心的距离
            if line_direction == "horizontal":
                distance_from_center = torch.abs(x_coords - center_x)
                max_distance = width // 2 + spacing_px
            elif line_direction == "vertical":
                distance_from_center = torch.abs(y_coords - center_y)
                max_distance = height // 2 + spacing_px
            elif line_direction == "diagonal":
                # 计算投影距离
                projected_coords = (x_coords - center_x) * torch.cos(angle_rad) + (y_coords - center_y) * torch.sin(angle_rad)
                distance_from_center = torch.abs(projected_coords)
                diagonal_length = int(width * torch.cos(angle_rad).abs() + height * torch.sin(angle_rad).abs())
                max_distance = diagonal_length // 2 + spacing_px
            
            # 2.1 第一部分：两条线从合一开始分开
            for i in range(split_frames):
                # 计算当前扩展比例
                raw_progress = i / max(1, split_frames - 1)
                # 应用缓动
                progress = apply_easing(raw_progress, expand_ease)
                
                # 间距从0增加到设定值
                current_spacing = progress * spacing_px
                
                if line_direction == "horizontal":
                    # 水平方向上，从中心向两侧扩展
                    left_line = center_x - current_spacing/2
                    right_line = center_x + current_spacing/2
                    transition_area = (x_coords >= left_line) & (x_coords <= right_line)
                elif line_direction == "vertical":
                    # 垂直方向上，从中心向两侧扩展
                    top_line = center_y - current_spacing/2
                    bottom_line = center_y + current_spacing/2
                    transition_area = (y_coords >= top_line) & (y_coords <= bottom_line)
                elif line_direction == "diagonal":
                    # 对角线方向，从中心向两侧扩展
                    projected_coords = (x_coords - center_x) * torch.cos(angle_rad) + (y_coords - center_y) * torch.sin(angle_rad)
                    left_line = -current_spacing/2
                    right_line = current_spacing/2
                    transition_area = (projected_coords >= left_line) & (projected_coords <= right_line)
                
                # 设置遮罩
                mask = transition_area.float()
                
                if invert == "True":
                    mask = 1 - mask
                
                # 混合两张图片
                blended = img1 * (1 - mask).unsqueeze(-1) + img2 * mask.unsqueeze(-1)
                output_frames.append(blended)
            
            # 2.2 第二部分：两条线继续向外扩展，使用更平滑的加速度曲线
            for i in range(expand_full_frames):
                # 计算当前扩展比例，使用平方根函数减缓扩展速度
                raw_progress = i / max(1, expand_full_frames - 1)
                
                # 应用自定义缓动，使扩展更加平滑
                if expand_ease == "linear":
                    # 线性扩展太快，使用平方根函数减缓
                    progress = torch.sqrt(torch.tensor(raw_progress))
                else:
                    # 使用选定的缓动函数，但调整为更平滑的曲线
                    progress = apply_easing(raw_progress, expand_ease)
                    # 进一步调整，使末尾扩展更慢
                    if raw_progress > 0.7:
                        # 在70%之后，进一步减缓速度
                        late_progress = (raw_progress - 0.7) / 0.3  # 重新映射到0-1
                        slow_factor = 1 - 0.5 * late_progress  # 末尾减速因子
                        progress = progress * slow_factor + raw_progress * (1 - slow_factor)
                
                # 计算当前扩展距离，使用非线性映射使扩展更平滑
                expand_ratio = progress * progress  # 使用二次函数减缓扩展速度
                expand_distance = expand_ratio * (max_distance - spacing_px/2)
                
                # 创建遮罩区域
                inner_boundary = expand_distance
                outer_boundary = expand_distance + spacing_px
                
                # 两个区域：中心区域 + 环形区域
                center_area = (distance_from_center <= inner_boundary)
                ring_area = (distance_from_center > inner_boundary) & (distance_from_center <= outer_boundary)
                
                # 合并区域
                transition_area = center_area | ring_area
                
                # 设置遮罩
                mask = transition_area.float()
                
                if invert == "True":
                    mask = 1 - mask
                
                # 混合两张图片
                blended = img1 * (1 - mask).unsqueeze(-1) + img2 * mask.unsqueeze(-1)
                output_frames.append(blended)
        
        # 将所有帧堆叠为一个批量图片
        output = torch.cat(output_frames, dim=0)
        
        return (output, int(fps))


class SequenceTransition:
    """
    实现两个序列帧之间的过渡效果，支持水平、垂直和斜线扫描过渡
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence1": ("IMAGE",),  # 第一个序列帧
                "sequence2": ("IMAGE",),  # 第二个序列帧
                "transition_type": (["horizontal", "vertical", "diagonal"], {"default": "horizontal"}),  # 过渡类型
                "diagonal_angle": ("FLOAT", {"default": 45.0, "min": -89.0, "max": 89.0, "step": 1.0}),  # 斜线角度（度）
                "transition_width": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),  # 过渡区域宽度
                "sharpness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 50.0, "step": 0.1}),  # 过渡边缘锐利度
                "transition_mode": (["smooth", "sharp", "linear", "step"], {"default": "smooth"}),  # 过渡模式
                "direction": (["forward", "backward"], {"default": "forward"}),  # 过渡方向
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),  # 视频帧率
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps_int")
    FUNCTION = "generate_transition"
    CATEGORY = "Transition"
    
    def generate_transition(self, sequence1, sequence2, transition_type, diagonal_angle, transition_width, sharpness, transition_mode, direction, fps):
        # 确保两个序列有相同的尺寸
        if sequence1.shape[1:] != sequence2.shape[1:]:
            # 将第二个序列调整为第一个序列的尺寸
            sequence2 = F.interpolate(sequence2.permute(0, 3, 1, 2), 
                                     size=(sequence1.shape[1], sequence1.shape[2]), 
                                     mode='bilinear').permute(0, 2, 3, 1)
        
        # 获取序列帧数
        frames1 = sequence1.shape[0]
        frames2 = sequence2.shape[0]
        
        # 使用较少的帧数作为输出帧数
        frames = min(frames1, frames2)
        
        # 如果帧数不同，截取相同数量的帧
        if frames1 > frames:
            sequence1 = sequence1[:frames]
        if frames2 > frames:
            sequence2 = sequence2[:frames]
        
        # 获取图像尺寸
        height, width = sequence1.shape[1], sequence1.shape[2]
        
        # 创建输出帧
        output_frames = []
        
        # 计算斜线的角度（转换为弧度）
        angle_rad = torch.tensor(diagonal_angle * 3.14159 / 180.0)
        
        # 创建坐标网格
        y_coords = torch.arange(height).float().view(1, height, 1).repeat(1, 1, width)
        x_coords = torch.arange(width).float().view(1, 1, width).repeat(1, height, 1)
        
        # 根据过渡类型创建基础坐标
        if transition_type == "horizontal":
            # 水平方向过渡
            base_coords = x_coords / width
        elif transition_type == "vertical":
            # 垂直方向过渡
            base_coords = y_coords / height
        elif transition_type == "diagonal":
            # 对角线方向过渡
            projected_coords = x_coords * torch.cos(angle_rad) + y_coords * torch.sin(angle_rad)
            # 归一化到0-1范围
            max_proj = torch.max(projected_coords)
            min_proj = torch.min(projected_coords)
            base_coords = (projected_coords - min_proj) / (max_proj - min_proj)
        
        # 反转方向（如果需要）
        if direction == "backward":
            base_coords = 1 - base_coords
        
        # 为每一帧创建过渡遮罩
        for i in range(frames):
            # 计算当前帧的过渡位置
            position = i / max(1, frames - 1)
            
            # 根据过渡模式创建不同的过渡效果
            if transition_mode == "smooth":
                # 平滑的sigmoid过渡
                center = position
                # 应用锐利度参数，值越大越锐利
                scaled_width = transition_width / sharpness
                mask = 1 / (1 + torch.exp(-(base_coords - center) / (scaled_width / 2)))
            
            elif transition_mode == "sharp":
                # 更锐利的过渡
                # 使用高次幂来创建更陡峭的过渡
                diff = base_coords - position
                # 应用锐利度参数
                power = 2.0 * sharpness
                # 创建陡峭的S形曲线
                mask = 1 / (1 + torch.exp(-diff * power / transition_width))
            
            elif transition_mode == "linear":
                # 线性过渡
                half_width = transition_width / 2
                # 计算线性过渡区域
                start = position - half_width
                end = position + half_width
                # 创建线性过渡遮罩
                mask = torch.clamp((base_coords - start) / (end - start), 0, 1)
            
            elif transition_mode == "step":
                # 阶梯式过渡（几乎无过渡区域）
                # 使用非常小的过渡宽度
                step_width = transition_width / (10 * sharpness)
                mask = (base_coords > position).float()
                
                # 添加一点点平滑过渡（可选）
                if step_width > 0:
                    smooth_region = torch.abs(base_coords - position) < step_width
                    smooth_mask = 0.5 + (base_coords - position) / (2 * step_width)
                    mask = torch.where(smooth_region, smooth_mask, mask)
            
            # 混合两个序列的对应帧
            blended = sequence1[i:i+1] * (1 - mask).unsqueeze(-1) + sequence2[i:i+1] * mask.unsqueeze(-1)
            output_frames.append(blended)
        
        # 将所有帧堆叠为一个批量图片
        output = torch.cat(output_frames, dim=0)
        
        return (output, int(fps))


class CircularTransition:
    """
    实现两张图片之间的圆形过渡效果，支持从中心向外扩展或从外向内收缩
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 起始图片
                "image2": ("IMAGE",),  # 结束图片
                "frames": ("INT", {"default": 24, "min": 2, "max": 240, "step": 1}),  # 过渡帧数
                "transition_mode": (["expand", "contract", "iris"], {"default": "expand"}),  # 过渡模式：扩展、收缩或光圈
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # 圆心X坐标（相对位置）
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # 圆心Y坐标（相对位置）
                "edge_softness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),  # 边缘柔和度
                "aspect_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),  # 椭圆纵横比
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),  # 椭圆旋转角度
                "ease_function": (["linear", "ease_in", "ease_out", "ease_in_out", "elastic", "bounce"], {"default": "ease_out"}),  # 缓动函数
                "invert": (["False", "True"], {"default": "False"}),  # 是否反转效果
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),  # 视频帧率
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps_int")
    FUNCTION = "generate_transition"
    CATEGORY = "Transition"
    
    def generate_transition(self, image1, image2, frames, transition_mode, center_x, center_y, edge_softness, aspect_ratio, rotation_angle, ease_function, invert, fps):
        # 确保两张图片有相同的尺寸
        if image1.shape[1:] != image2.shape[1:]:
            # 将第二张图调整为第一张图的尺寸
            image2 = F.interpolate(image2.permute(0, 3, 1, 2), 
                                   size=(image1.shape[1], image1.shape[2]), 
                                   mode='bilinear').permute(0, 2, 3, 1)
        
        # 取两张图片中的第一帧（如果是批量图片）
        img1 = image1[0:1]
        img2 = image2[0:1]
        
        # 创建过渡帧
        output_frames = []
        
        height, width = image1.shape[1], image1.shape[2]
        
        # 计算圆心坐标（像素）
        center_x_px = int(width * center_x)
        center_y_px = int(height * center_y)
        
        # 计算最大半径（从圆心到图像最远角落的距离）
        corners = [
            (0, 0),  # 左上角
            (width, 0),  # 右上角
            (0, height),  # 左下角
            (width, height)  # 右下角
        ]
        
        max_radius = 0
        for corner_x, corner_y in corners:
            # 计算圆心到角落的距离
            distance = ((corner_x - center_x_px) ** 2 + (corner_y - center_y_px) ** 2) ** 0.5
            max_radius = max(max_radius, distance)
        
        # 旋转角度转换为弧度
        rotation_rad = rotation_angle * 3.14159 / 180.0
        
        # 创建坐标网格
        y_coords = torch.arange(height).float().view(1, height, 1).repeat(1, 1, width)
        x_coords = torch.arange(width).float().view(1, 1, width).repeat(1, height, 1)
        
        # 计算相对于圆心的坐标
        rel_x = x_coords - center_x_px
        rel_y = y_coords - center_y_px
        
        # 应用旋转变换
        if rotation_angle != 0:
            rot_x = rel_x * torch.cos(torch.tensor(rotation_rad)) - rel_y * torch.sin(torch.tensor(rotation_rad))
            rot_y = rel_x * torch.sin(torch.tensor(rotation_rad)) + rel_y * torch.cos(torch.tensor(rotation_rad))
            rel_x, rel_y = rot_x, rot_y
        
        # 应用纵横比变换（椭圆效果）
        if aspect_ratio != 1.0:
            rel_x = rel_x * torch.sqrt(torch.tensor(aspect_ratio))
            rel_y = rel_y / torch.sqrt(torch.tensor(aspect_ratio))
        
        # 计算到圆心的距离
        distance_from_center = torch.sqrt(rel_x ** 2 + rel_y ** 2)
        
        # 归一化距离（0-1范围）
        normalized_distance = distance_from_center / max_radius
        
        # 缓动函数
        def apply_easing(t, ease_type):
            if ease_type == "linear":
                return t
            elif ease_type == "ease_in":
                return t * t
            elif ease_type == "ease_out":
                return 1 - (1 - t) * (1 - t)
            elif ease_type == "ease_in_out":
                return 0.5 * (torch.sin((t - 0.5) * 3.14159) + 1)
            elif ease_type == "elastic":
                # 简化的弹性效果
                p = 0.3
                return torch.pow(2, -10 * t) * torch.sin((t - p/4) * (2 * 3.14159) / p) + 1
            elif ease_type == "bounce":
                # 简化的弹跳效果
                if t < 0.5:
                    return 2 * t * t
                elif t < 0.75:
                    return 2 * (2 * t - 1) * (2 * t - 1) + 0.5
                else:
                    return 2 * (2 * t - 1.5) * (2 * t - 1.5) + 0.75
            return t
        
        # 光圈效果的参数
        iris_width = 0.2  # 光圈宽度（相对于最大半径）
        
        # 为每一帧创建过渡遮罩
        for i in range(frames):
            # 计算当前帧的过渡进度
            progress = i / max(1, frames - 1)
            
            # 应用缓动函数
            eased_progress = apply_easing(progress, ease_function)
            
            # 根据过渡模式创建不同的圆形过渡效果
            if transition_mode == "expand":
                # 从中心向外扩展的圆形
                # 当前半径（归一化）
                current_radius = eased_progress
                
                # 创建带有柔和边缘的圆形遮罩
                mask = 1 / (1 + torch.exp(-(normalized_distance - current_radius) / edge_softness))
                
            elif transition_mode == "contract":
                # 从外向内收缩的圆形
                # 当前半径（归一化）
                current_radius = 1 - eased_progress
                
                # 创建带有柔和边缘的圆形遮罩
                mask = 1 / (1 + torch.exp((normalized_distance - current_radius) / edge_softness))
                
            elif transition_mode == "iris":
                # 光圈效果（环形过渡）
                # 当前环的中心位置（从外到内）
                ring_center = 1 - eased_progress
                
                # 创建环形遮罩
                inner_edge = ring_center - iris_width/2
                outer_edge = ring_center + iris_width/2
                
                # 计算内外边缘的平滑过渡
                inner_mask = 1 / (1 + torch.exp((normalized_distance - inner_edge) / edge_softness))
                outer_mask = 1 / (1 + torch.exp(-(normalized_distance - outer_edge) / edge_softness))
                
                # 组合内外遮罩形成环形
                mask = inner_mask * outer_mask
            
            # 反转遮罩（如果需要）
            if invert == "True":
                mask = 1 - mask
            
            # 混合两张图片
            blended = img1 * (1 - mask).unsqueeze(-1) + img2 * mask.unsqueeze(-1)
            output_frames.append(blended)
        
        # 将所有帧堆叠为一个批量图片
        output = torch.cat(output_frames, dim=0)
        
        return (output, int(fps))


class CircularSequenceTransition:
    """
    实现两个序列帧之间的圆形过渡效果
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence1": ("IMAGE",),  # 第一个序列帧
                "sequence2": ("IMAGE",),  # 第二个序列帧
                "transition_mode": (["expand", "contract", "iris"], {"default": "expand"}),  # 过渡模式：扩展、收缩或光圈
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # 圆心X坐标（相对位置）
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # 圆心Y坐标（相对位置）
                "edge_softness": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),  # 边缘柔和度
                "aspect_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),  # 椭圆纵横比
                "rotation_angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),  # 椭圆旋转角度
                "ease_function": (["linear", "ease_in", "ease_out", "ease_in_out", "elastic", "bounce"], {"default": "ease_out"}),  # 缓动函数
                "invert": (["False", "True"], {"default": "False"}),  # 是否反转效果
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1}),  # 视频帧率
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps_int")
    FUNCTION = "generate_transition"
    CATEGORY = "Transition"
    
    def generate_transition(self, sequence1, sequence2, transition_mode, center_x, center_y, edge_softness, aspect_ratio, rotation_angle, ease_function, invert, fps):
        # 确保两个序列有相同的尺寸
        if sequence1.shape[1:] != sequence2.shape[1:]:
            # 将第二个序列调整为第一个序列的尺寸
            sequence2 = F.interpolate(sequence2.permute(0, 3, 1, 2), 
                                     size=(sequence1.shape[1], sequence1.shape[2]), 
                                     mode='bilinear').permute(0, 2, 3, 1)
        
        # 获取序列帧数
        frames1 = sequence1.shape[0]
        frames2 = sequence2.shape[0]
        
        # 使用较少的帧数作为输出帧数
        frames = min(frames1, frames2)
        
        # 如果帧数不同，截取相同数量的帧
        if frames1 > frames:
            sequence1 = sequence1[:frames]
        if frames2 > frames:
            sequence2 = sequence2[:frames]
        
        # 获取图像尺寸
        height, width = sequence1.shape[1], sequence1.shape[2]
        
        # 计算圆心坐标（像素）
        center_x_px = int(width * center_x)
        center_y_px = int(height * center_y)
        
        # 计算最大半径（从圆心到图像最远角落的距离）
        corners = [
            (0, 0),  # 左上角
            (width, 0),  # 右上角
            (0, height),  # 左下角
            (width, height)  # 右下角
        ]
        
        max_radius = 0
        for corner_x, corner_y in corners:
            # 计算圆心到角落的距离
            distance = ((corner_x - center_x_px) ** 2 + (corner_y - center_y_px) ** 2) ** 0.5
            max_radius = max(max_radius, distance)
        
        # 旋转角度转换为弧度
        rotation_rad = rotation_angle * 3.14159 / 180.0
        
        # 创建坐标网格
        y_coords = torch.arange(height).float().view(1, height, 1).repeat(1, 1, width)
        x_coords = torch.arange(width).float().view(1, 1, width).repeat(1, height, 1)
        
        # 计算相对于圆心的坐标
        rel_x = x_coords - center_x_px
        rel_y = y_coords - center_y_px
        
        # 应用旋转变换
        if rotation_angle != 0:
            rot_x = rel_x * torch.cos(torch.tensor(rotation_rad)) - rel_y * torch.sin(torch.tensor(rotation_rad))
            rot_y = rel_x * torch.sin(torch.tensor(rotation_rad)) + rel_y * torch.cos(torch.tensor(rotation_rad))
            rel_x, rel_y = rot_x, rot_y
        
        # 应用纵横比变换（椭圆效果）
        if aspect_ratio != 1.0:
            rel_x = rel_x * torch.sqrt(torch.tensor(aspect_ratio))
            rel_y = rel_y / torch.sqrt(torch.tensor(aspect_ratio))
        
        # 计算到圆心的距离
        distance_from_center = torch.sqrt(rel_x ** 2 + rel_y ** 2)
        
        # 归一化距离（0-1范围）
        normalized_distance = distance_from_center / max_radius
        
        # 缓动函数
        def apply_easing(t, ease_type):
            if ease_type == "linear":
                return t
            elif ease_type == "ease_in":
                return t * t
            elif ease_type == "ease_out":
                return 1 - (1 - t) * (1 - t)
            elif ease_type == "ease_in_out":
                return 0.5 * (torch.sin((t - 0.5) * 3.14159) + 1)
            elif ease_type == "elastic":
                # 简化的弹性效果
                p = 0.3
                return torch.pow(2, -10 * t) * torch.sin((t - p/4) * (2 * 3.14159) / p) + 1
            elif ease_type == "bounce":
                # 简化的弹跳效果
                if t < 0.5:
                    return 2 * t * t
                elif t < 0.75:
                    return 2 * (2 * t - 1) * (2 * t - 1) + 0.5
                else:
                    return 2 * (2 * t - 1.5) * (2 * t - 1.5) + 0.75
            return t
        
        # 光圈效果的参数
        iris_width = 0.2  # 光圈宽度（相对于最大半径）
        
        # 创建输出帧
        output_frames = []
        
        # 为每一帧创建过渡遮罩
        for i in range(frames):
            # 计算当前帧的过渡进度
            progress = i / max(1, frames - 1)
            
            # 应用缓动函数
            eased_progress = apply_easing(progress, ease_function)
            
            # 根据过渡模式创建不同的圆形过渡效果
            if transition_mode == "expand":
                # 从中心向外扩展的圆形
                # 当前半径（归一化）
                current_radius = eased_progress
                
                # 创建带有柔和边缘的圆形遮罩
                mask = 1 / (1 + torch.exp(-(normalized_distance - current_radius) / edge_softness))
                
            elif transition_mode == "contract":
                # 从外向内收缩的圆形
                # 当前半径（归一化）
                current_radius = 1 - eased_progress
                
                # 创建带有柔和边缘的圆形遮罩
                mask = 1 / (1 + torch.exp((normalized_distance - current_radius) / edge_softness))
                
            elif transition_mode == "iris":
                # 光圈效果（环形过渡）
                # 当前环的中心位置（从外到内）
                ring_center = 1 - eased_progress
                
                # 创建环形遮罩
                inner_edge = ring_center - iris_width/2
                outer_edge = ring_center + iris_width/2
                
                # 计算内外边缘的平滑过渡
                inner_mask = 1 / (1 + torch.exp((normalized_distance - inner_edge) / edge_softness))
                outer_mask = 1 / (1 + torch.exp(-(normalized_distance - outer_edge) / edge_softness))
                
                # 组合内外遮罩形成环形
                mask = inner_mask * outer_mask
            
            # 反转遮罩（如果需要）
            if invert == "True":
                mask = 1 - mask
            
            # 混合两个序列的对应帧
            blended = sequence1[i:i+1] * (1 - mask).unsqueeze(-1) + sequence2[i:i+1] * mask.unsqueeze(-1)
            output_frames.append(blended)
        
        # 将所有帧堆叠为一个批量图片
        output = torch.cat(output_frames, dim=0)
        
        return (output, int(fps))


# 节点列表，用于注册到ComfyUI
NODE_CLASS_MAPPINGS = {
    "LinearTransition": LinearTransition,
    "GradientTransition": GradientTransition,
    "DualLineTransition": DualLineTransition,
    "SequenceTransition": SequenceTransition,
    "CircularTransition": CircularTransition,
    "CircularSequenceTransition": CircularSequenceTransition
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "LinearTransition": "Linear Transition",
    "GradientTransition": "Gradient Transition",
    "DualLineTransition": "Dual Line Transition",
    "SequenceTransition": "Sequence Transition",
    "CircularTransition": "Circular Transition",
    "CircularSequenceTransition": "Circular Sequence Transition"
} 