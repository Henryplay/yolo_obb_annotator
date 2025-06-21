import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import math
import json
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.config_file = Path.home() / ".yolo_obb_annotator_config.json"
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {"last_directory": ""}
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def get_last_directory(self):
        """获取上次使用的目录"""
        try:
            last_dir = self.config.get("last_directory", "")
            if last_dir and Path(last_dir).exists() and Path(last_dir).is_dir():
                return last_dir
        except Exception:
            pass
        # 如果上次目录无效，返回用户主目录
        return str(Path.home())
    
    def set_last_directory(self, directory):
        """设置上次使用的目录"""
        try:
            if directory and Path(directory).exists():
                self.config["last_directory"] = str(directory)
                self.save_config()
        except Exception:
            pass

class FileManager:
    def __init__(self):
        self.root_dir = None
        self.images_dir = None
        self.labels_dir = None
        self.image_files = []
        self.current_index = 0
        
    def load_directory(self, directory):
        self.root_dir = Path(directory)
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"
        
        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise ValueError("目录必须包含images和labels子目录")
        
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        self.image_files = sorted([f for f in self.images_dir.iterdir() 
                                 if f.suffix.lower() in extensions])
        
        if not self.image_files:
            raise ValueError("images目录中没有找到图像文件")
            
        self.current_index = 0
        return len(self.image_files)
    
    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return True
        return False
    
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False
    
    def get_current_info(self):
        if not self.image_files:
            return "0/0"
        return f"{self.current_index + 1}/{len(self.image_files)}"
    
    def get_label_path(self, image_path):
        return self.labels_dir / (image_path.stem + '.txt')
    
    def get_current_image_path(self):
        if 0 <= self.current_index < len(self.image_files):
            return self.image_files[self.current_index]
        return None

class AnnotationHandler:
    def __init__(self):
        self.annotations = []
        self.selected_index = -1
        
    def load_annotations(self, label_path, img_width, img_height):
        self.annotations = []
        self.selected_index = -1
        if not label_path.exists():
            return
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:
                    class_id = int(parts[0])
                    points = []
                    for i in range(1, 9, 2):
                        x = float(parts[i]) * img_width
                        y = float(parts[i+1]) * img_height
                        points.append((x, y))
                    
                    self.annotations.append({
                        'class_id': class_id,
                        'points': points
                    })
    
    def save_annotations(self, label_path, img_width, img_height):
        with open(label_path, 'w') as f:
            for ann in self.annotations:
                line = str(ann['class_id'])
                for x, y in ann['points']:
                    nx = x / img_width
                    ny = y / img_height
                    line += f" {nx:.6f} {ny:.6f}"
                f.write(line + '\n')
    
    def point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def select_annotation(self, x, y):
        for i, ann in enumerate(self.annotations):
            if self.point_in_polygon((x, y), ann['points']):
                self.selected_index = i
                return i
        self.selected_index = -1
        return -1
    
    def update_class_id(self, new_class_id):
        if 0 <= self.selected_index < len(self.annotations):
            self.annotations[self.selected_index]['class_id'] = new_class_id
    
    def add_annotation(self, points, class_id):
        self.annotations.append({
            'class_id': class_id,
            'points': points
        })
    
    def delete_selected_annotation(self):
        if 0 <= self.selected_index < len(self.annotations):
            del self.annotations[self.selected_index]
            self.selected_index = -1
            return True
        return False
    
    def vector_angle(self, v1, v2):
        """计算两个向量之间的角度（度）"""
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
            
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # 防止数值误差
        angle = math.acos(cos_angle)
        return math.degrees(angle)
    
    def is_parallelogram(self, points, tolerance=2.0):
        """检测四个点是否构成平行四边形"""
        if len(points) != 4:
            return True  # 不是四个点，不需要检查
            
        A, B, C, D = points
        
        # 计算四条边的向量
        AB = (B[0] - A[0], B[1] - A[1])
        BC = (C[0] - B[0], C[1] - B[1])
        CD = (D[0] - C[0], D[1] - C[1])
        DA = (A[0] - D[0], A[1] - D[1])
        
        # 检查对边是否平行（向量角度差小于容差）
        angle1 = self.vector_angle(AB, CD)  # AB与CD的角度
        angle2 = self.vector_angle(BC, DA)  # BC与DA的角度
        
        # 平行线的角度差应该接近0度或180度
        parallel1 = min(angle1, 180 - angle1) < tolerance
        parallel2 = min(angle2, 180 - angle2) < tolerance
        
        return parallel1 and parallel2
    
    def correct_to_parallelogram(self, points):
        """将四个点修正为平行四边形，保持前三个点不变"""
        if len(points) != 4:
            return points
            
        A, B, C = points[:3]  # 保持前三个点不变
        
        # 根据平行四边形性质计算第四个点：D = A + C - B
        D_corrected = (A[0] + C[0] - B[0], A[1] + C[1] - B[1])
        
        return [A, B, C, D_corrected]

class ImageCanvas:
    def __init__(self, master, annotation_handler, class_names):
        self.canvas = tk.Canvas(master, bg='gray', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.annotation_handler = annotation_handler
        self.class_names = class_names
        self.image = None
        self.photo = None
        self.scale = 1.0
        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 图像拖动相关
        self.pan_x = 0
        self.pan_y = 0
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.last_pan_x = 0
        self.last_pan_y = 0
        
        self.temp_points = []
        self.temp_point_ids = []
        self.is_marking = False
        
        self.setup_events()
        
    def setup_events(self):
        """设置所有事件绑定"""
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Double-Button-1>", self.reset_zoom)
        
        # 鼠标滚轮缩放（跨平台）
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)    # Linux/Mac
        self.canvas.bind("<Button-5>", self.on_mousewheel)    # Linux/Mac
        
        # 键盘事件 - 直接在Canvas上绑定
        self.canvas.bind("<KeyPress-Escape>", self.on_escape)
        self.canvas.bind("<Escape>", self.on_escape)
        
        # 使canvas能够接收键盘焦点
        self.canvas.bind("<FocusIn>", lambda e: None)
        self.canvas.configure(takefocus=1)
        
    def load_image(self, image_path):
        self.image = cv2.imread(str(image_path))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.zoom_factor = 1.0  # 重置缩放
        self.pan_x = 0  # 重置拖动偏移
        self.pan_y = 0
        self.display_image()
        
    def display_image(self):
        if self.image is None:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        img_height, img_width = self.image.shape[:2]
        
        # 计算基础缩放比例
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        base_scale = min(scale_x, scale_y) * 0.9
        
        # 应用用户缩放
        self.scale = base_scale * self.zoom_factor
        
        new_width = int(img_width * self.scale)
        new_height = int(img_height * self.scale)
        resized = cv2.resize(self.image, (new_width, new_height))
        
        # 计算基础偏移（居中）+ 拖动偏移
        base_offset_x = (canvas_width - new_width) // 2
        base_offset_y = (canvas_height - new_height) // 2
        
        self.offset_x = base_offset_x + self.pan_x
        self.offset_y = base_offset_y + self.pan_y
        
        img_pil = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(img_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, 
                                anchor=tk.NW, image=self.photo)
        
        self.draw_annotations()
        
    def draw_annotations(self):
        for i, ann in enumerate(self.annotation_handler.annotations):
            points = [(x * self.scale + self.offset_x, 
                      y * self.scale + self.offset_y) for x, y in ann['points']]
            
            color = 'red' if i == self.annotation_handler.selected_index else 'green'
            width = 3 if i == self.annotation_handler.selected_index else 2
            
            flat_points = [coord for point in points for coord in point]
            self.canvas.create_polygon(flat_points, outline=color, 
                                     fill='', width=width, tags="annotation")
            
            class_id = ann['class_id']
            class_text = f"{class_id}: {self.class_names.get(class_id, '未知')}"
            x, y = points[0]
            self.canvas.create_text(x, y-10, text=class_text, 
                                  fill=color, anchor=tk.W, tags="annotation")
    
    def on_click(self, event):
        # 获取主窗口引用并确保它有焦点
        root = self.canvas.winfo_toplevel()
        
        if self.is_marking:
            self.add_temp_point(event.x, event.y)
            # 标记模式下，Canvas需要焦点来接收Esc键
            self.canvas.focus_set()
        else:
            # 记录拖动起始点
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.last_pan_x = self.pan_x
            self.last_pan_y = self.pan_y
            self.is_dragging = False  # 初始时不确定是否拖动
            
            # 选择标注框（如果没有发生拖动）
            x = (event.x - self.offset_x) / self.scale
            y = (event.y - self.offset_y) / self.scale
            self.annotation_handler.select_annotation(x, y)
            self.display_image()
            
            # 非标记模式下，确保主窗口有焦点以接收快捷键
            root.focus_force()
    
    def on_drag(self, event):
        if self.is_marking:
            return
            
        # 计算拖动距离
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        # 如果移动距离超过阈值，开始拖动
        if abs(dx) > 3 or abs(dy) > 3:
            self.is_dragging = True
            
            # 更新拖动偏移
            self.pan_x = self.last_pan_x + dx
            self.pan_y = self.last_pan_y + dy
            
            # 应用边界限制
            self.limit_pan()
            
            # 重新显示图像
            self.display_image()
    
    def on_release(self, event):
        if not self.is_marking and not self.is_dragging:
            # 如果没有拖动，则处理选择操作
            x = (event.x - self.offset_x) / self.scale
            y = (event.y - self.offset_y) / self.scale
            self.annotation_handler.select_annotation(x, y)
            self.display_image()
        
        self.is_dragging = False
    
    def limit_pan(self):
        """限制拖动范围，防止图像拖动过度"""
        if self.image is None:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_height, img_width = self.image.shape[:2]
        
        scaled_width = int(img_width * self.scale)
        scaled_height = int(img_height * self.scale)
        
        # 计算最大拖动范围
        max_pan_x = max(0, (scaled_width - canvas_width) // 2)
        max_pan_y = max(0, (scaled_height - canvas_height) // 2)
        
        # 限制拖动范围
        self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_x))
        self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_y))
    
    def on_escape(self, event):
        """处理Esc键事件"""
        if self.is_marking:
            if hasattr(self, 'mode_callback'):
                self.mode_callback(False)  # 通过回调切换模式
            else:
                self.set_marking_mode(False)  # fallback
                if hasattr(self, 'zoom_callback'):
                    self.zoom_callback("已退出标记模式")
            
            # 确保主窗口获得焦点
            root = self.canvas.winfo_toplevel()
            root.focus_force()
            return "break"  # 阻止事件继续传播
            
    def on_motion(self, event):
        pass
    
    def on_mousewheel(self, event):
        """处理鼠标滚轮缩放"""
        if self.image is None:
            return
            
        # 确定缩放方向
        if event.num == 4 or event.delta > 0:  # 向上滚动，放大
            zoom_delta = 1.2
        elif event.num == 5 or event.delta < 0:  # 向下滚动，缩小
            zoom_delta = 1 / 1.2
        else:
            return
            
        # 计算新的缩放因子
        new_zoom = self.zoom_factor * zoom_delta
        new_zoom = max(0.1, min(5.0, new_zoom))  # 限制缩放范围
        
        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.display_image()
            # 通知主窗口更新状态
            if hasattr(self, 'zoom_callback'):
                self.zoom_callback(f"缩放: {self.zoom_factor:.1f}x")
    
    def reset_zoom(self, event=None):
        """重置缩放和拖动到初始状态"""
        if self.zoom_factor != 1.0 or self.pan_x != 0 or self.pan_y != 0:
            self.zoom_factor = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.display_image()
            if hasattr(self, 'zoom_callback'):
                self.zoom_callback("缩放和视图已重置")
    
    def set_zoom_callback(self, callback):
        """设置缩放状态回调函数"""
        self.zoom_callback = callback
    
    def set_mode_callback(self, callback):
        """设置模式切换回调函数"""
        self.mode_callback = callback
    
    def add_temp_point(self, x, y):
        img_x = (x - self.offset_x) / self.scale
        img_y = (y - self.offset_y) / self.scale
        
        self.temp_points.append((img_x, img_y))
        
        point_id = self.canvas.create_oval(x-3, y-3, x+3, y+3, 
                                         fill='blue', outline='blue')
        self.temp_point_ids.append(point_id)
        
        if len(self.temp_points) == 4:
            self.create_new_annotation()
    
    def create_new_annotation(self):
        if len(self.temp_points) == 4:
            # 检查是否需要几何修正
            is_parallel = self.annotation_handler.is_parallelogram(self.temp_points)
            
            if not is_parallel:
                # 计算修正后的点
                corrected_points = self.annotation_handler.correct_to_parallelogram(self.temp_points)
                
                # 显示修正确认对话框
                self.show_correction_dialog(self.temp_points.copy(), corrected_points)
            else:
                # 已经是平行四边形，直接显示类别选择
                self.show_class_selection_dialog(self.temp_points.copy())
    
    def show_correction_dialog(self, original_points, corrected_points):
        """显示几何修正确认对话框"""
        dialog = tk.Toplevel()
        dialog.title("几何修正")
        dialog.geometry("400x300")
        dialog.transient(dialog.master)
        dialog.grab_set()
        
        # 说明文本
        tk.Label(dialog, text="检测到不规则四边形", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        tk.Label(dialog, text="是否自动修正为平行四边形？", 
                font=('Arial', 10)).pack(pady=5)
        
        # 创建预览画布
        preview_frame = tk.Frame(dialog)
        preview_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        preview_canvas = tk.Canvas(preview_frame, width=300, height=150, bg='white')
        preview_canvas.pack()
        
        # 绘制预览
        self.draw_correction_preview(preview_canvas, original_points, corrected_points)
        
        # 按钮区域
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def accept_correction():
            dialog.destroy()
            self.clear_temp_points()
            self.show_class_selection_dialog(corrected_points)
        
        def keep_original():
            dialog.destroy()
            self.show_class_selection_dialog(original_points)
        
        def cancel():
            dialog.destroy()
            self.clear_temp_points()
        
        tk.Button(button_frame, text="接受修正", command=accept_correction,
                 bg='lightgreen', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="保持原样", command=keep_original,
                 bg='lightyellow', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="重新标注", command=cancel,
                 bg='lightcoral', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
    
    def draw_correction_preview(self, canvas, original_points, corrected_points):
        """在预览画布中绘制修正前后的对比"""
        canvas_width = 300
        canvas_height = 150
        
        # 计算所有点的边界框，用于缩放
        all_points = original_points + corrected_points
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        
        # 计算缩放比例
        if max_x - min_x == 0 or max_y - min_y == 0:
            return
            
        scale_x = (canvas_width - 40) / (max_x - min_x)
        scale_y = (canvas_height - 40) / (max_y - min_y)
        scale = min(scale_x, scale_y)
        
        # 计算偏移量（居中）
        offset_x = (canvas_width - (max_x - min_x) * scale) / 2 - min_x * scale
        offset_y = (canvas_height - (max_y - min_y) * scale) / 2 - min_y * scale
        
        # 绘制原始四边形（红色虚线）
        orig_scaled = [(p[0] * scale + offset_x, p[1] * scale + offset_y) for p in original_points]
        flat_orig = [coord for point in orig_scaled for coord in point]
        canvas.create_polygon(flat_orig, outline='red', fill='', width=2, dash=(5, 5))
        canvas.create_text(20, 20, text="原始", fill='red', anchor=tk.W, font=('Arial', 10, 'bold'))
        
        # 绘制修正后四边形（绿色实线）
        corr_scaled = [(p[0] * scale + offset_x, p[1] * scale + offset_y) for p in corrected_points]
        flat_corr = [coord for point in corr_scaled for coord in point]
        canvas.create_polygon(flat_corr, outline='green', fill='', width=2)
        canvas.create_text(20, 40, text="修正后", fill='green', anchor=tk.W, font=('Arial', 10, 'bold'))
        
        # 标记被修正的点（第四个点）
        if len(orig_scaled) == 4 and len(corr_scaled) == 4:
            orig_point = orig_scaled[3]  # 第四个点
            corr_point = corr_scaled[3]
            
            # 绘制移动箭头
            canvas.create_line(orig_point[0], orig_point[1], 
                             corr_point[0], corr_point[1], 
                             fill='blue', width=2, arrow=tk.LAST)
            
            # 标记点
            canvas.create_oval(orig_point[0]-3, orig_point[1]-3, 
                             orig_point[0]+3, orig_point[1]+3, 
                             fill='red', outline='red')
            canvas.create_oval(corr_point[0]-3, corr_point[1]-3, 
                             corr_point[0]+3, corr_point[1]+3, 
                             fill='green', outline='green')
    
    def show_class_selection_dialog(self, points):
        """显示类别选择对话框"""
        dialog = tk.Toplevel()
        dialog.title("选择类别")
        dialog.geometry("300x120")
        dialog.transient(dialog.master)
        dialog.grab_set()
        
        tk.Label(dialog, text="请选择类别:").pack(pady=5)
        
        class_var = tk.StringVar()
        class_values = [f"{k}: {v}" for k, v in self.class_names.items()]
        class_combo = ttk.Combobox(dialog, textvariable=class_var, 
                                 values=class_values, state="readonly")
        class_combo.pack(pady=5)
        class_combo.set(class_values[0] if class_values else "0: 默认")
        
        def confirm():
            try:
                class_id = int(class_var.get().split(':')[0])
                self.annotation_handler.add_annotation(points, class_id)
                self.clear_temp_points()  # 清除临时点
                self.display_image()
                dialog.destroy()
            except (ValueError, IndexError):
                messagebox.showerror("错误", "请选择有效的类别")
        
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="确定", command=confirm).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="取消", 
                 command=lambda: [self.clear_temp_points(), dialog.destroy()]).pack(side=tk.LEFT, padx=5)
            
    def clear_temp_points(self):
        self.temp_points = []
        for point_id in self.temp_point_ids:
            self.canvas.delete(point_id)
        self.temp_point_ids = []
    
    def set_marking_mode(self, enabled):
        self.is_marking = enabled
        if not enabled:
            self.clear_temp_points()
        
        # 更新光标样式
        cursor = "dotbox" if enabled else "crosshair"
        self.canvas.configure(cursor=cursor)
        
        # 根据模式设置合适的焦点
        if enabled:
            # 标记模式下Canvas需要接收Esc键
            self.canvas.focus_set()
        else:
            # 退出标记模式时，确保主窗口有焦点以接收快捷键
            root = self.canvas.winfo_toplevel()
            root.focus_force()

class GUIManager:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO OBB 数据重新标注工具")
        self.root.geometry("1200x800")
        
        # 初始化配置管理器
        self.config_manager = ConfigManager()
        
        # 预设类别字典
        self.class_names = {
            0: "背景",
            1: "人员",
            2: "车辆", 
            3: "建筑",
            4: "植物",
            5: "动物",
            6: "设备",
            7: "标志",
            8: "其他",
            9: "待定"
        }
        
        self.file_manager = FileManager()
        self.annotation_handler = AnnotationHandler()
        
        self.setup_ui()
        self.setup_shortcuts()
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # 初始化完成后确保按钮状态正确，使用多次延迟确保UI完全就绪
        self.root.after(50, self.update_mode_button)
        self.root.after(100, self.update_mode_button)
        self.root.after(200, lambda: [self.update_mode_button(), self.root.focus_force()])
        
    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = tk.Frame(main_frame, width=350, bg='lightgray')
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        self.image_canvas = ImageCanvas(left_frame, self.annotation_handler, self.class_names)
        self.image_canvas.set_zoom_callback(self.update_status)
        self.image_canvas.set_mode_callback(self.on_canvas_mode_change)
        
        # 工具栏
        toolbar = tk.Frame(self.root, bg='darkgray', height=40)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Button(toolbar, text="加载目录", command=self.load_directory).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(toolbar, text="上一张", command=self.prev_image).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="下一张", command=self.next_image).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="保存", command=self.save_current).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.status_label = tk.Label(toolbar, text="请加载目录", bg='darkgray', fg='white')
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        self.image_info_label = tk.Label(toolbar, text="0/0", bg='darkgray', fg='white')
        self.image_info_label.pack(side=tk.RIGHT, padx=20)
        
        # 右侧面板
        tk.Label(right_frame, text="图像列表", bg='lightgray', font=('Arial', 12, 'bold')).pack(pady=10)
        
        self.image_listbox = tk.Listbox(right_frame, height=12)
        self.image_listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        # 标注控制区域
        control_frame = tk.Frame(right_frame, bg='lightgray')
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(control_frame, text="标注控制", bg='lightgray', font=('Arial', 12, 'bold')).pack()
        
        # 模式切换按钮
        self.mode_button = tk.Button(control_frame, text="进入标记模式", 
                                    command=self.toggle_mode_button,
                                    bg='lightgreen', fg='black', 
                                    font=('Arial', 10, 'bold'))
        self.mode_button.pack(fill=tk.X, pady=5)
        
        # 类别选择
        class_frame = tk.Frame(control_frame, bg='lightgray')
        class_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(class_frame, text="修改类别:", bg='lightgray').pack(anchor=tk.W)
        
        self.class_var = tk.StringVar()
        class_values = [f"{k}: {v}" for k, v in self.class_names.items()]
        self.class_combo = ttk.Combobox(class_frame, textvariable=self.class_var, 
                                       values=class_values, state="readonly")
        self.class_combo.pack(fill=tk.X, pady=2)
        self.class_combo.bind('<<ComboboxSelected>>', self.on_class_change)
        
        # 删除按钮
        tk.Button(control_frame, text="删除选中框", command=self.delete_selected,
                 bg='red', fg='white').pack(fill=tk.X, pady=5)
        
        # 快捷键提示
        tk.Label(right_frame, text="快捷键", bg='lightgray', font=('Arial', 12, 'bold')).pack(pady=(10,5))
        shortcuts = [
            "P: 标记点模式",
            "Esc: 退出标记模式", 
            "按钮: 切换标记模式",
            "D: 删除选中框",
            "Ctrl+S: 保存",
            "←→: 上一张/下一张",
            "滚轮: 缩放图像",
            "拖动: 移动图像",
            "双击: 重置缩放和视图"
        ]
        for shortcut in shortcuts:
            tk.Label(right_frame, text=shortcut, bg='lightgray', font=('Arial', 9)).pack(anchor=tk.W, padx=10)
        
        # 几何修正说明
        tk.Label(right_frame, text="几何修正", bg='lightgray', font=('Arial', 12, 'bold')).pack(pady=(10,5))
        geo_info = [
            "• 自动检测不规则四边形",
            "• 智能修正为平行四边形",
            "• 保持前三点位置不变",
            "• 提供修正预览确认"
        ]
        for info in geo_info:
            tk.Label(right_frame, text=info, bg='lightgray', font=('Arial', 9)).pack(anchor=tk.W, padx=10)
        
        # 功能说明
        tk.Label(right_frame, text="功能特性", bg='lightgray', font=('Arial', 12, 'bold')).pack(pady=(10,5))
        feature_info = [
            "• 目录记忆：自动记住上次目录",
            "• 智能缩放：鼠标滚轮控制",
            "• 拖动浏览：放大后可拖动查看",
            "• 快捷操作：丰富的键盘快捷键"
        ]
        for info in feature_info:
            tk.Label(right_frame, text=info, bg='lightgray', font=('Arial', 9)).pack(anchor=tk.W, padx=10)
        
    def setup_shortcuts(self):
        # 确保主窗口可以接收键盘事件，使用focus_force确保强制获取焦点
        self.root.bind('<KeyPress-p>', lambda e: [self.toggle_marking_mode(True), self.root.focus_force()])
        self.root.bind('<KeyRelease-p>', lambda e: [self.toggle_marking_mode(False), self.root.focus_force()])
        self.root.bind('<Escape>', lambda e: [self.escape_key_handler(), self.root.focus_force()])
        self.root.bind('<Control-s>', lambda e: [self.save_current(), self.root.focus_force()])
        
        # d键删除功能 - 确保在任何情况下都能工作
        self.root.bind('<d>', self.handle_delete_key)
        self.root.bind('<D>', self.handle_delete_key)
        
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        
        # 确保窗口能接收键盘事件
        self.root.focus_force()
        
        # 当点击主窗口任何地方时，确保焦点在主窗口
        self.root.bind('<Button-1>', lambda e: self.root.focus_force(), add='+')
        
    def handle_delete_key(self, event):
        """专门处理d键删除功能"""
        # 强制确保主窗口有焦点
        self.root.focus_force()
        
        # 执行删除操作
        if self.annotation_handler.delete_selected_annotation():
            self.image_canvas.display_image()
            self.update_status("已删除选中的标注框")
        else:
            self.update_status("没有选中的标注框")
        
        # 再次确保焦点在主窗口
        self.root.after(10, lambda: self.root.focus_force())
        return "break"  # 阻止事件继续传播
        
    def delete_selected_key(self):
        """通过键盘快捷键删除选中框（保留兼容性）"""
        return self.handle_delete_key(None)
    
    def escape_key_handler(self):
        """处理Esc键的主窗口级别事件"""
        if self.image_canvas.is_marking:
            self.set_marking_mode(False)
            self.update_status("已通过Esc键退出标记模式")
        
    def on_canvas_mode_change(self, enabled):
        """处理来自Canvas的模式切换请求"""
        # 直接设置Canvas状态（避免循环调用）
        self.image_canvas.is_marking = enabled
        
        # 更新UI状态
        self.update_mode_button()
        
        # 更新状态栏
        if enabled:
            self.status_label.config(text="标记模式: 点击4个点创建新框 (按Esc或点击按钮退出)")
        else:
            self.status_label.config(text="选择模式 (可拖动查看图像)")
            self.update_status("已通过Canvas退出标记模式")
        
        # 确保主窗口有焦点
        self.root.focus_force()
            
    def toggle_mode_button(self):
        """通过按钮切换标记模式"""
        current_mode = self.image_canvas.is_marking
        new_mode = not current_mode
        
        # 先立即更新按钮显示（预设状态）
        if new_mode:
            self.mode_button.config(text="退出标记模式", bg='lightcoral', fg='white')
        else:
            self.mode_button.config(text="进入标记模式", bg='lightgreen', fg='black')
        
        # 强制刷新UI显示
        self.mode_button.update_idletasks()
        self.root.update_idletasks()
        
        # 然后设置实际的模式状态
        self.set_marking_mode(new_mode)
        
        # 再次确保按钮状态正确
        self.root.after(10, self.update_mode_button)
        
        # 记录操作
        if new_mode:
            self.update_status("通过按钮进入标记模式")
        else:
            self.update_status("通过按钮退出标记模式")
    
    def set_marking_mode(self, enabled):
        """统一的模式设置方法"""
        # 先设置Canvas模式
        self.image_canvas.set_marking_mode(enabled)
        
        # 强制确保主窗口获得焦点（特别是退出标记模式时）
        self.root.focus_force()
        
        # 立即更新按钮状态
        self.update_mode_button()
        
        # 强制刷新UI
        self.root.update_idletasks()
        
        # 更新状态栏
        if enabled:
            self.status_label.config(text="标记模式: 点击4个点创建新框 (按Esc或点击按钮退出)")
        else:
            self.status_label.config(text="选择模式 (可拖动查看图像)")
            # 退出标记模式时特别确保焦点在主窗口
            self.root.after(50, lambda: self.root.focus_force())
    
    def update_mode_button(self):
        """更新模式按钮的显示状态"""
        if hasattr(self, 'mode_button') and self.mode_button.winfo_exists():  # 确保按钮存在且有效
            try:
                if self.image_canvas.is_marking:
                    self.mode_button.config(text="退出标记模式", bg='lightcoral', fg='white')
                else:
                    self.mode_button.config(text="进入标记模式", bg='lightgreen', fg='black')
                
                # 强制刷新按钮显示 - 多重刷新确保显示正确
                self.mode_button.update_idletasks()
                self.mode_button.update()
                # 刷新整个右侧面板
                if hasattr(self, 'mode_button') and self.mode_button.master:
                    self.mode_button.master.update_idletasks()
                    
            except tk.TclError:
                # 如果按钮已被销毁，忽略错误
                pass
        
    def toggle_marking_mode(self, enabled):
        """兼容原有的快捷键调用方式"""
        self.set_marking_mode(enabled)
            
    def load_directory(self):
        # 获取上次使用的目录作为初始目录
        initial_dir = self.config_manager.get_last_directory()
        
        directory = filedialog.askdirectory(
            title="选择数据目录",
            initialdir=initial_dir
        )
        
        if not directory:
            return
            
        try:
            num_images = self.file_manager.load_directory(directory)
            
            # 成功加载后保存目录到配置
            self.config_manager.set_last_directory(directory)
            
            self.update_status(f"已加载 {num_images} 张图像 (目录已记录)")
            
            self.image_listbox.delete(0, tk.END)
            for img_file in self.file_manager.image_files:
                self.image_listbox.insert(tk.END, img_file.name)
                
            if num_images > 0:
                self.image_listbox.selection_set(0)
                self.load_current_image()
            
            # 确保模式按钮状态正确
            self.update_mode_button()
            # 确保主窗口有焦点
            self.root.focus_set()
                
        except Exception as e:
            messagebox.showerror("错误", str(e))
            
    def load_current_image(self):
        image_path = self.file_manager.get_current_image_path()
        if not image_path:
            return
            
        # 先退出标记模式（如果处于标记模式）
        if hasattr(self, 'image_canvas') and self.image_canvas.is_marking:
            self.set_marking_mode(False)
            
        self.image_canvas.load_image(image_path)
        
        label_path = self.file_manager.get_label_path(image_path)
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        self.annotation_handler.load_annotations(label_path, w, h)
        
        self.image_canvas.display_image()
        self.update_image_info()
        self.update_mode_button()  # 确保按钮状态正确
        
        # 同步列表框选择
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(self.file_manager.current_index)
        self.image_listbox.see(self.file_manager.current_index)
        
        # 确保主窗口有焦点
        self.root.focus_set()
        
    def next_image(self):
        if self.file_manager.next_image():
            self.load_current_image()
            self.update_status("切换到下一张图像")
        else:
            self.update_status("已经是最后一张图像")
        # 确保主窗口有焦点
        self.root.focus_set()
            
    def prev_image(self):
        if self.file_manager.prev_image():
            self.load_current_image()
            self.update_status("切换到上一张图像")
        else:
            self.update_status("已经是第一张图像")
        # 确保主窗口有焦点
        self.root.focus_set()
            
    def delete_selected(self):
        if self.annotation_handler.delete_selected_annotation():
            self.image_canvas.display_image()
            self.update_status("已删除选中的标注框")
        else:
            self.update_status("没有选中的标注框")
        # 确保主窗口有焦点
        self.root.focus_set()
            
    def on_image_select(self, event):
        selection = self.image_listbox.curselection()
        if selection:
            self.file_manager.current_index = selection[0]
            self.load_current_image()
            
    def on_class_change(self, event):
        if self.annotation_handler.selected_index >= 0:
            try:
                new_class = int(self.class_var.get().split(':')[0])
                self.annotation_handler.update_class_id(new_class)
                self.image_canvas.display_image()
                class_name = self.class_names.get(new_class, '未知')
                self.update_status(f"已更新类别为 {new_class}: {class_name}")
            except (ValueError, IndexError):
                messagebox.showerror("错误", "请选择有效的类别")
                
    def save_current(self):
        image_path = self.file_manager.get_current_image_path()
        if not image_path:
            return
            
        label_path = self.file_manager.get_label_path(image_path)
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        self.annotation_handler.save_annotations(label_path, w, h)
        self.update_status("已保存标注")
        
    def update_status(self, message):
        self.status_label.config(text=message)
        
    def update_image_info(self):
        self.image_info_label.config(text=self.file_manager.get_current_info())
    
    def on_closing(self):
        """应用程序关闭时的清理工作"""
        try:
            # 保存配置
            self.config_manager.save_config()
        except Exception:
            pass
        finally:
            # 关闭窗口
            self.root.destroy()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = GUIManager()
    app.run()