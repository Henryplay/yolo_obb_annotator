# YOLO OBB Annotator

这是一个基于 PySide6 的 YOLO OBB (Oriented Bounding Box) 标注工具。

## 主要功能

- **加载目录**: 支持加载包含 `images` 和 `labels` 子目录的数据集。
- **图像导航**: 轻松切换上一张/下一张图像。
- **OBB 标注**: 在图像上绘制旋转矩形（四边形）标注。
- **类别选择**: 为每个标注选择对应的类别。
- **保存标注**: 将标注数据保存为 YOLO OBB 格式的 `.txt` 文件。
- **删除标注**: 删除已有的标注。
- **自动校正**: 对于非平行四边形的标注，提供自动校正为平行四边形的功能。

## 使用方法

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **运行工具**:
    ```bash
    python yolo_obb_annotator.py
    ```
3.  **加载目录**: 点击 "Load Directory" 按钮选择你的数据集根目录（该目录应包含 `images` 和 `labels` 子目录）。
4.  **开始标注**: 点击 "Start Marking" 按钮进入标注模式，然后在图像上依次点击四个点来定义一个 OBB。
5.  **选择类别**: 完成四点标注后，会弹出对话框让你选择该标注的类别。
6.  **保存**: 完成标注后，点击 "Save Annotations" 按钮保存当前图像的标注。
7.  **导航**: 使用 "Previous Image" 和 "Next Image" 按钮切换图像。
8.  **删除**: 选择一个标注后，点击 "Delete Selected" 按钮可以删除它。