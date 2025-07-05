import sys
import os
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsTextItem,
    QMessageBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
)
from PySide6.QtGui import (
    QPixmap,
    QImage,
    QPolygonF,
    QPen,
    QBrush,
    QColor,
    QPainter,
    QFont,
)
from PySide6.QtCore import Qt, QPointF, Signal


class ConfigManager:
    def __init__(self):
        self.config_file = Path.home() / ".yolo_obb_annotator_config.json"
        self.config = self.load_config()

    def load_config(self):
        try:
            if self.config_file.exists():
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"last_directory": ""}

    def save_config(self):
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_last_directory(self):
        try:
            last_dir = self.config.get("last_directory", "")
            if last_dir and Path(last_dir).exists() and Path(last_dir).is_dir():
                return last_dir
        except Exception:
            pass
        return str(Path.home())

    def set_last_directory(self, directory):
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
            raise ValueError(
                "Directory must contain 'images' and 'labels' subdirectories"
            )

        extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
        self.image_files = sorted(
            [f for f in self.images_dir.iterdir() if f.suffix.lower() in extensions]
        )

        if not self.image_files:
            raise ValueError("No image files found in 'images' directory")

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
        return self.labels_dir / (image_path.stem + ".txt")

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

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:
                    class_id = int(parts[0])
                    points = []
                    for i in range(1, 9, 2):
                        x = float(parts[i]) * img_width
                        y = float(parts[i + 1]) * img_height
                        points.append(QPointF(x, y))
                    self.annotations.append({"class_id": class_id, "points": points})

    def save_annotations(self, label_path, img_width, img_height):
        with open(label_path, "w") as f:
            for ann in self.annotations:
                line = str(ann["class_id"])
                for p in ann["points"]:
                    nx = p.x() / img_width
                    ny = p.y() / img_height
                    line += f" {nx:.6f} {ny:.6f}"
                f.write(line + "\n")

    def select_annotation(self, point):
        for i, ann in enumerate(self.annotations):
            poly = QPolygonF(ann["points"])
            if poly.containsPoint(point, Qt.OddEvenFill):
                self.selected_index = i
                return i
        self.selected_index = -1
        return -1

    def update_class_id(self, new_class_id):
        if 0 <= self.selected_index < len(self.annotations):
            self.annotations[self.selected_index]["class_id"] = new_class_id

    def add_annotation(self, points, class_id):
        self.annotations.append({"class_id": class_id, "points": points})

    def delete_selected_annotation(self):
        if 0 <= self.selected_index < len(self.annotations):
            del self.annotations[self.selected_index]
            self.selected_index = -1
            return True
        return False

    def vector_angle(self, p1, p2):
        v1 = (p1.x(), p1.y())
        v2 = (p2.x(), p2.y())
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 == 0 or mag2 == 0:
            return 0
        cos_angle = max(-1, min(1, dot_product / (mag1 * mag2)))
        angle = math.acos(cos_angle)
        return math.degrees(angle)

    def is_parallelogram(self, points, tolerance=5.0):
        if len(points) != 4:
            return True
        A, B, C, D = points
        AB = B - A
        BC = C - B
        CD = D - C
        DA = A - D
        angle1 = self.vector_angle(AB, -CD)
        angle2 = self.vector_angle(BC, -DA)
        parallel1 = angle1 < tolerance
        parallel2 = angle2 < tolerance
        return parallel1 and parallel2

    def correct_to_parallelogram(self, points):
        if len(points) != 4:
            return points
        A, B, C = points[:3]
        D_corrected = QPointF(A.x() + C.x() - B.x(), A.y() + C.y() - B.y())
        return [A, B, C, D_corrected]


class ImageCanvas(QGraphicsView):
    annotation_selected = Signal(int)
    mode_changed = Signal(bool)
    new_annotation_request = Signal(list)

    def __init__(self, annotation_handler, class_names, class_colors):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.annotation_handler = annotation_handler
        self.class_names = class_names
        self.class_colors = class_colors
        self.pixmap_item = None
        self.is_marking = False
        self.temp_point_items = []

    def load_image(self, image_path):
        self.scene.clear()
        self.clear_temp_points()
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.draw_annotations()

    def draw_annotations(self):
        for item in self.scene.items():
            if isinstance(item, (QGraphicsPolygonItem, QGraphicsTextItem)):
                self.scene.removeItem(item)

        for i, ann in enumerate(self.annotation_handler.annotations):
            poly = QPolygonF(ann["points"])
            class_id = ann["class_id"]

            base_color = self.class_colors[class_id % len(self.class_colors)]
            pen_color = (
                QColor("red")
                if i == self.annotation_handler.selected_index
                else base_color
            )
            pen_width = 3 if i == self.annotation_handler.selected_index else 2

            pen = QPen(pen_color, pen_width)
            self.scene.addPolygon(poly, pen)

            class_text = f"{class_id}: {self.class_names.get(class_id, 'Unknown')}"
            text_item = self.scene.addText(class_text, QFont("Arial", 12))
            text_item.setDefaultTextColor(pen_color)
            text_item.setPos(ann["points"][0] + QPointF(0, -15))

    def set_marking_mode(self, enabled):
        self.is_marking = enabled
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.CrossCursor)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
            self.clear_temp_points()
        self.mode_changed.emit(enabled)

    def clear_temp_points(self):
        for item in self.temp_point_items:
            self.scene.removeItem(item)
        self.temp_point_items = []

    def mousePressEvent(self, event):
        if self.is_marking:
            point = self.mapToScene(event.pos())
            if self.pixmap_item and self.pixmap_item.contains(point):
                dot = QGraphicsEllipseItem(point.x() - 3, point.y() - 3, 6, 6)
                dot.setBrush(QBrush(QColor("cyan")))
                dot.setPen(QPen(Qt.NoPen))
                self.scene.addItem(dot)
                self.temp_point_items.append(dot)

                if len(self.temp_point_items) == 4:
                    points = [p.rect().center() for p in self.temp_point_items]
                    self.new_annotation_request.emit(points)
                    self.clear_temp_points()
        else:
            point = self.mapToScene(event.pos())
            selected_index = self.annotation_handler.select_annotation(point)
            self.annotation_selected.emit(selected_index)
            self.draw_annotations()
            super().mousePressEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        self.scale(factor, factor)


class ClassSelectionDialog(QDialog):
    def __init__(self, class_names, default_id=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Class")
        layout = QVBoxLayout(self)

        self.label = QLabel("Please select a class for the new annotation:")
        layout.addWidget(self.label)

        self.combo = QComboBox()
        for i, name in class_names.items():
            self.combo.addItem(f"{i}: {name}", i)

        if default_id is not None:
            for i in range(self.combo.count()):
                if self.combo.itemData(i) == default_id:
                    self.combo.setCurrentIndex(i)
                    break

        layout.addWidget(self.combo)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_selected_class_id(self):
        return self.combo.currentData()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO OBB Annotator (PySide6 Edition)")
        self.setGeometry(100, 100, 1200, 800)

        self.config_manager = ConfigManager()
        self.file_manager = FileManager()
        self.annotation_handler = AnnotationHandler()

        self.class_names = {0: "barcode", 1: "qrcode", 2: "dm", 3: "pdf417"}
        self.class_colors = [
            QColor(230, 25, 75),
            QColor(60, 180, 75),
            QColor(255, 225, 25),
            QColor(0, 130, 200),
            QColor(245, 130, 48),
            QColor(145, 30, 180),
            QColor(70, 240, 240),
            QColor(240, 50, 230),
            QColor(210, 245, 60),
            QColor(250, 190, 212),
            QColor(0, 128, 128),
            QColor(220, 190, 255),
            QColor(170, 110, 40),
            QColor(255, 250, 200),
            QColor(128, 0, 0),
            QColor(170, 255, 195),
            QColor(128, 128, 0),
            QColor(255, 215, 180),
            QColor(0, 0, 128),
            QColor(128, 128, 128),
        ]
        self.last_selected_class_id = 9  # Default to 'pending'

        self.init_ui()
        self.apply_stylesheet()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.canvas = ImageCanvas(
            self.annotation_handler, self.class_names, self.class_colors
        )
        main_layout.addWidget(self.canvas, 3)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setFixedWidth(300)
        main_layout.addWidget(right_panel, 1)

        # File/Navigation Group
        file_nav_group = QVBoxLayout()
        self.load_btn = QPushButton("Load Directory")
        self.prev_btn = QPushButton("Previous Image")
        self.next_btn = QPushButton("Next Image")
        file_nav_group.addWidget(self.load_btn)
        file_nav_group.addWidget(self.prev_btn)
        file_nav_group.addWidget(self.next_btn)
        right_layout.addLayout(file_nav_group)

        # Annotation Actions Group
        annotation_actions_group = QVBoxLayout()
        self.save_btn = QPushButton("Save Annotations")
        self.mark_btn = QPushButton("Start Marking")
        self.delete_btn = QPushButton("Delete Selected")
        annotation_actions_group.addWidget(self.save_btn)
        annotation_actions_group.addWidget(self.mark_btn)
        annotation_actions_group.addWidget(self.delete_btn)
        right_layout.addLayout(annotation_actions_group)

        right_layout.addWidget(QLabel("Image Files:"))
        self.image_list = QListWidget()
        right_layout.addWidget(self.image_list)

        right_layout.addWidget(QLabel("Change Class:"))
        self.class_combo = QComboBox()
        for i, name in self.class_names.items():
            self.class_combo.addItem(f"{i}: {name}", i)
        right_layout.addWidget(self.class_combo)

        right_layout.addStretch()

        self.status_label = QLabel("Load a directory to begin.")
        self.statusBar().addWidget(self.status_label)

        self.load_btn.clicked.connect(self.load_directory)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_annotations)
        self.mark_btn.clicked.connect(self.toggle_marking_mode)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.image_list.currentItemChanged.connect(self.on_image_select)
        self.class_combo.currentIndexChanged.connect(self.on_class_change)
        self.canvas.annotation_selected.connect(self.on_annotation_selected)
        self.canvas.mode_changed.connect(self.on_mode_changed)
        self.canvas.new_annotation_request.connect(self.handle_new_annotation)

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QWidget { background-color: #2e2e2e; color: #e0e0e0; font-family: Arial; }
            QMainWindow { border: 1px solid #1e1e1e; }
            QPushButton { background-color: #4a4a4a; border: 1px solid #5a5a5a; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #5a5a5a; }
            QPushButton:pressed { background-color: #6a6a6a; }
            QListWidget { background-color: #3c3c3c; border: 1px solid #5a5a5a; }
            QListWidget::item:selected { background-color: #0078d7; }
            QComboBox { background-color: #3c3c3c; border: 1px solid #5a5a5a; padding: 5px; border-radius: 3px; }
            QGraphicsView { border: 1px solid #5a5a5a; }
            QStatusBar { background-color: #1e1e1e; }
            QLabel { padding-top: 5px; }
        """)

    def load_directory(self):
        initial_dir = self.config_manager.get_last_directory()
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", initial_dir
        )
        if directory:
            try:
                num_images = self.file_manager.load_directory(directory)
                self.config_manager.set_last_directory(directory)
                self.image_list.clear()
                for img_file in self.file_manager.image_files:
                    self.image_list.addItem(img_file.name)
                if num_images > 0:
                    self.image_list.setCurrentRow(0)
                self.status_label.setText(f"Loaded {num_images} images.")
            except ValueError as e:
                QMessageBox.critical(self, "Error", str(e))

    def load_current_image(self):
        image_path = self.file_manager.get_current_image_path()
        if image_path:
            img = cv2.imread(str(image_path))
            if img is None:
                self.status_label.setText(
                    f"Error: Could not read image {image_path.name}"
                )
                return
            h, w = img.shape[:2]
            label_path = self.file_manager.get_label_path(image_path)
            self.annotation_handler.load_annotations(label_path, w, h)
            self.canvas.load_image(image_path)
            self.update_class_combo_for_selection()

    def on_image_select(self, current, previous):
        if current:
            self.file_manager.current_index = self.image_list.row(current)
            self.load_current_image()

    def prev_image(self):
        if self.file_manager.prev_image():
            self.image_list.setCurrentRow(self.file_manager.current_index)

    def next_image(self):
        if self.file_manager.next_image():
            self.image_list.setCurrentRow(self.file_manager.current_index)

    def save_annotations(self):
        image_path = self.file_manager.get_current_image_path()
        if image_path:
            img = cv2.imread(str(image_path))
            if img is None:
                return
            h, w = img.shape[:2]
            label_path = self.file_manager.get_label_path(image_path)
            self.annotation_handler.save_annotations(label_path, w, h)
            self.status_label.setText(f"Annotations saved for {image_path.name}")

    def toggle_marking_mode(self):
        self.canvas.set_marking_mode(not self.canvas.is_marking)

    def on_mode_changed(self, is_marking):
        if is_marking:
            self.mark_btn.setText("Cancel Marking")
            self.mark_btn.setStyleSheet("background-color: #c0392b;")
        else:
            self.mark_btn.setText("Start Marking")
            self.mark_btn.setStyleSheet("")

    def delete_selected(self):
        if self.annotation_handler.delete_selected_annotation():
            self.canvas.draw_annotations()
            self.status_label.setText("Annotation deleted.")

    def on_annotation_selected(self, index):
        self.update_class_combo_for_selection()

    def on_class_change(self, index):
        if self.annotation_handler.selected_index != -1 and index >= 0:
            new_class_id = self.class_combo.itemData(index)
            self.annotation_handler.update_class_id(new_class_id)
            self.last_selected_class_id = new_class_id
            self.canvas.draw_annotations()

    def update_class_combo_for_selection(self):
        if self.annotation_handler.selected_index != -1:
            class_id = self.annotation_handler.annotations[
                self.annotation_handler.selected_index
            ]["class_id"]
            for i in range(self.class_combo.count()):
                if self.class_combo.itemData(i) == class_id:
                    self.class_combo.setCurrentIndex(i)
                    break
        else:
            self.class_combo.setCurrentIndex(-1)

    def handle_new_annotation(self, points):
        final_points = points
        if not self.annotation_handler.is_parallelogram(points):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Shape Correction")
            msg_box.setText("The marked shape is not a parallelogram.")
            msg_box.setInformativeText("Do you want to automatically correct it?")
            correct_btn = msg_box.addButton("Correct It", QMessageBox.AcceptRole)
            keep_btn = msg_box.addButton("Keep Original", QMessageBox.DestructiveRole)
            msg_box.addButton(QMessageBox.Cancel)
            msg_box.exec()

            if msg_box.clickedButton() == correct_btn:
                final_points = self.annotation_handler.correct_to_parallelogram(points)
            elif msg_box.clickedButton() == keep_btn:
                pass
            else:  # Cancel
                self.canvas.clear_temp_points()
                return

        class_dialog = ClassSelectionDialog(
            self.class_names, self.last_selected_class_id, self
        )
        if class_dialog.exec() == QDialog.Accepted:
            class_id = class_dialog.get_selected_class_id()
            self.last_selected_class_id = class_id
            self.annotation_handler.add_annotation(final_points, class_id)
            self.canvas.draw_annotations()
            self.status_label.setText("New annotation added.")

        self.canvas.clear_temp_points()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

