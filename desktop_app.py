import sys
import os
import cv2
import numpy as np
import rawpy
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QFileDialog, QLabel,
                               QSlider, QListWidget, QSplitter)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import qdarktheme

# 将 Matplotlib 的 3D 画布直接嵌入到 Qt 窗口中
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class FocusAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("焦点拓扑分析系统 (Lightroom 风格)")
        self.resize(1400, 800)

        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 使用 QSplitter 允许用户拖拽改变左右面板宽度
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # === 左侧面板：控制与列表 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 按钮
        self.btn_load = QPushButton("导入图片 (支持 RAW)")
        self.btn_load.clicked.connect(self.load_images)
        left_layout.addWidget(self.btn_load)

        self.btn_run = QPushButton("批量分析选中图片")
        self.btn_run.clicked.connect(self.run_analysis)
        left_layout.addWidget(self.btn_run)

        # 文件列表
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.display_selected_result)
        left_layout.addWidget(QLabel("等待处理的队列："))
        left_layout.addWidget(self.file_list)

        splitter.addWidget(left_panel)

        # === 右侧面板：图像预览区 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 上半部分：2D 图片对比
        top_images_layout = QHBoxLayout()
        self.label_original = QLabel("原图预览区")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("background-color: #202020;")

        self.label_heatmap = QLabel("热力图预览区")
        self.label_heatmap.setAlignment(Qt.AlignCenter)
        self.label_heatmap.setStyleSheet("background-color: #202020;")

        top_images_layout.addWidget(self.label_original)
        top_images_layout.addWidget(self.label_heatmap)
        right_layout.addLayout(top_images_layout)

        # 下半部分：3D 拓扑图嵌入
        self.figure_3d = plt.figure(facecolor='#1e1e1e')
        self.canvas_3d = FigureCanvas(self.figure_3d)
        right_layout.addWidget(self.canvas_3d)

        splitter.addWidget(right_panel)

        # 设置左右面板的初始比例 1:3
        splitter.setSizes([300, 1100])

        # 数据存储
        self.image_paths = []
        self.results = {}  # 存放处理好的结果，方便快速切换查看

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "Images (*.jpg *.png *.nef *.cr2 *.dng)"
        )
        if files:
            for f in files:
                if f not in self.image_paths:
                    self.image_paths.append(f)
                    self.file_list.addItem(os.path.basename(f))

    def run_analysis(self):
        # 这里是之前封装的核心算法，为了演示保持最简
        # 在真实的 App 开发中，这里会放进 QThread 防止主界面卡顿
        for i in range(self.file_list.count()):
            file_path = self.image_paths[i]

            # 1. 读取 (兼容 RAW)
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.nef', '.cr2', '.dng']:
                with rawpy.imread(file_path) as raw:
                    img_rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
                    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(file_path)

            # 缩放加速
            h, w = img.shape[:2]
            scale = 800 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

            # 2. 拉普拉斯与方差计算
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            blk = 16
            hy, hx = gray.shape[0] // blk, gray.shape[1] // blk
            heat = np.zeros((hy, hx))
            for y in range(hy):
                for x in range(hx):
                    heat[y, x] = lap[y * blk:(y + 1) * blk, x * blk:(x + 1) * blk].var()

            heat = np.log1p(heat)
            norm_h = cv2.normalize(heat, None, 0, 1, cv2.NORM_MINMAX)
            heat_full = cv2.resize(norm_h, (img.shape[1], img.shape[0]))
            color_map = cv2.applyColorMap(np.uint8(255 * heat_full), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.5, color_map, 0.5, 0)

            # 存入字典
            self.results[file_path] = {
                'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                'heatmap': heat
            }

        # 分析完成后，自动显示第一张
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)

    def display_selected_result(self, current_item):
        if not current_item:
            return

        # 获取对应的文件路径
        idx = self.file_list.row(current_item)
        file_path = self.image_paths[idx]

        if file_path in self.results:
            res = self.results[file_path]

            # 渲染 2D 到 QLabel
            self._show_cv_image_on_label(res['original'], self.label_original)
            self._show_cv_image_on_label(res['overlay'], self.label_heatmap)

            # 渲染 3D 到内嵌画布
            self.figure_3d.clear()
            ax = self.figure_3d.add_subplot(111, projection='3d')
            y_shape, x_shape = res['heatmap'].shape
            X, Y = np.meshgrid(np.arange(x_shape), np.arange(y_shape))
            ax.plot_surface(X, Y, res['heatmap'], cmap='magma', linewidth=0)
            ax.axis('off')

            # 设置 3D 面板的背景色，贴合暗黑主题
            self.figure_3d.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            self.canvas_3d.draw()

    def _show_cv_image_on_label(self, cv_rgb_img, label):
        """将 OpenCV 矩阵转换为 Qt 可显示的图像"""
        h, w, ch = cv_rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(cv_rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        # 等比例缩放以适应 label 大小
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 一键应用高级暗黑主题 (类似 Lightroom)
    qdarktheme.setup_theme("dark")
    window = FocusAnalyzerApp()
    window.show()
    sys.exit(app.exec())