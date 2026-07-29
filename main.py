import sys
import os
import hashlib
import json
import cv2
import numpy as np
import rawpy
import exifread  # 【新增】读取相机 EXIF 信息的库

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel,
                               QTreeWidget, QTreeWidgetItem, QSplitter,
                               QProgressBar, QMessageBox, QTreeWidgetItemIterator, QMenu)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QAction
import qdarktheme

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# ================= 缓存目录设置 (打包专用安全路径) =================
import platform
if platform.system() == "Windows":
    # Win 会存在: C:\Users\用户名\AppData\Local\FocusAnalyzer\focus_cache
    BASE_DIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "FocusAnalyzer")
else:
    # Mac 会存在: /Users/用户名/Library/Application Support/FocusAnalyzer/focus_cache
    BASE_DIR = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "FocusAnalyzer")

CACHE_DIR = os.path.join(BASE_DIR, "focus_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
SESSION_FILE = os.path.join(CACHE_DIR, "session.json")


def get_cache_paths(file_path):
    file_id = hashlib.md5(file_path.encode('utf-8')).hexdigest()
    return {
        'original': os.path.join(CACHE_DIR, f"{file_id}_orig.jpg"),
        'overlay': os.path.join(CACHE_DIR, f"{file_id}_over.jpg"),
        'heatmap': os.path.join(CACHE_DIR, f"{file_id}_heat.npy")
    }


# ================= EXIF 解析工具 =================
def get_exif_string(file_path):
    """提取相机的 EXIF 信息（含镜头识别）并格式化为字符串"""
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        if not tags:
            return "⚠️ EXIF 信息: 未包含拍摄参数 (可能是导出的图片剥离了数据)"

        # 1. 相机型号
        cam_model = str(tags.get('Image Model', '未知相机'))

        # 2. 镜头型号 (双重校验防漏)
        lens_model = str(tags.get('EXIF LensModel', ''))
        if not lens_model or lens_model.strip() == '':
            lens_model = str(tags.get('Image LensModel', '未知镜头'))

        # 3. 快门 (保留 1/200 这样的分数格式)
        shutter = str(tags.get('EXIF ExposureTime', '未知'))
        if shutter != '未知': shutter += 's'

        # 4. ISO
        iso = str(tags.get('EXIF ISOSpeedRatings', '未知'))
        if iso != '未知': iso = f"ISO {iso}"

        # 5. 光圈 (把分数 28/10 计算成 f/2.8)
        aperture_tag = tags.get('EXIF FNumber')
        aperture = '未知'
        if aperture_tag and aperture_tag.values:
            val = aperture_tag.values[0]
            if val.den != 0:
                aperture = f"f/{val.num / val.den:.1f}".replace('.0', '')

        # 6. 焦距
        focal_tag = tags.get('EXIF FocalLength')
        focal = '未知'
        if focal_tag and focal_tag.values:
            val = focal_tag.values[0]
            if val.den != 0:
                focal = f"{int(val.num / val.den)}mm"

        return f"📷 {cam_model}  |  🔭 {lens_model}  |  🎯 {focal}  |  ⭕ {aperture}  |  ⏱️ {shutter}  |  💡 {iso}"

    except Exception:
        return "⚠️ EXIF 信息: 读取失败"

# ================= 后台处理线程 =================
class AnalysisWorker(QThread):
    progress = Signal(int, int)
    finished = Signal()
    log = Signal(str)

    def __init__(self, files_to_process):
        super().__init__()
        self.files = files_to_process

    def run(self):
        total = len(self.files)
        for i, file_path in enumerate(self.files):
            cache = get_cache_paths(file_path)

            if os.path.exists(cache['original']) and os.path.exists(cache['heatmap']):
                self.log.emit(f"已跳过 (读取缓存): {os.path.basename(file_path)}")
                self.progress.emit(i + 1, total)
                continue

            self.log.emit(f"正在分析: {os.path.basename(file_path)}")
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.nef', '.cr2', '.cr3', '.arw', '.dng']:
                    with rawpy.imread(file_path) as raw:
                        img_rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
                        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                else:
                    img = cv2.imread(file_path)

                if img is None: continue

                h, w = img.shape[:2]
                scale = 800 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

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

                cv2.imwrite(cache['original'], img)
                cv2.imwrite(cache['overlay'], overlay)
                np.save(cache['heatmap'], heat)

            except Exception as e:
                self.log.emit(f"处理出错: {str(e)}")

            self.progress.emit(i + 1, total)
        self.finished.emit()


# ================= 主窗口 UI =================
class FocusAnalyzerPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("焦点拓扑分析系统")
        self.resize(1400, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- 左侧控制区 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("👇 请将图片或文件夹拖拽至此区域")
        self.tree.setStyleSheet("QTreeWidget::item { padding: 4px; }")
        self.tree.itemClicked.connect(self.display_cache)
        self.tree.itemChanged.connect(self.update_selected_count)

        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.keyPressEvent = self._tree_key_press_event

        self.tree.setAcceptDrops(True)
        self.tree.dragEnterEvent = self._drag_enter_event
        self.tree.dragMoveEvent = self._drag_move_event
        self.tree.dropEvent = self._drop_event
        left_layout.addWidget(self.tree)

        self.lbl_selected_count = QLabel("已选择: 0 个文件")
        self.lbl_selected_count.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 13px; margin-bottom: 5px;")
        self.lbl_selected_count.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_selected_count)

        self.btn_run = QPushButton("⚡ 开始分析选中的文件")
        self.btn_run.setStyleSheet("background-color: #2b579a; font-weight: bold; padding: 8px;")
        self.btn_run.clicked.connect(self.start_analysis)
        left_layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        left_layout.addWidget(self.status_label)

        splitter.addWidget(left_panel)

        # --- 右侧预览区 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        top_imgs = QHBoxLayout()
        self.label_orig = QLabel("选择文件以预览")
        self.label_orig.setAlignment(Qt.AlignCenter)
        self.label_orig.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
        self.label_orig.setMinimumSize(1, 1)  # 【新增】：锁死最小尺寸，禁止反向推挤布局

        self.label_heat = QLabel("热力图")
        self.label_heat.setAlignment(Qt.AlignCenter)
        self.label_heat.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
        self.label_heat.setMinimumSize(1, 1)  # 【新增】：锁死最小尺寸

        top_imgs.addWidget(self.label_orig)
        top_imgs.addWidget(self.label_heat)
        # 【核心修改 1】：给顶部的 2D 图像区域分配 2 份的空间 (Stretch = 2)
        right_layout.addLayout(top_imgs, 2)

        # 3D 图表区域
        self.fig_3d = plt.figure(facecolor='#1e1e1e')
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.canvas_3d.setMinimumSize(1, 1)  # 【新增】：防止 Matplotlib 画布推挤
        # 【核心修改 2】：给底部的 3D 图表分配 1 份的空间 (Stretch = 1)
        right_layout.addWidget(self.canvas_3d, 1)

        # EXIF 信息条
        self.label_exif = QLabel("📷 拍摄参数: 暂无数据")
        self.label_exif.setAlignment(Qt.AlignCenter)
        self.label_exif.setStyleSheet(
            "color: #a0a0a0; font-size: 14px; font-weight: bold; "
            "padding: 10px; background-color: #1e1e1e; border-top: 1px solid #333;"
        )
        # 【核心修改 3】：给参数条分配 0 份空间，让它保持原本固定的高度，不被拉伸
        right_layout.addWidget(self.label_exif, 0)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050])

        self.load_session()
        self.update_selected_count()

    # ================= 删除与缓存清理 =================
    def _show_context_menu(self, position):
        if not self.tree.selectedItems(): return
        menu = QMenu()

        # 选项 1：彻底移除
        delete_action = QAction("🗑️ 从列表中移除 (并清理缓存)", self)
        delete_action.triggered.connect(self.delete_selected_items)
        menu.addAction(delete_action)

        # 选项 2：仅清理缓存并重置状态
        clear_cache_action = QAction("🧹 仅清理缓存 (恢复为未分析状态)", self)
        clear_cache_action.triggered.connect(self.clear_cache_only)
        menu.addAction(clear_cache_action)

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _tree_key_press_event(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_items()
        else:
            QTreeWidget.keyPressEvent(self.tree, event)

    # 【新增功能】：仅清理缓存并恢复未分析状态
    def clear_cache_only(self):
        selected_items = self.tree.selectedItems()
        if not selected_items: return

        reply = QMessageBox.question(
            self, "确认清理",
            "确定要清理选中项目的缓存吗？\n(这将会把它们恢复为未分析状态，并自动打勾以便重新分析)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes: return

        self.tree.blockSignals(True)
        for item in selected_items:
            self._clear_item_cache_and_reset(item)

        self.tree.blockSignals(False)
        self.update_selected_count()

        # 为了防止画面还残留着刚才清掉的图片，重置右侧预览区
        self.label_orig.clear()
        self.label_heat.clear()
        self.label_orig.setText("缓存已清理，请重新分析")
        self.label_heat.setText("热力图 (暂无数据)")
        self.fig_3d.clear()
        self.canvas_3d.draw()
        self.label_exif.setText("📷 拍摄参数: 暂无数据")
        self.status_label.setText("就绪")

    def _clear_item_cache_and_reset(self, item):
        """递归清理缓存并重置 UI 状态的核心逻辑"""
        if item.childCount() > 0:
            for i in range(item.childCount()):
                self._clear_item_cache_and_reset(item.child(i))
        else:
            file_path = item.data(0, Qt.UserRole)
            if file_path:
                # 物理删除缓存文件
                for path in get_cache_paths(file_path).values():
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass

                # 恢复 UI 状态：去掉前面的 ✅，并重新打上勾
                current_text = item.text(0)
                if current_text.startswith("✅ "):
                    item.setText(0, current_text[2:])  # 截掉开头的 ✅ 和空格
                item.setCheckState(0, Qt.Checked)

    # 原有的彻底删除逻辑保持不变
    def delete_selected_items(self):
        # ... (这里保留你原本的 delete_selected_items 里的代码不变) ...
        selected_items = self.tree.selectedItems()
        if not selected_items: return

        reply = QMessageBox.question(
            self, "确认移除",
            "确定要移除选中的项目吗？\n(这将会清理它产生的本地缓存文件，但不会删除原图)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes: return

        self.tree.blockSignals(True)
        for item in selected_items:
            if item.treeWidget() is None: continue
            self._delete_item_and_cache(item)

        self.tree.blockSignals(False)
        self.update_selected_count()

        if self.tree.topLevelItemCount() == 0:
            self.label_orig.clear()
            self.label_heat.clear()
            self.label_orig.setText("列表已清空")
            self.label_heat.setText("")
            self.fig_3d.clear()
            self.canvas_3d.draw()
            self.label_exif.setText("📷 拍摄参数: 暂无数据")
            self.status_label.setText("已清理完毕")

    def _delete_item_and_cache(self, item):
        # ... (这里保留你原本的 _delete_item_and_cache 里的代码不变) ...
        if item.childCount() > 0:
            children = [item.child(i) for i in range(item.childCount())]
            for child in children: self._delete_item_and_cache(child)
        else:
            file_path = item.data(0, Qt.UserRole)
            if file_path:
                for path in get_cache_paths(file_path).values():
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass

        parent = item.parent()
        if parent:
            parent.removeChild(item)
            if parent.childCount() == 0:
                idx = self.tree.indexOfTopLevelItem(parent)
                if idx >= 0: self.tree.takeTopLevelItem(idx)
        else:
            idx = self.tree.indexOfTopLevelItem(item)
            if idx >= 0: self.tree.takeTopLevelItem(idx)

    # ================= 核心统计 =================
    def update_selected_count(self, item=None, column=None):
        count = 0
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            curr_item = iterator.value()
            if curr_item.childCount() == 0 and curr_item.checkState(0) == Qt.Checked:
                count += 1
            iterator += 1

        self.lbl_selected_count.setText(f"已选择: {count} 个文件")

        if count == 0:
            self.btn_run.setEnabled(False)
            self.btn_run.setStyleSheet("background-color: #555; color: #888; font-weight: bold; padding: 8px;")
        else:
            self.btn_run.setEnabled(True)
            self.btn_run.setStyleSheet("background-color: #2b579a; color: white; font-weight: bold; padding: 8px;")

    # ================= 拖拽支持与列表挂载 =================
    def _drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drag_move_event(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drop_event(self, event):
        urls = event.mimeData().urls()
        files_to_add = []

        self.tree.blockSignals(True)
        for url in urls:
            path = url.toLocalFile()
            if not path: continue
            if os.path.isdir(path):
                self.import_folder_by_path(path)
            elif os.path.isfile(path) and path.lower().endswith(('.jpg', '.png', '.nef', '.cr2', '.dng')):
                files_to_add.append(path)

        if files_to_add: self.import_files_by_paths(files_to_add)

        self.tree.blockSignals(False)
        self.update_selected_count()

    def add_tree_item(self, parent, name, full_path, is_folder=False):
        item = QTreeWidgetItem(parent)
        item.setText(0, name)
        item.setData(0, Qt.UserRole, full_path)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

        cache = get_cache_paths(full_path)
        if os.path.exists(cache['heatmap']):
            item.setCheckState(0, Qt.Unchecked)
            item.setText(0, f"✅ {name}")
        else:
            item.setCheckState(0, Qt.Checked)
        return item

    def import_folder_by_path(self, folder):
        exts = ('.jpg', '.png', '.nef', '.cr2', '.dng')
        valid_files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        if not valid_files: return

        clean_path = os.path.normpath(folder)
        folder_name = os.path.basename(clean_path)
        if not folder_name: folder_name = clean_path

        root = QTreeWidgetItem(self.tree)
        root.setText(0, f"📁 {folder_name}")
        root.setData(0, Qt.UserRole, folder)
        root.setFlags(root.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
        root.setCheckState(0, Qt.Checked)

        for f in valid_files:
            self.add_tree_item(root, f, os.path.join(folder, f), False)
        self.tree.expandAll()

    def import_files_by_paths(self, files):
        for f in files: self.add_tree_item(self.tree, os.path.basename(f), f, False)

    # ================= 启动分析与显示 =================
    def start_analysis(self):
        files_to_process = []
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.childCount() == 0 and item.checkState(0) == Qt.Checked:
                path = item.data(0, Qt.UserRole)
                if path and os.path.isfile(path):
                    files_to_process.append(path)
            iterator += 1

        if not files_to_process: return

        self.tree.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ 正在全力分析中...")
        self.btn_run.setStyleSheet("background-color: #b8860b; color: white; font-weight: bold; padding: 8px;")

        self.progress_bar.setMaximum(len(files_to_process))
        self.progress_bar.setValue(0)

        self.worker = AnalysisWorker(files_to_process)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(lambda msg: self.status_label.setText(msg))
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()

    def analysis_finished(self):
        self.tree.setEnabled(True)
        self.btn_run.setText("⚡ 开始分析选中的文件")
        self.status_label.setText("任务全部完成！")

        self.tree.blockSignals(True)
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.childCount() == 0:
                path = item.data(0, Qt.UserRole)
                if os.path.exists(get_cache_paths(path)['heatmap']):
                    item.setCheckState(0, Qt.Unchecked)
                    if not item.text(0).startswith("✅"):
                        item.setText(0, f"✅ {item.text(0)}")
            iterator += 1
        self.tree.blockSignals(False)
        self.update_selected_count()

    def display_cache(self, item, column):
        if item.childCount() > 0: return
        file_path = item.data(0, Qt.UserRole)
        cache = get_cache_paths(file_path)

        # 【新增】：每次点击图片，瞬间读取该图片的 EXIF 信息并展示
        self.label_exif.setText(get_exif_string(file_path))

        if os.path.exists(cache['heatmap']):
            self.status_label.setText(f"极速加载中: {item.text(0)}")
            orig = cv2.imread(cache['original'])
            over = cv2.imread(cache['overlay'])
            heat = np.load(cache['heatmap'])

            self._show_img(orig, self.label_orig)
            self._show_img(over, self.label_heat)

            self.fig_3d.clear()
            ax = self.fig_3d.add_subplot(111, projection='3d')
            y_shape, x_shape = heat.shape
            X, Y = np.meshgrid(np.arange(x_shape), np.arange(y_shape))
            ax.plot_surface(X, Y, heat, cmap='magma', linewidth=0)
            ax.axis('off')
            self.fig_3d.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            self.canvas_3d.draw()
            self.status_label.setText("就绪")
        else:
            self.label_orig.clear()
            self.label_heat.clear()
            self.label_orig.setText("图片尚未分析")
            self.label_heat.setText("热力图 (图片尚未分析)")
            self.fig_3d.clear()
            self.canvas_3d.draw()
            self.status_label.setText("就绪")

    def _show_img(self, cv_bgr, label):
        rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ================= 记忆功能 =================
    def load_session(self):
        if os.path.exists(SESSION_FILE):
            try:
                self.tree.blockSignals(True)
                with open(SESSION_FILE, 'r') as f:
                    paths = json.load(f)
                indep_files = []
                for path in paths:
                    if os.path.isdir(path):
                        self.import_folder_by_path(path)
                    elif os.path.isfile(path):
                        indep_files.append(path)
                if indep_files: self.import_files_by_paths(indep_files)
                self.tree.blockSignals(False)
            except:
                pass

    def closeEvent(self, event):
        paths = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            path = item.data(0, Qt.UserRole)
            if path: paths.append(path)
        with open(SESSION_FILE, 'w') as f:
            json.dump(paths, f)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    window = FocusAnalyzerPro()
    window.show()
    sys.exit(app.exec())