import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy  # 新增库
import os     # 新增库，用于获取文件后缀名
from mpl_toolkits.mplot3d import Axes3D

def generate_focus_heatmap(image_path, block_size=32, resize_max_dim=1200, alpha=0.5):
    # 1. 智能读取图片 (兼容 RAW 与普通格式)
    ext = os.path.splitext(image_path)[1].lower()

    if ext in ['.nef', '.cr2', '.cr3', '.arw', '.dng']:
        print(f"检测到 RAW 文件，正在使用 rawpy 解码: {image_path}")
        try:
            with rawpy.imread(image_path) as raw:
                # postprocess() 将传感器原始拜尔阵列数据转换为 RGB 图像
                # use_camera_wb=True: 使用拍摄时的白平衡，保证颜色正常
                # no_auto_bright=True: 关闭自动提亮，保留原始曝光
                img_rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)

                # rawpy 输出的是 RGB 格式，为了和后续的 OpenCV 逻辑无缝衔接，转换为 BGR
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"RAW 文件解码失败，请确保 rawpy 已更新到最新版: {e}")
    else:
        # 普通格式回退使用 OpenCV 读取
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片，请检查路径: {image_path}")

    # 缩放图片以提升计算速度
    h, w = img.shape[:2]
    scale = resize_max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 2. 转换为灰度图并计算拉普拉斯响应
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # 3. 使用滑动窗口计算网格方差
    h_small, w_small = gray.shape
    num_blocks_y = h_small // block_size
    num_blocks_x = w_small // block_size
    heatmap_grid = np.zeros((num_blocks_y, num_blocks_x))

    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            y_start, y_end = y * block_size, (y + 1) * block_size
            x_start, x_end = x * block_size, (x + 1) * block_size
            window = laplacian[y_start:y_end, x_start:x_end]
            heatmap_grid[y, x] = window.var()

    # 4. 对数变换与归一化
    heatmap_grid = np.log1p(heatmap_grid)
    norm_heatmap = cv2.normalize(heatmap_grid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # 5. 生成 2D 热力图与原图的叠加
    heatmap_full = cv2.resize(norm_heatmap, (w_small, h_small), interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_full), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    # 颜色转换 (BGR 转 RGB 给 Matplotlib 使用)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return img_rgb, overlay_rgb, heatmap_grid


def plot_results(img_rgb, overlay_rgb, heatmap_grid):
    """同时绘制 2D 热力图和 3D 拓扑图"""
    # --- 绘制 2D 图像 ---
    fig_2d = plt.figure(figsize=(14, 7))
    ax1 = fig_2d.add_subplot(1, 2, 1)
    ax1.set_title("Original Image")
    ax1.imshow(img_rgb)
    ax1.axis("off")

    ax2 = fig_2d.add_subplot(1, 2, 2)
    ax2.set_title("Focus Heatmap (Red = In Focus)")
    ax2.imshow(overlay_rgb)
    ax2.axis("off")
    plt.tight_layout()

    # --- 绘制 3D 拓扑图 ---
    fig_3d = plt.figure(figsize=(10, 8))
    ax3 = fig_3d.add_subplot(111, projection='3d')
    y_shape, x_shape = heatmap_grid.shape
    X, Y = np.meshgrid(np.arange(0, x_shape, 1), np.arange(0, y_shape, 1))

    surf = ax3.plot_surface(X, Y, heatmap_grid, cmap='magma', linewidth=0, antialiased=True, alpha=0.9)
    ax3.view_init(elev=45, azim=30)
    ax3.set_title("3D Focus Topology Map")
    fig_3d.colorbar(surf, shrink=0.5, aspect=5)

    # 展示所有图表
    plt.show()


if __name__ == "__main__":
    # ====== 在这里替换成你的图片名称 ======
    # 请确保图片和 main.py 放在同一个文件夹下
    TARGET_IMAGE = "1.jpg"

    try:
        print("正在分析图像焦点，请稍候...")
        original, heatmap_overlay, topology_data = generate_focus_heatmap(TARGET_IMAGE)
        plot_results(original, heatmap_overlay, topology_data)
    except Exception as e:
        print(f"发生错误: {e}")