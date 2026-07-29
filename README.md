# Focus Topology Analysis System (焦点拓扑分析系统)

一款专为摄影后期工作流设计的专业焦点分析与验证工具。通过底层算法生成热力图与 3D 拓扑图，帮助摄影师精准判断照片的物理合焦平面，告别“伪合焦”与跑焦困扰。

## ✨ 核心功能 (Features)

* **核心算法验证**：基于拉普拉斯算子 (Laplacian) 的梯度方差算法，客观测算像素级物理锐度与高频细节。
* **双视图比现代**：左侧提供原始图像，右侧提供对应生成的“算法热力图”，直观显现最清晰的物理平面。
* **3D 拓扑表面图 (Surface Plot)**：通过三维地形图的形式，降维打击景深偏移，精准判断前跑焦 (Front Focus) 与后跑焦 (Back Focus)。
* **原生缓存管理**：内建独立的高效缓存系统 (`focus_cache`)。支持在文件树中右键“仅清除缓存 (Clear Cache Only)”或“移除文件 (Remove)”，降低磁盘占用且不影响分析列表。
* **跨平台支持**：支持打包分发，兼容 macOS 与 Windows 11 系统。

## 🛠️ 技术栈 (Tech Stack)

* **开发语言**: Python
* **图形界面**: [如有用到填写如 PySide6 / PyQt]
* **图像处理**: OpenCV, NumPy
* **打包工具**: PyInstaller (单目录解压模式，启动无终端窗口)

## 🚀 快速运行 (Quick Start)

如果你想在本地开发环境中运行此项目：

1. 克隆仓库到本地：
   ```bash
   git clone [https://github.com/Collapsar-Echo/FocusAnalyzer.git](https://github.com/Collapsar-Echo/FocusAnalyzer.git)
   ```
2. 安装相关依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行主程序：
   ```bash
   python main.py
   ```

## 📦 编译与打包 (Build & Release)

项目配置了完整的 PyInstaller 打包指令。

**macOS 环境打包命令**：
```bash
pyinstaller FocusAnalyzer.spec
```
或
```bash
pyinstaller --noconsole --windowed --onedir --icon="logo.icns" --name "FocusAnalyzer" main.py
```
*(编译产物将位于 `dist/FocusAnalyzer.app` 目录中)*