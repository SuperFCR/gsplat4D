# GSplat4D

A clean and efficient implementation of 4D Gaussian Splatting for dynamic scene reconstruction, built on top of the high-performance [gsplat](https://github.com/nerfstudio-project/gsplat) library.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **High-Performance 4D Rendering**: Built on [gsplat](https://github.com/nerfstudio-project/gsplat) for optimized Gaussian splatting operations
- **Deformable Neural Networks**: Incorporates temporal deformation networks for dynamic scene modeling
- **Multiple Deformation Modes**: Supports both standard 3D deformation and 6DOF rigid transformations
- **Progressive Training**: Implements warm-up phases and progressive spherical harmonics training
- **Comprehensive Evaluation**: Built-in PSNR, SSIM, and LPIPS metrics for quality assessment
- **Flexible Data Loading**: Compatible with COLMAP datasets and custom 4D data formats
- **Multi-GPU Support**: Distributed training capabilities for large-scale scenes

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n gsplat4d python=3.8
conda activate gsplat4d

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install gsplat
pip install gsplat

# Install other dependencies
pip install numpy tqdm tyro Pillow imageio torchmetrics
```

### Installation

```bash
git clone https://github.com/SuperFCR/gsplat4D.git
cd gsplat4D
pip install -e .
```
## 📁 Data Format

GSplat4D expects data in the following structure:

```
data_dir/
├── images/           # Input images
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── sparse/           # COLMAP sparse reconstruction
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── poses_bounds.npy  # Camera poses (optional)
```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GSplat](https://github.com/nerfstudio-project/gsplat) for the high-performance Gaussian splatting backend
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) for the original 3DGS implementation
- [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) for deformation network inspiration

## 🔗 Related Work

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [4D Gaussian Splatting](https://guanjunwu.github.io/4dgs/)
- [GSplat Library](https://github.com/nerfstudio-project/gsplat)
- [Nerfstudio](https://github.co

  
