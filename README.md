# GSplat4D: 4D Gaussian Splatting for Dynamic Scene Reconstruction

GSplat4D is a 4D Gaussian Splatting implementation that combines the efficiency of [gsplat](https://github.com/nerfstudio-project/gsplat) with temporal deformation techniques inspired by [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) for dynamic scene reconstruction.

## Overview

This project extends 3D Gaussian Splatting to the temporal domain, enabling high-quality reconstruction and novel view synthesis of dynamic scenes. By leveraging deformation networks and temporal encoding, GSplat4D can capture and render complex dynamic scenes with temporal consistency.

### Key Features

- **4D Gaussian Splatting**: Extends 3D Gaussian Splatting to handle temporal dynamics
- **Efficient Rendering**: Built on top of the high-performance gsplat library
- **Temporal Deformation**: Incorporates deformation networks for smooth temporal transitions
- **COLMAP Integration**: Compatible with COLMAP camera poses and sparse reconstructions
- **Flexible Training**: Configurable training parameters and strategies
- **Multi-view Consistency**: Maintains consistency across multiple viewpoints and time steps

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- PyTorch 1.12+

### Dependencies

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install gsplat
pip install numpy scipy tqdm tyro
pip install imageio pillow
pip install torchmetrics
```

### Clone the Repository

```bash
git clone https://github.com/SuperFCR/gsplat4D.git
cd gsplat4D
```

## Usage

### Basic Training

To train a 4D Gaussian Splatting model on your dynamic scene data:

```bash
python simple_trainer.py --data_dir /path/to/your/data --result_dir results/your_scene
```

### Configuration Options

The training script supports various configuration options:

```bash
python simple_trainer.py \
    --data_dir /path/to/your/data \
    --result_dir results/your_scene \
    --max_steps 40000 \
    --batch_size 1 \
    --data_factor 1.0 \
    --test_every 8
```

### Advanced Configuration

For fine-tuning the training process, you can adjust additional parameters:

```bash
python simple_trainer.py \
    --data_dir /path/to/your/data \
    --result_dir results/your_scene \
    --max_steps 40000 \
    --warm_up 3000 \
    --position_lr_init 1.6e-4 \
    --position_lr_final 1.6e-6 \
    --sh_degree 3 \
    --lambda_dssim 0.2 \
    --init_opa 0.1 \
    --init_scale 1.0
```

Key parameters:
- `--warm_up`: Number of warmup steps (default: 3000)
- `--position_lr_init/final`: Learning rate schedule for position optimization
- `--sh_degree`: Spherical harmonics degree for appearance modeling (default: 3)
- `--lambda_dssim`: Weight for DSSIM loss (default: 0.2)
- `--depth_loss`: Enable depth supervision (default: False)
- `--is_6dof`: Use 6DoF deformation (default: False)

### Data Format

The input data should be organized in COLMAP format with temporal sequences:

```
data_dir/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── images/
│   ├── frame_000/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   ├── frame_001/
│   └── ...
└── transforms.json (optional)
```

## Architecture

GSplat4D consists of several key components:

1. **Deformation Network**: Handles temporal deformation of 3D Gaussians
2. **Temporal Encoding**: Encodes time information for consistent deformation
3. **4D Rasterization**: Renders deformed Gaussians using gsplat
4. **Optimization Strategy**: Adaptive densification and pruning strategies

## Results

The model outputs:
- Rendered images at novel viewpoints and time steps
- 4D Gaussian representations
- Training metrics and loss curves
- Test set evaluations (PSNR, SSIM, LPIPS)

## Acknowledgments

This project builds upon several excellent works:

- **[gsplat](https://github.com/nerfstudio-project/gsplat)**: We thank the gsplat team for providing an efficient and high-performance implementation of 3D Gaussian Splatting rasterization. Their CUDA kernels and optimization strategies form the foundation of our rendering pipeline.

- **[Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)**: We are grateful to the authors of Deformable 3D Gaussians for their pioneering work on temporal deformation of 3D Gaussians. Our deformation network design and training strategies are inspired by their methodology.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@software{gsplat4d2025,
  title={GSplat4D: 4D Gaussian Splatting for Dynamic Scene Reconstruction},
  author={Chaoran Feng},
  year={2025},
  url={https://github.com/SuperFCR/gsplat4D}
}
```

Please also cite the original works that inspired this project:

```bibtex
@article{gsplat2023,
  title={gsplat: An Open-Source Library for Gaussian Splatting},
  author={gsplat contributors},
  year={2023},
  url={https://github.com/nerfstudio-project/gsplat}
}

@article{yang2023deformable,
  title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
  author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Contact

For questions or issues, please open an issue on the GitHub repository.