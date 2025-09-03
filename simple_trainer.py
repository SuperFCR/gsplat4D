import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tyro
from random import randint
from PIL import Image
import imageio

# Import gsplat for actual Gaussian Splatting rendering
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

# Import the original DeformModel and required utilities
from utils.time_utils import DeformNetwork
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, knn, rgb_to_sh, get_linear_noise_func
from utils.colmap_utils import Parser, Dataset
from utils.rigid_utils import from_homogenous, to_homogenous
# Import metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


@dataclass
class Config4D:
    """Configuration for 4D Gaussian Splatting Reconstruction"""
    # Basic settings
    data_dir: str = "/share/magic_group/aigc/fcr/difix4d/data/4DNex-10M/dynamic_3_colmap/00015772"
    result_dir: str = "results/4d_reconstruction"
    data_factor: float = 1.0
    normalize_world_space: bool = True
    test_every: int = 8
    
    # Training parameters
    warm_up: int = 3000
    max_steps: int = 40_000
    batch_size: int = 1
    patch_size: Optional[int] = None
    
    # DeformModel specific parameters
    is_blender: bool = False
    is_6dof: bool = False
    
    # Training arguments for DeformModel
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    deform_lr_max_steps: int = 40000
    
    # Gaussian Splatting parameters
    sh_degree: int = 3
    init_opa: float = 0.1
    init_scale: float = 1.0
    
    # Loss parameters
    depth_loss: bool = False
    lambda_dssim: float = 0.2
    lpips_net: Literal["vgg", "alex"] = "alex"
    
    # Rendering parameters
    global_scale: float = 1.0
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False
    
    # Densification strategy
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(default_factory=DefaultStrategy)
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0007

    # Evaluation parameters
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 10_000, 20_000, 30_000, 40_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000, 40_000])


class DeformModel:
    """Original DeformModel class for 4D reconstruction"""
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


def create_4d_splats_with_optimizers(
    parser: Parser,
    is_blender: bool = False,
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    cfg = None,
) -> Tuple[torch.nn.ParameterDict, DeformModel, Dict[str, torch.optim.Optimizer]]:
    """Create 4D Gaussian splats and optimizers using original DeformModel"""
    
    if is_blender:
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3)) 
    else:
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
    
    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]
    
    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))
    
    # 4D parameter: canonical positions (positions at t=canonical_time)
    canonical_means = points.clone()
    params = [
        ("means", torch.nn.Parameter(canonical_means), 1.6e-4 / 10 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3 / 5),
        ("quats", torch.nn.Parameter(quats), 1e-3 / 5),
        ("opacities", torch.nn.Parameter(opacities), 5e-2 / 5),
    ]
    
    # Color representation using spherical harmonics
    if feature_dim is None:
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3 / 50))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20 / 50))
    else:
        features = torch.rand(N, feature_dim)
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))
    
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    
    # Create DeformModel
    deform_model = DeformModel(is_blender=is_blender, is_6dof=cfg.is_6dof)  
    deform_model.train_setting(cfg)
    
    # Scale learning rate based on batch size
    BS = batch_size * world_size
    optimizer_class = torch.optim.Adam
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    
    return splats, deform_model, optimizers


class Runner4D:
    """4D Gaussian Splatting trainer with actual gsplat rendering"""
    
    def __init__(self, cfg: Config4D, world_rank: int = 0, world_size: int = 1):
        self.cfg = cfg
        self.world_rank = world_rank
        self.world_size = world_size
        self.device = f"cuda:{world_rank}" if torch.cuda.is_available() else "cpu"
        
        # Create output directories
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        
        # Load 4D data
        self.load_4d_data()
        
        # Create 4D model
        self.splats, self.deform_model, self.optimizers = create_4d_splats_with_optimizers(
            parser=self.parser,
            is_blender=cfg.is_blender,
            scene_scale=self.scene_scale,
            cfg=cfg,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size
        )
        
        print(f"4D Model initialized. Number of GS: {len(self.splats['means'])}")
        print(f"DeformModel: is_blender={cfg.is_blender}, is_6dof={cfg.is_6dof}")
        
        # Initialize smooth noise function (same as deformable 3DGS)
        self.smooth_term = get_linear_noise_func(
            lr_init=0.1, 
            lr_final=1e-15, 
            lr_delay_mult=0.01, 
            max_steps=20000
        )
        
        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        
        # Loss functions and metrics
        self.l1_loss = torch.nn.L1Loss()
        
        # Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")
        
        # Background color
        bg_color = [1, 1, 1] if cfg.is_blender else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        
        # Training metrics history
        self.metrics_history = defaultdict(list)
        self.best_psnr = 0.0
        self.best_iteration = 0
        
    def load_4d_data(self):
        """Load 4D data"""
        print("Loading 4D data...")
        
        self.parser = Parser(
            data_dir=self.cfg.data_dir,
            factor=self.cfg.data_factor,
            normalize=self.cfg.normalize_world_space,
            test_every=self.cfg.test_every,
        )
        
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=self.cfg.patch_size,
            load_depths=self.cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        
        # Get viewpoint stack
        # self.viewpoint_stack = self.trainset.cameras.copy()
        self.total_frame = self.parser.cam_nums
        self.time_interval = 1.0 / self.total_frame
        self.scene_scale = self.parser.scene_scale * 1.1 * self.cfg.global_scale
        
        print(f"Scene scale: {self.scene_scale}")
        print(f"Total frames: {self.total_frame}")

    def get_deformed_points(self, fid: torch.Tensor, iteration: int = 0) -> torch.Tensor:
        
        """Get deformed point positions using frame ID (following deformable 3DGS)"""
        canonical_means = self.splats["means"]
        N = canonical_means.shape[0]
        
        # Create time input for all Gaussians (same as deformable 3DGS)
        time_input = fid.unsqueeze(0).expand(N, -1)
        
        # Add smooth noise (same as deformable 3DGS)
        if not self.cfg.is_blender:
            ast_noise = torch.randn(1, 1, device=self.device).expand(N, -1) * self.time_interval * self.smooth_term(iteration)
            time_input = time_input + ast_noise
        
        # Get deformation
        d_xyz, d_rotation, d_scaling = self.deform_model.step(canonical_means, time_input)
        
        return d_xyz, d_rotation, d_scaling
    
        # if self.cfg.is_6dof:
        #     if torch.is_tensor(d_xyz) is False:
        #         deformed_means = canonical_means 
        #     else:
        #         if d_xyz.dim() == 3:  # [N, 4, 4] 或 [N, 3, 3]
        #             if d_xyz.shape[-1] == 4:  # 4x4 
        #                 homogenous_points = to_homogenous(canonical_means)  # [N, 4]
        #                 deformed_homogenous = torch.bmm(d_xyz, homogenous_points.unsqueeze(-1)).squeeze(-1)  # [N, 4]
        #                 deformed_means = from_homogenous(deformed_homogenous)  # [N, 3]
        #             elif d_xyz.shape[-1] == 3:  # 3x3 matrix
        #                 deformed_means = torch.bmm(d_xyz, canonical_means.unsqueeze(-1)).squeeze(-1)  # [N, 3]
        #             else:
        #                 raise ValueError(f"Unexpected d_xyz shape: {d_xyz.shape}")
        #         elif d_xyz.dim() == 2 and d_xyz.shape[-1] == 3:  # [N, 3] 
        #             deformed_means = canonical_means + d_xyz
        #         else:
        #             raise ValueError(f"Unexpected d_xyz shape for 6DOF: {d_xyz.shape}")
        # else:
        #     deformed_means = canonical_means + d_xyz 
        
        # deformed_quats = canonical_quats + d_rotation if d_rotation is not None else canonical_quats
        # deformed_scales = canonical_scales + d_scaling if d_scaling is not None else canonical_scales
        
        # return deformed_means, deformed_quats, deformed_scales
        
    def rasterize_4d_splats(self, camtoworlds: torch.Tensor, Ks: torch.Tensor, 
                        width: int, height: int, d_xyz, d_rotation, d_scaling, 
                        is_6dof=False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """4D Gaussian rasterization using actual gsplat (following original 4DGS style)"""
        
        # Get canonical Gaussian parameters
        canonical_means = self.splats["means"]
        canonical_quats = self.splats["quats"]
        canonical_scales = self.splats["scales"]
        opacities = torch.sigmoid(self.splats["opacities"])
        
        # Apply 4D deformations (following original 4DGS render function)
        if is_6dof:
            if torch.is_tensor(d_xyz) is False:
                means3D = canonical_means
            else:
                # 6DOF transformation using homogeneous coordinates
                if d_xyz.dim() == 3 and d_xyz.shape[-1] == 4:  # [N, 4, 4] transformation matrices
                    homogenous_points = to_homogenous(canonical_means)
                    deformed_homogenous = torch.bmm(d_xyz, homogenous_points.unsqueeze(-1)).squeeze(-1)
                    means3D = from_homogenous(deformed_homogenous)
                else:
                    # Fallback to simple addition
                    means3D = canonical_means + d_xyz
        else:
            # Standard deformation: simple addition (like original 4DGS)
            if torch.is_tensor(d_xyz):
                means3D = canonical_means + d_xyz
            else:
                means3D = canonical_means
        
        # Apply rotation deformation
        if torch.is_tensor(d_rotation):
            rotations = canonical_quats + d_rotation
        else:
            rotations = canonical_quats
        
        # Apply scaling deformation
        if torch.is_tensor(d_scaling):
            scales = torch.exp(canonical_scales + d_scaling)  # Apply deformation then exp
        else:
            scales = torch.exp(canonical_scales)
        
        # Spherical harmonics colors
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)
        
        # Use gsplat rasterization with deformed parameters
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means3D,           
            quats=rotations,         
            scales=scales,          
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            **kwargs,
        )
        
        return render_colors, render_alphas, info
    
    def train(self, step=0):
        """4D training main loop following deformable 3DGS style"""
        cfg = self.cfg
        max_steps = cfg.max_steps
        init_step = step
        
        print(f"Starting 4D reconstruction training for {max_steps} steps...")
        
        # Learning rate schedulers
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        
        ema_loss_for_log = 0.0
        # viewpoint_stack = None # Wait
        
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
                
        # pbar = tqdm.tqdm(range(1, max_steps + 1))
        pbar = tqdm.tqdm(range(init_step, max_steps))
        
        for iteration in pbar:
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                # Progressive SH training (simplified version)
                pass
            
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
                
            c2w = data["camtoworld"].to(self.device)  # [1, 4, 4]
            Ks = data["K"].to(self.device)  # [1, 3, 3]
            gt_image = data["image"].to(self.device) / 255.0  # [1, H, W, 3]
            image_id = data["image_id"].to(self.device)
            height, width = gt_image.shape[1:3]
            
            fid = torch.tensor([image_id.float() / (self.total_frame - 1)], 
                          dtype=torch.float32, device=self.device)
            
            if iteration < cfg.warm_up:
                d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
                # Use canonical positions during warm-up
                # means, quats, scales = self.splats["means"], self.splats["quats"], self.splats["scales"]
            else:
                # Use deformed positions
                d_xyz, d_rotation, d_scaling = self.get_deformed_points(fid, iteration)

            # Render with 4D deformations (exactly like original 4DGS render function)
            render_colors, render_alphas, info = self.rasterize_4d_splats(
                camtoworlds=c2w,
                Ks=Ks,
                width=width,
                height=height,
                d_xyz=d_xyz,
                d_rotation=d_rotation,
                d_scaling=d_scaling,
                is_6dof=cfg.is_6dof,  
                sh_degree=min(iteration // 1000, cfg.sh_degree),
            )

            # Loss computation (following deformable 3DGS)
            Ll1 = self.l1_loss(render_colors, gt_image)
            
            # SSIM loss
            render_colors_permuted = render_colors.permute(0, 3, 1, 2)
            gt_image_permuted = gt_image.permute(0, 3, 1, 2)
            ssim_loss = 1.0 - self.ssim(render_colors_permuted, gt_image_permuted)
            
            # Combined loss (following deformable 3DGS)
            loss = (1.0 - cfg.lambda_dssim) * Ll1 + cfg.lambda_dssim * ssim_loss
            
            # Densification step pre-backward
            if iteration >= cfg.densify_from_iter:
                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=iteration,
                    info=info,
                )
            
            # Backward pass
            loss.backward()
            
            # Update progress
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                pbar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
            
            # Optimization step
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad()
            
            # Update DeformModel
            if self.deform_model.optimizer is not None:
                self.deform_model.optimizer.step()
                self.deform_model.optimizer.zero_grad()
            
            # Update learning rates
            current_deform_lr = self.deform_model.update_learning_rate(iteration)
            for scheduler in schedulers:
                scheduler.step()
            
            # Post-backward densification
            if iteration >= cfg.densify_from_iter and iteration < cfg.densify_until_iter:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=iteration,
                        info=info,
                        packed=cfg.packed,
                    )
                
                # Densification
                if iteration % cfg.densification_interval == 0:
                    size_threshold = 20 if iteration > cfg.opacity_reset_interval else None
                    # Here you would call densify_and_prune if using DefaultStrategy
                
                # Opacity reset
                if iteration % cfg.opacity_reset_interval == 0:
                    # Reset opacity - this would need to be implemented
                    pass
            
            # Record metrics
            self.metrics_history['loss'].append(loss.item())
            self.metrics_history['l1_loss'].append(Ll1.item())
            self.metrics_history['num_gaussians'].append(len(self.splats['means']))
            
            # Evaluation and saving
            if iteration in cfg.eval_steps:
                self.evaluate(iteration)
            
            if iteration in cfg.save_steps or iteration == max_steps:
                self.save_checkpoint(iteration)
        
        print(f"Best PSNR = {self.best_psnr} in Iteration {self.best_iteration}")

    @torch.no_grad()
    def evaluate(self, iteration: int, stage: str = "val"):
        """4D evaluation with time embedding"""
        print(f"Running 4D evaluation at iteration {iteration}...")
        cfg = self.cfg
        device = self.device
        
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        
        ellipse_time = 0
        metrics = defaultdict(list)
        
        for i, data in enumerate(tqdm.tqdm(valloader)):
            c2w = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            image_id = data["image_id"].to(device)
            height, width = pixels.shape[1:3]
            
            fid = torch.tensor([image_id.float() / max(self.total_frame - 1, 1)], 
                            dtype=torch.float32, device=device)
            
            torch.cuda.synchronize()
            tic = time.time()
            
            canonical_means = self.splats["means"]
            N = canonical_means.shape[0]
            
            time_input = fid.unsqueeze(0).expand(N, -1)
            
            # Add smooth noise
            if not self.cfg.is_blender:
                ast_noise = torch.randn(1, 1, device=self.device).expand(N, -1) * self.time_interval * self.smooth_term(iteration)
                time_input = time_input + ast_noise
            
            d_xyz, d_rotation, d_scaling = self.deform_model.step(canonical_means, time_input)
            render_colors, render_alphas, _ = self.rasterize_4d_splats(
                camtoworlds=c2w,
                Ks=Ks,
                width=width,
                height=height,
                d_xyz=d_xyz,
                d_rotation=d_rotation,
                d_scaling=d_scaling,
                is_6dof=cfg.is_6dof,
                sh_degree=cfg.sh_degree,
            )
            
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic
            
            # 结果处理
            colors = torch.clamp(render_colors, 0.0, 1.0)
            alphas = render_alphas
            
            # 保存图像（和标准3DGS一样）
            if self.world_rank == 0:
                # GT image
                pixels_path = f"{self.render_dir}/{stage}/{iteration}/GT/{i:04d}.png"
                os.makedirs(os.path.dirname(pixels_path), exist_ok=True)
                pixels_canvas = pixels.squeeze(0).cpu().numpy()
                pixels_canvas = (pixels_canvas * 255).astype(np.uint8)
                imageio.imwrite(pixels_path, pixels_canvas)
                
                # Pred image
                colors_path = f"{self.render_dir}/{stage}/{iteration}/Pred/{i:04d}.png"
                os.makedirs(os.path.dirname(colors_path), exist_ok=True)
                colors_canvas = colors.squeeze(0).cpu().numpy()
                colors_canvas = (colors_canvas * 255).astype(np.uint8)
                imageio.imwrite(colors_path, colors_canvas)
                
                # Alpha image 
                alphas_path = f"{self.render_dir}/{stage}/{iteration}/Alpha/{i:04d}.png"
                os.makedirs(os.path.dirname(alphas_path), exist_ok=True)
                alphas_canvas = (alphas < 0.5).squeeze(0).cpu().numpy()
                alphas_canvas = (alphas_canvas * 255).astype(np.uint8)
                Image.fromarray(alphas_canvas.squeeze(), mode='L').save(alphas_path)
                
                # time info
                time_info_path = f"{self.render_dir}/{stage}/{iteration}/time_info.txt"
                with open(time_info_path, "a") as f:
                    f.write(f"Frame {i:04d}: image_id={image_id.item()}, fid={fid.item():.4f}\n")
                
                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
        
        if self.world_rank == 0:
            ellipse_time /= len(valloader)
            
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update({
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
                "total_frames": self.total_frame,  # 4D特有
            })
            
            print(
                f"4D Evaluation Results:"
                f"\nPSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f}"
                f"\nTime: {stats['ellipse_time']:.3f}s/frame"
                f"\nNumber of GS: {stats['num_GS']}, Total Frames: {stats['total_frames']}"
            )
            
            if stats['psnr'] > self.best_psnr:
                self.best_psnr = stats['psnr']
                self.best_iteration = iteration
            
            # 确保 stats_dir 存在
            if not hasattr(self, 'stats_dir'):
                self.stats_dir = f"{cfg.result_dir}/stats"
                os.makedirs(self.stats_dir, exist_ok=True)
            
            with open(f"{self.stats_dir}/{stage}_step{iteration:04d}_4d.json", "w") as f:
                json.dump(stats, f)
            
            if hasattr(self, 'writer') and self.writer is not None:
                for k, v in stats.items():
                    self.writer.add_scalar(f"4d_{stage}/{k}", v, iteration)
                self.writer.flush()
        
    def save_checkpoint(self, iteration: int):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'splats': self.splats.state_dict(),
            'config': self.cfg,
            'metrics_history': dict(self.metrics_history),
            'strategy_state': self.strategy_state,
            'best_psnr': self.best_psnr,
            'best_iteration': self.best_iteration
        }
        
        # Save DeformModel
        self.deform_model.save_weights(self.ckpt_dir, iteration)
        
        # Save Gaussian parameters and training state
        torch.save(checkpoint, f"{self.ckpt_dir}/ckpt_{iteration}.pt")
        
        # Save optimizers
        optimizer_states = {k: v.state_dict() for k, v in self.optimizers.items()}
        torch.save(optimizer_states, f"{self.ckpt_dir}/optimizers_{iteration}.pt")
        
        print(f"Checkpoint saved at iteration {iteration}")


def main():
    """Main function for 4D reconstruction with gsplat"""
    cfg = Config4D(
        data_dir="/share/magic_group/aigc/fcr/difix4d/data/4DNex-10M/dynamic_3_colmap/00018004",
        result_dir="results/dynamic_3_colmap/00018004",
        max_steps=40_000,
        # DeformModel settings
        is_blender=False,
        is_6dof=False,
        position_lr_init=1.6e-4,
        position_lr_final=1.6e-6,
        position_lr_delay_mult=0.01,
        deform_lr_max_steps=40_000,
        # Gsplat settings
        packed=False,
        sparse_grad=False,
        antialiased=False,
        strategy=DefaultStrategy(verbose=True)
    )
    
    print("=" * 60)
    print("4D Gaussian Splatting Reconstruction (Clean Version)")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Data dir: {cfg.data_dir}")
    print(f"  - Training steps: {cfg.max_steps}")
    print(f"  - Warm-up steps: {cfg.warm_up}")
    print(f"  - DeformModel: blender={cfg.is_blender}, 6dof={cfg.is_6dof}")
    print(f"  - GSplat: packed={cfg.packed}, antialiased={cfg.antialiased}")
    print("=" * 60)
    
    # Create and run trainer
    runner = Runner4D(cfg)
    runner.train()
    
    print("=" * 60)
    print("4D Reconstruction Training Completed!")
    print(f"Results saved in: {cfg.result_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()