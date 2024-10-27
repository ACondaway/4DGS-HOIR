import torch
from lpips import LPIPS
import torch.nn as nn
import torch.nn.functional as F

from hugs.utils.sampler import PatchSampler # 用于将图片分成多个patch，然后计算每个patch的损失

from .utils import l1_loss, ssim

from torchvision import models

class HandObjectInteractionLoss(nn.Module):
    def __init__(
        self,
        l_mask_w=0.1,
        l_ssim_w=0.2,
        l_vgg_w=0.1,
        l_lpips_w=0.1,
        l_lbs_w=0.1,
        l_contact_w=0.1,
        l_penetration_w=0.1,
        num_patches=4,
        patch_size=32,
        use_patches=True,
        bg_color='white',
        tau=0.005,  # 接触阈值
        eta=0.01    # 穿透阈值
    ):
        super(HandObjectInteractionLoss, self).__init__()
        
        # 损失权重
        self.l_mask_w = l_mask_w
        self.l_ssim_w = l_ssim_w
        self.l_vgg_w = l_vgg_w
        self.l_lpips_w = l_lpips_w
        self.l_lbs_w = l_lbs_w
        self.l_contact_w = l_contact_w
        self.l_penetration_w = l_penetration_w
        
        self.use_patches = use_patches
        self.bg_color = bg_color
        self.tau = tau
        self.eta = eta
        
        # 初始化 LPIPS 感知损失
        self.lpips = LPIPS(net="vgg").to('cuda')
        for param in self.lpips.parameters():
            param.requires_grad = False
        
        # 初始化 VGG 感知损失
        self.vgg_loss_fn = VGGPerceptualLoss().to('cuda')
        
        # 初始化 PatchSampler
        if self.use_patches:
            self.patch_sampler = PatchSampler(num_patches=num_patches, patch_size=patch_size)
    
    def forward(self, data, render_pkg, gs_out):
        """
        计算单帧的损失，包括掩码损失、SSIM、VGG、LPIPS、LBS、接触损失和穿透损失。
        
        参数：
            data: 包含真实图像和掩码的字典
            render_pkg: 模型渲染的输出，包括预测图像
            gs_out: 模型的高斯球体输出，包含 LBS 权重和手部顶点位置等信息
        返回：
            total_loss: 总损失值
            loss_dict: 各项损失组成的字典
        """
        loss_dict = {}
        
        # 获取真实图像和预测图像
        gt_image = data['rgb']           # 真实图像，形状：[3, H, W]
        pred_img = render_pkg['render']  # 预测图像，形状：[3, H, W]
        
        # 获取左手、右手和物体的掩码
        mask_lhand = data['mask_lhand'].unsqueeze(0)  # [1, H, W]
        mask_rhand = data['mask_rhand'].unsqueeze(0)  # [1, H, W]
        mask_obj = data['mask_obj'].unsqueeze(0)      # [1, H, W]
        
        # 计算掩码区域的 L1 损失
        loss_l1_lhand = l1_loss(pred_img, gt_image, mask_lhand) * self.l_mask_w
        loss_l1_rhand = l1_loss(pred_img, gt_image, mask_rhand) * self.l_mask_w
        loss_l1_obj = l1_loss(pred_img, gt_image, mask_obj) * self.l_mask_w
        loss_dict['mask_lhand'] = loss_l1_lhand
        loss_dict['mask_rhand'] = loss_l1_rhand
        loss_dict['mask_obj'] = loss_l1_obj
        
        # 计算 SSIM 损失
        loss_ssim = (1.0 - ssim(pred_img, gt_image)) * self.l_ssim_w
        loss_dict['ssim'] = loss_ssim
        
        # 计算 VGG 感知损失
        loss_vgg = self.vgg_loss_fn(pred_img.unsqueeze(0), gt_image.unsqueeze(0)) * self.l_vgg_w
        loss_dict['vgg'] = loss_vgg
        
        # 计算 LPIPS 感知损失
        if self.l_lpips_w > 0.0:
            if self.use_patches:
                # 合并掩码，确保只在感兴趣区域计算
                combined_mask = torch.clamp(mask_lhand + mask_rhand + mask_obj, 0, 1)
                _, pred_patches, gt_patches = self.patch_sampler.sample(combined_mask, pred_img, gt_image)
                loss_lpips = self.lpips(pred_patches, gt_patches).mean() * self.l_lpips_w
            else:
                loss_lpips = self.lpips(pred_img.unsqueeze(0), gt_image.unsqueeze(0)).mean() * self.l_lpips_w
            loss_dict['lpips'] = loss_lpips
        
        # LBS 损失
        if 'lbs_weights_lhand' in gs_out and 'gt_lbs_weights_lhand' in gs_out:
            loss_lbs_lhand = F.mse_loss(
                gs_out['lbs_weights_lhand'], gs_out['gt_lbs_weights_lhand']
            ) * self.l_lbs_w
            loss_dict['lbs_lhand'] = loss_lbs_lhand
        if 'lbs_weights_rhand' in gs_out and 'gt_lbs_weights_rhand' in gs_out:
            loss_lbs_rhand = F.mse_loss(
                gs_out['lbs_weights_rhand'], gs_out['gt_lbs_weights_rhand']
            ) * self.l_lbs_w
            loss_dict['lbs_rhand'] = loss_lbs_rhand
        
        # 接触损失
        if self.l_contact_w > 0.0:
            if 'hand_joints' in gs_out and 'object_surface' in data:
                contact_loss = self.compute_contact_loss(gs_out['hand_joints'], data['object_surface'])
                loss_dict['contact'] = self.l_contact_w * contact_loss
        
        # 穿透损失
        if self.l_penetration_w > 0.0:
            if 'hand_joints' in gs_out and 'object_surface' in data:
                penetration_loss = self.compute_penetration_loss(gs_out['hand_joints'], data['object_surface'])
                loss_dict['penetration'] = self.l_penetration_w * penetration_loss
        
        # 总损失
        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict
    
    def compute_contact_loss(self, hand_joints, object_surface):
        """
        计算接触损失。
        参数:
            hand_joints: 手部关节位置，形状 [N, 3]
            object_surface: ObjectSurface 实例，提供 nearest_distance 方法
        返回:
            接触损失
        """
        # 计算每个手部关节到物体表面的最近距离
        distances = object_surface.nearest_distance(hand_joints)  # [N]
        
        # 创建掩码，标记距离小于 tau 的关节
        contact_mask = (distances < self.tau).float()  # [N]
        
        # 防止除以零
        epsilon = 1e-6
        
        # 计算分子和分母
        numerator = torch.sum(distances * contact_mask)
        denominator = torch.sum(contact_mask) + epsilon
        
        # 计算接触损失
        contact_loss = numerator / denominator
        return contact_loss
    
    def compute_penetration_loss(self, hand_joints, object_surface):
        """
        计算穿透损失。
        参数:
            hand_joints: 手部关节位置，形状 [N, 3]
            object_surface: ObjectSurface 实例，提供 signed_distance 和 get_normal 方法
        返回:
            穿透损失
        """
        # 获取手部关节的符号距离
        sdf_values = object_surface.signed_distance(hand_joints)  # [N]
        
        # 获取物体表面的法向量
        normals = object_surface.get_normal(hand_joints)  # [N, 3]
        
        # 计算手-物体偏移向量
        if hasattr(object_surface, 'vertices'):
            # 假设手关节对应的最近物体表面点为物体表面的最近点
            # 需要实现准确的最近点映射
            # 这里简单示例，将手关节指向物体表面的法向量方向
            # 实际应用中需要根据具体物体表面实现准确的偏移向量
            offset_vectors = normals  # [N, 3]
        else:
            raise NotImplementedError("ObjectSurface must have 'vertices' attribute for penetration loss.")
        
        # 计算点积
        dot_products = torch.sum(offset_vectors * normals, dim=1)  # [N]
        
        # 计算穿透损失
        penetration_terms = F.relu(-dot_products + self.eta)  # [N]
        penetration_loss = torch.sum(penetration_terms)
        return penetration_loss

class GlobalLoss(nn.Module):
    def __init__(
        self,
        l1_weight=0.1,
        l_ssim_w=0.2,
        l_vgg_w=0.1,
        l_lpips_w=0.1,
        num_patches=4,
        patch_size=32,
        use_patches=True,
        bg_color='white'
    ):
        super(GlobalLoss, self).__init__()
        
        # 损失权重
        self.l1_weight = l1_weight
        self.l_ssim_w = l_ssim_w
        self.l_vgg_w = l_vgg_w
        self.l_lpips_w = l_lpips_w
        
        self.use_patches = use_patches
        self.bg_color = bg_color
        
        # 初始化 LPIPS 感知损失
        self.lpips = LPIPS(net="vgg").to('cuda')
        for param in self.lpips.parameters():
            param.requires_grad = False
        
        # 初始化 VGG 感知损失
        self.vgg_loss_fn = VGGPerceptualLoss().to('cuda')
        
        # 初始化 PatchSampler
        if self.use_patches:
            self.patch_sampler = PatchSampler(num_patches=num_patches, patch_size=patch_size)
    
    def forward(self, data, render_pkg):
        """
        计算全局的损失，包括全局 L1、SSIM、VGG 和 LPIPS 感知损失。
        
        参数：
            data: 包含真实图像的字典
            render_pkg: 模型渲染的输出，包括预测图像
        返回：
            total_loss: 总损失值
            loss_dict: 各项损失组成的字典
        """
        loss_dict = {}
        
        # 获取真实图像和预测图像
        gt_image = data['rgb']           # 真实图像，形状：[3, H, W]
        pred_img = render_pkg['render']  # 预测图像，形状：[3, H, W]
        
        # 计算全局 L1 损失
        loss_l1 = l1_loss(pred_img, gt_image) * self.l1_weight
        loss_dict['l1'] = loss_l1
        
        # 计算全局 SSIM 损失
        loss_ssim = (1.0 - ssim(pred_img, gt_image)) * self.l_ssim_w
        loss_dict['ssim'] = loss_ssim
        
        # 计算全局 VGG 感知损失
        loss_vgg = self.vgg_loss_fn(pred_img.unsqueeze(0), gt_image.unsqueeze(0)) * self.l_vgg_w
        loss_dict['vgg'] = loss_vgg
        
        # 计算全局 LPIPS 感知损失
        if self.l_lpips_w > 0.0:
            if self.use_patches:
                # 假设全局损失不需要掩码，直接采样
                _, pred_patches, gt_patches = self.patch_sampler.sample(None, pred_img, gt_image)
                loss_lpips = self.lpips(pred_patches, gt_patches).mean() * self.l_lpips_w
            else:
                loss_lpips = self.lpips(pred_img.unsqueeze(0), gt_image.unsqueeze(0)).mean() * self.l_lpips_w
            loss_dict['lpips'] = loss_lpips
        
        # 总损失
        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # 使用预训练的 VGG19 网络
        vgg = models.vgg19(pretrained=True).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to('cuda')

        # VGG 期望输入图像进行均值和标准差规范化
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
    
    def forward(self, pred, gt):
        """
        计算 VGG 感知损失。
        参数:
            pred: 预测图像，形状 [N, 3, H, W]
            gt: 真实图像，形状 [N, 3, H, W]
        返回:
            感知损失值
        """
        pred = (pred + 1) / 2  # 假设输入在 [-1, 1] 范围内
        gt = (gt + 1) / 2
        pred = (pred - self.mean) / self.std
        gt = (gt - self.mean) / self.std

        pred_features = self.vgg(pred)
        gt_features = self.vgg(gt)
        loss = F.l1_loss(pred_features, gt_features)
        return loss

class DynamicDeformationLoss(nn.Module):
    def __init__(self, weight=0.1):
        """
        初始化动力学损失。
        参数:
            weight: 损失权重
        """
        super(DynamicDeformationLoss, self).__init__()
        self.weight = weight

    def forward(self, hand_poses):
        """
        计算动力学损失，基于手部姿态的二阶差分。
        参数:
            hand_poses: 手部姿态序列，形状 [batch_size, sequence_length, pose_dim]
                        sequence_length 应该为至少5，以计算二阶差分
        返回:
            加权后的动力学损失
        """
        # 确保sequence_length >= 5
        batch_size, seq_len, pose_dim = hand_poses.shape
        if seq_len < 5:
            raise ValueError("hand_poses的序列长度必须至少为5帧。")

        # 使用滑动窗口，每5帧计算一次二阶差分
        dyn_def_loss = 0.0
        count = 0
        epsilon = 1e-8  # 防止除零

        for i in range(2, seq_len - 2):
            # 前两帧和后两帧，用于计算二阶差分
            prev_pose = hand_poses[:, i - 2, :]  # [batch_size, pose_dim]
            before_prev_pose = hand_poses[:, i - 1, :]
            curr_pose = hand_poses[:, i, :]
            next_pose = hand_poses[:, i + 1, :]
            after_next_pose = hand_poses[:, i + 2, :]

            second_order_diff = after_next_pose - 2 * next_pose + curr_pose  # [batch_size, pose_dim]

            # 计算二阶差分的平方和
            dyn_def_loss += torch.mean(second_order_diff ** 2)
            count += 1

        # 平均动力学损失
        dyn_def_loss = dyn_def_loss / (count + epsilon)

        # 加权
        return self.weight * dyn_def_loss
    
class ObjectSurface:
    def __init__(self, vertices, normals=None):
        """
        初始化物体表面。
        参数:
            vertices: 物体表面的点云，形状 [M, 3]
            normals: 物体表面点的法向量，形状 [M, 3]，可选
        """
        self.vertices = vertices  # [M, 3]
        self.normals = normals    # [M, 3] or None

    def nearest_distance(self, points):
        """
        计算每个点到物体表面的最近距离。
        参数:
            points: 查询点，形状 [N, 3]
        返回:
            distances: 最近距离，形状 [N]
        """
        with torch.no_grad():
            distances = torch.cdist(points, self.vertices)  # [N, M]
            min_distances, _ = distances.min(dim=1)         # [N]
        return min_distances

    def signed_distance(self, points):
        """
        计算每个点的符号距离（SDF）。
        正值表示在物体外部，负值表示在物体内部。
        参数:
            points: 查询点，形状 [N, 3]
        返回:
            sdf_values: 符号距离，形状 [N]
        """
        # 简单示例：假设物体为球体，中心为0，半径为1
        # 实际应用中，需要使用精确的SDF计算方法
        center = torch.zeros(1, 3).to(points.device)
        radius = 1.0
        distances = torch.norm(points - center, dim=1) - radius  # [N]
        return distances

    def get_normal(self, points):
        """
        获取每个查询点最近物体表面点的法向量。
        参数:
            points: 查询点，形状 [N, 3]
        返回:
            normals: 法向量，形状 [N, 3]
        """
        if self.normals is None:
            raise NotImplementedError("Normals are not provided.")
        
        with torch.no_grad():
            distances = torch.cdist(points, self.vertices)  # [N, M]
            min_indices = distances.argmin(dim=1)           # [N]
            normals = self.normals[min_indices]             # [N, 3]
        return normals
