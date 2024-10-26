import os
import torch
import pickle
import numpy as np
import os.path as osp
import torch.nn as nn
from dataclasses import dataclass
from collections import namedtuple
from typing import Optional, Dict

import smplx
from smplx.lbs import vertices2joints, batch_rodrigues, blend_shapes
from smplx.utils import Struct, to_tensor, to_np, ModelOutput, Tensor, Array
from smplx.vertex_ids import vertex_ids

from .lbs import lbs_extra, lbs

TIP_IDS = {
    'thumb': 744,
    'index': 320,
    'middle': 443,
    'ring': 554,
    'pinky': 671,
}

@dataclass
class MANOOutput(ModelOutput):
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    betas: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    hand_pose: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    shape_offsets: Optional[Tensor] = None
    pose_offsets: Optional[Tensor] = None
    T: Optional[Tensor] = None
    A: Optional[Tensor] = None

@dataclass
class DualMANOOutput(ModelOutput):
    vertices: Optional[Dict[str, Tensor]] = None  
    joints: Optional[Dict[str, Tensor]] = None
    betas: Optional[Tensor] = None
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None
    left_global_orient: Optional[Tensor] = None 
    right_global_orient: Optional[Tensor] = None
    full_pose: Optional[Dict[str, Tensor]] = None
    lbs_weights: Optional[Dict[str, Tensor]] = None
    v_template: Optional[Dict[str, Tensor]] = None

    def __getitem__(self, key: str):
        if key not in ['left_hand', 'right_hand']:
            raise KeyError(f"Key {key} not found. Use 'left_hand' or 'right_hand'")
            
        side = key.split('_')[0]
        return MANOOutput(
            vertices=self.vertices[side] if self.vertices is not None else None,
            joints=self.joints[side] if self.joints is not None else None, 
            betas=self.betas,
            global_orient=getattr(self, f"{side}_global_orient"),
            hand_pose=getattr(self, f"{side}_hand_pose"),
            full_pose=self.full_pose[side] if self.full_pose is not None else None,
            lbs_weights=self.lbs_weights[side] if self.lbs_weights is not None else None,
            v_template=self.v_template[side] if self.v_template is not None else None
        )

class MANO(nn.Module):
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS
    NUM_BETAS = 10

    def __init__(
        self,
        model_path: str,
        is_rhand: bool = True,
        data_struct: Optional[Struct] = None,
        create_betas: bool = True,
        betas: Optional[Tensor] = None,
        create_global_orient: bool = True,
        global_orient: Optional[Tensor] = None,
        create_transl: bool = True,
        transl: Optional[Tensor] = None,
        create_hand_pose: bool = True,
        hand_pose: Optional[Tensor] = None,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        batch_size: int = 1,
        joint_mapper=None,
        v_template: Optional[Tensor] = None,
        dtype=torch.float32,
        **kwargs
    ) -> None:
        super(MANO, self).__init__()

        if data_struct is None:
            if osp.isdir(model_path):
                model_fn = f'MANO_{"RIGHT" if is_rhand else "LEFT"}.pkl'
                mano_path = osp.join(model_path, model_fn)
            else:
                mano_path = model_path
                self.is_rhand = 'RIGHT' in os.path.basename(model_path)
            
            assert osp.exists(mano_path), f'Path {mano_path} does not exist!'
            with open(mano_path, 'rb') as f:
                data_struct = Struct(**pickle.load(f, encoding='latin1'))

        self.dtype = dtype
        self.joint_mapper = joint_mapper
        self.batch_size = batch_size
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        self.flat_hand_mean = flat_hand_mean

        self._register_parameters(
            data_struct=data_struct,
            create_betas=create_betas,
            betas=betas,
            create_global_orient=create_global_orient,
            global_orient=global_orient,
            create_transl=create_transl,
            transl=transl,
            create_hand_pose=create_hand_pose,
            hand_pose=hand_pose,
            v_template=v_template,
            **kwargs
        )

        if self.use_pca:
            hand_components = data_struct.hands_components[:num_pca_comps]
            self.register_buffer('hand_components', to_tensor(hand_components, dtype=dtype))

        hand_mean = np.zeros_like(data_struct.hands_mean) if flat_hand_mean else data_struct.hands_mean
        self.register_buffer('hand_mean', to_tensor(hand_mean, dtype=dtype))
        
        self.register_buffer('pose_mean', self.create_mean_pose(data_struct))

    def _register_parameters(self, data_struct, **kwargs):
        self.faces = data_struct.f
        self.register_buffer('faces_tensor', to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long))

        if kwargs.get('create_betas', True):
            default_betas = torch.zeros([self.batch_size, self.NUM_BETAS], dtype=self.dtype)
            self.register_parameter('betas', nn.Parameter(default_betas, requires_grad=True))

        if kwargs.get('create_global_orient', True):
            default_orient = torch.zeros([self.batch_size, 3], dtype=self.dtype)
            self.register_parameter('global_orient', nn.Parameter(default_orient, requires_grad=True))

        if kwargs.get('create_transl', True):
            default_transl = torch.zeros([self.batch_size, 3], dtype=self.dtype)
            self.register_parameter('transl', nn.Parameter(default_transl, requires_grad=True))

        if kwargs.get('create_hand_pose', True):
            hand_pose_dim = self.num_pca_comps if self.use_pca else 3 * self.NUM_HAND_JOINTS
            default_hand_pose = torch.zeros([self.batch_size, hand_pose_dim], dtype=self.dtype)
            self.register_parameter('hand_pose', nn.Parameter(default_hand_pose, requires_grad=True))

        v_template = kwargs.get('v_template', data_struct.v_template)
        self.register_buffer('v_template', to_tensor(to_np(v_template), dtype=self.dtype))
        
        shapedirs = data_struct.shapedirs
        self.register_buffer('shapedirs', to_tensor(to_np(shapedirs), dtype=self.dtype))

        self.register_buffer('J_regressor', to_tensor(to_np(data_struct.J_regressor), dtype=self.dtype))

        posedirs = np.reshape(data_struct.posedirs, [-1, data_struct.posedirs.shape[-1]]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))

        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        self.register_buffer('lbs_weights', to_tensor(to_np(data_struct.weights), dtype=self.dtype))

    def create_mean_pose(self, data_struct):
        return torch.cat([
            torch.zeros([1, 3], dtype=self.dtype),
            self.hand_mean.reshape(1, -1)
        ], dim=1)

    def forward(
        self,
        betas: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        hand_pose: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> MANOOutput:
        betas = betas if betas is not None else self.betas
        global_orient = global_orient if global_orient is not None else self.global_orient
        hand_pose = hand_pose if hand_pose is not None else self.hand_pose
        apply_trans = transl is not None or hasattr(self, 'transl')
        
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        if self.use_pca:
            hand_pose = torch.einsum('bi,ij->bj', [hand_pose, self.hand_components])

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        full_pose += self.pose_mean

        if return_verts:
            vertices, joints = lbs(
                betas, full_pose, self.v_template,
                self.shapedirs, self.posedirs,
                self.J_regressor, self.parents,
                self.lbs_weights, dtype=self.dtype
            )

            if self.joint_mapper is not None:
                joints = self.joint_mapper(joints)

            if apply_trans:
                joints = joints + transl.unsqueeze(dim=1)
                vertices = vertices + transl.unsqueeze(dim=1)

        output = MANOOutput(
            vertices=vertices if return_verts else None,
            joints=joints if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=full_pose if return_full_pose else None
        )

        return output

class MANOLayer(smplx.MANOLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        super(MANOLayer, self).__init__(*args, **kwargs)
        self.register_buffer('joint_map', torch.tensor([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], dtype=torch.long))
        
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', 
                torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('extra_joints_idxs', to_tensor(list(vertex_ids['mano'].values()), dtype=torch.long))

    def forward(self, *args, **kwargs) -> MANOOutput:
        mano_output = super().forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
            
        mano_output.joints = joints
        return mano_output

class DualMANOLayer(nn.Module):
    def __init__(
        self,
        model_path: str,
        use_pca: bool = True,
        num_pca_comps: int = 6,
        flat_hand_mean: bool = False,
        create_betas: bool = True,
        **kwargs
    ):
        super(DualMANOLayer, self).__init__()

        self.left_hand = MANOLayer(
            model_path=model_path,
            is_rhand=False,
            use_pca=use_pca,
            num_pca_comps=num_pca_comps,
            flat_hand_mean=flat_hand_mean,
            create_betas=create_betas,
            **kwargs
        )
        
        self.right_hand = MANOLayer(
            model_path=model_path,
            is_rhand=True,
            use_pca=use_pca,
            num_pca_comps=num_pca_comps,
            flat_hand_mean=flat_hand_mean,
            create_betas=create_betas,
            **kwargs
        )

    def forward(
        self,
        betas: Optional[torch.Tensor] = None,
        left_hand_pose: Optional[torch.Tensor] = None,
        right_hand_pose: Optional[torch.Tensor] = None,
        left_global_orient: Optional[torch.Tensor] = None,
        right_global_orient: Optional[torch.Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> DualMANOOutput:
        left_output = self.left_hand(
            betas=betas,
            global_orient=left_global_orient,
            hand_pose=left_hand_pose,
            return_verts=return_verts,
            return_full_pose=return_full_pose,
        )

        right_output = self.right_hand(
            betas=betas,
            global_orient=right_global_orient,
            hand_pose=right_hand_pose,
            return_verts=return_verts,
            return_full_pose=return_full_pose,
        )

        return DualMANOOutput(
            vertices={'left': left_output.vertices, 'right': right_output.vertices} if return_verts else None,
            joints={'left': left_output.joints, 'right': right_output.joints} if return_verts else None,
            betas=betas,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            left_global_orient=left_global_orient,
            right_global_orient=right_global_orient,
            full_pose={'left': left_output.full_pose, 'right': right_output.full_pose} if return_full_pose else None,
            lbs_weights={'left': self.left_hand.lbs_weights, 'right': self.right_hand.lbs_weights},
            v_template={'left': self.left_hand.v_template, 'right': self.right_hand.v_template}
        )