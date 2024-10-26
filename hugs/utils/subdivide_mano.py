# Update
# Modified from HuGS, Apple.Inc 2024

import torch 
import trimesh 
import numpy as np 
from trimesh import grouping 
from trimesh.geometry import faces_to_edges  


from hugs.models.modules.mano_layer import MANO


def subdivide(
    vertices,
    faces,              
    face_index=None,               
    vertex_attributes=None
):     
    
    if face_index is None:         
        face_mask = np.ones(len(faces), dtype=bool)     
    else:         
        face_mask = np.zeros(len(faces), dtype=bool)         
        face_mask[face_index] = True      
        
    # the (c, 3) int array of vertex indices     
    faces_subset = faces[face_mask]      
    # find the unique edges of our faces subset     
    edges = np.sort(faces_to_edges(faces_subset), axis=1)     
    unique, inverse = grouping.unique_rows(edges)     
    # then only produce one midpoint per unique edge     
    mid = vertices[edges[unique]].mean(axis=1)     
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)      
    # the new faces_subset with correct winding     
    f = np.column_stack(
        [         
            faces_subset[:, 0],         
            mid_idx[:, 0],         
            mid_idx[:, 2],        
            mid_idx[:, 0],         
            faces_subset[:, 1],         
            mid_idx[:, 1],         
            mid_idx[:, 2],         
            mid_idx[:, 1],         
            faces_subset[:, 2],         
            mid_idx[:, 0],         
            mid_idx[:, 1],         
            mid_idx[:, 2]]     
    ).reshape((-1, 3))      
    
    # add the 3 new faces_subset per old face all on the end     
    # # by putting all the new faces after all the old faces     
    # # it makes it easier to understand the indexes     
    new_faces = np.vstack((faces[~face_mask], f))     
    # stack the new midpoint vertices on the end     
    new_vertices = np.vstack((vertices, mid))    
        
    if vertex_attributes is not None:       
        new_attributes = {}       
        for key, values in vertex_attributes.items():
            if key == 'v_id':             
                attr_mid = values[edges[unique][:, 0]]         
            if key == 'lbs_weights':             
                attr_mid = values[edges[unique]].mean(axis=1)         
            else:             
                attr_mid = values[edges[unique]].mean(axis=1)                      
            new_attributes[key] = np.vstack((values, attr_mid))     
    return new_vertices, new_faces, new_attributes



def _subdivide_mano_model(mano=None, smoothing=False): 
    if mano is None:         
        mano = MANO("data/mano")     
             
    n_verts = mano.v_template.shape[0]     
    init_posedirs = mano.posedirs.detach().cpu().numpy()     
    init_lbs_weights = mano.lbs_weights.detach().cpu().numpy()     
    init_shapedirs = mano.shapedirs.detach().cpu().numpy()     
    init_v_id = mano.v_id      
    init_shapedirs = init_shapedirs.reshape(n_verts, -1)     
    init_J_regressor = mano.J_regressor.detach().cpu().numpy().transpose(1, 0)          
    init_vertices = mano.v_template.detach().cpu().numpy()     
    init_faces = mano.faces      
    print("# vertices before subdivision:", init_vertices.shape)          
    sub_vertices, sub_faces, attr = subdivide(
        vertices=init_vertices,         
        faces=init_faces,         
        vertex_attributes={
            "v_id": init_v_id,             
            "lbs_weights": init_lbs_weights,            
            "shapedirs": init_shapedirs,             
            "J_regressor": init_J_regressor,         
        }     
    )
    
    if smoothing:
        sub_mesh = trimesh.Trimesh(vertices=sub_vertices, faces=sub_faces)         
        sub_mesh = trimesh.smoothing.filter_mut_dif_laplacian(
            sub_mesh, 
            lamb=0.5,
            iterations=5,
            volume_constraint=True,
            laplacian_operator=None
        )        
        sub_vertices = sub_mesh.vertices
                       
    new_mano = MANO("data/mano")     
    new_mano.lbs_weights = torch.from_numpy(attr["lbs_weights"]).float()
    posedirs = np.zeros((207, sub_vertices.shape[0] * 3)).astype(np.float32)
    shapedirs = attr["shapedirs"].reshape(-1, 3, 10)
    J_regressor = np.zeros_like(attr["J_regressor"].transpose(1, 0))
    J_regressor[:, :n_verts] = mano.J_regressor
    new_mano.posedirs = torch.from_numpy(posedirs).float()
    new_mano.shapedirs = torch.from_numpy(shapedirs).float()
    new_mano.v_template = torch.from_numpy(sub_vertices).float()
    new_mano.faces_tensor = torch.from_numpy(sub_faces).long()
    new_mano.J_regressor = torch.from_numpy(J_regressor).float()
    new_mano.faces = sub_faces     
    new_mano.v_id = attr["v_id"].astype(int)
    return new_mano


    def subdivide_mano_model(mano=None, smoothing=False, n_iter=1):     
    if mano is None:         
        from hugs.cfg.constants import MANO_PATH         
        mano = MANO(MANO_PATH)          
    # 778 or 6890?
    mano.v_id = np.arange(778)[..., None]     
    for _ in range(n_iter):         
        mano = _subdivide_mano_model(mano, smoothing)     
    return mano