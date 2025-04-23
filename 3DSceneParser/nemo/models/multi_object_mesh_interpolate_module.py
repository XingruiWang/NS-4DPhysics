import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
import pytorch3d.renderer
from pytorch3d.structures.utils import list_to_padded, padded_to_list
from nemo.utils import (
    pre_process_mesh_pascal,
    vertex_memory_to_face_memory,
    rotation_theta,
)



class MultiMeshInterpolateModule(nn.Module):
    # For multiple objects in one scene
    def __init__(
        self,
        mesh_loader,
        memory_bank,
        rasterizer,
        post_process=None,
        off_set_mesh=False,
        **kwargs
    ):  
        super().__init__()
        # Verts must be list of tensors
        verts, faces = mesh_loader.get_mesh_para(to_torch=True)
        assert isinstance(verts, list) and isinstance(faces, list)

        # Convert memory features of vertices to faces
        self.verts = verts
        self.faces = faces
        self.face_memory = None
        self.n_mesh = len(verts)

        self.reset()
        self.update_memory(memory_bank=memory_bank, faces=faces)
        
        self.rasterizer = rasterizer
        self.post_process = post_process

    def reset(self):
        self.verts_registered = None
        self.faces_registered = None
        self.face_mem_registered = None
        self.meshes_registered = None

    def update_memory(self, memory_bank, faces=None):
        if faces is None:
            faces = self.faces
        if torch.is_tensor(memory_bank):
            n_vert_start = torch.cumsum(torch.Tensor([0] + [t.shape[0] for t in self.verts]).type(torch.int64), dim=0)

            # (n_verts_total, c)
            memory_bank = [memory_bank[n_vert_start[ii]:n_vert_start[ii + 1]] for ii in range(self.n_mesh)]
        # Convert memory features of vertices to faces
        self.face_memory = [vertex_memory_to_face_memory(m, f).to(m.device) for m, f in zip(memory_bank, faces)]

    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super().to(device)
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        self.face_memory = [t.to(device) for t in self.face_memory]
        self.verts = [t.to(device) for t in self.verts]
        self.faces = [t.to(device) for t in self.faces]
        return self

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def register_verts(self, indexs, transforms=None):
        # For place holder, call this function with transforms=None, then register_faces to create Meshes to reduce computation overhead

        all_verts = []
        if transforms is None:
            for index_ in indexs:
                n_obj = len(index_)
                # rewrite
                to_cat = []
                for i in range(n_obj):
                    if index_[i] != -1:
                        to_cat.append(self.verts[index_[i]])
                all_verts.append(torch.cat(to_cat, dim=0))
                # all_verts.append(torch.cat([self.verts[index_[i]] for i in range(n_obj) if index_[i] != -1], dim=0))
        else:
            # index need to be list of tensor [k, ] * n
            for transform_, index_ in zip(transforms, indexs):
                
                n_obj = len(transform_)
                vert_list_this = [self.verts[index_[i]] for i in range(n_obj)]
                vert_padded_this = list_to_padded(vert_list_this)
                
                split_size=[]
                for t in vert_list_this:
                    split_size.append(t.shape[0])

                vert_get = padded_to_list(transform_.transform_points(vert_padded_this), split_size=split_size)
                if (index_ == -1).sum() > 0:
                    vert_get = vert_get[:-(index_ == -1).sum()]
                
                all_verts.append(torch.cat(vert_get, dim=0))
                # all_verts.append(torch.cat(vert_list_this, dim=0))

        self.verts_registered = all_verts
    
    @torch.no_grad()
    def register_faces(self, indexs,):
        all_faces = []
        all_face_mems = []

        for index_ in indexs:
            n_obj = len(index_)
            idx_shift = torch.cumsum(torch.Tensor([0] + [self.verts[index_[i]].shape[0] for i in range(n_obj - 1) if index_[i] != -1]), dim=0).to(self.faces[0].device)

            all_faces.append(torch.cat([self.faces[index_[i]] + idx_shift[i] for i in range(n_obj) if index_[i] != -1], dim=0))
            all_face_mems.append(torch.cat([self.face_memory[index_[i]] for i in range(n_obj) if index_[i] != -1], dim=0))

        self.faces_registered = all_faces
        self.face_mem_registered = torch.cat(all_face_mems, dim=0)

    def forward(self, indexs=None, transforms=None, **kwargs):
        # 梯度没了？？？？？？？
        if transforms is not None:
            assert indexs is not None
            self.register_verts(indexs, transforms)

        if indexs is not None and self.faces_registered is None:
            self.register_faces(indexs)
        # print("Shape of verts and faces", self.verts_registered[0].shape, self.faces_registered[0].shape)
        verts_registered = self.verts_registered
        meshes = Meshes(verts=verts_registered, faces=self.faces_registered)
        meshes = meshes.update_padded(list_to_padded(self.verts_registered))
        # meshes = meshes.update_padded(BackwardHook.apply(list_to_padded(self.verts_registered)))
        fragments = self.rasterizer(meshes, **kwargs)

        out_map = pytorch3d.renderer.mesh.utils.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, self.face_mem_registered
        ).squeeze(3)
        

        if self.post_process is not None:
            out_map = self.post_process(out_map)
        return out_map
    

class BackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x
        ):  
            return x
    
    @staticmethod
    def backward(ctx, grad_x):
        print(grad_x.max().item(), grad_x.min().item())
        return grad_x
