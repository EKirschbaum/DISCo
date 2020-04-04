
import torch
import torch.nn as nn
import torch.nn.functional as F

import inferno.utils.torch_utils as tu

class GetAffs(nn.Module):
    def __init__(self, offsets,dtype,device):
        super(GetAffs,self).__init__()
        self.offsets = offsets
        self.dtype = dtype
        self.device = device
        
    @property
    def dim(self):
        # infer dim from affinty orders
        return len(self.offsets[0])

    def fill_shift_kernel(self, shift_kernel, offset):
        assert self.dim == 2
        # The kernels have a shape similar to conv kernels in torch.
        # If we have direct nhood (diagonal_affinities = False), we have 3 output channels,
        # corresponding to (depth, height, width)
        # otheriwise (indirect nhood (diagonal_affinities = True), we have 9 output channels)
        shift_kernel = self.aff_shift_kernels_(shift_kernel,
                                            self.dim,
                                            offset)
        return shift_kernel

    
    def segmentation_to_affinity(self, segmentation, offset):
        # This assumes that segmentation is a uni-channel variable
        assert segmentation.size(1) == 1, str(segmentation.size(1))
        assert self.dim == 2
        # Make a kernel variable and convolve
        shift_kernels = self.fill_shift_kernel(segmentation.data.new(1, 1, 3, 3).zero_(),
                                               offset)
        #shift_kernels = Variable(shift_kernels, requires_grad=False)
        abs_offset = tuple(max(1, abs(off)) for off in offset)
        spatial_gradient = F.conv2d(input=segmentation,
                                    weight=shift_kernels,
                                    dilation=abs_offset,
                                    padding=abs_offset)
        # Binarize affinities
        binarized_affinities = \
            tu.where((spatial_gradient == 0).data,
                     spatial_gradient.data.new(*spatial_gradient.size()).fill_(1.),
                     spatial_gradient.data.new(*spatial_gradient.size()).fill_(0.))
        return(binarized_affinities)

    def aff_shift_kernels_(self, kernel, dim, offset):
        if dim == 3:
            assert len(offset) == 3
            kernel[0, 0, 1, 1, 1] = -1.
            s_z = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            s_x = 1 if offset[2] == 0 else (2 if offset[2] > 0 else 0)
            kernel[0, 0, s_z, s_y, s_x] = 1.
        elif dim == 2:
            assert len(offset) == 2
            kernel[0, 0, 1, 1] = -1.
            s_x = 1 if offset[0] == 0 else (2 if offset[0] > 0 else 0)
            s_y = 1 if offset[1] == 0 else (2 if offset[1] > 0 else 0)
            kernel[0, 0, s_x, s_y] = 1.
        else:
            raise NotImplementedError
        return kernel
    
    def get_affs(self,segmentation):
        affinities = []
        for offset in self.offsets:
            affinity = self.segmentation_to_affinity(segmentation,offset)
            affinities.append(affinity)
        return(torch.stack(affinities,dim=0))
        
