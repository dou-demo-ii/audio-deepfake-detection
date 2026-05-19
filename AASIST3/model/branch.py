import torch.nn as nn

from .hs_gal import HtrgGraphAttentionLayer
from .pool import GraphPool

class InferenceBranch(nn.Module):
    def __init__(self, gat_dims, temperature, pool_ratio, size):
        super().__init__()
        self.htrg_gat1 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperature, size=size
        )
        self.htrg_gat2 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperature, size=size
        )

        self.pool_hS = GraphPool(pool_ratio, gat_dims[1], 0.3, size=size)
        self.pool_hT = GraphPool(pool_ratio, gat_dims[1], 0.3, size=size)

    def forward(self, out_T, out_S, master):
        # Первая стадия
        out_T_res, out_S_res, master_res = self.htrg_gat1(out_T, out_S, master=master)

        # Пулинг
        out_S_res = self.pool_hS(out_S_res)
        out_T_res = self.pool_hT(out_T_res)

        # Вторая стадия с residual connection
        out_T_aug, out_S_aug, master_aug = self.htrg_gat2(out_T_res, out_S_res, master=master_res)
        
        out_T_final = out_T_res + out_T_aug
        out_S_final = out_S_res + out_S_aug
        master_final = master_res + master_aug

        return out_T_final, out_S_final, master_final