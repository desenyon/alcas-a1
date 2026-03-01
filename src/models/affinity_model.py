"""
ALCAS - Geometric GNN Affinity Model v4
Improvements over v3:
  - Virtual node for global context
  - hidden_dim=384 (fix underfitting)
  - dropout=0.15
  - Two cross-attention blocks at different depths
  - Virtual node states fed into prediction head

Feature dims (locked):
  ligand_x: 43, ligand_edge_attr: 11
  protein_x: 35, protein_edge_attr: 22
  cross_edge_attr: 18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


class GeometricEdgeConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, out_dim, n_rbf=16, dropout=0.15):
        super().__init__(aggr='add')
        self.register_buffer('rbf_centers', torch.linspace(0.0, 10.0, n_rbf))
        self.rbf_width = 10.0 / n_rbf

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim + n_rbf, out_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(out_dim, out_dim),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(out_dim, out_dim),
        )
        self.norm     = nn.LayerNorm(out_dim)
        self.dropout  = nn.Dropout(dropout)
        self.residual = nn.Linear(node_dim, out_dim) if node_dim != out_dim else nn.Identity()

    def rbf_encode(self, dist):
        return torch.exp(
            -((dist.unsqueeze(-1) - self.rbf_centers) ** 2) / (2 * self.rbf_width ** 2)
        )

    def forward(self, x, edge_index, edge_attr, pos):
        row, col = edge_index
        dist     = (pos[row] - pos[col]).norm(dim=-1)
        rbf      = self.rbf_encode(dist)
        edge_emb = self.edge_encoder(torch.cat([edge_attr, rbf], dim=-1))
        out      = self.propagate(edge_index, x=x, edge_emb=edge_emb)
        out      = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(self.dropout(out) + self.residual(x))

    def message(self, x_j, edge_emb):
        return self.message_mlp(torch.cat([x_j, edge_emb], dim=-1))


class VirtualNodeUpdate(nn.Module):
    """
    Global virtual node: aggregates all nodes -> updates VN state ->
    broadcasts back to all nodes. Gives every atom graph-level context
    without requiring deep message passing chains.
    """
    def __init__(self, hidden_dim, dropout=0.15):
        super().__init__()
        self.to_vn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.from_vn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vn_norm   = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, batch, vn):
        """
        x:     [N_total, H]
        batch: [N_total]
        vn:    [B, H]  current virtual node state
        """
        # Aggregate nodes -> virtual node
        node_mean = global_mean_pool(x, batch)                    # [B, H]
        vn_new    = self.vn_norm(vn + self.to_vn(
            torch.cat([node_mean, vn], dim=-1)
        ))
        # Broadcast virtual node -> nodes
        vn_per_node = vn_new[batch]                               # [N, H]
        x_new = self.node_norm(x + self.dropout(
            self.from_vn(torch.cat([x, vn_per_node], dim=-1))
        ))
        return x_new, vn_new


class EdgeGuidedCrossAttention(nn.Module):
    """
    Fully batched cross attention restricted to spatial cross edges (4.5 A).
    Both directions computed simultaneously via scatter_add_.
    """
    def __init__(self, hidden_dim, cross_edge_dim=18, dropout=0.15):
        super().__init__()
        H = hidden_dim
        self.hidden_dim = H
        self.scale      = H ** -0.5

        self.q_pro = nn.Linear(H, H)
        self.k_lig = nn.Linear(H, H)
        self.v_lig = nn.Linear(H, H)
        self.q_lig = nn.Linear(H, H)
        self.k_pro = nn.Linear(H, H)
        self.v_pro = nn.Linear(H, H)

        self.edge_bias = nn.Linear(cross_edge_dim, 1, bias=False)
        self.out_pro   = nn.Linear(H, H)
        self.out_lig   = nn.Linear(H, H)
        self.norm_pro  = nn.LayerNorm(H)
        self.norm_lig  = nn.LayerNorm(H)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, pro_x, lig_x, cross_ei, cross_ea):
        if cross_ei.shape[1] == 0:
            return pro_x, lig_x

        pi  = cross_ei[0]
        li  = cross_ei[1]
        H   = self.hidden_dim
        bias = self.edge_bias(cross_ea).squeeze(-1)

        def attend(Q_src, K_tgt, V_tgt, src_idx, tgt_idx, n_tgt):
            scores = (Q_src * K_tgt).sum(-1) * self.scale + bias
            scores = scores - scores.max()
            exp_s  = torch.exp(scores)
            denom  = torch.zeros(n_tgt, device=Q_src.device)
            denom.scatter_add_(0, tgt_idx, exp_s)
            w      = exp_s / (denom[tgt_idx] + 1e-8)
            out    = torch.zeros(n_tgt, H, device=Q_src.device)
            out.scatter_add_(0, tgt_idx.unsqueeze(-1).expand(-1, H),
                             w.unsqueeze(-1) * V_tgt)
            return out

        # Ligand -> Protein
        pro_upd = attend(self.q_pro(pro_x)[pi], self.k_lig(lig_x)[li],
                         self.v_lig(lig_x)[li], pi, pi, pro_x.shape[0])
        pro_x_new = self.norm_pro(pro_x + self.dropout(self.out_pro(pro_upd)))

        # Protein -> Ligand
        lig_upd = attend(self.q_lig(lig_x)[li], self.k_pro(pro_x)[pi],
                         self.v_pro(pro_x)[pi], li, li, lig_x.shape[0])
        lig_x_new = self.norm_lig(lig_x + self.dropout(self.out_lig(lig_upd)))

        return pro_x_new, lig_x_new


class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, batch):
        n_graphs = int(batch.max().item()) + 1
        gates    = self.gate(x) - self.gate(x).max()
        exp_g    = torch.exp(gates)
        denom    = torch.zeros(n_graphs, 1, device=x.device)
        denom.scatter_add_(0, batch.unsqueeze(1), exp_g)
        weights  = exp_g / (denom[batch] + 1e-8)
        attn_out = torch.zeros(n_graphs, x.shape[1], device=x.device)
        attn_out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), weights * x)
        return torch.cat([attn_out, global_mean_pool(x, batch)], dim=-1)  # [B, 2H]


class AffinityModel(nn.Module):
    """
    ALCAS Geometric GNN v4

    Block A (2 conv) -> VirtualNode -> CrossAttn ->
    Block B (2 conv) -> VirtualNode -> CrossAttn ->
    AttentivePool + VN states -> MLP head

    Head input: 2H (lig) + 2H (pro) + H (lig_vn) + H (pro_vn) = 6H
    """
    def __init__(
        self,
        ligand_node_dim:  int = 43,
        ligand_edge_dim:  int = 11,
        protein_node_dim: int = 35,
        protein_edge_dim: int = 22,
        cross_edge_dim:   int = 18,
        hidden_dim:       int = 384,
        dropout:          float = 0.15,
        n_rbf:            int = 16,
    ):
        super().__init__()
        H = hidden_dim

        self.lig_input = nn.Sequential(
            nn.Linear(ligand_node_dim, H), nn.SiLU(), nn.Dropout(dropout),
        )
        self.pro_input = nn.Sequential(
            nn.Linear(protein_node_dim, H), nn.SiLU(), nn.Dropout(dropout),
        )

        self.lig_vn_init = nn.Parameter(torch.zeros(1, H))
        self.pro_vn_init = nn.Parameter(torch.zeros(1, H))

        def make_convs(edge_dim):
            return nn.ModuleList([
                GeometricEdgeConv(H, edge_dim, H, n_rbf, dropout),
                GeometricEdgeConv(H, edge_dim, H, n_rbf, dropout),
            ])

        self.lig_convs_a = make_convs(ligand_edge_dim)
        self.pro_convs_a = make_convs(protein_edge_dim)
        self.lig_vn_a    = VirtualNodeUpdate(H, dropout)
        self.pro_vn_a    = VirtualNodeUpdate(H, dropout)
        self.cross_attn_1 = EdgeGuidedCrossAttention(H, cross_edge_dim, dropout)

        self.lig_convs_b = make_convs(ligand_edge_dim)
        self.pro_convs_b = make_convs(protein_edge_dim)
        self.lig_vn_b    = VirtualNodeUpdate(H, dropout)
        self.pro_vn_b    = VirtualNodeUpdate(H, dropout)
        self.cross_attn_2 = EdgeGuidedCrossAttention(H, cross_edge_dim, dropout)

        self.lig_pool = AttentivePooling(H)
        self.pro_pool = AttentivePooling(H)

        self.head = nn.Sequential(
            nn.Linear(H * 6, H * 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(H * 2, H),     nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(H, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.lig_vn_init)
        nn.init.zeros_(self.pro_vn_init)

    def forward(self, batch):
        lig_x   = batch['ligand_x']
        lig_ei  = batch['ligand_edge_index']
        lig_ea  = batch['ligand_edge_attr']
        lig_pos = batch['ligand_pos']
        lig_b   = batch['ligand_batch']

        pro_x   = batch['protein_x']
        pro_ei  = batch['protein_edge_index']
        pro_ea  = batch['protein_edge_attr']
        pro_pos = batch['protein_pos']
        pro_b   = batch['protein_batch']

        cross_ei = batch['cross_edge_index']
        cross_ea = batch['cross_edge_attr']

        B = int(lig_b.max().item()) + 1

        lig_x = self.lig_input(lig_x)
        pro_x = self.pro_input(pro_x)

        lig_vn = self.lig_vn_init.expand(B, -1).clone()
        pro_vn = self.pro_vn_init.expand(B, -1).clone()

        # Block A
        for conv in self.lig_convs_a:
            lig_x = conv(lig_x, lig_ei, lig_ea, lig_pos)
        for conv in self.pro_convs_a:
            pro_x = conv(pro_x, pro_ei, pro_ea, pro_pos)
        lig_x, lig_vn = self.lig_vn_a(lig_x, lig_b, lig_vn)
        pro_x, pro_vn = self.pro_vn_a(pro_x, pro_b, pro_vn)
        pro_x, lig_x  = self.cross_attn_1(pro_x, lig_x, cross_ei, cross_ea)

        # Block B
        for conv in self.lig_convs_b:
            lig_x = conv(lig_x, lig_ei, lig_ea, lig_pos)
        for conv in self.pro_convs_b:
            pro_x = conv(pro_x, pro_ei, pro_ea, pro_pos)
        lig_x, lig_vn = self.lig_vn_b(lig_x, lig_b, lig_vn)
        pro_x, pro_vn = self.pro_vn_b(pro_x, pro_b, pro_vn)
        pro_x, lig_x  = self.cross_attn_2(pro_x, lig_x, cross_ei, cross_ea)

        lig_graph = self.lig_pool(lig_x, lig_b)
        pro_graph = self.pro_pool(pro_x, pro_b)

        combined = torch.cat([lig_graph, pro_graph, lig_vn, pro_vn], dim=-1)
        return self.head(combined).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)