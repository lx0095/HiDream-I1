import math
import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
from .attention import FeedForwardSwiGLU

class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_routed_experts=4, num_activated_experts=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        ### select top-k experts
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            topk_weight, topk_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(logits, k=self.top_k)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        return topk_idx, topk_weight, row_idx

class MOEFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
    ):
        super().__init__()
        self.shared_experts = FeedForwardSwiGLU(dim, hidden_dim // 2)
        self.experts = nn.ModuleList([FeedForwardSwiGLU(dim, hidden_dim) for i in range(num_routed_experts)])
        self.gate = MoEGate(
            embed_dim = dim, 
            num_routed_experts = num_routed_experts, 
            num_activated_experts = num_activated_experts
        )
        self.num_activated_experts = num_activated_experts
        self.num_routed_experts = num_routed_experts

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        topk_idx, topk_weight, row_idx = self.gate(x) 
        y = self.moe_infer(x, topk_idx, topk_weight, row_idx).view(*orig_shape)
        y = y + self.shared_experts(identity)

        return y

    @torch.no_grad()
    def moe_infer(self, x, experts_idx, experts_weight, row_idx):
        expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, experts_idx, active_num=x.shape[0])
        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, self.num_routed_experts)
        expert_tokens = expert_tokens.cpu().numpy()

        up_out = torch_npu.npu_grouped_matmul(
            x=[expanded_x],
            weight=[expert_i.w1.weight.transpose(0, 1) for expert_i in self.experts],
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
        )
        gate_out = torch_npu.npu_grouped_matmul(
            x=[expanded_x],
            weight=[expert_i.w3.weight.transpose(0, 1) for expert_i in self.experts],
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
        )

        up_out = torch.cat(up_out, dim=0)
        gate_out = torch.cat(gate_out, dim=0)
        gate_up_out = F.silu(up_out) * gate_out

        down_out = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[expert_i.w2.weight.transpose(0, 1) for expert_i in self.experts],
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
        )
        down_out = torch.cat(down_out, dim=0)

        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            down_out,
            skip1=None,
            skip2=None,
            bias=None,
            scales=experts_weight,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=experts_weight,
        )

        return final_hidden_states
