import math
import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
from .attention import FeedForwardSwiGLU
from torch.distributed.nn.functional import all_gather

_LOAD_BALANCING_LOSS = []
def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)

def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()

def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS

def batched_load_balancing_loss():
    aux_losses_arr = get_load_balancing_loss()
    alpha = aux_losses_arr[0][-1]
    Pi = torch.stack([ent[1] for ent in aux_losses_arr], dim=0)
    fi = torch.stack([ent[2] for ent in aux_losses_arr], dim=0)

    fi_list = all_gather(fi)
    fi = torch.stack(fi_list, 0).mean(0)

    aux_loss = (Pi * fi).sum(-1).mean() * alpha
    return aux_loss

# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
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
        bsz, seq_len, h = hidden_states.shape    
        # print(bsz, seq_len, h)    
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        
        ### select top-k experts
        if self.scoring_func == 'softmax':
            topk_weight, topk_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(logits, k=self.top_k)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = logits.softmax(dim=-1)
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)

                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
                save_load_balancing_loss((aux_loss, Pi, fi, self.alpha))
        else:
            aux_loss = None
        return topk_idx, topk_weight, row_idx, aux_loss

# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
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
        wtype = x.dtype
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, row_idx, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        if self.training:
            flat_topk_idx = topk_idx.view(-1)
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=wtype)
            for i, expert in enumerate(self.experts): 
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=wtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape).to(dtype=wtype)
            #y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, topk_idx, topk_weight.view(-1, 1), row_idx).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights, row_idx):
        expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(x, row_idx, flat_expert_indices, active_num=x.shape[0])
        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, self.num_routed_experts)
        # expert_tokens = expert_tokens.to(torch.int64)
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
        # maybe has accuracy problem
        # gate_up_out = torch_npu.npu_swiglu(gate_up_out)

        down_out = torch_npu.npu_grouped_matmul(
            x=[gate_up_out],
            weight=[expert_i.w2.weight.transpose(0, 1) for expert_i in self.experts],
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
        )
        # TODO: Reorder device memory 2 times here, replace the current
        # implementation here when suitable operators become available.
        down_out = torch.cat(down_out, dim=0)
        final_hidden_states = torch_npu.npu_moe_finalize_routing(
            down_out,
            skip1=None,
            skip2=None,
            bias=None,
            scales=flat_expert_weights.view(-1, 2),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=flat_expert_indices,
        )
        #breakpoint()
        return final_hidden_states
