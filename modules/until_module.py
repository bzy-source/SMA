# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from modules.until_config import PretrainedConfig
from scipy.stats import entropy

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None,  *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model



def js_divergence(p, q):
    m = 0.5 * (p + q)
    kl_div_pm = F.kl_div(p.log(), m, reduction='batchmean')
    kl_div_qm = F.kl_div(q.log(), m, reduction='batchmean')
    js_div = 0.5 * kl_div_pm + 0.5 * kl_div_qm
    return js_div



##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class Triplet_Loss(nn.Module):
    def __init__(self,):
        super(Triplet_Loss, self).__init__()
    
    def forward(self, sim_matrix, margin=1.0):
        B = sim_matrix.size(0)
        # obtain positive sample and negative sample
        positive_samples = sim_matrix[torch.arange(B), torch.arange(B)]
        
        # 找到负样本的最大值（非对角线）
        negative_samples = sim_matrix[~torch.eye(B, dtype=bool)].view(B, -1).max(dim=1)[0]
        
        # 计算损失
        loss = F.relu(positive_samples - negative_samples + margin)
        
        return loss.mean()
    

class NegNCE(nn.Module):
    def __init__(self, ):
        super(NegNCE, self).__init__()
        self.c_pos_w = 1.0
        self.c_neg_w = 0.5
        self.margin = 0.0
    def forward(self, sim_matrix, logit_scale=1.0):  # temp = 100, refer to X-CLIP
        # sim_matrix: [batch_t, batch_v]
        logpt = F.softmax(sim_matrix * logit_scale, dim=-1)
        logpt = torch.clamp(logpt, 1e-6, 1 - 1e-6)
        positive_logit = torch.diag(logpt).unsqueeze(-1)
        mask = (torch.eye(logpt.size(0)) > .5).to(logpt.device)
        d1 = torch.diag(sim_matrix)
        x = sim_matrix
        max_margin = F.relu(self.margin + x - d1.view(-1, 1)) + F.relu(self.margin + x - d1.view(1, -1))
        max_margin = max_margin.masked_fill(mask, 0)
        hard_negative_logits = logpt[max_margin > 0.]
        # local_cross_entropy
        loss_pos = - torch.log(positive_logit)
        loss_neg = - torch.log(1 - hard_negative_logits)
        if len(loss_neg) > 0:
            sim_loss = self.c_pos_w * loss_pos.mean() + self.c_neg_w * loss_neg.mean()
        else:
            sim_loss = self.c_pos_w * loss_pos.mean()
        return sim_loss

class MILNCELoss(nn.Module):
    def __init__(self, batch_size=1, n_pair=1,):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair//2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()
        return sim_loss



class MaxMarginRankingLoss(nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

class PatchShiftModule(nn.Module):
    def __init__(self, net, video_frame, n_div):
        super().__init__()
        self.net = net
        self.video_frame = video_frame
        self.n_div = n_div
        logger.warning('Using patch shift!')
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        # here q == k == v, psm means patch shift output
        x = query # shape here is LND, not NLD (50, 384, 768)
        # print("query.shape: ", query.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        patch_len = x.shape[-2]
        fold = patch_len // self.n_div
        x = x.reshape(-1, self.video_frame, x.shape[-2], x.shape[-1])  # shape = [bs, frame, grid ** 2, width]   [bs, farme, patch_length, channel_dimension]
        psm = torch.zeros_like(x) # shape = [bs, frame, grid ** 2, width]
        psm[:,:,:,:] = x[:,:,:,:]

        # ##############channel shift##################
        # psm[:, 1:, :, :fold] = x[:, :-1, :, :fold]
        # psm[:, :-1, :, fold:2*fold] = x[:, 1:, :, fold:2*fold]
        # ##############channel shift##################

        # ##############patch channel shift##################
        # psm[:, 1:, 1:, :fold] = x[:, :-1, 1:, :fold]
        # psm[:, :-1, 1:, fold:2*fold] = x[:, 1:, 1:, fold:2*fold]
        # ##############patch channel shift##################

        # ##############cls channel shift##################
        # psm[:, 1:, :1, :fold] = x[:, :-1, :1, :fold]               #  沿时间轴往右
        # psm[:, :-1, :1, fold:2*fold] = x[:, 1:, :1, fold:2*fold]   #  沿时间轴往左
        # ##############cls channel shift##################


        ##############left and right shift##############
        lshift_indices = torch.arange(start=1, end=patch_len, step=fold)
        psm[:, 1:, lshift_indices, :] = x[:, :-1, lshift_indices, :] # f_t = f_t-1
        rshift_indices = torch.arange(start=1+3, end=patch_len, step=fold)
        psm[:, :-1, rshift_indices, :] = x[:, 1:, rshift_indices, :] # f_t = f_t+1
        ##############left and right shift##############
        x = psm.reshape(-1, patch_len, x.shape[-1])
        x = x.permute(1, 0, 2)  # NLD -> LND

        return self.net(x, x, x, need_weights=need_weights, attn_mask=attn_mask)

def make_patch_shift(net, video_frame=12, shift_layers=4, n_div=7):
    '''
    Args:
    net: CLIP
    video_frame: need predefine here
    shift_layers: layers to be shift
    '''
    
    def make_trans_patch_shift(stage, shift_layers):
        # net.clip.visual.transformer.resblocks[i] is a ResidualAttentionBlock type, contain net.attn -- a nn.MultiheadAttention
        # make a shift before net.attn, so it is a residual attn
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            # b is a ResidualAttentionBlock type, contain self.attn
            # if i==4 or i==6 or i==8 or i==10:
            #     blocks[i].attn = TokenShiftModule(b.attn, video_frame=video_frame, n_div=n_div)
            if i>=10 and i<=11:
                blocks[i].attn = PatchShiftModule(b.attn, video_frame=video_frame, n_div=n_div)
        return nn.Sequential(*blocks)

    net.clip.visual.transformer.resblocks = make_trans_patch_shift(net.clip.visual.transformer.resblocks, shift_layers=shift_layers)