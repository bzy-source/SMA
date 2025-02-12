from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from modules.modeling import CLIP4ClipPreTrainedModel, show_log, update_attr, check_attr
from modules.criterion import UncertaintyAwareLoss, VarianceLoss
from modules.module_agg import AggregationBlock

import numpy as np
from einops import rearrange, repeat


logger = logging.getLogger(__name__)
allgather = AllGather.apply


def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)

class my_model(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(my_model, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        self.lamda = self.task_config.lamda
        self.fi = self.task_config.fi
        self.visualize = self.task_config.visualize

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]    # shape [768, 3, 32, 32]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # shape [50, 768]
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]  # shape [512, 512]
        context_length = clip_state_dict["positional_embedding"].shape[0]  # shape [77, 512]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]    # shape [49408, 512]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]    # [512]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))
        self.patch_num = int((image_resolution/32) ** 2 + 1)
        self.embed_dim = embed_dim

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            self.word_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
            self.patch_position_embeddings = nn.Embedding(cross_config.max_position_embeddings+196, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        
        num_words = self.task_config.max_words
        num_frames = self.task_config.max_frames


        self.total_patch_num = num_frames*self.patch_num

        # recommend set True
        self.use_original_clip_for_frame_features = False   

        # for coarse-grained constrast weights
        self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)

        # for cross-grained constrast weights
        self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)        

        # for fine-grained constrast weights
        self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        self.patch_mat_weight =nn.parameter.Parameter(torch.eye(self.total_patch_num), requires_grad=True)
        self.frame_mat_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(self.task_config.txt_num_embeds), requires_grad=True)
        self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        self.patch_mat_weight2 = nn.parameter.Parameter(torch.eye(self.total_patch_num), requires_grad=True)
        self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(self.task_config.txt_num_embeds), requires_grad=True)
        self.pixel_mat_weight = nn.parameter.Parameter(torch.eye(self.task_config.select_frame*self.task_config.video_num_embeds), requires_grad=True)
        self.pixel_mat_weight2 = nn.parameter.Parameter(torch.eye(self.task_config.select_frame*self.task_config.video_num_embeds), requires_grad=True)

        # text video learnable query
        self.txt_num_embeds = self.task_config.txt_num_embeds
        self.vid_num_embeds = self.task_config.video_num_embeds

        self.txt_agg_block = AggregationBlock(
            depth = self.task_config.spm_depth,              
            input_channels = embed_dim,              
            input_axis = 1,                   
            num_latents = self.txt_num_embeds,
            latent_dim = embed_dim,
            num_classes = embed_dim,
            attn_dropout = self.task_config.dropout,
            ff_dropout = self.task_config.dropout,
            weight_tie_layers = self.task_config.spm_weight_sharing,
            pos_enc_type = self.task_config.spm_txt_pos_enc_type,
            pre_norm = self.task_config.spm_pre_norm,
            post_norm = self.task_config.spm_post_norm,
            activation = self.task_config.spm_activation,
            last_ln = self.task_config.spm_last_ln,
            ff_mult = self.task_config.spm_ff_mult,
            more_dropout = self.task_config.spm_more_dropout,
            first_order=self.task_config.first_order
        )

        self.vid_agg_block = AggregationBlock(
            depth = self.task_config.spm_depth,
            input_channels = embed_dim,
            input_axis = 1,
            num_latents = self.vid_num_embeds,
            latent_dim = embed_dim,
            num_classes = embed_dim,
            attn_dropout = self.task_config.dropout,
            ff_dropout = self.task_config.dropout,
            weight_tie_layers = self.task_config.spm_weight_sharing,
            pos_enc_type = self.task_config.spm_vid_pos_enc_type,
            pre_norm = self.task_config.spm_pre_norm,
            post_norm = self.task_config.spm_post_norm,
            activation = self.task_config.spm_activation,
            last_ln = self.task_config.spm_last_ln,
            ff_mult = self.task_config.spm_ff_mult,
            more_dropout = self.task_config.spm_more_dropout,
            first_order=self.task_config.first_order
        )

        self.query = nn.parameter.Parameter(torch.zeros(self.txt_num_embeds, embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.query)

        self.loss_fct = CrossEn()
        self.mse_loss = nn.MSELoss()
        self.loss_var = VarianceLoss()

        self.apply(self.init_weights) 

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])   # input_ids: torch.Size([150, 32])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) # token_type_ids: torch.Size([150, 32])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) # attention_mask: torch.Size([150, 32])
        video_mask = video_mask.view(-1, video_mask.shape[-1]) # video_mask: torch.Size([150, 12])

        # T x 3 x H x W
        video = torch.as_tensor(video).float() # torch.Size([150, 1, 12, 1, 3, 224, 224])
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        # [bs, 1, dim], [bs, num_words, dim], [bs, num_frames, dim], [bs, num_frames, patch_num, dim]
        (sequence_output, seq_features), (visual_output, visual_patch)= self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask, 
                                                                video, video_mask, shaped=True, video_frame=video_frame)
        if self.training:
            loss = 0.
            sim_matrix1, sim_matrix2 = self.get_similarity_logits(sequence_output, seq_features, visual_output, visual_patch, attention_mask, 
                                        video_mask, shaped=True, loose_type=self.loose_type)
            
            sim_loss1 = self.loss_fct(sim_matrix1)
            sim_loss2 = self.loss_fct(sim_matrix1.T)
            sim_loss3 = self.loss_fct(sim_matrix2)
            sim_loss4 = self.loss_fct(sim_matrix2.T)


            sim_loss = (sim_loss1 + sim_loss2)/2 + self.fi * (sim_loss3 + sim_loss4)/2
            
            if self.task_config.local_rank == 0:
                print(f" >>> sim_loss1:{sim_loss1}, sim_loss2:{sim_loss2}, sim_loss3:{sim_loss3}, sim_loss4:{sim_loss4}")

            logit_scale = self.clip.logit_scale.exp()

            query = self.query / self.query.norm(dim=-1,keepdim=True)
            qq_logits = logit_scale * torch.matmul(query, query.t())
            div_loss = self.loss_var(qq_logits)
            
            loss = loss + sim_loss + self.lamda * div_loss

            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden, seq_features, attn = self.clip.encode_text(input_ids, return_hidden=True) # sequence_hidden: torch.Size([bs, 512]), seq_features: torch.Size([bs, 32, 512])
        sequence_hidden, seq_features = sequence_hidden.float(), seq_features.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1)) # [bs, 1, dim]

        return sequence_hidden, seq_features

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden, visual_patch, attn = self.clip.encode_image(video, return_hidden=True, video_frame=video_frame) 
        # visual-hidden:[bs*frame_num, dim] visual_patch:[bs*frame_num, patch_num, dim] attn:[bs*frame, patch_num, patch_num]
        visual_hidden, visual_patch = visual_hidden.float(), visual_patch.float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1)) # [bs, frame_num, dim]
        patch_num = visual_patch.size(1)
        patch_dim = visual_patch.size(2)
        visual_patch = visual_patch.view(bs_pair, -1, patch_num, patch_dim)

        return visual_hidden, visual_patch

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output, seq_features = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)      # [bs, 1, dim], [bs, num_words, dim]
        visual_output, visual_patch_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)  # [bs, num_frames, dim], [bs, num_frames, patch_num, dim]


        return (sequence_output, seq_features), (visual_output, visual_patch_output)

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out


    def _loose_similarity(self, sequence_output, seq_features, visual_output, visual_patch, attention_mask, video_mask, sim_header="meanP"):
        """
            sequence_output: CLS token of text       # [bs, 1, dim]
            seq_features: all tokens of text         # [bs, num_words, dim]
            visual_output: all frames of video       # [bs, num_frames, dim]
            viusal_patch: all patches of video       # [bs, num_frames, patch_num, dim]
        """
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()
        seq_features = seq_features.contiguous()
        visual_patch = visual_patch.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            visual_output_original = visual_output
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output  # [bs, num_frames, dim]
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1) 
            frame_position_embeddings = self.frame_position_embeddings(position_ids) # [150, 12, 512]
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)[0]
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        # sentence-level textual feature
        sentence_output = sequence_output.squeeze(1)  # [150, 512]
         
        # word-level textual features
        # word_features = seq_features / seq_features.norm(dim=-1, keepdim=True)                   # [bs, num_words, dim]
        word_features = seq_features

        # video-level visual feature 
        video_output = visual_output
        # video_output = visual_output / visual_output.norm(dim=-1, keepdim=True)s
        video_output = self._mean_pooling_for_similarity_visual(video_output, video_mask)
        # video_output = video_output / video_output.norm(dim=-1, keepdim=True)                    # [bs, dim]

        # frame-level visual features       
        if self.use_original_clip_for_frame_features:
            frame_features = visual_output_original               # [bs, num_frames, dim]
        else:
            frame_features = visual_output                        # [bs, num_frames, dim]

        # patch-level viusal features
        # patch_features = visual_patch.view(visual_patch.shape[0], -1, visual_patch.shape[-1])
        patch_features = visual_patch

        # word agg and patch agg using ca
        b = word_features.shape[0]
        text_query = repeat(self.query, 'n d -> b n d', b = b)
        attention_mask_bool = attention_mask.bool()
        txt_agg, txt_agg_attn = self.txt_agg_block(text_query, word_features, attention_mask_bool)
        txt_agg = txt_agg.contiguous()


        v, f, p, dim = patch_features.shape
        b = v * f
        patch_token = patch_features.view(-1, p, dim)
        patch_token = patch_token[:, 1:, :]
        v_query = repeat(self.query, 'n d -> b n d', b = b)
        vid_agg, vid_agg_attn = self.vid_agg_block(v_query, patch_token)
        vid_agg = vid_agg.contiguous() 
        vid_agg = vid_agg.view(v, f, -1, dim)
        vid_agg_attn = vid_agg_attn.view(v, f, vid_agg_attn.shape[-2], vid_agg_attn.shape[-1])
        # word agg and patch agg using ca

                       
        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            video_mask = allgather(video_mask, self.task_config)
            video_output = allgather(video_output, self.task_config)
            frame_features = allgather(frame_features, self.task_config)
            patch_features = allgather(patch_features, self.task_config)
            sentence_output = allgather(sentence_output, self.task_config)
            word_features = allgather(word_features, self.task_config)
            txt_agg = allgather(txt_agg, self.task_config)
            vid_agg = allgather(vid_agg, self.task_config)
            torch.distributed.barrier()

        v_weight = torch.einsum('ad,bvd->abv', [sentence_output, frame_features]) # bs 512, bs 12 512 -> bs bs 12
        v_weight = torch.softmax(v_weight / self.task_config.temp, dim=-1)
        v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask])  
        video_feat_t_cond = torch.einsum('abv,bvd->abd', [v_weight, frame_features]) # bs bs 12, bs 12 512 -> bs bs 512
        
        # l2_norm features
        word_features = word_features / word_features.norm(dim=-1, keepdim=True)
        sentence_output = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]
        frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)
        video_output = video_output / video_output.norm(dim=-1, keepdim=True)
        video_feat_t_cond = video_feat_t_cond / video_feat_t_cond.norm(dim=-1, keepdim=True)
        txt_agg = txt_agg / txt_agg.norm(dim=-1, keepdim=True)
        vid_agg = vid_agg / vid_agg.norm(dim=-1, keepdim=True)


        # pooled-video sentence score
        pooled_video_sentence_logits = logit_scale * torch.einsum('ad,abd->ab', [sentence_output, video_feat_t_cond])

        sentence_frame = torch.matmul(sentence_output, frame_features.permute(0, 2, 1))
        sentence_frame = sentence_frame.permute(1, 0, 2)
        sorted_indices = torch.argsort(sentence_frame, dim=-1, descending=True) # [text, video, frame_num]
        sorted_indices = sorted_indices[:, :, :self.task_config.select_frame]

        txts_num = sorted_indices.shape[0]
        patch_agg = vid_agg.unsqueeze(0).expand(txts_num, -1, -1, -1, -1) # [text, video, frame_num, p, dim]
        txts_num, vid_num, frame_num, p, dim = patch_agg.shape
        flatten_patch_agg = patch_agg.reshape(-1, p*dim)
        sorted_indices = sorted_indices.reshape(txts_num*vid_num, -1)
        ind = torch.arange(txts_num*vid_num).view(-1, 1) * frame_num
        ind = ind.to(sorted_indices.device)
        sorted_indices = sorted_indices + ind
        
        selected_patch_agg = torch.gather(flatten_patch_agg, 0, sorted_indices.view(-1).unsqueeze(1).expand(-1, p*dim))
        selected_patch_agg = selected_patch_agg.view(txts_num, vid_num, self.task_config.select_frame, p, dim)

        # word_agg patch_agg similarity score #
        agg_score = torch.einsum('abfpd, apd -> abfp', [selected_patch_agg, txt_agg])
        agg_score = torch.sum(agg_score * torch.softmax(agg_score, dim=2), dim=2)
        agg_score = torch.sum(agg_score * torch.softmax(agg_score, dim=2), dim=2)

        return pooled_video_sentence_logits, agg_score


    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, seq_features, visual_output, visual_patch, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type: 
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            
            retrieve_logits1, retrieve_logits2 = self._loose_similarity(sequence_output, seq_features, visual_output, visual_patch, attention_mask, video_mask, sim_header=self.sim_header)
            return retrieve_logits1, retrieve_logits2
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction