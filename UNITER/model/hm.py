"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Uniter for HM model
"""
import torch
from torch import nn
from torch.nn import functional as F

from .model import UniterPreTrainedModel, UniterModel
from .attention import MultiheadAttention


class UniterForHm(UniterPreTrainedModel):
    def __init__(self, config, img_dim):
        super().__init__(config)

        self.uniter = UniterModel(config, img_dim)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        logits = self.dense(pooled_output)
        logits = self.dropout(logits)
        # logits = self.classifier(logits).squeeze(1)
        logits = self.classifier(logits)

        if compute_loss:
            # hm_loss = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction='none')
            hm_loss = F.cross_entropy(logits, targets, reduction='none')
            return hm_loss
        else:
            return logits


class UniterForHmPaired(UniterPreTrainedModel):
    """ Finetune UNITER for HM (paired format)
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.hm_output = nn.Linear(config.hidden_size*2, 1)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        pooled_output = self.uniter.pooler(sequence_output)
        # concat CLS of the pair
        n_pair = pooled_output.size(0) // 2
        reshaped_output = pooled_output.contiguous().view(n_pair, -1)
        logits = self.hm_output(reshaped_output).squeeze(1)

        if compute_loss:
            hm_loss = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction='none')
            return hm_loss
        else:
            return logits


class AttentionPool(nn.Module):
    """ attention pooling layer """
    def __init__(self, hidden_size, drop=0.0):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.ReLU())
        self.dropout = nn.Dropout(drop)

    def forward(self, input_, mask=None):
        """input: [B, T, D], mask = [B, T]"""
        score = self.fc(input_).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=input_.dtype) * -1e4
            score = score + mask
        norm_score = self.dropout(F.softmax(score, dim=1))
        output = norm_score.unsqueeze(1).matmul(input_).squeeze(1)
        return output


class UniterForHmPairedAttn(UniterPreTrainedModel):
    """ Finetune UNITER for HM
        (paired format with additional attention layer)
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.attn1 = MultiheadAttention(config.hidden_size,
                                        config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        self.attn2 = MultiheadAttention(config.hidden_size,
                                        config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        self.fc = nn.Sequential(
            nn.Linear(2*config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob))
        self.attn_pool = AttentionPool(config.hidden_size,
                                       config.attention_probs_dropout_prob)
        # self.hm_output = nn.Linear(2*config.hidden_size, 1)
        self.hm_output = nn.Linear(2 * config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        # separate left image and right image
        bs, tl, d = sequence_output.size()
        left_out, right_out = sequence_output.contiguous().view(
            bs//2, tl*2, d).chunk(2, dim=1)
        # bidirectional attention
        mask = attn_masks == 0
        left_mask, right_mask = mask.contiguous().view(bs//2, tl*2
                                                       ).chunk(2, dim=1)
        left_out = left_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)
        l2r_attn, _ = self.attn1(left_out, right_out, right_out,
                                 key_padding_mask=right_mask)
        r2l_attn, _ = self.attn2(right_out, left_out, left_out,
                                 key_padding_mask=left_mask)
        left_out = self.fc(torch.cat([l2r_attn, left_out], dim=-1)
                           ).transpose(0, 1)
        right_out = self.fc(torch.cat([r2l_attn, right_out], dim=-1)
                            ).transpose(0, 1)
        # attention pooling and final prediction
        left_out = self.attn_pool(left_out, left_mask)
        right_out = self.attn_pool(right_out, right_mask)
        # logits = self.hm_output(torch.cat([left_out, right_out], dim=-1)).squeeze(1)
        logits = self.hm_output(torch.cat([left_out, right_out], dim=-1))

        if compute_loss:
            # hm_loss = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction='none')
            hm_loss = F.cross_entropy(logits, targets, reduction='none')
            return hm_loss
        else:
            return logits


class UniterTripleAttn(UniterPreTrainedModel):
    """ Finetune UNITER for Hateful memes with
        triple format with additional attention layers
        [(img, label), (img, caption), (img, paraphrased label)]

        Idea by Vu Dinh Anh
    """
    
    def __init__(self, config, img_dim):
        super().__init__(config)

        self.number_of_elements = 3 # Triple format, therefore 3
        
        # Model parts
        self.uniter = UniterModel(config, img_dim)

        self.attn1 = MultiheadAttention(config.hidden_size, 
                                                config.num_attention_heads, 
                                                config.attention_probs_dropout_prob)
        self.attn2 = MultiheadAttention(config.hidden_size, 
                                                config.num_attention_heads, 
                                                config.attention_probs_dropout_prob)
        self.attn3 = MultiheadAttention(config.hidden_size, 
                                                config.num_attention_heads, 
                                                config.attention_probs_dropout_prob)

        self.fc = nn.Sequential(nn.Linear(2 * config.hidden_size, config.hidden_size), 
                                nn.ReLU(),
                                nn.Dropout(config.hidden_dropout_prob))

        self.attn_pool = AttentionPool(config.hidden_size,
                                        config.attention_probs_dropout_prob)

        self.hm_output = nn.Linear(self.number_of_elements * config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):
        
        sequence_output = self.uniter(input_ids, position_ids,
                                        img_feat, img_pos_feat,
                                        attn_masks, gather_index,
                                        output_all_encoded_layers=False,
                                        img_type_ids=img_type_ids)

        # Separate into self.number_of_elements
        bs, tl, d = sequence_output.size() # bs : batch size; tl : topleft; d : dimension (maybe)
        left_out, middle_out, right_out = sequence_output.contiguous().view(bs // self.number_of_elements,
                                                                            tl * self.number_of_elements, 
                                                                            d).chunk(self.number_of_elements, dim=1)

        # Tri-directional attention
        mask = attn_masks == 0
        left_mask, middle_mask, right_mask = mask.contiguous().view(bs // self.number_of_elements,
                                                                    tl * self.number_of_elements).chunk(self.number_of_elements, dim=1)

        left_out = left_out.transpose(0, 1)
        middle_out = middle_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)

        l2m_attn, _ = self.attn1(left_out, middle_out, middle_out,
                                        key_padding_mask=middle_mask) # left to middle
        m2r_attn, _ = self.attn2(middle_out, right_out, right_out,
                                        key_padding_mask=right_mask) # middle to right
        r2l_attn, _ = self.attn3(right_out, left_out, left_out,
                                        key_padding_mask=left_mask) # right to left

        # 3 Fully-connected layers
        left_out = self.fc(torch.cat([l2m_attn, left_out], dim=-1)
                            ).transpose(0, 1)
        middle_out = self.fc(torch.cat([m2r_attn, middle_out], dim=-1)
                            ).transpose(0, 1)
        right_out = self.fc(torch.cat([r2l_attn, right_out], dim=-1)
                            ).transpose(0, 1)

        # 3 Attention pools
        left_out = self.attn_pool(left_out, left_mask)
        middle_out = self.attn_pool(middle_out, middle_mask)
        right_out = self.attn_pool(right_out, right_mask)

        # Final prediction
        logits = self.hm_output(torch.cat([left_out, middle_out, right_out], dim=-1))

        if compute_loss:
            return F.cross_entropy(logits, targets, reduction='none')
        else: return logits


class UniterCyclicAttn(UniterPreTrainedModel):
    """ Finetune UNITER for Hateful memes with
        n format with cyclic attention layers
        
        Idea by Vu Dinh Anh to generalize UniterTripleAttn

        n = num_attn = number of attention layers
    """

    def __init__(self, config, img_dim, num_attn = 4):
        super().__init__(config)

        self.num_attn = num_attn

        # Model parts
        self.uniter = UniterModel(config, img_dim)

        self.attn_list = nn.ModuleList([MultiheadAttention(config.hidden_size, 
                                                            config.num_attention_heads, 
                                                            config.attention_probs_dropout_prob) 
                                                            for _ in range(self.num_attn)])

        self.fc = nn.Sequential(nn.Linear(2 * config.hidden_size, config.hidden_size), 
                                nn.ReLU(),
                                nn.Dropout(config.hidden_dropout_prob))

        self.attn_pool = AttentionPool(config.hidden_size,
                                        config.attention_probs_dropout_prob)

        self.hm_output = nn.Linear(self.num_attn * config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):

        sequence_output = self.uniter(input_ids, position_ids,
                                        img_feat, img_pos_feat,
                                        attn_masks, gather_index,
                                        output_all_encoded_layers=False,
                                        img_type_ids=img_type_ids)

        # Separate into self.num_att
        bs, tl, d = sequence_output.size() # bs : batch size; tl : topleft; d : dimension (maybe)
        out_list = sequence_output.contiguous().view(bs // self.num_attn,
                                                    tl * self.num_attn, 
                                                    d).chunk(self.num_attn, dim=1)

        # n-cyclic-directional attention
        mask = attn_masks == 0
        mask_list = mask.contiguous().view(bs // self.num_attn,
                                            tl * self.num_attn).chunk(self.num_attn, dim=1)

        out_list = torch.stack([out_.transpose(0, 1) for out_ in out_list])

        attn_out_list = []
        for i, attn in enumerate(self.attn_list):
            if i + 1 == self.num_attn:
                attn_out, _ = attn(out_list[i], out_list[0], out_list[0],
                                    key_padding_mask=mask_list[0])
                attn_out_list.append(attn_out)
            else:
                attn_out, _ = attn(out_list[i], out_list[i+1], out_list[i+1],
                                    key_padding_mask=mask_list[i+1])
                attn_out_list.append(attn_out)

        attn_out_list = torch.stack(attn_out_list)
        
        # n-fully-connected layers
        out_list = torch.stack([self.fc(torch.cat([attn_out_list[i], out_list[i]], dim=-1)
                                            ).transpose(0, 1) 
                                            for i in range(self.num_attn)])

        # n attention pools
        out_list = [self.attn_pool(out_list[i], mask_list[i]) 
                    for i in range(self.num_attn)]

        # Final prediction
        logits = self.hm_output(torch.cat(out_list, dim=-1))

        if compute_loss:
            return F.cross_entropy(logits, targets, reduction='none')
        else: return logits


class UniterQuadrupleAttn(UniterPreTrainedModel):
    """ Finetune UNITER for Hateful memes with
        triple format with additional attention layers
        [(img, label), (img, caption), (img, paraphrased label)]

        Unlike UniterTripleAttn, this has 4 attentions.

        paraphrased label <-> label <-> caption

        Idea by Vu Dinh Anh
    """
    def __init__(self, config, img_dim):
        super().__init__(config)

        self.num_form = 3 # Triple format, therefore 3
        self.num_attn = 4 # Four attentions, therefore 4

        # Model parts
        self.uniter = UniterModel(config, img_dim)

        self.attn_list = nn.ModuleList([MultiheadAttention(config.hidden_size,
                                                            config.num_attention_heads,
                                                            config.attention_probs_dropout_prob)
                                                            for _ in range(self.num_attn)])

        self.fc = nn.Sequential(nn.Linear(2 * config.hidden_size, config.hidden_size), 
                                nn.ReLU(),
                                nn.Dropout(config.hidden_dropout_prob))

        self.attn_pool = AttentionPool(config.hidden_size,
                                        config.attention_probs_dropout_prob)

        self.hm_output = nn.Linear(self.num_attn * config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):

        sequence_output = self.uniter(input_ids, position_ids,
                                        img_feat, img_pos_feat,
                                        attn_masks, gather_index,
                                        output_all_encoded_layers=False,
                                        img_type_ids=img_type_ids)

        # Separate into self.num_form
        bs, tl, d = sequence_output.size() # bs : batch size; tl : topleft; d : dimension (maybe)
        left_out, middle_out, right_out = sequence_output.contiguous().view(bs // self.num_form,
                                                                            tl * self.num_form, 
                                                                            d).chunk(self.num_form, dim=1)

        # Quad-directional attention
        mask = attn_masks == 0
        left_mask, middle_mask, right_mask = mask.contiguous().view(bs // self.num_form,
                                                                    tl * self.num_form).chunk(self.num_form, dim=1)

        left_out = left_out.transpose(0, 1)
        middle_out = middle_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)

        l2m_attn, _ = self.attn_list[0](left_out, middle_out, middle_out,
                                        key_padding_mask=middle_mask)

        m2l_attn, _ = self.attn_list[1](middle_out, left_out, left_out,
                                        key_padding_mask=left_mask)

        l2r_attn, _ = self.attn_list[2](left_out, right_out, right_out,
                                        key_padding_mask=right_mask)

        r2l_attn, _ = self.attn_list[3](right_out, left_out, left_out,
                                        key_padding_mask=left_mask)

        # 4 Fully-connected layers
        l2m_fc = self.fc(torch.cat([l2m_attn, left_out], dim=-1)
                        ).transpose(0, 1)

        m2l_fc = self.fc(torch.cat([m2l_attn, middle_out], dim=-1)
                        ).transpose(0, 1)

        l2r_fc = self.fc(torch.cat([l2r_attn, left_out], dim=-1)
                        ).transpose(0, 1)

        r2l_fc = self.fc(torch.cat([r2l_attn, right_out], dim=-1)
                        ).transpose(0, 1)

        # 4 Attention pools
        l2m_pool = self.attn_pool(l2m_fc, left_mask)
        m2l_pool = self.attn_pool(m2l_fc, middle_mask)
        l2r_pool = self.attn_pool(l2r_fc, left_mask)
        r2l_pool = self.attn_pool(r2l_fc, right_mask)

        # Final prediction
        logits = self.hm_output(torch.cat([l2m_pool, m2l_pool, l2r_pool, r2l_pool], dim=-1))

        if compute_loss:
            return F.cross_entropy(logits, targets, reduction='none')
        else: return logits


class UniterHextupleAttn(UniterPreTrainedModel):
    """ Finetune UNITER for Hateful memes with
        triple format with additional attention layers
        [(img, label), (img, caption), (img, paraphrased label)]

        Unlike UniterTripleAttn, this has 6 attentions.

        paraphrased label <-> label <-> caption <-> paraphrased label
        Imagine having 3 nodes that are fully connected.
        Each edge will be an attention.

        Idea by Vu Dinh Anh
    """
    def __init__(self, config, img_dim):
        super().__init__(config)

        self.num_form = 3 # Triple format, therefore 3
        self.num_attn = 6 # Hextuple attentions, therefore 6

        # Model parts
        self.uniter = UniterModel(config, img_dim)

        self.attn_list = nn.ModuleList([MultiheadAttention(config.hidden_size,
                                                            config.num_attention_heads,
                                                            config.attention_probs_dropout_prob)
                                                            for _ in range(self.num_attn)])

        self.fc = nn.Sequential(nn.Linear(2 * config.hidden_size, config.hidden_size), 
                                nn.ReLU(),
                                nn.Dropout(config.hidden_dropout_prob))

        self.attn_pool = AttentionPool(config.hidden_size,
                                        config.attention_probs_dropout_prob)

        self.hm_output = nn.Linear(self.num_attn * config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, position_ids, img_feat, img_pos_feat,
                attn_masks, gather_index,
                img_type_ids, targets, compute_loss=True):

        sequence_output = self.uniter(input_ids, position_ids,
                                        img_feat, img_pos_feat,
                                        attn_masks, gather_index,
                                        output_all_encoded_layers=False,
                                        img_type_ids=img_type_ids)

        # Separate into self.num_form
        bs, tl, d = sequence_output.size() # bs : batch size; tl : topleft; d : dimension (maybe)
        left_out, middle_out, right_out = sequence_output.contiguous().view(bs // self.num_form,
                                                                            tl * self.num_form, 
                                                                            d).chunk(self.num_form, dim=1)

        # Hex-directional attention
        mask = attn_masks == 0
        left_mask, middle_mask, right_mask = mask.contiguous().view(bs // self.num_form,
                                                                    tl * self.num_form).chunk(self.num_form, dim=1)

        left_out = left_out.transpose(0, 1)
        middle_out = middle_out.transpose(0, 1)
        right_out = right_out.transpose(0, 1)

        l2m_attn, _ = self.attn_list[0](left_out, middle_out, middle_out,
                                        key_padding_mask=middle_mask)

        m2r_attn, _ = self.attn_list[1](middle_out, right_out, right_out,
                                        key_padding_mask=right_mask)

        r2l_attn, _ = self.attn_list[2](right_out, left_out, left_out,
                                        key_padding_mask=left_mask)

        l2r_attn, _ = self.attn_list[3](left_out, right_out, right_out,
                                        key_padding_mask=right_mask)

        r2m_attn, _ = self.attn_list[4](right_out, middle_out, middle_out,
                                        key_padding_mask=middle_mask)

        m2l_attn, _ = self.attn_list[5](middle_out, left_out, left_out,
                                        key_padding_mask=left_mask)

        # 6 Fully-connected layers
        l2m_fc = self.fc(torch.cat([l2m_attn, left_out], dim=-1)
                        ).transpose(0, 1)

        m2r_fc = self.fc(torch.cat([m2r_attn, middle_out], dim=-1)
                        ).transpose(0, 1)

        r2l_fc = self.fc(torch.cat([r2l_attn, right_out], dim=-1)
                        ).transpose(0, 1)

        l2r_fc = self.fc(torch.cat([l2r_attn, left_out], dim=-1)
                        ).transpose(0, 1)

        r2m_fc = self.fc(torch.cat([r2m_attn, right_out], dim=-1)
                        ).transpose(0, 1)

        m2l_fc = self.fc(torch.cat([m2l_attn, middle_out], dim=-1)
                        ).transpose(0, 1)

        # 6 Attention pools
        l2m_pool = self.attn_pool(l2m_fc, left_mask)
        m2r_pool = self.attn_pool(m2r_fc, middle_mask)
        r2l_pool = self.attn_pool(r2l_fc, right_mask)
        l2r_pool = self.attn_pool(l2r_fc, left_mask)
        r2m_pool = self.attn_pool(r2m_fc, right_mask)
        m2l_pool = self.attn_pool(m2l_fc, middle_mask)

        # Final prediction
        logits = self.hm_output(torch.cat([l2m_pool, m2r_pool, r2l_pool, l2r_pool, r2m_pool, m2l_pool], dim=-1))

        # Final prediction

        if compute_loss:
            return F.cross_entropy(logits, targets, reduction='none')
        else: return logits