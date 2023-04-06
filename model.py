import random
import math
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class Predictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, dropout=0.):
        super().__init__()
        self.mlp_sub = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp_obj = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        ent_sub = self.dropout(self.mlp_sub(x))
        ent_obj = self.dropout(self.mlp_obj(y))

        outputs = self.biaffine(ent_sub, ent_obj)

        return outputs


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x



class FFNN(nn.Module):
    def __init__(self, input_dim, hid_dim, cls_num, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, cls_num)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, hid_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.q_linear = nn.Linear(hid_size, hid_size)
        self.k_linear = nn.Linear(hid_size, hid_size * 2)

        self.factor = math.sqrt(hid_size)

        self.gate1 = Gate(hid_size, dropout=dropout)
        self.gate2 = Gate(hid_size, dropout=dropout)

    def forward(self, x, s, g):
        # x [B, L, H]
        # s [B, K, H]
        # g [B, N, H]
        # x = self.dropout(x)
        # s = self.dropout(s)
        q = self.q_linear(x)
        k_v = self.k_linear(g)
        k, v = torch.chunk(k_v, chunks=2, dim=-1)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.factor
        # scores = self.dropout(scores)
        scores = torch.softmax(scores, dim=-1)
        g = torch.bmm(scores, v)
        g = g.unsqueeze(2).expand(-1, -1, s.size(1), -1)
        h = x.unsqueeze(2).expand(-1, -1, s.size(1), -1)
        s = s.unsqueeze(1).expand(-1, x.size(1), -1, -1)

        h = self.gate1(h, g)
        h = self.gate2(h, s)
        return h


class Gate(nn.Module):
    def __init__(self, hid_size, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(hid_size * 2, hid_size)
        self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(hid_size, hid_size)

    def forward(self, x, y):
        '''
        :param x: B, L, K, H
        :param y: B, L, K, H
        :return:
        '''
        o = torch.cat([x, y], dim=-1)
        o = self.dropout(o)
        gate = self.linear(o)
        gate = torch.sigmoid(gate)
        o = gate * x + (1 - gate) * y
        # o = F.gelu(self.linear2(self.dropout(o)))
        return o


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.inner_dim = config.tri_hid_size
        self.tri_hid_size = config.tri_hid_size
        self.eve_hid_size = config.eve_hid_size
        self.event_num = config.tri_label_num
        self.role_num = config.rol_label_num
        self.teacher_forcing = True
        self.gamma = config.gamma
        self.arg_hid_size = config.arg_hid_size
        # self.layers = config.layers
        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)

        # self.tri_hid_size = 256
        # self.arg_hid_size = 384

        self.dropout = nn.Dropout(config.dropout)
        self.tri_linear = nn.Linear(config.bert_hid_size, self.tri_hid_size * 2)
        self.arg_linear = nn.Linear(config.bert_hid_size, self.arg_hid_size * 2)
        self.role_linear = nn.Linear(config.bert_hid_size, config.eve_hid_size * config.rol_label_num * 2)
        self.eve_embedding = nn.Embedding(self.event_num + 1, config.bert_hid_size, padding_idx=self.event_num) #.from_pretrained(self.reset_event_parameters(config.vocab, config.tokenizer), freeze=False)
        # self.layer_norm = LayerNorm(config.bert_hid_size, config.bert_hid_size, conditional=True)

        self.gate = AdaptiveFusion(config.bert_hid_size, dropout=config.dropout)

    def reset_event_parameters(self, vocab, tokenizer):
        labels = [vocab.tri_id2label[i] for i in range(self.event_num)]
        inputs = tokenizer(labels)
        input_ids = pad_sequence([torch.LongTensor(x) for x in inputs["input_ids"]], True)
        attention_mask = pad_sequence([torch.BoolTensor(x) for x in inputs["attention_mask"]], True)
        mask = pad_sequence([torch.BoolTensor(x[1:-1]) for x in inputs["attention_mask"]], True)
        bert_embs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embs = bert_embs[0][:, 1:-1]
        min_value = bert_embs.min().item()
        bert_embs = torch.masked_fill(bert_embs, mask.eq(0).unsqueeze(-1), min_value)
        bert_embs, _ = torch.max(bert_embs, dim=1)
        bert_embs = torch.cat([bert_embs, torch.zeros((1, bert_embs.size(-1)))], dim=0)
        return bert_embs.detach()


    def _sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.cuda()
        return embeddings

    def _pointer(self, qw, kw, word_mask2d):
        B, L, K, H = qw.size()
        pos_emb = self._sinusoidal_position_embedding(B, L, H)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        grid_mask2d = word_mask2d.unsqueeze(1).expand(B, K, L, L).float()
        logits = logits * grid_mask2d - (1 - grid_mask2d) * 1e12
        return logits

    def forward(self, inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, event_idx=0):
        """
        :param inputs: [B, L]
        :param att_mask: [B, L]
        :param word_mask1d: [B, L]
        :param word_mask2d: [B, L, L]
        :param span_labels: [B, L, L, 2], [..., 0] is trigger span label, [..., 1] is argument span label
        :param tri_labels: [B, L, L, C]
        :param event_mask: [B, L]
        :param prob: float (0 - 1)
        :return:
        """
        outputs = {}


        L = word_mask1d.size(1)

        bert_embs = self.bert(input_ids=inputs, attention_mask=att_mask)  #
        bert_embs = bert_embs[0]

        B, _, H = bert_embs.size()

        bert_embs = bert_embs[:, 1:1 + L]

        B, L, H = bert_embs.size()

        eve_embs = self.eve_embedding(torch.arange(self.event_num).long().cuda())
        eve_embs = eve_embs[None, ...].expand(B, -1, -1)

        if self.training:
            # x = bert_embs
            # y = self.eve_embedding(torch.LongTensor([event_idx]).cuda())[None, ...].expand(B, L, -1)
            # cond_bert_embs = self.gate(x, y)
            #
            # drop_bert_embs = self.dropout(cond_bert_embs)
            #
            # span_reps = self.span_linear(drop_bert_embs).view(B, L, 1, self.inner_dim * 4)
            #
            # tri_qw, tri_kw, arg_qw, arg_kw = torch.chunk(span_reps, 4, dim=-1)
            #
            # tri_logits = self._pointer(tri_qw, tri_kw, word_mask2d).permute(0, 2, 3, 1).squeeze()
            # arg_logits = self._pointer(arg_qw, arg_kw, word_mask2d).permute(0, 2, 3, 1).squeeze()

            x = bert_embs
            y = self.eve_embedding(event_idx)
            # y = torch.cat([y, g_eve_embs], dim=2)
            cond_bert_embs = self.gate(x, y, eve_embs)
            # cond_arg_embs = self.arg_gate(x, y)
            # cond_role_embs = self.role_gate(x, y)
            # drop_tri_embs = cond_bert_embs
            drop_tri_embs = self.dropout(cond_bert_embs)
            # drop_arg_embs = self.dropout(cond_bert_embs)
            # drop_role_embs = self.dropout(cond_bert_embs)

            tri_reps = self.tri_linear(drop_tri_embs).view(B, L, -1, self.tri_hid_size * 2)
            tri_qw, tri_kw = torch.chunk(tri_reps, 2, dim=-1)
            arg_reps = self.arg_linear(drop_tri_embs).view(B, L, -1, self.arg_hid_size * 2)
            arg_qw, arg_kw = torch.chunk(arg_reps, 2, dim=-1)

            tri_logits = self._pointer(tri_qw, tri_kw, word_mask2d).permute(0, 2, 3, 1)
            arg_logits = self._pointer(arg_qw, arg_kw, word_mask2d).permute(0, 2, 3, 1)

            role_reps = self.role_linear(drop_tri_embs).view(B, L, -1, self.eve_hid_size * 2)
            role_qw, role_kw = torch.chunk(role_reps, 2, dim=-1)

            # role_qw = self.role_linear1(drop_tri_embs).view(B, L, -1, self.eve_hid_size)
            # role_kw = self.role_linear2(drop_arg_embs).view(B, L, -1, self.eve_hid_size)
            role_logits = self._pointer(role_qw, role_kw, triu_mask2d).permute(0, 2, 3, 1).view(B, L, L, -1,
                                                                                                self.role_num)

            return tri_logits, arg_logits, role_logits
        else:
            x = bert_embs
            y = self.eve_embedding(torch.LongTensor([i for i in range(self.event_num)]).cuda()).unsqueeze(0).expand(B, -1, -1)
            # y = torch.cat([y, g_eve_embs], dim=2)
            cond_bert_embs = self.gate(x, y, eve_embs)
            # cond_arg_embs = self.arg_gate(x, y)
            # cond_role_embs = self.role_gate(x, y)
            # cond_tri_embs = torch.cat([cond_tri_embs, bert_embs.unsqueeze(2)], dim=2)
            # cond_arg_embs = torch.cat([cond_arg_embs, bert_embs.unsqueeze(2)], dim=2)
            # cond_bert_embs = torch.cat([cond_bert_embs, bert_embs.unsqueeze(2)], dim=2)

            # drop_tri_embs = self.dropout(cond_tri_embs)

            # span_reps = self.tri_linear(drop_tri_embs).view(B, L, -1, self.inner_dim * 4)
            #
            # tri_qw, tri_kw, arg_qw, arg_kw = torch.chunk(span_reps, 4, dim=-1)
            # drop_tri_embs = cond_bert_embs
            drop_tri_embs = self.dropout(cond_bert_embs)
            # drop_arg_embs = self.dropout(cond_bert_embs)
            # drop_role_embs = self.dropout(cond_bert_embs)

            tri_reps = self.tri_linear(drop_tri_embs).view(B, L, -1, self.tri_hid_size * 2)
            tri_qw, tri_kw = torch.chunk(tri_reps, 2, dim=-1)
            arg_reps = self.arg_linear(drop_tri_embs).view(B, L, -1, self.arg_hid_size * 2)
            arg_qw, arg_kw = torch.chunk(arg_reps, 2, dim=-1)

            tri_logits = self._pointer(tri_qw, tri_kw, word_mask2d).permute(0, 2, 3, 1)
            arg_logits = self._pointer(arg_qw, arg_kw, word_mask2d).permute(0, 2, 3, 1)

            role_reps = self.role_linear(drop_tri_embs).view(B, L, -1, self.eve_hid_size * 2)
            role_qw, role_kw = torch.chunk(role_reps, 2, dim=-1)

            # role_qw = self.role_linear1(drop_tri_embs).view(B, L, -1, self.eve_hid_size)
            # role_kw = self.role_linear2(drop_arg_embs).view(B, L, -1, self.eve_hid_size)
            role_logits = self._pointer(role_qw, role_kw, triu_mask2d).permute(0, 2, 3, 1).view(B, L, L, -1,
                                                                                                self.role_num)

            # tri_g_logits = tri_logits[..., -1]
            # arg_g_logits = arg_logits[..., -1]
            #
            # tri_logits = tri_logits[..., :-1]
            # arg_logits = arg_logits[..., :-1]

            # tri_g_b_index, tri_g_x_index, tri_g_y_index = ((tri_g_logits > 0).long() + word_mask2d.long()).eq(2).nonzero(as_tuple=True)
            # arg_g_b_index, arg_g_x_index, arg_g_y_index = ((arg_g_logits > 0).long() + word_mask2d.long()).eq(
            #     2).nonzero(as_tuple=True)

            tri_b_index, tri_x_index, tri_y_index, tri_e_index = ((tri_logits > 0).long() + word_mask2d[..., None].long()).eq(2).nonzero(as_tuple=True)  # trigger index

            arg_b_index, arg_x_index, arg_y_index, arg_e_index = ((arg_logits > 0).long() + word_mask2d[..., None].long()).eq(2).nonzero(as_tuple=True)  # trigger index

            role_b_index, role_x_index, role_y_index, role_e_index, role_r_index = (role_logits > 0).nonzero(
                as_tuple=True)

            # tri_g_b_index = torch.cat([tri_g_b_index, tri_b_index], dim=0)
            # tri_g_x_index = torch.cat([tri_g_x_index, tri_x_index], dim=0)
            # tri_g_y_index = torch.cat([tri_g_y_index, tri_y_index], dim=0)

            outputs["ti"] = torch.cat([x.unsqueeze(-1) for x in [ tri_b_index, tri_x_index, tri_y_index]],
                                      dim=-1).cpu().numpy()

            outputs["tc"] = torch.cat([x.unsqueeze(-1) for x in [tri_b_index, tri_x_index, tri_y_index, tri_e_index]],
                                      dim=-1).cpu().numpy()

            outputs["ai"] = torch.cat([x.unsqueeze(-1) for x in [arg_b_index, arg_x_index, arg_y_index, arg_e_index]],
                                      dim=-1).cpu().numpy()

            # outputs["ac"] = None
            outputs["ac"] = torch.cat([x.unsqueeze(-1) for x in [role_b_index, role_y_index, role_e_index, role_r_index]],
                                      dim=-1).cpu().numpy()

            # outputs["as"] = torch.cat([x.unsqueeze(-1) for x in [arg_g_b_index, arg_g_x_index, arg_g_y_index]],
            #                           dim=-1).cpu().numpy()



            return outputs