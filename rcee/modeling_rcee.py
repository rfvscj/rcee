# coding=utf-8
import math
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertConfig
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0")


class RNCE(nn.Module):
    """

    """
    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, positive_key, negative_keys=None):
        return self.rnce(query, positive_key, negative_keys,
                         temperature=self.temperature,
                         reduction=self.reduction)

    def rnce(self, h_evid, h_orig, h_rand, temperature=0.1, reduction='mean'):
        # Check input dimensionality.
        if h_evid.dim() != 2:
            raise ValueError('<h_evid> must have 2 dimensions.')
        if h_orig.dim() != 2:
            raise ValueError('<h_orig> must have 2 dimensions.')
        if h_rand.dim() != 3:
            raise ValueError("<h_rand> must have 3 dimensions.")
        # Check matching number of samples.
        if len(h_evid) != len(h_orig):
            raise ValueError('<h_evid> and <h_orig> must must have the same number of samples.')
        if len(h_evid) != len(h_rand):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Normalize to unit vectors
        h_orig, h_evid, h_rand = self.normalize(h_orig, h_evid, h_rand)

        negative_logit = torch.sum(h_orig * h_evid, dim=1, keepdim=True)

        # Cosine between positive pairs
        h_orig = h_orig.unsqueeze(1)
        positive_logits = h_orig @ self.transpose(h_rand)
        positive_logits = positive_logits.squeeze(1)  # [batch, k]

        # First index in last dimension are the positive samples
        logits = 1 - torch.cat([negative_logit, positive_logits], dim=1)
        # print(logits)
        labels = torch.zeros(len(logits), dtype=torch.long, device=h_evid.device)  # 这里的意思是，正样本全在第一个

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)
    
    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask, ignore_cls=True):
        if ignore_cls:
            last_hidden_state = last_hidden_state[:, 1:]
            attention_mask = attention_mask[:, 1:]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return mean_embeddings


class RCEE(nn.Module):
    def __init__(self, model_path, alpha=0.2, beta=0.3, init=False):
        super(RCEE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if init:
            config = BertConfig.from_pretrained(model_path)
            self.bert = BertModel(config)
        else:
            self.bert = BertModel.from_pretrained(model_path)
        self.dim = self.bert.config.hidden_size
        self.trans_layer = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.dim * 4, self.dim),
        )
        self.mean = MeanPooling()
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(p=0.1)
        self.linear_span = nn.Linear(self.dim, 2)
        self.option_linear = nn.Linear(self.dim, 1)
        self.rnce = RNCE(temperature=0.1)
        self.criterion = nn.CrossEntropyLoss()  # 本身是实现了softmax的，不需要额外softmax

    def forward(self, dqo_ids, dqo_mask, choice_num, answers=None, pseudo_start=None, pseudo_end=None,
                rand_starts=None, rand_ends=None, doc_lens=None, super_start_list=None, super_end_list=None,
                order_index=None):
        bert_output = self.bert(dqo_ids, attention_mask=dqo_mask)
        pooler_dqo = bert_output[1]
        state_dqo = bert_output[0]
        # o_mask = 1 - dqo_mask.unsqueeze(dim=-1)
        # filtered_state = state_dqo.masked_fill(o_mask.bool(), torch.tensor(0, device=device))
        state_mean = self.mean(state_dqo, dqo_mask)
        trans_output = self.trans_layer(state_mean)
        h_dqo = self.norm(pooler_dqo + self.dropout(trans_output))
        opt_score = self.option_linear(h_dqo).view(-1, choice_num)
        if answers is None:
            the_ans = torch.argmax(opt_score, dim=1)
            # 排序，从大到小, 测试时能否也干掉假证据？
            _, sorted_index = torch.sort(opt_score, dim=1, descending=True)
            assert sorted_index.shape[0] == opt_score.shape[0]
            assert sorted_index.shape[1] == opt_score.shape[1]
        else:  # 训练时或更新时直接用正确选项
            the_ans = answers
            _, _sorted_index = torch.sort(opt_score, dim=1, descending=True)
            # 移除正确项的index
            sorted_index = []
            for b in range(_sorted_index.shape[0]):
                temp = [answers[b]]  # 正确项在第一个
                for i in range(choice_num):
                    if _sorted_index[b][i] != answers[b]:
                        temp.append(_sorted_index[b][i])
                sorted_index.append(temp)
            sorted_index = torch.tensor(sorted_index, dtype=torch.long).to(device)
            assert sorted_index.shape[0] == opt_score.shape[0]
            assert sorted_index.shape[1] == opt_score.shape[1]
        # have a rest，到这里只是对选项按正误/得分排了个序

        state_dqo = state_dqo.view(-1, choice_num, state_dqo.shape[-2], state_dqo.shape[-1])
        # 这是正确项/得分最高项对应的dqo连接，用这个去预测证据区间
        qa_pooler, qa_state = [], []
        pooler_dqo = pooler_dqo.view(-1, choice_num, pooler_dqo.shape[-1])
        for b in range(the_ans.shape[0]):
            qa_state.append(state_dqo[b, the_ans[b]].unsqueeze(dim=0))
            qa_pooler.append(pooler_dqo[b, the_ans[b]].unsqueeze(dim=0))
        qa_state = torch.cat(qa_state, dim=0)
        qa_pooler = torch.cat(qa_pooler, dim=0)
        # TODO 这里如何搞
        start_logits, end_logits = self.linear_span(qa_state).split(1, dim=-1)  # 注意这个操作
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        ignored_index = start_logits.size(1)

        # 从start_logits和end_logits中找出长度大于临界值的start和end
        can_starts, start_idxs = torch.topk(start_logits, dim=1, k=2)
        can_ends, end_idxs = torch.topk(end_logits, dim=1, k=2)
        start_p, end_p = [], []
        for b in range(start_logits.shape[0]):
            can_spans = torch.cartesian_prod(start_idxs[b], end_idxs[b])
            _start, _end = can_spans[0][0], can_spans[0][1]
            for i in range(can_spans.shape[0]):
                if can_spans[i][1] - can_spans[i][0] > 3:
                    _start, _end = can_spans[i][0], can_spans[i][1]
                    break
            start_p.append(_start.unsqueeze(dim=0))
            end_p.append(_end.unsqueeze(dim=0))
        start_p = torch.cat(start_p, dim=0)
        end_p = torch.cat(end_p, dim=0)

        # 测试时
        if answers is None:
            return opt_score, start_p, end_p

        else:  # 训练时
            nega_nums = rand_starts.shape[-1]
            # TODO 之后可否删掉？ 初始化为pseudo_span
            for i in range(len(order_index)):
                if super_start_list[order_index[i]] == -1:
                    super_start_list[order_index[i]] = pseudo_start[i].item()
                if super_end_list[order_index[i]] == -1:
                    super_end_list[order_index[i]] = pseudo_end[i].item()
            
            super_starts = [super_start_list[index] for index in order_index]
            super_ends = [super_end_list[index] for index in order_index]
            super_starts = torch.tensor(super_starts, dtype=torch.long).to(device)
            super_ends = torch.tensor(super_ends, dtype=torch.long).to(device)
            start_positions = super_starts.clamp(0, ignored_index)
            end_positions = super_ends.clamp(0, ignored_index)
            # print(start_positions, end_positions)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index, label_smoothing=0.2)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            evidence_loss = (start_loss + end_loss) / 2
            loss = self.criterion(opt_score, answers)

            _dqo_ids = dqo_ids.view(-1, choice_num, dqo_ids.shape[-1])
            _dqo_mask = dqo_mask.view(-1, choice_num, dqo_mask.shape[-1])
            _doc_lens = doc_lens.view(-1, choice_num)
            sorted_ids, sorted_mask, sorted_doc_lens = [], [], []
            for b in range(the_ans.shape[0]):
                sorted_ids.append(torch.cat([_dqo_ids[b, sorted_index[b][i].unsqueeze(dim=0)]
                                             for i in range(choice_num)], dim=0).unsqueeze(dim=0))
                sorted_mask.append(torch.cat([_dqo_mask[b, sorted_index[b][i].unsqueeze(dim=0)]
                                              for i in range(choice_num)], dim=0).unsqueeze(dim=0))
                sorted_doc_lens.append(torch.cat([_doc_lens[b, sorted_index[b][i].unsqueeze(dim=0)]
                                                  for i in range(choice_num)], dim=0).unsqueeze(dim=0))
            sorted_ids = torch.cat(sorted_ids, dim=0)
            sorted_mask = torch.cat(sorted_mask, dim=0)
            sorted_doc_lens = torch.cat(sorted_doc_lens, dim=0)

            # 对正确QA-pair随机擦除，取dr
            pooler_dr_list, state_dr_list = [], []
            trans_dr_list = []
            # print(sorted_ids.shape, rand_starts.shape)
            for i in range(nega_nums):
                dr_ids, dr_mask = self.get_deqo(sorted_ids[:, 0], sorted_mask[:, 0], sorted_doc_lens[:, 0],
                                                rand_starts[:, i], rand_ends[:, i])
                out = self.bert(dr_ids, attention_mask=dr_mask)
                # r_mask = 1 - dr_mask.unsqueeze(dim=-1)
                
                # filtered_dr = out[0].masked_fill(r_mask.bool(), torch.tensor(0, device=device))
                dr_mean = self.mean(out[0], dr_mask)
                trans_dr = self.trans_layer(dr_mean)
                pooler_dr_list.append(out[1])  # 4 * [batch, hid_dim]
                state_dr_list.append(out[0])
                trans_dr_list.append(trans_dr)

            h_temp = trans_output.detach().view(the_ans.shape[0], choice_num, -1)
            h_orig = pad_sequence([h_temp[b, the_ans[b]] for b in range(h_temp.shape[0])], batch_first=True)

            # 擦掉证据，以监督的为准，仅优化这个
            deqo_ids, deqo_mask = self.get_deqo(sorted_ids[:, 0], sorted_mask[:, 0],
                                                sorted_doc_lens[:, 0], start_positions, end_positions)
            bert_output = self.bert(deqo_ids, attention_mask=deqo_mask)
            pooler_de = bert_output[1]
            state_de = bert_output[0]
            # e_mask = 1 - deqo_mask.unsqueeze(dim=-1)
            # filtered_state_de = state_de.masked_fill(e_mask.bool(), torch.tensor(0, device=device))
            de_mean = self.mean(state_de, deqo_mask)
            h_evid = self.trans_layer(de_mean)
            # [k, ]
            pooler_dr = torch.cat([p.unsqueeze(dim=1) for p in pooler_dr_list], dim=1)
            state_dr = torch.cat([s.unsqueeze(dim=1) for s in state_dr_list], dim=1)
            trans_drs = torch.cat([t.unsqueeze(dim=1) for t in trans_dr_list], dim=1)
            # filtered_state_dr = []
            # for ni in range(nega_nums):
            #     s = state_dr_list[ni]
            #     filtered_s = pad_sequence(
            #         [torch.mean(torch.cat([s[b, :rand_starts[b][ni]], s[b, rand_ends[b][ni] + 1:]], dim=0), dim=0)
            #          for b in range(s.shape[0])], batch_first=True
            #     )
            #     filtered_state_dr.append(filtered_s.unsqueeze(dim=1))
            # filtered_state_dr = torch.cat(filtered_state_dr, dim=1)
                
            # state_dr = torch.cat([torch.mean(s, dim=1).unsqueeze(dim=1) for s in state_dr_list], dim=1).view(-1, nega_nums, self.dim)
            # filtered_state_dr = torch.cat(state_dr_list, dim=1)
            h_rand = trans_drs

            loss_con = self.rnce(h_evid, h_orig, h_rand)

            return loss + self.alpha * loss_con + self.beta * evidence_loss, loss, loss_con, evidence_loss

    
    # 从头来
    @torch.no_grad()
    def update_evidence(self, dqo_ids, dqo_mask, choice_num, answers=None, pseudo_start=None, pseudo_end=None,
                        doc_lens=None, super_start_list=None, super_end_list=None, base_len_list=None, order_index=None, momentum=0.8):
        bert_output = self.bert(dqo_ids, attention_mask=dqo_mask)
        pooler_dqo = bert_output[1]
        state_dqo = bert_output[0]
        state_mean = self.mean(state_dqo, dqo_mask)
        trans_output = self.trans_layer(state_mean)  # 这个阶段不加dropout
        h_dqo = self.norm(pooler_dqo + self.dropout(trans_output))
        opt_score = self.option_linear(h_dqo).view(-1, choice_num)
        the_ans = answers
        _, _sorted_index = torch.sort(opt_score, dim=1, descending=True)
        # 移除正确项的index
        sorted_index = []
        for b in range(_sorted_index.shape[0]):
            temp = [answers[b]]  # 正确项在第一个
            for i in range(choice_num):
                if _sorted_index[b][i] != answers[b]:
                    temp.append(_sorted_index[b][i])
            sorted_index.append(temp)
        sorted_index = torch.tensor(sorted_index, dtype=torch.long).to(device)
        assert sorted_index.shape[0] == opt_score.shape[0]
        assert sorted_index.shape[1] == opt_score.shape[1]
        # have a rest，到这里只是对选项按正误/得分排了个序

        state_dqo = state_dqo.view(-1, choice_num, state_dqo.shape[-2], state_dqo.shape[-1])
        # 这是正确项/得分最高项对应的dqo连接，用这个去预测证据区间
        ans_state = []
        for b in range(the_ans.shape[0]):
            ans_state.append(state_dqo[b, the_ans[b], :, :].unsqueeze(dim=0))
        ans_state = torch.cat(ans_state, dim=0)
        start_logits, end_logits = self.linear_span(ans_state).split(1, dim=-1)  # 注意这个操作
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        ignored_index = start_logits.size(1)
        # 从start_logits和end_logits中找出长度大于临界值的start和end
        can_starts, start_idxs = torch.topk(start_logits, dim=1, k=3)
        can_ends, end_idxs = torch.topk(end_logits, dim=1, k=3)
        valid_can_spans = []
        for b in range(start_logits.shape[0]):
            valid_sp = []
            _tail = []  # 用于把不合法区间放最后
            can_spans = torch.cartesian_prod(start_idxs[b], end_idxs[b])
            # 如此保证最少有一个
            _start, _end = can_spans[0][0], can_spans[0][1]
            # valid_sp.append((_start, _end))
            for i in range(1, can_spans.shape[0]):
                if can_spans[i][1] - can_spans[i][0] > 3:
                    _start, _end = can_spans[i][0], can_spans[i][1]
                    valid_sp.append((_start, _end))
                else:
                    _tail.append((1, 1))
            # start_p.append(valid_sp[0][0])
            # end_p.append(valid_sp[0][1])
            valid_sp += _tail
            valid_sp = valid_sp[:1]  # 固定为3个，节省算力
            valid_can_spans.append(valid_sp)
        # start_p = torch.cat(start_p, dim=0)
        # end_p = torch.cat(end_p, dim=0)

        # 初始化
        for i in range(len(order_index)):
            if super_start_list[order_index[i]] == -1:
                super_start_list[order_index[i]] = pseudo_start[i].item()
            if super_end_list[order_index[i]] == -1:
                super_end_list[order_index[i]] = pseudo_end[i].item()
            if base_len_list[order_index[i]] == -1:
                base_len_list[order_index[i]] = max(pseudo_end[i].item() - pseudo_start[i].item(), 0)
                
        super_starts = [super_start_list[index] for index in order_index]
        super_ends = [super_end_list[index] for index in order_index]
        super_starts = torch.tensor(super_starts, dtype=torch.long).to(device)
        super_ends = torch.tensor(super_ends, dtype=torch.long).to(device)
        start_positions = super_starts.clamp(0, ignored_index)
        end_positions = super_ends.clamp(0, ignored_index)

        # TODO, 需给出评分最高的答案的ids和mask，不能是new_dqo
        _dqo_ids = dqo_ids.view(-1, choice_num, dqo_ids.shape[-1])
        _dqo_mask = dqo_mask.view(-1, choice_num, dqo_mask.shape[-1])
        _doc_lens = doc_lens.view(-1, choice_num)

        sorted_ids, sorted_mask, sorted_doc_lens = [], [], []
        for b in range(the_ans.shape[0]):
            sorted_ids.append(torch.cat([_dqo_ids[b, sorted_index[b][i].unsqueeze(dim=0)]
                                         for i in range(choice_num)], dim=0).unsqueeze(dim=0))
            sorted_mask.append(torch.cat([_dqo_mask[b, sorted_index[b][i].unsqueeze(dim=0)]
                                          for i in range(choice_num)], dim=0).unsqueeze(dim=0))
            sorted_doc_lens.append(torch.cat([_doc_lens[b, sorted_index[b][i].unsqueeze(dim=0)]
                                              for i in range(choice_num)], dim=0).unsqueeze(dim=0))
        sorted_ids = torch.cat(sorted_ids, dim=0)
        sorted_mask = torch.cat(sorted_mask, dim=0)
        sorted_doc_lens = torch.cat(sorted_doc_lens, dim=0)

        cur_pooler_list, cur_state_list, cur_transs = [], [], []
        pred_pooler_list, pred_state_list, pred_transs = [], [], []
        for i in range(choice_num):
            cur_erased_ids, cur_erased_mask = self.get_deqo(sorted_ids[:, i], sorted_mask[:, i], sorted_doc_lens[:, i],
                                                            start_positions, end_positions)
            # pred_erased_ids, pred_erased_mask = self.get_deqo(sorted_ids[:, i], sorted_mask[:, i], sorted_doc_lens[:, i],
            #  batch, k, seq                                                 start_p, end_p)
            can_erased_ids, can_erased_mask = self.get_deqo_ext(sorted_ids[:, i], sorted_mask[:, i], sorted_doc_lens[:, i],
                                                                valid_can_spans)
            cur_out = self.bert(cur_erased_ids, attention_mask=cur_erased_mask)
            
            b_pooler, b_state, b_trans = [], [], []
            
            for b in range(len(cur_erased_ids)):
                # 里边相当于以候选证据条数作为batch了
                can_out = self.bert(can_erased_ids[b], attention_mask=can_erased_mask[b])
                b_pooler.append(can_out[1])
                b_state.append(can_out[0])
                can_mean = self.mean(can_out[0], can_erased_mask[b])
                can_trans = self.trans_layer(can_mean)
                b_trans.append(can_trans)
            
            b_pooler = pad_sequence(b_pooler, batch_first=True)
            b_state = pad_sequence(b_state, batch_first=True)
            b_trans = pad_sequence(b_trans, batch_first=True)
                 
            # pred_out = self.bert(pred_erased_ids, attention_mask=pred_erased_mask)
            cur_pooler_list.append(cur_out[1].unsqueeze(dim=1))
            cur_state_list.append(cur_out[0])
            pred_pooler_list.append(b_pooler.unsqueeze(dim=2))
            pred_state_list.append(b_state)
            cur_mean = self.mean(cur_out[0], cur_erased_mask)
            # pred_mean = self.mean(pred_out[0], pred_erased_mask)
            cur_trans = self.trans_layer(cur_mean)
            # pred_trans = self.trans_layer(pred_mean)
            cur_transs.append(cur_trans.unsqueeze(dim=1))
            # pred_transs.append(pred_trans.unsqueeze(dim=1))
            pred_transs.append(b_trans.unsqueeze(dim=2))


        # 这里把另几个选项只凭证据的得分也算出来，然后用softmax归一化，这样更有理据
        cur_poolers = torch.cat(cur_pooler_list, dim=1)        
        pred_poolers = torch.cat(pred_pooler_list, dim=2)
        

        cur_transs = torch.cat(cur_transs, dim=1)
        pred_transs = torch.cat(pred_transs, dim=2)
        

        cur_h = self.norm(cur_poolers + self.dropout(cur_transs))
        pred_h = self.norm(pred_poolers + self.dropout(pred_transs))

        opt_scores_cur = F.softmax(self.option_linear(cur_h).view(-1, choice_num), dim=1)
        opt_scores_pred = F.softmax(self.option_linear(pred_h).view(opt_scores_cur.shape[0], -1, choice_num), dim=2)

        new_cnt = 0
        for b in range(len(order_index)):

            # 长度惩罚，只惩罚预测的
            base_len = base_len_list[order_index[b]]
            # pred_len = end_p[b] - start_p[b]  # 长度差
            cur_len = super_end_list[order_index[b]] - super_start_list[order_index[b]]
            
            # dif_p = 0.2 * torch.log(pred_len / base_len).clamp(0, 3).item()

            cur_qa_score = opt_scores_cur[b, 0]
            # 先从候选中选最好的
            pm_start, pm_end = 1, 1
            min_score = 10.0
            pred_qa_score = opt_scores_pred[b]
            for ki in range(len(pred_qa_score)):
                can_score = pred_qa_score[ki, 0]
                can_len = valid_can_spans[b][ki][1] - valid_can_spans[b][ki][0]
                _crit = 0.2 * min(max(math.log((can_len + 1) / (base_len + 1)), 0), 3)
                if can_score + _crit < min_score:
                    min_score = can_score + _crit
                    pm_start, pm_end = valid_can_spans[b][ki][0], valid_can_spans[b][ki][1]
            

            if cur_qa_score > min_score + 0.01 and pm_end - pm_start > 3 or cur_len <= 3:
                super_start_list[order_index[b]] = pm_start
                super_end_list[order_index[b]] = pm_end
                # 更新基准长度
                base_len_list[order_index[b]] = momentum * base_len_list[order_index[b]] + (1 - momentum) * (pm_end - pm_start)

            if super_start_list[order_index[b]] != pseudo_start[b] or super_end_list[order_index[b]] != pseudo_end[b]:
                new_cnt += 1
                pass
        return new_cnt

    def get_deqo(self, source_ids, source_mask, doc_lens, start_pos, end_pos):
        """ 输入原序列和证据起止位置，擦除证据，擦除的token都替换为[MASK]，返回[CLS]DE[SEP]QO[SEP]的拼接
        source: 原序列ids
        doc_lens: 原序列中文档长度
        """
        deqo_ids = []
        deqo_mask = []

        for b in range(source_ids.shape[0]):
            if start_pos[b] > end_pos[b]:
                deqo_ids.append(source_ids[b])
                deqo_mask.append(source_mask[b])
            else:
                s = start_pos[b].clamp(0, doc_lens[b] - 1)
                e = end_pos[b].clamp(0, doc_lens[b] - 1)
                mask_span = torch.tensor([0] * (e - s + 1), dtype=torch.long).to(device)
                mask_mask = torch.tensor([0] * (e - s + 1), dtype=torch.long).to(device)
                deqo_id = torch.cat((source_ids[b][:s], mask_span, source_ids[b][e + 1:]), dim=0)
                deqo_ma = torch.cat((source_mask[b][:s], mask_mask, source_mask[b][e + 1:]), dim=0)
                # assert deqo_id.shape[-1] == 512
                if deqo_id[0] != 101:
                    # deqo_id = deqo_id[:doc_lens[b] - 1]  # nmd写错了
                    # deqo_ma = deqo_ma[:doc_lens[b] - 1]
                    deqo_id = deqo_id[:deqo_id.shape[-1] - 1]
                    deqo_ma = deqo_ma[:deqo_ma.shape[-1] - 1]
                    deqo_id = torch.cat((torch.tensor([101]).to(device), deqo_id), dim=0)
                    deqo_ma = torch.cat((torch.tensor([1]).to(device), deqo_ma), dim=0)
                if deqo_id[-1] != 102:
                    deqo_id = deqo_id[:deqo_id.shape[-1] - 1]
                    deqo_ma = deqo_ma[:deqo_ma.shape[-1] - 1]
                    deqo_id = torch.cat((deqo_id, torch.tensor([102]).to(device)), dim=0)
                    deqo_ma = torch.cat((deqo_ma, torch.tensor([1]).to(device)), dim=0)
                # assert deqo_id.shape[-1] == 512
                deqo_ids.append(deqo_id)
                deqo_mask.append(deqo_ma)
        deqo_ids = pad_sequence(deqo_ids, batch_first=True)
        deqo_mask = pad_sequence(deqo_mask, batch_first=True)
        # assert deqo_ids.shape == deqo_mask.shape
        # assert deqo_ids.shape[-1] == 512

        return deqo_ids, deqo_mask
    
    def get_deqo_ext(self, source_ids, source_mask, doc_lens, valid_can_spans):
        """ 
        valid_can_spans: batch size，不定项，(start, end)
        return: list, 不定项
        """
        
        deqo_ids = []
        deqo_mask = []

        for b in range(source_ids.shape[0]):
            k_deqo_ids, k_deqo_mask = [], []
            for ki in range(len(valid_can_spans[b])):
                if valid_can_spans[b][ki][0] > valid_can_spans[b][ki][1]:
                    k_deqo_ids.append(source_ids[b])
                    k_deqo_mask.append(source_mask[b])
                else:
                    s = min(max(valid_can_spans[b][ki][0], 0), doc_lens[b] - 1)
                    e = min(max(valid_can_spans[b][ki][1], 0), doc_lens[b] - 1)
                    mask_span = torch.tensor([0] * (e - s + 1), dtype=torch.long).to(device)
                    mask_mask = torch.tensor([0] * (e - s + 1), dtype=torch.long).to(device)
                    deqo_id = torch.cat((source_ids[b][:s], mask_span, source_ids[b][e + 1:]), dim=0)
                    deqo_ma = torch.cat((source_mask[b][:s], mask_mask, source_mask[b][e + 1:]), dim=0)
                    # assert deqo_id.shape[-1] == 512
                    if deqo_id[0] != 101:
                        # deqo_id = deqo_id[:doc_lens[b] - 1]  # nmd写错了
                        # deqo_ma = deqo_ma[:doc_lens[b] - 1]
                        deqo_id = deqo_id[:deqo_id.shape[-1] - 1]
                        deqo_ma = deqo_ma[:deqo_ma.shape[-1] - 1]
                        deqo_id = torch.cat((torch.tensor([101]).to(device), deqo_id), dim=0)
                        deqo_ma = torch.cat((torch.tensor([1]).to(device), deqo_ma), dim=0)
                    if deqo_id[-1] != 102:
                        deqo_id = deqo_id[:deqo_id.shape[-1] - 1]
                        deqo_ma = deqo_ma[:deqo_ma.shape[-1] - 1]
                        deqo_id = torch.cat((deqo_id, torch.tensor([102]).to(device)), dim=0)
                        deqo_ma = torch.cat((deqo_ma, torch.tensor([1]).to(device)), dim=0)
                    # assert deqo_id.shape[-1] == 512
                    k_deqo_ids.append(deqo_id)
                    k_deqo_mask.append(deqo_ma)
            k_deqo_ids = pad_sequence(k_deqo_ids, batch_first=True)
            k_deqo_mask = pad_sequence(k_deqo_mask, batch_first=True)        
            deqo_ids.append(k_deqo_ids.unsqueeze(dim=0))
            deqo_mask.append(k_deqo_mask.unsqueeze(dim=0))
        # assert deqo_ids.shape == deqo_mask.shape
        # assert deqo_ids.shape[-1] == 512
        deqo_ids = torch.cat(deqo_ids, dim=0)
        deqo_mask = torch.cat(deqo_mask, dim=0)

        return deqo_ids, deqo_mask
