# coding=utf-8
import re
import time

import json
import os
import torch
import numpy as np
import random

from nltk import sent_tokenize


def sentence_token_nltk(str):
    sent_tokenize_list = sent_tokenize(str)
    return sent_tokenize_list


def batch_accuracy(pre, label):
    pre = pre.argmax(dim=1)
    correct = torch.eq(pre, label).sum().float().item()
    accuracy = correct / float(len(label))
    return accuracy


def save_dataset(path, dataset):
    with open(path, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(sample + '\n')


def read_dataset(path, formatted=True):
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if formatted:
        return dataset
    else:
        data_list = dataset['data']
        formatted_data_list = []
        for orig_item in data_list:
            _document = orig_item['article']
            for i in range(len(orig_item['questions'])):
                _id = orig_item['id'] + '-' + str(i)
                _question = orig_item['questions'][i]
                _answer = orig_item['answers'][i]
                _evidence = orig_item['evidences'][i]
                _option = orig_item['options'][i]  # list
                formatted_item = {'id': _id, 'document': _document, 'question': _question,
                                  'option_list': _option, 'answerKey': _answer, 'evidence': _evidence}
                formatted_data_list.append(formatted_item)
        return formatted_data_list


def save_model(model, output_model_file, optimizer=None):
    if optimizer is None:
        torch.save(model.state_dict(), output_model_file)
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_model_file)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu


def kmp(haystack, needle):
    m, n = len(haystack), len(needle)
    # step1: compute next array
    i, j = 0, 1
    next = [0] * n
    while j < n:
        if needle[i] == needle[j]:  # 如果相等比较好理解，直接更新next数组
            next[j] = i + 1
            i += 1
            j += 1
        elif i > 0:  # 如果不相等的话，需要将i跳转到next[i-1]，但是如果i=0的话，则不存在next[i-1]，需要另外判断
            i = next[i - 1]
        else:  # 如果i=0，needle[i]!=needle[j]，说明只有j需要一直往后+1
            j += 1
    # step2: kmp search
    i, j = 0, 0
    while i < m and j < n:
        if haystack[i] == needle[j]:
            i += 1
            j += 1
        elif j > 0:
            j = next[j - 1]
        else:
            i += 1
    if j == n:
        return i - j, i - 1
    return 0, 0


def get_token_span(src, tgt):
    # tokenize后的
    span_start, span_end = kmp(src, tgt)
    if span_start == span_end and len(tgt) > 2:
        span_start, span_end = kmp(src, tgt[1:-1])
        # if span_end != 0:
        #     print("到这就说明我的这个改动有用")
        span_start -= 1
        span_end += 1
    return span_start, span_end


def get_orig_span(orig_text, src_tokens, input_start, input_end, tokenizer):
    try:
        space_index = []  # 记录空白符位置
        for i, c in enumerate(orig_text):
            if re.match(r'\s', c) is not None:
                space_index.append(i)
        doc_tokens = tokenizer.tokenize(orig_text)

        tok_to_orig_index = []

        accum = 0  # accum 记录长度，最后相等就是对了
        for (i, token) in enumerate(doc_tokens):
            if token == '[UNK]':
                tok_to_orig_index.append(accum)
                accum += 1
            elif re.match(r'^##', token) is not None:
                tok_to_orig_index.append(accum)
                accum += len(token) - 2
            else:
                tok_to_orig_index.append(accum)
                accum += len(token)
        tok_to_orig_index.append(accum)

        temp = 0  # 即后边的数要加上的
        space_i = 0
        for i in range(len(tok_to_orig_index)):
            # tok_to_orig_index[i] += temp
            if space_i == len(space_index):
                tok_to_orig_index[i] += temp
                continue
            if space_index[space_i] <= tok_to_orig_index[i] + temp:
                space_i += 1
                temp += 1
            tok_to_orig_index[i] += temp
        # print(input_start, input_end, len(tok_to_orig_index))
        input_start = min(input_start, len(tok_to_orig_index) - 1)
        input_end = min(input_end, len(tok_to_orig_index) - 1)
        new_start = tok_to_orig_index[input_start]
        new_end = tok_to_orig_index[input_end]
        return orig_text[new_start: new_end].strip()
    except IndexError:
        # 接受的是convert to string后的
        print("有些问题")
        tgt_tokens = src_tokens[input_start:input_end + 1]
        tgt_str = tokenizer.convert_tokens_to_string(tgt_tokens)
        return clean_space(tgt_str)


def clean_space(text):
    """"
    处理中文多余的空格
    """
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list, key=lambda i: len(i), reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i, new_i)
    return text


def cut_sent_fine(para):
    # 细粒度分割
    para = re.sub('([，；。！？\!\?\.])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([，；。！？\!\?\.][”’])([^，；。！？\!\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    fine_sent_list = para.split("\n")
    return fine_sent_list

# def cut_sent(para, max_len=50):
#     para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
#     para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
#     para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
#     para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
#     # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
#     para = para.rstrip()  # 段尾如果有多余的\n就去掉它
#     # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    
#     coarse_sent_list = para.split("\n")
#     new_sent_list = []
#     for sent in coarse_sent_list:
#         if len(sent) < max_len:
#             new_sent_list.append(sent)
#         else:
#             new_sent_list += cut_sent_fine(sent) 
    
#     if len(new_sent_list) > 1:
#         return new_sent_list
    
#     fine_sent_list = cut_sent_fine(para)
#     mid = len(fine_sent_list) // 2
#     left = "".join(fine_sent_list[:mid])
#     right = "".join(fine_sent_list[mid:])
#     return [left, right]
def cut_sent(para, max_len=80, min_len=5):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？!\?][”’])([^，。！？!\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    
    coarse_sent_list = para.split("\n")
    
    new_sent_list = []
    for sent in coarse_sent_list:
        if len(sent) < max_len:
            new_sent_list.append(sent)
        else:
            fine_cut = cut_sent_fine(sent)
            if len(fine_cut) > 1:
                for i in range(len(fine_cut) - 1):
                    if len(fine_cut[i]) < min_len:
                        fine_cut[i + 1] = fine_cut[i] + fine_cut[i + 1]
                        fine_cut[i] = ''
                # 去掉''
                new_cut = [s for s in fine_cut if len(s) > 0]
                    
            else:
                new_cut = fine_cut
                        
            new_sent_list += new_cut
    
    # new_sent_list = []
    # for sent in coarse_sent_list:
    #     if len(sent) < max_len:
    #         new_sent_list.append(sent)
    #     else:
    #         new_sent_list += cut_sent_fine(sent) 
    
    # if len(coarse_sent_list) > 1:
    #     return coarse_sent_list
    
    # fine_sent_list = cut_sent_fine(para)
    # mid = len(fine_sent_list) // 2
    # left = "".join(fine_sent_list[:mid])
    # right = "".join(fine_sent_list[mid:])
    # return [left, right]
    return new_sent_list


def merge_data(official_data_file, new_evidences, save_path=None):
    data_list = json.load(open(official_data_file, 'r', encoding='utf8'))['data']
    formatted_data_list = []
    for orig_item in data_list:
        _document = orig_item['article']
        for i in range(len(orig_item['questions'])):
            _id = orig_item['id'] + '-' + str(i)
            _question = orig_item['questions'][i]
            _answer = orig_item['answers'][i]
            if _id in new_evidences:
                _evidence = new_evidences[_id]['evidence']
            else:
                print("RNMD这不可能发生")
                _evidence = orig_item['evidences'][i]
            _option = orig_item['options'][i]  # list
            formatted_item = {'id': _id, 'document': _document, 'question': _question,
                              'option_list': _option, 'answerKey': _answer, 'evidence': _evidence}
            formatted_data_list.append(formatted_item)
    if save_path is not None:
        print("保存融合新证据后的数据")
        json.dump(formatted_data_list, open(save_path, 'w', encoding='utf8'), indent=2, ensure_ascii=False)
    return formatted_data_list


def get_erased_input(source_ids, source_mask, start_pos, end_pos,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """ 输入原序列和证据起止位置，擦除证据，擦除的token都替换为[MASK]，返回[CLS]DE[SEP]QO[SEP]的拼接
    source: 原序列ids
    doc_lens: 原序列中文档长度
    """
    erased_ids, erased_mask = None, None

    if start_pos >= end_pos:
        erased_ids = source_ids
        erased_mask = source_mask
    else:
        mask_span = torch.tensor([0] * (end_pos - start_pos + 1), dtype=torch.long).to(device)
        mask_mask = torch.tensor([0] * (end_pos - start_pos + 1), dtype=torch.long).to(device)
        deqo_id = torch.cat((source_ids[:start_pos], mask_span, source_ids[end_pos + 1:]), dim=0)
        deqo_ma = torch.cat((source_mask[:start_pos], mask_mask, source_mask[end_pos + 1:]), dim=0)
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
        erased_ids = deqo_id
        erased_mask = deqo_ma

    return erased_ids, erased_mask


if __name__ == "__main__":
    new_evidences = json.load(open("pseudo_evidence.json", 'rb'))
    new_dict = {}
    for item in new_evidences:
        new_dict[item['id']] = {'evidence': item['evidence']}
    merge_data("../data/c3/train-pseudo-c3.json", new_evidences=new_dict, save_path="./results/train_merged.json")