# coding=utf-8
import re
from transformers import BertTokenizer, get_cosine_schedule_with_warmup
from tqdm import trange
import os
import random
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from eval_expmrc import calc_f1, calc_f1_solo
from utils import read_dataset, set_seed, get_orig_span, kmp, cut_sent
from modeling_rcee import RCEE
import json
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 全局
best_dev_f1, best_dev_f1_ans, best_dev_f1_evid, best_f1_evid_only, base_len_list = 0, 0, 0, 0, 0

def get_input_feature(samples, tokenizer, max_len, max_len_qo, choice_num, order_index=None, for_dev=False):
    global super_start_list, super_end_list

    evidences = []
    answers = []
    documents, endings = [], []
    dqo_ids, dqo_mask = [], []
    doc_lens = []
    evidence_starts, evidence_ends = [], []
    rand_starts, rand_ends = [], []
    d_toks = []
    for i, sample in enumerate(samples):
        if 'answerKey' in sample:
            answerKey = sample['answerKey']
        else:
            answerKey = "A"
        question = sample['question']
        while len(sample['option_list']) < choice_num:
            sample['option_list'].append('')
        evidence = sample['evidence']  # emm
        if isinstance(evidence, list):
            evidence = evidence[0]
        elif isinstance(evidence, str):
            evidence = evidence
        else:
            raise TypeError("要么list要么str")
        e_tok = tokenizer.tokenize(evidence)
        document = sample['document']
        d_tok = tokenizer.tokenize(document)
        documents.append(document)
        sent_list = cut_sent(document)

        super_evidence = evidence
        if order_index is not None:
            if super_start_list[order_index[i]] != -1 and super_end_list[order_index[i]] != -1:
                super_evidence = get_orig_span(document, d_tok, super_start_list[order_index[i]] - 1,
                                               super_end_list[order_index[i]], tokenizer)

        # 算个f1最高的作为证据句，去掉，剩下的都是非证据句
        hi, maxf1 = 0, 0
        for i, sent in enumerate(sent_list):
            temp = calc_f1_solo(sent, super_evidence)
            if temp > maxf1:
                maxf1 = temp
                hi = i
        if len(sent_list) != 1:
            del sent_list[hi]
        nums_nega = 3
        if len(sent_list) >= nums_nega:
            rand_sents = random.sample(sent_list, k=nums_nega)
        else:
            rand_sents = random.choices(sent_list, k=nums_nega)

        for opt in sample['option_list']:
            if "_" in question or '▁' in question:
                try:  # 有概率在末尾有反斜杠等特殊情况
                    qa_cat = re.sub('[_▁]+', opt.strip('\\\w'), question)
                except:
                    qa_cat = " ".join([question, opt])
            else:
                qa_cat = " ".join([question, opt])
            endings.append(qa_cat)
            qa_tok = tokenizer.tokenize(qa_cat)
            if len(qa_tok) > max_len_qo:
                qa_tok = qa_tok[-max_len_qo:]
            doc_len = len(d_tok)
            if len(d_tok) + len(qa_tok) > max_len - 3:
                doc_len = max_len - 3 - len(qa_tok)
                d_tok = d_tok[:doc_len]

            dqo_tok = ['[CLS]'] + d_tok + ['[SEP]'] + qa_tok + ['[SEP]']
            dqo_m = [1] * len(dqo_tok)
            dqo_tok = dqo_tok + ['[PAD]'] * (max_len - len(dqo_tok))
            dqo_id = tokenizer.convert_tokens_to_ids(dqo_tok)
            dqo_m = dqo_m + [0] * (max_len - len(dqo_m))
            doc_lens.append(doc_len + 2)  # [CLS]和[SEP]，进入模型后都统一加上偏置

            assert len(dqo_id) == max_len
            assert len(dqo_m) == max_len
            dqo_ids.append(dqo_id)
            dqo_mask.append(dqo_m)
            # 若因分词导致问题，缩一缩证据token
            evidence_start, evidence_end = kmp(d_tok, e_tok)
            if evidence_start == evidence_end and len(e_tok) > 2:
                evidence_start, evidence_end = kmp(d_tok, e_tok[1:-1])
                # if evidence_end != 0:
                #     print("到这就说明我的这个改动有用")
                # evidence_start -= 1
                evidence_end += 1

            evidence_start = min(doc_len - 1, evidence_start)
            evidence_end = min(doc_len - 1, evidence_end)
            es = max(evidence_start, 0)
            ee = min(evidence_end, 511)
        # if ee < es:
        #     print("span:", es, ee, evidence_start, evidence_end, document, '\n', evidence)
        evidence_starts.append(es + 1)  # 加1是偏移一个cls
        evidence_ends.append(ee + 1)

        rand_start_o, rand_end_o = [], []
        for rand_sent in rand_sents:
            r_tok = tokenizer.tokenize(rand_sent)
            rand_start, rand_end = kmp(d_tok, r_tok)
            rand_start_o.append(rand_start + 1)  # 偏移一个cls，还可以细修
            rand_end_o.append(rand_end + 1)

        rand_starts.append(rand_start_o)
        rand_ends.append(rand_end_o)

        if answerKey in '123456':
            answer = ord(answerKey) - ord('1')
        else:
            answer = ord(answerKey) - ord('A')
        answers.append(answer)
        if for_dev:
            d_toks.append(d_tok)
            evidences.append(sample['evidence'])
    dqo_ids = torch.tensor(dqo_ids, dtype=torch.long).to(device)
    dqo_mask = torch.tensor(dqo_mask, dtype=torch.long).to(device)
    doc_lens = torch.tensor(doc_lens, dtype=torch.long).to(device)
    answers = torch.tensor(answers, dtype=torch.long).to(device)
    evidence_starts = torch.tensor(evidence_starts, dtype=torch.long).to(device)
    evidence_ends = torch.tensor(evidence_ends, dtype=torch.long).to(device)
    rand_starts = torch.tensor(rand_starts, dtype=torch.long, device=device)
    rand_ends = torch.tensor(rand_ends, dtype=torch.long, device=device)

    if for_dev:
        return dqo_ids, dqo_mask, answers, evidence_starts, evidence_ends, evidences, d_toks, documents, doc_lens, rand_starts, rand_ends
    return dqo_ids, dqo_mask, answers, evidence_starts, evidence_ends, doc_lens, rand_starts, rand_ends


@torch.no_grad()
def eval(model, test_examples, tokenizer, eval_batch_size, choice_num, max_len, max_len_qo):
    results = {}
    model.eval()
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    preds, evidences = [], []
    p_anss, anss = [], []
    for step in step_trange:
        beg_index = step * eval_batch_size
        end_index = min((step + 1) * eval_batch_size, len(test_examples))
        batch_example = [example for example in test_examples[beg_index:end_index]]
        dqo_ids, dqo_mask, answers, e_start, e_end, evidence, tok_dqs, documents, doc_lens, _, _ = get_input_feature(
            batch_example, tokenizer, max_len,
            max_len_qo,
            args.choice_num,
            for_dev=True)
        scores, start_p, end_p = model(dqo_ids, dqo_mask, choice_num, doc_lens=doc_lens)

        scores = scores.cpu().detach().tolist()
        answers = answers.cpu().detach().tolist()
        start_p = start_p.cpu().detach().tolist()
        end_p = end_p.cpu().detach().tolist()
        batch_e = []

        for p, a, example, s, e, t, d in zip(scores, answers, batch_example, start_p, end_p, tok_dqs, documents):
            p_ans = p.index(max(p))
            p_anss.append(p_ans)
            anss.append(a)
            # e_span = t[s - 1:e]  # 这里有个偏移
            if s + 2 <= e:  # 太短认为证据在末尾
                e_pred = get_orig_span(d, t, s - 1, e, tokenizer)  # TODO 这里要不要-1来着，好像是不用
            else:
                e_pred = d[-len(d) // 4:]  # tricks
            batch_e.append(e_pred)
            qid = example['id']
            answer = chr(ord('A') + p_ans)
            results[qid] = {"answer": answer, "evidence": e_pred}
        preds += batch_e
        evidences += evidence
    # for i in range(10):  # 展示一些实例
    #     print("prediction: ", sources[i], "golden: ", evidences[i])
    f1_all, f1_ans, f1_evid = calc_f1(p_anss, anss, preds, evidences)
    results_save_path = "./results/f1all-" + str(round(f1_all, 4)) + ".json"
    json.dump(results, open(results_save_path, 'w', encoding='utf8'), indent=2, ensure_ascii=False)
    return f1_all, f1_ans, f1_evid


def train(args, model, tokenizer, optimizer, train_examples, dev_examples, episode):
    global best_dev_f1, best_dev_f1_ans, best_dev_f1_evid, best_f1_evid_only  # python 对于字符串和数字，改变参数不影响外边的值
    global best_start_list, best_end_list
    global super_start_list, super_end_list

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    epochs = args.epoch_per_episode if episode < args.episodes - 1 else 5
    # dev_f1, dev_f1_ans, dev_f1_evid = eval(model, dev_examples, tokenizer, args.eval_batch_size, args.choice_num, args.max_len, args.max_len_qo)
    # print('dev_f1:', dev_f1, 'f1_ans:', dev_f1_ans, 'f1_evid:', dev_f1_evid)
    step_count = len(train_examples) // train_batch_size
    if step_count * train_batch_size < len(train_examples):
        step_count += 1
    warm_up_ratio = 0.2
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_ratio * step_count // args.gradient_accumulation_steps,
                                                num_training_steps=step_count * epochs // args.gradient_accumulation_steps)
    for epoch in range(epochs):

        order = list(range(len(train_examples)))
        random.seed(args.seed + episode * 10 + epoch)
        random.shuffle(order)
        model.train()

        step_trange = trange(step_count)
        tr_loss, nb_tr_steps = 0, 0
        tr_loss_a, tr_loss_con, tr_loss_e = 0, 0, 0

        # update(args, model, tokenizer, train_examples, order, epoch)

        for step in step_trange:
            beg_index = step * train_batch_size
            end_index = min((step + 1) * train_batch_size, len(train_examples))
            order_index = order[beg_index:end_index]
            batch_example = [train_examples[index] for index in order_index]
            dqo_ids, dqo_mask, answers, e_start, e_end, doc_lens, rand_starts, rand_ends = get_input_feature(
                batch_example,
                tokenizer=tokenizer,
                max_len=args.max_len,
                max_len_qo=args.max_len_qo,
                choice_num=args.choice_num,
                order_index=order_index
            )
            
            # Runs the forward pass with autocasting.
            with autocast(device_type='cuda', dtype=torch.float16):
                loss, loss_a, loss_con, loss_e = model(dqo_ids, dqo_mask, args.choice_num, answers,
                                                       e_start, e_end, rand_starts, rand_ends, doc_lens,
                                                       super_start_list, super_end_list, order_index)

                loss = loss.mean()
                loss_e = loss_e.mean()
                loss_con = loss_con.mean()
                loss_a = loss_a.mean()
                tr_loss += loss.item()
                tr_loss_e += loss_e.item()
                tr_loss_con += loss_con.item()
                tr_loss_a += loss_a.item()
                nb_tr_steps += 1
                loss = loss / args.gradient_accumulation_steps
                loss_e = loss_e / args.gradient_accumulation_steps
                loss_con = loss_con / args.gradient_accumulation_steps
                loss_a = loss_a / args.gradient_accumulation_steps
            # loss.backward()
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # optimizer.step()
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = (scale > scaler.get_scale())
                if not skip_lr_sched:
                    scheduler.step()
                optimizer.zero_grad()
            loss_show = ' Epoch:' + str(episode) + '-' + str(epoch) + " loss:" + str(
                round(tr_loss / nb_tr_steps, 4)) + " loss_a:" + str(
                round(tr_loss_a / nb_tr_steps, 4)) + " loss_con:" + str(
                round(tr_loss_con / nb_tr_steps, 4)) + " loss_e:" + str(round(tr_loss_e / nb_tr_steps, 4))
            step_trange.set_postfix_str(loss_show)

        dev_f1, dev_f1_ans, dev_f1_evid = eval(model, dev_examples, tokenizer, args.eval_batch_size, args.choice_num,
                                               args.max_len, args.max_len_qo)

        print('dev_f1:', dev_f1, 'f1_ans:', dev_f1_ans, 'f1_evid:', dev_f1_evid)

        # f1_evid更好就更新证据，f1_all更好就保存模型
        if dev_f1 > best_dev_f1:  # 以总分为准还是以证据得分为准？
            best_dev_f1 = dev_f1
            best_dev_f1_ans = dev_f1_ans
            best_dev_f1_evid = dev_f1_evid
            file_name = f'f1_{dev_f1}_ans_{dev_f1_ans}_evid_{dev_f1_evid}_lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}_ga_{args.gradient_accumulation_steps}_alpha_{args.alpha}_beta_{args.beta}.bin'
            output_model_path = os.path.join(args.results_save_path, file_name)
            torch.save(model.state_dict(), output_model_path)
            print('new best dev f1:', dev_f1, 'f1_ans:', dev_f1_ans, 'f1_evid:', dev_f1_evid)

        if dev_f1_evid > best_f1_evid_only:
            best_f1_evid_only = dev_f1_evid
            update(args, model, tokenizer, train_examples, order, epoch)


@torch.no_grad()
def update(args, model, tokenizer, train_examples, order, epoch):
    global best_dev_f1, best_dev_f1_ans, best_dev_f1_evid  # python 对于字符串和数字，改变参数不影响外边的值
    global best_start_list, best_end_list
    global super_start_list, super_end_list
    global base_len_list
    model.eval()
    train_batch_size = args.train_batch_size
    step_count = len(train_examples) // train_batch_size
    if step_count * train_batch_size < len(train_examples):
        step_count += 1
    step_trange = trange(step_count)
    new_cnt = 0
    for step in step_trange:
        beg_index = step * train_batch_size
        end_index = min((step + 1) * train_batch_size, len(train_examples))
        order_index = order[beg_index:end_index]
        batch_example = [train_examples[index] for index in order_index]
        dqo_ids, dqo_mask, answers, e_start, e_end, doc_lens, _, _ = get_input_feature(
            batch_example,
            tokenizer=tokenizer,
            max_len=args.max_len,
            max_len_qo=args.max_len_qo,
            choice_num=args.choice_num,
        )
        batch_new_cnt = model.update_evidence(dqo_ids, dqo_mask, args.choice_num, answers, e_start, e_end, doc_lens,
                                              super_start_list, super_end_list, base_len_list, order_index, args.m)
        new_cnt += batch_new_cnt
        step_trange.set_postfix_str(
            "更新证据中..., 新证据占比: " + str(round(new_cnt / ((step + 1) * args.train_batch_size), 4)))
    if not os.path.exists("analysis"):
        os.mkdir("analysis")
    print("保存更新的证据")
    with open("analysis/evidences" + str(episode) + '-' + str(epoch) + ".json", 'w', encoding='utf8') as fw:
        ep_list = []
        for idx in range(len(super_start_list)):
            ep = get_orig_span(train_examples[idx]['document'], tokenizer.tokenize(train_examples[idx]['document']),
                               super_start_list[idx] - 1, super_end_list[idx], tokenizer)
            ep_list.append(ep)
        json.dump(ep_list, fw, indent=2, ensure_ascii=False)


def get_group_parameters(model):
    params = list(model.named_parameters())
    llr = ['encoder.layer.10', 'encoder.layer.11']
    no_decay = ['bias,', 'LayerNorm.weight', 'LayerNorm.bias']
    other = ['trans_layer','option_linear', 'pooler']
    no_main = llr + no_decay + other

    param_group = [
        {'params':[p for n,p in params if not any(nd in n for nd in no_main)],'weight_decay':1e-2,'lr':2e-5},
        {'params':[p for n,p in params if not any(nd in n for nd in other) and not any(ll in n for ll in llr) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':2e-5},
        {'params':[p for n,p in params if any(nd in n for nd in llr) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':5e-5},
        {'params':[p for n,p in params if any(nd in n for nd in llr) and not any(nd in n for nd in no_decay) ],'weight_decay':1e-2,'lr':5e-5},
        {'params':[p for n,p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':3e-4},
        {'params':[p for n,p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay) ],'weight_decay':1e-2,'lr':3e-4},
    ]
    return param_group


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="zh", type=str)
    parser.add_argument("--model_path", default='../plms/macbert', type=str)
    parser.add_argument("--choice_num", default=4, type=int)
    parser.add_argument("--data_path_train",
                        default='../data/c3/formatted/train.json',
                        type=str)
    parser.add_argument("--data_path_dev",
                        default='../data/c3/formatted/dev.json',
                        type=str)
    parser.add_argument("--results_save_path", default='./results/', type=str)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument("--init_checkpoint", default=None, type=str)
    parser.add_argument("--base_train", default=False, type=str)
    parser.add_argument("--base_model", default=None, type=str)
    parser.add_argument("--max_len", default=512, type=int, )
    parser.add_argument("--max_len_qo", default=128, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--episodes", default=1, type=int, help="retrain the model every episode")
    parser.add_argument("--epoch_per_episode", default=2, type=int)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--m', type=float, default=0.8, help="momentum")
    parser.add_argument('--seed', type=int, default=123456)
    args = parser.parse_args()

    set_seed(args.seed)
    train_examples = read_dataset(args.data_path_train)
    # 记录super_span
    super_start_list, super_end_list = [-1] * len(train_examples), [-1] * len(train_examples)
    # 记录验证集上最好的super_span
    best_start_list, best_end_list = [-1] * len(train_examples), [-1] * len(train_examples)
    # 记录基准长度
    base_len_list = [-1] * len(train_examples)

            

    dev_examples = read_dataset(args.data_path_dev)

    print(json.dumps({"lr": args.lr, "model": args.model_path, "seed": args.seed,
                      "bs": args.train_batch_size,
                      'gradient_accumulation_steps': args.gradient_accumulation_steps,
                      "episodes": args.episodes,
                      "epoch_per_episode": args.epoch_per_episode,
                      "train_path": args.data_path_train,
                      "dev_path": args.data_path_dev,
                      "train_size": len(train_examples),
                      "dev_size": len(dev_examples),
                      "alpha": args.alpha, "beta": args.beta}, indent=2))

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    os.makedirs(args.results_save_path, exist_ok=True)

    # 开始episode
    for episode in range(args.episodes):
        model = RCEE(args.model_path, args.alpha, args.beta)
        param_group = get_group_parameters(model)
        optimizer = torch.optim.AdamW(param_group, lr=args.lr, eps=1e-7)
        model.to(device)
        train(args, model, tokenizer, optimizer, train_examples, dev_examples, episode)
        print('best dev f1:', best_dev_f1, 'best_dev_f1_ans:', best_dev_f1_ans, 'best_dev_f1_evid:', best_dev_f1_evid)
