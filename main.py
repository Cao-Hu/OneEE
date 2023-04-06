import argparse
import random

import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
import torch.nn.functional as F
import config
import data_loader
import utils
from model import Model


def compute_kl_loss(p, q):
    p_loss = F.kl_div(p, q, reduction='none')
    q_loss = F.kl_div(q, p, reduction='none')

    # pad_mask is for seq-level tasks
    # if pad_mask is not None:
    #     p_loss.masked_fill_(pad_mask, 0.)
    #     q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    # p_loss = p_loss.sum()
    # q_loss = q_loss.sum()
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def compute_dis_loss(x):
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.sum((x - x_mean) ** 2, dim=-1) / x.size(-1)
    loss = torch.sqrt(var)
    loss = loss.mean()
    return loss


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_true = y_true.float().detach()
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=warmup_steps,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        total_tc_r = 0
        total_tc_p = 0
        total_tc_c = 0

        total_ai_r = 0
        total_ai_p = 0
        total_ai_c = 0

        total_ac_r = 0
        total_ac_p = 0
        total_ac_c = 0

        # overlap = []

        alpha = epoch / config.epochs
        # gamma = gamma ** 2
        for i, data_batch in enumerate(data_loader):
            data_batch = [data.cuda() for data in data_batch[:-2]] + [data_batch[-2], data_batch[-1]]
            inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, event_idx, _, role_labels_num = data_batch
            # event_idx = [i for i in range(10)]
            # random.shuffle(event_idx)
            # event_idx = event_idx[:5]
            # input_event_idx = torch.cat([event_idx, torch.zeros((event_idx.size(0), 1), dtype=torch.long).cuda().fill_(11)], dim=-1)
            input_event_idx = event_idx
            tri_logits, arg_logits, role_logits = model(inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, input_event_idx)
            N = event_idx.size(-1)
            # loss_mask = ((tri_labels > 0).long() + (arg_labels > 0).long()).gt(0)
            # r_logits = torch.cat([tri_logits.unsqueeze(-1), arg_logits.unsqueeze(-1)], dim=-1)

            b_index = torch.arange(inputs.size(0)).long().cuda() * config.tri_label_num
            event_idx = event_idx + b_index.unsqueeze(-1)

            B, L, L, _ = tri_labels.size()

            K = config.rol_label_num
            tri_labels = tri_labels.view(-1, L, L)[event_idx].view(B, -1, L, L).permute(0, 2, 3, 1)
            arg_labels = arg_labels.view(-1, L, L)[event_idx].view(B, -1, L, L).permute(0, 2, 3, 1)
            role_labels = role_labels.view(-1, L, L, K)[event_idx].view(B, -1, L, L, K).permute(0, 2, 3, 1, 4)

            tri_mask = (tri_labels.sum(dim=-1).eq(0).long() + word_mask2d.long()).eq(2)
            arg_mask = (arg_labels.sum(dim=-1).eq(0).long() + word_mask2d.long()).eq(2)
            role_mask = (role_labels.sum(dim=-2).eq(0).long() + triu_mask2d[..., None].long()).eq(2)
            # tri_mask = ((tri_labels == tri_g_labels).long() + word_mask2d[..., None].long()).eq(2)
            # arg_mask = ((arg_labels == arg_g_labels).long() + word_mask2d[..., None].long()).eq(2)
            # role_mask = ((role_labels == role_g_labels).long() + triu_mask2d[..., None, None].long()).eq(2)

            # tri_g_labels = torch.cat([tri_labels, tri_g_labels], dim=-1)
            # arg_g_labels = torch.cat([arg_labels, arg_g_labels], dim=-1)
            # role_g_labels = torch.cat([role_labels, role_g_labels], dim=-2)
            loss1 = multilabel_categorical_crossentropy(tri_logits[word_mask2d], tri_labels[word_mask2d])
            loss2 = multilabel_categorical_crossentropy(arg_logits[word_mask2d], arg_labels[word_mask2d])
            loss3 = multilabel_categorical_crossentropy(role_logits[triu_mask2d], role_labels[triu_mask2d])
            loss = config.gamma * loss1 + loss2 + loss3

            # loss_r1 = compute_dis_loss(tri_logits[tri_mask])
            # loss_r2 = compute_dis_loss(arg_logits[arg_mask])
            # loss_r3 = compute_dis_loss(role_logits.transpose(-1, -2)[role_mask])
            # loss_r = loss_r1 + loss_r2 + loss_r3
            # loss = loss + alpha * config.gamma * loss_r

            # tri_logits = tri_logits[..., :-1]
            # arg_logits = arg_logits[..., :-1]
            # role_logits = role_logits[..., :-1, :]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            tri_outputs = tri_logits > 0
            total_tc_r += tri_labels.long().sum().item()
            total_tc_p += tri_outputs.sum().item()
            total_tc_c += (tri_outputs + tri_labels.long()).eq(2).sum().item()

            arg_outputs = arg_logits > 0
            total_ai_r += arg_labels.long().sum().item()
            total_ai_p += arg_outputs.sum().item()
            total_ai_c += (arg_outputs + arg_labels.long()).eq(2).sum().item()

            role_outputs = role_logits > 0
            total_ac_r += role_labels.sum().item()
            total_ac_p += role_outputs.sum().item()
            total_ac_c += (role_outputs + role_labels.long()).eq(2).sum().item()

            self.scheduler.step()
        tri_f1, tri_r, tri_p = utils.calculate_f1(total_tc_r, total_tc_p, total_tc_c)
        arg_f1, arg_r, arg_p = utils.calculate_f1(total_ai_r, total_ai_p, total_ai_c)
        role_f1, role_r, role_p = utils.calculate_f1(total_ac_r, total_ac_p, total_ac_c)

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "Tri F1", "Arg F1", "Role F1"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [tri_f1, arg_f1, role_f1]])
        logger.info("\n{}".format(table))
        # print(np.mean(overlap))
        # print(np.mean(loss2_list))
        # print(np.mean(loss3_list))
        # print(np.mean(loss4_list))
        # print(np.mean(loss5_list))

        return tri_f1 + arg_f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        total_results = {k + "_" + t: 0 for k in ["ti", "tc", "ai", "ac"] for t in ["r", "p", "c"]}
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                data_batch = [data.cuda() for data in data_batch[:-2]] + [data_batch[-2], data_batch[-1]]
                inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, event_idx, tuple_labels, _ = data_batch
                results = model(inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels)

                results = utils.decode(results, tuple_labels, config.tri_args)
                for key, value in results.items():
                    total_results[key] += value

        ti_f1, ti_r, ti_p = utils.calculate_f1(total_results["ti_r"], total_results["ti_p"], total_results["ti_c"])
        tc_f1, tc_r, tc_p = utils.calculate_f1(total_results["tc_r"], total_results["tc_p"], total_results["tc_c"])
        ai_f1, ai_r, ai_p = utils.calculate_f1(total_results["ai_r"], total_results["ai_p"], total_results["ai_c"])
        ac_f1, ac_r, ac_p = utils.calculate_f1(total_results["ac_r"], total_results["ac_p"], total_results["ac_c"])
        title = "EVAL" if not is_test else "TEST"

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Trigger I"] + ["{:3.4f}".format(x) for x in [ti_f1, ti_p, ti_r]])
        table.add_row(["Trigger C"] + ["{:3.4f}".format(x) for x in [tc_f1, tc_p, tc_r]])
        table.add_row(["Argument I"] + ["{:3.4f}".format(x) for x in [ai_f1, ai_p, ai_r]])
        table.add_row(["Argument C"] + ["{:3.4f}".format(x) for x in [ac_f1, ac_p, ac_r]])

        logger.info("\n{}".format(table))
        return (ti_f1 + ai_f1 + tc_f1 + ac_f1) / 4

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/fewfc.json')
    parser.add_argument('--device', type=int, default=1)

    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--tri_hid_size', type=int)
    parser.add_argument('--eve_hid_size', type=int)
    parser.add_argument('--arg_hid_size', type=int)
    parser.add_argument('--node_type_size', type=int)
    parser.add_argument('--event_sample', type=int)
    parser.add_argument('--layers', type=int)

    parser.add_argument('--dropout', type=float)
    parser.add_argument('--graph_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warm_epochs', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--grad_clip_norm', type=float)
    parser.add_argument('--gamma', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    if config.seed >= 0:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    datasets = data_loader.load_data(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=2,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    warmup_steps = config.warm_epochs * len(datasets[0])

    logger.info("Building Model")
    model = Model(config)

    model = model.cuda()

    trainer = Trainer(model)

    best_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        if i >= 5:
            f1 = trainer.eval(i, dev_loader)
            test_f1 = trainer.eval(i, test_loader, is_test=True)
            if f1 > best_f1:
                best_f1 = f1
                best_test_f1 = test_f1
                trainer.save("model.pt")
    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load("model.pt")
    trainer.eval("Final", test_loader, True)