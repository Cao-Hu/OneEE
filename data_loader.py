import json
import os
import random
import numpy as np
import prettytable as pt
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dis2idx = np.zeros((1000), dtype="int64")
# dis2idx[1] = 1
# dis2idx[2:] = 2
# dis2idx[4:] = 3
# dis2idx[8:] = 4
# dis2idx[16:] = 5
# dis2idx[32:] = 6
# dis2idx[64:] = 7
# dis2idx[128:] = 8
# dis2idx[256:] = 9
# mask_token_id = -1
# neg_num = 3

SAMPLE_NUM = 0

class Vocabulary(object):
    PAD = "<pad>"
    UNK = "<unk>"
    ARG = "<arg>"

    def __init__(self):
        self.tri_label2id = {}  # trigger
        self.tri_id2label = {}
        self.tri_id2count = defaultdict(int)
        self.tri_id2prob = {}

        self.rol_label2id = {}  # role
        self.rol_id2label = {}

    def label2id(self, label, type):
        label = label.lower()
        if type == "tri":
            return self.tri_label2id[label]
        elif type == "rol":
            return self.rol_label2id[label]
        else:
            raise Exception("Wrong Label Type!")

    def add_label(self, label, type):
        label = label.lower()

        if type == "tri":
            if label not in self.tri_label2id:
                self.tri_label2id[label] = len(self.tri_id2label)
                self.tri_id2label[self.tri_label2id[label]] = label
                self.tri_id2count[self.tri_label2id[label]] += 1
        elif type == "rol":
            if label not in self.rol_label2id:
                self.rol_label2id[label] = len(self.rol_id2label)
                self.rol_id2label[self.rol_label2id[label]] = label
        else:
            raise Exception("Wrong Label Type!")

    def get_prob(self):
        total = np.sum(list(self.tri_id2count.values()))
        for k, v in self.tri_id2count.items():
            self.tri_id2prob[k] = v / total

    @property
    def tri_label_num(self):
        return len(self.tri_label2id)

    @property
    def rol_label_num(self):
        return len(self.rol_label2id)

    @property
    def label_num(self):
        return self.tri_label_num


def collate_fn(data):
    inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, tuple_labels, event_list, training = map(
        list, zip(*data))

    batch_size = len(inputs)
    max_tokens = np.max([x.shape[0] for x in word_mask1d])

    inputs = pad_sequence(inputs, True)
    att_mask = pad_sequence(att_mask, True)
    word_mask1d = pad_sequence(word_mask1d, True)

    def pad_2d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    def pad_3d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :, :x.shape[1], :x.shape[2]] = x
        return new_data

    def pad_4d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :, :x.shape[1], :x.shape[2], :] = x
        return new_data
    word_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    word_mask2d = pad_2d(word_mask2d, word_mat)
    triu_mat = torch.zeros((batch_size, max_tokens, max_tokens), dtype=torch.bool)
    triu_mask2d = pad_2d(triu_mask2d, triu_mat)
    tri_mat = torch.zeros((batch_size, tri_labels[0].size(0), max_tokens, max_tokens), dtype=torch.bool)
    tri_labels = pad_3d(tri_labels, tri_mat)
    arg_mat = torch.zeros((batch_size, arg_labels[0].size(0), max_tokens, max_tokens), dtype=torch.bool)
    arg_labels = pad_3d(arg_labels, arg_mat)
    role_mat = torch.zeros((batch_size, role_labels[0].size(0), max_tokens, max_tokens, role_labels[0].size(-1)), dtype=torch.bool)
    role_labels = pad_4d(role_labels, role_mat)

    _tuple_labels = {k: set() for k in ["ti", "tc", "ai", "ac"]}
    if not training[0]:
        for i, x in enumerate(tuple_labels):
            for k, v in x.items():
                _tuple_labels[k] = _tuple_labels[k] | set([(i,) + t for t in x[k]])
        role_label_num = len(_tuple_labels["ac"])
    else:
        role_label_num = np.sum([len(x["ac"]) for x in tuple_labels])

    # all_event_idx = [i for i in range(tri_labels.size(1))]
    # event_idx = []
    # for b in range(inputs.size(0)):
    #     pos_events, neg_events, neg_probs = event_list[b]
    #     random.shuffle(pos_events)
    #     random.shuffle(neg_events)
    #     if len(neg_events) >= SAMPLE_NUM:
    #         if len(pos_events) >= 1:
    #             neg_list = neg_events[:SAMPLE_NUM]
    #             # neg_list = random.choices(neg_events, weights=neg_probs, k=SAMPLE_NUM)
    #             pos_list = pos_events[:1]
    #         else:
    #             neg_list = neg_events[:SAMPLE_NUM+1]
    #             # neg_list = random.choices(neg_events, weights=neg_probs, k=SAMPLE_NUM)
    #             pos_list = pos_events[:]
    #     else:
    #         neg_list = neg_events[:]
    #         pos_list = pos_events[:1+SAMPLE_NUM-len(neg_list)]
    #     event_idx.append(pos_list + neg_list)
    event_idx = []
    for b in range(inputs.size(0)):
        pos_event, neg_events = event_list[b]
        neg_list = random.choices(neg_events, k=SAMPLE_NUM)
        event_idx.append([pos_event] + neg_list)
    event_idx = torch.LongTensor(event_idx)
    return inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, event_idx, _tuple_labels, role_label_num


class RelationDataset(Dataset):
    def __init__(self, inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels,
                 role_labels, gold_tuples, event_list):
        self.inputs = inputs
        self.att_mask = att_mask
        self.word_mask1d = word_mask1d
        self.word_mask2d = word_mask2d
        self.triu_mask2d = triu_mask2d
        self.tri_labels = tri_labels
        self.arg_labels = arg_labels
        self.role_labels = role_labels
        self.tuple_labels = gold_tuples
        self.event_list = event_list
        self.training = True

    def eval_data(self):
        self.training = False

    def __getitem__(self, item):
        return torch.LongTensor(self.inputs[item]), \
               torch.LongTensor(self.att_mask[item]), \
               torch.BoolTensor(self.word_mask1d[item]), \
               torch.BoolTensor(self.word_mask2d[item]), \
               torch.BoolTensor(self.triu_mask2d[item]), \
               torch.BoolTensor(self.tri_labels[item]), \
               torch.BoolTensor(self.arg_labels[item]), \
               torch.BoolTensor(self.role_labels[item]), \
               self.tuple_labels[item], \
               self.event_list[item], \
               self.training

    def __len__(self):
        return len(self.inputs)


def process_bert(data, tokenizer, vocab):
    inputs = []
    att_mask = []
    word_mask1d = []
    word_mask2d = []
    triu_mask2d = []
    arg_labels = []
    tri_labels = []
    role_labels = []
    gold_tuples = []
    event_list = []

    total_event_set = set([i for i in range(vocab.tri_label_num)])

    # data = data[:100]
    for ins_id, instance in tqdm.tqdm(enumerate(data), total=len(data)):

        _inputs = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
            [x for x in instance["content"].lower()]) + [tokenizer.sep_token_id]
        length = len(_inputs) - 2

        _word_mask1d = np.array([1] * length)
        _word_mask2d = np.triu(np.ones((length, length), dtype=np.bool))
        # _triu_mask2d = np.triu(np.ones((length, length), dtype=np.bool), k=1)
        _triu_mask2d = np.ones((length, length), dtype=np.bool)
        np.fill_diagonal(_triu_mask2d, 0)
        _tri_labels = np.zeros((vocab.tri_label_num, length, length), dtype=np.bool)
        _arg_labels = np.zeros((vocab.tri_label_num, length, length), dtype=np.bool)
        _role_labels = np.zeros((vocab.tri_label_num, length, length, vocab.rol_label_num), dtype=np.bool)
        _att_mask = np.array([1] * len(_inputs))
        if "event_type" in instance:
            pos_event = vocab.label2id(instance["event_type"], "tri")
        else:
            pos_event = 0
        event_set = set()
        _gold_tuples = {k: set() for k in ["ti", "tc", "ai", "ac"]}
        events = instance["events"]
        for event in events:
            trigger = event["trigger"]
            t_s, t_e = trigger["span"]
            t_e = t_e - 1
            event_type = vocab.label2id(event["type"], "tri")
            _tri_labels[event_type, t_s, t_e] = 1
            _gold_tuples["ti"].add((t_s, t_e))
            _gold_tuples["tc"].add((t_s, t_e, event_type))
            event_set.add(event_type)
            args = event["args"]
            for k, v in args.items():
                for arg in v:
                    a_s, a_e = arg["span"]
                    a_e = a_e - 1
                    role = vocab.label2id(k, "rol")
                    _arg_labels[event_type, a_s, a_e] = 1
                    _role_labels[event_type, t_s:t_e+1, a_s:a_e+1, role] = 1
                    # if t_s < a_s:
                    #     _role_labels[t_s, a_s, event_type, role] = 1
                    # else:
                    #     _role_labels[a_s, t_s, event_type, role] = 1
                    _gold_tuples["ai"].add((a_s, a_e, event_type))
                    _gold_tuples["ac"].add((a_s, a_e, event_type, role))

        neg_event = list(total_event_set - event_set)
        # neg_probs = [vocab.tri_id2prob[x] for x in neg_event]

        inputs.append(_inputs)
        att_mask.append(_att_mask)
        word_mask1d.append(_word_mask1d)
        word_mask2d.append(_word_mask2d)
        triu_mask2d.append(_triu_mask2d)
        arg_labels.append(_arg_labels)
        tri_labels.append(_tri_labels)
        role_labels.append(_role_labels)
        gold_tuples.append(_gold_tuples)
        event_list.append((pos_event, neg_event))

    return inputs, att_mask, word_mask1d, word_mask2d, triu_mask2d, tri_labels, arg_labels, role_labels, gold_tuples, event_list


def fill_vocab(vocab, dataset):
    statistic = {"tri_num": 0, "arg_num": 0}
    for instance in dataset:
        events = instance["events"]
        for eve in events:
            vocab.add_label(eve["type"], "tri")
            args = eve["args"]
            for k, v in args.items():
                vocab.add_label(k, "rol")
            statistic["arg_num"] += len(args)
        statistic["tri_num"] += len(events)
    return statistic


def load_data(config):
    global EVENT_NUN
    global SAMPLE_NUM
    with open("./data/{}/new_train.json".format(config.dataset), "r", encoding="utf-8") as f:
        train_data = [json.loads(x) for x in f.readlines()]
    with open("./data/{}/dev.json".format(config.dataset), "r", encoding="utf-8") as f:
        dev_data = [json.loads(x) for x in f.readlines()]
    with open("./data/{}/test.json".format(config.dataset), "r", encoding="utf-8") as f:
        test_data = [json.loads(x) for x in f.readlines()]

    # train_data = train_data + dev_data
    # dev_data = test_data
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    config.tokenizer = tokenizer
    vocab = Vocabulary()
    train_statistic = fill_vocab(vocab, train_data)
    vocab.get_prob()
    dev_statistic = fill_vocab(vocab, dev_data)
    test_statistic = fill_vocab(vocab, test_data)

    with open("./data/{}/ty_args.json".format(config.dataset), "r", encoding="utf-8") as f:
        tri_args = json.load(f)
    config.tri_args = set()
    for k, vs in tri_args.items():
        for v in vs:
            k_i, v_i = vocab.label2id(k, "tri"), vocab.label2id(v, "rol")
            config.tri_args.add((k_i, v_i))

    table = pt.PrettyTable([config.dataset, "#sentence", "#event", "#argument"])
    table.add_row(["train", len(train_data)] + [train_statistic[key] for key in ["tri_num", "arg_num"]])
    table.add_row(["dev", len(dev_data)] + [dev_statistic[key] for key in ["tri_num", "arg_num"]])
    table.add_row(["test", len(test_data)] + [test_statistic[key] for key in ["tri_num", "arg_num"]])
    config.logger.info("\n{}".format(table))

    config.tri_label_num = vocab.tri_label_num
    config.rol_label_num = vocab.rol_label_num
    config.label_num = vocab.tri_label_num
    config.vocab = vocab

    EVENT_NUN = config.tri_label_num
    SAMPLE_NUM = config.event_sample

    print("Processing train data...")
    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    print("Processing dev data...")
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    print("Processing test data...")
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))

    dev_dataset.eval_data()
    test_dataset.eval_data()
    return train_dataset, dev_dataset, test_dataset


def load_lexicon(emb_path, vocab, emb_dim=50):
    emb_dict = load_pretrain_emb(emb_path)
    embed_size = emb_dim
    scale = np.sqrt(3.0 / emb_dim)
    embedding = np.random.uniform(-scale, scale, (len(vocab.word2id) + len(emb_dict), embed_size))

    for k, v in emb_dict.items():
        k = k.lower()
        index = len(vocab.word2id)
        vocab.word2id[k] = index
        vocab.id2word[index] = k

        embedding[index, :] = v
    embedding[0] = np.zeros(embed_size)
    embedding = torch.FloatTensor(embedding)
    return embedding


def load_embedding(emb_path, emb_dim, vocab):
    wvmodel = load_pretrain_emb(emb_path)
    embed_size = emb_dim
    scale = np.sqrt(3.0 / emb_dim)
    embedding = np.random.uniform(-scale, scale, (len(vocab), embed_size))
    hit = 0
    for token, i in vocab.items():
        if token in wvmodel:
            hit += 1
            embedding[i, :] = wvmodel[token]
    print("File: {} Total hit: {} rate {:.4f}".format(emb_path, hit, hit / len(vocab)))
    embedding[0] = np.zeros(embed_size)
    embedding = torch.FloatTensor(embedding)
    return embedding


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, "r", encoding="utf-8") as file:
        for line in tqdm.tqdm(file):
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict