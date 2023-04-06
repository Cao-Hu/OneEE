import logging
import time
import pickle


def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def decode(outputs, labels, tri_args):
    results = {}
    arg_dict = {}

    for key in ["ti", "tc", "ai", "ac"]:
        pred = outputs[key]
        if pred is None:
            pred_set = set()
        else:
            pred_set = set([tuple(x.tolist()) for x in pred])
        if key == "ai":
            for b, x, y, e in pred_set:
                for c in range(x, y + 1):
                    if (b, c) in arg_dict:
                        arg_dict[(b, c)].append((b, x, y))
                    else:
                        arg_dict[(b, c)] = [(b, x, y)]
        if key in ["ac"]:
            new_pred_set = set()
            for b, x, e, r in pred_set:
                if (b, x) in arg_dict:
                    for prefix in arg_dict[(b, x)]:
                        new_pred_set.add(prefix + (e, r))
            pred_set = set([x for x in new_pred_set if (x[-2], x[-1]) in tri_args])
        results[key + "_r"] = len(labels[key])
        results[key + "_p"] = len(pred_set)
        results[key + "_c"] = len(pred_set & labels[key])

    return results

def calculate_f1(r, p, c):
    if r == 0 or p == 0 or c == 0:
        return 0, 0, 0
    r = c / r
    p = c / p
    f1 = (2 * r * p) / (r + p)
    return f1, r, p