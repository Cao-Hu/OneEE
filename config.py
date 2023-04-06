import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config["dataset"]

        self.bert_hid_size = config["bert_hid_size"]
        self.tri_hid_size = config["tri_hid_size"]
        self.eve_hid_size = config["eve_hid_size"]
        self.arg_hid_size = config["arg_hid_size"]
        # self.node_type_size = config["node_type_size"]
        self.event_sample = config["event_sample"]
        # self.layers = config["layers"]

        self.dropout = config["dropout"]
        # self.graph_dropout = config["graph_dropout"]

        self.epochs = config["epochs"]
        self.warm_epochs = config["warm_epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.grad_clip_norm = config["grad_clip_norm"]
        self.gamma = config["gamma"]

        self.bert_name = config["bert_name"]
        self.bert_learning_rate = config["bert_learning_rate"]

        self.seed = config["seed"]

        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())