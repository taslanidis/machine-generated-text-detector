import torch
from torch import nn
import torch.amp
from transformers import AutoModel


class FrozenRoberta(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name
        # self.n_classes = n_classes
        self.encoder = AutoModel.from_pretrained(self.name, num_labels=2)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

        pooler = list(self.encoder.pooler.parameters())
        self.last_params = pooler + list(self.classifier.parameters())

    def save_parameters(self, path):
        pooler_dict = self.encoder.pooler.state_dict()
        classifier_dict = self.classifier.state_dict()
        union = {**pooler_dict, **classifier_dict}
        torch.save(union, path)

    def load_parameters(self, path):
        union = torch.load(path)
        pooler_dict = {k: v for k, v in union.items() if 'dense' in k}
        classifier_dict = {k: v for k, v in union.items() if 'dense' not in k}
        self.encoder.pooler.load_state_dict(pooler_dict)
        self.classifier.load_state_dict(classifier_dict)

    def forward(self, *args, **kwargs):

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = self.encoder(*args, **kwargs)

        last_hidden_state = outputs.last_hidden_state.float()

        pooled_output = self.encoder.pooler(last_hidden_state)
        linear_output = self.classifier(pooled_output)

        return linear_output
