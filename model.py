from collections import OrderedDict
import torch.nn as nn
from loss import ArcMarginProduct

class ASLModel(nn.Module):
    def __init__(self, p, in_features, n_class):
        super(ASLModel, self).__init__()
        self.dropout = nn.Dropout(p)
        self.layer0 = nn.Linear(in_features, 1024)
        self.layer1 = nn.Linear(1024, 512)
        self.layer2 = nn.Linear(512, n_class)
        
    def forward(self, x):
        x = self.layer0(x)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
class ASLLinearModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        first_out_features: int,
        num_classes: int,
        num_blocks: int,
        drop_rate: float,
        loss_fn,
        arcface
    ):
        super().__init__()

        blocks = []
        out_features = first_out_features
        for idx in range(num_blocks):
            # if idx == num_blocks - 1:
            #     out_features = num_classes

            blocks.append(self._make_block(in_features, out_features, drop_rate))

            in_features = out_features
            out_features = out_features // 2
        
        self.model = nn.Sequential(*blocks)
        self.loss_fn = loss_fn
        self.fc_probs = nn.Linear(256, num_classes)
        self.arcface = arcface

    def _make_block(self, in_features, out_features, drop_rate):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x, y):
        x = self.model(x)
        if self.arcface:
            arcface = self.loss_fn(x, y)
            return self.fc_probs(x), arcface
        else:
            return self.fc_probs(x)
        
