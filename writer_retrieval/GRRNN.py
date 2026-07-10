"""
    This code is for the following paper:

    Sheng He and Lambert Schomaker
    GR-RNN: Global-Context Residual Recurrent Neural Networks for Writer Identification
    Pattern Recognition

    @email: heshengxgd@gmail.com
    @author: Sheng He
    @Github: https://github.com/shengfly/writer-identification

    Modified for writer retrieval:
    - In training mode  → returns class logits (unchanged)
    - In eval mode      → returns L2-normalised embedding (glfa, 512-d)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# VGG backbone
# ---------------------------------------------------------------------------
class VGGnet(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
        layers = [64, 128, 256, 512]
        self.conv1 = self._conv(input_channel, layers[0])
        self.maxp1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = self._conv(layers[0], layers[1])
        self.maxp2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = self._conv(layers[1], layers[2])
        self.maxp3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = self._conv(layers[2], layers[3])
        self.maxp4 = nn.MaxPool2d(2, stride=2)

        self.featv = nn.AdaptiveAvgPool2d((1, 2))
        self.avg   = nn.AdaptiveAvgPool2d(1)

    def _conv(self, inplance, outplance, nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance, outplance, kernel_size=3,
                                  stride=1, padding=1, bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance
        return nn.Sequential(*conv)

    def forward(self, x):
        x = self.conv1(x); x = self.maxp1(x)
        x = self.conv2(x); x = self.maxp2(x)
        x = self.conv3(x); x = self.maxp3(x)
        xfeat = x                               # (B, 256, H/8, W/8)
        x = self.conv4(x); x = self.maxp4(x)
        x = torch.flatten(self.avg(x), 1)       # (B, 512) — global feature
        return x, xfeat


# ---------------------------------------------------------------------------
# GRU cell (custom)
# ---------------------------------------------------------------------------
class GRUcell(nn.Module):
    def __init__(self, inplance, hidden_size, bias=True):
        super().__init__()
        self.inplance    = inplance
        self.hidden_size = hidden_size
        self.bias        = bias
        self.x2h = nn.Linear(inplance,     3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size,  3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(1))
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate   = torch.tanh(i_n + (resetgate * h_n))
        hy        = (1 - inputgate) * newgate + inputgate * hidden
        return hy


# ---------------------------------------------------------------------------
# GR-RNN  (identification during training, retrieval embedding at eval)
# ---------------------------------------------------------------------------
class GrnnNet(nn.Module):
    def __init__(self, input_channel, num_classes=105, mode='vertical'):
        super().__init__()
        self.mode = mode
        self.net  = VGGnet(input_channel)
        self.avg  = nn.AdaptiveAvgPool2d(1)
        self.ada  = nn.Linear(256, 512)
        self.rnn  = GRUcell(512, 512)
        self.classifier = nn.Linear(512, num_classes)

    # ------------------------------------------------------------------
    # Helper: run the GR-RNN recurrence and return the accumulated state
    # ------------------------------------------------------------------
    def _run_vertical(self, glf, feat):
        seq = feat.size()[-1] // 2
        glfa = None
        for n in range(seq):
            s    = 2 * n
            patch = feat[:, :, :, s:s + 2]
            lx   = torch.flatten(self.avg(patch), 1)
            lx   = self.ada(lx)
            glf  = self.rnn(lx, glf) + lx
            glfa = glf if n == 0 else glfa + glf
        return glfa

    def _run_horizontal(self, glf, feat):
        seq  = feat.size()[-2]
        glfa = None
        for n in range(seq):
            patch = feat[:, :, n, :].unsqueeze(2)
            lx    = torch.flatten(self.avg(patch), 1)
            lx    = self.ada(lx)
            glf   = self.rnn(lx, glf) + lx
            glfa  = glf if n == 0 else glfa + glf
        return glfa

    # ------------------------------------------------------------------
    def forward(self, x):
        glf, feat = self.net(x)

        if 'vertical' in self.mode:
            glfa = self._run_vertical(glf, feat)
        elif 'horzontal' in self.mode:          # keep original typo
            glfa = self._run_horizontal(glf, feat)
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. "
                             "Choose 'vertical' or 'horzontal'.")

        if self.training:
            # ---- classification path (training) ----
            return self.classifier(glfa)
        else:
            # ---- retrieval path (eval) ----
            # L2-normalise so cosine similarity == dot product
            return F.normalize(glfa, p=2, dim=1)

    # ------------------------------------------------------------------
    # Convenience: extract raw (un-normalised) embedding at eval time
    # ------------------------------------------------------------------
    def get_embedding(self, x):
        """Return raw glfa without L2 normalisation."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            glf, feat = self.net(x)
            if 'vertical' in self.mode:
                emb = self._run_vertical(glf, feat)
            else:
                emb = self._run_horizontal(glf, feat)
        if was_training:
            self.train()
        return emb


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    x = torch.rand(2, 1, 64, 128)

    mod = GrnnNet(1, num_classes=105, mode='vertical')

    # training mode → logits
    mod.train()
    logits = mod(x)
    print('Train  output shape:', logits.shape)   # (2, 105)

    # eval mode → L2-normalised embedding
    mod.eval()
    with torch.no_grad():
        emb = mod(x)
    print('Eval   output shape:', emb.shape)       # (2, 512)
    print('Embedding norms    :', emb.norm(dim=1)) # should be ~1.0