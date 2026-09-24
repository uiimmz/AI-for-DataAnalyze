# -*- coding: utf-8 -*-
"""
03_transformer.py —— 8.4 路径三：深度学习自动提特征 · Transformer 编码器
对应教材：自注意力机制对关键时间步加权，与 LSTM 的循环结构形成对比
数据：同 02_lstm.py，原始 96Hz 惯性序列
跑法：python 03_transformer.py
"""
from common import *
from seqdata import build_seq_dataset, train_torch_model, torch, nn, torch_skip, \
    WIN, SEQ_SIG

EPOCHS, D_MODEL, NHEAD, NLAYER = 15, 64, 4, 2


if nn is not None:
    class TransformerNet(nn.Module):
        """线性投影 + 可学习位置编码 + TransformerEncoder。输入 (B, T, C) -> logit (B,)"""

        def __init__(self, c_in, d=D_MODEL, nhead=NHEAD, nlayer=NLAYER, t=WIN):
            super().__init__()
            self.proj = nn.Linear(c_in, d)
            self.pos = nn.Parameter(torch.zeros(1, t, d))   # 位置编码（T 固定）
            self.norm = nn.LayerNorm(d)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                               dim_feedforward=4 * d, dropout=0.3,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=nlayer)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            h = self.norm(self.proj(x) + self.pos)
            h = self.enc(h)                 # (B, T, d)
            return self.head(h.mean(dim=1)).squeeze(-1)   # 时间维平均池化
else:
    TransformerNet = None               # 没装 torch，main 里会先友好退出


if __name__ == "__main__":
    if torch_skip("03_transformer.py"):
        sys.exit(0)
    X, y, g = build_seq_dataset()
    if X is None:
        sys.exit(0)
    C = X.shape[2]
    # 位置编码按 WIN 开形状，X 的长度也必须等于 WIN —— 不等就是缓存/配置对不上，
    # 与其在 forward 里抛一句广播 RuntimeError，不如在这里说清楚。
    assert X.shape[1] == WIN, \
        f"序列长度 {X.shape[1]} 与 common.py 的 WIN={WIN} 不一致：缓存是旧设置建的，" \
        f"删掉 results/seq_dataset.npz 重跑即可"

    y_prob = train_torch_model(lambda: TransformerNet(C), X, y, g,
                               epochs=EPOCHS, tag="Transformer")
    y_pred = (y_prob >= 0.5).astype(int)
    save_metrics([evaluate(y, y_pred, y_prob, "Transformer")], "8.4_Transformer指标.csv")

    save_pred("transformer", y, y_prob, SEQ_SIG)   # 供 8.5 与 01b/02 同图对比
    print("\n完成。结果已存到 results/")
