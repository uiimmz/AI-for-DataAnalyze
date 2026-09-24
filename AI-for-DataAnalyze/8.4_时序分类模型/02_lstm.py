# -*- coding: utf-8 -*-
"""
02_lstm.py —— 8.4 路径二：深度学习自动提特征 · LSTM
对应教材：LSTM 通过输入门/遗忘门/输出门解决 RNN 梯度消失，端到端学时序特征
数据：year2帕金森病患者与健康老年人行走运动参数数据（原始 96Hz 惯性序列）
跑法：python 02_lstm.py
"""
from common import *
from seqdata import build_seq_dataset, train_torch_model, torch, nn, torch_skip, SEQ_SIG

EPOCHS, HIDDEN = 15, 64


if nn is not None:
    class LSTMNet(nn.Module):
        """双向 LSTM + 全连接分类头。输入 (B, T, C) -> 输出 logit (B,)"""

        def __init__(self, c_in, hidden=HIDDEN):
            super().__init__()
            self.lstm = nn.LSTM(c_in, hidden, num_layers=2, batch_first=True,
                                bidirectional=True, dropout=0.3)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hidden, 1))

        def forward(self, x):
            out, _ = self.lstm(x)          # out: (B, T, 2H)
            return self.head(out[:, -1]).squeeze(-1)   # 取最后一个时间步
else:
    LSTMNet = None                      # 没装 torch，main 里会先友好退出


if __name__ == "__main__":
    if torch_skip("02_lstm.py"):
        sys.exit(0)
    X, y, g = build_seq_dataset()
    if X is None:
        sys.exit(0)
    C = X.shape[2]

    y_prob = train_torch_model(lambda: LSTMNet(C), X, y, g,
                               epochs=EPOCHS, tag="LSTM")
    y_pred = (y_prob >= 0.5).astype(int)
    save_metrics([evaluate(y, y_pred, y_prob, "LSTM")], "8.4_LSTM指标.csv")

    save_pred("lstm", y, y_prob, SEQ_SIG)   # 供 8.5 与 01b/03 同图对比
    print("\n完成。结果已存到 results/")
