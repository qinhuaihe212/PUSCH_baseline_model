# 5G NR PUSCH 复现：B0 基线

初学者可先读 [代码逐段解读](./代码解读.md)，再对照 `pusch_b0.py` 运行。

本目录中的 `pusch_b0.py` 对应演示图中的发射端与传统接收机 B0：TB/CB CRC、LDPC、速率匹配、扰码、QPSK、DMRS 资源映射和 OFDM 由 Sionna 完成；接收端使用 Sionna 默认的 DMRS LS 信道估计、LMMSE 检测/软解映射和 TB 译码。脚本通过直通与 AWGN 信道统计信息比特 BER、传输块 BLER，并可导出带发送真值的时域 IQ。

## 运行

建议 Python 3.11+、PyTorch 2.9+，参见 [Sionna 安装说明](https://nvlabs.github.io/sionna/installation.html)。在本目录：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python pusch_b0.py --domain freq --snr-db 0,5,10,15 --export-iq output
```

Linux/WSL 中将上述 `py -3.11` 换成 `python3.11`，将虚拟环境中的解释器路径换成 `.venv/bin/python`。若要连同 OFDM 调制/解调一起评估，用 `--domain time`。`--batches` 和 `--batch-size` 控制每个信噪比点的统计样本数；默认规模只是冒烟验证，绘制可靠的低 BLER 曲线须增加样本数。

`output/frame_0000.c64` 是连续复数 IQ（每点 float32 I、float32 Q），`frame_0000.npz` 含 IQ 和 TB 信息比特真值，`frame_0000.json` 含 PRB、MCS、DMRS、FFT/CP、采样率与帧编号。同一发送波形可用于后续硬件回放；采集后仍需另加帧同步、频偏校正、增益/采样率标定以及与发送标签的帧号对齐。当前脚本不宣称 X410 实测链路已闭环。

## 与后续 B1/B2 的接口

- **B1**：保留相同 PUSCH 发射机和 TB 译码器，换掉 B0 的检测/软解映射，网络需要输出与 Sionna `LayerDemapper`/`TBDecoder` 匹配的编码比特 LLR。先固定 Sionna 版本和码字长度，再做小批次过拟合检查。
- **B2**：图中的 G-ECCT 是 [FM-Receiver 论文](https://arxiv.org/abs/2607.12555)提出的分组、符号级译码器，和普通 ECCT 并非同一个现成模块。要接入 5G LDPC，需明确 CB 分段、速率匹配逆过程、扰码、分组方式以及校验矩阵与 Sionna 的位序对应关系，才能和 B0 作同一测试集比较。

可直接参考 [Sionna 官方 PUSCH 教程](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html)、[Sionna 源码](https://github.com/NVlabs/sionna)、[NVIDIA neural_rx](https://github.com/NVlabs/neural_rx) 与 [ECCT 原作者实现](https://github.com/yonilc/ecct)。`neural_rx` 使用旧版 Sionna 0.18、TensorFlow 2.15，不可直接与本脚本的 Sionna 2.2/PyTorch 环境混装；ECCT 原仓库主要展示短码示例，也不是 G-ECCT。它们适合阅读架构与训练流程，再针对当前配置移植。
