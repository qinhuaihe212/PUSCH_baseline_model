"""单用户 5G NR PUSCH 传统接收机基线（B0），并导出带标签的时域 IQ。

依赖 Sionna PHY 2.2.x。这里模拟的是理想对齐的链路，不包含真实 SDR 所需的帧同步。

常见缩写：TB = 传输块，DMRS = 解调参考信号，IQ = 复数基带采样，
CP = 循环前缀，FFT/IFFT = 傅里叶变换/逆变换，AWGN = 加性高斯白噪声。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import sionna.phy
from sionna.phy.channel import AWGN
from sionna.phy.nr import PUSCHConfig, PUSCHReceiver, PUSCHTransmitter


def make_config(prbs: int, mcs: int) -> PUSCHConfig:
    """建立发射和接收共用的 PUSCH 参数。

    PRB 是物理资源块，每块包含 12 个子载波；MCS 决定调制方式和编码率。
    图中的 TB CRC、LDPC、速率匹配和扰码由 Sionna 的发射机内部完成。
    """
    cfg = PUSCHConfig()
    # 缩小频域资源，便于初学者在普通电脑上快速跑通示例。
    cfg.carrier.n_size_grid = prbs
    # 首轮实验只用一个发射天线端口和一层数据流。
    cfg.num_antenna_ports = 1
    cfg.num_layers = 1
    cfg.tb.mcs_table = 1
    cfg.tb.mcs_index = mcs  # MCS 表 1 的 0～9 对应 QPSK，编码率各不相同。
    # DMRS 是接收端估计信道的已知导频；端口 0 对应当前的单层传输。
    cfg.dmrs.dmrs_port_set = [0]
    cfg.dmrs.additional_position = 0  # 不额外增加 DMRS 符号位置。
    # 在创建发射机之前，让 Sionna 检查参数组合是否有效。
    cfg.check_config()
    if int(cfg.tb.num_bits_per_symbol) != 2:
        raise ValueError("Choose a QPSK MCS (table 1, index 0..9).")
    return cfg


def error_counts(bits: torch.Tensor, decoded: torch.Tensor) -> tuple[int, int, int, int]:
    """统计误比特数、总比特数、错误 TB 数和总 TB 数。

    张量形状为 [批大小, 用户数, 每个 TB 的信息比特数]。
    最后一维是一个 TB；只要其中任意一位错了，这个 TB 就计为错误。
    BER = 误比特数 / 总比特数，BLER = 错误 TB 数 / 总 TB 数。
    """
    wrong = bits.to(torch.int32) != decoded.to(torch.int32)
    return (
        int(wrong.sum().item()),  # 全部错误比特
        int(wrong.numel()),  # 全部信息比特
        int(wrong.any(dim=-1).sum().item()),  # 至少有 1 位出错的 TB
        int(wrong.shape[0] * wrong.shape[1]),  # 批大小 × 用户数
    )


def export_frame(cfg: PUSCHConfig, tx_freq: PUSCHTransmitter, folder: Path) -> None:
    """生成一个可回放的时域波形，并保存与之对应的发送比特。"""
    folder.mkdir(parents=True, exist_ok=True)
    # 先取得一组随机信息比特；频域波形在这里不用，因此用 _ 忽略。
    _, bits = tx_freq(1)
    # return_bits=False 表示把上面的同一组比特作为输入，而非重新随机生成。
    tx_time = PUSCHTransmitter(cfg, output_domain="time", return_bits=False)
    # 输出形状：[批大小, 用户, 发射天线, 时域采样点]；这里只取第一路。
    # detach/cpu/numpy 用于将 PyTorch 张量转换成可保存的 NumPy 数组。
    waveform = tx_time(bits).detach().cpu().numpy()[0, 0, 0].astype(np.complex64)
    payload = bits.detach().cpu().numpy()[0, 0].astype(np.uint8)
    grid = tx_time.resource_grid
    # 元数据让后续采集的 IQ 能找到相同的子载波、导频和发送真值配置。
    metadata = {
        "frame_id": 0,  # 本次导出的文件序号；不是 5G 帧号。
        "nr_frame_number": int(cfg.carrier.frame_number),
        "nr_slot_number": int(cfg.carrier.slot_number),
        "prbs": int(cfg.carrier.n_size_grid),
        "mcs_table": int(cfg.tb.mcs_table),
        "mcs_index": int(cfg.tb.mcs_index),
        "modulation": "QPSK",
        "num_layers": 1,
        "dmrs_port_set": [0],
        "tb_size_bits": int(cfg.tb_size),
        "fft_size": int(grid.fft_size),
        "cp_samples": int(grid.cyclic_prefix_length),
        "ofdm_symbols": int(grid.num_ofdm_symbols),
        "subcarrier_spacing_hz": float(grid.subcarrier_spacing),
        "sample_rate_hz": float(grid.fft_size * grid.subcarrier_spacing),  # Fs = N_FFT × Δf
        "iq_format": "complex64, little-endian float32 I then Q",
        "synchronization_preamble": False,
    }
    # .c64 是供回放工具读取的原始复数采样；.npz 便于 Python 读取波形和标签。
    waveform.tofile(folder / "frame_0000.c64")
    np.savez_compressed(folder / "frame_0000.npz", iq_tx=waveform, tb_bits=payload)
    (folder / "frame_0000.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(waveform)} IQ samples and {len(payload)} TB bits to {folder}")


def main() -> None:
    """解析命令行参数，构建链路，并在多个信噪比下统计 BER/BLER。"""
    p = argparse.ArgumentParser(description=__doc__)
    # freq：直接在资源栅格上加噪；time：包括 IFFT、CP、FFT 等 OFDM 步骤。
    p.add_argument("--domain", choices=("freq", "time"), default="freq")
    # 每个信噪比点重复 batches 个批次，每批包含 batch-size 个传输块。
    p.add_argument("--snr-db", default="0,5,10,15", help="Comma-separated Es/N0-like points")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--prbs", type=int, default=4)
    p.add_argument("--mcs", type=int, default=4, help="QPSK MCS in table 1 (0..9)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-iq", type=Path)
    args = p.parse_args()
    if args.batch_size < 1 or args.batches < 1:
        p.error("--batch-size and --batches must be positive")
    if not 0 <= args.mcs <= 9:
        p.error("--mcs must be in 0..9 for QPSK")
    snrs = [float(s) for s in args.snr_db.split(",")]

    # 固定随机种子，便于重复运行时比较结果。
    torch.manual_seed(args.seed)
    sionna.phy.config.seed = args.seed
    cfg = make_config(args.prbs, args.mcs)
    # 发射机内部完成编码、调制、DMRS 映射和可选的 OFDM 时域调制。
    tx = PUSCHTransmitter(cfg, output_domain=args.domain)
    # 默认接收机依次做 DMRS-LS 信道估计、LMMSE 检测、软解映射和 TB 译码。
    # 时域 AWGN 样本从第 0 个采样点开始，没有传播时延，所以 l_min=0。
    rx = PUSCHReceiver(tx, input_domain=args.domain,
                       l_min=0 if args.domain == "time" else None)
    channel = AWGN()  # 给发射信号叠加指定方差的复数高斯噪声。
    print(f"B0: {args.domain=}, PRBs={args.prbs}, MCS={args.mcs}, TB={cfg.tb_size} bits")
    print("SNR(dB)       BER          BLER       bit_errors / bits   block_errors / blocks")

    # 只进行推理，不训练神经网络；关闭梯度可节省内存。
    with torch.no_grad():
        # 直通检查：不经过噪声信道，接收机仍需一个很小的假设噪声方差。
        x, bits = tx(1)
        decoded = rx(x, 1e-6)
        be, _, bl, _ = error_counts(bits, decoded)
        print(f"direct-pass: bit_errors={be}, block_errors={bl}")

        for snr in snrs:
            # 分别累计：误比特、总比特、错误 TB、总 TB。
            counts = [0, 0, 0, 0]
            # 将 dB 转为线性噪声方差。这里是简单 AWGN 扫描，并非完整的 Eb/N0 换算。
            no = 10.0 ** (-snr / 10.0)
            for _ in range(args.batches):
                x, bits = tx(args.batch_size)
                # y 必须是通过信道后的信号；bits 只用于评估，不输入接收机。
                y = channel(x, no)
                decoded = rx(y, no)
                counts = [a + b for a, b in zip(counts, error_counts(bits, decoded))]
            be, nb, bl, nbl = counts
            print(f"{snr:7.1f}   {be/nb:10.4e}   {bl/nbl:10.4e}   {be:8d} / {nb:<8d}   {bl:5d} / {nbl}")

        if args.export_iq:
            # 无论仿真选 freq 还是 time，导出文件始终使用时域 IQ。
            tx_freq = tx if args.domain == "freq" else PUSCHTransmitter(cfg)
            export_frame(cfg, tx_freq, args.export_iq)


if __name__ == "__main__":
    main()
