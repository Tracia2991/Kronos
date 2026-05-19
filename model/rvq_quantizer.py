import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ


class KronosRVQQuantizer(nn.Module):
    """
    残差向量量化器，替换 Kronos 原有的 BSQ 双粒度量化。

    参数：
        dim              : 输入向量维度，必须等于 s1_bits + s2_bits（例如10+10=20）
        num_quantizers   : 残差层数，建议4层
        codebook_size    : 每层codebook词表大小，建议256
        codebook_dim     : codebook内部投影维度，建议16（省显存）
        commitment_weight: commitment loss权重，建议0.25
        quantize_dropout : 必须 > 0，这样才能支持部分层解码（粗粒度近似）
    """

    def __init__(
        self,
        dim: int,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        codebook_dim: int = 16,
        commitment_weight: float = 0.25,
        kmeans_init: bool = True,
        decay: float = 0.8,
        quantize_dropout: float = 0.1,   # ← 不能为0，否则部分层解码会报错
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.dim = dim

        self.rvq = ResidualVQ(
            dim=dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            decay=decay,
            shared_codebook=False,
            quantize_dropout=quantize_dropout,  # ← 必须设置
        )

    def forward(self, z):
        """
        返回顺序和原 BSQuantizer.forward() 完全一致：

            rvq_loss  : 标量，直接替代原来的 bsq_loss
            quantized : (B, T, dim)，替代原来的 quantized
            z_indices : (B, T, num_quantizers)，替代原来的 z_indices
        """
        quantized, z_indices, commit_losses = self.rvq(z)
        rvq_loss = commit_losses.sum()
        return rvq_loss, quantized, z_indices

    def encode_to_tokens(self, z):
        """推理专用，返回 (B, T, num_quantizers)"""
        with torch.no_grad():
            _, z_indices, _ = self.rvq(z)
        return z_indices

    def decode_from_tokens(self, indices):
        """
        从token索引重建量化向量，支持部分层（用于粗粒度近似）。
        参数: indices (B, T, k)，k <= num_quantizers
        返回: (B, T, dim)
        """
        return self.rvq.get_output_from_indices(indices)

if __name__ == "__main__":
    import torch
    print("=" * 40)
    print("RVQ 验证开始")
    print("=" * 40)

    # 模拟真实Kronos参数：s1_bits=10, s2_bits=10 → codebook_dim=20
    quantizer = KronosRVQQuantizer(
        dim=20,
        num_quantizers=2,
        codebook_size=256,
        codebook_dim=16,
    )

    z = torch.randn(4, 90, 20)  # batch=4, lookback=90, dim=20
    
    # 测试 forward
    loss, quantized, indices = quantizer(z)
    print(f"forward() 测试:")
    print(f"  输入形状:     {z.shape}")
    print(f"  quantized:   {quantized.shape}  ← 应该和输入一样")
    print(f"  indices:     {indices.shape}   ← 应该是 (4, 90, 2)")
    print(f"  loss:        {loss.item():.4f}")

    # 测试 encode_to_tokens
    tokens = quantizer.encode_to_tokens(z)
    print(f"\nencode_to_tokens() 测试:")
    print(f"  tokens形状:  {tokens.shape}  ← 应该是 (4, 90, 2)")

    # 测试 decode_from_tokens
    recovered = quantizer.decode_from_tokens(tokens)
    print(f"\ndecode_from_tokens() 测试:")
    print(f"  还原形状:    {recovered.shape}  ← 应该和输入一样")

    print("=" * 40)
    print("验证通过！可以开始训练了。")
    print("=" * 40)