using NumpyDotNet;
using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Informers.Layers
{
    public class FullAttention : nn.Module<Tensor, Tensor, Tensor, IMasking, (Tensor, Tensor)>
    {
        public Dropout dropout;
        public bool mask_flag;
        public bool output_attention;
        public double? scale;

        public FullAttention(bool mask_flag = true, int factor = 5, double? scale = null, double attention_dropout = 0.1,
                             bool output_attention = false) : base("FullAttention")
        {
            this.scale = scale;
            this.mask_flag = mask_flag;
            this.output_attention = output_attention;
            this.dropout = nn.Dropout(attention_dropout);
        }

        public override (Tensor, Tensor) forward(Tensor queries, Tensor keys, Tensor values, IMasking attn_mask)
        {
            var (B, L, H, E) = queries.shape.ToLong4();
            var (_, S, _, D) = values.shape.ToLong4();
            var scale = this.scale ?? 1.0 / sqrt(E);
            var scores = torch.einsum("blhe,bshe->bhls", queries, keys);
            if (this.mask_flag) {
                if (attn_mask is null) {
                    attn_mask = new TriangularCausalMask(B, L, device: queries.device);
                }
                scores.masked_fill_(attn_mask.mask, -np.Inf);
            }
            var A = this.dropout.forward(torch.softmax(scale * scores, dim: -1));
            var V = torch.einsum("bhls,bshd->blhd", A, values);
            if (this.output_attention) {
                return (V.contiguous(), A);
            } else {
                return (V.contiguous(), null);
            }
        }
    }

}
