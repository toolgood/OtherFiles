using static TorchSharp.torch;
using TorchSharp.Modules;
using NumpyDotNet;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class TSTEncoder : nn.Module
    {
        public ModuleList<TSTEncoderLayer> layers;
        public bool res_attention;

        public TSTEncoder(int q_len, int d_model, int n_heads, int? d_k = null, int? d_v = null, int? d_ff = null,
                          string norm = "BatchNorm", double attn_dropout = 0.0, double dropout = 0.0,
                          string activation = "gelu", bool res_attention = false, int n_layers = 1,
                          bool pre_norm = false, bool store_attn = false) : base("TSTEncoder")
        {
            layers = nn.ModuleList((from i in Enumerable.Range(0, n_layers)
                                    select new TSTEncoderLayer(q_len, d_model, n_heads: n_heads, d_k: d_k, d_v: d_v, d_ff: d_ff, norm: norm, attn_dropout: attn_dropout, dropout: dropout, activation: activation, res_attention: res_attention, pre_norm: pre_norm, store_attn: store_attn)).ToArray());
            this.res_attention = res_attention;
        }

        public virtual Tensor forward(Tensor src, Tensor key_padding_mask = null, Tensor attn_mask = null)
        {
            Tensor output = src;
            Tensor scores = null;
            if (res_attention) {
                foreach (var mod in layers) {
                    (output, scores) = mod.forward(output, prev: scores, key_padding_mask: key_padding_mask, attn_mask: attn_mask);
                }
                return output;
            } else {
                foreach (var mod in layers) {
                    (output, _) = mod.forward(output, key_padding_mask: key_padding_mask, attn_mask: attn_mask);
                }
                return output;
            }
        }
    }


}
