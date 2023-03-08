using System.Collections;
using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class TSTiEncoder : nn.Module
    {
        public Dropout dropout;
        public TSTEncoder encoder;
        public int patch_len;
        public int patch_num;
        public int seq_len;
        public Linear W_P;
        public Tensor W_pos;

        public TSTiEncoder(int c_in, int patch_num, int patch_len, int max_seq_len = 1024, int n_layers = 3,
                           int d_model = 128, int n_heads = 16, int? d_k = null, int? d_v = null, int d_ff = 256,
                           string norm = "BatchNorm", double attn_dropout = 0.0, double dropout = 0.0,
                           string act = "gelu", bool store_attn = false, string key_padding_mask = "auto",
                           object padding_var = null, object attn_mask = null, bool res_attention = true,
                           bool pre_norm = false, string pe = "zeros", bool learn_pe = true, bool verbose = false,
                           Hashtable kwargs = null) : base("TSTiEncoder")
        {
            this.patch_num = patch_num;
            this.patch_len = patch_len;
            // Input encoding
            var q_len = patch_num;
            W_P = nn.Linear(patch_len, d_model);
            seq_len = q_len;
            // Positional encoding
            W_pos = PatchTST_layers.positional_encoding(pe, learn_pe, q_len, d_model);
            // Residual dropout
            this.dropout = nn.Dropout(dropout);
            // Encoder
            encoder = new TSTEncoder(q_len, d_model, n_heads, d_k: d_k, d_v: d_v, d_ff: d_ff, norm: norm, attn_dropout: attn_dropout, dropout: dropout, pre_norm: pre_norm, activation: act, res_attention: res_attention, n_layers: n_layers, store_attn: store_attn);
        }

        public virtual Tensor forward(Tensor x)
        {
            // x: [bs x nvars x patch_len x patch_num]
            var n_vars = x.shape[1];
            // Input encoding
            x = x.permute(0, 1, 3, 2);
            x = W_P.forward(x);
            var u = reshape(x, new long[] { x.shape[0] * x.shape[1], x.shape[2], x.shape[3] });
            u = dropout.forward(u + W_pos);
            // Encoder
            var z = encoder.forward(u);
            z = reshape(z, new long[] { -1, n_vars, z.shape[^2], z.shape[^1] });
            z = z.permute(0, 1, 3, 2);
            return z;
        }
    }


}
