using static TorchSharp.torch;
using TorchSharp.Modules;
using static TorchSharp.torch.nn;
using ToolGood.SoarSky.StockFormer.Utils;

namespace ToolGood.SoarSky.StockFormer.NsTransformers.Layers
{
    public class DecoderLayer : nn.Module
    {
        public Module<Tensor, Tensor> activation;
        public Conv1d conv1;
        public Conv1d conv2;
        public AttentionLayer cross_attention;
        public Dropout dropout;
        public LayerNorm norm1;
        public LayerNorm norm2;
        public LayerNorm norm3;
        public AttentionLayer self_attention;

        public DecoderLayer(
            AttentionLayer self_attention,
            AttentionLayer cross_attention,
            int d_model,
            int? d_ff = null,
            double dropout = 0.1,
            string activation = "relu") : base("DecoderLayer")
        {
            d_ff = d_ff ?? 4 * d_model; //d_ff || 4 * d_model;
            this.self_attention = self_attention;
            this.cross_attention = cross_attention;
            this.conv1 = nn.Conv1d(inputChannel: d_model, outputChannel: d_ff.Value, kernelSize: 1);
            this.conv2 = nn.Conv1d(inputChannel: d_ff.Value, outputChannel: d_model, kernelSize: 1);
            this.norm1 = nn.LayerNorm(d_model);
            this.norm2 = nn.LayerNorm(d_model);
            this.norm3 = nn.LayerNorm(d_model);
            this.dropout = nn.Dropout(dropout);
            this.activation = activation == "relu" ? nn.ReLU() : nn.GELU();//F.relu : F.gelu;
        }

        public virtual Tensor forward(Tensor x, Tensor cross, IMasking x_mask = null, IMasking cross_mask = null,
                                      Tensor tau = null, Tensor delta = null)
        {
            // Note that delta only used for Self-Attention(x_enc with x_enc) 
            // and Cross-Attention(x_enc with x_dec), 
            // but not suitable for Self-Attention(x_dec with x_dec)
            x = x + this.dropout.forward(this.self_attention.forward(x, x, x, attn_mask: x_mask, tau: tau, delta: null).Item1);
            x = this.norm1.forward(x);
            x = x + this.dropout.forward(this.cross_attention.forward(x, cross, cross, attn_mask: cross_mask, tau: tau, delta: delta).Item1);
            var y = x = this.norm2.forward(x);
            y = this.dropout.forward(this.activation.forward(this.conv1.forward(y.transpose(-1, 1))));
            y = this.dropout.forward(this.conv2.forward(y).transpose(-1, 1));
            return this.norm3.forward(x + y);
        }
    }

}
