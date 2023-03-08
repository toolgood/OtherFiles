using ToolGood.SoarSky.StockFormer.Informers.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ToolGood.SoarSky.StockFormer.Informers.Layers
{
    public class EncoderLayer : nn.Module
    {
        public Module<Tensor, Tensor> activation;
        public AttentionLayer attention;
        public Conv1d conv1;
        public Conv1d conv2;
        public Dropout dropout;
        public LayerNorm norm1;
        public LayerNorm norm2;

        public EncoderLayer(
            AttentionLayer attention,
            int d_model,
            int? d_ff = null,
            double dropout = 0.1,
            string activation = "relu") : base("EncoderLayer")
        {
            d_ff = d_ff ?? 4 * d_model; //d_ff || 4 * d_model;
            this.attention = attention;
            this.conv1 = nn.Conv1d(inputChannel: d_model, outputChannel: d_ff.Value, kernelSize: 1);
            this.conv2 = nn.Conv1d(inputChannel: d_ff.Value, outputChannel: d_model, kernelSize: 1);
            this.norm1 = nn.LayerNorm(d_model);
            this.norm2 = nn.LayerNorm(d_model);
            this.dropout = nn.Dropout(dropout);
            this.activation = activation == "relu" ? nn.ReLU() : nn.GELU();//F.relu : F.gelu;
        }

        public virtual (Tensor, Tensor) forward(Tensor x, IMasking attn_mask = null)
        {
            // x [B, L, D]
            // x = x + this.dropout.forward(this.attention.forward(
            //     x, x, x,
            //     attn_mask = attn_mask
            // ))
            var (new_x, attn) = this.attention.forward(x, x, x, attn_mask: attn_mask);
            x = x + this.dropout.forward(new_x);
            var y = x = this.norm1.forward(x);
            y = this.dropout.forward(this.activation.forward(this.conv1.forward(y.transpose(-1, 1))));
            y = this.dropout.forward(this.conv2.forward(y).transpose(-1, 1));
            return (this.norm2.forward(x + y), attn);
        }
    }



}
