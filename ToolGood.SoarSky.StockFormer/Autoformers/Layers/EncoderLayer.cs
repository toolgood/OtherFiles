using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Layers
{
    // 
    //     Autoformer encoder layer with the progressive decomposition architecture
    //     
    public class EncoderLayer : nn.Module
    {
        public Module<Tensor,Tensor> activation;
        public AutoCorrelationLayer attention;
        public Conv1d conv1;
        public Conv1d conv2;
        public series_decomp decomp1;
        public series_decomp decomp2;
        public Dropout dropout;

        public EncoderLayer(
            AutoCorrelationLayer attention,
            int d_model,
            int? d_ff = null,
            int moving_avg = 25,
            double dropout = 0.1,
            string activation = "relu") : base("EncoderLayer")
        {
            d_ff = d_ff ?? 4 * d_model; //d_ff || 4 * d_model;
            this.attention = attention;
            this.conv1 = nn.Conv1d(inputChannel: d_model, outputChannel: d_ff.Value, kernelSize: 1, bias: false);
            this.conv2 = nn.Conv1d(inputChannel: d_ff.Value, outputChannel: d_model, kernelSize: 1, bias: false);
            this.decomp1 = new series_decomp(moving_avg);
            this.decomp2 = new series_decomp(moving_avg);
            this.dropout = nn.Dropout(dropout);
            this.activation = activation == "relu" ? nn.ReLU() : nn.GELU();//F.relu : F.gelu;
        }

        public virtual (Tensor, Tensor) forward(Tensor x, IMasking attn_mask = null)
        {
            var (new_x, attn) = this.attention.forward(x, x, x, attn_mask: attn_mask);
            x = x + this.dropout.forward(new_x);
            (x, _) = this.decomp1.forward(x);
            var y = x;
            y = this.dropout.forward(this.activation.forward(this.conv1.forward(y.transpose(-1, 1))));
            y = this.dropout.forward(this.conv2.forward(y).transpose(-1, 1));
            var (res, _) = this.decomp2.forward(x + y);
            return (res, attn);
        }
}
}
