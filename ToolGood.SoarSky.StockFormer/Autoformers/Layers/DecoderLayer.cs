using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Layers
{
    // 
    //     Autoformer decoder layer with the progressive decomposition architecture
    //     
    public class DecoderLayer : nn.Module
    {
        public Module<Tensor, Tensor> activation;
        public Conv1d conv1;
        public Conv1d conv2;
        public AutoCorrelationLayer cross_attention;
        public series_decomp decomp1;
        public series_decomp decomp2;
        public series_decomp decomp3;
        public Dropout dropout;
        public Conv1d projection;
        public AutoCorrelationLayer self_attention;

        public DecoderLayer(
            AutoCorrelationLayer self_attention,
            AutoCorrelationLayer cross_attention,
            int d_model,
            int c_out,
            int? d_ff = null,
            int moving_avg = 25,
            double dropout = 0.1,
            string activation = "relu") : base("DecoderLayer")
        {
            d_ff = d_ff ?? 4 * d_model; //d_ff || 4 * d_model;
            this.self_attention = self_attention;
            this.cross_attention = cross_attention;
            this.conv1 = nn.Conv1d(inputChannel: d_model, outputChannel: d_ff.Value, kernelSize: 1, bias: false);
            this.conv2 = nn.Conv1d(inputChannel: d_ff.Value, outputChannel: d_model, kernelSize: 1, bias: false);
            this.decomp1 = new series_decomp(moving_avg);
            this.decomp2 = new series_decomp(moving_avg);
            this.decomp3 = new series_decomp(moving_avg);
            this.dropout = nn.Dropout(dropout);
            this.projection = nn.Conv1d(inputChannel: d_model, outputChannel: c_out, kernelSize: 3, stride: 1, padding: 1, paddingMode: TorchSharp.PaddingModes.Circular, bias: false);
            this.activation = activation == "relu" ? nn.ReLU() : nn.GELU();//F.relu : F.gelu;
            this.RegisterComponents();

        }

        public virtual (Tensor, Tensor) forward(Tensor x, Tensor cross, IMasking x_mask = null, IMasking cross_mask = null)
        {
            x = x + this.dropout.forward(this.self_attention.forward(x, x, x, attn_mask: x_mask).Item1);
            (x, var trend1) = this.decomp1.forward(x);
            x = x + this.dropout.forward(this.cross_attention.forward(x, cross, cross, attn_mask: cross_mask).Item1);
            (x, var trend2) = this.decomp2.forward(x);
            var y = x;
            y = this.dropout.forward(this.activation.forward(this.conv1.forward(y.transpose(-1, 1))));
            y = this.dropout.forward(this.conv2.forward(y).transpose(-1, 1));
            (x, var trend3) = this.decomp3.forward(x + y);
            var residual_trend = trend1 + trend2 + trend3;
            residual_trend = this.projection.forward(residual_trend.permute(0, 2, 1)).transpose(1, 2);
            return (x, residual_trend);
        }

    }
}
