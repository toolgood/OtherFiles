using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.NsAutoformers.Layers
{
    public class DecoderLayer : nn.Module
    {
        public nn.Module<Tensor, Tensor> activation;
        public Conv1d conv1;
        public Conv1d conv2;
        public series_decomp decomp1;
        public series_decomp decomp2;
        public series_decomp decomp3;
        public Dropout dropout;
        public Conv1d projection;

        public AutoCorrelationLayer self_attention;
        public AutoCorrelationLayer cross_attention;

        public DecoderLayer(
            AutoCorrelationLayer self_attention,
            AutoCorrelationLayer cross_attention,
            int d_model, int c_out, int? d_ff = null,
            int moving_avg = 25, double dropout = 0.1,
            string activation = "relu") : base("DecoderLayer")
        {
            d_ff = d_ff ?? 4 * d_model;
            this.self_attention = self_attention;
            this.cross_attention = cross_attention;
            conv1 = nn.Conv1d(inputChannel: d_model, outputChannel: d_ff.Value, kernelSize: 1, bias: false);
            conv2 = nn.Conv1d(inputChannel: d_ff.Value, outputChannel: d_model, kernelSize: 1, bias: false);
            decomp1 = new series_decomp(moving_avg);
            decomp2 = new series_decomp(moving_avg);
            decomp3 = new series_decomp(moving_avg);
            this.dropout = nn.Dropout(dropout);
            projection = nn.Conv1d(inputChannel: d_model, outputChannel: c_out, kernelSize: 3, stride: 1, padding: 1, paddingMode: TorchSharp.PaddingModes.Circular, bias: false);
            //this.activation = activation == "relu" ? F.relu : F.gelu;
            this.activation = activation == "relu" ? nn.ReLU() : nn.GELU();



            register_module("self_attention", this.self_attention);
            register_module("cross_attention", this.cross_attention);
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("decomp1", decomp1);
            register_module("decomp2", decomp2);
            register_module("decomp3", decomp3);
            register_module("dropout", this.dropout);
            register_module("projection", projection);
            register_module("activation", this.activation);

            RegisterComponents();

        }

        public virtual (Tensor, Tensor) forward(
            Tensor x, Tensor cross,
            IMasking x_mask = null, IMasking cross_mask = null,
            Tensor tau = null, Tensor delta = null)
        {
            // Note that delta only used for Self-Attention(x_enc with x_enc) 
            // and Cross-Attention(x_enc with x_dec), 
            // but not suitable for Self-Attention(x_dec with x_dec)
            x = x + dropout.forward(self_attention.forward(x, x, x, attn_mask: x_mask, tau: tau, delta: null).Item1);
            (x, var trend1) = decomp1.forward(x);
            x = x + dropout.forward(cross_attention.forward(x, cross, cross, attn_mask: cross_mask, tau: tau, delta: delta).Item1);
            (x, var trend2) = decomp2.forward(x);
            var y = x;
            y = dropout.forward(activation.forward(conv1.forward(y.transpose(-1, 1))));
            y = dropout.forward(conv2.forward(y).transpose(-1, 1));
            (x, var trend3) = decomp3.forward(x + y);
            var residual_trend = trend1 + trend2 + trend3;
            residual_trend = projection.forward(residual_trend.permute(0, 2, 1)).transpose(1, 2);
            return (x, residual_trend);
        }
    }
}
