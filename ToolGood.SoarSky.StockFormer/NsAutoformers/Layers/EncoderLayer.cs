using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp.Modules;
using nn = TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace ToolGood.SoarSky.StockFormer.NsAutoformers.Layers
{
    public class EncoderLayer : nn.Module
    {
        public nn.Module<Tensor, Tensor> activation;
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
            d_ff = d_ff ?? 4 * d_model;
            this.attention = attention;
            conv1 = nn.Conv1d(inputChannel: d_model, outputChannel: d_ff.Value, kernelSize: 1, bias: false);
            conv2 = nn.Conv1d(inputChannel: d_ff.Value, outputChannel: d_model, kernelSize: 1, bias: false);
            decomp1 = new series_decomp(moving_avg);
            decomp2 = new series_decomp(moving_avg);
            this.dropout = nn.Dropout(dropout);
            //this.activation = activation == "relu" ? F.relu : F.gelu;
            this.activation = activation == "relu" ? nn.ReLU() : nn.GELU();

            register_module("attention", this.attention);
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("decomp1", decomp1);
            register_module("decomp2", decomp2);
            register_module("dropout", this.dropout);
            register_module("activation", this.activation);

            RegisterComponents();
        }

        public virtual (Tensor, Tensor) forward(Tensor x, IMasking attn_mask = null, Tensor tau = null, Tensor delta = null)
        {
            var (new_x, attn) = attention.forward(x, x, x, attn_mask: attn_mask, tau: tau, delta: delta);
            x = x + dropout.forward(new_x);
            (x, _) = decomp1.forward(x);
            var y = x;
            y = dropout.forward(activation.forward(conv1.forward(y.transpose(-1, 1))));
            y = dropout.forward(conv2.forward(y).transpose(-1, 1));
            var (res, _) = decomp2.forward(x + y);
            return (res, attn);
        }
    }
}
