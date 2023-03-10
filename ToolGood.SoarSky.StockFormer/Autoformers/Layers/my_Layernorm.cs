using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Layers
{
    public class my_Layernorm : nn.Module
    {
        public LayerNorm layernorm;

        public my_Layernorm(int channels) : base("my_Layernorm")
        {
            this.layernorm = nn.LayerNorm(channels);
        }

        public virtual Tensor forward(Tensor x)
        {
            var x_hat = this.layernorm.forward(x);
            var bias = torch.mean(x_hat, dimensions: new long[] { 1 }).unsqueeze(1).repeat(1, x.shape[1], 1);
            return x_hat - bias;
        }
    }
}
