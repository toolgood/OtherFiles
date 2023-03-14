using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Layers
{
    // 
    //     Series decomposition block
    //     
    public class series_decomp : nn.Module
    {
        public moving_avg moving_avg;

        public series_decomp(int kernel_size) : base("series_decomp")
        {
            this.moving_avg = new moving_avg(kernel_size, stride: 1);
            this.RegisterComponents();

        }

        public virtual (Tensor, Tensor) forward(Tensor x)
        {
            var moving_mean = this.moving_avg.forward(x);
            var res = x - moving_mean;
            return (res, moving_mean);
        }
    }
}
