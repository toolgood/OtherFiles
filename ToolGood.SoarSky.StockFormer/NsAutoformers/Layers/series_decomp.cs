using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.NsAutoformers.Layers
{
    public class series_decomp : nn.Module
    {
        public moving_avg moving_avg;

        public series_decomp(int kernel_size) : base("series_decomp")
        {
            moving_avg = new moving_avg(kernel_size, stride: 1);

            register_module("moving_avg", moving_avg);
            RegisterComponents();
        }

        public virtual (Tensor, Tensor) forward(Tensor x)
        {
            var moving_mean = moving_avg.forward(x);
            var res = x - moving_mean;
            return (res, moving_mean);
        }
    }
}
