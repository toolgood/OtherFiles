using static TorchSharp.torch;
using TorchSharp.Modules;
using TorchSharp;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Layers
{
    // 
    //     Moving average block to highlight the trend of time series
    //     
    public class moving_avg : nn.Module
    {
        public AvgPool1d avg;
        public int kernel_size;

        public moving_avg(int kernel_size, int stride) : base("moving_avg")
        {
            this.kernel_size = kernel_size;
            this.avg = nn.AvgPool1d(kernelSize: kernel_size, stride: stride, padding: 0);
        }

        public virtual Tensor forward(Tensor x)
        {
            // padding on the both ends of time series
            var front = x[TensorIndex.Colon, TensorIndex.Slice(0, 1, null), TensorIndex.Colon].repeat(1, (int)((this.kernel_size - 1) / 2), 1);
            //var last = x.size(1) - 1;
            var end = x[TensorIndex.Colon, TensorIndex.Slice(-1, null), TensorIndex.Colon].repeat(1, (int)((this.kernel_size - 1) / 2), 1);
            x = torch.cat(new List<Tensor> { front, x, end }, dim: 1);
            x = this.avg.forward(x.permute(0, 2, 1));
            x = x.permute(0, 2, 1);
            return x;
        }
    }
}
