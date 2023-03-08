using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class moving_avg : nn.Module
    {
        public AvgPool1d avg;
        public int kernel_size;

        public moving_avg(int kernel_size, int stride) : base("moving_avg")
        {
            this.kernel_size = kernel_size;
            avg = nn.AvgPool1d(kernelSize: kernel_size, stride: stride, padding: 0);
        }

        public virtual Tensor forward(Tensor x)
        {
            // padding on the both ends of time series
            var front = x[TensorIndex.Ellipsis, TensorIndex.Slice(0, 1, null), TensorIndex.Ellipsis].repeat(1, (kernel_size - 1) / 2, 1);
            var last = x.size(1) - 1;
            var end = x[TensorIndex.Ellipsis, last, TensorIndex.Ellipsis].repeat(1, (kernel_size - 1) / 2, 1);
            x = cat(new List<Tensor> { front, x, end }, dim: 1);
            x = avg.forward(x.permute(0, 2, 1));
            x = x.permute(0, 2, 1);
            return x;
        }
    }

}
