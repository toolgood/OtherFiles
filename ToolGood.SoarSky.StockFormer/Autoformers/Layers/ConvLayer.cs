using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Layers
{
    public class ConvLayer : nn.Module
    {
        public ELU activation;
        public Conv1d downConv;
        public MaxPool1d maxPool;
        public BatchNorm1d norm;

        public ConvLayer(long c_in) : base("ConvLayer")
        {
            var padding = 1;// torch.@__version__ >= "1.5.0" ? 1 : 2;
            this.downConv = nn.Conv1d(inputChannel: c_in, outputChannel: c_in, kernelSize: 3, padding: padding, paddingMode: TorchSharp.PaddingModes.Circular);
            this.norm = nn.BatchNorm1d(c_in);
            this.activation = nn.ELU();
            this.maxPool = nn.MaxPool1d(kernelSize: 3, stride: 2, padding: 1);
        }

        public virtual Tensor forward(Tensor x)
        {
            x = this.downConv.forward(x.permute(0, 2, 1));
            x = this.norm.forward(x);
            x = this.activation.forward(x);
            x = this.maxPool.forward(x);
            x = x.transpose(1, 2);
            return x;
        }
    }



}
