using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ToolGood.SoarSky.StockFormer.Informers.Utils
{
    public interface IDataEmbedding
    {
        Tensor forward(Tensor x, Tensor x_mark);
    }

    public class DataEmbedding : nn.Module, IDataEmbedding
    {

        public Dropout dropout;

        public PositionalEmbedding position_embedding;

        public Module<Tensor, Tensor> temporal_embedding;

        public TokenEmbedding value_embedding;

        public DataEmbedding(
            int c_in,
            int d_model,
            string embed_type = "fixed",
            string freq = "h",
            double dropout = 0.1) : base("DataEmbedding")
        {
            this.value_embedding = new TokenEmbedding(c_in: c_in, d_model: d_model);
            this.position_embedding = new PositionalEmbedding(d_model: d_model);
            this.temporal_embedding = embed_type != "timeF" ? new TemporalEmbedding(d_model: d_model, embed_type: embed_type, freq: freq) : new TimeFeatureEmbedding(d_model: d_model, embed_type: embed_type, freq: freq);
            this.dropout = nn.Dropout(p: dropout);
        }

        public virtual Tensor forward(Tensor x, Tensor x_mark)
        {
            x = this.value_embedding.forward(x) + this.position_embedding.forward(x) + this.temporal_embedding.forward(x_mark);
            return this.dropout.forward(x);
        }
    }

}
