using ToolGood.SoarSky.StockFormer.Autoformers.Layers;
using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp;
using static TorchSharp.torch;
using Decoder = ToolGood.SoarSky.StockFormer.Autoformers.Layers.Decoder;
using Encoder = ToolGood.SoarSky.StockFormer.Autoformers.Layers.Encoder;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Models
{
    public class Autoformer : nn.Module
    {

        public DataEmbedding_wo_pos dec_embedding;

        public Decoder decoder;

        public series_decomp decomp;

        public DataEmbedding_wo_pos enc_embedding;

        public Encoder encoder;

        public int label_len;

        public bool output_attention;

        public int pred_len;

        public int seq_len;

        public Autoformer(AutoformerConfig configs) : base("Autoformer")
        {
            this.seq_len = configs.seq_len;
            this.label_len = configs.label_len;
            this.pred_len = configs.pred_len;
            this.output_attention = configs.output_attention;
            // Decomp
            var kernel_size = configs.moving_avg;
            this.decomp = new series_decomp(kernel_size);
            // Embedding
            // The series-wise connection inherently contains the sequential information.
            // Thus, we can discard the position embedding of transformers.
            this.enc_embedding = new DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout);
            this.dec_embedding = new DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout);
            // Encoder
            this.encoder = new Encoder((from l in Enumerable.Range(0, configs.e_layers)
                                        select new EncoderLayer(new AutoCorrelationLayer(new AutoCorrelation(false, configs.factor, attention_dropout: configs.dropout, output_attention: configs.output_attention), configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, moving_avg: configs.moving_avg, dropout: configs.dropout, activation: configs.activation)).ToArray(), norm_layer: new my_Layernorm(configs.d_model));
            // Decoder
            this.decoder = new Decoder((from l in Enumerable.Range(0, configs.d_layers)
                                        select new DecoderLayer(new AutoCorrelationLayer(new AutoCorrelation(true, configs.factor, attention_dropout: configs.dropout, output_attention: false), configs.d_model, configs.n_heads), new AutoCorrelationLayer(new AutoCorrelation(false, configs.factor, attention_dropout: configs.dropout, output_attention: false), configs.d_model, configs.n_heads), configs.d_model, configs.c_out, configs.d_ff, moving_avg: configs.moving_avg, dropout: configs.dropout, activation: configs.activation)).ToArray(), norm_layer: new my_Layernorm(configs.d_model), projection: nn.Linear(configs.d_model, configs.c_out, hasBias: true));
            this.RegisterComponents();
        }

        public virtual (Tensor, List<Tensor>) forward(
            Tensor x_enc,
            Tensor x_mark_enc,
            Tensor x_dec,
            Tensor x_mark_dec,
            IMasking enc_self_mask = null,
            IMasking dec_self_mask = null,
            IMasking dec_enc_mask = null)
        {
            // decomp init
            var mean = torch.mean(x_enc, dimensions: new long[] { 1 }).unsqueeze(1).repeat(1, this.pred_len, 1);
            var zeros = torch.zeros(new long[] { x_dec.shape[0], this.pred_len, x_dec.shape[2] }, device: x_enc.device);
            var (seasonal_init, trend_init) = this.decomp.forward(x_enc);
            // decoder input
            trend_init = torch.cat(new List<Tensor> { trend_init[TensorIndex.Colon, TensorIndex.Slice(-this.label_len), TensorIndex.Colon], mean }, dim: 1);
            seasonal_init = torch.cat(new List<Tensor> { seasonal_init[TensorIndex.Colon, TensorIndex.Slice(-this.label_len), TensorIndex.Colon], zeros }, dim: 1);
            // enc
            var enc_out = this.enc_embedding.forward(x_enc, x_mark_enc);
            (enc_out, var attns) = this.encoder.forward(enc_out, attn_mask: enc_self_mask);
            // dec
            var dec_out = this.dec_embedding.forward(seasonal_init, x_mark_dec);
            var (seasonal_part, trend_part) = this.decoder.forward(dec_out, enc_out, x_mask: dec_self_mask, cross_mask: dec_enc_mask, trend: trend_init);
            // final
            dec_out = trend_part + seasonal_part;
            if (this.output_attention) {
                return (dec_out[TensorIndex.Colon, TensorIndex.Slice(-this.pred_len), TensorIndex.Colon], attns);
            } else {
                return (dec_out[TensorIndex.Colon, TensorIndex.Slice(-this.pred_len), TensorIndex.Colon], null);
            }
        }
    }
}
