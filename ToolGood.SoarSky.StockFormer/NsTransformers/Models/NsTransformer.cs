using ToolGood.SoarSky.StockFormer.NsTransformers.Layers;
using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp;
using static Tensorboard.TensorShapeProto.Types;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.NsTransformers.Models
{
    public class NsTransformer : nn.Module
    {

        public DataEmbedding dec_embedding;

        public Decoder decoder;

        public Projector delta_learner;

        public DataEmbedding enc_embedding;

        public Encoder encoder;

        public int label_len;

        public bool output_attention;

        public int pred_len;

        public int seq_len;

        public Projector tau_learner;

        public NsTransformer(NsTransformerConfig configs) : base("Model")
        {
            this.pred_len = configs.pred_len;
            this.seq_len = configs.seq_len;
            this.label_len = configs.label_len;
            this.output_attention = configs.output_attention;
            // Embedding
            this.enc_embedding = new DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout);
            this.dec_embedding = new DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout);
            // Encoder
            this.encoder = new Encoder((from l in Enumerable.Range(0, configs.e_layers)
                                        select new EncoderLayer(new AttentionLayer(new DSAttention(false, configs.factor, attention_dropout: configs.dropout, output_attention: configs.output_attention), configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, dropout: configs.dropout, activation: configs.activation)).ToArray(), norm_layer: torch.nn.LayerNorm(configs.d_model));
            // Decoder
            this.decoder = new Decoder((from l in Enumerable.Range(0, configs.d_layers)
                                        select new DecoderLayer(new AttentionLayer(new DSAttention(true, configs.factor, attention_dropout: configs.dropout, output_attention: false), configs.d_model, configs.n_heads), new AttentionLayer(new DSAttention(false, configs.factor, attention_dropout: configs.dropout, output_attention: false), configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, dropout: configs.dropout, activation: configs.activation)).ToArray(), norm_layer: torch.nn.LayerNorm(configs.d_model), projection: nn.Linear(configs.d_model, configs.c_out, hasBias: true));
            this.tau_learner = new Projector(enc_in: configs.enc_in, seq_len: configs.seq_len, hidden_dims: configs.p_hidden_dims, hidden_layers: configs.p_hidden_layers, output_dim: 1);
            this.delta_learner = new Projector(enc_in: configs.enc_in, seq_len: configs.seq_len, hidden_dims: configs.p_hidden_dims, hidden_layers: configs.p_hidden_layers, output_dim: configs.seq_len);

            this.RegisterComponents();
        }

        public virtual (Tensor, List<Tensor>) forward(Tensor x_enc, Tensor x_mark_enc, Tensor x_dec, Tensor x_mark_dec,
                                                      IMasking enc_self_mask = null, IMasking dec_self_mask = null,
                                                      IMasking dec_enc_mask = null)
        {
            var x_raw = x_enc.clone().detach();
            // Normalization
            var mean_enc = x_enc.mean(new long[] { 1 }, keepdim: true).detach();
            x_enc = x_enc - mean_enc;
            var std_enc = torch.sqrt(torch.var(x_enc, dimensions: 1, keepdim: true, unbiased: false) + 1E-05).detach();
            x_enc = x_enc / std_enc;
            // x_dec_new = torch.cat([x_enc[:, -self.label_len: , :], torch.zeros_like(x_dec[:, -self.pred_len:, :])], dim = 1).to(x_enc.device).clone()

            var x_dec_new = torch.cat(new List<Tensor> {
                                   x_enc[TensorIndex.Colon,TensorIndex.Slice(-this.label_len,null),TensorIndex.Colon],
                                   torch.zeros_like(x_dec[TensorIndex.Colon,TensorIndex.Slice(-this.pred_len,null),TensorIndex.Colon])
                                }, dim: 1).to(x_enc.device).clone();
            var tau = this.tau_learner.forward(x_raw, std_enc).exp();
            var delta = this.delta_learner.forward(x_raw, mean_enc);
            // Model Inference
            var enc_out = this.enc_embedding.forward(x_enc, x_mark_enc);
            (enc_out, var attns) = this.encoder.forward(enc_out, attn_mask: enc_self_mask, tau: tau, delta: delta);
            var dec_out = this.dec_embedding.forward(x_dec_new, x_mark_dec);
            dec_out = this.decoder.forward(dec_out, enc_out, x_mask: dec_self_mask, cross_mask: dec_enc_mask, tau: tau, delta: delta);
            // De-normalization
            dec_out = dec_out * std_enc + mean_enc;
            if (this.output_attention) {
                return (dec_out[TensorIndex.Colon, TensorIndex.Slice(-this.pred_len), TensorIndex.Colon], attns);
            } else {
                return (dec_out[TensorIndex.Colon, TensorIndex.Slice(-this.pred_len), TensorIndex.Colon], null);
            }
        }
    }

}
