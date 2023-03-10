using System.Reflection;
using ToolGood.SoarSky.StockFormer.NsAutoformers.Layers;
using ToolGood.SoarSky.StockFormer.Utils;
using TorchSharp;
using TorchSharp.Utils;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.NsAutoformers.Models
{
    public class NsAutoformer : nn.Module
    {
        public DataEmbedding_wo_pos dec_embedding;
        public Decoder decoder;
        public series_decomp decomp;
        public Projector delta_learner;
        public DataEmbedding_wo_pos enc_embedding;
        public Encoder encoder;
        public int label_len;
        public bool output_attention;
        public int pred_len;
        public int seq_len;
        public Projector tau_learner;



        public NsAutoformer(NsAutoformerConfig configs) : base("NsAutoformer")
        {
            seq_len = configs.seq_len;
            label_len = configs.label_len;
            pred_len = configs.pred_len;
            output_attention = configs.output_attention;
            // Decomp
            var kernel_size = configs.moving_avg;
            decomp = new series_decomp(kernel_size);
            // Embedding
            // The series-wise connection inherently contains the sequential information.
            // Thus, we can discard the position embedding of transformers.
            enc_embedding = new DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout);
            dec_embedding = new DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout);
            // Encoder
            encoder = new Encoder((from l in Enumerable.Range(0, configs.e_layers)
                                        select new EncoderLayer(new AutoCorrelationLayer(new DSAutoCorrelation(false, configs.factor, attention_dropout: configs.dropout, output_attention: configs.output_attention), configs.d_model, configs.n_heads), configs.d_model, configs.d_ff, moving_avg: configs.moving_avg, dropout: configs.dropout, activation: configs.activation)).ToArray(), norm_layer: new my_Layernorm(configs.d_model));
            // Decoder
            decoder = new Decoder((from l in Enumerable.Range(0, configs.d_layers)
                                        select new DecoderLayer(new AutoCorrelationLayer(new DSAutoCorrelation(true, configs.factor, attention_dropout: configs.dropout, output_attention: false), configs.d_model, configs.n_heads), new AutoCorrelationLayer(new DSAutoCorrelation(false, configs.factor, attention_dropout: configs.dropout, output_attention: false), configs.d_model, configs.n_heads), configs.d_model, configs.c_out, configs.d_ff, moving_avg: configs.moving_avg, dropout: configs.dropout, activation: configs.activation)).ToArray(), norm_layer: new my_Layernorm(configs.d_model), projection: nn.Linear(configs.d_model, configs.c_out, hasBias: true));
            tau_learner = new Projector(enc_in: configs.enc_in, seq_len: configs.seq_len, hidden_dims: configs.p_hidden_dims, hidden_layers: configs.p_hidden_layers, output_dim: 1);
            delta_learner = new Projector(enc_in: configs.enc_in, seq_len: configs.seq_len, hidden_dims: configs.p_hidden_dims, hidden_layers: configs.p_hidden_layers, output_dim: configs.seq_len);


            this.register_module("decomp", decomp);
            this.register_module("enc_embedding", enc_embedding);
            this.register_module("dec_embedding", dec_embedding);
            this.register_module("encoder", encoder);
            this.register_module("decoder", decoder);
            this.register_module("tau_learner", tau_learner);
            this.register_module("delta_learner", delta_learner);

            RegisterComponents();
        }


        public virtual (Tensor, List<Tensor>) forward(
            Tensor x_enc, Tensor x_mark_enc, Tensor x_dec, Tensor x_mark_dec,
            IMasking enc_self_mask = null, IMasking dec_self_mask = null, IMasking dec_enc_mask = null)
        {
            var x_raw = x_enc.clone().detach();
            // Normalization
            var mean_enc = x_enc.mean(new long[] { 1 }, keepdim: true).detach();
            x_enc = x_enc - mean_enc;
            var std_enc = sqrt(var(x_enc, dimensions: 1, keepdim: true, unbiased: false) + 1E-05).detach();
            x_enc = x_enc / std_enc;
            var x_dec_new = cat(new List<Tensor> {
                    x_enc[TensorIndex.Ellipsis,-label_len,TensorIndex.Ellipsis],
                    zeros_like(x_dec[TensorIndex.Ellipsis,-pred_len,TensorIndex.Ellipsis])
                }, dim: 1).to(x_enc.device).clone();
            var tau = tau_learner.forward(x_raw, std_enc).exp();
            var delta = delta_learner.forward(x_raw, mean_enc);
            // Model Inference
            // decomp init
            var mean = torch.mean(x_enc, dimensions: new long[] { 1 }).unsqueeze(1).repeat(1, pred_len, 1);
            var zeros = torch.zeros(new long[] { x_dec_new.shape[0], pred_len, x_dec_new.shape[2] }, device: x_enc.device);
            var (seasonal_init, trend_init) = decomp.forward(x_enc);
            // decoder input
            trend_init = cat(new List<Tensor> { trend_init[TensorIndex.Ellipsis, -label_len, TensorIndex.Ellipsis], mean }, dim: 1);
            seasonal_init = cat(new List<Tensor> { seasonal_init[TensorIndex.Ellipsis, -label_len, TensorIndex.Ellipsis], zeros }, dim: 1);
            // enc
            var enc_out = enc_embedding.forward(x_enc, x_mark_enc);
            (enc_out, var attns) = encoder.forward(enc_out, attn_mask: enc_self_mask, tau: tau, delta: delta);
            // dec
            var dec_out = dec_embedding.forward(seasonal_init, x_mark_dec);
            var (seasonal_part, trend_part) = decoder.forward(dec_out, enc_out, x_mask: dec_self_mask, cross_mask: dec_enc_mask, trend: trend_init, tau: tau, delta: null);
            // final
            dec_out = trend_part + seasonal_part;
            // De-normalization
            dec_out = dec_out * std_enc + mean_enc;


            if (this.output_attention) {
                return (dec_out[TensorIndex.Ellipsis, -this.pred_len, TensorIndex.Ellipsis], attns);
            } else {
                return (dec_out[TensorIndex.Ellipsis, -this.pred_len, TensorIndex.Ellipsis], null);
            }
        }


        public static Dictionary<string, Tensor> state_dict2(nn.Module module, Dictionary<string, Tensor> destination = null, string prefix = null)
        {
            if (destination == null)
            {
                destination = new Dictionary<string, Tensor>();
            }

            foreach (var item3 in module.named_parameters())
            {
                string key = string.IsNullOrEmpty(prefix) ? item3.name ?? "" : prefix + "." + item3.name;
                destination.TryAdd(key, item3.parameter);
            }

            foreach (var item4 in module.named_buffers())
            {
                string key2 = string.IsNullOrEmpty(prefix) ? item4.name ?? "" : prefix + "." + item4.name;
                destination.TryAdd(key2, item4.buffer);
            }

            var fi = typeof(nn.Module).GetField("_internal_submodules", BindingFlags.Instance | BindingFlags.GetField | BindingFlags.IgnoreCase | BindingFlags.NonPublic);
            OrderedDict<string, nn.Module> _internal_submodules = fi.GetValue(module) as OrderedDict<string, nn.Module>;

            //module._internal_submodules
            foreach (var internal_submodule in _internal_submodules)
            {
                string item = internal_submodule.Item1;
                nn.Module item2 = internal_submodule.Item2;
                string prefix2 = string.IsNullOrEmpty(prefix) ? item ?? "" : prefix + "." + item;
                state_dict2(item2, destination, prefix2);
            }

            return destination;
        }
    }

}
