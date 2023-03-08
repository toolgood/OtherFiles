using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorboard.ApiDef.Types;
using static TorchSharp.torch;
using ToolGood.SoarSky.StockFormer.Informers.Layers;
using TorchSharp.Modules;
using TorchSharp;
using ToolGood.SoarSky.StockFormer.Informers.Utils;

using TriangularCausalMask = ToolGood.SoarSky.StockFormer.Informers.Utils.TriangularCausalMask;
using ProbMask = ToolGood.SoarSky.StockFormer.Informers.Utils.ProbMask;
using Encoder = ToolGood.SoarSky.StockFormer.Informers.Layers.Encoder;
using EncoderLayer = ToolGood.SoarSky.StockFormer.Informers.Layers.EncoderLayer;
using ConvLayer = ToolGood.SoarSky.StockFormer.Informers.Layers.ConvLayer;
using Decoder = ToolGood.SoarSky.StockFormer.Informers.Layers.Decoder;
using DecoderLayer = ToolGood.SoarSky.StockFormer.Informers.Layers.DecoderLayer;
using FullAttention = ToolGood.SoarSky.StockFormer.Informers.Layers.FullAttention;
using ProbAttention = ToolGood.SoarSky.StockFormer.Informers.Layers.ProbAttention;
using AttentionLayer = ToolGood.SoarSky.StockFormer.Informers.Layers.AttentionLayer;
using DataEmbedding = ToolGood.SoarSky.StockFormer.Informers.Utils.DataEmbedding;

namespace ToolGood.SoarSky.StockFormer.Informers.Models
{
    public class Informer : nn.Module
    {
        public Conv1d end_conv2;
        public Conv1d end_conv1;
        public string attn;
        public DataEmbedding dec_embedding;
        public Decoder decoder;
        public DataEmbedding enc_embedding;
        public Encoder encoder;
        public bool output_attention;
        public int pred_len;
        public Linear projection;


        public Informer(int enc_in, int dec_in, long c_out, int seq_len, int label_len, int out_len, int factor = 5,
                        int d_model = 512, int n_heads = 8, int e_layers = 3, int d_layers = 2, int d_ff = 512,
                        double dropout = 0.0, string attn = "prob", string embed = "fixed", string freq = "h",
                        string activation = "gelu", bool output_attention = false, bool distil = true, bool mix = true,
                        Device device = null) : base("Informer")
        {
            this.pred_len = out_len;
            this.attn = attn;
            this.output_attention = output_attention;
            // Encoding
            this.enc_embedding = new DataEmbedding(enc_in, d_model, embed, freq, dropout);
            this.dec_embedding = new DataEmbedding(dec_in, d_model, embed, freq, dropout);
            // Attention

            if (attn == "prob") {
                this.encoder = new Encoder((from l in Enumerable.Range(0, e_layers)
                                            select new EncoderLayer(new AttentionLayer(new ProbAttention(false, factor, attention_dropout: dropout, output_attention: output_attention), d_model, n_heads, mix: false), d_model, d_ff, dropout: dropout, activation: activation)).ToArray(), distil ? (from l in Enumerable.Range(0, e_layers - 1)
                                                                                                                                                                                                                                                                                                       select new ConvLayer(d_model)).ToArray() : null, norm_layer: torch.nn.LayerNorm(d_model));
                // Decoder
                this.decoder = new Decoder((from l in Enumerable.Range(0, d_layers)
                                            select new DecoderLayer(new AttentionLayer(new ProbAttention(true, factor, attention_dropout: dropout, output_attention: false), d_model, n_heads, mix: mix), new AttentionLayer(new FullAttention(false, factor, attention_dropout: dropout, output_attention: false), d_model, n_heads, mix: false), d_model, d_ff, dropout: dropout, activation: activation)).ToArray(), norm_layer: torch.nn.LayerNorm(d_model));



            } else {
                this.encoder = new Encoder((from l in Enumerable.Range(0, e_layers)
                                            select new EncoderLayer(new AttentionLayer(new FullAttention(false, factor, attention_dropout: dropout, output_attention: output_attention), d_model, n_heads, mix: false), d_model, d_ff, dropout: dropout, activation: activation)).ToArray(), distil ? (from l in Enumerable.Range(0, e_layers - 1)
                                                                                                                                                                                                                                                                                                       select new ConvLayer(d_model)).ToArray() : null, norm_layer: torch.nn.LayerNorm(d_model));
                // Decoder
                this.decoder = new Decoder((from l in Enumerable.Range(0, d_layers)
                                            select new DecoderLayer(new AttentionLayer(new FullAttention(true, factor, attention_dropout: dropout, output_attention: false), d_model, n_heads, mix: mix), new AttentionLayer(new FullAttention(false, factor, attention_dropout: dropout, output_attention: false), d_model, n_heads, mix: false), d_model, d_ff, dropout: dropout, activation: activation)).ToArray(), norm_layer: torch.nn.LayerNorm(d_model));

            }

            // this.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
            // this.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
            this.projection = nn.Linear(d_model, c_out, hasBias: true);
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
            var enc_out = this.enc_embedding.forward(x_enc, x_mark_enc);
            (enc_out, var attns) = this.encoder.forward(enc_out, attn_mask: enc_self_mask);
            var dec_out = this.dec_embedding.forward(x_dec, x_mark_dec);
            dec_out = this.decoder.forward(dec_out, enc_out, x_mask: dec_self_mask, cross_mask: dec_enc_mask);
            dec_out = this.projection.forward(dec_out);
            // dec_out = this.end_conv1.forward(dec_out)
            // dec_out = this.end_conv2.forward(dec_out.transpose(2,1)).transpose(1,2)
            if (this.output_attention) {
                return (dec_out[TensorIndex.Ellipsis, -this.pred_len, TensorIndex.Ellipsis], attns);
            } else {
                return (dec_out[TensorIndex.Ellipsis, -this.pred_len, TensorIndex.Ellipsis], null);
            }
        }
    }
}
