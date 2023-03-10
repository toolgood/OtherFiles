using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using ToolGood.SoarSky.StockFormer.PatchTSTs.Layers;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Models
{
    public class PatchTST : nn.Module
    {
        public series_decomp decomp_module;
        public bool decomposition;
        public PatchTST_backbone model;
        public PatchTST_backbone model_res;
        public PatchTST_backbone model_trend;

        public PatchTST(PatchTSTConfig configs, int max_seq_len = 1024, int? d_k = null, int? d_v = null,
                        string norm = "BatchNorm", double attn_dropout = 0.0, string act = "gelu",
                        string key_padding_mask = "auto", object padding_var = null, object attn_mask = null,
                        bool res_attention = true, bool pre_norm = false, bool store_attn = false, string pe = "zeros",
                        bool learn_pe = true, bool pretrain_head = false, string head_type = "flatten",
                        bool verbose = false, Hashtable kwargs = null) : base("Model")
        {
            // load parameters
            var c_in = configs.enc_in;
            var context_window = configs.seq_len;
            var target_window = configs.pred_len;
            var n_layers = configs.e_layers;
            var n_heads = configs.n_heads;
            var d_model = configs.d_model;
            var d_ff = configs.d_ff;
            var dropout = configs.dropout;
            var fc_dropout = configs.fc_dropout;
            var head_dropout = configs.head_dropout;
            var individual = configs.individual;
            var patch_len = configs.patch_len;
            var stride = configs.stride;
            var padding_patch = configs.padding_patch;
            var revin = configs.revin;
            var affine = configs.affine;
            var subtract_last = configs.subtract_last;
            var decomposition = configs.decomposition;
            var kernel_size = configs.kernel_size;
            // model
            this.decomposition = decomposition;
            if (this.decomposition) {
                decomp_module = new series_decomp(kernel_size);
                model_trend = new PatchTST_backbone(c_in: c_in, context_window: context_window, target_window: target_window, patch_len: patch_len, stride: stride, max_seq_len: max_seq_len, n_layers: n_layers, d_model: d_model, n_heads: n_heads, d_k: d_k, d_v: d_v, d_ff: d_ff, norm: norm, attn_dropout: attn_dropout, dropout: dropout, act: act, key_padding_mask: key_padding_mask, padding_var: padding_var, attn_mask: attn_mask, res_attention: res_attention, pre_norm: pre_norm, store_attn: store_attn, pe: pe, learn_pe: learn_pe, fc_dropout: fc_dropout, head_dropout: head_dropout, padding_patch: padding_patch, pretrain_head: pretrain_head, head_type: head_type, individual: individual, revin: revin, affine: affine, subtract_last: subtract_last, verbose: verbose, kwargs);
                model_res = new PatchTST_backbone(c_in: c_in, context_window: context_window, target_window: target_window, patch_len: patch_len, stride: stride, max_seq_len: max_seq_len, n_layers: n_layers, d_model: d_model, n_heads: n_heads, d_k: d_k, d_v: d_v, d_ff: d_ff, norm: norm, attn_dropout: attn_dropout, dropout: dropout, act: act, key_padding_mask: key_padding_mask, padding_var: padding_var, attn_mask: attn_mask, res_attention: res_attention, pre_norm: pre_norm, store_attn: store_attn, pe: pe, learn_pe: learn_pe, fc_dropout: fc_dropout, head_dropout: head_dropout, padding_patch: padding_patch, pretrain_head: pretrain_head, head_type: head_type, individual: individual, revin: revin, affine: affine, subtract_last: subtract_last, verbose: verbose, kwargs);
            } else {
                model = new PatchTST_backbone(c_in: c_in, context_window: context_window, target_window: target_window, patch_len: patch_len, stride: stride, max_seq_len: max_seq_len, n_layers: n_layers, d_model: d_model, n_heads: n_heads, d_k: d_k, d_v: d_v, d_ff: d_ff, norm: norm, attn_dropout: attn_dropout, dropout: dropout, act: act, key_padding_mask: key_padding_mask, padding_var: padding_var, attn_mask: attn_mask, res_attention: res_attention, pre_norm: pre_norm, store_attn: store_attn, pe: pe, learn_pe: learn_pe, fc_dropout: fc_dropout, head_dropout: head_dropout, padding_patch: padding_patch, pretrain_head: pretrain_head, head_type: head_type, individual: individual, revin: revin, affine: affine, subtract_last: subtract_last, verbose: verbose, kwargs);
            }
            this.RegisterComponents();
        }

        public virtual Tensor forward(Tensor x)
        {
            // x: [Batch, Input length, Channel]
            if (decomposition) {
                var (res_init, trend_init) = decomp_module.forward(x);
                res_init = res_init.permute(0, 2, 1);
                trend_init = trend_init.permute(0, 2, 1);
                var res = model_res.forward(res_init);
                var trend = model_trend.forward(trend_init);
                x = res + trend;
                x = x.permute(0, 2, 1);
            } else {
                x = x.permute(0, 2, 1);
                x = model.forward(x);
                x = x.permute(0, 2, 1);
            }
            return x;
        }
    }

}
