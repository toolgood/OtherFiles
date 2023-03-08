using NumpyDotNet;
using System.Diagnostics;
using ToolGood.SoarSky.StockFormer.Informers.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Informers.Layers
{

    public class ProbAttention : nn.Module<Tensor, Tensor, Tensor, IMasking, (Tensor, Tensor)>
    {
        public Dropout dropout;
        public int factor;
        public bool mask_flag;
        public bool output_attention;
        public double? scale;

        public ProbAttention(bool mask_flag = true, int factor = 5, double? scale = null, double attention_dropout = 0.1,
                             bool output_attention = false) : base("ProbAttention")
        {
            this.factor = factor;
            this.scale = scale;
            this.mask_flag = mask_flag;
            this.output_attention = output_attention;
            this.dropout = nn.Dropout(attention_dropout);
        }

        public virtual (Tensor, Tensor) _prob_QK(Tensor Q, Tensor K, int sample_k, int n_top)
        {
            // n_top: c*ln(L_q)
            // Q [B, H, L, D]
            var (B, H, L_K, E) = K.shape.ToLong4();
            var (_, _, L_Q, _) = Q.shape.ToLong4();
            // calculate the sampled Q_K
            var K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E);
            var index_sample = torch.randint(L_K, (L_Q, sample_k));
            var K_sample = K_expand[TensorIndex.Ellipsis, TensorIndex.Ellipsis,
                TensorIndex.Tensor(torch.arange(L_Q).unsqueeze(1)), 
                TensorIndex.Tensor(index_sample), TensorIndex.Ellipsis];
            var Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2);
            // find the Top_k query with sparisty measurement
            var M = Q_K_sample.max(-1).values - torch.div(Q_K_sample.sum(-1), L_K);
            var M_top = M.topk(n_top, sorted: false).indexes;
            // use the reduced Q to calculate Q_K
            var Q_reduce = Q[torch.arange(B)[TensorIndex.Ellipsis, null, null], torch.arange(H)[null, TensorIndex.Ellipsis, null], M_top, TensorIndex.Ellipsis];
            var Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1));
            return (Q_K, M_top);
        }

        public virtual Tensor _get_initial_context(Tensor V, long L_Q)
        {
            Tensor contex;
            var (B, H, L_V, D) = V.shape.ToLong4();
            if (!this.mask_flag) {
                // V_sum = V.sum(dim=-2)
                var V_sum = V.mean(new long[] { -2 });
                contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[^1]).clone();
            } else {
                // use mask
                Debug.Assert(L_Q == L_V);
                contex = V.cumsum(dim: -2);
            }
            return contex;
        }

        public virtual (Tensor, Tensor) _update_context(
            Tensor context_in,
            Tensor V,
            Tensor scores,
            long index,
            long L_Q,
            IMasking attn_mask)
        {
            var (B, H, L_V, D) = V.shape.ToLong4();
            if (this.mask_flag) {
                attn_mask = new ProbMask(B, H, L_Q, index, scores, device: V.device);
                scores.masked_fill_(attn_mask.mask, -np.Inf);
            }
            var attn = torch.softmax(scores, dim: -1);
            context_in[torch.arange(B)[TensorIndex.Ellipsis, null, null], torch.arange(H)[null, TensorIndex.Ellipsis, null], index, TensorIndex.Ellipsis] = torch.matmul(attn, V).type_as(context_in);
            if (this.output_attention) {
                var attns = (torch.ones(new long[] { B, H, L_V, L_V }) / L_V).type_as(attn).to(attn.device);
                attns[torch.arange(B)[TensorIndex.Ellipsis, null, null], torch.arange(H)[null, TensorIndex.Ellipsis, null], index, TensorIndex.Ellipsis] = attn;
                return (context_in, attns);
            } else {
                return (context_in, null);
            }
        }

        public override (Tensor, Tensor) forward(Tensor queries, Tensor keys, Tensor values, IMasking attn_mask)
        {
            var (B, L_Q, H, D) = queries.shape.ToLong4();
            var (_, L_K, _, _) = keys.shape.ToLong4();
            queries = queries.transpose(2, 1);
            keys = keys.transpose(2, 1);
            values = values.transpose(2, 1);
            var U_part = this.factor * (int)((float)Math.Ceiling(Math.Log(L_K)));//.astype("int").item();
            var u = this.factor * (int)Math.Ceiling(Math.Log(L_Q));//.astype("int").item();
            U_part = U_part < (int)L_K ? U_part : (int)L_K;
            u = u < (int)L_Q ? u : (int)L_Q;
            var (scores_top, index) = this._prob_QK(queries, keys, sample_k: U_part, n_top: u);
            // add scale factor
            var scale = this.scale ?? 1.0 / sqrt(D);
            if (scale is not null) {
                scores_top = scores_top * scale;
            }
            // get the context
            var context = this._get_initial_context(values, L_Q);
            // update the context with selected top_k queries
            (context, var attn) = this._update_context(context, values, scores_top, index.item<long>(), L_Q, attn_mask);
            return (context.transpose(2, 1).contiguous(), attn);
        }
    }

}
