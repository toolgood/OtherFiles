using System.Collections;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Layers
{
    public class PatchTST_backbone : Module
    {
        public TSTiEncoder backbone;
        public Module<Tensor, Tensor> head;
        public int head_nf;
        public string head_type;
        public bool individual;
        public int n_vars;
        public string padding_patch;
        public ReplicationPad1d padding_patch_layer;
        public int patch_len;
        public bool pretrain_head;
        public bool revin;
        public RevIN revin_layer;
        public int stride;

        public PatchTST_backbone(int c_in, int context_window, int target_window, int patch_len, int stride,
                                 int max_seq_len = 1024, int n_layers = 3, int d_model = 128, int n_heads = 16,
                                 int? d_k = null, int? d_v = null, int d_ff = 256, string norm = "BatchNorm",
                                 double attn_dropout = 0.0, double dropout = 0.0, string act = "gelu",
                                 string key_padding_mask = "auto", object padding_var = null, object attn_mask = null,
                                 bool res_attention = true, bool pre_norm = false, bool store_attn = false,
                                 string pe = "zeros", bool learn_pe = true, double fc_dropout = 0.0,
                                 double head_dropout = 0, string padding_patch = null, bool pretrain_head = false,
                                 string head_type = "flatten", bool individual = false, bool revin = true,
                                 bool affine = true, bool subtract_last = false, bool verbose = false,
                                 Hashtable kwargs = null) : base("PatchTST_backbone")
        {
            // RevIn
            this.revin = revin;
            if (this.revin)
            {
                revin_layer = new RevIN(c_in, affine: affine, subtract_last: subtract_last);
            }
            // Patching
            this.patch_len = patch_len;
            this.stride = stride;
            this.padding_patch = padding_patch;
            var patch_num = Convert.ToInt32((context_window - patch_len) / stride + 1);
            if (padding_patch == "end")
            {
                // can be modified to general case
                padding_patch_layer = ReplicationPad1d(stride); // nn.ReplicationPad1d((0, stride));
                patch_num += 1;
            }
            // Backbone 
            backbone = new TSTiEncoder(c_in, patch_num: patch_num, patch_len: patch_len, max_seq_len: max_seq_len, n_layers: n_layers, d_model: d_model, n_heads: n_heads, d_k: d_k,
                d_v: d_v, d_ff: d_ff,
                attn_dropout: attn_dropout, dropout: dropout, act: act, key_padding_mask: key_padding_mask, padding_var: padding_var,
                attn_mask: attn_mask, res_attention: res_attention, pre_norm: pre_norm, store_attn: store_attn, pe: pe, learn_pe: learn_pe, verbose: verbose, kwargs: kwargs);
            // Head
            head_nf = d_model * patch_num;
            n_vars = c_in;
            this.pretrain_head = pretrain_head;
            this.head_type = head_type;
            this.individual = individual;
            if (this.pretrain_head)
            {
                head = create_pretrain_head(head_nf, c_in, fc_dropout);
            }
            else if (head_type == "flatten")
            {
                head = new Flatten_Head(this.individual, n_vars, head_nf, target_window, head_dropout: head_dropout);
            }
        }

        public virtual Tensor forward(Tensor z)
        {
            // z: [bs x nvars x seq_len]
            // norm
            if (revin)
            {
                z = z.permute(0, 2, 1);
                z = revin_layer.forward(z, "norm");
                z = z.permute(0, 2, 1);
            }
            // do patching
            if (padding_patch == "end")
            {
                z = padding_patch_layer.forward(z);
            }
            z = z.unfold(dimension: -1, size: patch_len, step: stride);
            z = z.permute(0, 1, 3, 2);
            // model
            z = backbone.forward(z);
            z = head.forward(z);
            // denorm
            if (revin)
            {
                z = z.permute(0, 2, 1);
                z = revin_layer.forward(z, "denorm");
                z = z.permute(0, 2, 1);
            }
            return z;
        }

        public virtual Sequential create_pretrain_head(long head_nf, long vars, double dropout)
        {
            return Sequential(Dropout(dropout), Conv1d(head_nf, vars, 1));
        }
    }


}
