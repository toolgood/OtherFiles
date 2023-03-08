using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.utils;
using TorchSharp.Modules;
using ToolGood.SoarSky.StockFormer.PatchTSTs.Exps;
using static Tensorboard.ApiDef.Types;
using ToolGood.SoarSky.StockFormer.DataProvider;

namespace ToolGood.SoarSky.StockFormer
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Informer();


 
        }

        public static void Informer()
        {
            var expConfig = new ToolGood.SoarSky.StockFormer.Informers.Exps.Exp_Config();
            expConfig.data = "ETTh1";
            expConfig.features = "M";
            expConfig.seq_len = 96;
            expConfig.label_len = 48;
            expConfig.pred_len = 48;
            expConfig.e_layers = 2;
            expConfig.d_layers = 1;
            expConfig.attn = "prob";
            var exp = new ToolGood.SoarSky.StockFormer.Informers.Exps.Exp_Main(expConfig);
            exp.train(exp.ToString());


        }

        public static void PatchTST_Test()
        {
            var expConfig = new ToolGood.SoarSky.StockFormer.PatchTSTs.Exps.ExpConfig();
            expConfig.features = "M";
            expConfig.seq_len = 336;
            expConfig.enc_in = 7;
            expConfig.e_layers = 3;
            expConfig.n_heads = 4;
            expConfig.d_model = 16;
            expConfig.d_ff = 128;
            expConfig.dropout = 0.3;
            expConfig.fc_dropout = 0.3;
            expConfig.head_dropout = 0;
            expConfig.patch_len = 16;
            expConfig.train_epochs = 100;
            expConfig.batch_size = 128;
            expConfig.learning_rate = 0.0001;

            var exp = new ToolGood.SoarSky.StockFormer.PatchTSTs.Exps.Exp_Main(expConfig);
            exp.train(exp.ToString());
        }
    }
}
