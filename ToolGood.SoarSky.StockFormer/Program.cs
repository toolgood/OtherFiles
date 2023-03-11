namespace ToolGood.SoarSky.StockFormer
{
    internal class Program
    {
        static void Main(string[] args)
        {


            //Informer();
            Autoformer();

            //NsTransformer();
            //NsAutoformer();
            //  PatchTST();



        }
        public static void Autoformer()
        {
            //--is_training 1   --root_path ./dataset/ETT-small/   --data_path ETTh1.csv   --model_id ETTh1_96_24   --model Autoformer
            //--data ETTh1   --features M   --seq_len 96   --label_len 48   --pred_len 24   --e_layers 2   --d_layers 1   --factor 3
            //--enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --itr 1
            var expConfig = new ToolGood.SoarSky.StockFormer.Autoformers.Exps.Exp_Config();
            //expConfig.is_training=true;
            expConfig.data = "ETTh1";
            expConfig.features = "M";
            expConfig.seq_len = 96;
            expConfig.label_len = 48;
            expConfig.pred_len = 24;

            expConfig.e_layers = 2;
            expConfig.d_layers = 1;
            expConfig.factor = 3;
            expConfig.enc_in = 7;
            expConfig.dec_in = 7;
            expConfig.c_out = 7;

            var exp = new ToolGood.SoarSky.StockFormer.Autoformers.Exps.Exp_Main(expConfig);
            var model = exp._build_model();
            model.save("Autoformer.pth");
            exp.train(exp.ToString());

        }

        public static void NsTransformer()
        {
            var expConfig = new ToolGood.SoarSky.StockFormer.NsTransformers.Exps.Exp_Config();

            expConfig.data = "ETTh1";
            expConfig.features = "M";
            expConfig.seq_len = 96;
            expConfig.label_len = 48;
            expConfig.pred_len = 96;

            expConfig.e_layers = 2;
            expConfig.d_layers = 1;
            expConfig.enc_in = 7;
            expConfig.dec_in = 7;
            expConfig.c_out = 7;
            expConfig.p_hidden_dims = new List<int> { 256, 256 };
            expConfig.p_hidden_layers = 2;

            var exp = new ToolGood.SoarSky.StockFormer.NsTransformers.Exps.Exp_Main(expConfig);
            var model = exp._build_model();
            model.save("NsTransformer.pth");
            exp.train(exp.ToString());
        }

        public static void NsAutoformer()
        {
            var expConfig = new ToolGood.SoarSky.StockFormer.NsAutoformers.Models.NsAutoformerConfig();

            expConfig.data = "ETTh1";
            expConfig.features = "M";
            expConfig.seq_len = 96;
            expConfig.label_len = 48;
            expConfig.pred_len = 96;

            expConfig.e_layers = 2;
            expConfig.d_layers = 1;
            expConfig.enc_in = 7;
            expConfig.dec_in = 7;
            expConfig.c_out = 7;
            expConfig.p_hidden_dims = new List<int> { 256, 256 };
            expConfig.p_hidden_layers = 2;

            var exp = new ToolGood.SoarSky.StockFormer.NsAutoformers.Exps.Exp_Main(expConfig);
            var model = exp._build_model();
            model.save("NsAutoformer.pth");
            //   exp.train(exp.ToString());
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
            var model = exp._build_model();
            model.save("Informer.pth");
            //   exp.train(exp.ToString());
        }

        public static void PatchTST()
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
            var model = exp._build_model();
            model.save("PatchTST.pth");
            //   exp.train(exp.ToString());
        }
    }
}
