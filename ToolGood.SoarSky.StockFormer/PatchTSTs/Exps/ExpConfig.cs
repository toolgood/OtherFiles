using ToolGood.SoarSky.StockFormer.DataProvider;
using ToolGood.SoarSky.StockFormer.PatchTSTs.Models;
using static Tensorboard.ApiDef.Types;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Exps
{
    public class ExpConfig : PatchTSTConfig, IConfig
    {
        #region data loader

        public string model { get; set; } = "PatchTST";

        public string data { get; set; } = "ETTh1";
        /// <summary>
        ///数据文件的根路径（默认为 ./data/ETT/）
        ///The root path of the data file (defaults to ./data/ETT/)
        /// </summary>
        public string root_path { get; set; } = "./Datasets/ETT-small/";

        public string data_path { get; set; } = "ETTh1.csv";

        /// <summary>
        /// 预测任务，选项：[M，S，MS]；M： 多变量预测多变量，S：单变量预测单变量，MS：多变量预测单因素
        /// orecasting task, options:[M, S, MS]; M:multivariate predict multivariate,S:univariate predict univariate, MS:multivariate predict univariate
        /// </summary>
        public string features { get; set; } = "MS";

        public string target { get; set; } = "OT";

        /// <summary>
        ///时间特征编码的频率（默认为 h）。这可以设置为 s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly)。你也可以使用更详细的频率，如 15min 或 3h
        ///Freq for time features encoding (defaults to h). This can be set to s,t,h,d,b,w,m (s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly).You can also use more detailed freq like 15min or 3h
        /// </summary>
        public string freq { get; set; } = "d";
        /// <summary>
        ///模型检查点的位置（默认为 ./checkpoints/）
        ///Location of model checkpoints (defaults to ./checkpoints/)
        /// </summary>
        public string checkpoints { get; set; } = "./checkpoints/";


        #endregion

        #region optimization

        /// <summary>
        ///Data loader 的 num_works（默认为 0）
        ///The num_works of Data loader (defaults to 0)
        /// </summary>
        public int num_workers { get; set; } = 10;
        /// <summary>
        ///训练时期（默认为 6）
        ///Train epochs (defaults to 6)
        /// </summary>
        public int train_epochs { get; set; } = 100;
        /// <summary>
        ///训练输入数据的批量大小（默认为 32）
        ///The batch size of training input data (defaults to 32)
        /// </summary>
        public int batch_size { get; set; } = 128;
        /// <summary>
        ///提前停止耐心（默认为 3）
        ///Early stopping patience (defaults to 3)
        /// </summary>
        public int patience { get; set; } = 100;
        /// <summary>
        ///优化器学习率（默认为 0.0001）
        ///Optimizer learning rate (defaults to 0.0001)
        /// </summary>
        public double learning_rate { get; set; } = 0.0001;
        /// <summary>
        ///调整学习率的方法（默认为 type1）
        ///Ways to adjust the learning rate (defaults to type1)
        /// </summary>
        public string lradj { get; set; } = "type3";

        #endregion


        public double pct_start { get; set; } = 0.3;
        /// <summary>
        /// use automatic mixed precision training
        /// </summary>
        public bool test_flop { get; set; } = true;

        /// <summary>
        ///时间特征编码（默认为 timeF）。这可以设置为timeF，fixed，learned, date
        ///Time features encoding (defaults to timeF). This can be set to timeF, fixed, learned
        /// </summary>
        public string embed { get; set; } = "timeF";


        public override string ToString()
        {
            return $"{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_dm{d_model}_nh{n_heads}_el{e_layers}_df{d_ff}_eb{embed}";

        }

    }

}
