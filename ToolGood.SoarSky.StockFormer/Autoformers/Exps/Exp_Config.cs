using ToolGood.SoarSky.StockFormer.Autoformers.Models;
using ToolGood.SoarSky.StockFormer.DataProvider;

namespace ToolGood.SoarSky.StockFormer.Autoformers.Exps
{
    public class Exp_Config : AutoformerConfig, IConfig
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
        public override string freq { get; set; } = "d";
        /// <summary>
        ///模型检查点的位置（默认为 ./checkpoints/）
        ///Location of model checkpoints (defaults to ./checkpoints/)
        /// </summary>
        public string checkpoints { get; set; } = "./checkpoints/";


        #endregion

    }
}
