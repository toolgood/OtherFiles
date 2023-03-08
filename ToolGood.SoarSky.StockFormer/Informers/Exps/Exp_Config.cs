using ToolGood.SoarSky.StockFormer.DataProvider;

namespace ToolGood.SoarSky.StockFormer.Informers.Exps
{
    public class Exp_Config : IConfig
    {
        #region data loader

        public string model { get; set; } = "Informer";

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
        public string freq { get; set; } = "h";
        /// <summary>
        ///模型检查点的位置（默认为 ./checkpoints/）
        ///Location of model checkpoints (defaults to ./checkpoints/)
        /// </summary>
        public string checkpoints { get; set; } = "./checkpoints/";


        #endregion

        #region forecasting task
        /// <summary>
        ///Informer 编码器的输入序列长度（默认为 96）
        ///Input sequence length of Informer encoder (defaults to 96)
        /// </summary>
        public int seq_len { get; set; } = 96;
        /// <summary>
        ///Informer 解码器的起始令牌长度（默认为 48）
        ///Start token length of Informer decoder (defaults to 48)
        /// </summary>
        public int label_len { get; set; } = 48;
        /// <summary>
        ///预测序列长度（默认为 24​​）
        ///Prediction sequence length (defaults to 24)
        /// </summary>
        public int pred_len { get; set; } = 24;

        #endregion

        #region model define

        /// <summary>
        ///编码器输入大小（默认为 7）
        ///Encoder input size (defaults to 7)
        /// </summary>
        public int enc_in { get; set; } = 7;
        /// <summary>
        ///解码器输入大小（默认为 7）
        ///Decoder input size (defaults to 7)
        /// </summary>
        public int dec_in { get; set; } = 7;
        /// <summary>
        ///输出大小（默认为 7）
        ///Output size (defaults to 7)
        /// </summary>
        public int c_out { get; set; } = 7;

        /// <summary>
        ///模型尺寸（默认为 512）
        ///Dimension of model (defaults to 512)
        /// </summary>
        public int d_model { get; set; } = 512;

        /// <summary>
        ///头数（默认为 8）
        ///Num of heads (defaults to 8)
        /// </summary>
        public int n_heads { get; set; } = 8;

        /// <summary>
        ///编码器层数（默认为 2）
        ///Num of encoder layers (defaults to 2)
        /// </summary>
        public int e_layers { get; set; } = 2;
        /// <summary>
        ///解码器层数（默认为 1）
        ///Num of decoder layers (defaults to 1)
        /// </summary>
        public int d_layers { get; set; } = 1;

        /// <summary>
        ///fcn 的维度（默认为 2048）
        ///Dimension of fcn (defaults to 2048)
        /// </summary>
        public int d_ff { get; set; } = 2048;

        /// <summary>
        ///Probsparse attn 因子（默认为 5）
        ///Probsparse attn factor (defaults to 5)
        /// </summary>
        public int factor { get; set; } = 5;

        /// <summary>
        ///是否在编码器中使用蒸馏，使用该参数表示不使用蒸馏（默认为True）
        ///Whether to use distilling in encoder, using this argument means not using distilling (defaults to True)
        /// </summary>
        public bool distil { get; set; } = true;


        /// <summary>
        ///辍学概率（默认为 0.05）
        ///The probability of dropout (defaults to 0.05)
        /// </summary>
        public double dropout { get; set; } = 0.05;
        /// <summary>
        ///时间特征编码（默认为 timeF）。这可以设置为timeF，fixed，learned, date
        ///Time features encoding (defaults to timeF). This can be set to timeF, fixed, learned
        /// </summary>
        public string embed { get; set; } = "timeF";

        /// <summary>
        ///是否在encoder中输出attention，使用该参数表示输出attention（默认为False）
        ///Whether to output attention in encoder, using this argument means outputing attention (defaults to False)
        /// </summary>
        public bool output_attention { get; set; } = true;
        /// <summary>
        ///激活函数（默认为gelu）
        ///Activation function (defaults to gelu)
        /// </summary>
        public string activation { get; set; } = "gelu";
        #endregion

        #region optimization

        /// <summary>
        ///Data loader 的 num_works（默认为 0）
        ///The num_works of Data loader (defaults to 0)
        /// </summary>
        public int num_workers { get; set; } = 0;
        /// <summary>
        ///训练时期（默认为 6）
        ///Train epochs (defaults to 6)
        /// </summary>
        public int train_epochs { get; set; } = 6;
        /// <summary>
        ///训练输入数据的批量大小（默认为 32）
        ///The batch size of training input data (defaults to 32)
        /// </summary>
        public int batch_size { get; set; } = 32;
        /// <summary>
        ///提前停止耐心（默认为 3）
        ///Early stopping patience (defaults to 3)
        /// </summary>
        public int patience { get; set; } = 3;
        /// <summary>
        ///优化器学习率（默认为 0.0001）
        ///Optimizer learning rate (defaults to 0.0001)
        /// </summary>
        public double learning_rate { get; set; } = 0.0001;
        /// <summary>
        ///调整学习率的方法（默认为 type1）
        ///Ways to adjust the learning rate (defaults to type1)
        /// </summary>
        public string lradj { get; set; } = "type1";

        #endregion

        public int padding { get; set; } = 0;
        public bool mix { get; set; } = false;
        public string attn { get; set; } = "prob";
    }
}
