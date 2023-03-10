using ToolGood.SoarSky.StockFormer.DataProvider;
using ToolGood.SoarSky.StockFormer.Informers.Models;
using ToolGood.SoarSky.StockFormer.NsAutoformers.Models;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;

namespace ToolGood.SoarSky.StockFormer.NsAutoformers.Exps
{
    public abstract class Exp_Basic
    {
        public NsAutoformerConfig args;
        public Device device;
        public NsAutoformer model;

        public Exp_Basic(NsAutoformerConfig args)
        {
            this.args = args;
            device = cuda.is_available() ? CUDA : CPU;

            model = _build_model();
            model.to(device);
        }
        public virtual NsAutoformer _build_model()
        {
            return new NsAutoformer(args);
        }

        public virtual (Dataset, DataLoader) _get_data(string flag)
        {
            return DataFactory.data_provider(args, flag);
        }

        public virtual OptimizerHelper _select_optimizer()
        {
            return optim.Adam(model.parameters(), lr: args.learning_rate);
        }

        public virtual Loss<Tensor, Tensor, Tensor> _select_criterion()
        {
            return nn.MSELoss();
        }



        protected virtual void adjust_learning_rate(OptimizerHelper optimizer, int epoch, NsAutoformerConfig args)
        {
            Dictionary<int, double> lr_adjust;
            if (args.lradj == "type1") {
                lr_adjust = new Dictionary<int, double> { { epoch, args.learning_rate * Math.Pow(0.5, (epoch - 1) / 1) } };
            } else if (args.lradj == "type2") {
                lr_adjust = new Dictionary<int, double> {
                    { 2, 5E-05},
                    { 4, 1E-05},
                    { 6, 5E-06},
                    { 8, 1E-06},
                    { 10, 5E-07},
                    { 15, 1E-07},
                    { 20, 5E-08}};
            } else if (args.lradj == "type3") {
                lr_adjust = new Dictionary<int, double> { { epoch, args.learning_rate } };
            } else if (args.lradj == "type4") {
                lr_adjust = new Dictionary<int, double> { { epoch, args.learning_rate * Math.Pow(0.9, (epoch - 1) / 1) } };
            } else {
                lr_adjust = new Dictionary<int, double> { { epoch, args.learning_rate * Math.Pow(0.95, (epoch - 1) / 1) } };
            }
            if (lr_adjust.keys().Contains(epoch)) {
                var lr = lr_adjust[epoch];
                foreach (var param_group in optimizer.ParamGroups) {
                    param_group.LearningRate = lr;
                    //param_group["lr"] = lr;
                }
                Console.WriteLine("Updating learning rate to {0}".format(lr));
            }
        }
    }

}
