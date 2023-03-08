using NumpyDotNet;
using System.Collections;
using ToolGood.SoarSky.StockFormer.DataProvider;
using ToolGood.SoarSky.StockFormer.PatchTSTs.Models;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using static TorchSharp.torch.utils.data;

namespace ToolGood.SoarSky.StockFormer.PatchTSTs.Exps
{
    public class Exp_Main
    {
        public PatchTST model;
        public Device device;
        public ExpConfig args;

        public Exp_Main(ExpConfig args)
        {
            device = cuda.is_available() ? CUDA : CPU;
            this.args = args;
            model = _build_model().to(device);
        }

        public virtual PatchTST _build_model()
        {
            var model = new PatchTST(args,
                max_seq_len: 1024,
                 d_k: null,
                 d_v: null,
                 norm: "BatchNorm",
                 attn_dropout: 0.0,
                 act: "gelu",
                 key_padding_mask: "auto",
                 padding_var: null,
                 attn_mask: null,
                 res_attention: true,
                 pre_norm: false,
                 store_attn: false,
                 pe: "zeros",
                 learn_pe: true,
                 pretrain_head: false,
                 head_type: "flatten",
                 verbose: false,
                 kwargs: null
                );

            return model;
        }

        public virtual (Dataset, DataLoader) _get_data(string flag)
        {
            var (data_set, data_loader) = DataFactory.data_provider(args, flag);
            return (data_set, data_loader);
        }

        public virtual OptimizerHelper _select_optimizer()
        {
            var model_optim = Adam(model.parameters(), lr: args.learning_rate);
            return model_optim;
        }

        public virtual Loss<Tensor, Tensor, Tensor> _select_criterion()
        {
            var criterion = nn.MSELoss();
            return criterion;
        }
        private (Tensor, Tensor, Tensor, Tensor) GetTensor(Dictionary<string, Tensor> dict)
        {
            var batch_x = dict["batch_x"];
            var batch_y = dict["batch_y"];
            var batch_x_mark = dict["batch_x_mark"];
            var batch_y_mark = dict["batch_y_mark"];
            return (batch_x, batch_y, batch_x_mark, batch_y_mark);
        }

        public virtual double vali(object vali_data, DataLoader vali_loader, Loss<Tensor, Tensor, Tensor> criterion)
        {
            var total_loss = new List<double>();
            model.eval();
            using (var _no_grad = no_grad())
            {
                foreach (var dict in vali_loader)
                {
                    var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                    batch_x = batch_x.@float().to(device);
                    batch_y = batch_y.@float();
                    batch_x_mark = batch_x_mark.@float().to(device);
                    batch_y_mark = batch_y_mark.@float().to(device);
                    // decoder input
                    var dec_inp = zeros_like(batch_y[TensorIndex.Ellipsis, -args.pred_len, TensorIndex.Ellipsis]).@float();
                    dec_inp = cat(new List<Tensor> { batch_y[TensorIndex.Ellipsis, TensorIndex.Slice(null, args.label_len, null), TensorIndex.Ellipsis], dec_inp }, dim: 1).@float().to(device);
                    // encoder - decoder
                    var outputs = model.forward(batch_x);

                    var f_dim = args.features == "MS" ? -1 : 0;
                    outputs = outputs[TensorIndex.Ellipsis, -args.pred_len, f_dim];
                    batch_y = batch_y[TensorIndex.Ellipsis, -args.pred_len, f_dim].to(device);
                    var pred = outputs.detach().cpu();
                    var @true = batch_y.detach().cpu();
                    var loss = criterion.forward(pred, @true);
                    total_loss.append(loss.item<double>());
                }
            }
            var total_loss2 = total_loss.Average();
            model.train();
            return total_loss2;
        }

        public virtual PatchTST train(string setting)
        {
            Tensor outputs;
            var (train_data, train_loader) = _get_data(flag: "train");
            var (vali_data, vali_loader) = _get_data(flag: "val");
            var (test_data, test_loader) = _get_data(flag: "test");
            var path = os.path.join(args.checkpoints, setting);
            if (!os.path.exists(path))
            {
                os.makedirs(path);
            }
            var time_now = DateTime.Now;
            var train_steps = (int)train_loader.Count;
            var early_stopping = new EarlyStopping(patience: args.patience, verbose: true);
            var model_optim = _select_optimizer();
            var criterion = _select_criterion();

            var scheduler = OneCycleLR(optimizer: model_optim, steps_per_epoch: train_steps, pct_start: args.pct_start, epochs: args.train_epochs, max_lr: args.learning_rate);
            foreach (var epoch in Enumerable.Range(0, args.train_epochs))
            {
                var iter_count = 0;
                var train_loss = new List<double>();
                model.train();
                var epoch_time = DateTime.Now;
                var i = -1;
                foreach (var dict in train_loader)
                {
                    var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                    i++;
                    iter_count += 1;
                    model_optim.zero_grad();
                    batch_x = batch_x.@float().to(device);
                    batch_y = batch_y.@float().to(device);
                    batch_x_mark = batch_x_mark.@float().to(device);
                    batch_y_mark = batch_y_mark.@float().to(device);
                    // decoder input
                    //var _pred_len = batch_y.size(1) - 1 - this.args.pred_len;
                    var dec_inp = zeros_like(batch_y[TensorIndex.Ellipsis, TensorIndex.Slice(-args.pred_len), TensorIndex.Ellipsis]).@float();
                    dec_inp = cat(new List<Tensor> { batch_y[TensorIndex.Ellipsis, TensorIndex.Slice(null, args.label_len, null), TensorIndex.Ellipsis], dec_inp }, dim: 1).@float().to(device);
                    // encoder - decoder

                    outputs = model.forward(batch_x);
                    // print(outputs.shape,batch_y.shape)
                    var f_dim = args.features == "MS" ? -1 : 0;
                    outputs = outputs[TensorIndex.Ellipsis, -args.pred_len, f_dim];
                    batch_y = batch_y[TensorIndex.Ellipsis, -args.pred_len, f_dim].to(device);
                    var loss = criterion.forward(outputs, batch_y);
                    train_loss.append(loss.item<double>());

                    if ((i + 1) % 100 == 0)
                    {
                        Console.WriteLine("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item<double>()));
                        var speed = (DateTime.Now - time_now) / iter_count;
                        var left_time = speed * ((args.train_epochs - epoch) * train_steps - i);
                        Console.WriteLine("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time));
                        iter_count = 0;
                        time_now = DateTime.Now;
                    }

                    loss.backward();
                    model_optim.step();

                    if (args.lradj == "TST")
                    {
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout: false);
                        scheduler.step();
                    }
                }
                Console.WriteLine("Epoch: {} cost time: {}".format(epoch + 1, DateTime.Now - epoch_time));
                var train_loss2 = train_loss.Average();
                var vali_loss = vali(vali_data, vali_loader, criterion);
                var test_loss = vali(test_data, test_loader, criterion);
                Console.WriteLine("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss));
                early_stopping.__call__(vali_loss, model, path);
                if (early_stopping.early_stop)
                {
                    Console.WriteLine("Early stopping");
                    break;
                }
                if (args.lradj != "TST")
                {
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, args);
                }
                else
                {
                    Console.WriteLine("Updating learning rate to {}".format(scheduler.get_last_lr()[0]));
                }
            }
            var best_model_path = path + "/" + "checkpoint.pth";
            model.load(best_model_path);
            return model;
        }

        public virtual void test(string setting, bool test = false)
        {
            var (test_data, test_loader) = _get_data(flag: "test");
            if (test)
            {
                Console.WriteLine("loading model");
                var best_model_path = os.path.join("./checkpoints/" + setting, "checkpoint.pth");
                model.load(best_model_path);
                //this.model.load_state_dict(torch.load(best_model_path));
            }
            var preds = new List<Tensor>();
            var trues = new List<Tensor>();
            var inputx = new List<Tensor>();
            var folder_path = "./test_results/" + setting + "/";
            if (!os.path.exists(folder_path))
            {
                os.makedirs(folder_path);
            }
            model.eval();
            Tensor batch_x = null;
            using (var _no_grad = no_grad())
            {
                var i = -1;
                foreach (var dict in test_loader)
                {
                    (batch_x, var batch_y, var batch_x_mark, var batch_y_mark) = GetTensor(dict);
                    i++;
                    batch_x = batch_x.@float().to(device);
                    batch_y = batch_y.@float().to(device);
                    batch_x_mark = batch_x_mark.@float().to(device);
                    batch_y_mark = batch_y_mark.@float().to(device);
                    // decoder input
                    var dec_inp = zeros_like(batch_y[TensorIndex.Ellipsis, -args.pred_len, TensorIndex.Ellipsis]).@float();
                    dec_inp = cat(new List<Tensor> {
                            batch_y[TensorIndex.Ellipsis,TensorIndex.Slice(null,args.label_len,null),TensorIndex.Ellipsis],
                            dec_inp
                        }, dim: 1).@float().to(device);
                    // encoder - decoder
                    var outputs = model.forward(batch_x);


                    var f_dim = args.features == "MS" ? -1 : 0;
                    // print(outputs.shape,batch_y.shape)
                    outputs = outputs[TensorIndex.Ellipsis, -args.pred_len, f_dim];
                    batch_y = batch_y[TensorIndex.Ellipsis, -args.pred_len, f_dim].to(device);
                    outputs = outputs.detach().cpu();//.numpy();
                    batch_y = batch_y.detach().cpu();//.numpy();
                    var pred = outputs;
                    var @true = batch_y;
                    preds.append(pred);
                    trues.append(@true);
                    inputx.append(batch_x.detach().cpu()/*.numpy()*/);
                    if (i % 20 == 0)
                    {
                        var input = batch_x.detach().cpu();//.numpy();
                        //var gt = np.concatenate((input[0, TensorIndex.Ellipsis, ^1], @true[0, TensorIndex.Ellipsis, ^1]), axis: 0);
                        //var pd = np.concatenate((input[0, TensorIndex.Ellipsis, ^1], pred[0, TensorIndex.Ellipsis, ^1]), axis: 0);
                        //   visual(gt, pd, os.path.join(folder_path, i.ToString() + ".pdf"));
                    }
                }
            }
            //if (this.args.test_flop) {
            //    test_params_flop((batch_x.shape[1], batch_x.shape[2]));
            //    return;
            //    //exit();
            //}
            //preds = np.array(preds);
            //trues = np.array(trues);
            //inputx = np.array(inputx);
            //preds = preds.reshape(-1, preds.shape[^2], preds.shape[^1]);
            //trues = trues.reshape(-1, trues.shape[^2], trues.shape[^1]);
            //inputx = inputx.reshape(-1, inputx.shape[^2], inputx.shape[^1]);
            //// result save
            //folder_path = "./results/" + setting + "/";
            //if (!os.path.exists(folder_path)) {
            //    os.makedirs(folder_path);
            //}
            //var (mae, mse, rmse, mape, mspe, rse, corr) = Metrics.metric(preds, trues);
            //Console.WriteLine("mse:{}, mae:{}, rse:{}".format(mse, mae, rse));
            ////var f = open("result.txt", "a");
            ////f.write(setting + "  \n");
            ////f.write("mse:{}, mae:{}, rse:{}".format(mse, mae, rse));
            ////f.write("\n");
            ////f.write("\n");
            ////f.close();
            //// np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
            //// np.save(folder_path + "pred.npy", preds);
            //// np.save(folder_path + 'true.npy', trues)
            //// np.save(folder_path + 'x.npy', inputx)
            return;
        }

        public virtual void predict(string setting, bool load = false)
        {
            var (pred_data, pred_loader) = _get_data(flag: "pred");
            if (load)
            {
                var path = os.path.join(args.checkpoints, setting);
                var best_model_path = path + "/" + "checkpoint.pth";

                model.load(best_model_path);
                //this.model.load_state_dict(torch.load(best_model_path));
            }
            var preds = new List<Tensor>();
            model.eval();
            using (var _no_grad = no_grad())
            {
                var i = -1;
                foreach (var dict in pred_loader)
                {
                    (var batch_x, var batch_y, var batch_x_mark, var batch_y_mark) = GetTensor(dict);
                    i++;

                    batch_x = batch_x.@float().to(device);
                    batch_y = batch_y.@float();
                    batch_x_mark = batch_x_mark.@float().to(device);
                    batch_y_mark = batch_y_mark.@float().to(device);
                    // decoder input
                    var dec_inp = zeros(new long[] {
                            batch_y.shape[0],
                            args.pred_len,
                            batch_y.shape[2]
                        }).@float().to(batch_y.device);
                    dec_inp = cat(new List<Tensor> {
                            batch_y[TensorIndex.Ellipsis,TensorIndex.Slice(null,args.label_len,null),TensorIndex.Ellipsis],
                            dec_inp
                        }, dim: 1).@float().to(device);
                    // encoder - decoder
                    var outputs = model.forward(batch_x);
                    var pred = outputs.detach().cpu();//.numpy();
                    preds.append(pred);
                }
            }
            var preds2 = cat(preds, 1);
            preds2 = preds2.reshape(-1, preds2.shape[^2], preds2.shape[^1]);
            // result save
            var folder_path = "./results/" + setting + "/";
            if (!os.path.exists(folder_path))
            {
                os.makedirs(folder_path);
            }
            //np.save(folder_path + "real_prediction.npy", preds);
            return;
        }

        public static void test_params_flop(nn.Module model, (Tensor, Tensor) x_shape)
        {
            long model_params = 0;
            foreach (var parameter in model.parameters())
            {
                model_params += parameter.numel();
                Console.WriteLine("INFO: Trainable parameter count: {:.2f}M".format(model_params / 1000000.0));
            }
            //using (var torch.cuda.device(0)) {
            //    (macs, @params) = get_model_complexity_info(model.cuda(), x_shape, as_strings: true, print_per_layer_stat: true);
            //    // print('Flops:' + flops)
            //    // print('Params:' + params)
            //    Console.WriteLine("{:<30}  {:<8}".format("Computational complexity: ", macs));
            //    Console.WriteLine("{:<30}  {:<8}".format("Number of parameters: ", @params));
            //}
        }


        public static void adjust_learning_rate(OptimizerHelper optimizer, LRScheduler scheduler, int epoch, ExpConfig args,
                                                bool printout = true)
        {
            Dictionary<int, double> lr_adjust = null;
            // lr = args.learning_rate * (0.2 ** (epoch // 2))
            if (args.lradj == "type1")
            {
                lr_adjust = new Dictionary<int, double> { { epoch, args.learning_rate * Math.Pow(0.5, (epoch - 1) / 1) } };
            }
            else if (args.lradj == "type2")
            {
                lr_adjust = new Dictionary<int, double> {
                    { 2, 5E-05},
                    { 4, 1E-05},
                    { 6, 5E-06},
                    { 8, 1E-06},
                    { 10, 5E-07},
                    { 15, 1E-07},
                    { 20, 5E-08}};
            }
            else if (args.lradj == "type3")
            {
                lr_adjust = new Dictionary<int, double> {
                    { epoch, epoch < 3 ? args.learning_rate : args.learning_rate * Math.Pow(0.9, (epoch - 3) / 1)}};
            }
            else if (args.lradj == "constant")
            {
                lr_adjust = new Dictionary<int, double> {
                    { epoch, args.learning_rate}};
            }
            else if (args.lradj == "3")
            {
                lr_adjust = new Dictionary<int, double> {
                    { epoch, epoch < 10 ? args.learning_rate : args.learning_rate * 0.1}};
            }
            else if (args.lradj == "4")
            {
                lr_adjust = new Dictionary<int, double> {
                    { epoch, epoch < 15 ? args.learning_rate : args.learning_rate * 0.1}};
            }
            else if (args.lradj == "5")
            {
                lr_adjust = new Dictionary<int, double> {
                    { epoch, epoch < 25 ? args.learning_rate : args.learning_rate * 0.1}};
            }
            else if (args.lradj == "6")
            {
                lr_adjust = new Dictionary<int, double> {
                    { epoch, epoch < 5 ? args.learning_rate : args.learning_rate * 0.1}};
            }
            else if (args.lradj == "TST")
            {
                lr_adjust = new Dictionary<int, double> {
                    { epoch, scheduler.get_last_lr()[0]}};
            }
            if (lr_adjust.keys().Contains(epoch))
            {
                var lr = lr_adjust[epoch];
                foreach (var param_group in optimizer.ParamGroups)
                {
                    param_group.LearningRate = lr;
                    //param_group["lr"] = lr;
                }
                if (printout)
                {
                    Console.WriteLine("Updating learning rate to {}".format(lr));
                }
            }
        }
    }
    public static class LRSchedulerExt
    {
        public static IList<double> get_last_lr(this LRScheduler lrScheduler)
        {
            var type = typeof(LRScheduler);
            var fi = type.GetField("get_last_lr", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            return fi.GetValue(lrScheduler) as IList<double>;
        }


    }

}
