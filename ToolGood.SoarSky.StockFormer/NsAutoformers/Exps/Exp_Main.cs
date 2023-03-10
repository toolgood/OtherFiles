using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torch;
using TorchSharp;
using ToolGood.SoarSky.StockFormer.NsAutoformers.Models;
using ToolGood.SoarSky.StockFormer.Informers.Models;
using ToolGood.SoarSky.StockFormer.Utils;

namespace ToolGood.SoarSky.StockFormer.NsAutoformers.Exps
{
    public class Exp_Main : Exp_Basic
    {
        public Exp_Main(NsAutoformerConfig args) : base(args)
        {
        }


        private (Tensor, Tensor, Tensor, Tensor) GetTensor(Dictionary<string, Tensor> dict)
        {
            var batch_x = dict["batch_x"];
            var batch_y = dict["batch_y"];
            var batch_x_mark = dict["batch_x_mark"];
            var batch_y_mark = dict["batch_y_mark"];
            return (batch_x, batch_y, batch_x_mark, batch_y_mark);
        }

        public virtual double vali(Dataset vali_data, DataLoader vali_loader, Loss<Tensor, Tensor, Tensor> criterion)
        {
            var total_loss = new List<double>();
            model.eval();
            var f_dim = args.features == "MS" ? -1 : 0;

            using (var temp = no_grad())
            {
                foreach (var dict in vali_loader)
                {
                    var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                    batch_x = batch_x.@float().to(device);
                    batch_y = batch_y.@float().to(device);
                    batch_x_mark = batch_x_mark.@float().to(device);
                    batch_y_mark = batch_y_mark.@float().to(device);

                    // decoder input
                    var dec_inp = zeros_like(batch_y[TensorIndex.Ellipsis, -args.pred_len, TensorIndex.Ellipsis]).@float();
                    dec_inp = cat(new List<Tensor> { batch_y[TensorIndex.Ellipsis, TensorIndex.Slice(null, args.label_len), TensorIndex.Ellipsis], dec_inp }, dim: 1).@float().to(device);
                    // encoder - decoder
                    Tensor outputs = model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark).Item1;

                    batch_y = batch_y[TensorIndex.Ellipsis, -args.pred_len, f_dim].to(device);
                    var pred = outputs.detach();//.cpu();
                    var @true = batch_y.detach();//.cpu();
                    var loss = criterion.forward(pred, @true);
                    total_loss.Add(loss.item<double>());
                }
            }
            var total_loss2 = (double)np.average(np.array(total_loss.ToArray()));
            model.train();
            return total_loss2;
        }

        public virtual nn.Module train(string setting)
        {
            var (train_data, train_loader) = _get_data(flag: "train");
            var (vali_data, vali_loader) = _get_data(flag: "val");
            var (test_data, test_loader) = _get_data(flag: "test");
            var path = os.path.join(args.checkpoints, setting);
            if (!os.path.exists(path))
            {
                os.makedirs(path);
            }

            var time_now = DateTime.Now;
            var train_steps = train_loader.Count;
            var early_stopping = new EarlyStopping(patience: args.patience, verbose: true);
            var model_optim = _select_optimizer();
            var criterion = _select_criterion();

            var f_dim = args.features == "MS" ? -1 : 0;

            foreach (var epoch in Enumerable.Range(0, args.train_epochs))
            {
                var iter_count = 0;
                var train_loss = new List<double>();
                model.train();
                var epoch_time = DateTime.Now;
                int i = -1;
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
                    var dec_inp = zeros_like(batch_y[TensorIndex.Ellipsis, -args.pred_len, TensorIndex.Ellipsis]).@float();
                    dec_inp = cat(new List<Tensor> { batch_y[TensorIndex.Ellipsis, TensorIndex.Slice(null, args.label_len), TensorIndex.Ellipsis], dec_inp }, dim: 1).@float().to(device);
                    // encoder - decoder
                    Tensor outputs = model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark).Item1;

                    batch_y = batch_y[TensorIndex.Ellipsis, -args.pred_len, f_dim].to(device);
                    Tensor loss = criterion.forward(outputs, batch_y);
                    train_loss.Add(loss.item<double>());

                    if ((i + 1) % 100 == 0)
                    {
                        Console.WriteLine(string.Format("\titers: {0}, epoch: {1} | loss: {2:.7f}", i + 1, epoch + 1, (double)loss));
                        var speed = (DateTime.Now - time_now) / iter_count;
                        var left_time = speed * ((args.train_epochs - epoch) * train_steps - i);
                        Console.WriteLine(string.Format("\tspeed: {0:.4f}s/iter; left time: {1:.4f}s", speed, left_time));
                        iter_count = 0;
                        time_now = DateTime.Now;
                    }

                    loss.backward();
                    model_optim.step();
                }
                Console.WriteLine("Epoch: {0} cost time: {1}".format(epoch + 1, DateTime.Now - epoch_time));
                var train_loss2 = np.average(np.array(train_loss.ToArray()));
                var vali_loss = vali(vali_data, vali_loader, criterion);
                var test_loss = vali(test_data, test_loader, criterion);
                Console.WriteLine("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss2, vali_loss, test_loss));
                early_stopping.__call__(vali_loss, model, path);
                if (early_stopping.early_stop)
                {
                    Console.WriteLine("Early stopping");
                    break;
                }
                adjust_learning_rate(model_optim, epoch + 1, args);
            }

            var best_model_path = path + "/" + "checkpoint.pth";
            model.save(best_model_path);
            //model.load_state_dict(torchEx.load(best_model_path));
            return model;
        }

        public virtual void test(string setting, bool test)
        {
            var (test_data, test_loader) = _get_data(flag: "test");

            var preds = new List<Tensor>();
            var trues = new List<Tensor>();
            var folder_path = "./test_results/" + setting + "/";
            if (!os.path.exists(folder_path))
            {
                os.makedirs(folder_path);
            }

            model.eval();
            using (var temp = no_grad())
            {
                int i = -1;
                var f_dim = args.features == "MS" ? -1 : 0;
                foreach (var dict in test_loader)
                {
                    var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);
                    i++;
                    batch_x = batch_x.@float().to(device);
                    batch_y = batch_y.@float().to(device);
                    batch_x_mark = batch_x_mark.@float().to(device);
                    batch_y_mark = batch_y_mark.@float().to(device);
                    // decoder input
                    var dec_inp = zeros_like(batch_y[TensorIndex.Ellipsis, -args.pred_len, TensorIndex.Ellipsis]).@float();
                    dec_inp = cat(new List<Tensor> { batch_y[TensorIndex.Ellipsis, TensorIndex.Slice(null, args.label_len), TensorIndex.Ellipsis], dec_inp }, dim: 1).@float().to(device);
                    // encoder - decoder
                    Tensor outputs = model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark).Item1;

                    batch_y = batch_y[TensorIndex.Ellipsis, -args.pred_len, f_dim].to(device);
                    outputs = outputs.detach();//.cpu().numpy();
                    batch_y = batch_y.detach();//.cpu().numpy();
                    var pred = outputs;
                    var @true = batch_y;
                    preds.Add(pred);
                    trues.Add(@true);
                    if (i % 20 == 0)
                    {
                        //input = batch_x.detach().cpu().numpy();
                        //gt = np.concatenate((input[0, ":", ^1], @true[0, ":", ^1]), axis: 0);
                        //pd = np.concatenate((input[0, ":", ^1], pred[0, ":", ^1]), axis: 0);
                        //visual(gt, pd, os.path.join(folder_path, i.ToString() + ".pdf"));
                    }
                }
            }
            var preds2 = np.array(preds.ToArray());
            var trues2 = np.array(trues.ToArray());
            Console.WriteLine("test shape:", preds2.shape, trues2.shape);
            preds2 = preds2.reshape(-1, (int)preds2.shape.iDims[^2], preds2.shape.iDims[^1]);
            trues2 = trues2.reshape(-1, trues2.shape.iDims[^2], trues2.shape.iDims[^1]);
            Console.WriteLine("test shape:", preds2.shape, trues2.shape);
            // result save
            folder_path = "./results/" + setting + "/";
            if (!os.path.exists(folder_path))
            {
                os.makedirs(folder_path);
            }
            //var (mae, mse, rmse, mape, mspe) = metric(preds, trues);
            //Console.WriteLine("mse:{}, mae:{}".format(mse, mae));
            //var f = open("result.txt", "a");
            //f.write(setting + "  \n");
            //f.write("mse:{}, mae:{}".format(mse, mae));
            //f.write("\n");
            //f.write("\n");
            //f.close();
            //np.save(folder_path + "metrics.npy", np.array(new List<object> {
            //        mae,
            //        mse,
            //        rmse,
            //        mape,
            //        mspe
            //    }));
            //np.save(folder_path + "pred.npy", preds);
            //np.save(folder_path + "true.npy", trues);
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
                //model.load_state_dict(torchEx.load(best_model_path));
            }
            var preds = new List<Tensor>();
            model.eval();
            using (var temp = no_grad())
            {
                foreach (var dict in pred_loader)
                {
                    var (batch_x, batch_y, batch_x_mark, batch_y_mark) = GetTensor(dict);


                    batch_x = batch_x.@float().to(device);
                    batch_y = batch_y.@float().to(device);
                    batch_x_mark = batch_x_mark.@float().to(device);
                    batch_y_mark = batch_y_mark.@float().to(device);
                    // decoder input
                    var dec_inp = zeros_like(batch_y[TensorIndex.Ellipsis, -args.pred_len, TensorIndex.Ellipsis]).@float();
                    dec_inp = cat(new List<Tensor> { batch_y[TensorIndex.Ellipsis, TensorIndex.Slice(null, args.label_len), TensorIndex.Ellipsis], dec_inp }, dim: 1).@float().to(device);
                    // encoder - decoder
                    Tensor outputs = model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark).Item1;
                    var pred = outputs.detach();
                    preds.Add(pred);
                }
            }
            //preds = np.array(preds);
            //preds = preds.reshape(-1, preds.shape[^2], preds.shape[^1]);
            //// result save
            //var folder_path = "./results/" + setting + "/";
            //if (!os.path.exists(folder_path)) {
            //    os.makedirs(folder_path);
            //}
            //np.save(folder_path + "real_prediction.npy", preds);
            return;
        }
    }

}
