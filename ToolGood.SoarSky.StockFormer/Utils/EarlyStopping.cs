﻿using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;

namespace ToolGood.SoarSky.StockFormer.Utils
{
    public class EarlyStopping
    {
        public double? best_score;
        public int counter;
        public double delta;
        public bool early_stop;
        public int patience;
        public double val_loss_min;
        public bool verbose;

        public EarlyStopping(int patience = 7, bool verbose = false, double delta = 0)
        {
            this.patience = patience;
            this.verbose = verbose;
            this.counter = 0;
            this.best_score = null;
            this.early_stop = false;
            this.val_loss_min = np.Inf;
            this.delta = delta;
        }

        public virtual void @__call__(double val_loss, nn.Module model, string path)
        {
            var score = -val_loss;
            if (best_score is null) {
                best_score = score;
                save_checkpoint(val_loss, model, path);
            } else if (score < best_score + delta) {
                counter += 1;
                Console.WriteLine($"EarlyStopping counter: {counter} out of {patience}");
                if (counter >= patience) {
                    early_stop = true;
                }
            } else {
                best_score = score;
                save_checkpoint(val_loss, model, path);
                counter = 0;
            }
        }

        public virtual void save_checkpoint(double val_loss, nn.Module model, string path)
        {
            if (verbose) {
                Console.WriteLine($"Validation loss decreased ({val_loss_min} --> {val_loss}).  Saving model ...");
            }
            model.save(path + "/" + "checkpoint.pth");
            this.val_loss_min = val_loss;
        }
    }

}
