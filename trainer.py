import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from time import time
import sys
from datetime import datetime, timezone, timedelta

class Trainer(object):
    def __init__(self, batch_size, val_batch_size, num_workers, train_dataset, val_dataset, model, model_name,
                  max_epoch=200, initial_lr=0.001,
                 model_path = 'CheckpointAndLog/',
                 log_path = 'CheckpointAndLog/'
                ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.model_name = model_name
        self.max_epoch = max_epoch
        self.initial_lr = initial_lr

        self.model = model.to(self.device)


        self.best_val_loss = 2 ** 31
        self.log_file = None

        self.all_train_loss = []
        self.all_val_loss = []

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       drop_last=True,
                                       num_workers=self.num_workers,
                                       sampler=None)

        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=True,
                                     num_workers=self.num_workers)


        self.best_model_save_path = model_path + self.model_name + '/CheckPoint/'
        if not os.path.exists(self.best_model_save_path):
            os.makedirs(self.best_model_save_path)

        self.log_save_path = log_path + self.model_name + '/Log/'
        self.fig_save_path = log_path + self.model_name + '/Log/'
        if not os.path.exists(self.log_save_path):
            os.makedirs(self.log_save_path)

        self.save_log("batch_size: " ,batch_size, "  lr: ",initial_lr)

    def train_step(self, epoch):
        self.save_log("\nEpoch: ", epoch + 1)
        self.model.train()
        train_loss = 0

        for idx, (source_data, source_mask, target_data) in enumerate(self.train_loader):
            source_data = source_data.to(self.device)
            source_mask = source_mask.to(self.device)
            target_data = target_data.to(self.device)


            train_step_loss = self.model.train_step(source_data,
                                  source_mask,
                                  target_data,
                                  epoch = epoch,
                                  idx = idx,
                                  dir = self.log_save_path)
            train_loss += train_step_loss

        train_mean_loss = train_loss / len(self.train_loader)


        log_str = "Train Loss:{:.5f} ".format(train_mean_loss)
        self.save_log(log_str)

        return train_mean_loss

    def val_step(self, epoch):
        torch.cuda.empty_cache()

        print("val epoch step:", epoch + 1)
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for idx, (patch_data, label_data) in enumerate(self.val_loader):

                patch_data, label_data = patch_data.to(self.device), label_data.to(self.device)
                step_val_loss = self.model.val_step(patch_data, label_data)
                val_loss += step_val_loss

        val_mean_loss = val_loss / len(self.val_loader)
        print("best val loss: ", self.best_val_loss)

        eval_dic = {'loss': val_mean_loss}
        log_str = "Val Loss:{:.5f} ".format(val_mean_loss)
        self.save_log(log_str)

        if self.best_val_loss > val_mean_loss:
            self.best_val_loss = val_mean_loss
            model_file = self.best_model_save_path + self.model_name + "_" + str(self.max_epoch) + "_" + str(self.initial_lr) + ".pkl"
            self.save_best_checkpoint(model_file, epoch)

            log_str = "Saving parameters to " + model_file
            self.save_log(log_str)

        return eval_dic

    def poly_lr(self, epoch, max_epochs, initial_lr, exponent=0.9):
        return initial_lr * (1 - epoch / max_epochs) ** exponent

    def lr_decay(self, epoch, max_epochs, initial_lr):
        lr = self.poly_lr(epoch, max_epochs, initial_lr, exponent=1.5)
        self.model.updata_lr(lr)
        print_str = "Learning rate adjusted to {}".format(lr)
        self.save_log(print_str)

    def plot_progress(self):

        x_epoch = list(range(len(self.all_train_loss)))

        plt.plot(x_epoch, self.all_train_loss, color="b", linestyle="--", marker="*", label='train')
        plt.plot(x_epoch, self.all_val_loss, color="r", linestyle="--", marker="*", label='val')
        plt.legend()
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        plt.savefig(self.fig_save_path + "Total_loss_" + self.model_name + "_" + str(self.max_epoch) + "_" + str(self.initial_lr) + ".jpg")
        plt.close()

    def save_best_checkpoint(self, model_file, epoch):
        check_point = {
            'net_dict': self.model.state_dict(),
            'epoch': epoch,
            'batch_size': self.batch_size,
            'train_loss': self.all_train_loss,
            'val_loss': self.all_val_loss,
            'initial_lr': self.initial_lr
        }
        torch.save(check_point, model_file)

    def save_log(self, *args, also_print_to_console=True, add_timestamp=True):
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp,tz=timezone(timedelta(hours=+8)))

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            if not os.path.isdir(self.log_save_path):
                os.mkdir(self.log_save_path)
            timestamp = datetime.now()
            self.log_file = os.path.join(self.log_save_path, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                ctr += 1
        if also_print_to_console:
            print(*args)

    def run_train(self):
        for epoch in range(0, self.max_epoch):
            train_loss = self.train_step(epoch)
            self.all_train_loss.append(train_loss)

            eval_dict = self.val_step(epoch)
            val_loss = eval_dict['loss']
            self.all_val_loss.append(val_loss)

            self.lr_decay(epoch, self.max_epoch, self.initial_lr)
            self.plot_progress()
