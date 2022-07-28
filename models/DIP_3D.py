import torch
import torch.nn as nn
import pytorch_lightning as pl

import os
if (os.path.isfile(os.getcwd() + "/seed.txt")):
    with open(os.getcwd() + "/seed.txt", 'r') as file:
        random_seed = file.read().rstrip()
    if (eval(random_seed)):
        pl.seed_everything(2)
        
class DIP_3D(pl.LightningModule):

    def __init__(self, hyperparameters_config, method):
        super().__init__()

        # Defining variables from hyperparameters_config
        self.lr = hyperparameters_config['lr']
        self.opti_DIP = hyperparameters_config['opti_DIP']
        self.sub_iter_DIP = hyperparameters_config['sub_iter_DIP']
        self.skip = hyperparameters_config['skip_connections']
        self.method = method
        if (hyperparameters_config['mlem_sequence'] is None):
            self.post_reco_mode = True
            self.suffix = self.suffix_func(hyperparameters_config)
        else:
            self.post_reco_mode = False

        # Defining CNN variables
        L_relu = 0.2
        num_channel = [16, 32, 64, 128]
        pad = [0, 0, 0]

        # Layers in CNN architecture
        self.deep1 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(1, num_channel[0], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], num_channel[0], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.down1 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], num_channel[0], 3, stride=(2, 2, 2), padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.deep2 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], num_channel[1], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[1], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.down2 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[1], 3, stride=(2, 2, 2), padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.deep3 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[2], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[2], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.down3 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[2], 3, stride=(2, 2, 2), padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.deep4 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[3], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[3]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[3], num_channel[3], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[3]),
                                   nn.LeakyReLU(L_relu))

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
                                 nn.ReplicationPad3d(1),
                                 nn.Conv3d(num_channel[3], num_channel[2], 3, stride=(1, 1, 1), padding=pad[0]),
                                 nn.BatchNorm3d(num_channel[2]),
                                 nn.LeakyReLU(L_relu))

        self.deep5 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[2], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[2], num_channel[2], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
                                 nn.ReplicationPad3d(1),
                                 nn.Conv3d(num_channel[2], num_channel[1], 3, stride=(1, 1, 1), padding=pad[0]),
                                 nn.BatchNorm3d(num_channel[1]),
                                 nn.LeakyReLU(L_relu))

        self.deep6 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[1], (3, 3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[1], num_channel[1], (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
                                 nn.ReplicationPad3d(1),
                                 nn.Conv3d(num_channel[1], num_channel[0], 3, stride=(1, 1, 1), padding=pad[0]),
                                 nn.BatchNorm3d(num_channel[0]),
                                 nn.LeakyReLU(L_relu))

        self.deep7 = nn.Sequential(nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], num_channel[0], (3, 3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm3d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad3d(1),
                                   nn.Conv3d(num_channel[0], 1, (3, 3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm3d(1))

        self.positivity = nn.ReLU() # Final ReLU to enforce positivity of ouput image
        # self.positivity = nn.SiLU() # Final SiLU, smoother than ReLU but not positive
        # self.positivity = nn.Softplus() # Final SiLU to enforce positivity of ouput image, smoother than ReLU

    def forward(self, x):
        # Encoder
        out1 = self.deep1(x)
        out = self.down1(out1)
        out2 = self.deep2(out)
        out = self.down2(out2)
        out3 = self.deep3(out)
        out = self.down3(out3)
        out = self.deep4(out)

        # Decoder
        out = self.up1(out)
        if (self.skip >= 1):
            out_skip1 = out3 + out
            out = self.deep5(out_skip1)
        else:
            out = self.deep5(out)
        out = self.up2(out)
        if (self.skip >= 2):
            out_skip2 = out2 + out
            out = self.deep6(out_skip2)
        else:
            out = self.deep6(out)
        out = self.up3(out)
        if (self.skip >= 3):
            out_skip3 = out1 + out
            out = self.deep7(out_skip3)
        else:
            out = self.deep7(out)

        if (self.method == 'Gong'):
            out = self.positivity(out)
        return out

    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD

    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        out = self.forward(image_net_input_torch)
        # Save image over epochs
        if (self.post_reco_mode):
            self.post_reco(out)
        loss = self.DIP_loss(out, image_corrupt_torch)
        # logging using tensorboard logger
        self.logger.experiment.add_scalar('loss', loss,self.current_epoch)        
        return loss

    def configure_optimizers(self):
        # Optimization algorithm according to command line

        """
        Optimization of the DNN with SGLD
        """

        if (self.opti_DIP == 'Adam'):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) # Optimizing using Adam
        elif (self.opti_DIP == 'LBFGS' or self.opti_DIP is None): # None means no argument was given in command line
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr, history_size=10, max_iter=4) # Optimizing using L-BFGS
        return optimizer

    def post_reco(self,out):
        from utils.utils_func import save_img
        if ((self.current_epoch%(self.sub_iter_DIP // 10) == 0)):
            try:
                out_np = out.detach().numpy()[0,0,:,:]
            except:
                out_np = out.cpu().detach().numpy()[0,0,:,:]
            subroot = '/home/meraslia/sgld/hernan_folder/data/Algo/'
            experiment = 24
            save_img(out_np, subroot+'Block2/out_cnn/' + format(experiment) + '/out_' + 'DIP' + '_post_reco_epoch=' + format(self.current_epoch) + self.suffix + '.img') # The saved images are not destandardized !!!!!! Do it when showing images in tensorboard
                    
    def suffix_func(self,hyperparameters_config):
        suffix = "config"
        for key, value in hyperparameters_config.items():
            suffix +=  "_" + key + "=" + str(value)
        return suffix