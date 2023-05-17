import hyperspy.api as hys
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F

class load_data():
    def __init__(self, adr, dat_dim, dat_unit, cr_range=None, dat_scale=1, rescale=True, DM_file=False):
        self.file_adr = adr
        self.num_img = len(adr)
        self.dat_dim = dat_dim
        if dat_dim == 4:
            cr_range = None
        self.dat_unit = dat_unit
        self.cr_range = cr_range
        
        if cr_range:
            self.dat_dim_range = np.arange(cr_range[0], cr_range[1], cr_range[2]) * dat_scale
            self.num_dim = len(self.dat_dim_range)
        
        if dat_dim == 3:
            self.data_storage, self.data_shape = data_load_3d(adr, cr_range, rescale, DM_file)
        
        else:
            self.data_storage, self.data_shape = data_load_4d(adr, rescale)
            
        self.original_data_shape = self.data_shape.copy()
             
    def binning(self, bin_y, bin_x, str_y, str_x, offset=0, rescale_0to1=True):
        dataset = []
        data_shape_new = []
        
        for img in self.data_storage:
            print(img.shape)
            processed = binning_SI(img, bin_y, bin_x, str_y, str_x, offset, self.num_dim, rescale_0to1) # include the step for re-scaling the actual input
            print(processed.shape)
            data_shape_new.append(processed.shape)
            dataset.append(processed)

        data_shape_new = np.asarray(data_shape_new)
        print(data_shape_new)
        
        self.data_storage = dataset
        self.data_shape = data_shape_new
        
    def find_center(self, cbox_edge, center_remove, result_visual=True, log_scale=True):
        if self.dat_dim != 4:
            print("data dimension error")
            return
        
        self.center_pos = []
        
        for i in range(self.num_img):
            mean_dp = np.mean(self.data_storage[i], axis=(0, 1))
            cbox_outy = int(mean_dp.shape[0]/2 - cbox_edge/2)
            cbox_outx = int(mean_dp.shape[1]/2 - cbox_edge/2)
            center_box = mean_dp[cbox_outy:-cbox_outy, cbox_outx:-cbox_outx]
            Y, X = np.indices(center_box.shape)
            com_y = np.sum(center_box * Y) / np.sum(center_box)
            com_x = np.sum(center_box * X) / np.sum(center_box)
            c_pos = [np.around(com_y+cbox_outy), np.around(com_x+cbox_outx)]
            self.center_pos.append(c_pos)
        print(self.center_pos)
        
        if result_visual:
            np.seterr(divide='ignore')
            for i in range(self.num_img):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                if log_scale:
                    ax.imshow(np.log(np.mean(self.data_storage[i], axis=(0, 1))), cmap="viridis")
                else:
                    ax.imshow(np.mean(self.data_storage[i], axis=(0, 1)), cmap="viridis")
                ax.scatter(self.center_pos[i][1], self.center_pos[i][0], c="r", s=10)
                ax.axis("off")
                plt.show()
        
        if center_remove != 0:
            self.center_removed = True
            data_cr = []
            for i in range(self.num_img):
                ri = radial_indices(self.data_storage[i].shape[2:], [center_remove, 100], center=self.center_pos[i])
                data_cr.append(np.multiply(self.data_storage[i], ri))
                
            self.data_storage = data_cr
            
            if result_visual:
                for i in range(self.num_img):
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                    if log_scale:
                        ax.imshow(np.log(np.mean(self.data_storage[i], axis=(0, 1))), cmap="viridis")
                    else:
                        ax.imshow(np.mean(self.data_storage[i], axis=(0, 1)), cmap="viridis")
                    ax.scatter(self.center_pos[i][1], self.center_pos[i][0], c="r", s=10)
                    ax.axis("off")
                    plt.show()
                    
    def make_input(self, min_val=0.0, max_normalize=True, rescale_0to1=False, log_scale=False, radial_flat=True, w_size=0, radial_range=None, final_dim=2):

        dataset_flat = []
        if self.dat_dim == 3:
            for i in range(self.num_img):
                dataset_flat.extend(self.data_storage[i].clip(min=min_val).reshape(-1, self.num_dim).tolist())

            dataset_flat = np.asarray(dataset_flat)
            print(dataset_flat.shape)
            
            
        if self.dat_dim == 4:
            self.radial_flat = radial_flat
            self.w_size = w_size
            self.radial_range = radial_range
            
            dataset = []
            
            if radial_flat:
                self.k_indx = []
                self.k_indy = []
                self.a_ind = []

                for r in range(radial_range[0], radial_range[1], radial_range[2]):
                    tmp_k, tmp_a = indices_at_r((radial_range[1]*2, radial_range[1]*2), r, (radial_range[1], radial_range[1]))
                    self.k_indx.extend(tmp_k[0].tolist())
                    self.k_indy.extend(tmp_k[1].tolist())
                    self.a_ind.extend(tmp_a.tolist())

                self.s_length = len(self.k_indx)
                
                if final_dim == 1:

                    for i in range(self.num_img):
                        flattened = circle_flatten(self.data_storage[i], radial_range, self.center_pos[i])

                        tmp = np.zeros((radial_range[1]*2, radial_range[1]*2))
                        tmp[self.k_indy, self.k_indx] = np.sum(flattened, axis=(0, 1))

                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        ax.imshow(tmp, cmap="viridis")
                        ax.axis("off")
                        fig.tight_layout()
                        plt.show()

                        dataset.append(flattened)
                        
                else:
                    for i in range(self.num_img):
                        flattened = circle_flatten(self.data_storage[i], radial_range, self.center_pos[i])

                        tmp = np.zeros((radial_range[1]*2, radial_range[1]*2))
                        tmp[self.k_indy, self.k_indx] = np.sum(flattened, axis=(0, 1))

                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        ax.imshow(tmp, cmap="viridis")
                        ax.axis("off")
                        fig.tight_layout()
                        plt.show()

                        dataset.append(flattened)
                    
                
            else:
                for i in range(self.num_img):
                    flattened = flattening(self.data_storage[i], flat_option="box", crop_dist=w_size, c_pos=self.center_pos[i])
                    if final_dim == 1:
                        dataset.append(flattened)
                    else:
                        dataset.append(flattened.reshape(self.data_shape[i][0], self.data_shape[i][1], self.w_size*2, self.w_size*2))

                self.s_length = (w_size*2)**2
                
            for i in range(self.num_img):
                print(dataset[i].shape)
                if final_dim == 1:
                    dataset_flat.extend(dataset[i].reshape(-1, self.s_length))
                else:
                    dataset_flat.extend(dataset[i].reshape(-1, self.w_size*2, self.w_size*2))
                    
            dataset_flat = np.asarray(dataset_flat)
            print(dataset_flat.shape)
            
        if log_scale:
            dataset_flat[np.where(dataset_flat==0.0)] = 1.0
            dataset_flat = np.log(dataset_flat)
            print(np.min(dataset_flat), np.max(dataset_flat))
            
        if max_normalize:
            if final_dim == 1:
                print(np.max(dataset_flat, axis=1).shape)
                dataset_flat = dataset_flat / np.max(dataset_flat, axis=1)[:, np.newaxis]
            else:
                dataset_flat = dataset_flat / np.max(dataset_flat, axis=(1,2))[:, np.newaxis, np.newaxis]
            print(np.min(dataset_flat), np.max(dataset_flat))
            
        if rescale_0to1:
            for i in range(len(dataset_flat)):
                dataset_flat[i] = zero_one_rescale(dataset_flat[i])
                
        dataset_flat = dataset_flat.clip(min=min_val)
        print(np.min(dataset_flat), np.max(dataset_flat))
        self.total_num = len(dataset_flat)
        self.dataset_flat = dataset_flat
        self.ri = np.random.choice(self.total_num, self.total_num, replace=False)

        self.dataset_input = dataset_flat[self.ri]
        self.dataset_input = self.dataset_input.astype(np.float32)


class VAE2DCNN_encoder(nn.Module):
    def __init__(self, final_length, channels, kernels, strides, paddings, pooling, z_dim):
        super(VAE2DCNN_encoder, self).__init__()
        
        self.z_dim = z_dim
        self.final_length = final_length
        self.channels = channels
        
        enc_net = []
        enc_net.append(nn.Conv2d(1, channels[0], kernels[0], stride=strides[0], 
                                 padding=paddings[0], bias=True))
        enc_net.append(nn.BatchNorm2d(channels[0]))
        enc_net.append(nn.Tanh())
        if pooling[0] != 1:
            enc_net.append(nn.AvgPool2d(pooling[0]))
        for i in range(1, len(channels)):
            enc_net.append(nn.Conv2d(channels[i-1], channels[i], kernels[i], stride=strides[i],
                                     padding=paddings[i], bias=True))
            enc_net.append(nn.BatchNorm2d(channels[i]))
            enc_net.append(nn.Tanh())
            if pooling[i] != 1:
                enc_net.append(nn.AvgPool2d(pooling[i]))
            
        enc_net.append(nn.Flatten())
        enc_net.append(nn.Linear(self.final_length**2*channels[-1], 2*self.z_dim))
        
        self.encoder = nn.Sequential(*enc_net)
        
        
    def encode(self, x):
        
        latent = self.encoder(x)
        mu = latent[:, :self.z_dim]
        logvar = latent[:, self.z_dim:]
        
        return mu, logvar
        
                
    def reparametrization(self, mu, logvar):
        
        return mu+torch.exp(0.5*logvar)*torch.randn_like(logvar)
        
        
    def forward(self, x):
        x = x.unsqueeze(1)
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        
        return mu, logvar, z
    
class VAE2DCNN_decoder(nn.Module):
    def __init__(self, z_dim, final_length, channels, dec_kernels, dec_strides, dec_paddings, dec_outpads, f_kernel):
        super(VAE2DCNN_decoder, self).__init__()
        
        self.z_dim = z_dim
        self.final_length = final_length
        self.channels = channels
        
        self.init_decoder = nn.Linear(self.z_dim, self.final_length**2*channels[-1])
        
        dec_net = []
        for i in range(len(channels)-1, 0, -1):
            dec_net.append(nn.ConvTranspose2d(channels[i], channels[i-1], dec_kernels[i], dec_strides[i],
                                              dec_paddings[i], output_padding=dec_outpads[i], bias=True))
            dec_net.append(nn.BatchNorm2d(channels[i-1]))
            dec_net.append(nn.Tanh())
            
        dec_net.append(nn.ConvTranspose2d(channels[0], 1, dec_kernels[0], dec_strides[0], 
                                          dec_paddings[0], output_padding=dec_outpads[0], bias=True))
        dec_net.append(nn.BatchNorm2d(1))
        dec_net.append(nn.Tanh())
        
        dec_net.append(nn.Conv2d(1, 1, f_kernel, bias=True))
        dec_net.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*dec_net)
    
    def forward(self, z):
        init_decoded = self.init_decoder(z)
        init_decoded = init_decoded.view(-1, self.channels[-1], self.final_length, self.final_length)
        
        return self.decoder(init_decoded)
    

class VAEFCNN_encoder(nn.Module):
    def __init__(self, in_dim, z_dim, h_dim):
        super(VAEFCNN_encoder, self).__init__()
        
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        
        enc_net = []
        enc_net.append(nn.Linear(self.in_dim, self.h_dim[0]))
        enc_net.append(nn.Tanh())
        for i in range(1, len(h_dim)):
            enc_net.append(nn.Linear(self.h_dim[i-1], self.h_dim[i]))
            enc_net.append(nn.Tanh())
        enc_net.append(nn.Linear(self.h_dim[-1], 2*z_dim))
        
        self.encoder = nn.Sequential(*enc_net)
        
        
    def encode(self, x):
        
        latent = self.encoder(x)
        mu = latent[:, :self.z_dim]
        logvar = latent[:, self.z_dim:]
        
        return mu, logvar
    
    def reparametrization(self, mu, logvar):
        
        return mu+torch.exp(0.5*logvar)*torch.randn_like(logvar)        
        
    def forward(self, x):
        
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        
        return mu, logvar, z  
    
    
    
class VAEFCNN_decoder(nn.Module):
    def __init__(self, z_dim, h_dim, in_dim):
        super(VAEFCNN_decoder, self).__init__()
        
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.in_dim = in_dim
        
        dec_net = []
        dec_net.append(nn.Linear(z_dim, self.h_dim[-1]))
        dec_net.append(nn.BatchNorm1d(self.h_dim[-1]))
        dec_net.append(nn.Tanh())
        for i in range(len(h_dim), 1, -1):
            dec_net.append(nn.Linear(self.h_dim[i-1], self.h_dim[i-2]))
            dec_net.append(nn.BatchNorm1d(self.h_dim[i-2]))
            dec_net.append(nn.Tanh())
        dec_net.append(nn.Linear(self.h_dim[0], self.in_dim))
        dec_net.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*dec_net)
        
    def forward(self, z):       
        
        return self.decoder(z)
    
    
class ivVAEFCNN_encoder(nn.Module):
    def __init__(self, in_dim, h_dim, z_dim, rot_check=True, trans_check=True, trans_std=0.1):
        super(ivVAEFCNN_encoder, self).__init__()
        
        self.img_z_dim = z_dim        
        
        self.rot_check=rot_check
        self.trans_check=trans_check
        self.trans_std=trans_std
        
        self.z_dim = self.img_z_dim
        if self.rot_check:
            self.z_dim += 1
        if self.trans_check:
            self.z_dim += 2
        if not self.rot_check and not self.trans_check:
            print("Warning! at least one invariant property must be chosen")
            return
        
        enc_net = []
        enc_net.append(nn.Linear(in_dim, h_dim[0]))
        enc_net.append(nn.Tanh())
        for i in range(1, len(h_dim)):
            enc_net.append(nn.Linear(h_dim[i-1], h_dim[i]))
            enc_net.append(nn.Tanh())
        enc_net.append(nn.Linear(h_dim[-1], 2*self.z_dim))
        
        self.encoder = nn.Sequential(*enc_net)
        
    def encode(self, x):
        
        latent = self.encoder(x)
        mu = latent[:, :self.z_dim]
        logvar = latent[:, self.z_dim:]
        
        return mu, logvar
        
    def rotation(self, coord, z):
        rot_matrix = torch.stack((torch.cos(z), torch.sin(z), -torch.sin(z), torch.cos(z)), dim=1)
        rot_matrix = rot_matrix.view(-1, 2, 2)
        
        return torch.bmm(coord, rot_matrix)
        
        
    def translation(self, coord, z):
        trans_z = z * self.trans_std
        trans_z = trans_z.unsqueeze(1)
        
        return coord + trans_z
        
                
    def reparametrization(self, mu, logvar):
        
        return mu+torch.exp(0.5*logvar)*torch.randn_like(logvar)
        
        
    def forward(self, x, coord):
        if coord.size(0) != x.size(0):
            coord = coord[:x.size(0)]
        
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        
        rot_mu=None
        rot_logvar=None
        rot_z=None
        
        trans_mu=None
        trans_logvar=None
        trans_z=None
        
        if self.rot_check:
            rot_mu = mu[:, 0]
            mu = mu[:, 1:]
            
            rot_logvar = logvar[:, 0]
            logvar = logvar[:, 1:]
            
            rot_z = z[:, 0]
            z = z[:, 1:]
            
            coord = self.rotation(coord, rot_z)
            
        if self.trans_check:

            trans_mu = mu[:, :2]
            mu = mu[:, 2:]
            
            trans_logvar = logvar[:, :2]
            logvar = logvar[:, 2:]

            trans_z = z[:, :2]
            z = z[:, 2:]
            
            coord = self.translation(coord, trans_z)
        
        return coord, mu, logvar, z, rot_mu, rot_logvar, rot_z, trans_mu, trans_logvar, trans_z
    
    
class ivVAE2DCNN_encoder(nn.Module):
    def __init__(self, final_length, channels, kernels, strides, paddings, pooling, z_dim, 
                 rot_check=True, trans_check=True, trans_std=0.1):
        super(ivVAE2DCNN_encoder, self).__init__()
        
        self.img_z_dim = z_dim        
        self.final_length = final_length
        self.channels = channels
        
        self.rot_check=rot_check
        self.trans_check=trans_check
        self.trans_std=trans_std
        
        self.z_dim = self.img_z_dim
        if self.rot_check:
            self.z_dim += 1
        if self.trans_check:
            self.z_dim += 2
        if not self.rot_check and not self.trans_check:
            print("Warning! at least one invariant property must be chosen")
            return
        
        enc_net = []
        enc_net.append(nn.Conv2d(1, channels[0], kernels[0], stride=strides[0], 
                                 padding=paddings[0], bias=True))
        enc_net.append(nn.BatchNorm2d(channels[0]))
        enc_net.append(nn.Tanh())
        if pooling[0] != 1:
            enc_net.append(nn.AvgPool2d(pooling[0]))
        for i in range(1, len(channels)):
            enc_net.append(nn.Conv2d(channels[i-1], channels[i], kernels[i], stride=strides[i],
                                     padding=paddings[i], bias=True))
            enc_net.append(nn.BatchNorm2d(channels[i]))
            enc_net.append(nn.Tanh())
            if pooling[i] != 1:
                enc_net.append(nn.AvgPool2d(pooling[i]))
            
        enc_net.append(nn.Flatten())
        enc_net.append(nn.Linear(final_length**2*channels[-1], 2*self.z_dim, bias=False))
        
        self.encoder = nn.Sequential(*enc_net)
        
    def encode(self, x):
        
        latent = self.encoder(x)
        mu = latent[:, :self.z_dim]
        logvar = latent[:, self.z_dim:]
        
        return mu, logvar
        
        
    def rotation(self, coord, z):
        rot_matrix = torch.stack((torch.cos(z), torch.sin(z), -torch.sin(z), torch.cos(z)), dim=1)
        rot_matrix = rot_matrix.view(-1, 2, 2)
        
        return torch.bmm(coord, rot_matrix)
        
        
    def translation(self, coord, z):
        trans_z = z * self.trans_std
        trans_z = trans_z.unsqueeze(1)
        
        return coord + trans_z
        
                
    def reparametrization(self, mu, logvar):
        
        return mu+torch.exp(0.5*logvar)*torch.randn_like(logvar)
        
        
    def forward(self, x, coord):
        if coord.size(0) != x.size(0):
            coord = coord[:x.size(0)]
        
        x = x.unsqueeze(1)
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        
        rot_mu=None
        rot_logvar=None
        rot_z=None
        
        trans_mu=None
        trans_logvar=None
        trans_z=None
        
        if self.rot_check:
            rot_mu = mu[:, 0]
            mu = mu[:, 1:]
            
            rot_logvar = logvar[:, 0]
            logvar = logvar[:, 1:]
            
            rot_z = z[:, 0]
            z = z[:, 1:]
            
            coord = self.rotation(coord, rot_z)
            
        if self.trans_check:

            trans_mu = mu[:, :2]
            mu = mu[:, 2:]
            
            trans_logvar = logvar[:, :2]
            logvar = logvar[:, 2:]

            trans_z = z[:, :2]
            z = z[:, 2:]
            
            coord = self.translation(coord, trans_z)
        
        return coord, mu, logvar, z, rot_mu, rot_logvar, rot_z, trans_mu, trans_logvar, trans_z
    

class ivVAEFCNN_decoder(nn.Module):   
    def __init__(self, n_coord, n_dim, z_dim, hid_dim, num_hid, w_size, bi_lin=False):
        super(ivVAEFCNN_decoder, self).__init__()
        
        self.n_coord = n_coord
        self.n_dim = n_dim
        self.z_dim = z_dim
        self.bi_lin = bi_lin
        self.w_size = w_size
        
        self.linear_coord = nn.Linear(n_dim, hid_dim, bias=False)
        self.linear_img = nn.Linear(z_dim, hid_dim, bias=False)
        if bi_lin:
            self.bi_linear = nn.Bilinear(n_dim, z_dim, hid_dim, bias=False)
    
        
        dec_net = []
        for i in range(num_hid):
            dec_net.append(nn.Linear(hid_dim, hid_dim, bias=True))
            dec_net.append(nn.BatchNorm1d(hid_dim))
            dec_net.append(nn.Tanh())
            
        dec_net.append(nn.Linear(hid_dim, 1, bias=True))
        dec_net.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*dec_net)
        
        
    def forward(self, coord, z):
        img_tmp = self.linear_img(z)
        z = z.unsqueeze(1)
        z = z.expand(z.size(0), self.n_coord, self.z_dim).contiguous()
        if self.bi_lin:
            bi_tmp = self.bi_linear(coord, z)
        coord = coord.view(coord.size(0)*coord.size(1), -1).contiguous()
        coord_tmp = self.linear_coord(coord)
        
        #print(img_tmp.shape, bi_tmp.shape, coord_tmp.shape)
        img_tmp = img_tmp.unsqueeze(1)
        coord_tmp = coord_tmp.view(z.size(0), self.n_coord, -1)
        #print(img_tmp.shape, bi_tmp.shape, coord_tmp.shape)
        
        
        if self.bi_lin:
            init_dec = coord_tmp + img_tmp + bi_tmp
        else:
            init_dec = coord_tmp + img_tmp
        #print(init_dec.shape)
        
        init_dec = init_dec.view(z.size(0)*self.n_coord, -1)
        output = self.decoder(init_dec.contiguous())
        
        return output.view(z.size(0), self.n_coord)
    

class linFE_decoder(nn.Module):
    def __init__(self, z_dim, in_dim):
        super(linFE_decoder, self).__init__()
        
        self.z_dim = z_dim
        self.in_dim = in_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.in_dim, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, z):       
        
        return self.decoder(z)

    

def reconstruction_loss(out, tar, mean=False, loss_fn="BCE"):

    if mean:
        reduce = "mean"
    else:
        reduce = "sum"

    if loss_fn == "BCE":
        return F.binary_cross_entropy(out, tar, reduction=reduce)
    
    elif loss_fn == "MSE":
        return F.mse_loss(out, tar, reduction=reduce)
    

def VAE_KLD(mu, logvar, mean=False, mode="normal", beta=4.0, gamma=1000.0, C_max=25, C_stop_iter=1E5, glob_iter=0):
    if mean:
        kld = -0.5*torch.mean(1+logvar-mu**2-logvar.exp())

    else:
        kld = -0.5*torch.sum(1+logvar-mu**2-logvar.exp())

    if mode == "normal":
        return kld
    elif mode == "beta":
        return beta * kld
    elif mode == "gamma":
        C = torch.clamp(C_max/C_stop_iter*glob_iter, 0, C_max.data[0])
        return gamma*(kld-C).abs()
    else:
        print("invalid mode option!")
        return

def ivVAE_KLD(mu, logvar, rot_mu, rot_logvar, ang_std, 
              mean=False, mode="normal", beta=4.0, 
              gamma=1000.0, C_max=25, C_stop_iter=1E5, glob_iter=0):
    
    if mean:
        kld = -0.5*torch.mean(1+logvar-mu**2-logvar.exp())
        rot_kld = torch.mean(-rot_logvar + np.log(ang_std) + (torch.exp(rot_logvar)**2 + 
                                                              rot_mu**2)/2/ang_std**2 - 0.5)

    else:
        kld = -0.5*torch.sum(1+logvar-mu**2-logvar.exp())
        rot_kld = torch.sum(-rot_logvar + np.log(ang_std) + (torch.exp(rot_logvar)**2 + 
                                                              rot_mu**2)/2/ang_std**2 - 0.5)

    if mode == "normal":
        return kld + rot_kld
    elif mode == "beta":
        return beta * (kld+rot_kld)  
    elif mode == "gamma":
        C = torch.clamp(C_max/C_stop_iter*glob_iter, 0, C_max.data[0])
        return gamma*((kld-C).abs() + (rot_kld-C).abs())
    else:
        print("invalid mode option!")
        return


def data_load_3d(adr, crop=None, rescale=True, DM_file=True):
    """
    load a spectrum image
    """
    storage = []
    shape = []
    for i, ad in enumerate(adr):
        if DM_file:
            if crop:
                temp = hys.load(ad)
                print(temp.axes_manager[2])
                temp = temp.isig[crop[0]:crop[1]]
                temp = temp.data
                if rescale:
                    temp = temp/np.max(temp)
                temp = temp.clip(min=0.0)
                print(temp.shape)
                
            else:
                temp = hys.load(ad).data
                if rescale:
                    temp = temp/np.max(temp)
                temp = temp.clip(min=0.0)
                print(temp.shape)
        
        else:
            if crop:
                temp = tifffile.imread(ad)
                temp = temp[:, :, crop[0]:crop[1]]
                if rescale:
                    temp = temp/np.max(temp)
                temp = temp.clip(min=0.0)
                print(temp.shape)
                
            else:
                temp = tifffile.imread(ad)
                if rescale:
                    temp = temp/np.max(temp)
                temp = temp.clip(min=0.0)
                print(temp.shape)                
                
        shape.append(temp.shape)
        storage.append(temp)       
    
    shape = np.asarray(shape)
    return storage, shape


def data_load_4d(adr, rescale=False):
    storage = []
    shape = []   
    for i, ad in enumerate(adr):
        tmp = tifffile.imread(ad)
        if rescale:
            tmp = tmp / np.max(tmp)
        tmp = tmp.clip(min=0.0)
        print(tmp.shape)
        if len(tmp.shape) == 3:
            try:
                tmp = tmp.reshape(int(tmp.shape[0]**(1/2)), int(tmp.shape[0]**(1/2)), tmp.shape[1], tmp.shape[2])
                print("The scanning shape is automatically corrected")
            except:
                print("The input data is not 4-dimensional")
                print("Please confirm that all options are correct")
            
        shape.append(list(tmp.shape[:2]))
        storage.append(tmp)
    
    shape = np.asarray(shape)
    return storage, shape

def zero_one_rescale(spectrum):
    """
    normalize one spectrum from 0.0 to 1.0
    """
    spectrum = spectrum.clip(min=0.0)
    min_val = np.min(spectrum)
    
    rescaled = spectrum - min_val
    
    if np.max(rescaled) != 0:
        rescaled = rescaled / np.max(rescaled)
    
    return rescaled

def binning_SI(si, bin_y, bin_x, str_y, str_x, offset, depth, rescale=True):
    """
    re-bin a spectrum image
    """
    si = np.asarray(si)
    rows = range(0, si.shape[0]-bin_y+1, str_y)
    cols = range(0, si.shape[1]-bin_x+1, str_x)
    new_shape = (len(rows), len(cols))
    
    binned = []
    for i in rows:
        for j in cols:
            temp_sum = np.mean(si[i:i+bin_y, j:j+bin_x, offset:(offset+depth)], axis=(0, 1))
            if rescale:
                binned.append(zero_one_rescale(temp_sum))
            else:
                binned.append(temp_sum)
            
    binned = np.asarray(binned).reshape(new_shape[0], new_shape[1], depth)
    
    return binned


def radial_indices(shape, radial_range, center=None):
    y, x = np.indices(shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    
    r = np.hypot(y - center[0], x - center[1])
    ri = np.ones(r.shape)
    
    if len(np.unique(radial_range)) > 1:
        ri[np.where(r <= radial_range[0])] = 0
        ri[np.where(r > radial_range[1])] = 0
        
    else:
        r = np.round(r)
        ri[np.where(r != round(radial_range[0]))] = 0
    
    return ri

def flattening(fdata, flat_option="box", crop_dist=None, c_pos=None):
    
    fdata_shape = fdata.shape
    if flat_option == "box":
        if crop_dist:     
            box_size = np.array([crop_dist, crop_dist])
        
            h_si = np.floor(c_pos[0]-box_size[0]).astype(int)
            h_fi = np.ceil(c_pos[0]+box_size[0]).astype(int)
            w_si = np.floor(c_pos[1]-box_size[1]).astype(int)
            w_fi = np.ceil(c_pos[1]+box_size[1]).astype(int)

            tmp = fdata[:, :, h_si:h_fi, w_si:w_fi]
            
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(np.log(np.mean(tmp, axis=(0, 1))), cmap="viridis")
            ax.axis("off")
            plt.show()
            
            tmp = tmp.reshape(fdata_shape[0], fdata_shape[1], -1)
            return tmp

        else:
            tmp = fdata.reshape(fdata_shape[0], fdata_shape[1], -1)
            return tmp

        
    elif flat_option == "radial":
        if len(crop_dist) != 3:
            print("Warning! 'crop_dist' must be a list containing 3 elements")
            
        tmp = circle_flatten(fdata, crop_dist, c_pos)
        return tmp
        
    else:
        print("Warning! Wrong option ('flat_option')")
        return
    
def circle_flatten(f_stack, radial_range, c_pos):
    k_indx = []
    k_indy = []
    
    for r in range(radial_range[0], radial_range[1], radial_range[2]):
        tmp_k, tmp_a = indices_at_r(f_stack.shape[2:], r, c_pos)
        k_indx.extend(tmp_k[0].tolist())
        k_indy.extend(tmp_k[1].tolist())
    
    k_indx = np.asarray(k_indx)
    k_indy = np.asarray(k_indy)
    flat_data = f_stack[:, :, k_indy, k_indx]
    
    return flat_data

def indices_at_r(shape, radius, center=None):
    y, x = np.indices(shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    r = np.hypot(y - center[0], x - center[1])
    r = np.around(r)
    
    ri = np.where(r == radius)
    
    angle_arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            angle_arr[i, j] = np.angle(complex(x[i, j]-center[1], y[i, j]-center[0]), deg=True)
            
    angle_arr = angle_arr + 180
    angle_arr = np.around(angle_arr)
    
    ai = np.argsort(angle_arr[ri])
    r_sort = (ri[1][ai], ri[0][ai])
    a_sort = np.sort(angle_arr[ri])
        
    return r_sort, a_sort

def reshape_coeff(coeffs, new_shape):
    """
    reshape a coefficient matrix to restore the original scanning shapes.
    """
    coeff_reshape = []
    for i in range(len(new_shape)):
        temp = coeffs[:int(new_shape[i, 0]*new_shape[i, 1]), :]
        coeffs = np.delete(coeffs, range(int(new_shape[i, 0]*new_shape[i, 1])), axis=0)
        temp = np.reshape(temp, (new_shape[i, 0], new_shape[i, 1], -1))
        #print(temp.shape)
        coeff_reshape.append(temp)
        
    return coeff_reshape