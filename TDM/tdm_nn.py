import numpy as np
import torch
import torch.nn as nn
import functorch

# import tensorflow as tf
class Full_network:
    def __init__(self, 
                ae_network,
                fwd_net,
                bwd_net,
                params,
                 ):
        self.ae_network = ae_network
        self.fwd_network = fwd_net
        self.bwd_network = bwd_net
        self.device = params['device']
        self.state_dim = params['state_dim']
        self.act_dim = params['act_dim']
        self.latent_dim = params['latent_out']
        self.latent_s_dim = params['latent_s_dim']
        self.activation = params['activation']
        
    def encoder_res(self,dic):
        z, state_decode, act_decode, self.encoder_weights, self.encoder_biases, self.decoder_weights, self.decoder_biases = self.process_autoencoder(self.ae_network, dic)
        return z, state_decode, act_decode
    
    def dynamic_pred(self,dic):
        pred_results = {}
        z, zp, state_decode, act_decode = self.process_autoencoder(self.ae_network, dic)
        z_s = z[:,:self.latent_s_dim]
        z_a = z[:,self.latent_s_dim:]
        z_sp = zp[:,:self.latent_s_dim]
        dz_s = self.dz_s(dic['x'], dic['ds'], self.ae_network.encoder)
        dz_sp = self.dz_s(dic['xp'], dic['dsp'], self.ae_network.encoder)
        
        fwd_dyna_predict = self.fwd_network(z)
        bwd_dyna_predict = self.bwd_network(zp)
        pred_z_sp = z_s + fwd_dyna_predict
        pre_zp = torch.cat((pred_z_sp, z_a), -1)
        pred_delta_zsp = self.bwd_network(pre_zp)
        pred_zs = z_sp + pred_delta_zsp
        
        dz_s_decode = self.decode_dz_s(z_s, fwd_dyna_predict, self.ae_network.s_decoder)
        dz_sp_decode = self.decode_dz_s(z_sp, bwd_dyna_predict, self.ae_network.s_decoder)

        pred_results['dz_s'] = dz_s
        pred_results['dz_sp'] = dz_sp
        pred_results['state_decode'] = state_decode
        pred_results['act_decode'] = act_decode
        pred_results['fwd_dyna_predict'] = fwd_dyna_predict
        pred_results['bwd_dyna_predict'] = bwd_dyna_predict
        pred_results['dz_s_decode'] = dz_s_decode
        pred_results['dz_sp_decode'] = dz_sp_decode
        pred_results['consist'] = pred_zs - z_s
        pred_results['dyna_consist'] = fwd_dyna_predict + bwd_dyna_predict
        return pred_results


    def process_autoencoder(self, network, dic):
        """
        Construct a nonlinear autoencoder.

        Arguments:

        Returns:
            z -
            x_decode -
            encoder_weights - List of tensorflow arrays containing the encoder weights
            encoder_biases - List of tensorflow arrays containing the encoder biases
            decoder_weights - List of tensorflow arrays containing the decoder weights
            decoder_biases - List of tensorflow arrays containing the decoder biases
        """
        x = dic['x']
        xp = dic['xp']
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float().to(self.device)
        if type(xp) == np.ndarray:
            xp = torch.from_numpy(xp).float().to(self.device)
        z, s_decoded, a_decoded = network(x)
        zp, _, _ = network(xp) 
        return z, zp, s_decoded, a_decoded
    
    def dz_s(self, x, ds, func):
        jeco_matrix = functorch.vmap(functorch.jacrev(func))(x)
        partial_s = jeco_matrix[:,:,:self.state_dim] # B x Z_DIM x S_DIM
        ds = torch.unsqueeze(ds, -1) # B x S_DIM x 1
        dz_s = torch.squeeze(torch.bmm(partial_s, ds))[:,:self.latent_s_dim] # B x Z_DIM
        return dz_s

    def decode_dz_s(self, z_s, dz_s, func):
        jeco_matrix = functorch.vmap(functorch.jacrev(func))(z_s)
        partial_s = jeco_matrix[:,:,:self.latent_s_dim] # B x S_DIM x Z_DIM
        dz_s = torch.unsqueeze(dz_s, -1) # B x Z_DIM x 1
        ds = torch.squeeze(torch.bmm(partial_s, dz_s)) # B x S_DIM
        return ds
    
class ForwardNet(nn.Module):
    def __init__(self, 
                 state_dim, 
                 act_dim,
                 device=None,
                 seed=123,
                 ):
        super(ForwardNet, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.state_dim, self.act_dim, self.device = state_dim, act_dim, device
        self.input_dim = state_dim + act_dim
        self.fc_layers = nn.Sequential(
            torch.nn.Linear(self.input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.state_dim)
        )
        
    def forward(self, inputs):
        if type(inputs) == np.ndarray:
            inputs = torch.from_numpy(inputs).float().to(self.device)
        else:
            inputs = inputs.float().to(self.device)
        d_s = self.fc_layers(inputs)
        return d_s
    
class BackwardNet(nn.Module):
    def __init__(self, 
                 state_dim, 
                 act_dim,
                 device=None,
                 seed=123,
                 ):
        super(BackwardNet, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.state_dim, self.act_dim, self.device = state_dim, act_dim, device
        self.input_dim = state_dim + act_dim
        
        self.fc_layers = nn.Sequential(
            torch.nn.Linear(self.input_dim, 512),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.state_dim)
        )
        
    def forward(self, inputs):
        if type(inputs) == np.ndarray:
            inputs = torch.from_numpy(inputs).float().to(self.device)
        else:
            inputs = inputs.float().to(self.device)
        d_s = self.fc_layers(inputs)
        return d_s

class AutoEncoder(torch.nn.Module):
    def __init__(self, 
                input_dim,
                out_dim,
                act_dim,
                state_dim,
                z_stat_dim,
                z_act_dim,
                device,
                seed=111,
                 ):
        super(AutoEncoder, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.z_stat_dim = z_stat_dim
        self.device = device
        self.encoder = nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_dim)
        )
 
        self.s_decoder = nn.Sequential(
            torch.nn.Linear(z_stat_dim, 128),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, state_dim)
        )
        
        self.a_decoder = nn.Sequential(
            torch.nn.Linear(z_act_dim, 128),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, act_dim)
            
        )
    def forward(self, x):
        z = self.encoder(x)
        z_state = z[:,:self.z_stat_dim]
        z_act = z[:,self.z_stat_dim:]
        s_decoded = self.s_decoder(z_state)
        a_decoded = self.a_decoder(z_act)
        return z, s_decoded, a_decoded