import numpy as np
import torch
import torch.optim as optim
import os
import wandb
import datetime
import pickle
from tdm_nn import Full_network, AutoEncoder, ForwardNet, BackwardNet

def train_network(training_data, val_data, params):
    # SET UP NETWORK
    
    input_dim = params['input_dim']
    state_dim = params['state_dim']# 
    act_dim = params['act_dim'] # 
    latent_out = params['latent_out']# 
    z_stat_dim = params['latent_s_dim']
    z_act_dim = latent_out - z_stat_dim
    # assert(latent_out == act_dim + z_stat_dim)
    learning_rate = params['learning_rate']
    l1_lambda = params['l1_rate']
    total_epochs = params['max_epochs']
    
    ae_network =  AutoEncoder(input_dim, latent_out, act_dim, state_dim, z_stat_dim, z_act_dim, params['device'], seed=params['seed']).to(params['device'])
    fwd_network = ForwardNet(z_stat_dim, z_act_dim, device=params['device'], seed=params['seed']).to(params['device'])
    bwd_network = BackwardNet(z_stat_dim, z_act_dim, device=params['device'], seed=params['seed']).to(params['device'])
    dynamic_network = Full_network(ae_network, fwd_network, bwd_network, params)
    
    optimizer = optim.Adam([{"params":dynamic_network.ae_network.parameters()},
                            {"params":dynamic_network.fwd_network.parameters()}, 
                            {"params":dynamic_network.bwd_network.parameters()}], 
                            lr=learning_rate)
    validation_dict = create_feed_dictionary(val_data, params, idxs=None)

    print('--------------------START TRAINING-------------------------------')
    
    for epoch in range(total_epochs):
        rand_idx = torch.LongTensor(np.random.permutation(params['train_size'] -1))
        for j in range(params['batch_num']):
            optimizer.zero_grad()
            batch_idxs = rand_idx[j*params['batch_size']:(j+1)*params['batch_size']]
            train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
            results_dic = dynamic_network.dynamic_pred(train_dict)
            train_total_loss, losses  = define_loss(train_dict, results_dic, params, epoch, validation=False)
            l1_norm_fwd = sum(p.abs().sum() for p in dynamic_network.fwd_network.parameters())
            l1_norm_bwd = sum(p.abs().sum() for p in dynamic_network.bwd_network.parameters())
            reg_loss = l1_lambda * (l1_norm_fwd + l1_norm_bwd)
            train_total_loss += reg_loss
            train_total_loss.backward()
            optimizer.step()

            
        wandb.log({
            'epoch': epoch,
            "training_loss": train_total_loss,
            "dnyamic_dz_s":losses['dnyamic_dz_s'],
            "state_decode":losses['state_decode'],
            "dnyamic_dz_s_decoded":losses['dnyamic_dz_s_decoded'],
            "act_decode":losses['act_decode'],
            "dnyamic_dz_sp":losses['dnyamic_dz_sp'],
            "dnyamic_dz_sp_decoded":losses['dnyamic_dz_sp_decoded'],
            'model_consist':losses['model_consist'],
            "dyna_consist": losses['dyna_consist']
            }
            )

        if epoch % 2 == 0:
            valid_result_dic = dynamic_network.dynamic_pred(validation_dict)
            valid_total_loss, valid_losses  = define_loss(validation_dict, valid_result_dic, params, epoch, validation=True)
            wandb.log({
                'epoch': epoch,
                "valid_loss_dz_s":valid_losses['dnyamic_dz_s'],
                "valid_loss_dz_s_decoded": valid_losses['dnyamic_dz_s_decoded'],
                'valid_loss_state_decoder': valid_losses['state_decode'],
                'valid_loss_action_decoder': valid_losses['act_decode'],
                "valid_loss_dz_sp_decoded": valid_losses['dnyamic_dz_sp_decoded'],
                "valid_loss_dz_sp":valid_losses['dnyamic_dz_sp'],
                'valid_loss_model_consist':losses['model_consist'],
                "dyna_consist":losses['dyna_consist']
                }
                )
    OUT_DIR = params['data_path']
    if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)
    pickle.dump(ae_network, open(OUT_DIR + 'AE_params.pkl', 'wb'))
    pickle.dump(fwd_network, open(OUT_DIR + 'Dyna_fwd_params.pkl', 'wb'))
    pickle.dump(bwd_network, open(OUT_DIR + 'Dyna_bwd_params.pkl', 'wb'))
    
def create_feed_dictionary(data, params, idxs=None):

    if idxs is None:
        idxs = np.arange(data['s'].shape[0])
    feed_dict =  {}
    feed_dict['s'] = torch.from_numpy(data['s'][idxs]).float().to(params['device'])
    feed_dict['act'] = torch.from_numpy(data['act'][idxs]).float().to(params['device'])
    feed_dict['sp'] = torch.from_numpy(data['sp'][idxs]).float().to(params['device'])
    feed_dict['next_act'] = torch.from_numpy(data['next_act'][idxs]).float().to(params['device'])
    feed_dict['da'] = torch.from_numpy(data['da'][idxs]).float().to(params['device'])
    feed_dict['ds'] = torch.from_numpy(data['ds'][idxs]).float().to(params['device'])
    feed_dict['dsp'] = torch.from_numpy(data['dsp'][idxs]).float().to(params['device'])
    feed_dict['x'] = torch.cat((feed_dict['s'],feed_dict['act']), -1)
    feed_dict['dx'] = torch.cat((feed_dict['ds'],feed_dict['da']), -1)
    feed_dict['xp'] = torch.cat((feed_dict['sp'],feed_dict['act']), -1)
    feed_dict['dxp'] = torch.cat((feed_dict['dsp'],feed_dict['da']), -1)
    
    return feed_dict

def define_loss(data_dic, results_dic, params, epoch, validation=False):
        """
        Create the loss functions.
        """
        losses =  {}
        if validation:
            with torch.no_grad():
                losses['state_decode'] = torch.mean(torch.sum((data_dic['s'] - results_dic['state_decode'])**2, -1))
                losses['act_decode'] = torch.mean(torch.sum((data_dic['act'] - results_dic['act_decode'])**2, -1))
                losses['total_decode_loss'] = losses['act_decode'] + losses['state_decode']
                losses['dnyamic_dz_s'] = torch.mean(torch.sum((results_dic['dz_s'] - results_dic['fwd_dyna_predict'])**2, -1))
                losses['dnyamic_dz_s_decoded'] = torch.mean(torch.sum((data_dic['ds'] - results_dic['dz_s_decode'])**2, -1))
                losses['dnyamic_dz_sp'] = torch.mean(torch.sum((results_dic['dz_sp'] - results_dic['bwd_dyna_predict'])**2, -1))
                losses['dnyamic_dz_sp_decoded'] = torch.mean(torch.sum((data_dic['dsp'] - results_dic['dz_sp_decode'])**2, -1))
                losses['dyna_consist'] = torch.mean(torch.sum((results_dic['dyna_consist'])**2, -1))
                losses['model_consist'] = torch.mean(torch.sum((results_dic['consist'])**2, -1))
                loss =  losses['total_decode_loss'] + losses['dnyamic_dz_s'] +  losses['dnyamic_dz_s_decoded']
                return loss, losses     
        else:
            with torch.set_grad_enabled(True):
                losses['state_decode'] = torch.mean(torch.sum((data_dic['s'] - results_dic['state_decode'])**2, -1))
                losses['act_decode'] = torch.mean(torch.sum((data_dic['act'] - results_dic['act_decode'])**2, -1))
                losses['dnyamic_dz_s'] = torch.mean(torch.sum((results_dic['dz_s'] - results_dic['fwd_dyna_predict'])**2, -1))
                losses['dnyamic_dz_s_decoded'] = torch.mean(torch.sum((data_dic['ds'] - results_dic['dz_s_decode'])**2, -1))
                losses['dnyamic_dz_sp'] = torch.mean(torch.sum((results_dic['dz_sp'] - results_dic['bwd_dyna_predict'])**2, -1))
                losses['dnyamic_dz_sp_decoded'] = torch.mean(torch.sum((data_dic['dsp'] - results_dic['dz_sp_decode'])**2, -1))
                losses['dyna_consist'] = torch.mean(torch.sum((results_dic['dyna_consist'])**2, -1))
                losses['model_consist'] = torch.mean(torch.sum((results_dic['consist'])**2, -1))
                if epoch >= params['pre_train_epoch']:
                    loss = params['loss_weight_state_decoder'] * (losses['state_decode']) \
                    + params['loss_weight_act_decoder'] * (losses['act_decode'])\
                    + params['loss_weight_dynamic_z_s'] * (losses['dnyamic_dz_s'] + losses['dnyamic_dz_sp'])  \
                    + params['losss_model_consist'] * (losses['model_consist'] + losses['dyna_consist']) \
                    + params['loss_weight_dynamic_z_s_decode'] * (losses['dnyamic_dz_s_decoded'] + losses['dnyamic_dz_sp_decoded']) 
                else:
                    loss = params['loss_weight_state_decoder'] * (losses['state_decode']) \
                          + params['loss_weight_act_decoder'] * (losses['act_decode'])\
                          + params['loss_weight_dynamic_z_s_decode'] * (losses['dnyamic_dz_s_decoded'] + losses['dnyamic_dz_sp_decoded']) \
                          + params['losss_model_consist'] * (losses['model_consist'] + losses['dyna_consist']) \
        
        return loss, losses