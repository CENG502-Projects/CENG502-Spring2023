import json
import os
import sys
import time
from os import path as osp
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


from model_temporal import LSTMSeqNetwork, BilinearLSTMSeqNetwork, TCNSeqNetwork, CTINNetwork
from utils import load_config, MSEAverageMeter
from data_glob_speed import GlobSpeedSequence, SequenceToSequenceDataset
from transformations import ComposeTransform, RandomHoriRotateSeq
from metric import compute_absolute_trajectory_error, compute_relative_trajectory_error, compute_ate_rte
from scipy.interpolate import interp1d

'''
Temporal models with loss functions in global coordinate frame
Configurations
    - Model types 
        TCN - type=tcn
        LSTM_simple - type=lstm, lstm_bilinear
        CTIN - type=ctin Contextual Transformer for Inertial Navigation        
'''

torch.multiprocessing.set_sharing_strategy('file_system')
_nano_to_sec = 1e09
_input_channel, _output_channel = 6, 2
device = 'cpu'


class GlobalPosLoss(torch.nn.Module):
    def __init__(self, mode='full', history=None):
        """
        Calculate position loss in global coordinate frame
        Target :- Global Velocity
        Prediction :- Global Velocity
        """
        super(GlobalPosLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        assert mode in ['full', 'part']
        self.mode = mode
        if self.mode == 'part':
            assert history is not None
            self.history = history
        elif self.mode == 'full':
            self.history = 1

    def forward(self, pred, targ):
        gt_pos = torch.cumsum(targ[:, 1:, ], 1)
        pred_pos = torch.cumsum(pred[:, 1:, ], 1)
        if self.mode == 'part':
            gt_pos = gt_pos[:, self.history:, :] - gt_pos[:, :-self.history, :]
            pred_pos = pred_pos[:, self.history:, :] - pred_pos[:, :-self.history, :]
        loss = self.mse_loss(pred_pos, gt_pos)
        return torch.mean(loss)

class CTINMultiTaskLoss(torch.nn.Module):
    def __init__(self):
        """
        Calculate Multi Task Loss in global coordinate frame
        Targets : - Global Velocity, Global Position Difference
        Prediction : - Global Velocity
        """
        super(CTINMultiTaskLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, pred, targ, pred_cov):
        """
        pred : [batch_size, window_size, 2] contains predicted velocities
        pred_cov : [batch_size, window_size, 2] contains covariances for predicted velocities
        targ : [batch_size, window_size, 2] contains ground truth velocities
        """
        gt_pos = torch.cumsum(targ[:, 1:, ], 1)
        pred_pos = torch.cumsum(pred[:, 1:, ], 1)

        lv_p = self.mse_loss(pred_pos, gt_pos)
        diffs = torch.square(targ[:, 1:, ] - pred[:, 1:, ])
        lv_e = torch.cumsum(diffs, 1) # cumulative error of estimated velocities 
        lv_e = torch.mean(lv_e)
        lv = lv_p + lv_e # integral velocity loss

        # lc has dimension [batch_size, window_size-1]  
        
        lc = diffs[:, :, None, :] @ (1 / (torch.diag_embed(pred_cov[:, 1:, ]) + 1e-5))
        lc = lc @ diffs[:, :, :, None]
        lc = torch.mean(lc)
        lc += torch.mean(0.5 * torch.log(torch.abs(pred_cov[:, 1:, 0] * pred_cov[:, 1:, 1]) + 1e-5) )
        loss = lv + lc
        return loss

def write_config(args, **kwargs):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "pytorch_global_position"
            if kwargs:
                values['kwargs'] = kwargs
            json.dump(values, f, sort_keys=True)


def get_dataset(root_dir, data_list, args, **kwargs):
    input_format, output_format = [0, 3, 6], [0, _output_channel]
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, [], False

    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms.append(RandomHoriRotateSeq(input_format, output_format))
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True
    transforms = ComposeTransform(transforms)

    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    elif args.dataset == 'ridi':
        from data_ridi import RIDIGlobSpeedSequence
        seq_type = RIDIGlobSpeedSequence
    dataset = SequenceToSequenceDataset(seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
                                        random_shift=random_shift, transform=transforms, shuffle=shuffle,
                                        grv_only=grv_only, **kwargs)

    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)


def get_model(args, **kwargs):
    config = {}
    if kwargs.get('dropout'):
        config['dropout'] = kwargs.get('dropout')

    if args.type == 'tcn':
        network = TCNSeqNetwork(_input_channel, _output_channel, args.kernel_size,
                                layer_channels=args.channels, **config)
        print("TCN Network. Receptive field: {} ".format(network.get_receptive_field()))
    elif args.type == 'lstm_bi':
        print("Bilinear LSTM Network")
        network = BilinearLSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, device,
                                         lstm_layers=args.layers, lstm_size=args.layer_size, **config).to(device)
    elif args.type == 'lstm':
        print("Simple LSTM Network")
        network = LSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, device,
                                 lstm_layers=args.layers, lstm_size=args.layer_size, **config).to(device)
    else:
        print("CTIN")
        network = CTINNetwork(_input_channel, _output_channel, args.batch_size, 200, 1, 4, device).to(device)

    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. trainable parameters: {}'.format(pytorch_total_params))
    return network


def get_loss_function(history, args, **kwargs):
    if args.type == 'tcn':
        config = {'mode': 'part',
                  'history': history}
    else:
        config = {'mode': 'full'}

    #criterion = GlobalPosLoss(**config)
    criterion = CTINMultiTaskLoss()
    return criterion


def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep
    return result[:-1]


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset_from_list(args.data_dir, args.train_list, args, mode='train', **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True)
    end_t = time.time()

    print('Training set loaded. Time usage: {:.3f}s'.format(end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.data_dir, args.val_list, args, mode='val', **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print('Validation set loaded')

    global device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.out_dir:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
        copyfile(args.train_list, osp.join(args.out_dir, "train_list"))
        if args.val_list is not None:
            copyfile(args.val_list, osp.join(args.out_dir, "validation_list"))
        write_config(args, **kwargs)

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    network = get_model(args, **kwargs).to(device)
    history = network.get_receptive_field() if args.type == 'tcn' else args.window_size // 2
    criterion = get_loss_function(history, args, **kwargs)

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.75, verbose=True, eps=1e-12)
    quiet_mode = kwargs.get('quiet', False)
    use_scheduler = kwargs.get('use_scheduler', True)

    log_file = None
    if args.out_dir:
        log_file = osp.join(args.out_dir, 'logs', 'log.txt')
        if osp.exists(log_file):
            if args.continue_from is None:
                os.remove(log_file)
            else:
                copyfile(log_file, osp.join(args.out_dir, 'logs', 'log_old.txt'))

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        with open(osp.join(str(Path(args.continue_from).parents[1]), 'config.json'), 'r') as f:
            model_data = json.load(f)

        if device.type == 'cpu':
            checkpoints = torch.load(args.continue_from, map_location=lambda storage, location: storage)
        else:
            checkpoints = torch.load(args.continue_from, map_location={model_data['device']: args.device})

        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
    if kwargs.get('force_lr', False):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    step = 0
    best_val_loss = np.inf
    train_errs = np.zeros(args.epochs)

    print("Starting from epoch {}".format(start_epoch))
    print(args.epochs)
    try:
        for epoch in range(start_epoch, args.epochs):
            log_line = ''
            network.train()
            train_vel = MSEAverageMeter(3, [2], _output_channel)
            train_loss = 0
            start_t = time.time()
            
            print(device, "device")
            for bid, batch in enumerate(train_loader):
                feat, targ, _, _ = batch
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                predicted, pred_cov = network(feat)
                train_vel.add(predicted.cpu().detach().numpy(), targ.cpu().detach().numpy())
                loss = criterion(predicted, targ, pred_cov)
                train_loss += loss.cpu().detach().numpy()
                loss.backward()
                optimizer.step()
                step += 1
                if not (step % 1000):
                    print("step", step, "epoch", epoch)

            train_errs[epoch] = train_loss / train_mini_batches
            end_t = time.time()
            if not quiet_mode:
                print('-' * 25)
                print('Epoch {}, time usage: {:.3f}s, loss: {}, vel_loss {}/{:.6f}'.format(
                    epoch, end_t - start_t, train_errs[epoch], train_vel.get_channel_avg(), train_vel.get_total_avg()))
            log_line = format_string(log_line, epoch, optimizer.param_groups[0]['lr'], train_errs[epoch],
                                     *train_vel.get_channel_avg())

            saved_model = False
            if val_loader:
                network.eval()
                val_vel = MSEAverageMeter(3, [2], _output_channel)
                val_loss = 0
                for bid, batch in enumerate(val_loader):
                    feat, targ, _, _ = batch
                    feat, targ = feat.to(device), targ.to(device)
                    optimizer.zero_grad()
                    pred = network(feat)
                    val_vel.add(pred.cpu().detach().numpy(), targ.cpu().detach().numpy())
                    val_loss += criterion(pred, targ).cpu().detach().numpy()
                val_loss = val_loss / val_mini_batches
                log_line = format_string(log_line, val_loss, *val_vel.get_channel_avg())
                if not quiet_mode:
                    print('Validation loss: {} vel_loss: {}/{:.6f}'.format(val_loss, val_vel.get_channel_avg(),
                                                                           val_vel.get_total_avg()))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_model = True
                    if args.out_dir:
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'loss': train_errs[epoch],
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Best Validation Model saved to ', model_path)
                if use_scheduler:
                    scheduler.step(val_loss)

            if args.out_dir and not saved_model and (epoch + 1) % args.save_interval == 0:  # save even with validation
                model_path = osp.join(args.out_dir, 'checkpoints', 'icheckpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print('Model saved to ', model_path)

            if log_file:
                log_line += '\n'
                with open(log_file, 'a') as f:
                    f.write(log_line)
            if np.isnan(train_loss):
                print("Invalid value. Stopping training.")
                break
    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training completed')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)


def recon_traj_with_preds_global(dataset, preds, ind=None, seq_id=0, type='preds', **kwargs):
    ind = ind if ind is not None else np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int)

    if type == 'gt':
        pos = dataset.gt_pos[seq_id][:, :2]
    else:
        ts = dataset.ts[seq_id]
        # Compute the global velocity from local velocity.
        dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
        pos = preds * dts
        pos[0, :] = dataset.gt_pos[seq_id][0, :2]
        pos = np.cumsum(pos, axis=0)
    veloc = preds
    ori = dataset.orientations[seq_id]

    return pos, veloc, ori

def run_test(network, data_loader, device, eval_mode=True):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ, _, _) in enumerate(data_loader):
        p1, cov = network(feat.to(device))
        pred = p1.cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all

def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    breakpoint()
    pos = interp1d(ts_ext, pos, axis=0)(ts)
    return pos


def test(args, **kwargs):
    global device, _output_channel
    import matplotlib.pyplot as plt

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        root_dir = osp.split(args.test_path)[0]
        test_data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        root_dir = args.data_dir if args.data_dir else osp.split(args.test_list)[0]
        with open(args.test_list) as f:
            test_data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_path or test_list must be specified.')

    # Load the first sequence to update the input and output size
    _ = get_dataset(root_dir, [test_data_list[0]], args, mode='test')

    if args.out_dir and not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(osp.join(str(Path(args.model_path).parents[1]), 'config.json'), 'r') as f:
        model_data = json.load(f)

    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        checkpoint = torch.load(args.model_path, map_location={model_data['device']: args.device})

    network = get_model(args, **kwargs)
    network.load_state_dict(checkpoint.get('model_state_dict'))
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []
    log_file = None
    if args.test_list and args.out_dir:
        log_file = osp.join(args.out_dir, osp.split(args.test_list)[-1].split('.')[0] + '_log.txt')
        with open(log_file, 'w') as f:
            f.write(args.model_path + '\n')
            f.write('Seq traj_len velocity ate rte\n')

    pred_per_min = 200 * 60


    for idx, data in enumerate(test_data_list):

        seq_dataset = get_dataset(root_dir, test_data_list, args, mode='test', **kwargs)
        seq_loader = DataLoader(seq_dataset, batch_size=1, shuffle=False)
        ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=np.int)
        # don't take covariance here


        targets, preds = run_test(network, seq_loader, device, True)
        losses = np.mean((targets - preds) ** 2, axis=0)
        preds_seq.append(preds)
        targets_seq.append(targets)
        losses_seq.append(losses)

        pos_pred = recon_traj_with_preds(seq_dataset, preds[:, -1, :])[:, :2]
        pos_gt = seq_dataset.gt_pos[0][:, :2]

        traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
        ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
        ate_all.append(ate)
        rte_all.append(rte)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

        print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))

        # Plot figures
        kp = preds.shape[1]
        if kp == 2:
            targ_names = ['vx', 'vy']
        elif kp == 3:
            targ_names = ['vx', 'vy', 'vz']

        plt.figure('{}'.format(data), figsize=(16, 9))
        plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
        plt.plot(pos_pred[:, 0], pos_pred[:, 1])
        plt.plot(pos_gt[:, 0], pos_gt[:, 1])
        plt.title(data)
        plt.axis('equal')
        plt.legend(['Predicted', 'Ground truth'])
        plt.subplot2grid((kp, 2), (kp - 1, 0))
        plt.plot(pos_cum_error)
        plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
        for i in range(kp):
            plt.subplot2grid((kp, 2), (i, 1))
            plt.plot(ind, preds[:, i])
            plt.plot(ind, targets[:, i])
            plt.legend(['Predicted', 'Ground truth'])
            plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
        plt.tight_layout()

        if args.show_plot:
            plt.show()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                    np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
            plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

        plt.close('all')

    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)
    # Export a csv file
    if args.out_dir is not None and osp.isdir(args.out_dir):
        with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
            if losses_seq.shape[1] == 2:
                f.write('seq,vx,vy,avg,ate,rte\n')
            else:
                f.write('seq,vx,vy,vz,avg,ate,rte\n')
            for i in range(losses_seq.shape[0]):
                f.write('{},'.format(test_data_list[i]))
                for j in range(losses_seq.shape[1]):
                    f.write('{:.6f},'.format(losses_seq[i][j]))
                f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))

    print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}'.format(
        np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all)))
    return losses_avg      


if __name__ == '__main__':
    """
    Run file with individual arguments or/and config file. If argument appears in both config file and args, 
    args is given precedence.
    """
    default_config_file = osp.abspath(osp.join(osp.abspath(__file__), '../../config/temporal_model_defaults.json'))

    import argparse

    parser = argparse.ArgumentParser(description="Run seq2seq model in train/test mode [required]. Optional "
                                                 "configurations can be specified as --key [value..] pairs",
                                     add_help=True)
    parser.add_argument('--config', type=str, help='Configuration file [Default: {}]'.format(default_config_file),
                        default=default_config_file)
    # common
    parser.add_argument('--type', type=str, choices=['tcn', 'lstm', 'lstm_bi', 'ctin'], help='Model type')
    parser.add_argument('--data_dir', type=str, help='Directory for data files if different from list path.')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, help='Gaussian for smoothing features')
    parser.add_argument('--target_sigma', type=float, help='Gaussian for smoothing target')
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, help='Cuda device (e.g:- cuda:0) or cpu')
    parser.add_argument('--dataset', type=str, choices=['ronin', 'ridi'])
    # tcn
    tcn_cmd = parser.add_argument_group('tcn', 'configuration for TCN')
    tcn_cmd.add_argument('--kernel_size', type=int)
    tcn_cmd.add_argument('--channels', type=str, help='Channel sizes for TCN layers (comma separated)')
    # lstm
    lstm_cmd = parser.add_argument_group('lstm', 'configuration for LSTM')
    lstm_cmd.add_argument('--layers', type=int)
    lstm_cmd.add_argument('--layer_size', type=int)

    mode = parser.add_subparsers(title='mode', dest='mode', help='Operation: [train] train model, [test] evaluate model')
    mode.required = True
    # train
    train_cmd = mode.add_parser('train')
    train_cmd.add_argument('--train_list', type=str)
    train_cmd.add_argument('--val_list', type=str)
    train_cmd.add_argument('--continue_from', type=str, default=None)
    train_cmd.add_argument('--epochs', type=int)
    train_cmd.add_argument('--save_interval', type=int)
    train_cmd.add_argument('--lr', '--learning_rate', type=float)
    # test
    test_cmd = mode.add_parser('test')
    test_cmd.add_argument('--test_path', type=str, default=None)
    test_cmd.add_argument('--test_list', type=str, default=None)
    test_cmd.add_argument('--model_path', type=str, default=None)
    test_cmd.add_argument('--fast_test', action='store_true')
    test_cmd.add_argument('--show_plot', action='store_true')

    '''
    Extra arguments
    Set True: use_scheduler, 
              quite (no output on stdout), 
              force_lr (force lr when a model is loaded from continue_from)
    float:  dropout, 
            max_ori_error (err. threshold for priority grv in degrees)
            max_velocity_norm (filter outliers in training) 
    '''

    args, unknown_args = parser.parse_known_args()
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    args, kwargs = load_config(default_config_file, args, unknown_args)

    print(args, kwargs)
    if args.mode == 'train':
        train(args, **kwargs)
    elif args.mode == 'test':
        if not args.model_path:
            raise ValueError("Model path required")
        args.batch_size = 1
        test(args, **kwargs)
