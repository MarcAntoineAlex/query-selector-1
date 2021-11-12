
import time

import numpy as np
import torch


from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn

import ipc
from config import build_parser, Config
from model import Transformer
from data_loader import Dataset_ETT_hour, Dataset_ETT_minute
from metrics import metric
from architect import Architect


def get_model(args):
    return Transformer(args.embedding_size, args.hidden_size, args.input_len, args.dec_seq_len, args.pred_len,
                       output_len=args.output_len,
                       n_heads=args.n_heads, n_encoder_layers=args.n_encoder_layers,
                       n_decoder_layers=args.n_decoder_layers, enc_attn_type=args.encoder_attention,
                       dec_attn_type=args.decoder_attention, dropout=args.dropout)


def get_W(mdl):
    return mdl.W()


def _get_data(args, flag):
    if not args.data == 'ETTm1':
        Data = Dataset_ETT_hour
    else:
        Data = Dataset_ETT_minute
    # timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False;
        drop_last = True;
        batch_size = 32
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False;
        drop_last = False;
        batch_size = 1;
        # freq = args.detail_freq
        # Data = Dataset_Pred
    else:
        shuffle_flag = True;
        drop_last = True;
        batch_size = args.batch_size
        # freq = args.freq

    data_set = Data(
        root_path='data',
        data_path=args.data+'.csv',
        flag=flag,
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        # timeenc=timeenc,
        # freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader


def run_metrics(caption, preds, trues):
    preds = np.array(preds)
    trues = np.array(trues)
    # print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # print('test shape:', preds.shape, trues.shape)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('{} ; MSE: {}, MAE: {}'.format(caption, mse, mae))
    return mse, mae


def run_iteration(teacher, student, trn_loader, val_loader, next_loader, architect, args, message =''):
    preds = []
    trues = []
    total_loss = 0
    elem_num = 0
    steps = 0
    data_count = 0
    target_device = 'cuda:{}'.format(args.local_rank)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(trn_loader):
        try:
            val_data = next(val_iter)
        except:
            val_iter = iter(val_loader)
            val_data = next(val_iter)

        try:
            next_data = next(next_iter)
        except:
            next_iter = iter(next_loader)
            next_data = next(next_iter)



        trn_x = torch.tensor(batch_x, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        trn_y = torch.tensor(batch_y, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        val_data[0] = torch.tensor(val_data[0], dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        val_data[1] = torch.tensor(val_data[1], dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        next_data[0] = torch.tensor(next_data[0], dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        next_data[1] = torch.tensor(next_data[1], dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)

        # assert torch.abs(next_data[0] - trn_x).max().item() == 0

        elem_num += len(trn_x)
        steps += 1

        # teacher.optimA.zero_grad()
        # architect.unrolled_backward(args, (trn_x, trn_y), val_data, next_data, 0.00005, teacher.optim, student.optim, data_count)
        # teacher.optimA.zero_grad()

        teacher.optim.zero_grad()
        result = teacher(trn_x)
        loss = nn.functional.mse_loss(result.squeeze(2), trn_y.squeeze(2), reduction='mean')  # todo: critere
        preds.append(result.detach().cpu().numpy())
        trues.append(trn_y.detach().cpu().numpy())

        unscaled_loss = loss.item()
        total_loss += unscaled_loss
        loss.backward()
        teacher.optim.step()

        student.optim.zero_grad()
        result = student(next_data[0])
        loss_s1 = nn.functional.mse_loss(result.squeeze(2), next_data[1].squeeze(2), reduction='mean')
        target = teacher(next_data[0])
        loss_s2 = nn.functional.mse_loss(result.squeeze(2), target.squeeze(2), reduction='mean')
        loss_s = loss_s1 * 0.5 + loss_s2
        loss_s.backward()
        student.optim.step()
        data_count += batch_x.shape[0]
        print("{} Loss at step {}: t{}  s{}, mean for epoch: {}, mem_alloc: {}".format(message, steps, unscaled_loss,
              loss_s.item(), total_loss / steps,torch.cuda.max_memory_allocated()))
    return preds, trues


def test(model, test_loader, args, message=''):
    preds = []
    trues = []
    total_loss = 0
    elem_num = 0
    steps = 0
    model.eval()
    target_device = 'cuda:{}'.format(args.local_rank)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch = torch.tensor(batch_x, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)
        target = torch.tensor(batch_y, dtype=torch.float16 if args.fp16 else torch.float32, device=target_device)

        elem_num += len(batch)
        steps += 1

        result = model(batch)
        loss = nn.functional.mse_loss(result.squeeze(2), target.squeeze(2), reduction='mean')  # todo: critere
        preds.append(result.detach().cpu().numpy())
        trues.append(target.detach().cpu().numpy())
        unscaled_loss = loss.item()
        total_loss += unscaled_loss
        # print("{} Loss at step {}: {}, mean for epoch: {}, mem_alloc: {}".format(message, steps, unscaled_loss,
        #                                                                          total_loss / steps,
        #                                                                          torch.cuda.max_memory_allocated()))
    model.train()
    return preds, trues


def preform_experiment(args):
    teacher = get_model(args)
    student = get_model(args)
    params = list(get_W(teacher))
    print('Number of parameters: {}'.format(len(params)))

    teacher.to('cuda')
    teacher.optim = Adam(params, lr=0.00005, weight_decay=1e-2)
    teacher.optimA = Adam(teacher.A(), lr=0.3, weight_decay=0.)
    student.to('cuda')
    student.optim = Adam(student.W(), lr=0.00005, weight_decay=1e-2)

    train_data, train_loader = _get_data(args, flag='train')
    valid_data, valid_loader = _get_data(args, flag='test')
    next_data, next_loader = _get_data(args, flag='train')  # todo: check
    test_data, test_loader = _get_data(args, flag='test')


    architect = Architect(args, nn.MSELoss(), teacher, student)

    start = time.time()
    for iter in range(1, args.iterations + 1):
        preds, trues = run_iteration(teacher, student, train_loader, valid_loader, next_loader, architect, args,
                                     message=' Run {:>3}, iteration: {:>3}:  '.format(args.run_num, iter))
        mse, mae = run_metrics("Loss after iteration {}".format(iter), preds, trues)

        v_preds, v_trues = test(teacher, test_loader, args, message="Validation set teacher")
        v_preds_s, v_trues_s = test(student, test_loader, args, message="Validation set student")
        mse_t, mae_t = run_metrics("Loss for validation set teacher", v_preds, v_trues)
        mse_s, mae_s = run_metrics("Loss for validation set student", v_preds_s, v_trues_s)
        # print("Time per iteration {}, memory {}".format((time.time() - start)/iter, torch.cuda.memory_stats()))

    # print(torch.cuda.max_memory_allocated())

    if args.debug:
        teacher.record()
    return mse_t, mae_t, mse_s, mae_s


def critere(model, pred, true, data_count, reduction='mean'):
    weights = model.arch[data_count:data_count + pred.shape[0]]
    weights = torch.softmax(weights, dim=0) ** 0.5
    if reduction != 'mean':
        crit = nn.MSELoss(reduction=reduction)
        return crit(pred * weights, true * weights).mean(dim=(-1, -2))
    else:
        crit = nn.MSELoss()
        return crit(pred * weights, true * weights)

def main():
    parser = build_parser()
    args = parser.parse_args(None)
    conf = Config.from_file('settings/tuned/ts_query-selector_{}.json'.format(args.setting))
    print(conf.to_json())
    args.data = conf.data
    args.seq_len = conf.seq_len
    args.pred_len = conf.pred_len
    args.dec_seq_len = conf.dec_seq_len
    args.hidden_size = conf.seq_len
    args.n_encoder_layers = conf.n_encoder_layers
    args.n_decoder_layers = conf.n_decoder_layers
    args.decoder_attention = conf.decoder_attention
    args.n_heads = conf.heads
    args.batch_size = conf.batch_size
    args.embedding_size = conf.embedding_size
    args.iterations = conf.iterations
    args.exps = conf.exps
    args.dropout = conf.dropout

    results = {'mse_t':[], 'mae_t':[], 'mse_s':[], 'mae_s':[]}
    for i in range(10):
        mse_t, mae_t, mse_s, mae_s = preform_experiment(args)
        results['mse_t'].append(mse_t)
        results['mae_t'].append(mae_t)
        results['mse_s'].append(mse_s)
        results['mae_s'].append(mae_s)
    print('Teacher: Mse {} Mae {} Student: Mse {}  Mae{}'.format(np.array(results['mse_t']).mean(), np.array(results['mae_t']).mean(),
                                                                 np.array(results['mse_s']).mean(), np.array(results['mae_s']).mean()))


if __name__ == '__main__':
    main()


