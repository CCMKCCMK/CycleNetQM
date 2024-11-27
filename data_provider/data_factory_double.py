from data_provider.data_loader import Dataset_Electricity
from torch.utils.data import DataLoader

def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:  # train/val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Dataset_Electricity(
        root_path=args.root_path,
        residual_csv_path=args.residual_csv_path,
        cycle_csv_path=args.cycle_csv_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        cycle=args.cycle
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader