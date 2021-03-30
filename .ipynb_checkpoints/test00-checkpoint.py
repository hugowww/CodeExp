from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

for i in range(1,10):
    writer.add_hparams(
        hparam_dict={'lr': 0.1*i, 'bsize': i},
        metric_dict={'haccuracy': 10*i, 'hloss': 10*i}
    )
    
with SummaryWriter('run/exp-6', flush_secs=10) as w:
    for i in range(5):
        w.add_hparams({'lr': 0.1*i, 'bsize': i},
                      {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})