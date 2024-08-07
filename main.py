from numpy import index_exp
import torch

import utility
import data
import model
import loss
from option import args
# if args.assistant:
#     from trainer_autoassist import Trainer
# else:
if args.dynamic:
    from trainer_dynamic import Trainer
elif args.efficient:
    from trainer_efficient import Trainer
elif args.switchable:
    from trainer_switchable import Trainer
elif args.meantime:
    from trainer_meantime import Trainer
elif args.succession:
    from trainer_succession import Trainer
elif args.match:
    from trainer_match import Trainer
elif args.decision:
    from trainer_decision import Trainer
elif args.check:
    from trainer_check import Trainer
elif args.ada:
    from trainer_ada import Trainer
else:
    from trainer import Trainer
import os
# if torch.cuda.is_available():
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            if args.switchable:
                loader = []
                for part in args.data_part_list:
                    args.file_suffix = part
                    loader.append(data.Data(args))
            else:
                loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            # utility.print_params(_model,checkpoint,args)
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()
            # TODO add final test

            checkpoint.done()

if __name__ == '__main__':
    main()
