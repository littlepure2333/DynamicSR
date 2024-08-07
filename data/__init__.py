from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            for d in args.data_train:
                if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                    m = import_module('data.benchmark')
                    datasets.append(getattr(m, 'Benchmark')(args, train=True, name=d))
                elif d in ['Set5_PSNR', 'Set14_PSNR', 'B100_PSNR', 'Urban100_PSNR']:
                    m = import_module('data.benchmark_psnr')
                    datasets.append(getattr(m, 'Benchmark_PSNR')(args, train=True, name=d))
                elif d == 'DIV2K_PSNR':
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, name='DIV2K'))
                elif d == 'DIV2K_SWITCHABLE':
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, name='DIV2K'))
                elif d == 'DIV2K_MEANTIME':
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, name='DIV2K'))
                elif d == 'DIV2K_DYNAMIC':
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, name='DIV2K'))
                else:
                    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                    m = import_module('data.' + module_name.lower())
                    datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = dataloader.DataLoader(
                MyConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        for d in args.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d, benchmark=True)
            elif d in ['Set5_PSNR', 'Set14_PSNR', 'B100_PSNR', 'Urban100_PSNR']:
                d = d.split("_PSNR")[0]
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(args, train=False, name=d)
            elif d == 'DIV2K_PSNR':
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    testset = getattr(m, module_name)(args, train=False, name='DIV2K')
            elif d == 'DIV2K_SWITCHABLE':
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    testset = getattr(m, module_name)(args, train=False, name='DIV2K')
            elif d == 'DIV2K_MEANTIME':
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    testset = getattr(m, module_name)(args, train=False, name='DIV2K')
            elif d == 'DIV2K_DYNAMIC':
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    testset = getattr(m, module_name)(args, train=False, name='DIV2K')
            elif d == 'TEST8K':
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    testset = getattr(m, module_name)(args, train=False)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(args, train=False, name=d)

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads,
                )
            )
