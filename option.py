import argparse
from urllib import parse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true', default=False,
                    help='Enables debug mode')
parser.add_argument('--template', default='EDSR_decision',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--device', type=str, default="0,1",
                    help='indicate cuda visible devices')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../../dataset',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--cutblur', type=float, default=None,
                    help='alpha value in cutblur, 0 denotes no cutblur and modified EDSR, None denotes normal EDSR')
parser.add_argument('--patchnet', action='store_true', default=False,
                    help='use patchnet to online hard patches mining')
parser.add_argument('--ohem', action='store_true', default=False,
                    help='use online hard examples mining (cooperate with args.data_partion)')
parser.add_argument('--data_partion', type=float, default=1,
                    help='allow the first what partion to train, range[-1,1], negative denotes reverse order')
parser.add_argument('--final_data_partion', type=float, default=None,
                    help='final partion to train')
parser.add_argument('--file_suffix', type=str, default='_psnr_np.pt',
                    help='metrics index file suffix loaded in dataloader')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--chop', action='store_true', default=False,
                    help='enable memory-efficient forward')
parser.add_argument('--no_augment', action='store_true', default=False,
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--f', type=int, default=64,
                    help='number of feature maps in the first conv')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true', default=False,
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Option for Residual channel attention network (RCAN)
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# Training specifications
parser.add_argument('--reset', action='store_true', default=False,
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true', default=False,
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true', default=False,
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',  default=False,
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', default=False,
                    help='save output results')
parser.add_argument('--save_gt', action='store_true', default=False,
                    help='save low-resolution and high-resolution images together')
parser.add_argument('--ssim', action='store_true', default=False,
                    help='caculate and log SSIM (time-consuming for large image)')
parser.add_argument('--lpips_alex', action='store_true', default=False,
                    help='calculate IPIPS using alexnet')
parser.add_argument('--lpips_vgg', action='store_true', default=False,
                    help='calculate IPIPS using vggnet')
parser.add_argument('--save_psnr_list', action='store_true', default=False,
                    help="save every test image's psnr in a list")

# auto assist specifications
parser.add_argument("--assistant", action="store_true", default=False, 
                    help="use auto assistant")
parser.add_argument("--base_prob", type=float, default=0.3, 
                    help="base pass probability (default: 0.3)")
parser.add_argument("--sec_method", type=str, default="mean", 
                    help="method to shrink examples (default: mean)")

# dynamic SR
parser.add_argument("--dynamic", action="store_true", default=False, 
                    help="use dynamic SR")
parser.add_argument("--efficient", action="store_true", default=False, 
                    help="use multi-exit SR efficiently")
parser.add_argument("--switchable", action="store_true", default=False, 
                    help="use switchable SR")
parser.add_argument("--succession", action="store_true", default=False, 
                    help="optimize multi-exit SR in succession")
parser.add_argument("--shared_tail", action="store_true", default=False,
                    help="if share the parameter of tail(upsampler)")
parser.add_argument("--conv_thre", type=int, default=15,
                    help="convergence threshold, how many epoch the metric didn't imporve then converged")
parser.add_argument("--freeze", action="store_true", default=False, 
                    help="whether freeze previous exit's parameters")
parser.add_argument("--meantime", action="store_true", default=False, 
                    help="use switchable SR and meantime data")
parser.add_argument('--cap_mult_list', type=tuple, default=(0.33, 0.66, 1.0),
                    help='capacity multiple list')
parser.add_argument('--data_part_list', type=tuple, default=('easy_x2_descending', 'midd_x2_descending', 'hard_x2_descending'),
                    help='data part list')
parser.add_argument('--exit_interval', type=int, default=1,
                    help='every N layers will output an exit')
parser.add_argument("--match", action="store_true", default=False,
                    help="match bins with multi-exits")
parser.add_argument("--bins", type=int, default=320,
                    help="defines the number of equal-width bins")
parser.add_argument("--statistics_file", type=str,
                    help="the dataset statistics file path")
parser.add_argument("--n_test_samples", type=int, default=20,
                    help="the number of test samples on every bins")
parser.add_argument("--decision", action="store_true", default=False,
                    help='use early-exit decision maker')
parser.add_argument("--exit_threshold", type=float, default=0.8,
                    help='early exit decision threshold')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

