import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from option import args
import data

if __name__ == '__main__':
    args.dir_data = "/data/shizun/"
    args.scale = "2"
    args.ext ='sep'
    loader = data.Data(args)
