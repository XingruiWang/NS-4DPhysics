import argparse
import os
import utils
import torch


class BaseOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--run_dir', default='scratch/test_run', type=str, help='experiment directory')
        self.parser.add_argument('--dataset', default='superclevr', type=str, help='dataset')
        self.parser.add_argument('--load_checkpoint_path', default=None, type=str, help='load checkpoint path')
        self.parser.add_argument('--gpu_ids', default='0', type=str, help='ids of gpu to be used')
        
        self.parser.add_argument('--obj_ann_path', type=str, help='superclevr object ann path')
        self.parser.add_argument('--img_dir', type=str, help='superclevr image directory')
        self.parser.add_argument('--scene_path', type=str, help='superclevr scene file')
        self.parser.add_argument('--type', default='part', type=str, help='superclevr scene file')

        
        self.parser.add_argument('--clevr_mini_img_dir', default='../../data/raw/CLEVR_mini/images', type=str, help='clevr-mini image directory')
        self.parser.add_argument('--clevr_mini_ann_path', default='../../data/attr_net/objects/clevr_mini_objs.json', type=str, help='clevr-mini objects annotation file')
        
        self.parser.add_argument('--concat_img', default=1, type=int, help='concatenate original image when sent to network')
        self.parser.add_argument('--split_id', default=3500, type=int, help='splitting index between train and val images')
        self.parser.add_argument('--batch_size', default=64, type=int, help='batch size')
        self.parser.add_argument('--num_workers', default=1, type=int, help='number of workers for loading')
        self.parser.add_argument('--learning_rate', default=0.002, type=float, help='learning rate')
        self.parser.add_argument('--pred_bbox', default='', type=str, help='learning rate')
        self.parser.add_argument('--output_file', default='', type=str, help='learning rate')
        self.parser.add_argument('--resume', default='', type=str, help='learning rate')
        self.parser.add_argument('--multiple_dataset', action="store_true")
        self.parser.add_argument('--use_aug_ds', action="store_true")
        self.parser.add_argument('--bbox_mode', choices=['XYXY', 'XYWH'], default='XYWH')

        

        

        self.initialized = True

    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        # parse gpu id list
        str_gpu_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_gpu_ids:
            if str_id.isdigit() and int(str_id) >= 0:
                self.opt.gpu_ids.append(int(str_id))
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])
        else:
            print('| using cpu')
            self.opt.gpu_ids = []

        # print and save options
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        utils.mkdirs(self.opt.run_dir)

        if self.is_train:
            filename = 'train_opt.txt'
        else:
            filename = 'test_opt.txt'
        file_path = os.path.join(self.opt.run_dir, filename)
        with open(file_path, 'wt') as fout:
            fout.write('| options\n')
            for k, v in sorted(args.items()):
                fout.write('%s: %s\n' % (str(k), str(v)))

        return self.opt


class TrainOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--num_iters', default=100000, type=int, help='total number of iterations')
        self.parser.add_argument('--num_epochs', default=80, type=int, help='total number of iterations')
        self.parser.add_argument('--display_every', default=20, type=int, help='display training information every N iterations')
        self.parser.add_argument('--checkpoint_every', default=1, type=int, help='save every N iterations')
        self.parser.add_argument('--shuffle_data', default=1, type=int, help='shuffle dataloader')
        
        self.is_train = True

class ValOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--split', default='val')
        self.parser.add_argument('--load_path', required = True)

        self.parser.add_argument('--shuffle_data', default=1, type=int, help='shuffle dataloader')
        self.is_train = False

class TestOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--split', default='test')
        self.parser.add_argument('--load_path', required = True)

        self.parser.add_argument('--shuffle_data', default=1, type=int, help='shuffle dataloader')
        self.is_train = False


def get_options(mode):
    if mode == 'train':
        opt = TrainOptions().parse()
    elif mode == 'test':
        opt = TestOptions().parse()
    elif mode == 'val':
        opt = ValOptions().parse()
    else:
        raise ValueError('Invalid mode for option parsing: %s' % mode)
    return opt