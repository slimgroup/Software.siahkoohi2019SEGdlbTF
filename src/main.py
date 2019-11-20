import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import wavefield_reconstrcution

parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment_dir', dest='experiment_dir', default='wave_recon', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=150, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--image_size0', dest='image_size0', type=int, default=172, help='then crop to this size')
parser.add_argument('--image_size1', dest='image_size1', type=int, default=172, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=2, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=2, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=10000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=1000, help='print the debug information every print_freq iterations')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./log', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=1000.0, help='weight on L1 term in objective')
parser.add_argument('--data_path', dest='data_path', type=str, default='/home/ec2-user/data/', help='path of the train/test dataset')
parser.add_argument('--freq', dest='frequency', type=float, default=10.0, help='frequency of data')
parser.add_argument('--sampling_rate', dest='sampling_rate', type=float, default=0.1, help='sampling rate')
parser.add_argument('--sampling_scheme', dest='sampling_scheme', type=str, default='random', help='sampling scheme')
args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = wavefield_reconstrcution(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)

if __name__ == '__main__':
    tf.app.run()
