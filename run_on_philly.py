import argparse
import os
os.system("ls")
parser = argparse.ArgumentParser(description='Helper run')
# general
parser.add_argument('--modelDir', help='model directory', type=str)
parser.add_argument('--dataDir', help='data directory', type=str)
parser.add_argument('--logDdir', help='log directory', type=str)
parser.add_argument('--auth', help='auth', required=True, type=str)
parser.add_argument('--path', help='path', required=True, type=str)
parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
parser.add_argument('--branch', help="branch of code", type=str)
args, rest = parser.parse_known_args()
print args
print rest
os.system("git clone https://{0}@github.com/xvjiarui/Deep-Feature-Flow.git -b {1} dff".format(args.auth, args.branch))
os.chdir("dff")
os.system("ls & sh init.sh")
os.system("ls & python {0} --cfg {1} --modelDir {2} --dataDir {3}".format(args.path, args.cfg, args.modelDir, args.dataDir))