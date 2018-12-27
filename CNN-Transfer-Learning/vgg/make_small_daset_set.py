
import argparse

parser = argparse.ArgumentParser(description='Transfer learning using VGG')
parser.add_argument('--traindir', required=True)
parser.add_argument('--validdir', required=True)
parser.add_argument('--classes', required=True, nargs='+')

