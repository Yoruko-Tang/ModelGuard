import os
import os.path as osp
import sys


def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)

def parse_defense_kwargs(kwargs_str):
    kwargs = dict()
    for entry in kwargs_str.split(','):
        if len(entry) < 1:
            continue
        key, value = entry.split(':')
        assert key not in kwargs, 'Argument ({}:{}) conflicts with ({}:{})'.format(key, value, key, kwargs[key])
        try:
            # Cast into int if possible
            value = int(value)
        except ValueError:
            try:
                # Try with float
                value = float(value)
            except ValueError:
                # Give up
                pass
        kwargs[key] = value
    return kwargs

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull,'w')
    
    def __exit__(self,exc_type,exc_val,exc_tb):
        sys.stdout.close()
        sys.stdout=self._original_stdout


BBOX_CHOICES = ['none', 'topk', 'rounding',
                'reverse_sigmoid', 'reverse_sigmoid_wb',
                'rand_noise', 'rand_noise_wb',
                'mad', 'mad_wb','mld']
