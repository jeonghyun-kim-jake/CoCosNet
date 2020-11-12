import os
import random
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset

from data.base_dataset import BaseDataset, get_params, get_transform

import torch
import torchvision.transforms as transforms

from PIL import Image

from os.path import isfile, join, abspath


class BFADE20KDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=150)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        instance_dir = opt.instance_dir
        instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        image_files = []
        instance_files = []

        for p in image_paths:            
          if p.endswith('.jpg') or  p.endswith('.png') :
            image_files.append(p)
        
        for p in instance_paths:             
          if p.endswith('.png') and not  p.endswith('.png.png') :
            instance_files.append(p)

        assert len(instance_files) == len(image_files), "The #images in {} and {} do not match.".format(len(instance_files),len(image_files))

        return instance_files, image_files

    def get_ref(self, opt):
        extra = '_test' if opt.phase == 'test' else ''
        with open('./data/ade20k_ref{}.txt'.format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else:
                val = [items[1], items[-1]]
            ref_dict[key] = val
        train_test_folder = ('training', 'validation')
        return ref_dict, train_test_folder

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label_tensor, params1 = self.get_label_tensor(label_path)
        file_name = os.path.basename(label_path)

        # input image (real images)
        image_path = self.image_paths[index]
        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), \
                "The label_path %s and image_path %s don't match." % \
                (label_path, image_path)

        # input image
        image = Image.open(image_path)
        image = image.convert('RGB')
        transform_image = get_transform(self.opt, params1)
        image_tensor = transform_image(image)

        ref_tensor = 0
        label_ref_tensor = 0

        # input's segment
        segment_path = join(self.opt.instance_dir, file_name)
        path_ref = image_path

        # input_image --> ref
        image_ref = Image.open(path_ref).convert('RGB')

        # ref label -> expansion
        path_ref_label = join(self.opt.segment_dir, file_name)
        label_ref_tensor, params = self.get_label_tensor(path_ref_label)
        transform_image = get_transform(self.opt, params)

        ref_tensor = transform_image(image_ref)

        self_ref_flag = torch.zeros_like(ref_tensor)



        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      'self_ref': self_ref_flag,
                      'ref': ref_tensor,
                      'label_ref': label_ref_tensor
                      }
                      
        print("\n\n====")
        print("image_path", image_path)
        print("label_path", label_path)
        print("segment_path", segment_path)
        print("====\n\n")
        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
