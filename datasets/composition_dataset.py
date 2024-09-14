from itertools import product

import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets.augment.transform import transform_image, transform_image_aug


def _preprocess_utzappos(objects, attributes):
    custom_map = {
        'faux.fur': 'faux_fur',
        'faux.leather': 'faux_leather',
        'full.grain.leather': 'full_grain_leather',
        'hair.calf': 'hair_calf_leather',
        'patent.leather': 'patent_leather',
        'boots.ankle': 'ankle_boots',
        'boots.knee.high': 'knee_high_boots',
        'boots.mid-calf': 'midcalf_boots',
        'shoes.boat.shoes': 'boat_shoes',
        'shoes.clogs.and.mules': 'clogs_shoes',
        'shoes.flats': 'flats_shoes',
        'shoes.heels': 'heels',
        'shoes.loafers': 'loafers',
        'shoes.oxfords': 'oxford_shoes',
        'shoes.sneakers.and.athletic.shoes': 'sneakers',
        'traffic_light': 'traffic_light',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower',
        'nubuck' : 'grainy_leather',
    }
    objects_new = []
    for i, obj in enumerate(objects):
        if obj.lower() in custom_map:
            obj = custom_map[obj.lower()]
        if '_' in obj:
            obj = obj.replace("_", " ")
        objects_new.append(obj.lower())

    attributes_new = [attr.replace(".", " ").lower() for attr in attributes]

    return objects_new, attributes_new



def pairwise_class_names(root, split):
    """ parse the dataset classnames
    """
    def parse_pairs(pair_list):
        with open(pair_list, 'r') as f:
            pairs = f.read().strip().split('\n')
            # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
            pairs = [t.split() for t in pairs]
            pairs = list(map(tuple, pairs))
        attrs, objs = zip(*pairs)

        # preprocess words
        if root.split('/')[-1] == 'ut-zappos':
            objs, attrs = _preprocess_utzappos(objs, attrs)
            pairs = [(a, o) for a, o in zip(attrs, objs)]

        return attrs, objs, pairs

    tr_attrs, tr_objs, tr_pairs = parse_pairs(
        '%s/%s/train_pairs.txt' % (root, split))
    vl_attrs, vl_objs, vl_pairs = parse_pairs(
        '%s/%s/val_pairs.txt' % (root, split))
    ts_attrs, ts_objs, ts_pairs = parse_pairs(
        '%s/%s/test_pairs.txt' % (root, split))

    all_attrs, all_objs = sorted(
        list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs)))
    all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))
    all_pairs_open = list(product(all_attrs, all_objs))

    out = {
        'all_attrs': all_attrs,
        'all_objs': all_objs,
        'all_pairs': all_pairs,
        'all_pairs_open': all_pairs_open,
        'train_pairs': tr_pairs,
        'val_pairs': vl_pairs,
        'test_pairs': ts_pairs
    }
    return out


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            num_aug=0,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False
            # inductive=True
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world

        # new addition
        # if phase == 'train':
        #     self.inductive = inductive
        # else:
        #     self.inductive = False

        self.feat_dim = None
        if num_aug > 0:
            self.transform = transform_image_aug(num_aug=num_aug, augmix=True)
        else:
            self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if self.root.split('/')[-1] == 'ut-zappos':
                objs, attrs = _preprocess_utzappos([obj], [attr])
                obj, attr = objs[0], attrs[0]

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data


    def parse_split(self):
        out = pairwise_class_names(self.root, self.split)
        return out['all_attrs'], out['all_objs'], out['all_pairs'], out['train_pairs'], out['val_pairs'], out['test_pairs']


    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data

    def __len__(self):
        return len(self.data)
