import argparse
import os
import os.path as osp
import lmdb
import pickle
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def loads_data(buf):
    """
    Deserialize object from buf.
    Args:
        buf: serialized data from dums_data.
    """
    return pickle.loads(buf)


def dumps_data(obj):
    """
    Serialize an object.
    """
    return pickle.dumps(obj)


def folder2lmdb(json_data, img_dir, mask_dir, output_dir, split, write_frequency=1000):
    """
    Generate lmdb wiev for fast data extraction
    """
    lmdb_path = osp.join(output_dir, "%s.lmdb" % split)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=10995116277 * 3, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    tbar = tqdm(json_data)
    for idx, item in enumerate(tbar):
        img = raw_reader(osp.join(img_dir, item['img_name']))
        mask = raw_reader(osp.join(mask_dir, f"{item['segment_id']}.png"))
        data = {'img': img, 'mask': mask, 'cat': item['cat'],
                'seg_id': item['segment_id'], 'img_name': item['img_name'],
                'num_sents': item['sentences_num'], 'sents': [i['sent'] for i in item['sentences']]}
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data(data))
        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def parse_args():
    parser = argparse.ArgumentParser(description='COCO Folder to LMDB.')
    parser.add_argument('-j', '--json-dir', type=str,
                        required=True,
                        help='the name of json file.')
    parser.add_argument('-i', '--img-dir', type=str,
                        required=True,
                        help='the folder of images.')
    parser.add_argument('-m', '--mask-dir', type=str,
                        required=True,
                        help='the folder of masks.')
    parser.add_argument('-o', '--output-dir', type=str,
                        required=True,
                        help='the folder of output lmdb file.')
    parser.add_argument('-s', '--split', type=str,
                        default='train',
                        help='the split type.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.split = osp.basename(args.json_dir).split(".")[0]
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_dir, 'rb') as f:
        json_data = json.load(f)

    folder2lmdb(json_data, args.img_dir, args.mask_dir, args.output_dir, args.split)
