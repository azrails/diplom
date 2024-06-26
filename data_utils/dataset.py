import os
import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from .folder_to_lmdb import loads_data

DataParams = namedtuple("DataParams", ['inv_mat', 'img', 'mask', 'img_size', 'sents'])

class ReferenceDataset(Dataset):
    def __init__(self, lmdb_path, mode, image_size, contrastive=False, contrastive_prob=0.5, with_img=False):
        self.lmdb_path = lmdb_path
        self.mode = mode
        self.input_size = image_size
        self.db_connection = None
        self.__load_meta()
        self.contrastive = contrastive
        self.with_img = with_img
        self.contrastive_prob = contrastive_prob
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)

    def __load_meta(self):
        self.__init_db()
        self.db_connection.close()
        self.db_connection = None

    def __init_db(self):
        """
        init connection to lmdb db and extract meta information
        """
        self.db_connection = lmdb.open(
            self.lmdb_path,
            subdir=os.path.isdir(self.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with self.db_connection.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))


    def __len__(self):
        return self.length
    

    def extract_data(self, index):
        with self.db_connection.begin(write=False) as txn:
            data = txn.get(self.keys[index])
        
        data = loads_data(data)
        #chose sentence
        sent_idx = np.random.choice(data['num_sents'])
        sent = data['sents'][sent_idx]

        #extract and convert image and mask
        img = cv2.imdecode(
            np.frombuffer(data['img'], np.uint8),
            cv2.IMREAD_COLOR
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imdecode(
            np.frombuffer(data['mask'], np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        img_size = img.shape[:2]

        #downscale matrix
        mat, mat_inv = self.get_transform_mat(img_size, True)

        #img and mask transofrms
        #Помнить что тут перестановка размеров т.к входное (480, 640)
        transformed_mask = cv2.warpAffine(
            mask,
            mat,
            dsize=self.input_size[::-1],
            flags=cv2.INTER_LINEAR,
            borderValue=0.
        )
        transformed_img = cv2.warpAffine(
            img,
            mat,
            dsize=self.input_size[::-1],
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        )
        tensor_img = self.convert_to_tensor(transformed_img)
        tensor_mask = self.convert_to_tensor(transformed_mask, mask=True)
        return tensor_img, tensor_mask, sent, transformed_img


    def __getitem__(self, index):
        if self.db_connection is None:
            #initialize db connection
            self.__init_db()

        is_contrastive = False
        tensor_img, tensor_mask, sent, image = self.extract_data(index)
        if self.contrastive is True:
            is_contrastive = np.random.random() > self.contrastive_prob
            if is_contrastive:
                neg_idx = index
                while neg_idx == index:
                    neg_idx = np.random.choice(self.length)
                negative_tensor_img, negative_tensor_mask, negative_sent, _ = self.extract_data(neg_idx) 
                choise = np.random.choice(2)
                if choise == 0:
                    sent = negative_sent
                else:
                    tensor_img = negative_tensor_img
                    tensor_mask = negative_tensor_mask

        if self.mode == "train" or self.mode == "val":
            train_mask = self.extract_segment(tensor_img, tensor_mask)
            if self.contrastive:
                if self.with_img is False:
                    return train_mask, sent, int(not is_contrastive)
                return tensor_img, train_mask, sent, int(not is_contrastive)

            if self.with_img is False:
                return train_mask, sent
            return tensor_img, train_mask, sent


        if self.mode == "test":
            train_mask = self.extract_segment(tensor_img, tensor_mask)
            return image, tensor_img, tensor_mask, train_mask, sent
        #block used inly in test
        # params = DataParams(mat_inv, img, mask, img_size, data['sents'])
        # return tensor_img, tensor_mask, params

    @staticmethod
    def extract_segment(img: torch.Tensor, mask: torch.Tensor, negative:bool = False) -> torch.tensor:
        if negative == True:
            mask = torch.where(mask > 0, 0., 1.)
        else:
            mask = torch.where(mask > 0, 1., 0.)
        mask = mask[None, :, :].repeat(3, 1, 1)
        return img * mask


    def convert_to_tensor(self, img, mask:bool=False):
        """
        Converts image or mask to tensor,
        flag means image and corresponding normalization
        """
        if mask:
            img = torch.from_numpy(img)
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not torch.is_floating_point(img):
            img = img.to(torch.float32)
        
        #normalize image
        if not mask:
            img.div_(255.).sub_(self.mean).div_(self.std)
        return img


    def get_transform_mat(self, img_size, inverse=False):
        """
        Getting affine matrix transformation for downscaling image
        """
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None


    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"lmdb_path={self.lmdb_dir}, " + \
            f"mode={self.mode}, " + \
            f"image_size={self.input_size}, "


def affine_transform(img: np.ndarray, to_size: tuple[int, int], inverse:bool =False):
    ori_h, ori_w = img.shape[:2]
    inp_h, inp_w = to_size
    scale = min(inp_h / ori_h, inp_w / ori_w)
    new_h, new_w = ori_h * scale, ori_w * scale
    bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

    src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
    dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                    [bias_x, new_h + bias_y]], np.float32)

    mat = cv2.getAffineTransform(src, dst)
    transformed_img = cv2.warpAffine(
            img,
            mat,
            dsize=to_size[::-1],
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
    )
    return transformed_img

