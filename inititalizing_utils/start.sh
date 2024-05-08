#!bash
python ../data_utils/prepare_data.py --data_root . --output_dir . --dataset refcoco --generate_mask
python ../data_utils/folder_to_lmdb.py -j anns/refcoco/train.json -i images/train2014 -m masks/refcoco/ -o lmdb/refcoco
python ../data_utils/folder_to_lmdb.py -j anns/refcoco/val.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../data_utils/folder_to_lmdb.py -j anns/refcoco/testA.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco
python ../data_utils/folder_to_lmdb.py -j anns/refcoco/testB.json -i images/train2014/ -m masks/refcoco -o lmdb/refcoco