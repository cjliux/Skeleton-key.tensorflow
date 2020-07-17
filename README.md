# Skeleton-key-tensorflow

The implementation of the model in paper "Skeleton Key: Image Captioning by Skeleton-Attribute Decomposition"
**Forked from https://github.com/feiyu1990/Skeleton-key**
* Fix bugs in the original code

Prerequisite
The model uses tensorflow, and the preprocessing of the captions requires Stanford NLP Core and you need to download COCO dataset first.

Dataset preprocessing
Use create_data.py to create the skeleton-attribute dataset from COCO.

Test
Download the pre-trained model at Drive, and put the model under ./model Use run_inference.py to test the model on the 5000-split test set.


