#! /bin/bash
echo "running bash script ... "
python3 -W ignore main.py \
--model_type linear_basis_cust \
--num_bases 3 \
--domain yelp2013 \
--vocab_dir ../predefined_vocab/yelp2013/42939.vocab \
--pretrained_word_em_dir ../predefined_vocab/yelp2013/word_vectors.npy \
--train_datadir ../dataset/yelp2013/processed_data/train.txt \
--dev_datadir ../dataset/yelp2013/processed_data/dev.txt \
--test_datadir ../dataset/yelp2013/processed_data/test.txt \
--word_dim 300 \
--state_size 256 \
--valid_step 1000 \
