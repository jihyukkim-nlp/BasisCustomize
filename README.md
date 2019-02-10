# BasisCustomize
Categorical Metadata Representation for Customized Text Classification

This PyTorch code was used in the experiments of the research paper

Jihyeok Kim*, Reinald Kim Amplayo*, Kyungjae Lee, Sua Sung, Minji Seo, and Seung-won Hwang. **Categorical-Metadata-Representation-for-Customized-Text-Classification**. _TACL_, 2019.
(* Authors have equal contributions)

### Run the Code!

#### Prerequisite

1) PyTorch 1.0
2) Other requirements are listed in `requirements.txt`.

#### 1. Preprocess Dataset

We provided a shell script `dataset/yelp2013/download_yelp.sh` that downloads and preprocess the Yelp 2013 dataset. Preprocessing can be similarly done with other datasets as well (see below for download links).

We also provided the vocabulary and word vectors used in our experiments (in the `predefined_vocab/yelp2013` directory) to better replicate the results reported in the paper.

#### 2. Train and Test the Models

The `src/main.py` trains the model using the given training and dev sets, and subsequently tests the model on the given test set. There are multiple arguments that need to be set, but the most important (and mandatory) ones are the following:

- `model_type`: the type and method of customization, which can be assigned as either `BiLSTM` (no customization), or `<location>[_basis]_cust`, where `<location>` can be any of the following: word, encoder, attention, linear, bias.
- `domain`: the dataset directory name (e.g. yelp2013)
- `num_bases`: the number of bases (only required when basis customization is used)

An example execution is:

~~~bash
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
~~~

### Download the Datasets!

There are three datasets used in the paper: Yelp 2013, AAPR, and PolMed.

To download Yelp 2013, refer to the following <a href="https://drive.google.com/open?id=1PxAkmPLFMnfom46FMMXkHeqIxDbA16oy">link</a> from the original authors.

Although they were constructed by different authors (please refer to these links for <a href="https://github.com/lancopku/AAPR">AAPR</a> and <a href="https://www.figure-eight.com/">PolMed</a>, we use specific data splits for the AAPR and PolMed datasets.
Download our splits <a href="https://github.com/zizi1532/BasisCustomize/releases/download/1.0/datasets.zip">here</a>.

### Cite the Paper!

To cite the paper/code/data splits, please use this BibTex:

```
@article{kim2019categorical,
	Author = {Jihyeok Kim and Reinald Kim Amplayo and Kyungjae Lee and Sua Sung and Minji Seo and Seung-won Hwang},
	Journal = {TACL},
	Year = {2019},
	Title = {Categorical Metadata Representation for Customized Text Classification}
}
```

If using specific datasets, please also cite the original authors of the datasets:

Yelp 2013
```
@inproceedings{tang2015learning,
	Author = {Duyu Tang and Bing Qin and Ting Liu},
	Booktitle = {ACL},
	Location = {Beijing, China},
	Year = {2015},
	Title = {Learning Semantic Representations of Users and Products for Document Level Sentiment Classification},
}
```

AAPR
```
@inproceedings{tang2015learning,
	Author = {Pengcheng Yang and Xu Sun and Wei Li and Shuming Ma},
	Booktitle = {ACL: Short Papers},
	Location = {Melbourne, Australia},
	Year = {2018},
	Title = {Automatic Academic Paper Rating Based on Modularized Hierarchical Convolutional Neural Network},
}
```

If there are any questions, please send Jihyeok Kim an email: zizi1532 at yonsei dot ac dot kr
