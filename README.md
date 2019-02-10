# BasisCustomize
Categorical Metadata Representation for Customized Text Classification

This Pytorch code was used in the experiments of the research paper

Jihyeok Kim*, Reinald Kim Amplayo*, Kyungjae Lee, Sua Sung, Minji Seo, and Seung-won Hwang. **Categorical-Metadata-Representation-for-Customized-Text-Classification**. _TACL_, 2019.
(* Authors have equal contributions)

### Run the Code!
#### Clone github code
~~~bash
$ git clone "https://github.com/zizi1532/BasisCustomize.git"
~~~
#### Prerequisite
(Experiment has been done in python 3.5, Linux 16.04 LTS, cuda 8.0, Geforce GTX 1080Ti)
1) Make virtual environment
~~~bash
$ virtualenv basis_customize --python=python3.5
$ source basis_customize/bin/activate
~~~
2) Install pytorch (Stable 1.0)  - follow instruction in https://pytorch.org/
3) Install required python packages
~~~bash
$ cd BasisCustomize
$ pip install -r requirements.txt # required python packages
$ apt-get install p7zip # required for unzip yelp2013 dataset
~~~

#### 1. DownLoad & Preprocess Dataset
1) Yelp2013
~~~bash
$ cd dataset/yelp2013
$ ./download_yelp.sh
~~~
2) AAPR
3) Polmed

#### 2. Train
~~~bash
$ cd src
$ python main.py {arguments}
~~~
For example
~~~bash
$ cat run.sh
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
$ ./run.sh
~~~
#### Arguments

|arg|	description|	default|
|---|---|---|
|model_type|	model type: BiLSTM, word_cust, attention_cust, linear_cust, bias_cust, word_basis_cust, encoder_basis_cust, attention_basis_cust, linear_basis_cust, bias_basis_cust	|mandatory|
|domain|	dataset: yelp2013, aapr, polmed	|mandatory|
|num_bases| number of bases (required for basis model type)| mandatory if model using basis customize else 0|
|vocab_dir| directory of vocabulary; each predefined vocabulary is in ../predefined_vocab/{domain}/|
|train_datadir| directory of train data; in ../dataset/{domain}/train.txt|default path; yelp2013 train.txt|
|dev_datadir| directory of development data; in ../dataset/{domain}/dev.txt|default path; yelp2013 dev.txt|
|test_datadir| directory of test data; in ../dataset/{domain}/test.txt|default path; yelp2013 test.txt|
|word_dim| word vector dimension|300|
|meta_dim| latent vector dimension of meta unit|128|
|valid_step| evaluate with development set every {valid_step} iteration|1000|
|batch_size| - |32|
|epoch| maximum number of epoch| 10|
|device| cpu or cuda (specify index if you have multiple gpu, e.g. cuda:0 or cuda:1,.)|cuda|
|pretrained_word_em_dir| directory of pretrained word vector| each pretrained word vectors is in ../predefined_vocab/{domain}/|
|max_grad_norm| for gradient cliping| 3.0|

 

TODO by Jihyeok

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
