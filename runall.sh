#!/bin/bash
# My first script

theano-cache clear
theano-cache purge.
cd data_train/
perl create_train.pl
cd ../extract_gmm_10cv/
perl gmm_all.pl
cd ../GCCA_toolbox/
matlab -r runallbnrc_sp1
cd ../extract_gmm_10cv/
matlab -r highml_sp
perl data_all2.pl
cd ../cnn_data/
perl cnn_all2.pl
cd ../cnn
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python2 convolutional_mlp.py
perl cnn_features.pl
matlab -r format2
matlab -r mpqa_rnn
cd ../gp/gp
perl rungp.pl
perl runcal.pl
matlab -r gp_err
