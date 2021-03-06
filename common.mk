# common.mk
SHELL:=/bin/bash
ROOT:=$(shell pwd)
export LC_ALL=C
# -----


# initial setup (adjust to your own environment)
KALDI_ROOT=/home/dyszek/tools/kaldi
KALDI_STEPS=steps# make sure that this symbolic link exists (it should point to egs/wsj/s5/steps)
KALDI_UTILS=utils# make sure that this symbolic link exists (it should point to egs/wsj/s5/utils)
KALDI_CMD=run.pl
NJOBS:=6
OPENFST_ROOT=${KALDI_ROOT}/tools/openfst
MITLM_ROOT=/home/dyszek/tools/mitlm/local-build
IRSTLM_ROOT=${KALDI_ROOT}/tools/irstlm
# -----


# environment variable setup
KALDI_SRC=${KALDI_ROOT}/src
export PATH:=${KALDI_UTILS}:${KALDI_STEPS}:${KALDI_SRC}/bin:${KALDI_SRC}/featbin:${KALDI_SRC}/fstbin:${KALDI_SRC}/latbin:${KALDI_SRC}/lmbin:${KALDI_SRC}/gmmbin:${KALDI_SRC}/nnet3bin:${KALDI_SRC}/chainbin:${KALDI_SRC}/nnet2bin:${KALDI_SRC}/nnetbin:${MITLM_ROOT}/bin:${OPENFST_ROOT}/bin:${IRSTLM_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH:=${OPENFST_ROOT}/lib:${LD_LIBRARY_PATH}
# -----


# main directories
ALIGN_DIR=alignments
AM_TRAIN_DIR=trained-ams
CONFIG_DIR=config
CORPORA_DIR=corpora
DATA_DIR=data
EVALUATION_DIR=evaluation
LEXICON_DIR=lexicons
LM_TRAIN_DIR=trained-lms
# -----


# auxiliary functions
dir_guard=mkdir -p $(@D)
LINK_MFCC_FEATS_TYPE:=mfcc
link_mfcc_data=$(shell mkdir -p $(@D)/data ;\
        ln -sf ${ROOT}/$(word 1, $(^D))/text $(@D)/data/ ;\
        ln -sf ${ROOT}/$(word 1, $(^D))/wav.scp $(@D)/data/ ;\
        ln -sf ${ROOT}/$(word 1, $(^D))/utt2spk $(@D)/data/ ;\
        ln -sf ${ROOT}/$(word 1, $(^D))/spk2utt $(@D)/data/ ;\
        ln -sf ${ROOT}/$(word 1, $(^D))/feats-${LINK_MFCC_FEATS_TYPE}.scp $(@D)/data/feats.scp ;\
        ln -sf ${ROOT}/$(word 1, $(^D))/cmvn-${LINK_MFCC_FEATS_TYPE}.scp $(@D)/data/cmvn.scp)
# -----


# language model (n-gram) training
LEXICON_VERSION:=lexicon-4.0.0
INPUT_CORPUS_LEXICON=${LEXICON_DIR}/${LEXICON_VERSION}.txt

LM_NGRAM_TRAIN_NAME:=
LM_NGRAM_TRAIN_DIR=${LM_TRAIN_DIR}/${LM_NGRAM_TRAIN_NAME}
LM_NGRAM_TSV_SOURCE_CORPORA:=# paths should be glued with underscore ('_')
LM_NGRAM_TSV_PATHS=$(shell echo -e "${LM_NGRAM_TSV_SOURCE_CORPORA}" | tr '_' '\n' | perl -ne 'print "corpora/$$1/$$1.tsv " if /(.*)/')
LM_NGRAM_ORDER:=2
LM_NGRAM_SMOOTHING:=ModKN
LM_NGRAM_PRUNE:=false
LM_NGRAM_PRUNE_THRESHOLDS:='1e-06,1e-06'
# -----


# input for various jobs
INPUT_CORPUS_NAME:=
INPUT_CORPUS_DIR=${DATA_DIR}/${INPUT_CORPUS_NAME}
INPUT_DATA_DIR=${INPUT_CORPUS_DIR}/data
INPUT_LM_NAME:=${INPUT_CORPUS_NAME}
INPUT_LM_DIR=${LM_TRAIN_DIR}/${INPUT_LM_NAME}
INPUT_LANG_DIR=${INPUT_LM_DIR}/lang
INPUT_LANG_DICT_DIR=${INPUT_LANG_DIR}/dict
INPUT_CORPUS_TSV=${CORPORA_DIR}/${INPUT_CORPUS_NAME}/${INPUT_CORPUS_NAME}.tsv
# -----


# GMM-HMM acoustic model training
AM_TRAIN_CORPUS_NAME:=
TRAIN_DATA_DIR=${DATA_DIR}/${AM_TRAIN_CORPUS_NAME}/data
AM_TRAIN_LM_NAME:=
TRAIN_LANG_DIR=${LM_TRAIN_DIR}/${AM_TRAIN_LM_NAME}/lang

AM_TRAIN_MONO_FEAT_TYPE:=mfcc
AM_TRAIN_MONO_DELTAS_DIM:=2
AM_TRAIN_MONO_NAME:=am-mono/feattype=${AM_TRAIN_MONO_FEAT_TYPE}-deltasdim=${AM_TRAIN_MONO_DELTAS_DIM}
AM_TRAIN_MONO_DIR=${AM_TRAIN_DIR}/${AM_TRAIN_CORPUS_NAME}/${AM_TRAIN_LM_NAME}/${AM_TRAIN_MONO_NAME}
AM_ALIGN_MONO_NAME:=${AM_TRAIN_MONO_NAME}
AM_ALIGN_MONO_DIR=${ALIGN_DIR}/${AM_TRAIN_CORPUS_NAME}/${AM_TRAIN_LM_NAME}/${AM_ALIGN_MONO_NAME}

AM_TRAIN_TRI1_FEAT_TYPE:=${AM_TRAIN_MONO_FEAT_TYPE}
AM_TRAIN_TRI1_DELTAS_DIM:=${AM_TRAIN_MONO_DELTAS_DIM}
AM_TRAIN_TRI1_LEAVES:=
AM_TRAIN_TRI1_PDFS:=
AM_TRAIN_TRI1_NAME:=am-tri1/feattype=${AM_TRAIN_TRI1_FEAT_TYPE}-deltasdim=${AM_TRAIN_TRI1_DELTAS_DIM}-leaves=${AM_TRAIN_TRI1_LEAVES}-pdfs=${AM_TRAIN_TRI1_PDFS}
AM_TRAIN_TRI1_DIR=${AM_TRAIN_DIR}/${AM_TRAIN_CORPUS_NAME}/${AM_TRAIN_LM_NAME}/${AM_TRAIN_TRI1_NAME}
AM_ALIGN_TRI1_NAME:=${AM_TRAIN_TRI1_NAME}
AM_ALIGN_TRI1_DIR=${ALIGN_DIR}/${AM_TRAIN_CORPUS_NAME}/${AM_TRAIN_LM_NAME}/${AM_ALIGN_TRI1_NAME}

AM_TRAIN_TRI2_FEAT_TYPE:=${AM_TRAIN_TRI1_FEAT_TYPE}
AM_TRAIN_TRI2_DELTAS_DIM:=${AM_TRAIN_TRI1_DELTAS_DIM}
AM_TRAIN_TRI2_LEAVES:=
AM_TRAIN_TRI2_PDFS:=
AM_TRAIN_TRI2_NAME:=am-tri2/feattype=${AM_TRAIN_TRI2_FEAT_TYPE}-deltasdim=${AM_TRAIN_TRI2_DELTAS_DIM}-leaves=${AM_TRAIN_TRI2_LEAVES}-pdfs=${AM_TRAIN_TRI2_PDFS}
AM_TRAIN_TRI2_DIR=${AM_TRAIN_DIR}/${AM_TRAIN_CORPUS_NAME}/${AM_TRAIN_LM_NAME}/${AM_TRAIN_TRI2_NAME}
AM_ALIGN_TRI2_NAME:=${AM_TRAIN_TRI2_NAME}
AM_ALIGN_TRI2_DIR=${ALIGN_DIR}/${AM_TRAIN_CORPUS_NAME}/${AM_TRAIN_LM_NAME}/${AM_ALIGN_TRI2_NAME}
# -----


# TDNN acoustic model training
AM_TRAIN_TDNN_ALIGN_NAME:=${AM_ALIGN_TRI2_NAME}
AM_TRAIN_TDNN_ALIGN_DIR:=${AM_ALIGN_TRI2_DIR}

AM_TRAIN_TDNN_EPOCHS:=8
AM_TRAIN_TDNN_SPLICE_INDEXES:="-4,-3,-2,-1,0,1,2,3,4_0_-2,2_0_-4,4_0"
AM_TRAIN_TDNN_SPLICE_INDEXES_VERSION:=0.0.0
AM_TRAIN_TDNN_FEAT_TYPE:=mfcc-high-res
AM_TRAIN_TDNN_DELTAS_DIM:=0
AM_TRAIN_TDNN_CMN:=false
AM_TRAIN_TDNN_CVN:=false
AM_TRAIN_TDNN_INITIAL_LR:=0.005
AM_TRAIN_TDNN_FINAL_LR:=0.0005
AM_TRAIN_TDNN_PNORM_INPUT_DIM:=2000
AM_TRAIN_TDNN_PNORM_OUTPUT_DIM:=250
AM_TRAIN_TDNN_RELU_DIM:=800

AM_TRAIN_TDNN_NAME=am-tdnn/feattype=${AM_TRAIN_TDNN_FEAT_TYPE}-deltasdim=${AM_TRAIN_TDNN_DELTAS_DIM}-cmn=${AM_TRAIN_TDNN_CMN}-cvm=${AM_TRAIN_TDNN_CVN}-spliceidxversion=${AM_TRAIN_TDNN_SPLICE_INDEXES_VERSION}-epochs=${AM_TRAIN_TDNN_EPOCHS}-initiallr=${AM_TRAIN_TDNN_INITIAL_LR}-finallr=${AM_TRAIN_TDNN_FINAL_LR}-pnorminputdim=${AM_TRAIN_TDNN_PNORM_INPUT_DIM}-pnormoutputdim=${AM_TRAIN_TDNN_PNORM_OUTPUT_DIM}/${AM_TRAIN_TDNN_ALIGN_NAME}

AM_TRAIN_TDNN_DIR=${AM_TRAIN_DIR}/${AM_TRAIN_CORPUS_NAME}/${AM_TRAIN_LM_NAME}/${AM_TRAIN_TDNN_NAME}
# -----


# TDNN acoustic model discriminative training
AM_TRAIN_DISCR_TDNN_CORPUS_NAME:=
AM_TRAIN_DISCR_TDNN_LM_NAME:=
TRAIN_DISCR_TDNN_DATA_DIR=${DATA_DIR}/${AM_TRAIN_DISCR_TDNN_CORPUS_NAME}/data
TRAIN_DISCR_TDNN_LANG_DIR=${LM_TRAIN_DIR}/${AM_TRAIN_DISCR_TDNN_LM_NAME}/lang

AM_TRAIN_DISCR_TDNN_ALIGN_NAME=${AM_TRAIN_TDNN_NAME}
AM_TRAIN_DISCR_TDNN_ALIGN_DIR=${ALIGN_DIR}/${AM_TRAIN_DISCR_TDNN_CORPUS_NAME}/${AM_TRAIN_DISCR_TDNN_ALIGN_NAME}/${AM_TRAIN_DISCR_TDNN_LM_NAME}

AM_TRAIN_DISCR_TDNN_DENLATS_DIR=${AM_TRAIN_TDNN_DIR}/denlats/${AM_TRAIN_DISCR_TDNN_CORPUS_NAME}/${AM_TRAIN_DISCR_TDNN_LM_NAME}

AM_TRAIN_DISCR_TDNN_DEGS_FRAMES_PER_EG:=150
AM_TRAIN_DISCR_TDNN_DEGS_FRAMES_OVERLAP:=30
AM_TRAIN_DISCR_TDNN_DEGS_DIR=${AM_TRAIN_DISCR_TDNN_DENLATS_DIR}/degs

AM_TRAIN_DISCR_TDNN_CRITERION:=smbr
AM_TRAIN_DISCR_TDNN_DROP_FRAMES:=true
AM_TRAIN_DISCR_TDNN_EPOCHS:=4
AM_TRAIN_DISCR_TDNN_EFFECTIVE_LR:=0.0000125
AM_TRAIN_DISCR_TDNN_LAST_LAYER_FACTOR:=1.0
AM_TRAIN_DISCR_TDNN_MAX_PAR_CHANGE:=1.0
AM_TRAIN_DISCR_TDNN_ONESIL:=true
AM_TRAIN_DISCR_TDNN_MINIBATCH_SIZE:=64

AM_TRAIN_DISCR_TDNN_NAME=criterion=${AM_TRAIN_DISCR_TDNN_CRITERION}-dropframes=${AM_TRAIN_DISCR_TDNN_DROP_FRAMES}-epoch=${AM_TRAIN_DISCR_TDNN_EPOCHS}-seffectivelr=${AM_TRAIN_DISCR_TDNN_EFFECTIVE_LR}-lastlayerfactor=${AM_TRAIN_DISCR_TDNN_LAST_LAYER_FACTOR}-maxparchange=${AM_TRAIN_DISCR_TDNN_MAX_PAR_CHANGE}-onesilclass=${AM_TRAIN_DISCR_TDNN_ONESIL}-minibatch=${AM_TRAIN_DISCR_TDNN_MINIBATCH_SIZE}
AM_TRAIN_DISCR_TDNN_DIR=${AM_TRAIN_DISCR_TDNN_DENLATS_DIR}/${AM_TRAIN_DISCR_TDNN_NAME}
# -----


# evaluation variables
EVAL_CORPUS_NAME:=
EVAL_DATA_DIR=${DATA_DIR}/${EVAL_CORPUS_NAME}/data

EVAL_AM_NAME:=
EVAL_AM_TYPE:=gmm
EVAL_AM_TDNN_DISCR:=false
EVAL_AM_TDNN_DISCR_EPOCH:=4

EVAL_AM_FEAT_TYPE:=mfcc
EVAL_AM_TRAIN_CORPUS_NAME:=
EVAL_LM_TRAIN_CORPUS_NAME:=
EVAL_AM_DIR=${AM_TRAIN_DIR}/${EVAL_AM_TRAIN_CORPUS_NAME}/${EVAL_LM_TRAIN_CORPUS_NAME}/${EVAL_AM_NAME}

EVAL_LM_NAME:=
EVAL_LM_DIR=
EVAL_LANG_DIR=${LM_TRAIN_DIR}/${EVAL_LM_NAME}/lang

EVAL_COMPOSITE_NAME:=
EVAL_COMPOSITE_GRAPH_DIR=${EVALUATION_DIR}/${EVAL_COMPOSITE_NAME}/graph

EVAL_DECODER_BEAM:=13.0
EVAL_DECODER_ACWT:=0.083333
EVAL_DECODER_MAX_ACTIVE:=7000
EVAL_DECODER_CONFIG_NAME:=beam=${EVAL_DECODER_BEAM}-acwt=${EVAL_DECODER_ACWT}-maxactive=${EVAL_DECODER_MAX_ACTIVE}

EVAL_DECODE_DIR=${EVALUATION_DIR}/${EVAL_COMPOSITE_NAME}/${EVAL_CORPUS_NAME}/${EVAL_DECODER_CONFIG_NAME}
EVAL_REPORT_DIR=${EVALUATION_DIR}/${EVAL_COMPOSITE_NAME}/${EVAL_CORPUS_NAME}/${EVAL_DECODER_CONFIG_NAME}/report
# -----
