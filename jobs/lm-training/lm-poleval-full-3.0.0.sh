#!/bin/bash

# Settable Makefile variables
MAKE_LEXICON_VERSION=lexicon-4.0.0
MAKE_LM_NGRAM_TRAIN_NAME=lm-poleval-full-3.0.0
MAKE_INPUT_LM_NAME=${MAKE_LM_NGRAM_TRAIN_NAME}
MAKE_LM_NGRAM_TSV_SOURCE_CORPORA="\
poleval-train-0.0.0_\
poleval-dev-0.0.0_\
psc-0.0.2"

MAKE_LM_NGRAM_ORDER=3
MAKE_LM_NGRAM_SMOOTHING=ModKN
MAKE_LM_NGRAM_PRUNE=true
MAKE_LM_NGRAM_PRUNE_THRESHOLDS='1e-07,1e-08'

MAKE_PARAMS=$1

# Variables parsing
MAKE_VARIABLES=$(set | grep MAKE |\
	grep -v "MAKE_PARAMS" |\
	grep -v "MAKE_VARIABLES" |\
	grep -v "MAKE_TARGET" | perl -ne 'print "$1 " if /MAKE_(.*)/')

# Makefile target for this job
MAKE_TARGET=lm-ngram-train

echo -e "Launching job: remake ${MAKE_TARGET} ${MAKE_VARIABLES} ${MAKE_PARAMS}"
remake ${MAKE_TARGET} ${MAKE_VARIABLES} ${MAKE_PARAMS}

MAKE_TARGET=prepare-data-lang

echo -e "Launching job: remake ${MAKE_TARGET} ${MAKE_VARIABLES} ${MAKE_PARAMS}"
remake ${MAKE_TARGET} ${MAKE_VARIABLES} ${MAKE_PARAMS}
