#!/bin/bash

MAKE_INPUT_CORPUS_NAME=poleval-eval-0.0.0
MAKE_PARAMS=$1

MAKE_VARIABLES=$(set | grep MAKE | grep -v "MAKE_PARAMS" | grep -v "MAKE_VARIABLES" | grep -v "MAKE_TARGET" | perl -ne 'print "$1 " if /MAKE_(.*)/')

MAKE_TARGET=prepare-data-data

echo -e "Launching job: remake ${MAKE_TARGET} ${MAKE_VARIABLES} ${MAKE_PARAMS}"
remake ${MAKE_TARGET} ${MAKE_VARIABLES} ${MAKE_PARAMS}
