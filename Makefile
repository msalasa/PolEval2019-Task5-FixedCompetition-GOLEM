include common.mk


# evaluation
eval-report: ${EVAL_REPORT_DIR}/overall.dtl

eval-decode: ${EVAL_DECODE_DIR}/done
# - - - - -


# language model (n-gram) training
lm-ngram-train: ${LM_NGRAM_TRAIN_DIR}/done
# - - - - -


# acoustic model training
am-train-discr-tdnn: ${AM_TRAIN_DISCR_TDNN_DIR}/done

am-align-tdnn: ${AM_TRAIN_DISCR_TDNN_ALIGN_DIR}/done

am-train-tdnn: ${AM_TRAIN_TDNN_DIR}/done

am-align-tri2: ${AM_ALIGN_TRI2_DIR}/done

am-train-tri2: ${AM_TRAIN_TRI2_DIR}/done

am-align-tri1: ${AM_ALIGN_TRI1_DIR}/done

am-train-tri1: ${AM_TRAIN_TRI1_DIR}/done

am-align-mono: ${AM_ALIGN_MONO_DIR}/done

am-train-mono: ${AM_TRAIN_MONO_DIR}/done
# - - - - -


# data preparation (kaldi directory structure)
prepare-data: ${INPUT_DATA_DIR}/done ${INPUT_LANG_DIR}/done

prepare-data-data: ${INPUT_DATA_DIR}/done

prepare-data-lang: ${INPUT_LANG_DIR}/done
# - - - - -


# # #
# EVALUATION
# # #

# Scoring using sclite (part of SCTK toolkit)
${EVAL_REPORT_DIR}/overall.dtl: ${EVAL_REPORT_DIR}/results.tsv ${EVAL_DECODE_DIR}/rtf-info \
		${EVAL_REPORT_DIR}/kaldi-score/done	${EVAL_COMPOSITE_GRAPH_DIR}/../composite-info
	${dir_guard}
	cat $(word 1, $^) | cut -d$$'\t' -f1,2 | perl -ne 'print "$$2 ($$1)\n" if /^(.*?)\t(.*)/' > $(@D)/refs.trn
	cat $(word 1, $^) | cut -d$$'\t' -f1,3 | perl -ne 'print "$$2 ($$1)\n" if /^(.*?)\t(.*)/' > $(@D)/hypos.trn
	./local/sclite -r $(@D)/refs.trn -h $(@D)/hypos.trn -i spu_id -o dtl -n overall -O $(@D)

# Scoring using Kaldi script (grid-search of LM-scale parameter)
${EVAL_REPORT_DIR}/kaldi-score/done: ${EVAL_DATA_DIR}/done ${EVAL_COMPOSITE_GRAPH_DIR}/HCLG.fst ${EVAL_DECODE_DIR}/done
	${dir_guard}
	${KALDI_STEPS}/score_kaldi.sh $(word 1, $(^D)) $(word 2, $(^D)) $(word 3, $(^D))
	touch $@
	
# TSV file with decoding results (references \t hypotheses)
${EVAL_REPORT_DIR}/results.tsv: ${EVAL_DATA_DIR}/done ${EVAL_DECODE_DIR}/done
	${dir_guard}
	paste -d$$'\t'	\
		<(cat $(word 1, $(^D))/text | cut -d' ' -f1) \
		<(cat $(word 1, $(^D))/text | cut -d' ' -f2-) \
		<(cat $(word 2, $(^D))/log/decode.*.log | grep "^[0-9]\{6\}-[0-9]\{6\}.* .*" |\
			grep -v "Log-like" | grep -v "final-state" | sort -V | cut -d' ' -f2- |\
			sed "s/<unk>//g") | sed "s/ \s\+/ /g" > $@

# Average Real-time Factor (utterance duration / decoding time)
${EVAL_DECODE_DIR}/rtf-info: ${EVAL_DECODE_DIR}/done
	${dir_guard}
	grep "real-time" ${EVAL_DECODE_DIR}/log/decode.*.log > $@
	cat $@ | grep -o "sec is.*" | cut -d' ' -f3 | awk '{sum+=$$1; count+=1}END{print "\naverage rtf: "sum/count}' >> $@

# Main decoding target	
${EVAL_DECODE_DIR}/done: \
		${EVAL_DATA_DIR}/done ${EVAL_COMPOSITE_GRAPH_DIR}/HCLG.fst \
		${EVAL_DECODE_DIR}/../am-links-done
	${dir_guard}
	$(eval LINK_MFCC_FEATS_TYPE:=${EVAL_AM_FEAT_TYPE})
	${link_mfcc_data}
	decode_opts="" ;\
	if [[ "${EVAL_AM_TYPE}" == "gmm" ]]; then \
		decode_script=${KALDI_STEPS}/decode.sh ;\
	elif [[ "${EVAL_AM_TYPE}" == "tdnn" ]]; then \
		decode_script=${KALDI_STEPS}/nnet3/decode.sh ;\
	elif [[ "${EVAL_AM_TYPE}" == "chain-tdnn" ]]; then \
		decode_script=${KALDI_STEPS}/nnet3/decode.sh ;\
		decode_opts="--post-decode-acwt 10.0 --frames-per-chunk 150" ;\
	else \
		echo -e "Unknown AM type. Exiting..." ;\
		exit 1 ;\
	fi ;\
	$${decode_script} --skip-scoring true --nj ${NJOBS} --cmd ${KALDI_CMD} $${decode_opts} \
		--beam ${EVAL_DECODER_BEAM} --acwt ${EVAL_DECODER_ACWT} \
		--max-active ${EVAL_DECODER_MAX_ACTIVE} \
		$(word 2, $(^D)) $(@D)/data $(@D)
	touch $@

# Decoding graph composition
${EVAL_COMPOSITE_GRAPH_DIR}/HCLG.fst: \
		${EVAL_LANG_DIR}/done ${EVAL_AM_DIR}/done
	${dir_guard}
	if [[ "${EVAL_AM_TDNN_DISCR}" == "true" ]]; then \
		ln -sf ${ROOT}/$(word 2, $(^D))/epoch${EVAL_AM_TDNN_DISCR_EPOCH}.mdl $(word 2, $(^D))/final.mdl ;\
	fi
	mkgraph_opts="" ;\
	if [[ "${EVAL_AM_TYPE}" == "chain-tdnn" ]]; then \
		mkgraph_opts="--self-loop-scale 1.0" ;\
	fi ;\
	${KALDI_UTILS}/mkgraph.sh $${mkgraph_opts} $(word 1, $(^D)) $(word 2, $(^D)) $(@D)

# File with info about composite
${EVAL_COMPOSITE_GRAPH_DIR}/../composite-info:
	${target_header}
	${dir_guard}
	echo -e "source-composite-name:\t${EVAL_COMPOSITE_NAME}\
			  \nacoustic-model-type:\t${EVAL_AM_TYPE}\
			  \ninput-features-type:\t${EVAL_AM_FEAT_TYPE}\
			  \nacoustic-model-fulldir:\t${EVAL_AM_DIR}\
			  \nlanguage-model-name:\t${EVAL_LM_NAME}" > $@
	touch $@

# Links to AM components (needed by decode.sh)
${EVAL_DECODE_DIR}/../am-links-done: ${EVAL_AM_DIR}/done
	${dir_guard}
	if [[ "${EVAL_AM_TDNN_DISCR}" == "true" ]]; then \
		ln -sf ${ROOT}/$(^D)/epoch${EVAL_AM_TDNN_DISCR_EPOCH}.mdl $(@D)/final.mdl ;\
	else \
		ln -sf ${ROOT}/$(^D)/final.mdl $(@D)/final.mdl ;\
	fi;
	ln -sf ${ROOT}/$(^D)/final.occs $(@D)/
	ln -sf ${ROOT}/$(^D)/tree $(@D)/
	- ln -sf ${ROOT}/$(^D)/final.mat $(@D)/
	- ln -sf ${ROOT}/$(^D)/cmvn_opts $(@D)/
	- ln -sf ${ROOT}/$(^D)/frame_subsampling_factor $(@D)/
	touch $@


# #
# LANGUAGE MODEL TRAINING
# #

${LM_NGRAM_TRAIN_DIR}/done: ${LM_NGRAM_TRAIN_DIR}/${LM_NGRAM_TRAIN_NAME}.arpa
	${dir_gaurd}
	echo -e "LM Training Finished!"
	touch $@

${LM_NGRAM_TRAIN_DIR}/${LM_NGRAM_TRAIN_NAME}.arpa: ${LM_NGRAM_TRAIN_DIR}/corpus.txt
	${dir_guard}	
	estimate-ngram -order ${LM_NGRAM_ORDER} -smoothing ${LM_NGRAM_SMOOTHING} \
		-text $^ -write-lm $@ -write-vocab $(@D)/vocab.txt
	if [[ "${LM_NGRAM_PRUNE}" == "true" ]]; then \
		prune-lm --threshold=${LM_NGRAM_PRUNE_THRESHOLDS} $@ $(@D)/pruned.arpa ;\
		mv $(@D)/pruned.arpa $@ ;\
	fi;
	sed -i "/<\/s>/d" $(@D)/vocab.txt
	echo -e "ngram-order\t${LM_NGRAM_ORDER}" > $(@D)/lm-info
	echo -e "smoothing-alg\t${LM_NGRAM_SMOOTHING}" >> $(@D)/lm-info
	echo -e "source-sen-count\t$$(cat $^ | wc -l)" >> $(@D)/lm-info
	echo -e "source-word-count\t$$(cat $(@D)/vocab.txt | wc -l)" >> $(@D)/lm-info
	echo -e "source-corpora\t${LM_NGRAM_TSV_SOURCE_CORPORA}" >> $(@D)/lm-info
	echo -e "lexicon\t${LEXICON_VERSION}" >> $(@D)/lm-info

${LM_NGRAM_TRAIN_DIR}/corpus.txt: ${LM_NGRAM_TSV_PATHS}
	${dir_guard}
	cat $^ | cut -d$$'\t' -f2 | tr -d '[:punct:][:digit:]' | sed -e "s/ \s\+/ /g" -e "/^$$/d" -e "s/^ //g" -e "s/ $$//g" | sort -u > $@


# #
# ACOUSTIC MODEL TRAINING
# #

# discriminative training of TDNN AM
${AM_TRAIN_DISCR_TDNN_DIR}/done: ${AM_TRAIN_DISCR_TDNN_DEGS_DIR}/done 
	${dir_guard}
	${KALDI_STEPS}/nnet3/train_discriminative.sh --cmd ${KALDI_CMD} --effective-lrate ${AM_TRAIN_DISCR_TDNN_EFFECTIVE_LR} \
		--max-param-change ${AM_TRAIN_DISCR_TDNN_MAX_PAR_CHANGE} --criterion ${AM_TRAIN_DISCR_TDNN_CRITERION} \
		--drop-frames ${AM_TRAIN_DISCR_TDNN_DROP_FRAMES} --num-epochs ${AM_TRAIN_DISCR_TDNN_EPOCHS} \
		--one-silence-class ${AM_TRAIN_DISCR_TDNN_ONESIL} --minibatch-size ${AM_TRAIN_DISCR_TDNN_MINIBATCH_SIZE} \
		--num-jobs-nnet 1 --run-diagnostics false\
		--last-layer-factor ${AM_TRAIN_DISCR_TDNN_LAST_LAYER_FACTOR} \
		$(word 1, $(^D)) $(@D)
	touch $@

# examples generation for discriminative training
${AM_TRAIN_DISCR_TDNN_DEGS_DIR}/done: ${TRAIN_DISCR_TDNN_DATA_DIR}/done ${TRAIN_DISCR_TDNN_LANG_DIR}/done ${AM_TRAIN_TDNN_DIR}/done ${AM_TRAIN_DISCR_TDNN_ALIGN_DIR}/done ${AM_TRAIN_DISCR_TDNN_DENLATS_DIR}/done
	${dir_guard}
	$(eval LINK_MFCC_FEATS_TYPE:=${AM_TRAIN_TDNN_FEAT_TYPE})
	${link_mfcc_data}
	model_left_context=`nnet3-am-info $(word 3, $(^D))/final.mdl | grep "left-context:" | awk '{print $$2}'` ;\
	model_right_context=`nnet3-am-info $(word 3, $(^D))/final.mdl | grep "right-context:" | awk '{print $$2}'` ;\
	echo -e "$${model_left_context}\t$${model_right_context}" ;\
	${KALDI_STEPS}/nnet3/get_egs_discriminative.sh --cmd "${KALDI_CMD} --mem 4GB" --max-jobs-run 2 \
		--cmvn-opts "$$(cat $(word 3, $(^D))/cmvn_opts)" --adjust-priors true \
		--left-context $${model_left_context} --right-context $${model_right_context} \
		--frames-per-eg ${AM_TRAIN_DISCR_TDNN_DEGS_FRAMES_PER_EG} --frames-overlap-per-eg ${AM_TRAIN_DISCR_TDNN_DEGS_FRAMES_OVERLAP} \
		$(@D)/data $(word 2, $(^D)) $(word 4, $(^D)) $(word 5, $(^D)) $(word 3, $(^D))/final.mdl $(@D)
	touch $@
		
# generation of denominator lattices for discriminative training
${AM_TRAIN_DISCR_TDNN_DENLATS_DIR}/done: ${TRAIN_DISCR_TDNN_DATA_DIR}/done ${TRAIN_DISCR_TDNN_LANG_DIR}/done ${AM_TRAIN_TDNN_DIR}/done
	${dir_guard}
	$(eval LINK_MFCC_FEATS_TYPE:=${AM_TRAIN_TDNN_FEAT_TYPE})
	${link_mfcc_data}
	${KALDI_STEPS}/nnet3/make_denlats.sh --cmd ${KALDI_CMD} --determinize true --nj ${NJOBS} --sub-split 20 \
		$(@D)/data $(word 2, $(^D)) $(word 3, $(^D)) $(@D)
	touch $@

# train data alignment (TDNN AM)
${AM_TRAIN_DISCR_TDNN_ALIGN_DIR}/done: ${TRAIN_DISCR_TDNN_DATA_DIR}/done ${TRAIN_LANG_DIR}/done ${AM_TRAIN_TDNN_DIR}/done
	${target_header}
	${dir_guard}
	$(eval LINK_MFCC_FEATS_TYPE:=${AM_TRAIN_TDNN_FEAT_TYPE})
	${link_mfcc_data}
	${KALDI_STEPS}/nnet3/align.sh --cmd ${KALDI_CMD} --use-gpu false --nj ${NJOBS} \
		$(@D)/data $(word 2, $(^D)) $(word 3, $(^D)) $(@D)
	touch $@

# TDNN AM training (Time Delay Neural Network)
${AM_TRAIN_TDNN_DIR}/done:\
		${TRAIN_DATA_DIR}/done ${TRAIN_LANG_DIR}/done \
		${AM_TRAIN_TDNN_ALIGN_DIR}/done
	${dir_guard}
	$(eval LINK_MFCC_FEATS_TYPE:=${AM_TRAIN_TDNN_FEAT_TYPE})
	${link_mfcc_data}
	echo -e "${AM_TRAIN_TDNN_SPLICE_INDEXES}" | tr '_' ' ' > $(@D)/splice_indexes
	echo -e "--norm-means=${AM_TRAIN_TDNN_CMN} --norm-vars=${AM_TRAIN_TDNN_CVN}" > $(@D)/cmvn_opts
	${KALDI_STEPS}/nnet3/train_tdnn.sh --num-epochs ${AM_TRAIN_TDNN_EPOCHS} \
		--num-jobs-initial 1 --num-jobs-final 1 \
		--splice-indexes "$$(cat $(@D)/splice_indexes)" --feat-type raw \
		--cmvn-opts "$$(cat $(@D)/cmvn_opts)" --initial-effective-lrate ${AM_TRAIN_TDNN_INITIAL_LR} \
		--final-effective-lrate ${AM_TRAIN_TDNN_FINAL_LR} --cmd ${KALDI_CMD} \
		--pnorm-input-dim ${AM_TRAIN_TDNN_PNORM_INPUT_DIM} --pnorm-output-dim ${AM_TRAIN_TDNN_PNORM_OUTPUT_DIM} \
		$(@D)/data $(word 2, $(^D)) $(word 3, $(^D)) $(@D)
	touch $@

# train data alignment (GMM-HMM AM with triphones 2nd pass)
${AM_ALIGN_TRI2_DIR}/done: ${TRAIN_DATA_DIR}/done ${TRAIN_LANG_DIR}/done ${AM_TRAIN_TRI2_DIR}/done
	${dir_guard}
	${link_mfcc_data}
	${KALDI_STEPS}/align_si.sh --nj ${NJOBS} --cmd ${KALDI_CMD} \
		$(@D)/data $(word 2, $(^D)) $(word 3, $(^D)) $(@D)
	touch $@

# 2nd pass of GMM-HMM AM with triphones training (for more precise alignments)
${AM_TRAIN_TRI2_DIR}/done: ${TRAIN_DATA_DIR}/done ${TRAIN_LANG_DIR}/done ${AM_ALIGN_TRI1_DIR}/done
	${dir_guard}
	${link_mfcc_data}
	${KALDI_STEPS}/train_deltas.sh --cmd ${KALDI_CMD} \
		${AM_TRAIN_TRI2_LEAVES} ${AM_TRAIN_TRI2_PDFS} \
		$(@D)/data $(word 2, $(^D)) $(word 3, $(^D)) $(@D)
	touch $@

# train data alignment (GMM-HMM AM with triphones 1st pass)
${AM_ALIGN_TRI1_DIR}/done: ${TRAIN_DATA_DIR}/done ${TRAIN_LANG_DIR}/done ${AM_TRAIN_TRI1_DIR}/done
	${dir_guard}
	${link_mfcc_data}
	${KALDI_STEPS}/align_si.sh --nj ${NJOBS} --cmd ${KALDI_CMD} \
		$(@D)/data $(word 2, $(^D)) $(word 3, $(^D)) $(@D)
	touch $@

# 1st pass of GMM-HMM AM with triphones training
${AM_TRAIN_TRI1_DIR}/done: ${TRAIN_DATA_DIR}/done ${TRAIN_LANG_DIR}/done ${AM_ALIGN_MONO_DIR}/done
	${dir_guard}
	${link_mfcc_data}
	${KALDI_STEPS}/train_deltas.sh --cmd ${KALDI_CMD} \
		${AM_TRAIN_TRI1_LEAVES} ${AM_TRAIN_TRI1_PDFS} \
		$(@D)/data $(word 2, $(^D)) $(word 3, $(^D)) $(@D)
	touch $@

# train data alignment (GMM-HMM AM with monophones)
${AM_ALIGN_MONO_DIR}/done: ${TRAIN_DATA_DIR}/done ${TRAIN_LANG_DIR}/done ${AM_TRAIN_MONO_DIR}/done
	${dir_guard}
	${link_mfcc_data}
	${KALDI_STEPS}/align_si.sh --nj ${NJOBS} --cmd ${KALDI_CMD} \
		$(@D)/data $(word 2, $(^D)) $(word 3, $(^D)) $(@D)
	touch $@

# GMM-HMM AM with monohones training
${AM_TRAIN_MONO_DIR}/done: ${TRAIN_DATA_DIR}/done ${TRAIN_LANG_DIR}/done
	${dir_guard}
	${link_mfcc_data}
	${KALDI_STEPS}/train_mono.sh --nj ${NJOBS} --cmd ${KALDI_CMD} \
		$(@D)/data $(word 2, $(^D)) $(@D)
	touch $@


# #
# DATA PREPARATION
# #

# lang directory validation
${INPUT_LANG_DIR}/done: \
		${INPUT_LANG_DIR}/L-done ${INPUT_LANG_DIR}/G-done
	${dir_guard}
	echo -e "Lang preparation finished!"
	${KALDI_UTILS}/validate_lang.pl --skip-determinization-check $(@D)
	touch $@

# G.fst composition
${INPUT_LANG_DIR}/G-done: ${INPUT_LANG_DIR}/L-done
	${dir_guard}
	cat ${INPUT_LM_DIR}/${INPUT_LM_NAME}.arpa |\
		${KALDI_UTILS}/find_arpa_oovs.pl $(@D)/words.txt > $(@D)/lm-oovs.txt
	cat ${INPUT_LM_DIR}/${INPUT_LM_NAME}.arpa |\
		arpa2fst --disambig-symbol=#0 --read-symbol-table=$(@D)/words.txt - $(@D)/G.fst
	- fstisstochastic $(@D)/G.fst
	touch $@

# lang directory preparation
${INPUT_LANG_DIR}/L-done: \
		${INPUT_LANG_DICT_DIR}/lexicon.txt \
		${INPUT_LANG_DICT_DIR}/extra_questions.txt \
		${INPUT_LANG_DICT_DIR}/nonsilence_phones.txt \
		${INPUT_LANG_DICT_DIR}/optional_silence.txt \
		${INPUT_LANG_DICT_DIR}/silence_phones.txt
	${dir_guard}
	${KALDI_UTILS}/prepare_lang.sh $(word 1, $(^D)) "<unk>" $(@D)/local/lang $(@D)
	touch $@

${INPUT_LANG_DICT_DIR}/nonsilence_phones.txt: ${INPUT_LANG_DICT_DIR}/lexicon.txt
	${dir_guard}
	cat $^ | cut -d$$' ' -f2- | tr ' ' '\n' | sort -u |\
		grep -v "sil" | grep -v "nsn" | grep -v "spn" > $@

# final phonetic transcriptions
${INPUT_LANG_DICT_DIR}/lexicon.txt: \
		${INPUT_LANG_DICT_DIR}/lexicon-iv.txt ${INPUT_LANG_DICT_DIR}/lexicon-oov.txt
	${dir_guard}
	(cat $^; echo -e "<unk> spn\nSPEAKERNOISE spn") | sort > $@

# phonetic transcriptions
${INPUT_LANG_DICT_DIR}/lexicon-iv.txt: \
		${INPUT_CORPUS_LEXICON} ${INPUT_LM_DIR}/vocab.txt
	${dir_guard}
	awk 'NR==FNR{words[$$1]; next;} ($$1 in words)' $(word 2, $^) $(word 1, $^) |\
		egrep -v '<.?s>' > $@

${INPUT_LANG_DICT_DIR}/lexicon-oov.txt: \
		${INPUT_LANG_DICT_DIR}/vocab-oov.txt
	${dir_guard}
	touch $@

# out-of-vocabulary word extraction
${INPUT_LANG_DICT_DIR}/vocab-oov.txt: \
		${INPUT_CORPUS_LEXICON} ${INPUT_LM_DIR}/vocab.txt
	${dir_guard}
	awk 'NR==FNR{words[$$1]; next;} !($$1 in words)' $(word 1, $^) $(word 2, $^) |\
		egrep -v '<.?s>' > $@

# vocabulary extraction
${INPUT_LM_DIR}/vocab.txt:
	${dir_guard}
	remake lm-ngram-train \
		LM_NGRAM_TRAIN_NAME=${INPUT_LM_NAME} \
		LM_NGRAM_TSV_SOURCE_CORPORA=${INPUT_CORPUS_NAME} \
		INPUT_LM_NAME=" "

# phones representing silence
# - sil - pure silence
# - nsn - non-speaker noise
# - spn - speaker noise
${INPUT_LANG_DICT_DIR}/silence_phones.txt:
	${target_header}
	${dir_guard}
	echo -e "sil\nspn\nnsn" > $@

${INPUT_LANG_DICT_DIR}/optional_silence.txt:
	${target_header}
	${dir_guard}
	echo -e "sil" > $@

${INPUT_LANG_DICT_DIR}/extra_questions.txt:
	${target_header}
	${dir_guard}
	touch $@

# data directory validation
${INPUT_DATA_DIR}/done: \
		${INPUT_DATA_DIR}/text ${INPUT_DATA_DIR}/wav.scp \
		${INPUT_DATA_DIR}/utt2spk ${INPUT_DATA_DIR}/spk2utt \
		${INPUT_DATA_DIR}/feats-mfcchighres.scp ${INPUT_DATA_DIR}/cmvn-mfcchighres.scp \
		${INPUT_DATA_DIR}/feats-mfcc.scp ${INPUT_DATA_DIR}/cmvn-mfcc.scp
	${dir_guard}
	${KALDI_UTILS}/validate_data_dir.sh --no-feats $(@D)
	echo -e "Data preparation finished!"
	touch $@

# paths to high-resolution cepstral mmean normalization feature archives
${INPUT_DATA_DIR}/cmvn-mfcchighres.scp: ${INPUT_DATA_DIR}/feats-mfcchighres.scp
	${dir_guard}
	cp $^ $(@D)/feats.scp
	${KALDI_STEPS}/compute_cmvn_stats.sh $(@D) $(@D)/make_mfcchighres $(@D)/mfcchighres
	mv $(@D)/cmvn.scp $@
	rm $(@D)/feats.scp

# paths to cepstral mean normalization feature archives
${INPUT_DATA_DIR}/cmvn-mfcc.scp: ${INPUT_DATA_DIR}/feats-mfcc.scp
	${dir_guard}
	cp $^ $(@D)/feats.scp
	${KALDI_STEPS}/compute_cmvn_stats.sh $(@D) $(@D)/make_mfcc $(@D)/mfcc
	mv $(@D)/cmvn.scp $@
	rm $(@D)/feats.scp

# paths to high-resolution MFCC feature archives
${INPUT_DATA_DIR}/feats-mfcchighres.scp: ${INPUT_DATA_DIR}/wav.scp
	${dir_guard}
	${KALDI_STEPS}/make_mfcc.sh --nj ${NJOBS} --cmd ${KALDI_CMD} --mfcc-config ${CONFIG_DIR}/mfcchighres.conf $(@D) $(@D)/make_mfcchighres $(@D)/mfcchighres
	mv $(@D)/feats.scp $@

# paths to MFCC feature archives
${INPUT_DATA_DIR}/feats-mfcc.scp: ${INPUT_DATA_DIR}/wav.scp
	${dir_guard}
	${KALDI_STEPS}/make_mfcc.sh --nj ${NJOBS} --cmd ${KALDI_CMD} --mfcc-config ${CONFIG_DIR}/mfcc.conf $(@D) $(@D)/make_mfcc $(@D)/mfcc
	mv $(@D)/feats.scp $@

# utterance transcriptions
${INPUT_DATA_DIR}/text: ${INPUT_CORPUS_TSV}
	${dir_guard}
	paste -d' ' <(cut -d$$'\t' -f1 $^) <( cut -d$$'\t' -f2 $^ | tr -d '[:punct:]') | sort > $@

# paths to audio files
${INPUT_DATA_DIR}/wav.scp: ${INPUT_CORPUS_TSV}
	${dir_guard}
	cut -d$$'\t' -f1,3 $^ | tr '\t' ' ' | sort > $@

# mapping between utterance and speaker
${INPUT_DATA_DIR}/utt2spk: ${INPUT_CORPUS_TSV}
	${dir_guard}
	cat $^ | perl -ne 'print "$$1 $$1\n" if /^(.*?)\t.*/' > $@

# mapping between speaker and utterance
${INPUT_DATA_DIR}/spk2utt: ${INPUT_DATA_DIR}/utt2spk
	${dir_guard}
	${KALDI_UTILS}/utt2spk_to_spk2utt.pl $^ > $@
