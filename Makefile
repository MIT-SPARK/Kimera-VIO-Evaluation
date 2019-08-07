.PHONY: help euroc_evaluation

help:
	@evaluation/main_evaluation.py --help

euroc_evaluation:
	@evaluation/main_evaluation.py -r -a --save_plots --save_boxplots --save_results experiments/full_euroc.yaml
