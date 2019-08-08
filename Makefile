.PHONY: help euroc_evaluation regression_tests

help:
	@evaluation/main_evaluation.py --help

euroc_evaluation:
	@evaluation/main_evaluation.py -r -a --save_plots --save_boxplots --save_results experiments/full_euroc.yaml

regression_tests:
	@evaluation/regression_tests.py -r -a --save_results experiments/regression_test.yaml
