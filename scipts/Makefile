.PHONY: help euroc_evaluation regression_tests metric_semantic

help:
	@evaluation/main_evaluation.py --help

euroc_evaluation:
	@evaluation/main_evaluation.py -r -a -v --save_plots --save_boxplots --save_results --write_website experiments/full_euroc.yaml

uHumans1_evaluation:
	@evaluation/uHumans_evaluation.py -a -v --save_plots --save_boxplots --save_results experiments/uHumans1.yaml

uHumans2_evaluation:
	@evaluation/uHumans_evaluation.py -a -v --save_plots --save_boxplots --save_results experiments/uHumans2.yaml

regression_tests:
	@evaluation/regression_tests.py -r -a --save_results experiments/regression_test.yaml

metric_semantic:
	@evaluation/metric_semantic_evaluation.py ~/Downloads/tesse_multiscene_office1_3d_semantic_v5.ply ~/Downloads/tesse_semantics_2.ply
