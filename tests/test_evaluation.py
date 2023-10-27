"""Test evaluation code."""
import pytest


@pytest.mark.skip(reason="needs to be updated")
def test_default_functionality():
    """Test parser functionality."""
    ev.run(args)

    test_output_dir = os.path.join(
        os.getcwd(), "evaluation/tests/test_results/V1_01_easy/Euroc/"
    )
    # Check that we have generated a results file.
    results_file = os.path.join(test_output_dir, "results_vio.yaml")
    assert os.path.isfile(results_file)
    # Remove file so that we do not re-test and get a false negative...
    os.remove(results_file)

    results_file = os.path.join(test_output_dir, "results_pgo.yaml")
    assert os.path.isfile(results_file)
    # Remove file so that we do not re-test and get a false negative...
    os.remove(results_file)

    # Check that we have generated boxplots.
    boxplots_file = os.path.join(test_output_dir, "../../datasets_ape_boxplots.pdf")
    assert os.path.isfile(boxplots_file)
    # Remove file so that we do not re-test and get a false negative...
    os.remove(boxplots_file)

    # Check that we have generated APE table.
    ape_table_file = os.path.join(test_output_dir, "../../APE_table.tex")
    assert os.path.isfile(ape_table_file)
    # Remove file so that we do not re-test and get a false negative...
    os.remove(ape_table_file)

    # Check that we have generated plots.
    plot_filename = "plots.pdf"
    plot_filepath = os.path.join(test_output_dir, plot_filename)
    print(
        "Checking plot with filename: %s \n At path: %s"
        % (plot_filename, plot_filepath)
    )
    assert os.path.isfile(plot_filepath)
    # Remove file so that we do not re-test and get a false negative...
    os.remove(plot_filepath)
