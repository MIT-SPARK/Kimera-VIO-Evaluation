"""Test logger interface."""
import kimera_eval
import logging


def test_logger(capsys):
    """Add coverage for logging utilities."""
    kimera_eval.configure_logging(level="DEBUG")
    logging.debug("test debug")
    logging.info("test info")
    logging.warning("test warning")
    logging.error("test error")
    logging.critical("test critical")

    out, err = capsys.readouterr()
    out_lines = [x for x in out.split("\n") if len(x)]
    err_lines = [x for x in err.split("\n") if len(x)]
    assert len(out_lines) == 2
    assert "test debug" in out_lines[0]
    assert "test info" in out_lines[1]
    assert len(err_lines) == 3
    assert "test warning" in err_lines[0]
    assert "test error" in err_lines[1]
    assert "test critical" in err_lines[2]
