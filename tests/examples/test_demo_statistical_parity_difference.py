from notebook_runner import notebook_run
import os


def test_demo_statistical_parity_difference_metrics():
    nb, errors = notebook_run(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'examples', 'demo_statistical_parity_difference.ipynb'))

    if len(errors) > 0:
        for err in errors:
            for tbi in err['traceback']:
                print(tbi)
        raise AssertionError("Errors in notebook 'demo_statistical_parity_difference' testcases")
