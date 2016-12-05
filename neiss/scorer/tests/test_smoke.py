import pytest
import numpy as np
from scorer.scorer import Scorer

def test_smoke():
    """
    Simple smoke test, designed to see if anything is critically wrong with MVP functionality. Only expectation is that
    model predicts a class given an input.
    :return: None
    """
    sample_feats = np.reshape([0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                            0.0,0.0, 0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                            0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                            0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,6,5.0,0.00258081716229,
                            0.00137894457704,0.00221744814915, 0.00140578204266,0.00103914097679,0.000878734622144,
                            0.00714285714286], newshape=(1,-1))
    s = Scorer()

    assert s.score(sample_feats)[0] in {1,2,4,5,6,8,9} # no 3 or 7 codes in coding scheme

if __name__ == '__main__':
    test_smoke()