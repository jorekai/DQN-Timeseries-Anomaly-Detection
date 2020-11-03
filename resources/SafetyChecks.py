def verifyBatchShape(state_batch, expectation):
    try:
        #  check for equivalence of array shapes
        msg = "Shape mismatch for Experience Replay, shape expected: {}, shape received: {}".format(state_batch.shape,
                                                                                                    expectation)
        assert state_batch.shape == expectation, msg
    except AssertionError:
        raise
