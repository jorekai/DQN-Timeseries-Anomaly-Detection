def verifyBatchShape(state_batch, expectation):
    try:
        #  check for equivalence of array shapes
        msg = "Shape mismatch for Experience Replay, shape expected: {}, shape received: {}".format(expectation,
                                                                                                    state_batch.shape)
        assert state_batch.shape == expectation, msg
    except AssertionError:
        raise
