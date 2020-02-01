import pytest
from src.ai import Dqn


def f():
    print("Running test...")
    raise SystemExit(1)


# -------------------------------------------------------------------


def test_my_test():
    with pytest.raises(SystemExit):
        f()


# -------------------------------------------------------------------


def test_dqn():
    brain = Dqn(5, 3, 0.9)
    last_reward = 0.1
    last_signal = [0.0, 0.0, 0.0, -0.05, -0.05]
    try:
        # Should return a tensor instance
        action = brain.update(last_reward, last_signal)
    except TypeError:
        raise AssertionError('The test exploded!')

    print("Brain: %s, Action: %s" % (brain, action))


def main():
    test_dqn()
    print(r'All tests passed. Success! \o/')


main()
