"""Explicit joint-name and causal time alignment for the lesson's ROS data."""
import numpy as np

JOINTS = ('crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3',
          'crane_plus_joint4', 'crane_plus_joint_hand')


def reorder(names, positions, required=JOINTS):
    if len(names) != len(positions) or len(names) != len(set(names)):
        raise ValueError('Invalid or duplicate joint names')
    values = dict(zip(names, positions))
    if any(name not in values for name in required):
        raise ValueError('Missing joint: refusing silent zero fill')
    result = np.asarray([values[name] for name in required], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite joint state')
    return result


def causal_sample(times_ns, values, target_ns, max_age_ns=None):
    times = np.asarray(times_ns, dtype=np.int64)
    target = np.asarray(target_ns, dtype=np.int64)
    values = np.asarray(values)
    if not len(times) or len(times) != len(values) or np.any(np.diff(times) < 0):
        raise ValueError('Empty, mismatched or unsorted series')
    idx = np.searchsorted(times, target, side='right')-1
    if np.any(idx < 0):
        raise ValueError('No past observation; future leakage forbidden')
    age = target-times[idx]
    if max_age_ns is not None and np.any(age > max_age_ns):
        raise ValueError('Stale observation')
    return values[idx], age


def uniform_times(start_ns, end_ns, fps=20):
    if fps <= 0:
        raise ValueError('Invalid sampling frequency')
    period = 1_000_000_000//fps
    if fps <= 0 or period*fps != 1_000_000_000 or end_ns <= start_ns:
        raise ValueError('Invalid sampling grid')
    return np.arange(start_ns, end_ns, period, dtype=np.int64)


def action_offsets(chunk_size=16, fps=20):
    """Index zero corresponds to the command issued at observation time."""
    return [i/fps for i in range(chunk_size)]


def action_chunk(actions, index, chunk_size=16):
    """End padding is explicitly masked, not added as real episode frames."""
    actions = np.asarray(actions)
    if not 0 <= index < len(actions):
        raise IndexError(index)
    indices = np.arange(index, index+chunk_size)
    return actions[np.minimum(indices, len(actions)-1)], indices >= len(actions)
