"""Apply the explicitly registered sampling seed to each random source."""

ALLOWED_SEEDS = (42, 43)


def require_sampling_seed(seed):
    """Reject implicit, Boolean or unsupported sampling identities."""
    if type(seed) is not int or seed not in ALLOWED_SEEDS:
        raise ValueError("The registered sampling seed must be 42 or 43")
    return seed


def reset_sampling_seed(seed, *, python_random, numpy_random, torch):
    """Reset all registered model random streams before initialization or sampling."""
    seed = require_sampling_seed(seed)
    python_random.seed(seed)
    numpy_random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
