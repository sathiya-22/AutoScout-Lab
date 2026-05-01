import enum

class ExplorationMetric(enum.Enum):
    """
    Defines the various exploration metrics collected by RLGuard's telemetry module.
    These metrics are used to characterize the LLM's exploration behavior
    within the RL environment and detect potential 'exploration hacking'.
    """
    ACTION_DISTRIBUTION_ENTROPY = "action_distribution_entropy"
    """Measures the entropy of the LLM's action distribution, indicating
    the diversity or predictability of chosen actions."""

    NOVELTY_SCORE = "novelty_score"
    """Quantifies how novel or unseen the current state or trajectory is
    compared to previously visited ones."""

    REWARD_VARIANCE = "reward_variance"
    """Indicates the variability in rewards received over a period,
    which can signal stable or erratic reward exploitation/exploration."""

    TRAJECTORY_DIVERSITY = "trajectory_diversity"
    """Assesses the uniqueness and variety of paths taken by the LLM
    within the environment."""

    SUBGOAL_COMPLETION_RATE = "subgoal_completion_rate"
    """Measures the frequency or success rate of achieving predefined
    sub-goals within a task, if applicable."""

    POLICY_ENTROPY = "policy_entropy"
    """Measures the overall entropy of the LLM's policy (e.g., output
    logits distribution), similar to action distribution entropy but
    can apply more broadly to LLM output generation."""

    KL_DIVERGENCE_FROM_BASELINE = "kl_divergence_from_baseline"
    """Measures the Kullback-Leibler divergence of the current policy
    from a learned or reference baseline policy, indicating policy shift."""

    STATE_VISITATION_COUNT = "state_visitation_count"
    """Tracks the frequency of visiting specific states or state clusters,
    useful for identifying overly repetitive or neglected areas."""

    ACTION_REPEAT_RATE = "action_repeat_rate"
    """The rate at which the LLM repeats the same action consecutively
    or frequently, possibly indicating a stuck policy or lack of exploration."""

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"<ExplorationMetric.{self.name}>"

# A convenience tuple of all defined exploration metrics
ALL_EXPLORATION_METRICS = tuple(ExplorationMetric)