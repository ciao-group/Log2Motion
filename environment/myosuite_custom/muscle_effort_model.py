import numpy as np

class MuscleEffortModel:
    """
        Computes effort per timestep based on relative muscle forces and optional muscle weights.

        This muscle model is based on the consumed endurance model,
        as described in: https://dl.acm.org/doi/pdf/10.1145/2556288.2557130

        Attributes:
            _model: MuJoCo model object (sim.model)
            _dt: float, timestep duration
            _weight: float, scaling factor for effort cost
            _muscle_weights: np.ndarray, normalized muscle weights
            _effort_cost: float, scalar effort cost for current timestep
            _consumed_endurance: float, consumed endurance value
            _endurance: float, minimum endurance across muscles
            _strength: np.ndarray, relative muscle forces (0-1) per muscle
        """
    def __init__(self, model, dt: float, muscle_weights: np.ndarray = None, weight: float = 1.0):
        """
        Initialize the MuscleEffortModel.

        Args:
            model: MuJoCo model object (sim.model)
            dt (float): Timestep duration
            muscle_weights (np.ndarray, optional): Array of shape (n_muscles,). If None, weights are normalized by max force.
            weight (float): Scaling factor for effort cost
        """
        self._model = model
        self._dt = dt
        self._weight = weight
        if muscle_weights is None:
            max_forces = self._model.actuator_gainprm[:, 2]  # peak force per muscle
            self._muscle_weights = max_forces / np.max(max_forces)  # normalize
        else:
            self._muscle_weights = muscle_weights
        self.reset()

    def reset(self) -> None:
        """
        Reset internal state variables.

        Sets:
            _effort_cost (float): 0.0
            _consumed_endurance (float): 0.0
            _endurance (float): np.inf
        """
        self._effort_cost = 0.0
        self._consumed_endurance = 0.0
        self._endurance = np.inf

    def compute(self, data) -> float:
        """
        Compute muscle effort cost for the current timestep.

        Args:
            data: MuJoCo data object (sim.data)

        Returns:
            float: Scalar effort cost for this timestep
        """
        self._strength = self._compute_strength(data)
        self._endurance = self._compute_endurance(self._strength)
        self._consumed_endurance = self._compute_consumed_endurance()
        self._effort_cost = self._weight * self._consumed_endurance
        return self._effort_cost

    def _compute_strength(self, data) -> np.ndarray:
        """
        Compute relative muscle forces (0-1) per muscle, apply weights.

        Args:
            data: MuJoCo data object (sim.data)

        Returns:
            np.ndarray: Weighted relative muscle forces per muscle
        """
        applied_forces = data.actuator_force
        max_forces = self._model.actuator_gainprm[:, 2]
        strength = np.abs(applied_forces / max_forces).clip(0, 1)
        if self._muscle_weights is not None:
            strength *= self._muscle_weights
        return strength

    def _compute_endurance(self, strength: np.ndarray) -> float:
        """
        Compute minimum endurance across muscles based on strength.

        Args:
            strength (np.ndarray): Weighted relative muscle forces per muscle

        Returns:
            float: Minimum endurance value across muscles
        """
        endurance = np.full_like(strength, np.inf)
        active = strength > 0.15
        endurance[active] = (1236.5 / ((strength[active] * 100 - 15) ** 0.618)) - 72.5
        return np.min(endurance)

    def _compute_consumed_endurance(self) -> float:
        """
        Compute consumed endurance based on dt and current endurance.

        Returns:
            float: Consumed endurance value for current timestep
        """
        if self._endurance < np.inf:
            return (self._dt / self._endurance) * 100
        return 0.0
