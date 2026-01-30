from typing import Any, Union
from myosuite.physics.sim_scene import SimScene, SimBackend

from myosuite_custom import custom_mj_sim_scene
from myosuite_custom.mj_renderer_custom import CustomMJRenderer


class SimSceneCustom(SimScene):
    def __init__(self, *args, **kwargs):
        # Initialize the base MJRenderer
        super().__init__(*args, **kwargs)
        self.renderer = CustomMJRenderer(self.sim)

    @staticmethod
    def create(*args, backend: Union[SimBackend, int], **kwargs) -> 'SimScene':
        """Creates a new simulation scene.

        Args:
            *args: Positional arguments to pass to the simulation.
            backend: The simulation backend to use to load the simulation.
            **kwargs: Keyword arguments to pass to the simulation.

        Returns:
            A SimScene object.
        """
        # backend = SimBackend(backend)
        # if backend == SimBackend.MUJOCO_PY:
        #     from myosuite.physics import mjpy_sim_scene  # type: ignore
        #     return mjpy_sim_scene.MjPySimScene(*args, **kwargs)
        # elif backend == SimBackend.MUJOCO:
        # from myosuite.physics import mj_sim_scene  # type: ignore
        return custom_mj_sim_scene.CustomDMSimScene(*args, **kwargs)
        # else:
        #     raise NotImplementedError(backend)


    # Get sim as per the sim_backend
    @staticmethod
    def get_sim(model_handle: Any) -> 'SimSceneCustom':
        # sim_backend = SimBackend.get_sim_backend()
        # if sim_backend == SimBackend.MUJOCO_PY:
        #     return SimScene.create(model_handle=model_handle, backend=SimBackend.MUJOCO_PY)
        # elif sim_backend == SimBackend.MUJOCO:
        return SimSceneCustom.create(model_handle=model_handle, backend=SimBackend.MUJOCO)
        # else:
        #     raise ValueError("Unknown sim_backend: {}. Available choices: MUJOCO_PY, MUJOCO")