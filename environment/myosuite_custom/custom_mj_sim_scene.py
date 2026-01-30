from typing import Any

from myosuite.physics.mj_sim_scene import DMSimScene

from myosuite_custom.mj_renderer_custom import CustomMJRenderer
# from myosuite_custom.sim_scene_custom import SimSceneCustom


class CustomDMSimScene(DMSimScene):
    def __init__(self, *args, **kwargs):
        # Initialize the base MJRenderer
        super().__init__(*args, **kwargs)

    def _create_renderer(self, sim: Any) -> CustomMJRenderer:
        """Creates a renderer for the given simulation."""
        return CustomMJRenderer(sim)