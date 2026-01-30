from mujoco import Renderer


class CustomRenderer(Renderer):
    def __init__(self, *args, **kwargs):
        # Initialize the base Renderer
        super().__init__(*args, **kwargs)

    @property
    def mjr_context(self):
        """Returns the MJR context."""
        return self._mjr_context