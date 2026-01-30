import mujoco
from mujoco import mjr_uploadTexture, MjrContext, mjtFontScale
from myosuite.renderer.mj_renderer import MJRenderer

from myosuite_custom.renderer_custom import CustomRenderer


class CustomMJRenderer(MJRenderer):
    def __init__(self, *args, **kwargs):
        # Initialize the base MJRenderer
        super().__init__(*args, **kwargs)
    def upd_texture(self, texture_id, data, window=True, model=None):
        if not self._window and not self._renderer:
            return
        if window:
            self._window.update_texture(texture_id)
            self.refresh_window()
        else:
            # self._renderer.update_texture(texture_id=texture_id)
            # while hasattr(data, "_mjdata"):
            #     data = data._mjdata
            # if type(data).__name__ != "MjData":
            #     raise TypeError(f"Expected mujoco._structs.MjData, got {type(data)}")
            # self._renderer.update_scene(data=data.ptr)
            # ctx = MjrContext(model.ptr, mjtFontScale.mjFONTSCALE_150)
            mjr_uploadTexture(model.ptr, self._renderer.mjr_context, texture_id)
            # self._renderer.update_texture(texture_id=texture_id)

    def setup_renderer(self, model, height, width):
        self._renderer = CustomRenderer(model, height=height, width=width)
        self._scene_option = mujoco.MjvOption()
        self._update_renderer_settings(self._scene_option)