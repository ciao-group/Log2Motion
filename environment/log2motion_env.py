import copy
import json
import os
from copy import deepcopy
from typing import List, Tuple
from android_env.components.simulators.emulator import emulator_simulator
from myosuite.utils import gym;
from myosuite_custom.base_v0_custom import BaseV0Custom
from myosuite_custom.muscle_effort_model import MuscleEffortModel

register = gym.register
import collections
import numpy as np
import mujoco




class Log2MotionEnv(BaseV0Custom):
    def __init__(self, env_config=None, obsd_model_path=None, seed=42):
        self.contact_counter = 0
        self.last_swipe = False
        self.last_contact = False
        self.muscle_effort_model = None
        self.start_bar_target_name = None
        self.contact_started = False
        self.outside_of_phone = False
        self.initialized_env = False
        self._swipe_goal = False
        self.swipe = False
        # Ensure touch sensor exists in the model
        self.touch_sensor_name = "touch_sensor_1"
        self.screen_touch_sensor_name = "smartphone_geom_touch"
        self.touch_sensor_idx = None
        self.screen_touch_sensor_idx = None
        self.muscle_condition = env_config.get('muscle_condition', "")
        self.hold_threshold = env_config.get('hold_threshold', 1)
        self.hold_information = 0
        self.sigma_signal_dependent = env_config.get('sigma_signal_dependent', 0.103 * 1.0)
        self.sigma_constant = env_config.get('constant_noise_level', 0.185 * 1.0)
        self.rng = np.random.default_rng(seed)

        self.grid = None
        self.grid_path = env_config.get('grid_path', 'environment/button_placement.json')
        self.start_qpos_data = {
            'return_qpos_button': [-0.0387, 0.0163, -0.0164, 0.0387, -0.0079, 0.0633, 0.0287, -0.0285,
                                   -0.0634, 0.0079, 0.8679, 0.1598, -0.8679, 0.1706, 1.4994, 1.0111,
                                   0.37, 0.8161, -0.6476, 0.7408, -0.8455, -1.3892, -0.0092, 0.2795,
                                   -0.0211, -0.0165, 1.6787, 0.2622, 1.6427, 1.593, 1.6259, 0.1831,
                                   1.6091, 1.6128, 1.5901, 0.2327, 1.5884, -0.0018]
        }
        android_config = env_config.get("android_config", None)
        self.screen_resolution = env_config.get('screen_resolution', [2400, 1080])
        self.frame_skip_conf = env_config.get('frame_skip', 1)

        self.previous_action = None
        self.solved_goal = False
        self.mistake_counter = 0
        self.mistake_counter_pos = {}
        self.use_android = env_config.get('use_android', False)
        if self.use_android:
            print("Starting Android emulator...")
            self.and_instance = self._android_setup(android_config['avd_name'], android_config['android_avd_home'],
                                                    android_config['android_sdk_root'], android_config['emulator_path'],
                                                    android_config['adb_path'])
            self.and_instance.launch()
            self.image_data = None
        model_path = env_config['model_path']

        self.MAX_TIME = env_config.get('MAX_TIME',2.5)
        self.max_timestep = env_config.get('max_timestep', int(self.MAX_TIME / (0.002 * self.frame_skip_conf)))
        self.done_th = env_config.get('max_distance_reset', 0.4)

        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        self.sim.data.qpos = deepcopy(self.start_qpos_data['return_qpos_button'])

        self.touch_sensor_idx, self.touch_sensor_dim = self._find_touch_sensor_index(self.touch_sensor_name)

        self.screen_touch_sensor_idx, self.screen_touch_sensor_dim = self._find_touch_sensor_index(
            self.screen_touch_sensor_name)

        # add to contact height button height form the inner center of the phone
        geo_size = copy.deepcopy(self.sim.model.geom_size[self.sim.model.geom_name2id(f"{'touch_area_1'}_geom")])
        smartphone_geo_size = copy.deepcopy(
            self.sim.model.geom_size[self.sim.model.geom_name2id('smartphone_geom')])
        self.contact_height = env_config.get('contact_height', 0.0001) + geo_size[2]
        self.phone_surface_distance = env_config.get('phone_surface_distance', 0.001) + smartphone_geo_size[2]
        self.FINGER_TIP_QVEL_IDS = self._get_index_finger_ids_qvel()
        self.tendon_ids = {}
        for tendon_id in range(self.sim.model.ntendon):
            tendon_name = self.sim.model.id2name(tendon_id, 'tendon')
            if tendon_name:
                self.tendon_ids[tendon_name] = tendon_id
        self.target = {
            'touch_area_name': 'touch_area_1',
            'unique_button_name': 'start_button',
            'tendon_id': self.tendon_ids.get('IFtip_err_1', None),
            'site_id': self.sim.model.site_name2id('touch_area_1'),
            'geom_id': self.sim.model.geom_name2id('touch_area_1_geom'),
            'position': copy.deepcopy(self.sim.data.site_xpos[self.sim.model.site_name2id('touch_area_1')]),
            'geo_pos': copy.deepcopy(self.sim.data.geom_xpos[self.sim.model.geom_name2id("touch_area_1_geom")]),
            'geo_size': geo_size,
            'smartphone_body_id': self.sim.model.body_name2id('smartphone'),
            'smartphone_body_pos': copy.deepcopy(self.sim.data.body_xpos[self.sim.model.body_name2id("smartphone")]),
            'smartphone_pos_geo': copy.deepcopy(
                self.sim.data.geom_xpos[self.sim.model.geom_name2id("smartphone_geom")]),
            # geometry postion of the screen
            'smartphone_screen_geom_id': self.sim.model.geom_name2id('smartphone_geom'),
            'smartphone_screen_geo_size': smartphone_geo_size,
            'contact_height': self.contact_height,
            'target_height_from_body_pos':
                copy.deepcopy(self.sim.data.geom_xpos[self.sim.model.geom_name2id("touch_area_1_geom")])[2] -
                copy.deepcopy(self.sim.data.body_xpos[self.sim.model.body_name2id("smartphone")])[2],
        }

        obs_key = ['qvel', 'act', 'distance_to_target', 'IFtip_pos_norm', 'contact_information',
                   'current_target_position', 'action_type', 'button_size']

        self.fatigue_value = 0.0
        self.fatigue_reset_vec = None
        self.fatigue_reset_random = False
        self._setup(
            obs_keys=obs_key,
            weighted_reward_keys={
                'bonus': 1.0
            },
            muscle_condition="",
            fatigue_reset_vec=self.fatigue_reset_vec,
            fatigue_reset_random=self.fatigue_reset_random,
            frame_skip=self.frame_skip_conf,
        )
        self.muscle_effort_model = MuscleEffortModel(self.sim.model, self.dt)
        self.device_id = 2
        self.START_QVEL = deepcopy(self.sim.init_qvel)
        self.reset()

    # calls BaseV0 setup method, initializes the environment with the given obs_keys, reward_keys and frame_skipping
    def _setup(self, obs_keys, weighted_reward_keys, frame_skip=10,
               muscle_condition="",
               fatigue_reset_vec=None,
               fatigue_reset_random=False,
               **kwargs):
        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, frame_skip=frame_skip,
                       muscle_condition=muscle_condition, fatigue_reset_vec=fatigue_reset_vec,
                       fatigue_reset_random=fatigue_reset_random, **kwargs)

    def _android_setup(self,
                       avd_name: str,
                       android_avd_home: str = '~/.android/avd',
                       android_sdk_root: str = '~/Android/Sdk',
                       emulator_path: str = '~/Android/Sdk/emulator/emulator',
                       adb_path: str = '~/Android/Sdk/platform-tools/adb',
                       run_headless: bool = False):

        """Loads an Android instance.

        Args:
          avd_name: Name of the AVD (Android Virtual Device).
          android_avd_home: Path to the AVD (Android Virtual Device).
          android_sdk_root: Root directory of the SDK.
          emulator_path: Path to the emulator binary.
          adb_path: Path to the ADB (Android Debug Bridge).
        Returns:
          env: An Android instance.
        """

        # Create simulator.
        return emulator_simulator.EmulatorSimulator(
            adb_controller_args=dict(
                adb_path=os.path.expanduser(adb_path),
                adb_server_port=5037,
            ),
            emulator_launcher_args=dict(
                avd_name=avd_name,
                android_avd_home=os.path.expanduser(android_avd_home),
                android_sdk_root=os.path.expanduser(android_sdk_root),
                emulator_path=os.path.expanduser(emulator_path),
                run_headless=run_headless,
                gpu_mode='swiftshader_indirect',
            ),
        )

    def change_the_conaff(self, value=0.0):
        """
        Change the contact affinity of the 'smartphone_geom' in the simulation model.
        :param value: float, optional
            The new contact affinity value to assign to 'smartphone_geom'. Default is 0.0.
        """
        if value != 1.0 or value != 1:
            self.swipe = True
            self.last_swipe = True
        else:
            self.swipe = False

        self.sim.model.geom_conaffinity[self.sim.model.geom_name2id('smartphone_geom')] = value

    def _find_touch_sensor_index(self, touch_sensor_name):
        """
        Finds the index and dimension of a touch sensor in the simulation model.

        Iterates through all sensors and returns the start index and dimension
        of the specified touch sensor.

        :param touch_sensor_name: str
            The name of the touch sensor to find.
        :return: tuple(int, int)
            A tuple containing the start index and dimension of the sensor.
        :raises ValueError:
            If the sensor with the specified name is not found.
        """
        sensor_start_idx = 0
        for i in range(self.sim.model.nsensor):
            if (self.sim.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH and
                    self.sim.model.sensor(i).name == touch_sensor_name):
                return sensor_start_idx, self.sim.model.sensor_dim[i]
            sensor_start_idx += self.sim.model.sensor_dim[i]

        raise ValueError(f"Touch sensor '{touch_sensor_name}' not found in the model!")

    def _is_point_in_range_screen(self, target_pos, target_geo_size, tip_pos):
        """
        Check if a point (x, y, z) is inside the invisible space above an object.

        :param tip_pos: (x, y, z) coordinates of the point
        :return: True if the point is inside the invisible space, False otherwise
        """
        result = 0
        if self.swipe:
            x_min, x_max = target_pos[0] - target_geo_size[0] - 0.003, target_pos[0] + target_geo_size[0] + 0.003
            y_min, y_max = target_pos[1] - target_geo_size[1] - 0.003, target_pos[1] + target_geo_size[1] + 0.003
            z_min, z_max = target_pos[2] + target_geo_size[2], target_pos[2] + target_geo_size[
                2] + 0.0075  # finger tip center is up to 5mm above the screen, finger tip radius is 5mm + 2 mm thickness of the screen
            if (x_min <= tip_pos[0] <= x_max) and (y_min <= tip_pos[1] <= y_max) and (z_min <= tip_pos[2] <= z_max):
                result = 1
        else:
            xy_dist = np.linalg.norm(tip_pos[:2] - self.target['position'].ravel()[:2])
            if xy_dist < self.target['geo_size'].ravel()[
                0]:  # check if its in range of xy of the button cannot touch the screen if yes
                return 0  # preventing errors form the screen touch sensor
            if self.sim.data.sensordata[
                # XY distance check
                self.screen_touch_sensor_idx: self.screen_touch_sensor_idx + self.screen_touch_sensor_dim] > 0.0:
                result = 1

        return result  # Return the result

    def _get_contact_information(self, tip_pos):
        """
        Determines if the fingertip is in contact with the target.

        Returns 1 if contact is detected, otherwise 0. Contact can be detected
        either by geometric distance checks or by the touch sensor data.

        :param tip_pos: array-like
            The (x, y, z) position of the fingertip.
        :return: int
            1 if contact is detected, 0 otherwise.
        """

        def is_contact_valid(tip_pos, target_pos, button_radius=0.005, z_tolerance=(-0.006, 0.0)):
            """
            Check if fingertip is in contact with target.

            tip_pos: np.array([x, y, z]) fingertip position
            target_pos: np.array([x, y, z]) target position
            button_radius: max distance in XY plane
            z_tolerance: tuple (min_offset, max_offset) from target z
            """

            z_offset = tip_pos[2] - (target_pos[2])  # finger tip radius
            radius = 0.006  # finger tip radius
            if self.swipe:
                radius = 0.0075
            if not (z_tolerance[0] <= z_offset <= z_tolerance[1] + radius):
                return 0

            # XY distance check
            xy_dist = np.linalg.norm(tip_pos[:2] - target_pos[:2])
            if xy_dist > button_radius:
                return 0
            return 1

        geo_size = self.target['geo_size'].ravel()
        tip_pos = np.array(tip_pos).ravel()
        current_target_position = self.target['position'].ravel()
        if self.swipe:
            if is_contact_valid(tip_pos, target_pos=current_target_position, button_radius=geo_size[0]) == 1:
                # print("distance activated")
                return 1
            elif self.sim.data.sensordata[
                self.touch_sensor_idx: self.touch_sensor_idx + self.touch_sensor_dim] > 0.0:
                # print("touch sensor activated")
                return 1
            # elif self._is_point_in_range(target_pos=current_target_position, target_geo_size=geo_size, tip_pos=tip_pos):
            #     return 1
            else:
                return 0
        else:
            if is_contact_valid(tip_pos, target_pos=current_target_position, button_radius=geo_size[0]) == 1:
                return 1
            if self.sim.data.sensordata[
                self.touch_sensor_idx: self.touch_sensor_idx + self.touch_sensor_dim].sum() > 0.0:
                return 1
            else:
                return 0

    def get_obs_dict(self, sim):
        def normalize_centered(pos, min_val, max_val):
            center = (min_val + max_val) / 2
            scale = (max_val - min_val) / 2
            return (pos - center) / scale

        # Min and max for each axis
        x_min, x_max = -0.1, 0.1
        y_min, y_max = -0.37, -0.12
        z_min, z_max = 1.0, 1.26
        current_tip_position = sim.data.site_xpos[sim.model.site_name2id('IFtip')].copy().ravel()
        normalized_fingertip = np.array([
            normalize_centered(current_tip_position[0], x_min, x_max),
            normalize_centered(current_tip_position[1], y_min, y_max),
            normalize_centered(current_tip_position[2], z_min, z_max)
        ])

        obs_dict = {}
        obs_dict['muscle_cost'] = np.array(
            [self._get_muscle_effort()])
        obs_dict['time'] = np.array([sim.data.time])
        # joint positions
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        # actuator values
        obs_dict['act'] = sim.data.act[:].copy() if sim.model.na > 0 else np.zeros_like(obs_dict['qpos'])
        obs_dict['IFtip_pos'] = sim.data.site_xpos[sim.model.site_name2id('IFtip')].copy()

        obs_dict['IFtip_pos_norm'] = normalized_fingertip
        if self.muscle_condition == "fatigue":
            obs_dict['action_type'] = np.array(
                [np.mean(self.unwrapped.muscle_fatigue.MF.copy())])  # 0 nothing fatigue 1

        else:
            obs_dict['action_type'] = np.array([self.fatigue_value])
        # finger tip joint velocities
        obs_dict['qvel'] = sim.data.qvel[self.FINGER_TIP_QVEL_IDS].copy() * self.dt
        obs_dict['qvel_full'] = deepcopy(sim.data.qvel) * self.dt
        obs_dict['qvel_real'] = deepcopy(sim.data.qvel)
        obs_dict['qvel_real_if_tip'] = sim.data.qvel[self.FINGER_TIP_QVEL_IDS].copy()
        # obs_dict['button_size'] = np.array([self.contact_height], dtype=np.float32)
        obs_dict['button_size'] = np.array(
            [self.target['geo_size'][0]])
        current_target_position = self.target['position'].ravel()
        normalized_current_target_position = np.array([
            normalize_centered(current_target_position[0], x_min, x_max),
            normalize_centered(current_target_position[1], y_min, y_max),
            normalize_centered(current_target_position[2], z_min, z_max)
        ])

        obs_dict['current_target_position'] = normalized_current_target_position
        obs_dict['current_target_position_global'] = copy.deepcopy(
            self.sim.data.site_xpos[self.sim.model.site_name2id('touch_area_1')])
        current_dist_to_touch = np.linalg.norm(current_tip_position - current_target_position)
        obs_dict['distance_to_target'] = np.array([current_dist_to_touch])

        contact_exists = self._get_contact_information(current_tip_position)
        if contact_exists == 1:
            self.last_contact = True
        if contact_exists == 0 and self._is_point_in_range_screen(target_pos=self.target['smartphone_pos_geo'],
                                                                  target_geo_size=self.target[
                                                                      'smartphone_screen_geo_size'],
                                                                  tip_pos=current_tip_position) == 1:
            # This part check if the fingertip is touching the screen surface if yes then set the flag to 2:
            if not self.swipe:
                if self.last_contact:
                    self.contact_counter = 0
                    contact_exists = 0
                else:
                    contact_exists = 2
            else:
                contact_exists = 2
                self.last_swipe = True
        else:
            if self.contact_counter > 20:
                self.last_contact = False
            self.contact_counter += 1
            if not hasattr(self, 'swipe_entry_count'):
                self.swipe_entry_count = 0
            if self.last_swipe:
                self.last_swipe = False

        obs_dict['contact_information'] = np.array([contact_exists], dtype=np.int32)

        return obs_dict

    def _done_if_outside_of_phone_geometry(self, finger_tip, z_range=0.2):
        """
            Checks if the fingertip (x, y,z) lies outside the bounds of an object defined by its center and dimensions.
            Z center plus 15 cm above the object
            :param x: X-coordinate of the fingertip.
            :param y: Y-coordinate of the fingertip.
            :return: True if the fingertip is outside the object's bounds, otherwise False.
            """
        # phone center
        finger_tip = finger_tip.ravel()
        x, y, z = finger_tip[0], finger_tip[1], finger_tip[2]
        cx, cy, cz = self.target['smartphone_body_pos'][0], self.target['smartphone_body_pos'][1], \
            self.target['smartphone_body_pos'][2]
        width, height = 0.1, 0.16
        # Calculate the bounds of the object (bounding box)
        left = cx - width
        right = cx + width
        bottom = cy - height
        top = cy + height
        front = cz - z_range
        back = cz + z_range

        # Check if the fingertip (point) is outside the bounds in x, y, or z
        if x < left or x > right or y < bottom or y > top or z < front or z > back:
            self.outside_of_phone = True
            return True
        return False


    def get_reward_dict(self, obs_dict):
        contact_exists = obs_dict['contact_information']
        self.solved_goal = False

        if contact_exists == 1:
            if self.hold_information is None:
                self.hold_information = 0
            self.hold_information += 1
        elif contact_exists == 2:

            if self.target['unique_button_name'] == 'return_qpos_button':
                terminate = True

            if self.hold_information == None:
                self.hold_information = 0
                self.mistake_counter += 1
                self.mistake_counter_pos[self.mistake_counter] = obs_dict['IFtip_pos']
                self.wrong_contact_penalty = -1
            elif self.hold_information >= 1:
                self.wrong_contact_penalty = -1
                self.hold_information = 0
                self.mistake_counter += 1
                self.mistake_counter_pos[self.mistake_counter] = obs_dict['IFtip_pos']
            else:
                self.mistake_counter += 1
                self.mistake_counter_pos[self.mistake_counter] = obs_dict['IFtip_pos']

                self.wrong_contact_penalty = -1
                self.hold_information = 0
        else:
            if self.hold_information is None or self.hold_information <= -1:
                self.hold_information = 0
            self.hold_information = 0

        if self.hold_information >= self.hold_threshold:
            self.hold_information = 0
            solved = True
            self.solved_goal = True
        else:
            self.solved_goal = False
            solved = False

        # Termination conditions
        done = (
                obs_dict['time'] > self.MAX_TIME
                or solved
                or self._done_if_outside_of_phone_geometry(obs_dict['IFtip_pos'])
        )

        # Reward dictionary
        rwd_dict = collections.OrderedDict((
            ('bonus', 0),
            ('sparse', 0),
            ('act_reg', 0),
            ('solved', solved),
            ('done', done)
        ))
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])

        return rwd_dict

    def _get_muscle_effort(self):
        """
        Computes and returns the current muscle effort from the model.

        If no muscle effort model is assigned, returns 0.0. Otherwise, uses the
        assigned muscle effort model to compute effort based on the current
        simulation data.

        :return: float
            The computed muscle effort. Returns 0.0 if no muscle effort model is set.
        """
        if self.muscle_effort_model is None:
            return 0.0
        else:
            return self.muscle_effort_model.compute(self.sim.data)

    def update_screen(self, window=True):
        """
        Captures the current screen image from the device, processes it, and updates the environment's texture.

        This method:
            - Captures a screenshot from the device using `get_screenshot`.
            - Rotates the image by 180 degrees to adjust for device orientation.
            - Reshapes and flattens the image to match the texture's expected format.
            - Updates the texture in the simulation with the processed image data.
        """
        if self.use_android:
            self.image_data = self.and_instance.get_screenshot()
            # Rotate the image by 180 degrees
            self.image_data = np.rot90(self.image_data, 2)  # Rotating twice (90 + 90)
            texture_id = self.sim.model.name2id("smartphone_texture", mujoco.mjtObj.mjOBJ_TEXTURE)
            tex_adr = self.sim.model.tex_adr[texture_id]
            self.image_data.reshape((self.screen_resolution[0], self.screen_resolution[1], 3)).astype(np.uint8)

            self.image_data = np.array(self.image_data).flatten()
            # Note: Ensure your `image_data` is a flat numpy array with correct dimensions
            self.sim.model.tex_rgb[tex_adr:tex_adr + self.image_data.size] = self.image_data
            self.sim.forward()
            self.sim.renderer.upd_texture(texture_id, self.sim.data.ptr, window, self.sim.model)

    def call_screen(self, touches: List[Tuple[int, int, bool, int]]):
        """
        Sends touch input events to the Android instance if `use_android` is enabled.

        :param touches: List[Tuple[int, int, bool, int]]
            A list of touch events, where each tuple contains:
            - x (int): x-coordinate of the touch
            - y (int): y-coordinate of the touch
            - is_down (bool): True if the touch is pressing down, False if releasing
            - finger_id (int): Identifier for the touch/finger
        """
        if self.use_android:
            self.and_instance.send_touch(touches)

    def apply_motor_noise(self, action):
        """
        Applies signal-dependent and constant noise to the action.

        Args:
            action (np.array): Action vector, values between -1 and 1.

        Returns:
            np.array: Noisy action.
        """

        # Signal-dependent noise (lognormal example)
        signal_dependent_noise = self.rng.lognormal(
            mean=0.0,
            sigma=self.sigma_signal_dependent,
            size=action.shape
        ) - 1.0  # Center around 0

        # Constant noise (normal)
        constant_noise = self.rng.normal(
            loc=0.0,
            scale=self.sigma_constant,
            size=action.shape
        )

        # Apply noise
        noisy_action = (1 + signal_dependent_noise) * action + constant_noise

        # Clip to [-1, 1]
        noisy_action = np.clip(noisy_action, -1.0, 1.0)

        return noisy_action

    def step(self, a, **kwargs):
        a = self.apply_motor_noise(a.copy())
        if self.previous_action is None:
            self.previous_action = np.zeros(a.shape)
        # Compute relative action
        relative_action = self.previous_action + a

        # Clip actions within valid range (if necessary)
        a = np.clip(relative_action, -1.0, 1.0)

        # Store the action for the next step
        self.previous_action = a.copy()

        if self.normalize_act:
            robotic_act_ind = self.sim.model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
            a[robotic_act_ind] = (
                    np.mean(self.sim.model.actuator_ctrlrange[robotic_act_ind], axis=-1)
                    + a[robotic_act_ind]
                    * (self.sim.model.actuator_ctrlrange[robotic_act_ind, 1]
                       - self.sim.model.actuator_ctrlrange[robotic_act_ind, 0]) / 2.0)

        ind = [
            32, 33, 34, 35, 36, 37, 38,
            39,
            40, 41, 42, 43, 44,
            47,
            49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

        mask_array = np.full(63, -2.0)
        for idx in ind:
            if idx < len(mask_array):
                mask_array[idx] = -0.7

        mask_array[30] = 0.7773  # PT
        mask_array[31] = 0.7773
        mask_array[32] = 0.7773  # FDS5, 8
        mask_array[33] = 0.7773  # FDS4, 9
        mask_array[34] = 0.7773  # FDS3, 10

        mask_array[36] = 0.7773  # FDP5, 12
        mask_array[37] = 0.7773  # FDP4, 13
        mask_array[38] = 0.7773  # FDP3, 14
        mask_array[43] = 0.7773  # EDC2, 10

        mask_array[55] = 0.7773  # LU_RB3
        mask_array[58] = 0.7773  # LU_RB4
        mask_array[61] = 0.7773  # LU_RB5
        mask_array[48] = 0.7773  # FPL
        mask_array[50] = 0.7773  # OP

        for idx in ind:
            a[idx] = mask_array[idx]
        obs, rwd, done, smth, info = super().step(a, **kwargs)
        info['mistake_counter_pos_dict'] = self.mistake_counter_pos
        if done:
            info['is_success'] = self.solved_goal
            smth = self.solved_goal or self._swipe_goal
        return obs, float(rwd), done, smth, info

    def find_closest_button(self, grid, norm_x, norm_y):
        """
        Find the closest button to a given normalized position.

        :param grid: Dictionary with button positions.
        :param norm_x: Normalized X coordinate (0 to 1, top-left origin).
        :param norm_y: Normalized Y coordinate (0 to 1, top-left origin).
        :return: The name of the closest button.
        """
        closest_button = None
        min_distance = float('inf')

        for button_name, data in grid.items():
            button_x, button_y = data['norm_position']

            # Calculate Euclidean distance
            distance = np.sqrt((norm_x - button_x) ** 2 + (norm_y - button_y) ** 2)

            # Update closest button if this one is closer
            if distance < min_distance:
                min_distance = distance
                closest_button = button_name

        return closest_button

    def find_closest_button_xy(self, grid, x, y):
        """
        Find the closest button to a given position.

        :param grid: Dictionary with button positions.
        :param x: X coordinate.
        :param y: Y coordinate.
        :return: The name of the closest button.
        """
        closest_button = None
        min_distance = float('inf')

        for button_name, data in grid.items():
            button_x, button_y = data['position']

            # Calculate Euclidean distance
            distance = np.sqrt((x - button_x) ** 2 + (y - button_y) ** 2)

            # Update closest button if this one is closer
            if distance < min_distance:
                min_distance = distance
                closest_button = button_name

        return closest_button

    def move_button_xyz(self, x, y, z):
        """
        MOve the button to a given position.

        :param grid: Dictionary with button positions.
        :param x: X local phone coordinate.
        :param y: Y local phone coordinate.
        :param z: Z local phone coordinate.
        """
        site1_index = self.target['site_id']
        geom1_index = self.target['geom_id']

        self.sim.model.site_pos[site1_index] = [x, y,
                                                z]
        self.sim.model.geom_pos[geom1_index] = [x, y,
                                                z]

        self.sim.forward()
        self.target['position'] = self.sim.data.site_xpos[
            self.sim.model.site_name2id(self.target['touch_area_name'])]

    def _move_buttons_grid(self, button_target_name=None):
        """
        Moves buttons to positions defined in a grid, optionally targeting a specific button.
        If the grid is not loaded yet, it loads the button positions from a JSON file.
        If `button_target_name` is provided, the target button's position is updated
        in both the site and geometry of the simulation model.
        :param button_target_name: str, optional
            The name of the button to move. If None, only the grid is loaded (if not already).
        """
        def load_grid_from_file(file_path):
            """
            Loads button positions from a JSON file and converts positions to tuples.
            :param file_path: str
                Path to the JSON file containing the grid.
            :return: dict
                A dictionary mapping button names to their position and properties.
            """
            with open(file_path, 'r') as f:
                grid = json.load(f)
            # Convert positions back to tuples if needed
            for v in grid.values():
                v['position'] = tuple(v['position'])
            return grid

        # Check if slide_bar is already created
        if self.grid is None:
            # Load grid from file
            self.grid = load_grid_from_file(self.grid_path)
            return
        if button_target_name is not None:
            selected_name = button_target_name
            x_center_variation, y_center_variation = self.grid[selected_name]['position']
            site1_index = self.target['site_id']
            geom1_index = self.target['geom_id']

            if selected_name == 'start_button':
                self.sim.model.site_pos[site1_index] = [x_center_variation, y_center_variation,
                                                        0.03]
                self.sim.model.geom_pos[geom1_index] = [x_center_variation, y_center_variation,
                                                        0.03]
            else:
                self.sim.model.site_pos[site1_index] = [x_center_variation, y_center_variation,
                                                        self.target['target_height_from_body_pos']]
                self.sim.model.geom_pos[geom1_index] = [x_center_variation, y_center_variation,
                                                        self.target['target_height_from_body_pos']]

            self.sim.forward()
            self.target['position'] = self.sim.data.site_xpos[
                self.sim.model.site_name2id(self.target['touch_area_name'])]
            # Assign the button's unique name to the target
            self.target['unique_button_name'] = selected_name

    def normalize_iftip_position(self, iftip_pos):
        """
        Normalize IFtip_pos (world coordinates) to screen-space [0,1] using smartphone screen bounds.

        Args:
            iftip_pos: Tuple (x, y, z) of fingertip position in world coordinates.

        Returns:
            Tuple (norm_x, norm_y) normalized to [0,1] for screen coordinates.
        """
        x, y, _ = iftip_pos

        # Get screen geometry bounds
        x_center, y_center = self.target['smartphone_body_pos'][0], self.target['smartphone_body_pos'][1]
        x_min = -self.target['smartphone_screen_geo_size'][0] + x_center
        x_max = self.target['smartphone_screen_geo_size'][0] + x_center
        y_min = -self.target['smartphone_screen_geo_size'][1] + y_center
        y_max = self.target['smartphone_screen_geo_size'][1] + y_center

        # Normalize
        norm_x = np.abs((x - x_max) / (x_max - x_min))  # left → right
        norm_y = (y - y_min) / (y_max - y_min)  # bottom → top
        # Clip to valid range [0,1]
        norm_x = np.clip(norm_x, 0, 1)
        norm_y = np.clip(norm_y, 0, 1)

        return norm_x, norm_y

    def touch_action(self, button_target_name=None, show_marker=True):
        """
        Make a touch interaction on the display.

        :param button_target_name: Name of the button to touch.
        :param show_marker: If the marker on the screen is to be shown.
        """
        site_id = self.target['site_id']
        tendon_id = self.target['tendon_id']
        if not show_marker:
            # Make the marker invisible
            self.sim.model.tendon_rgba[tendon_id] = [0.0, 0.0, 0.0, 0.0]
            self.sim.model.site_rgba[site_id] = np.array([0.0, 0.0, 0.0, 0.0])
        self.solved_goal = False
        self._move_buttons_grid(button_target_name)
        if show_marker:
            # Red, visible
            self.sim.model.tendon_rgba[tendon_id] = [1.0, 0.0, 0.0, 0.5]
            self.sim.model.site_rgba[site_id] = np.array([1.0, 0.0, 0.0, 0.5])

    def _get_joint_names(self):
        """
        Return a list of joint names according to the index ID of the joint angles
        """
        return [self.sim.model.joint(jnt_id).name for jnt_id in range(1, self.sim.model.njnt)]

    def _get_index_finger_ids_qvel(self):
        # List of joints involved in swiping motion (adjust if necessary)
        fingertip_joint_names = [
            # "mcp2_flexion", "pm2_flexion",
            "md2_flexion",  # Index finger
            # "mcp3_flexion", "pm3_flexion", "md3_flexion",  # Middle finger
            # "mp_flexion", "ip_flexion"  # Thumb
        ]

        # Get the indices of these joints in qvel
        fingertip_joint_indices = [
            self.sim.model.joint_name2id(joint) for joint in fingertip_joint_names if joint in self._get_joint_names()
        ]
        return fingertip_joint_indices

    def change_button_size(self, button_size=0.007, if_reset=True):
        """
        Adjust the size of the interactive button area in the environment.

        Parameters:
            button_size (float): The new size (radius) to apply to the button area.
                                 Affects both the geom and site size for touch detection.
            if_reset (bool): If it should call the self.reset().
        """
        self.current_button_size = button_size
        site1_index = self.sim.model.site_name2id("touch_area_1")
        geom1_index = self.sim.model.geom_name2id("touch_area_1_geom")

        self.sim.model.geom_size[geom1_index] = [button_size, self.target["geo_size"][1], 0.0]
        self.sim.model.site_size[site1_index] = [button_size, self.target["geo_size"][1], 0.0]
        self.sim.forward()
        if if_reset:
            self.prev_solved_qpos = None
            self.solved_goal = False
            self.hold_information = None
            self.target['geo_size'] = copy.deepcopy(
                self.sim.model.geom_size[self.sim.model.geom_name2id(f"{'touch_area_1'}_geom")]).ravel()
            self.reset()
        else:
            self.target['geo_size'] = copy.deepcopy(
                self.sim.model.geom_size[self.sim.model.geom_name2id(f"{'touch_area_1'}_geom")]).ravel()

    def _reset(self, reset_qpos=None, reset_qvel=None, seed=None, **kwargs):
        init_qpos = copy.deepcopy(self.sim.data.qpos[:].copy())
        init_qvel = copy.deepcopy(self.sim.data.qvel[:].copy())

        if not self.initialized_env or self.outside_of_phone:
            if self.muscle_effort_model is not None:
                self.muscle_effort_model.reset()
            if not self.initialized_env:
                self.initialized_env = True
                # self._move_slide_bar()
                self._move_buttons_grid()
            init_qpos = copy.deepcopy(self.start_qpos_data['return_qpos_button'])
            init_qvel = deepcopy(self.START_QVEL)
            self.outside_of_phone = False
            self.previous_action = None

        if reset_qpos != None:
            if self.muscle_effort_model is not None:
                self.muscle_effort_model.reset()
            reset_qpos = deepcopy(reset_qpos)
            reset_qvel = deepcopy(self.START_QVEL)
            self.previous_action = None
        else:
            reset_qpos = copy.deepcopy(init_qpos)
            reset_qvel = copy.deepcopy(init_qvel)
        if self.last_swipe or self.swipe:
            reset_qvel = deepcopy(self.START_QVEL)
            self.previous_action = None
        self.previous_action = None
        reset_qvel = deepcopy(self.START_QVEL)
        self.init_qpos[:] = reset_qpos
        self.init_qvel[:] = reset_qvel
        self.solved_goal = False
        self._swipe_goal = False
        self.hold_information = None
        self.mistake_counter = 0
        self.mistake_counter_pos = {}

        self.robot.reset(reset_pos=reset_qpos, reset_vel=reset_qvel, seed=42, **kwargs)
        return self.get_obs()
