import time
import numpy as np
import skvideo
from stable_baselines3 import PPO


_SCREEN_RESOLUTION= [2400, 1080]


# Define a threshold for determining if the action is a tap
_SWIPE_DISTANCE_THRESHOLD = 0.04
_HOME_BUTTON = (0.97, 0.5)  # Center - bottom YX
_RETURN_BUTTON = (0.97, 0.2)  # Left - bottom YX
_RECENT_APPS = (0.97, 0.8)  # Right - bottom YX


class Log2Motion:
    def __init__(self, env, policy_paths_dict, show_screen=True, button_size=0.005, collect_frames=False):
        """
        Initialize the policy selector with PPO policies and the action dictionary.

        :param env: The Phone environment
        :param policy_paths_dict: Dict of paths to the trained PPO policy models. Keys 'default','swipe'
        :param show_screen: Boolean to indicate whether to render the environment screen
        :param button_size: Size of the buttons in the environment (default is 0.005)
        :param collect_frames: Boolean to indicate whether to collect frames during interactions
        """
        self.env = env
        self.policy_keys = policy_paths_dict.keys() or ['swipe',
                                                        'default']  # Default keys if not provided
        self.policies = {key: PPO.load(path, env=self.env, device='cpu') for key, path in policy_paths_dict.items()}
        self.button_size = button_size  # Size of the buttons in the environment
        self.frames = {}
        self.episode_counter = 0
        self.show_screen = show_screen
        self.collect_frames = collect_frames
        self.results = {}
        self.show_marker = True
    def start_render(self, show_marker=True):
        """
        Initializes the environment for rendering and sets up the viewer.
        """
        self.show_marker = show_marker
        if self.show_screen:
            # Render the environment
            self.env.mj_render()
        elif self.collect_frames:
            frame = self.env.sim.renderer.render_offscreen(
                width=2560,
                height=1440,
                camera_id="smartphone_view",
            )
        # Reset the environment before starting the replay loop
        self.env.change_button_size(self.button_size)
        self.env.reset(reset_qpos=self.env.start_qpos_data["return_qpos_button"])
        self.results = {}  # Initialize results for the first episode
        self.episode_counter = 0
        self.env.update_screen(self.show_screen)

    def replay(self, action_list, multiple_episode=False, resting_position=True, button_size=0.005):
        """
        Replay a sequence of actions and handle environment interactions.

        :param action_list: List[dict]
            Ordered list of action dictionaries containing 'action_type', 'yx_touch', and 'yx_lift'.
        :param multiple_episode: bool
            Whether to perform multiple episodes, starting from home.
        :param resting_position: bool
            Whether to move to resting position after each action.
        :param button_size: float
            Default button size for tap/swipe actions.
        :return: dict
            Results of the replayed actions.
        """

        def handle_single_button(target_coords, policy_key='default', move_rest=True, button_name=None):
            """Helper to perform button press actions consistently."""
            x, y = target_coords
            self._select_target(x, y, show_marker=self.show_marker)
            policy = self.policies[policy_key]
            if button_name:
                print(f"Press {button_name}, policy {policy_key}")
            self._do_interaction(policy=policy)
            self._perform_touch((x, y))
            if move_rest:
                self._move_to_resting_position_with_time_interrupt(policy=policy, swipe=False)

        # Handle multiple episode start
        if multiple_episode:
            handle_single_button((_HOME_BUTTON[1], _HOME_BUTTON[0]), move_rest=resting_position, button_name="home")

        # Process each action in sequence
        for action_dict in action_list:
            action_type = action_dict['action_type']
            start_target = action_dict.get('yx_touch')
            goal_target = action_dict.get('yx_lift')

            # Normalize action_type to string
            action_type_str = str(action_type)

            if action_type_str in ['ActionType.PRESS_BACK']:
                handle_single_button((_RETURN_BUTTON[1], _RETURN_BUTTON[0]), move_rest=resting_position,
                                     button_name="back")
            elif action_type_str in ['ActionType.PRESS_HOME']:
                handle_single_button((_HOME_BUTTON[1], _HOME_BUTTON[0]), move_rest=resting_position, button_name="home")
            elif action_type_str in ['ActionType.DUAL_POINT']:
                # Tap action
                if self._is_tap_action(start_target, goal_target):
                    handle_single_button((start_target[1], start_target[0]), move_rest=resting_position)
                # Swipe action
                else:
                    # Start target
                    self._select_target(start_target[1], start_target[0], show_marker=self.show_marker)
                    self._do_interaction(policy=self.policies['default'])
                    # Swipe to goal target
                    self.env.change_the_conaff(value=0)
                    self._select_target(goal_target[1], goal_target[0], show_marker=False)
                    self._do_interaction(policy=self.policies['swipe'], swipe=True)
                    self.env.change_the_conaff(value=1)
                    # Return to resting position
                    if resting_position:
                        self._move_to_resting_position_with_time_interrupt(policy=self.policies['default'], swipe=True)
            elif action_type_str in ['ActionType.STATUS_TASK_COMPLETE']:
                policy = self.policies['default']
                self._move_to_resting_position_with_time_interrupt(policy=policy, swipe=False)
                return self.results
            else:
                print(f"NOT SUPPORTED ACTION TYPE: {action_type}")
                continue

        return self.results

    def _perform_touch(self, pos):
        """
        Performs a touch event at a specified position on the screen.

        :param pos: tuple of float
            Normalized (x, y) coordinates of the touch on the screen.
            Values should be in the range [0, 1].
        """
        if self.env.use_android:
            x_start, y_start = pos
            x, y = int(x_start * _SCREEN_RESOLUTION[1]), int(y_start * _SCREEN_RESOLUTION[0])
            # Touch down
            self.env.call_screen(touches=[[x, y, True, 0]])
            # Touch up
            self.env.call_screen(touches=[[x, y, False, 0]])
            time.sleep(0.5)
            self.env.update_screen(self.show_screen)

    def _select_target(self, norm_x, norm_y, show_marker=True):
        """
            Selects and activates the button closest to the given normalized coordinates.

            This method identifies the nearest button on the environment's grid based on
            the provided normalized x and y coordinates, and activates the touch action.

            Parameters:
                norm_x (float): Normalized x-coordinate of the target position.
                norm_y (float): Normalized y-coordinate of the target position.

            Returns:
                None
            """
        button_name = self.env.find_closest_button(self.env.grid, norm_x, norm_y)
        self.env.touch_action(button_target_name=button_name,show_marker=show_marker)


    def _move_to_resting_position_with_time_interrupt(self, policy, random_threshold=0.35, swipe=False):
        """
        Moves the agent to the 'resting_position' button while interrupting the action
        if the simulation time exceeds a specified threshold.

        :param policy: A policy object.
        :param random_threshold: float, optional
            Maximum simulation time (in seconds) allowed for the movement before interruption.
            Default is 0.35.
        :param swipe: bool, optional
            Whether to perform a swipe action. Default is False.
        """
        if swipe:
            self.env.touch_action(button_target_name='resting_position2', show_marker=False)
        else:
            self.env.touch_action(button_target_name='resting_position', show_marker=False)
        observation, _ = self.env.reset()
        interaction_frames = []
        data_list = []
        counter = 0
        # Track simulation time
        start_time = self.env.get_obs_dict(self.env.sim)['time']
        iteration_count = 0  # Counter for iterations
        while (self.env.get_obs_dict(self.env.sim)['time'] - start_time) < random_threshold:
            current_time = self.env.get_obs_dict(self.env.sim)['time']
            elapsed = current_time - start_time

            if self.show_screen:
                self.env.mj_render()
            if self.collect_frames:
                if iteration_count % 3 == 0:
                    self.env.update_screen(self.show_screen)
                    frame = self.env.sim.renderer.render_offscreen(
                        width=2560,
                        height=1440,
                        camera_id="smartphone_view",
                    )
                    interaction_frames.append(frame)
            iteration_count += 1

            chosen_action, _ = policy.predict(observation)
            observation, reward, done, solved, info = self.env.step(chosen_action)
            obs = info['obs_dict']
            obs = {
                'IFtip_pos': obs['IFtip_pos'],
                'time': obs['time'],
                'muscle_cost': obs['muscle_cost'],
                'contact_information': obs['contact_information'],
            }
            data_list.append(obs)

            if elapsed >= 0.25:
                counter += 1
                if counter % 2 == 0:
                    self.env.update_screen(self.show_screen)

        # Once time exceeds threshold, handle cleanup here
        self.env.update_screen(self.show_screen)
        self.episode_counter += 1
        if self.collect_frames:
            frame = self.env.sim.renderer.render_offscreen(
                width=2560,
                height=1440,
                camera_id="smartphone_view",
            )
            interaction_frames.append(frame)
            self.frames[self.episode_counter] = {
                'frames': interaction_frames,
                'time': current_time
            }
        self.results[self.episode_counter] = data_list
        observation, _ = self.env.reset()


    def _do_interaction(self, policy, swipe=False):
        """
        Executes a single interaction loop using the given policy in the environment.
        If `swipe` is True, fingertip positions (IFtip_pos) are used to
        update the external screen as simulated touch input.

        :param policy: A policy object.
        :param swipe: bool, optional
            Whether to simulate swipe actions on the screen using IFtip_pos.
            Default is False.
        """
        self.env.update_screen(window=self.show_screen)
        observation, _ = self.env.reset()

        iteration_count = 0
        interaction_frames = []
        data_list = []
        last_xy = None  # To store the last touch position for siwpe
        # Initialize a buffer for the last 5 IFtip positions
        iftip_buffer = []
        def handle_swipe(obs, last_xy=None, last_touch=False):
            """Update screen touch based on IFtip_pos."""

            iftip_pos = obs['IFtip_pos']
            norm_xy = self.env.normalize_iftip_position(iftip_pos)
            x = int(norm_xy[0] * env_config['screen_resolution'][1])
            y = int(norm_xy[1] * env_config['screen_resolution'][0])

            touches_to_send = []

            if last_xy is not None:
                last_x, last_y = last_xy
                for i in range(1, 2):  # 2 intermediate points
                    alpha = i / 4.0
                    inter_x = int(last_x + alpha * (x - last_x))
                    inter_y = int(last_y + alpha * (y - last_y))
                    touches_to_send.append([inter_x, inter_y, True, 0])
            if last_touch:
                touches_to_send.append([x, y, False, 0])
            else:
                touches_to_send.append([x, y, True, 0])

            for t in touches_to_send:
                self.env.call_screen(touches=[t])
                time.sleep(0.0001)

            return (x, y)

        # Collect frames
        if self.collect_frames:
            self.env.update_screen(self.show_screen)
            frame = self.env.sim.renderer.render_offscreen(
                width=2560, height=1440, camera_id="smartphone_view"
            )
            interaction_frames.append(frame)

        while True:
            if self.show_screen:
                self.env.mj_render()
            # Get action from policy
            chosen_action, _ = policy.predict(observation)
            observation, reward, done, solved, info = self.env.step(chosen_action)
            # Save obs data
            obs = info['obs_dict']
            obs = {
                'IFtip_pos': obs['IFtip_pos'],
                'time': obs['time'],
                'muscle_cost': obs['muscle_cost'],
                'contact_information': obs['contact_information'],
            }
            data_list.append(obs)
            # Collect frames
            if self.collect_frames and swipe:
                iftip_buffer.append(obs['IFtip_pos'])
                if len(iftip_buffer) == 7:
                    # Call handle_swipe for all 5 positions at once
                    for pos in iftip_buffer:
                        last_xy = handle_swipe({'IFtip_pos': pos}, last_xy=last_xy)
                        time.sleep(0.0001)
                # After the first burst, handle_swipe for each new step
                elif len(iftip_buffer) > 7:
                    last_xy = handle_swipe(obs, last_xy=last_xy)
            if self.collect_frames and iteration_count % 3 == 0:
                self.env.update_screen(self.show_screen)
                frame = self.env.sim.renderer.render_offscreen(
                    width=2560, height=1440, camera_id="smartphone_view"
                )
                interaction_frames.append(frame)


            iteration_count += 1

            # Handle episode termination
            if done:
                if solved:
                    current_time = self.env.get_obs_dict(self.env.sim)['time']
                    self.env.update_screen(self.show_screen)
                    self.episode_counter += 1
                    if swipe:
                        handle_swipe(obs, last_xy=last_xy, last_touch=True)
                    if self.collect_frames:
                        frame = self.env.sim.renderer.render_offscreen(
                            width=2560, height=1440, camera_id="smartphone_view"
                        )
                        interaction_frames.append(frame)
                        self.frames[self.episode_counter] = {
                            'frames': interaction_frames,
                            'time': current_time
                        }
                    self.results[self.episode_counter] = data_list
                    observation, _ = self.env.reset()
                    break
                else:
                    interaction_frames = []
                    data_list = []
                    observation, _ = self.env.reset()
                    break

    def save_video_from_episodes(self, output_path="output_video.mp4"):
        """
        Combines recorded episode frames and saves them as a video file.

        :param output_path: str, optional
            Path to save the output video (default is "output_video.mp4").

        :raises TypeError:
            If an episode has an unexpected data type.
        :raises ValueError:
            If total frames or total time is zero, preventing video creation.
        """
        if self.episode_counter == 0:
            print('No recorded episodes!!!!')
            return

        all_frames = []
        total_frames = 0
        total_time = 0.0

        # Sort episodes to preserve order (in case keys are not ordered)
        for key in sorted(self.frames.keys()):
            episode = self.frames[key]
            if isinstance(episode, dict):
                frames_list = episode['frames']
                time = episode['time']
            elif isinstance(episode, list) and len(episode) > 0 and isinstance(episode[0], dict):
                frames_list = episode[0]['frames']
                time = episode[0]['time']
            else:
                raise TypeError(f"Unexpected episode type: {type(episode)}")
            all_frames.extend(frames_list)
            total_frames += len(frames_list)
            total_time += time

        # Avoid division by zero
        if total_time == 0 or total_frames == 0:
            raise ValueError("Total time or frame count is zero; cannot calculate average FPS or create video.")
        # Save combined video
        skvideo.io.vwrite(
            output_path,
            np.asarray(all_frames),
            outputdict={
                "-r": str(30),
                "-pix_fmt": "yuv420p"
            },
        )
        print(f"Video saved to {output_path}")

    def _is_tap_action(self, normalized_start_yx, normalized_end_yx):
        """
        Determines if an action is a tap based on the distance between start and end target positions.

        A tap is considered if the distance between the start and end target is less than or equal to a specified threshold.

        :param normalized_start_yx: The start target position as a normalized tuple (y, x)
        :param normalized_end_yx: The end target position as a normalized tuple (y, x)
        :return: True if the action is a tap (i.e., distance is below the threshold), False otherwise
        """
        normalized_start_yx = np.array([normalized_start_yx[0], normalized_start_yx[1]])
        normalized_end_yx = np.array([normalized_end_yx[0], normalized_end_yx[1]])
        distance = np.linalg.norm(normalized_start_yx - normalized_end_yx)
        return distance <= _SWIPE_DISTANCE_THRESHOLD

