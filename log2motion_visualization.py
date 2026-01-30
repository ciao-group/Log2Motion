import sys
from pathlib import Path
from myosuite import gym

from environment.log2motion import Log2Motion
def main():
    # Set up paths
    current_dir = Path("").resolve()
    sys.path.append(str(current_dir / 'environment'))

    # Android emulator configuration
    android_config = {
        'avd_name': 'Pixel6API36',
        'android_avd_home': 'C:/Users/USER_NAME/.android/avd',
        'android_sdk_root': 'C:/Users/USER_NAME/AppData/Local/Android/Sdk',
        'emulator_path': 'C:/Users/USER_NAME/AppData/Local/Android/Sdk/emulator/emulator',
        'adb_path': 'C:/Users/USER_NAME/AppData/Local/Android/Sdk/platform-tools/adb'
    }

    # Environment configuration
    env_config = {
        'model_path': (current_dir / 'environment/scene/log2motion_scene.xml').as_posix(),
        'android_config': android_config,
        'screen_resolution': [2400, 1080],
        "use_android": True,
    }

    # Register the environment
    def register_env(env_id='Log2MotionEnv', entry_point='log2motion_env:Log2MotionEnv', env_config=None):
        gym.register(
            id=env_id,
            entry_point=entry_point,
            kwargs={"env_config": env_config}
        )

    register_env(env_config=env_config)

    # Create environment
    env = gym.make("Log2MotionEnv")


    # Initialize replay environment
    replay_env = Log2Motion(
        env=env.unwrapped,
        policy_paths_dict={
            "swipe": "policies/swipe",
            "default": "policies/normal"
        },
        show_screen=False
    )

    # Define episode actions
    episode_list = [
        {'time': [], 'yx_touch': (-1.0, -1.0), 'yx_lift': (-1.0, -1.0), 'action_type': "ActionType.PRESS_HOME"},
        {'time': [], 'yx_touch': (0.8091, 0.5899), 'yx_lift': (0.8091, 0.5899), 'action_type': "ActionType.DUAL_POINT"},
        {'time': [], 'yx_touch': (0.8216, 0.5105), 'yx_lift': (0.8216, 0.5105), 'action_type': "ActionType.DUAL_POINT"},
        {'time': [], 'yx_touch': (0.9036, 0.1837), 'yx_lift': (0.9036, 0.1837), 'action_type': "ActionType.DUAL_POINT"},
        {'time': [], 'yx_touch': (0.9036, 0.8237), 'yx_lift': (0.9036, 0.8237), 'action_type': "ActionType.DUAL_POINT"},
        {'time': [], 'yx_touch': (0.9036, 0.8237), 'yx_lift': (0.9036, 0.8237), 'action_type': "ActionType.DUAL_POINT"},
        {'time': [], 'yx_touch': (0.0544, 0.9420), 'yx_lift': (0.0544, 0.9420), 'action_type': "ActionType.DUAL_POINT"},
        {'time': [], 'yx_touch': (0.4505, 0.5985), 'yx_lift': (0.4505, 0.5985), 'action_type': "ActionType.DUAL_POINT"},
        {'time': [], 'yx_touch': (-1.0, -1.0), 'yx_lift': (-1.0, -1.0), 'action_type': "ActionType.STATUS_TASK_COMPLETE"},
    ]

    # Start replay
    replay_env.show_screen = True
    replay_env.start_render(show_marker=True)
    replay_env.frames = {}
    replay_env.collect_frames = True
    results = replay_env.replay(
        action_list=episode_list,
        multiple_episode=False,
        resting_position=True
    )



if __name__ == "__main__":
    main()
