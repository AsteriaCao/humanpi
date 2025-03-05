import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_90"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = f"data/libero/videos/{task_suite_name.split('_')[-1]}"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    # num_tasks_in_suite = task_suite.n_tasks
    num_tasks_in_suite = 1
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_description = "estimate_world_position"

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            tmp_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < 100:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    # img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],                      #(3,)
                                    _quat2axisangle(obs["robot0_eef_quat"]),    #(4,) -> (3,)
                                    obs["robot0_gripper_qpos"],                 #(2,)
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])
                    action = action_plan.popleft()
                    # action = np.array([0.+ t * 0.01, 0. , 0., 0., 0., 0., 0.])
                    # Execute action in environment
                    
                    from robosuite.utils.camera_utils import transform_from_pixels_3d_to_world, get_camera_transform_matrix, project_points_from_world_to_camera, transform_from_pixels_to_world, get_real_depth_map
                    import cv2 

                    point = obs['robot0_eef_pos']
                    camera_name = "robot0_eye_in_hand"
                    # camera_name = "agentview"
                    suffix = f"{camera_name}_forward_correct_inverse_2nd_depthmap_2_inverse"
                    # path = f"/storage/qiguojunLab/caojinjin/codes/openpi/dataloader/data/image_inverse_agentview_1_{t}.png"
                    tmp_img = np.ascontiguousarray(obs[f"{camera_name}_image"][::-1, ::-1])

                    # visualize depth map 
                    
                    depth_map = obs[f"{camera_name}_depth"][::-1, ::-1]
                    real_depth_map = get_real_depth_map(env.sim, depth_map)
                    # normalized_depth_map = cv2.normalize(np.squeeze(real_depth_map), None, 0, 1, cv2.NORM_MINMAX)
                    
                    # visual_depth_map = cv2.normalize(np.squeeze(real_depth_map), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    # path = "/storage/qiguojunLab/caojinjin/codes/openpi/dataloader/data/depth_inverse1.png"
                    # cv2.imwrite(path, visual_depth_map)
                   
                    transform_matrix = get_camera_transform_matrix(env.sim, camera_name, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)
                    tmp_point, point_3d = project_points_from_world_to_camera(point, transform_matrix, LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)

                    pos_y, pos_x = int(tmp_point[0]), int(tmp_point[1])
                    
                    camera_to_world = np.linalg.inv(transform_matrix)
                    # estimated error is large
                    # estimated_obj_pos = transform_from_pixels_to_world(
                    #     pixels=tmp_point,
                    #     depth_map=real_depth_map,
                    #     # depth_map=np.expand_dims(normalized_depth_map, axis=-1),
                    #     camera_to_world_transform=camera_to_world,
                    # )
                    estimated_obj_pos = transform_from_pixels_3d_to_world(
                        pixels_3d=point_3d,
                        # depth_map=real_depth_map, 
                        camera_to_world_transform=camera_to_world
                    )
                    z_err = np.abs(point[2] - estimated_obj_pos[2])
                    print(z_err)
                    
                    cv2.circle(tmp_img, (pos_x, pos_y), 5, (0, 0, 255), -1) 
                    # cv2.imwrite(path, tmp_img)
                    tmp_images.append(tmp_img)
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break
            
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            # suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_inverse_{task_segment}_{suffix}_point.mp4",
                [np.asarray(x) for x in tmp_images],
                fps=10,
            )
            import pdb; pdb.set_trace()
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # 20240225: add camera_depths 
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_depths": True}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
