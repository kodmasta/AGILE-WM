import argparse
from pathlib import Path
from types import SimpleNamespace
import re

import numpy as np

from agile_wm.paths import default_controller_checkpoint_path, default_rollouts_dir
from agile_wm.runtime import resolve_repo_path
from controller import Controller, MODE_ZCH
from env import make_env


def load_config(path: str) -> dict:
    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split("#", 1)[0].strip()
            if not key:
                continue
            config[key] = value
    return config


def bool_from_string(value: str) -> bool:
    return value.lower() not in ("0", "false", "no", "off", "none")


def get_next_episode_index(out_dir: Path) -> int:
    pattern = re.compile(r"^episode_(\d{6})\.npz$")
    max_index = -1
    if out_dir.exists():
        for file in out_dir.iterdir():
            if file.is_file():
                match = pattern.match(file.name)
                if match:
                    idx = int(match.group(1))
                    if idx > max_index:
                        max_index = idx
    return max_index + 1


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config_path",
        type=str,
        default="configs/carracing.config",
        help="Path to the carracing config file.",
    )
    known_args, remaining = config_parser.parse_known_args()

    config = {}
    config_path = Path(known_args.config_path)
    if config_path.exists():
        config = load_config(str(config_path))

    parser = argparse.ArgumentParser(
        description="Collect CarRacing rollouts using an encoded latent VAE+RNN state and a controller model."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(config_path),
        help="Path to the carracing config file.",
    )
    parser.add_argument("--env_name", type=str, default=config.get("env_name", "CarRacing"))
    parser.add_argument("--exp_name", type=str, default=config.get("exp_name", "WorldModels"))
    parser.add_argument(
        "--controller_path",
        type=str,
        default=None,
        help="Path to the controller JSON file from og.",
    )
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=int(config.get("max_frames", 10000)))
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base RNG seed. If omitted, each run will sample fresh rollout seeds.",
    )
    parser.add_argument("--out_dir", type=str, default=str(default_rollouts_dir()))
    parser.add_argument(
        "--render_mode",
        type=str,
        default=(None if config.get("render_mode", "None").lower() in ("none", "false", "0") else config.get("render_mode", None)),
        choices=[None, "human", "rgb_array"],
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save every Nth episode to disk.",
    )
    parser.add_argument(
        "--with_obs",
        action="store_true",
        help="Return and save raw environment frames alongside latent state observations.",
    )
    parser.add_argument(
        "--z_size",
        type=int,
        default=int(config.get("z_size", 64)),
        help="Latent z dimensionality for the VAE and controller input.",
    )
    parser.add_argument("--rnn_size", type=int, default=int(config.get("rnn_size", 512)))
    parser.add_argument("--state_space", type=int, default=int(config.get("state_space", 2)))
    parser.add_argument("--a_width", type=int, default=int(config.get("a_width", 3)))
    parser.add_argument("--exp_mode", type=int, default=int(config.get("exp_mode", MODE_ZCH)))
    parser.add_argument("--vae_learning_rate", type=float, default=float(config.get("vae_learning_rate", 1e-3)))
    parser.add_argument("--vae_kl_tolerance", type=float, default=float(config.get("vae_kl_tolerance", 0.5)))
    parser.add_argument("--rnn_learning_rate", type=float, default=float(config.get("rnn_learning_rate", 1e-3)))
    parser.add_argument("--rnn_grad_clip", type=float, default=float(config.get("rnn_grad_clip", 1.0)))
    parser.add_argument("--rnn_num_mixture", type=int, default=int(config.get("rnn_num_mixture", 5)))
    parser.add_argument("--rnn_r_pred", type=int, default=int(config.get("rnn_r_pred", 1)))
    parser.add_argument("--rnn_d_pred", type=int, default=int(config.get("rnn_d_pred", 0)))
    parser.add_argument("--rnn_batch_size", type=int, default=int(config.get("rnn_batch_size", 100)))
    parser.add_argument("--rnn_max_seq_len", type=int, default=int(config.get("rnn_max_seq_len", 500)))
    parser.add_argument("--rnn_d_true_weight", type=float, default=float(config.get("rnn_d_true_weight", 1.0)))
    parser.add_argument(
        "--rnn_temperature",
        type=float,
        default=float(config.get("rnn_temperature", 1.0)),
        help="Temperature used by the MDN-RNN sampling routine.",
    )
    args = parser.parse_args()
    if args.controller_path is None:
        args.controller_path = str(default_controller_checkpoint_path(args.exp_name, args.env_name))
    return args


def build_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        env_name=args.env_name,
        exp_name=args.exp_name,
        render_mode=args.render_mode,
        full_episode=False,
        load_model=True,
        z_size=args.z_size,
        rnn_size=args.rnn_size,
        state_space=args.state_space,
        a_width=args.a_width,
        max_frames=args.max_steps,
        exp_mode=args.exp_mode,
        vae_learning_rate=args.vae_learning_rate,
        vae_kl_tolerance=args.vae_kl_tolerance,
        rnn_learning_rate=args.rnn_learning_rate,
        rnn_grad_clip=args.rnn_grad_clip,
        rnn_num_mixture=args.rnn_num_mixture,
        rnn_r_pred=args.rnn_r_pred,
        rnn_d_pred=args.rnn_d_pred,
        rnn_d_true_weight=args.rnn_d_true_weight,
        rnn_batch_size=args.rnn_batch_size,
        rnn_max_seq_len=args.rnn_max_seq_len,
        rnn_input_seq_width=args.z_size + args.a_width,
        rnn_temperature=args.rnn_temperature,
    )


def collect_rollouts(args: argparse.Namespace) -> None:
    model_args = build_model_args(args)

    controller = Controller(model_args)
    controller_path = resolve_repo_path(args.controller_path)
    if not controller_path.exists():
        raise FileNotFoundError(f"Controller file not found: {controller_path}")
    controller.load_model(str(controller_path))

    env = make_env(
        model_args,
        render_mode=args.render_mode,
        full_episode=True,
        load_model=True,
        with_obs=args.with_obs,
    )

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(args.seed)

    start_idx = get_next_episode_index(out_dir)

    for episode_idx in range(args.num_episodes):
        save_idx = start_idx + episode_idx
        episode_seed = int(rng.integers(0, 2**31 - 1))
        env_obs, _ = env.reset(seed=episode_seed)

        if args.with_obs:
            obs, frame = env_obs
        else:
            obs = env_obs
            frame = None

        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_terminated = []
        episode_truncated = []

        if args.with_obs:
            episode_obs.append(np.asarray(frame, dtype=np.uint8))

        total_reward = 0.0

        for step_idx in range(args.max_steps):
            action = controller.get_action(obs)
            next_env_obs, reward, terminated, truncated, _ = env.step(action)

            if args.with_obs:
                next_obs, next_frame = next_env_obs
                episode_obs.append(np.asarray(next_frame, dtype=np.uint8))
            else:
                next_obs = next_env_obs

            episode_actions.append(np.asarray(action, dtype=np.float32))
            episode_rewards.append(np.float32(reward))
            episode_terminated.append(bool(terminated))
            episode_truncated.append(bool(truncated))

            total_reward += float(reward)
            obs = next_obs

            if terminated or truncated:
                break

        print(
            f"episode={episode_idx} steps={len(episode_rewards)} "
            f"return={total_reward:.2f} seed={episode_seed}"
        )

        if episode_idx % args.save_every == 0:
            episode_path = out_dir / f"episode_{save_idx:06d}.npz"
            np.savez_compressed(
                episode_path,
                obs=np.asarray(episode_obs, dtype=np.uint8 if args.with_obs else np.float32),
                actions=np.asarray(episode_actions, dtype=np.float32),
                rewards=np.asarray(episode_rewards, dtype=np.float32),
                terminated=np.asarray(episode_terminated, dtype=np.bool_),
                truncated=np.asarray(episode_truncated, dtype=np.bool_),
                seed=np.asarray([episode_seed], dtype=np.int64),
            )

    env.close()

if __name__ == "__main__":
    parsed_args = parse_args()
    collect_rollouts(parsed_args)
