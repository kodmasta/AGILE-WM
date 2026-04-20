from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import tensorflow as tf
from gymnasium import spaces
from gymnasium.utils import seeding
from PIL import Image

from agile_wm.paths import first_existing, world_model_leaf_candidates
from vae.vae import CVAE
from rnn.rnn import MDNRNN, rnn_init_state, rnn_next_state, rnn_sim

try:
    from ppaquette_gym_doom.doom_take_cover import DoomTakeCoverEnv
    _HAS_DOOM = True
except Exception:
    DoomTakeCoverEnv = object
    _HAS_DOOM = False


ObsType = Union[np.ndarray, list[np.ndarray]]


def _load_saved_model_weights(model: tf.keras.Model, path: str) -> None:
    saved = tf.saved_model.load(path)
    model.set_weights([var.numpy() for var in saved.variables])


def _to_numpy(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def _resize_rgb(frame: np.ndarray, top: int, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    cropped = frame[:top, :, :]
    img = Image.fromarray(cropped, mode="RGB").resize(size)
    return np.asarray(img, dtype=np.uint8)


def _resolve_model_dir(args: Any, leaf: str) -> Path:
    candidates = world_model_leaf_candidates(args.exp_name, args.env_name, leaf)
    return first_existing(candidates) or candidates[0]


class CarRacingWrapper(gym.Env):
    """
    Gymnasium-native wrapper around CarRacing that returns 64x64 RGB frames.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, full_episode: bool = False, render_mode: Optional[str] = None):
        super().__init__()
        self.full_episode = full_episode
        self.env = gym.make("CarRacing-v3", render_mode=render_mode)

        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
        )

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        return _resize_rgb(frame, top=84, size=(64, 64))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._process_frame(obs), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_frame(obs)

        if self.full_episode:
            terminated = False
            truncated = False

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()


class CarRacingMDNRNN(CarRacingWrapper):
    """
    CarRacing with observations replaced by [z, h] or [z, c, h].
    """

    def __init__(
        self,
        args: Any,
        *,
        load_model: bool = True,
        full_episode: bool = False,
        with_obs: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__(full_episode=full_episode, render_mode=render_mode)
        self.args = args
        self.with_obs = with_obs

        self.vae = CVAE(args)
        self.rnn = MDNRNN(args)

        if load_model:
            _load_saved_model_weights(self.vae, str(_resolve_model_dir(args, "tf_vae")))
            _load_saved_model_weights(self.rnn, str(_resolve_model_dir(args, "tf_rnn")))

        self.rnn_states = rnn_init_state(self.rnn)

        obs_dim = args.z_size + args.rnn_size * args.state_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.N_tiles: Optional[int] = None

    def encode_obs(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32) / 255.0
        x = x.reshape(1, 64, 64, 3)
        z = self.vae.encode(x)[0]
        return _to_numpy(z, dtype=np.float32)

    def _build_state(self, z: np.ndarray) -> np.ndarray:
        h = _to_numpy(tf.squeeze(self.rnn_states[0]), dtype=np.float32)

        if self.rnn.args.state_space == 2:
            c = _to_numpy(tf.squeeze(self.rnn_states[1]), dtype=np.float32)
            return np.concatenate([z, c, h], axis=-1).astype(np.float32)

        return np.concatenate([z, h], axis=-1).astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[ObsType, dict]:
        self.rnn_states = rnn_init_state(self.rnn)
        obs, info = super().reset(seed=seed, options=options)

        z = self.encode_obs(obs)
        z_state = self._build_state(z)

        raw_track = getattr(self.env.unwrapped, "track", None)
        if raw_track is not None:
            self.N_tiles = len(raw_track)

        if self.with_obs:
            return [z_state, obs], info
        return z_state, info

    def step(self, action: np.ndarray) -> tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)

        z = self.encode_obs(obs)
        z_state = self._build_state(z)
        self.rnn_states = rnn_next_state(self.rnn, z, action, self.rnn_states)

        if self.with_obs:
            return [z_state, obs], reward, terminated, truncated, info
        return z_state, reward, terminated, truncated, info

    def close(self) -> None:
        super().close()
        tf.keras.backend.clear_session()
        gc.collect()


class DoomTakeCoverMDNRNN(gym.Env):
    """
    Gymnasium-style adapter around the legacy DoomTakeCover env.
    This keeps the old backend but exposes a modern API.
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        args: Any,
        *,
        render_mode: Optional[str] = None,
        load_model: bool = True,
        with_obs: bool = False,
    ):
        if not _HAS_DOOM:
            raise ImportError(
                "ppaquette_gym_doom is not installed or failed to import."
            )

        super().__init__()
        self.args = args
        self.with_obs = with_obs

        self.legacy_env = DoomTakeCoverEnv()
        self.legacy_env.no_render = render_mode != "human"

        self.vae = CVAE(args)
        self.rnn = MDNRNN(args)

        if load_model:
            _load_saved_model_weights(self.vae, str(_resolve_model_dir(args, "tf_vae")))
            _load_saved_model_weights(self.rnn, str(_resolve_model_dir(args, "tf_rnn")))

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(),
            dtype=np.float32,
        )

        self.obs_size = self.rnn.args.z_size + self.rnn.args.rnn_size * self.rnn.args.state_space
        self.observation_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(self.obs_size,),
            dtype=np.float32,
        )
        self.pixel_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
        )

        self.np_random, _ = seeding.np_random(None)
        self.rnn_states = None
        self.z = None
        self.current_obs = None
        self.restart = 1
        self.frame_count = 0

    def seed(self, seed: Optional[int] = None) -> list[int]:
        if seed is not None:
            tf.random.set_seed(seed)
        self.np_random, actual_seed = seeding.np_random(seed)
        return [actual_seed]

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        return _resize_rgb(frame, top=400, size=(64, 64))

    def _encode(self, img: np.ndarray) -> np.ndarray:
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = x.reshape(1, 64, 64, 3)
        z = self.vae.encode(x)[0]
        return _to_numpy(z, dtype=np.float32)

    def _current_state(self) -> np.ndarray:
        h = _to_numpy(tf.keras.backend.flatten(self.rnn_states[0]), dtype=np.float32)

        if self.rnn.args.state_space == 2:
            c = _to_numpy(tf.keras.backend.flatten(self.rnn_states[1]), dtype=np.float32)
            return np.concatenate([self.z, c, h], axis=0).astype(np.float32)

        return np.concatenate([self.z, h], axis=0).astype(np.float32)

    @staticmethod
    def _to_full_action(action: float) -> list[int]:
        threshold = 0.3333
        full_action = [0] * 43

        if action < -threshold:
            full_action[11] = 1
        elif action > threshold:
            full_action[10] = 1

        return full_action

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[ObsType, dict]:
        if seed is not None:
            self.seed(seed)

        obs = self.legacy_env._reset()
        small_obs = self._process_frame(obs)

        self.current_obs = small_obs
        self.rnn_states = rnn_init_state(self.rnn)
        self.z = self._encode(small_obs)
        self.restart = 1
        self.frame_count = 0

        state = self._current_state()
        info: dict = {}

        if self.with_obs:
            return [state, self.current_obs], info
        return state, info

    def step(self, action: Union[float, np.ndarray]) -> tuple[ObsType, float, bool, bool, dict]:
        self.frame_count += 1

        action_scalar = float(np.asarray(action).reshape(()))
        self.rnn_states = rnn_next_state(self.rnn, self.z, action_scalar, self.rnn_states)

        full_action = self._to_full_action(action_scalar)
        obs, reward, done, info = self.legacy_env._step(full_action)

        small_obs = self._process_frame(obs)
        self.current_obs = small_obs
        self.z = self._encode(small_obs)
        self.restart = int(done)

        terminated = bool(done)
        truncated = False
        state = self._current_state()

        if self.with_obs:
            return [state, self.current_obs], float(reward), terminated, truncated, info
        return state, float(reward), terminated, truncated, info

    def render(self):
        # legacy env handles rendering internally via no_render
        return None

    def close(self) -> None:
        self.legacy_env.close()
        tf.keras.backend.clear_session()
        gc.collect()


class DreamDoomTakeCoverMDNRNN(gym.Env):
    """
    Fully simulated Gymnasium env using the learned MDN-RNN dynamics.
    """

    metadata = {"render_modes": []}

    def __init__(self, args: Any, *, load_model: bool = True):
        super().__init__()
        self.args = args

        initial_z_dir = _resolve_model_dir(args, "tf_initial_z")
        model_path = initial_z_dir.parent
        with open(os.path.join(model_path, "tf_initial_z/initial_z.json"), "r") as f:
            initial_mu, initial_logvar = json.load(f)

        self.initial_mu_logvar = np.array(
            [list(elem) for elem in zip(initial_mu, initial_logvar)],
            dtype=np.float32,
        )

        self.vae = CVAE(args)
        self.rnn = MDNRNN(args)

        if load_model:
            _load_saved_model_weights(self.vae, str(model_path / "tf_vae"))
            _load_saved_model_weights(self.rnn, str(model_path / "tf_rnn"))

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(),
            dtype=np.float32,
        )

        obs_size = self.rnn.args.z_size + self.rnn.args.rnn_size * self.rnn.args.state_space
        self.observation_space = spaces.Box(
            low=-50.0,
            high=50.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.np_random, _ = seeding.np_random(None)
        self.rnn_states = None
        self.o = None
        self._training = True

    def seed(self, seed: Optional[int] = None) -> list[int]:
        if seed is not None:
            tf.random.set_seed(seed)
        self.np_random, actual_seed = seeding.np_random(seed)
        return [actual_seed]

    def _sample_init_z(self) -> np.ndarray:
        idx = self.np_random.integers(0, self.initial_mu_logvar.shape[0])
        init_mu, init_logvar = self.initial_mu_logvar[idx]
        init_mu = init_mu / 10000.0
        init_logvar = init_logvar / 10000.0
        init_z = init_mu + np.exp(init_logvar / 2.0) * self.np_random.standard_normal(init_logvar.shape)
        return init_z.astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)

        self.rnn_states = rnn_init_state(self.rnn)
        z = np.expand_dims(self._sample_init_z(), axis=0)
        self.o = z

        if self.rnn.args.state_space == 2:
            obs = tf.concat([z, self.rnn_states[1], self.rnn_states[0]], axis=-1)
        else:
            obs = tf.concat([z, self.rnn_states[0]], axis=-1)

        return _to_numpy(tf.squeeze(obs), dtype=np.float32), {}

    def step(self, action: Union[float, np.ndarray]) -> tuple[np.ndarray, float, bool, bool, dict]:
        rnn_states_p1, z_tp1, r_tp1, d_tp1 = rnn_sim(
            self.rnn,
            self.o,
            self.rnn_states,
            action,
            training=self._training,
        )

        self.rnn_states = rnn_states_p1
        self.o = z_tp1

        if self.rnn.args.state_space == 2:
            obs = tf.concat([z_tp1, self.rnn_states[1], self.rnn_states[0]], axis=-1)
        else:
            obs = tf.concat([z_tp1, self.rnn_states[0]], axis=-1)

        terminated = bool(np.asarray(_to_numpy(d_tp1)).reshape(()))
        truncated = False
        reward = float(np.asarray(_to_numpy(r_tp1)).reshape(()))

        return _to_numpy(tf.squeeze(obs), dtype=np.float32), reward, terminated, truncated, {}

    def render(self):
        return None

    def close(self) -> None:
        tf.keras.backend.clear_session()
        gc.collect()


def make_env(
    args: Any,
    *,
    dream_env: bool = False,
    seed: int = -1,
    render_mode: Optional[str] = None,
    full_episode: bool = True,
    with_obs: bool = False,
    load_model: bool = True,
):
    """
    Gymnasium-native environment factory.
    """
    if args.env_name == "DoomTakeCover-v0":
        if dream_env:
            env = DreamDoomTakeCoverMDNRNN(
                args=args,
                load_model=load_model,
            )
        else:
            env = DoomTakeCoverMDNRNN(
                args=args,
                render_mode=render_mode,
                load_model=load_model,
                with_obs=with_obs,
            )
    elif "CarRacing" in args.env_name or args.env_name == "CarRacing-v3":
        if dream_env:
            raise ValueError("Training in dreams for CarRacing is not supported.")
        env = CarRacingMDNRNN(
            args=args,
            full_episode=full_episode,
            with_obs=with_obs,
            load_model=load_model,
            render_mode=render_mode,
        )
    else:
        raise ValueError(f"Unsupported env_name: {args.env_name}")

    if seed >= 0:
        env.reset(seed=seed)

    return env
