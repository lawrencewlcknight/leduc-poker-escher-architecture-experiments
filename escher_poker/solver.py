"""ESCHER solver implementation used by the Leduc poker experiments.

This file is intentionally close to the thesis notebook implementation so that
results remain traceable to the original experimental development.
"""

from __future__ import annotations

import collections
import contextlib
from datetime import datetime
import glob
import os
import random
import resource
import time

import numpy as np
import tensorflow as tf

from open_spiel.python import policy
import pyspiel
from open_spiel.python.algorithms import exploitability, expected_game_score

from .constants import (
    AVERAGE_POLICY_TRAIN_SHUFFLE_SIZE,
    REGRET_TRAIN_SHUFFLE_SIZE,
    VALUE_TRAIN_SHUFFLE_SIZE,
)
from .fixed_sampling import fixed_sampling_policy
from .networks import PolicyNetwork, RegretNetwork, ValueNetwork
from .regret_targets import (
    AUTHOR_STATE_VALUE,
    PAPER_POLICY_WEIGHTED_Q,
    VALID_REGRET_TARGET_BASELINES,
    compute_regret_target,
)
from .regret_target_processing import (
    BATCH_RMS,
    BATCH_STANDARDIZE,
    BATCH_STANDARDIZE_CLIP,
    CLIP,
    EMA_STD,
    FIXED_UTILITY_SCALE,
    VALID_REGRET_TARGET_PROCESSING,
)
from .replay import (
    RESERVOIR,
    VALID_REGRET_REPLAY_MODES,
    ReservoirBuffer,
    make_regret_replay_buffer,
)

class ESCHERSolver(policy.Policy):
    def __init__(self,
                 game,
                 policy_network_layers=(256, 128),
                 regret_network_layers=(256, 128),
                 value_network_layers=(256, 128),
                 num_iterations: int = 100,
                 num_traversals: int = 130000,
                 num_val_fn_traversals: int = 100,
                 learning_rate: float = 1e-3,
                 learning_rate_schedule: str = 'constant',
                 learning_rate_end: float = None,
                 learning_rate_decay_rate: float = 0.1,
                 learning_rate_warmup_iterations: int = 0,
                 batch_size_regret: int = 10000,
                 batch_size_value: int = 2024,
                 batch_size_average_policy: int = 10000,
                 memory_capacity: int = int(1e5),
                 regret_replay_mode: str = RESERVOIR,
                 regret_replay_rare_history_quota: int = 64,
                 regret_replay_weight_floor: float = 1e-6,
                 policy_network_train_steps: int = 15000,
                 regret_network_train_steps: int = 5000,
                 value_network_train_steps: int = 4048,
                 check_exploitability_every: int = 20,
                 compute_exploitability: bool = True,
                 reinitialize_regret_networks: bool = True,
                 reinitialize_value_network: bool = True,
                 save_regret_networks: str = None,
                 save_regret_memories: str = None,
                 tfrecord_compression: str = None,
                 append_legal_actions_mask=False,
                 save_average_policy_memories: str = None,
                 save_policy_weights=True,
                 expl: float = 1.0,
                 val_expl: float = 0.01,
                 importance_sampling_threshold: float = 100.,
                 importance_sampling: bool = True,
                 clear_value_buffer: bool = True,
                 val_bootstrap=False,
                 use_balanced_probs: bool = False,
                 balanced_sampling_mix: float = 1.0,
                 track_sampling_coverage: bool = False,
                 val_op_prob=0.,
                 infer_device='cpu',
                 debug_val=False,
                 play_against_random=False,
                 train_device='cpu',
                 experiment_string=None,
                 all_actions=True,
                 random_policy_path=None,
                 verbose: bool = True,
                 use_reach_weighted_avg_policy_loss: bool = False,
                 average_policy_weighting: str = "linear",
                 reuse_regret_traversals_for_value: bool = False,
                 on_policy_joint_regret_updates: bool = False,
                 value_test_traversals: int = 20,
                 bootstrap_value_with_separate_traversal: bool = False,
                 zero_regret_fallback: str = "argmax",
                 policy_network_activation: str = "leakyrelu",
                 regret_network_activation: str = "leakyrelu",
                 value_network_activation: str = "leakyrelu",
                 policy_network_layer_norm: bool = True,
                 regret_network_layer_norm: bool = True,
                 value_network_layer_norm: bool = True,
                 policy_network_residual_mode: str = "same_width",
                 regret_network_residual_mode: str = "same_width",
                 value_network_residual_mode: str = "same_width",
                 policy_network_head_depth: int = 0,
                 regret_network_head_depth: int = 0,
                 policy_network_head_units: int = None,
                 regret_network_head_units: int = None,
                 regret_network_output_mode: str = "direct",
                 regret_target_baseline: str = AUTHOR_STATE_VALUE,
                 regret_target_processing: str = "none",
                 regret_target_clip_value: float = 1.0,
                 regret_target_standardize_epsilon: float = 1e-6,
                 regret_target_fixed_scale: float = 1.0,
                 regret_target_ema_decay: float = 0.99,
                 *args, **kwargs):
        """Initialize the ESCHER algorithm.

        Args:
          game: Open Spiel game.
          policy_network_layers: (list[int]) Layer sizes of average_policy net MLP.
          regret_network_layers: (list[int]) Layer sizes of regret net MLP.
          value_network_layers: (list[int]) Layer sizes of value net MLP.
          num_iterations: Number of iterations.
          num_traversals: Number of traversals per iteration.
          num_val_fn_traversals: Number of history value function traversals per iteration
          learning_rate: Learning rate.
          batch_size_regret: (int) Batch size to sample from regret memories.
          batch_size_average_policy: (int) Batch size to sample from average_policy memories.
          memory_capacity: Number of samples that can be stored in memory.
          regret_replay_mode: Regret replay backend. Supported values are
            ``"reservoir"``, ``"all_samples"``, ``"infoset_stratified"``,
            ``"rare_history_quota"``, and
            ``"counterfactual_reach_weighted"``.
          regret_replay_rare_history_quota: Minimum protected samples per
            infoset in the rare-history-quota backend.
          regret_replay_weight_floor: Minimum positive priority weight in the
            counterfactual-reach-weighted backend.
          policy_network_train_steps: Number of policy network training steps (one
            policy training iteration at the end).
          regret_network_train_steps: Number of regret network training steps
            (per iteration).
          reinitialize_regret_networks: Whether to re-initialize the regret
            network before training on each iteration.
          save_regret_networks: If provided, all regret network itearations
            are saved in the given folder. This can be useful to implement SD-CFR
            https://arxiv.org/abs/1901.07621
          save_average_policy_memories: saves the collected average_policy memories as a
            tfrecords file in the given location. This is not affected by
            memory_capacity. All memories are saved to disk and not kept in memory
          save_regret_memories: If provided, regret memories are written to
            disk-backed TFRecord shards instead of in-memory reservoir buffers.
          use_balanced_probs: Use the fixed leaf-balanced policy for actions
            sampled on behalf of the updating player.
          balanced_sampling_mix: Convex mixture weight between uniform actions
            (0) and the exact leaf-balanced policy (1). This is used only when
            ``use_balanced_probs`` is true.
          track_sampling_coverage: Enumerate the fixed sampling policy and
            collect empirical infoset-coverage diagnostics. Intended for small
            games because exact tree enumeration is required.
          tfrecord_compression: Optional TFRecord compression type for
            disk-backed regret memory, for example ``"GZIP"``.
          regret_target_processing: Optional preprocessing applied only to
            sampled regret targets in the supervised regret-network loss.
            Replay buffers always retain raw regret targets. Supported values
            are ``"none"``, ``"standardize"``, ``"clip"``,
            ``"standardize_clip"``, ``"fixed_utility_scale"``,
            ``"batch_rms"``, and ``"ema_std"``.
          regret_network_output_mode: Regret-network output head structure.
            ``"direct"`` keeps the current action outputs, ``"centered"``
            centres action outputs over legal actions, and ``"dueling"`` adds
            a scalar state-value head to centred legal-action outputs.
          regret_target_baseline: Formula used for the raw instantaneous regret
            target. ``"author_state_value"`` reproduces the public authors'
            code, using ``Q_hat(h, a) - V_hat(h)``.
            ``"paper_policy_weighted_q"`` implements Equation 7 / Algorithm 2,
            using ``Q_hat(h, a) - sum_a pi(a|h) Q_hat(h, a)``.
          regret_target_clip_value: Symmetric clipping threshold for
            regret-target processing modes that include clipping.
          regret_target_standardize_epsilon: Minimum standard deviation used
            when standardising a batch of regret targets.
          regret_target_fixed_scale: Positive game-wide divisor used by
            ``"fixed_utility_scale"`` processing.
          regret_target_ema_decay: Decay applied to the persistent first and
            second target moments used by ``"ema_std"`` processing.
          average_policy_weighting: Weighting scheme for supervised
            average-policy regression samples. ``"linear"`` applies the
            standard CFR iteration weighting; ``"uniform"`` gives each sampled
            average-policy memory equal weight.
          infer_device: device used for TF-operations in the traversal branch.
            Format is anything accepted by tf.device
          train_device: device used for TF-operations in the NN training steps.
            Format is anything accepted by tf.device
        """
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)
        self._game = game
        self._save_policy_weights = save_policy_weights
        self._compute_exploitability = bool(compute_exploitability)
        self._play_against_random = play_against_random
        self._append_legal_actions_mask = append_legal_actions_mask
        self._num_random_games = 2000
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
            # `_traverse_game_tree` does not take into account this option.
            raise ValueError('Simulatenous games are not supported.')
        self._batch_size_regret = batch_size_regret
        self._batch_size_value = batch_size_value
        self._batch_size_average_policy = batch_size_average_policy
        self._regret_replay_mode = str(regret_replay_mode).lower()
        if self._regret_replay_mode not in VALID_REGRET_REPLAY_MODES:
            raise ValueError(
                "regret_replay_mode must be one of "
                f"{sorted(VALID_REGRET_REPLAY_MODES)}, got "
                f"{regret_replay_mode!r}."
            )
        self._regret_replay_rare_history_quota = int(
            regret_replay_rare_history_quota
        )
        if self._regret_replay_rare_history_quota <= 0:
            raise ValueError("regret_replay_rare_history_quota must be positive.")
        self._regret_replay_weight_floor = float(regret_replay_weight_floor)
        if (
            not np.isfinite(self._regret_replay_weight_floor)
            or self._regret_replay_weight_floor <= 0.0
        ):
            raise ValueError("regret_replay_weight_floor must be positive and finite.")
        self._policy_network_train_steps = policy_network_train_steps
        self._regret_network_train_steps = regret_network_train_steps
        self._value_network_train_steps = value_network_train_steps
        self._policy_network_layers = policy_network_layers
        self._regret_network_layers = regret_network_layers
        self._value_network_layers = value_network_layers
        self._policy_network_activation = policy_network_activation
        self._regret_network_activation = regret_network_activation
        self._value_network_activation = value_network_activation
        self._policy_network_layer_norm = bool(policy_network_layer_norm)
        self._regret_network_layer_norm = bool(regret_network_layer_norm)
        self._value_network_layer_norm = bool(value_network_layer_norm)
        self._policy_network_residual_mode = str(policy_network_residual_mode)
        self._regret_network_residual_mode = str(regret_network_residual_mode)
        self._value_network_residual_mode = str(value_network_residual_mode)
        self._policy_network_head_depth = int(policy_network_head_depth)
        self._regret_network_head_depth = int(regret_network_head_depth)
        self._policy_network_head_units = policy_network_head_units
        self._regret_network_head_units = regret_network_head_units
        valid_regret_network_output_modes = {"direct", "centered", "dueling"}
        self._regret_network_output_mode = str(regret_network_output_mode).lower()
        if self._regret_network_output_mode not in valid_regret_network_output_modes:
            raise ValueError(
                "regret_network_output_mode must be one of "
                f"{sorted(valid_regret_network_output_modes)}, got "
                f"{regret_network_output_mode!r}."
            )
        self._regret_target_baseline = str(regret_target_baseline).lower()
        if self._regret_target_baseline not in VALID_REGRET_TARGET_BASELINES:
            raise ValueError(
                "regret_target_baseline must be one of "
                f"{sorted(VALID_REGRET_TARGET_BASELINES)}, got "
                f"{regret_target_baseline!r}."
            )
        if self._regret_target_baseline == PAPER_POLICY_WEIGHTED_Q:
            if not all_actions:
                raise ValueError(
                    "paper_policy_weighted_q requires all_actions=True so every "
                    "legal Q-value is available."
                )
            if importance_sampling:
                raise ValueError(
                    "paper_policy_weighted_q requires importance_sampling=False; "
                    "Equation 7 is the no-importance-sampling ESCHER target."
                )
        self._regret_target_processing = str(regret_target_processing).lower()
        if self._regret_target_processing not in VALID_REGRET_TARGET_PROCESSING:
            raise ValueError(
                "regret_target_processing must be one of "
                f"{sorted(VALID_REGRET_TARGET_PROCESSING)}, got "
                f"{regret_target_processing!r}."
            )
        self._regret_target_clip_value = float(regret_target_clip_value)
        if self._regret_target_clip_value <= 0.0:
            raise ValueError("regret_target_clip_value must be positive.")
        self._regret_target_standardize_epsilon = float(
            regret_target_standardize_epsilon
        )
        if self._regret_target_standardize_epsilon <= 0.0:
            raise ValueError("regret_target_standardize_epsilon must be positive.")
        self._regret_target_fixed_scale = float(regret_target_fixed_scale)
        if (
            not np.isfinite(self._regret_target_fixed_scale)
            or self._regret_target_fixed_scale <= 0.0
        ):
            raise ValueError("regret_target_fixed_scale must be positive and finite.")
        self._regret_target_ema_decay = float(regret_target_ema_decay)
        if not 0.0 <= self._regret_target_ema_decay < 1.0:
            raise ValueError("regret_target_ema_decay must be in [0, 1).")
        self._num_players = game.num_players()
        self._last_raw_regret_target_variance = [
            np.nan for _ in range(self._num_players)
        ]
        self._last_processed_regret_target_variance = [
            np.nan for _ in range(self._num_players)
        ]
        self._last_processed_regret_target_abs_mean = [
            np.nan for _ in range(self._num_players)
        ]
        self._last_regret_target_standardization_mean = [
            np.nan for _ in range(self._num_players)
        ]
        self._last_regret_target_standardization_scale = [
            np.nan for _ in range(self._num_players)
        ]
        self._last_regret_target_clip_fraction = [
            np.nan for _ in range(self._num_players)
        ]
        self._last_regret_target_sign_flip_fraction = [
            np.nan for _ in range(self._num_players)
        ]
        self._last_raw_regret_target_positive_fraction = [
            np.nan for _ in range(self._num_players)
        ]
        self._last_processed_regret_target_positive_fraction = [
            np.nan for _ in range(self._num_players)
        ]
        self._reset_regret_target_consistency_diagnostics()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        hist_state = np.append(self._root_node.information_state_tensor(0),
                                self._root_node.information_state_tensor(1))        

        self._value_embedding_size = len(hist_state)
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._num_val_fn_traversals = num_val_fn_traversals
        self._reinitialize_regret_networks = reinitialize_regret_networks
        self._reinit_value_network = reinitialize_value_network
        self._num_actions = game.num_distinct_actions()
        self._iteration = 1
        self._base_learning_rate = float(learning_rate)
        self._learning_rate_schedule = str(learning_rate_schedule)
        self._learning_rate_end = (
            float(learning_rate_end)
            if learning_rate_end is not None
            else self._base_learning_rate
        )
        self._learning_rate_decay_rate = float(learning_rate_decay_rate)
        self._learning_rate_warmup_iterations = int(learning_rate_warmup_iterations)
        self._learning_rate = self._base_learning_rate
        self._save_regret_networks = save_regret_networks
        self._save_regret_memories = save_regret_memories
        if self._save_regret_memories and self._regret_replay_mode != RESERVOIR:
            raise ValueError(
                "Non-reservoir regret replay modes require in-memory replay; "
                "save_regret_memories must be unset."
            )
        self._tfrecord_compression = tfrecord_compression
        self._regret_memories_tfrecord_dir = None
        self._regret_tfrecordfiles = [None for _ in range(self._num_players)]
        self._regret_memory_counts = [0 for _ in range(self._num_players)]
        self._save_average_policy_memories = save_average_policy_memories
        self._infer_device = infer_device
        self._train_device = train_device
        self._memories_tfrecordpath = None
        self._memories_tfrecordfile = None
        self._average_policy_tfrecord_dir = None
        self._average_policy_shard_index = 0
        self._check_exploitability_every = check_exploitability_every
        self._expl = expl
        self._val_expl = val_expl
        self._importance_sampling = importance_sampling
        self._importance_sampling_threshold = importance_sampling_threshold
        self._clear_value_buffer = clear_value_buffer
        self._nodes_visited = 0
        self._example_info_state = [None, None]
        self._example_hist_state = None
        self._example_legal_actions_mask = [None, None]
        self._squared_errors = []
        self._squared_errors_child = []
        self._balanced_probs = {}
        self._balanced_prob_players = {}
        self._use_balanced_probs = bool(use_balanced_probs)
        self._balanced_sampling_mix = float(balanced_sampling_mix)
        if (
            not np.isfinite(self._balanced_sampling_mix)
            or not 0.0 <= self._balanced_sampling_mix <= 1.0
        ):
            raise ValueError("balanced_sampling_mix must be finite and in [0, 1].")
        self._track_sampling_coverage = bool(track_sampling_coverage)
        self._fixed_sampling_static_diagnostics = {}
        self._sampling_coverage_counts = [
            collections.Counter() for _ in range(self._num_players)
        ]
        self._sampling_reach_stats = [
            {"count": 0, "sum": 0.0, "min": np.inf, "max": 0.0}
            for _ in range(self._num_players)
        ]
        self._val_op_prob = val_op_prob
        self._val_bootstrap = val_bootstrap
        self._debug_val = debug_val
        self._experiment_string = experiment_string
        self._all_actions = all_actions
        self._random_policy_path = random_policy_path
        self._use_reach_weighted_avg_policy_loss = bool(
            use_reach_weighted_avg_policy_loss
        )
        valid_average_policy_weighting = {"linear", "uniform"}
        self._average_policy_weighting = str(average_policy_weighting).lower()
        if self._average_policy_weighting not in valid_average_policy_weighting:
            raise ValueError(
                "average_policy_weighting must be one of "
                f"{sorted(valid_average_policy_weighting)}, got "
                f"{average_policy_weighting!r}."
            )
        self._reuse_regret_traversals_for_value = bool(
            reuse_regret_traversals_for_value
        )
        self._on_policy_joint_regret_updates = bool(on_policy_joint_regret_updates)
        self._value_test_traversals = int(value_test_traversals)
        self._bootstrap_value_with_separate_traversal = bool(
            bootstrap_value_with_separate_traversal
        )
        if zero_regret_fallback not in {"argmax", "uniform"}:
            raise ValueError(
                "zero_regret_fallback must be either 'argmax' or 'uniform'."
            )
        self._use_uniform_zero_regret_fallback = zero_regret_fallback == "uniform"
        self._avg_policy_obs_count = 0

        # Initialize file save locations
        if self._save_regret_networks:
            os.makedirs(self._save_regret_networks, exist_ok=True)

        if self._save_average_policy_memories:
            if os.path.isdir(self._save_average_policy_memories):
                self._average_policy_tfrecord_dir = self._save_average_policy_memories
            else:
                self._average_policy_tfrecord_dir = os.path.split(
                    self._save_average_policy_memories
                )[0]
                os.makedirs(self._average_policy_tfrecord_dir, exist_ok=True)
            pattern = os.path.join(
                self._average_policy_tfrecord_dir,
                "average_policy_memories_*.tfrecord*",
            )
            for old_file in glob.glob(pattern):
                try:
                    os.remove(old_file)
                except OSError:
                    pass
            self._memories_tfrecordpath = self._average_policy_tfrecord_path()

        if self._save_regret_memories:
            self._regret_memories_tfrecord_dir = str(self._save_regret_memories)
            os.makedirs(self._regret_memories_tfrecord_dir, exist_ok=True)
            pattern = os.path.join(
                self._regret_memories_tfrecord_dir,
                "regret_memories_p*_iter*.tfrecord*",
            )
            for old_file in glob.glob(pattern):
                try:
                    os.remove(old_file)
                except OSError:
                    pass

        # Initialize policy network, loss, optimizer
        self._reinitialize_policy_network()

        # Persistent target moments are solver state, not network state. They
        # therefore survive Experiment 28's per-iteration network reinitialisation.
        with tf.device(self._train_device):
            self._regret_target_ema_mean = [
                tf.Variable(
                    0.0,
                    trainable=False,
                    dtype=tf.float32,
                    name=f"regret_target_ema_mean_player_{player}",
                )
                for player in range(self._num_players)
            ]
            self._regret_target_ema_second_moment = [
                tf.Variable(
                    0.0,
                    trainable=False,
                    dtype=tf.float32,
                    name=f"regret_target_ema_second_moment_player_{player}",
                )
                for player in range(self._num_players)
            ]
            self._regret_target_ema_initialized = [
                tf.Variable(
                    False,
                    trainable=False,
                    dtype=tf.bool,
                    name=f"regret_target_ema_initialized_player_{player}",
                )
                for player in range(self._num_players)
            ]

        # Initialize regret networks, losses, optimizers
        self._regret_networks = []
        self._regret_networks_train = []
        self._loss_regrets = []
        self._optimizer_regrets = []
        self._regret_train_step = []
        for player in range(self._num_players):
            with tf.device(self._infer_device):
                regret_network = RegretNetwork(
                    self._embedding_size, self._regret_network_layers,
                    self._num_actions,
                    activation=self._regret_network_activation,
                    use_layer_norm=self._regret_network_layer_norm,
                    residual_mode=self._regret_network_residual_mode,
                    head_depth=self._regret_network_head_depth,
                    head_units=self._regret_network_head_units,
                    output_mode=self._regret_network_output_mode)
                self._build_network_once(regret_network, self._embedding_size)
                self._regret_networks.append(regret_network)
            with tf.device(self._train_device):
                regret_network_train = RegretNetwork(
                    self._embedding_size,
                    self._regret_network_layers, self._num_actions,
                    activation=self._regret_network_activation,
                    use_layer_norm=self._regret_network_layer_norm,
                    residual_mode=self._regret_network_residual_mode,
                    head_depth=self._regret_network_head_depth,
                    head_units=self._regret_network_head_units,
                    output_mode=self._regret_network_output_mode)
                self._build_network_once(regret_network_train, self._embedding_size)
                self._regret_networks_train.append(regret_network_train)
                self._loss_regrets.append(tf.keras.losses.MeanSquaredError())
                self._optimizer_regrets.append(
                    tf.keras.optimizers.Adam(learning_rate=self._learning_rate))
                self._regret_train_step.append(self._get_regret_train_graph(player))

        self._create_memories(memory_capacity)

        # Initialize value networks, losses, optimizers
        self._val_network = ValueNetwork(
            self._value_embedding_size,
            self._value_network_layers,
            activation=self._value_network_activation,
            use_layer_norm=self._value_network_layer_norm,
            residual_mode=self._value_network_residual_mode,
        )
        self._val_network_train = ValueNetwork(
            self._value_embedding_size,
            self._value_network_layers,
            activation=self._value_network_activation,
            use_layer_norm=self._value_network_layer_norm,
            residual_mode=self._value_network_residual_mode,
        )
        self._build_network_once(self._val_network, self._value_embedding_size)
        self._build_network_once(self._val_network_train, self._value_embedding_size)
        self._loss_value = tf.keras.losses.MeanSquaredError()
        self._optimizer_value = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        self._value_train_step = self._get_value_train_graph()
        self._value_test_step = self._get_value_test_graph()
        self._verbose = verbose

    def _reinitialize_policy_network(self):
        """Reinitalize policy network and optimizer for training."""
        with tf.device(self._train_device):
            self._policy_network = PolicyNetwork(self._embedding_size,
                                                 self._policy_network_layers,
                                                 self._num_actions,
                                                 activation=self._policy_network_activation,
                                                 use_layer_norm=self._policy_network_layer_norm,
                                                 residual_mode=self._policy_network_residual_mode,
                                                 head_depth=self._policy_network_head_depth,
                                                 head_units=self._policy_network_head_units)
            self._build_network_once(self._policy_network, self._embedding_size)
            self._optimizer_policy = tf.keras.optimizers.Adam(
                learning_rate=self._learning_rate)
            self._loss_policy = tf.keras.losses.MeanSquaredError()

    def _reinitialize_regret_network(self, player):
        """Reinitalize player's regret network and optimizer for training."""
        with tf.device(self._train_device):
            self._regret_networks_train[player] = RegretNetwork(
                self._embedding_size, self._regret_network_layers,
                self._num_actions,
                activation=self._regret_network_activation,
                use_layer_norm=self._regret_network_layer_norm,
                residual_mode=self._regret_network_residual_mode,
                head_depth=self._regret_network_head_depth,
                head_units=self._regret_network_head_units,
                output_mode=self._regret_network_output_mode)
            self._build_network_once(
                self._regret_networks_train[player],
                self._embedding_size,
            )
            self._optimizer_regrets[player] = tf.keras.optimizers.Adam(
                learning_rate=self._learning_rate)
            self._regret_train_step[player] = (
                self._get_regret_train_graph(player))

    def get_example_info_state(self, player):
        return self._example_info_state[player]

    def get_example_hist_state(self):
        return self._example_hist_state

    def get_example_legal_actions_mask(self, player):
        return self._example_legal_actions_mask[player]

    def _reinitialize_value_network(self):
        """Reinitalize player's value network and optimizer for training."""
        with tf.device(self._train_device):
            self._val_network_train = ValueNetwork(
                self._value_embedding_size,
                self._value_network_layers,
                activation=self._value_network_activation,
                use_layer_norm=self._value_network_layer_norm,
                residual_mode=self._value_network_residual_mode,
            )
            self._build_network_once(
                self._val_network_train,
                self._value_embedding_size,
            )
            self._optimizer_value = tf.keras.optimizers.Adam(
                learning_rate=self._learning_rate)
            self._value_train_step = (self._get_value_train_graph())

    def _build_network_once(self, model, input_size):
        """Create Keras variables before the network enters a tf.function."""
        dummy_x = tf.zeros((1, input_size), dtype=tf.float32)
        dummy_mask = tf.ones((1, self._num_actions), dtype=tf.float32)
        model((dummy_x, dummy_mask), training=False)

    def _learning_rate_for_iteration(self, iteration):
        """Return the configured learning rate for a CFR iteration index."""
        iteration = int(iteration)
        total = max(int(self._num_iterations), 1)
        warmup = max(int(self._learning_rate_warmup_iterations), 0)
        if warmup > 0 and iteration < warmup:
            frac = (iteration + 1) / float(warmup)
            return self._base_learning_rate * (0.1 + 0.9 * frac)

        denom = max(total - warmup, 1)
        progress = min(max((iteration - warmup) / float(denom), 0.0), 1.0)
        schedule = self._learning_rate_schedule
        if schedule == 'constant':
            return self._base_learning_rate
        if schedule == 'linear_decay':
            return (
                self._base_learning_rate
                + progress * (self._learning_rate_end - self._base_learning_rate)
            )
        if schedule == 'cosine_decay':
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return (
                self._learning_rate_end
                + (self._base_learning_rate - self._learning_rate_end) * cosine
            )
        if schedule == 'step_decay':
            if progress < 0.5:
                return self._base_learning_rate
            stepped = self._base_learning_rate * self._learning_rate_decay_rate
            return max(stepped, self._learning_rate_end)
        raise ValueError(f'Unknown learning-rate schedule: {schedule}')

    def _set_learning_rate_for_iteration(self, iteration):
        learning_rate = float(self._learning_rate_for_iteration(iteration))
        self._learning_rate = learning_rate
        self._set_optimizer_learning_rate(getattr(self, "_optimizer_policy", None), learning_rate)
        self._set_optimizer_learning_rate(getattr(self, "_optimizer_value", None), learning_rate)
        for optimizer in getattr(self, "_optimizer_regrets", []):
            self._set_optimizer_learning_rate(optimizer, learning_rate)
        return learning_rate

    @staticmethod
    def _set_optimizer_learning_rate(optimizer, learning_rate):
        if optimizer is None:
            return
        try:
            optimizer.learning_rate.assign(float(learning_rate))
        except Exception:
            optimizer.learning_rate = float(learning_rate)

    @property
    def regret_buffers(self):
        return self._regret_memories

    @property
    def average_policy_buffer(self):
        return self._average_policy_memories

    def clear_regret_buffers(self):
        for p in range(self._num_players):
            self._regret_memories[p].clear()

    def _create_memories(self, memory_capacity):
        """Create memory buffers and associated feature descriptions."""
        self._average_policy_memories = ReservoirBuffer(memory_capacity)
        self._regret_memories = [
            make_regret_replay_buffer(
                self._regret_replay_mode,
                memory_capacity,
                rare_history_quota=self._regret_replay_rare_history_quota,
                weight_floor=self._regret_replay_weight_floor,
            )
            for _ in range(self._num_players)
        ]
        self._value_memory = ReservoirBuffer(memory_capacity)
        self._value_memory_test = ReservoirBuffer(memory_capacity)

        self._average_policy_feature_description = {
            'info_state': tf.io.FixedLenFeature([self._embedding_size], tf.float32),
            'action_probs': tf.io.FixedLenFeature([self._num_actions], tf.float32),
            'iteration': tf.io.FixedLenFeature([1], tf.float32),
            'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32),
            'reach_prob': tf.io.FixedLenFeature(
                [1],
                tf.float32,
                default_value=[1.0],
            ),
            'obs_index': tf.io.FixedLenFeature(
                [1],
                tf.float32,
                default_value=[0.0],
            ),
        }
        self._regret_feature_description = {
            'info_state': tf.io.FixedLenFeature([self._embedding_size], tf.float32),
            'iteration': tf.io.FixedLenFeature([1], tf.float32),
            'samp_regret': tf.io.FixedLenFeature([self._num_actions], tf.float32),
            'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32)
        }
        self._value_feature_description = {
            'hist_state': tf.io.FixedLenFeature([self._value_embedding_size], tf.float32),
            'iteration': tf.io.FixedLenFeature([1], tf.float32),
            'samp_value': tf.io.FixedLenFeature([1], tf.float32),
            'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32),
        }

    def get_val_weights(self):
        return self._val_network.get_weights()

    def set_val_weights(self, weights):
        self._val_network.set_weights(weights)

    def get_num_calls(self):
        if self._save_regret_memories:
            return int(sum(self._regret_memory_counts))
        num_calls = 0
        for p in range(self._num_players):
            num_calls += self._regret_memories[p].get_num_calls()
        return num_calls

    def set_iteration(self, iteration):
        self._iteration = iteration

    def get_weights(self):
        regret_weights = [self._regret_networks[player].get_weights() for player in range(self._num_players)]
        return regret_weights

    def get_policy_weights(self):
        policy_weights = self._policy_network.get_weights()
        return policy_weights

    def set_policy_weights(self, policy_weights):
        self._reinitialize_policy_network()
        self._policy_network.set_weights(policy_weights)

    def get_regret_memories(self, player):
        return self._regret_memories[player].get_data()

    def get_regret_memory_count(self, player):
        if self._save_regret_memories:
            return int(self._regret_memory_counts[player])
        return int(len(self._regret_memories[player].get_data()))

    def get_regret_replay_diagnostics(self, player):
        """Return backend-independent regret replay composition diagnostics."""
        if self._save_regret_memories:
            count = int(self._regret_memory_counts[player])
            return {
                "stored_count": count,
                "stream_count": count,
                "retention_fraction": 1.0 if count else 0.0,
                "unique_infosets": np.nan,
                "samples_per_infoset_min": np.nan,
                "samples_per_infoset_mean": np.nan,
                "samples_per_infoset_max": np.nan,
                "samples_per_infoset_cv": np.nan,
                "stored_weight_mean": np.nan,
            }
        diagnostics = dict(self._regret_memories[player].diagnostics())
        diagnostics.setdefault("stored_weight_mean", np.nan)
        return diagnostics

    def get_regret_memory_storage_bytes(self, player=None):
        if not self._save_regret_memories or not self._regret_memories_tfrecord_dir:
            return 0
        if player is None:
            pattern = os.path.join(
                self._regret_memories_tfrecord_dir,
                "regret_memories_p*_iter*.tfrecord*",
            )
        else:
            pattern = self._regret_tfrecord_pattern(player)
        return int(sum(os.path.getsize(path) for path in glob.glob(pattern)))

    def _current_rss_mb(self):
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return float(rss / (1024.0 ** 2) if rss > 10_000_000 else rss / 1024.0)
        except Exception:
            return np.nan

    def _regret_tfrecord_pattern(self, player):
        return os.path.join(
            self._regret_memories_tfrecord_dir,
            f"regret_memories_p{player}_iter*.tfrecord*",
        )

    def _open_regret_memory_writer(self, player):
        if not self._save_regret_memories:
            return
        if self._regret_tfrecordfiles[player] is not None:
            self._close_regret_memory_writer(player)
        shard_path = os.path.join(
            self._regret_memories_tfrecord_dir,
            f"regret_memories_p{player}_iter{self._iteration:06d}.tfrecord",
        )
        options = (
            tf.io.TFRecordOptions(compression_type=self._tfrecord_compression)
            if self._tfrecord_compression
            else None
        )
        self._regret_tfrecordfiles[player] = tf.io.TFRecordWriter(
            shard_path,
            options=options,
        )

    def _close_regret_memory_writer(self, player):
        writer = self._regret_tfrecordfiles[player]
        if writer is not None:
            writer.close()
            self._regret_tfrecordfiles[player] = None

    def _average_policy_tfrecord_path(self):
        return os.path.join(
            self._average_policy_tfrecord_dir,
            f"average_policy_memories_{self._average_policy_shard_index:06d}.tfrecord",
        )

    def _average_policy_tfrecord_pattern(self):
        return os.path.join(
            self._average_policy_tfrecord_dir,
            "average_policy_memories_*.tfrecord*",
        )

    def _open_average_policy_memory_writer(self):
        if not self._save_average_policy_memories:
            return
        if self._memories_tfrecordfile is not None:
            self._close_average_policy_memory_writer()
        self._memories_tfrecordpath = self._average_policy_tfrecord_path()
        self._memories_tfrecordfile = tf.io.TFRecordWriter(self._memories_tfrecordpath)

    def _close_average_policy_memory_writer(self):
        if self._memories_tfrecordfile is not None:
            self._memories_tfrecordfile.close()
            self._memories_tfrecordfile = None

    def _advance_average_policy_memory_writer(self):
        if not self._save_average_policy_memories:
            return
        self._close_average_policy_memory_writer()
        self._average_policy_shard_index += 1
        self._open_average_policy_memory_writer()

    def get_value_memory(self):
        return self._value_memory.get_data()

    def clear_value_memory(self):
        self._value_memory.clear()

    def get_value_memory_test(self):
        return self._value_memory_test.get_data()

    def get_average_policy_memories(self):
        return self._average_policy_memories.get_data()

    def get_average_policy_memory_count(self):
        if self._save_average_policy_memories:
            return int(self._avg_policy_obs_count)
        return int(len(self.get_average_policy_memories()))

    def get_num_nodes(self):
        return self._nodes_visited

    def get_squared_errors(self):
        return self._squared_errors

    def reset_squared_errors(self):
        self._squared_errors = []

    def get_squared_errors_child(self):
        return self._squared_errors_child

    def reset_squared_errors_child(self):
        self._squared_errors_child = []

    def _reset_regret_target_consistency_diagnostics(self):
        """Reset online raw-target diagnostics for the current ESCHER iteration."""
        self._regret_target_consistency = [
            {
                "count": 0,
                "bellman_residual_sum": 0.0,
                "bellman_residual_abs_sum": 0.0,
                "bellman_residual_sq_sum": 0.0,
                "policy_weighted_target_abs_sum": 0.0,
                "all_legal_targets_negative_count": 0,
            }
            for _ in range(self._num_players)
        ]

    def _record_regret_target_consistency(
        self,
        player,
        target_result,
        legal_actions_mask,
    ):
        """Accumulate Bellman and policy-centering diagnostics for one target."""
        stats = self._regret_target_consistency[player]
        residual = float(target_result.bellman_residual)
        weighted_target = float(target_result.policy_weighted_target)
        legal = np.asarray(legal_actions_mask, dtype=np.float64) > 0.0
        legal_targets = np.asarray(target_result.target, dtype=np.float64)[legal]

        stats["count"] += 1
        stats["bellman_residual_sum"] += residual
        stats["bellman_residual_abs_sum"] += abs(residual)
        stats["bellman_residual_sq_sum"] += residual * residual
        stats["policy_weighted_target_abs_sum"] += abs(weighted_target)
        if legal_targets.size and np.all(legal_targets < 0.0):
            stats["all_legal_targets_negative_count"] += 1

    def _regret_target_consistency_summary(self, player):
        """Return scalar diagnostics accumulated for ``player`` this iteration."""
        stats = self._regret_target_consistency[player]
        count = int(stats["count"])
        if count == 0:
            return {
                "count": 0,
                "bellman_residual_mean": np.nan,
                "bellman_residual_abs_mean": np.nan,
                "bellman_residual_rmse": np.nan,
                "policy_weighted_target_abs_mean": np.nan,
                "all_legal_targets_negative_fraction": np.nan,
            }
        return {
            "count": count,
            "bellman_residual_mean": stats["bellman_residual_sum"] / count,
            "bellman_residual_abs_mean": stats["bellman_residual_abs_sum"] / count,
            "bellman_residual_rmse": np.sqrt(
                stats["bellman_residual_sq_sum"] / count
            ),
            "policy_weighted_target_abs_mean": (
                stats["policy_weighted_target_abs_sum"] / count
            ),
            "all_legal_targets_negative_fraction": (
                stats["all_legal_targets_negative_count"] / count
            ),
        }

    def clear_val_memories_test(self):
        self._value_memory_test.clear()

    def clear_val_memories(self):
        self._value_memory.clear()

    def traverse_game_tree_n_times(self, n, p, train_regret=False, train_value=False,
                                   record_value=False,
                                   track_mean_squares=True, on_policy_prob=0., expl=0.6, val_test=False):
        for i in range(n):
            if i > 0:
                track_mean_squares = False
            self._traverse_game_tree(self._root_node, p, my_reach=1.0, opp_reach=1.0, sample_reach=1.0,
                                     my_sample_reach=1.0, chance_reach=1.0,
                                     train_regret=train_regret, train_value=train_value,
                                     record_value=record_value,
                                     track_mean_squares=track_mean_squares, on_policy_prob=on_policy_prob,
                                     expl=expl, val_test=val_test)

    def traverse_game_tree_joint_on_policy_n_times(self, n, track_mean_squares=False):
        """Collect regret samples from trajectories under the current joint policy."""
        for i in range(n):
            self._traverse_game_tree_joint_on_policy(
                self._root_node,
                track_mean_squares=(track_mean_squares and i == 0),
                last_action=0,
            )

    def init_regret_net(self):
        # initialize regret network
        for p in range(self._num_players):
            example_info_state = self.get_example_info_state(p)
            example_legal_actions_mask = self.get_example_legal_actions_mask(p)
            self.traverse_game_tree_n_times(1, p, track_mean_squares=False)
            self._init_main_regret_network(example_info_state, example_legal_actions_mask, p)

    def init_val_net(self):
        example_hist_state = self.get_example_hist_state()
        example_legal_actions_mask = self.get_example_legal_actions_mask(0)
        self._init_main_val_network(example_hist_state, example_legal_actions_mask)

    def play_game_against_random(self):
        # play one game per player
        reward = 0
        for player in [0, 1]:
            state = self._game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes, probs = zip(*state.chance_outcomes())
                    aidx = np.random.choice(range(len(outcomes)), p=probs)
                    action = outcomes[aidx]
                else:
                    cur_player = state.current_player()
                    legal_actions = state.legal_actions(cur_player)
                    legal_actions_mask = tf.constant(
                        state.legal_actions_mask(cur_player), dtype=tf.float32)
                    obs = tf.constant(state.observation_tensor(), dtype=tf.float32)
                    if len(obs.shape) == 1:
                        obs = tf.expand_dims(obs, axis=0)
                    if cur_player == player:
                        probs = self._policy_network((obs, legal_actions_mask), training=False)
                        probs = probs.numpy()[0]
                        probs /= probs.sum()
                        action = np.random.choice(range(state.num_distinct_actions()), p=probs)
                    elif cur_player == 1 - player:
                        action = random.choice(state.legal_actions())
                    else:
                        print("Got player ", str(cur_player))
                        break
                state.apply_action(action)
            reward += state.returns()[player]
        return reward

    def play_n_games_against_random(self, n):
        total_reward = 0
        for i in range(n):
            reward = self.play_game_against_random()
            total_reward += reward
        return total_reward / (2 * n)

    def print_mse(self):
        # track MSE
        squared_errors = self.get_squared_errors()
        self.reset_squared_errors()
        squared_errors_child = self.get_squared_errors_child()
        self.reset_squared_errors_child()
        print(sum(squared_errors) / len(squared_errors), "Mean Squared Errors")
        print(sum(squared_errors_child) / len(squared_errors_child), "Mean Squared Errors Child")

    def solve(self, save_path_convs=None):
        """Run ESCHER training and collect thesis-style diagnostics.

        Returns:
          regret_losses: dict[player -> list[float]]
          policy_loss: final policy-network loss
          convs: NashConv values at evaluation checkpoints
          nodes: nodes touched at evaluation checkpoints
          average_policy_values: player-0 average-policy values at checkpoints
          diagnostics: dict of per-checkpoint diagnostic arrays/lists
        """
        def _to_float(x):
            if x is None:
                return np.nan
            try:
                return float(x.numpy())
            except Exception:
                try:
                    return float(np.asarray(x))
                except Exception:
                    return np.nan

        regret_losses = collections.defaultdict(list)
        value_losses = []
        convs = []
        nodes = []
        average_policy_values = []
        diagnostics = collections.defaultdict(list)
        policy_loss = np.nan
        last_regret_losses = {p: np.nan for p in range(self._num_players)}
        last_value_loss = np.nan
        last_value_test_loss = np.nan
        solve_start_time = time.time()
        timestr = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())

        if self._use_balanced_probs:
            self._prepare_fixed_sampling_policy()
        elif self._track_sampling_coverage:
            self._fixed_sampling_static_diagnostics = (
                self._compute_fixed_sampling_static_diagnostics()
            )

        with tf.device(self._infer_device):
            with contextlib.ExitStack() as stack:
                if self._save_average_policy_memories:
                    self._open_average_policy_memory_writer()
                    stack.callback(self._close_average_policy_memory_writer)

                # Build networks/memories with a minimal traversal before the main loop.
                self.traverse_game_tree_n_times(1, 0, track_mean_squares=False)

                for i in range(self._num_iterations + 1):
                    current_lr = self._set_learning_rate_for_iteration(i)
                    self._reset_regret_target_consistency_diagnostics()
                    if self._verbose:
                        print(i)
                    if self._verbose and self._experiment_string is not None:
                        print(self._experiment_string)

                    # Initialise / refresh current regret and value networks.
                    self.init_regret_net()
                    self.init_val_net()

                    # Train the history-value function. Baseline ESCHER uses a
                    # separate value-traversal pass; the reuse ablation fills
                    # value memory during player-0 regret traversals below.
                    if not self._reuse_regret_traversals_for_value:
                        self.traverse_game_tree_n_times(
                            self._num_val_fn_traversals, 0, train_value=True,
                            track_mean_squares=False, on_policy_prob=self._val_op_prob,
                            expl=self._val_expl)
                        self.traverse_game_tree_n_times(
                            self._value_test_traversals, 0, train_value=True,
                            track_mean_squares=False,
                            on_policy_prob=self._val_op_prob, expl=self._val_expl,
                            val_test=True)
                        if self._reinit_value_network:
                            self._reinitialize_value_network()
                        value_loss = self._learn_value_network()
                        value_losses.append(value_loss)
                        last_value_loss = _to_float(value_loss)
                        last_value_test_loss = _to_float(self._get_value_test_loss())
                        if self._clear_value_buffer:
                            self.clear_val_memories_test()
                            self.clear_val_memories()
                    elif self._bootstrap_value_with_separate_traversal and i == 0:
                        self.traverse_game_tree_n_times(
                            self._num_val_fn_traversals, 0, train_value=True,
                            track_mean_squares=False, on_policy_prob=self._val_op_prob,
                            expl=self._val_expl)
                        if self._reinit_value_network:
                            self._reinitialize_value_network()
                        value_loss = self._learn_value_network()
                        value_losses.append(value_loss)
                        last_value_loss = _to_float(value_loss)
                        if self._clear_value_buffer:
                            self.clear_val_memories()

                    # Train regret networks. Baseline ESCHER uses separate
                    # player-specific regret traversal batches. The on-policy
                    # ablation uses one joint trajectory batch and writes regret
                    # samples for whichever player acts at visited decision nodes.
                    track_mse = True
                    if self._on_policy_joint_regret_updates:
                        opened_regret_writers = []
                        try:
                            if self._save_regret_memories:
                                for p in range(self._num_players):
                                    self._open_regret_memory_writer(p)
                                    opened_regret_writers.append(p)
                            self.traverse_game_tree_joint_on_policy_n_times(
                                self._num_traversals,
                                track_mean_squares=track_mse,
                            )
                        finally:
                            if self._save_regret_memories:
                                for p in reversed(opened_regret_writers):
                                    self._close_regret_memory_writer(p)
                        num_nodes = self.get_num_nodes()
                        for p in range(self._num_players):
                            if self._reinitialize_regret_networks:
                                self._reinitialize_regret_network(p)
                            regret_loss = self._learn_regret_network(p)
                            regret_losses[p].append(regret_loss)
                            last_regret_losses[p] = _to_float(regret_loss)
                            if self._save_regret_networks:
                                os.makedirs(self._save_regret_networks, exist_ok=True)
                                self._regret_networks[p].save(
                                    os.path.join(self._save_regret_networks,
                                                 f'regretnet_p{p}_it{self._iteration:04}'))
                    else:
                        for p in range(self._num_players):
                            record_value = bool(
                                self._reuse_regret_traversals_for_value and p == 0
                            )
                            if self._save_regret_memories:
                                self._open_regret_memory_writer(p)
                            try:
                                self.traverse_game_tree_n_times(
                                    self._num_traversals, p, train_regret=True,
                                    record_value=record_value,
                                    track_mean_squares=track_mse, expl=self._expl)
                            finally:
                                if self._save_regret_memories:
                                    self._close_regret_memory_writer(p)
                            num_nodes = self.get_num_nodes()
                            if self._reinitialize_regret_networks:
                                self._reinitialize_regret_network(p)
                            regret_loss = self._learn_regret_network(p)
                            regret_losses[p].append(regret_loss)
                            last_regret_losses[p] = _to_float(regret_loss)
                            if self._save_regret_networks:
                                os.makedirs(self._save_regret_networks, exist_ok=True)
                                self._regret_networks[p].save(
                                    os.path.join(self._save_regret_networks,
                                                 f'regretnet_p{p}_it{self._iteration:04}'))

                    if self._reuse_regret_traversals_for_value:
                        self.traverse_game_tree_n_times(
                            self._value_test_traversals, 0, train_value=True,
                            track_mean_squares=False,
                            on_policy_prob=self._val_op_prob, expl=self._val_expl,
                            val_test=True)
                        if self._reinit_value_network:
                            self._reinitialize_value_network()
                        value_loss = self._learn_value_network()
                        value_losses.append(value_loss)
                        last_value_loss = _to_float(value_loss)
                        last_value_test_loss = _to_float(self._get_value_test_loss())
                        if self._clear_value_buffer:
                            self.clear_val_memories_test()
                            self.clear_val_memories()

                    # Evaluate the learned average policy at fixed checkpoints.
                    self._iteration += 1
                    if self._compute_exploitability and i % self._check_exploitability_every == 0:
                        if self._save_average_policy_memories:
                            self._close_average_policy_memory_writer()
                        self._reinitialize_policy_network()
                        policy_loss = self._learn_average_policy_network()
                        policy_loss_float = _to_float(policy_loss)
                        if self._save_average_policy_memories:
                            self._average_policy_shard_index += 1
                            self._open_average_policy_memory_writer()

                        if self._save_policy_weights:
                            save_path_model = (save_path_convs if save_path_convs is not None else './tmp/results') + "/" + timestr
                            os.makedirs(save_path_model, exist_ok=True)
                            model_path = save_path_model + "/policy_nodes_" + str(num_nodes)
                            self._policy_network.save_weights(model_path)
                            self.save_policy_network(model_path + "full_model")

                        average_policy = policy.tabular_policy_from_callable(
                            self._game, self.action_probabilities)
                        avg_policy_value = expected_game_score.policy_value(
                            self._game.new_initial_state(), [average_policy] * self._num_players)
                        conv = exploitability.nash_conv(self._game, average_policy)

                        convs.append(float(conv))
                        nodes.append(float(num_nodes))
                        average_policy_values.append(float(avg_policy_value[0]))

                        diagnostics["iteration"].append(int(i + 1))
                        diagnostics["solver_iteration"].append(int(self._iteration))
                        diagnostics["wall_clock_seconds"].append(float(time.time() - solve_start_time))
                        diagnostics["learning_rate"].append(float(current_lr))
                        diagnostics["policy_loss"].append(policy_loss_float)
                        diagnostics["value_loss"].append(float(last_value_loss))
                        diagnostics["value_test_loss"].append(float(last_value_test_loss))
                        diagnostics["regret_loss_player_0"].append(float(last_regret_losses.get(0, np.nan)))
                        diagnostics["regret_loss_player_1"].append(float(last_regret_losses.get(1, np.nan)))
                        diagnostics["average_policy_buffer_size"].append(int(self.get_average_policy_memory_count()))
                        diagnostics["regret_buffer_size_player_0"].append(int(self.get_regret_memory_count(0)))
                        diagnostics["regret_buffer_size_player_1"].append(int(self.get_regret_memory_count(1)))
                        for p in range(self._num_players):
                            replay_diagnostics = self.get_regret_replay_diagnostics(p)
                            for name in [
                                "stream_count",
                                "retention_fraction",
                                "unique_infosets",
                                "samples_per_infoset_min",
                                "samples_per_infoset_mean",
                                "samples_per_infoset_max",
                                "samples_per_infoset_cv",
                                "stored_weight_mean",
                            ]:
                                diagnostics[f"regret_replay_{name}_player_{p}"].append(
                                    float(replay_diagnostics.get(name, np.nan))
                                )
                            if self._track_sampling_coverage:
                                coverage = self.get_sampling_coverage_diagnostics(p)
                                for name, value in coverage.items():
                                    diagnostics[
                                        f"sampling_coverage_{name}_player_{p}"
                                    ].append(float(value))
                        for name, value in (
                            self._fixed_sampling_static_diagnostics.items()
                        ):
                            diagnostics[f"fixed_sampling_{name}"].append(float(value))
                        diagnostics["regret_storage_bytes_player_0"].append(int(self.get_regret_memory_storage_bytes(0)))
                        diagnostics["regret_storage_bytes_player_1"].append(int(self.get_regret_memory_storage_bytes(1)))
                        diagnostics["regret_storage_bytes_total"].append(int(self.get_regret_memory_storage_bytes()))
                        diagnostics["raw_regret_target_variance_player_0"].append(float(self._last_raw_regret_target_variance[0]))
                        diagnostics["raw_regret_target_variance_player_1"].append(float(self._last_raw_regret_target_variance[1]))
                        diagnostics["processed_regret_target_variance_player_0"].append(float(self._last_processed_regret_target_variance[0]))
                        diagnostics["processed_regret_target_variance_player_1"].append(float(self._last_processed_regret_target_variance[1]))
                        diagnostics["processed_regret_target_abs_mean_player_0"].append(float(self._last_processed_regret_target_abs_mean[0]))
                        diagnostics["processed_regret_target_abs_mean_player_1"].append(float(self._last_processed_regret_target_abs_mean[1]))
                        diagnostics["regret_target_standardization_mean_player_0"].append(float(self._last_regret_target_standardization_mean[0]))
                        diagnostics["regret_target_standardization_mean_player_1"].append(float(self._last_regret_target_standardization_mean[1]))
                        diagnostics["regret_target_standardization_scale_player_0"].append(float(self._last_regret_target_standardization_scale[0]))
                        diagnostics["regret_target_standardization_scale_player_1"].append(float(self._last_regret_target_standardization_scale[1]))
                        diagnostics["regret_target_processing_mean_player_0"].append(float(self._last_regret_target_standardization_mean[0]))
                        diagnostics["regret_target_processing_mean_player_1"].append(float(self._last_regret_target_standardization_mean[1]))
                        diagnostics["regret_target_processing_scale_player_0"].append(float(self._last_regret_target_standardization_scale[0]))
                        diagnostics["regret_target_processing_scale_player_1"].append(float(self._last_regret_target_standardization_scale[1]))
                        diagnostics["regret_target_clip_fraction_player_0"].append(float(self._last_regret_target_clip_fraction[0]))
                        diagnostics["regret_target_clip_fraction_player_1"].append(float(self._last_regret_target_clip_fraction[1]))
                        diagnostics["regret_target_sign_flip_fraction_player_0"].append(float(self._last_regret_target_sign_flip_fraction[0]))
                        diagnostics["regret_target_sign_flip_fraction_player_1"].append(float(self._last_regret_target_sign_flip_fraction[1]))
                        diagnostics["raw_regret_target_positive_fraction_player_0"].append(float(self._last_raw_regret_target_positive_fraction[0]))
                        diagnostics["raw_regret_target_positive_fraction_player_1"].append(float(self._last_raw_regret_target_positive_fraction[1]))
                        diagnostics["processed_regret_target_positive_fraction_player_0"].append(float(self._last_processed_regret_target_positive_fraction[0]))
                        diagnostics["processed_regret_target_positive_fraction_player_1"].append(float(self._last_processed_regret_target_positive_fraction[1]))
                        for p in range(self._num_players):
                            consistency = self._regret_target_consistency_summary(p)
                            diagnostics[f"regret_target_sample_count_player_{p}"].append(
                                int(consistency["count"])
                            )
                            for name in [
                                "bellman_residual_mean",
                                "bellman_residual_abs_mean",
                                "bellman_residual_rmse",
                                "policy_weighted_target_abs_mean",
                                "all_legal_targets_negative_fraction",
                            ]:
                                diagnostics[f"regret_target_{name}_player_{p}"].append(
                                    float(consistency[name])
                                )
                        diagnostics["peak_rss_mb"].append(float(self._current_rss_mb()))
                        diagnostics["value_buffer_size"].append(int(len(self.get_value_memory())))
                        diagnostics["value_test_buffer_size"].append(int(len(self.get_value_memory_test())))

                        if save_path_convs:
                            np.save(save_path_convs + "_convs.npy", np.array(convs, dtype=float))
                            np.save(save_path_convs + "_nodes.npy", np.array(nodes, dtype=float))
                            np.save(save_path_convs + "_average_policy_values.npy", np.array(average_policy_values, dtype=float))

                        if self._verbose:
                            print(self._iteration, num_nodes, conv, avg_policy_value)

        # Train the final policy network so the returned solver is immediately playable.
        self._reinitialize_policy_network()
        policy_loss = self._learn_average_policy_network()
        return regret_losses, policy_loss, convs, nodes, average_policy_values, diagnostics

    def save_policy_network(self, outputfolder):
        """Saves the policy network to the given folder."""
        os.makedirs(outputfolder, exist_ok=True)
        self._policy_network.save(outputfolder)

    def train_policy_network_from_file(self,
                                       tfrecordpath,
                                       iteration=None,
                                       batch_size_average_policy=None,
                                       policy_network_train_steps=None,
                                       reinitialize_policy_network=True):
        """Trains the policy network from a previously stored tfrecords-file."""
        self._memories_tfrecordpath = tfrecordpath
        if iteration:
            self._iteration = iteration
        if batch_size_average_policy:
            self._batch_size_average_policy = batch_size_average_policy
        if policy_network_train_steps:
            self._policy_network_train_steps = policy_network_train_steps
        if reinitialize_policy_network:
            self._reinitialize_policy_network()
        policy_loss = self._learn_average_policy_network()
        return policy_loss

    def _add_to_average_policy_memory(
        self,
        info_state,
        iteration,
        average_policy_action_probs,
        legal_actions_mask,
        reach_prob=1.0,
    ):
        # pylint: disable=g-doc-args
        """Adds the given average_policy data to the memory.

        Uses either a tfrecordsfile on disk if provided, or a reservoir buffer.
        """
        self._avg_policy_obs_count += 1
        serialized_example = self._serialize_average_policy_memory(
            info_state,
            iteration,
            average_policy_action_probs,
            legal_actions_mask,
            reach_prob,
            self._avg_policy_obs_count,
        )
        if self._save_average_policy_memories:
            self._memories_tfrecordfile.write(serialized_example)
        else:
            self._average_policy_memories.add(serialized_example)

    def _serialize_average_policy_memory(
        self,
        info_state,
        iteration,
        average_policy_action_probs,
        legal_actions_mask,
        reach_prob=1.0,
        obs_index=0,
    ):
        """Create serialized example to store a average_policy entry."""
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'info_state':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=info_state)),
                    'action_probs':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=average_policy_action_probs)),
                    'iteration':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[iteration])),
                    'legal_actions':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=legal_actions_mask)),
                    'reach_prob':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[float(reach_prob)])),
                    'obs_index':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[float(obs_index)])),
                }))
        return example.SerializeToString()

    def _deserialize_average_policy_memory(self, serialized):
        """Deserializes a batch of average_policy examples for the train step."""
        tups = tf.io.parse_example(serialized, self._average_policy_feature_description)
        return (tups['info_state'], tups['action_probs'], tups['iteration'],
                tups['legal_actions'], tups['reach_prob'], tups['obs_index'])

    def _add_to_regret_memory(
        self,
        player,
        info_state,
        iteration,
        samp_regret,
        legal_actions_mask,
        counterfactual_reach=1.0,
        sampling_reach=1.0,
    ):
        """Adds regret data either to RAM replay or disk-backed TFRecord replay."""
        infoset_key = np.asarray(info_state, dtype=np.float32).tobytes()
        self._record_sampling_coverage(player, infoset_key, sampling_reach)
        serialized_example = self._serialize_regret_memory(
            info_state,
            iteration,
            samp_regret,
            legal_actions_mask,
        )
        if self._save_regret_memories:
            writer = self._regret_tfrecordfiles[player]
            if writer is None:
                raise RuntimeError(
                    "Disk-backed regret memory is enabled but no TFRecord writer "
                    "is open for the active regret traversal."
                )
            writer.write(serialized_example)
            self._regret_memory_counts[player] += 1
        else:
            replay_weight = float(counterfactual_reach)
            if not np.isfinite(replay_weight) or replay_weight <= 0.0:
                replay_weight = self._regret_replay_weight_floor
            self._regret_memories[player].add(
                serialized_example,
                key=infoset_key,
                weight=max(replay_weight, self._regret_replay_weight_floor),
            )

    def _serialize_regret_memory(self, info_state, iteration, samp_regret,
                                 legal_actions_mask):
        """Create serialized example to store an regret entry."""
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'info_state':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=info_state)),
                    'iteration':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[iteration])),
                    'samp_regret':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=samp_regret)),
                    'legal_actions':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=legal_actions_mask))
                }))
        return example.SerializeToString()

    def _serialize_value_memory(self, hist_state, iteration, samp_value, legal_actions_mask):
        """Create serialized example to store a value entry."""
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'hist_state':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=hist_state)),
                    'iteration':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[iteration])),
                    'samp_value':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[samp_value])),
                    'legal_actions':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=legal_actions_mask))
                }))
        return example.SerializeToString()

    def _deserialize_regret_memory(self, serialized):
        """Deserializes a batch of regret examples for the train step."""
        tups = tf.io.parse_example(serialized, self._regret_feature_description)
        return (tups['info_state'], tups['samp_regret'], tups['iteration'],
                tups['legal_actions'])

    def _deserialize_value_memory(self, serialized):
        """Deserializes a batch of regret examples for the train step."""
        tups = tf.io.parse_example(serialized, self._value_feature_description)
        return (tups['hist_state'], tups['samp_value'], tups['iteration'], tups['legal_actions'])

    def _baseline(self, state, aidx):  # pylint: disable=unused-argument
        # Default to vanilla outcome sampling
        return 0

    def _baseline_corrected_child_value(self, state, sampled_aidx,
                                        aidx, child_value, sample_prob):
        # Applies Eq. 9 of Schmid et al. '19
        baseline = self._baseline(state, aidx)
        if aidx == sampled_aidx:
            return baseline + (child_value - baseline) / sample_prob
        else:
            return baseline

    def _exact_value(self, state, update_player):
        state = state.clone()
        if state.is_terminal():
            return state.player_return(update_player)
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            val = 0
            for aidx in range(len(outcomes)):
                new_state = state.child(outcomes[aidx])
                val += probs[aidx] * self._exact_value(new_state, update_player)
            return val
        cur_player = state.current_player()
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        _, policy = self._sample_action_from_regret(state, cur_player)
        val = 0
        for aidx in range(num_legal_actions):
            new_state = state.child(legal_actions[aidx])
            val += policy[aidx] * self._exact_value(new_state, update_player)
        return val

    def _get_balanced_probs(self, state):
        """Populate the fixed policy that gives every descendant leaf equal reach."""
        if state.is_terminal():
            return 1
        elif state.is_chance_node():
            legal_actions = state.legal_actions()
            num_nodes = 0
            for action in legal_actions:
                num_nodes += self._get_balanced_probs(state.child(action))
            return num_nodes
        else:
            legal_actions = state.legal_actions()
            num_nodes = 0
            balanced_probs = np.zeros((state.num_distinct_actions()))
            for action in legal_actions:
                nodes = self._get_balanced_probs(state.child(action))
                balanced_probs[action] = nodes
                num_nodes += nodes
            balanced_probs /= balanced_probs.sum()
            infoset_key = state.information_state_string()
            cur_player = int(state.current_player())
            previous = self._balanced_probs.get(infoset_key)
            if previous is not None and not np.allclose(
                previous,
                balanced_probs,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Leaf-balanced probabilities are inconsistent across "
                    f"histories in infoset {infoset_key!r}."
                )
            previous_player = self._balanced_prob_players.get(infoset_key)
            if previous_player is not None and previous_player != cur_player:
                raise ValueError(
                    "Information-state strings collide across players: "
                    f"{infoset_key!r}."
                )
            self._balanced_probs[infoset_key] = balanced_probs
            self._balanced_prob_players[infoset_key] = cur_player
            return num_nodes

    def _fixed_sampling_policy(self, state):
        """Return the configured fixed distribution at one decision state."""
        legal_mask = np.asarray(state.legal_actions_mask(), dtype=np.float64)
        if self._use_balanced_probs:
            infoset_key = state.information_state_string()
            if infoset_key not in self._balanced_probs:
                raise RuntimeError(
                    "Balanced sampling policy was not prepared before traversal."
                )
            balanced_probs = self._balanced_probs[infoset_key]
            mix = self._balanced_sampling_mix
        else:
            balanced_probs = legal_mask
            mix = 0.0
        return fixed_sampling_policy(legal_mask, balanced_probs, mix)

    def _prepare_fixed_sampling_policy(self):
        """Build the immutable balanced table and optional exact diagnostics."""
        self._balanced_probs = {}
        self._balanced_prob_players = {}
        self._get_balanced_probs(self._root_node)
        if self._track_sampling_coverage:
            self._fixed_sampling_static_diagnostics = (
                self._compute_fixed_sampling_static_diagnostics()
            )

    def _compute_fixed_sampling_static_diagnostics(self):
        """Enumerate own-policy reach to every player decision history."""
        action_probabilities = []
        own_history_reaches = [[] for _ in range(self._num_players)]
        infosets = [set() for _ in range(self._num_players)]

        def walk(state, own_reaches):
            if state.is_terminal():
                return
            if state.is_chance_node():
                for action in state.legal_actions():
                    walk(state.child(action), own_reaches)
                return

            player = int(state.current_player())
            infosets[player].add(state.information_state_string())
            own_history_reaches[player].append(float(own_reaches[player]))
            sampling_policy = self._fixed_sampling_policy(state)
            legal_actions = state.legal_actions()
            action_probabilities.extend(
                float(sampling_policy[action]) for action in legal_actions
            )
            for action in legal_actions:
                child_reaches = list(own_reaches)
                child_reaches[player] *= float(sampling_policy[action])
                walk(state.child(action), child_reaches)

        walk(self._root_node, [1.0 for _ in range(self._num_players)])
        diagnostics = {
            "effective_balanced_mix": (
                self._balanced_sampling_mix if self._use_balanced_probs else 0.0
            ),
            "legal_action_probability_min": (
                float(np.min(action_probabilities))
                if action_probabilities
                else np.nan
            ),
        }
        for player in range(self._num_players):
            reaches = np.asarray(
                own_history_reaches[player],
                dtype=np.float64,
            )
            diagnostics[f"infoset_count_player_{player}"] = int(
                len(infosets[player])
            )
            diagnostics[f"history_count_player_{player}"] = int(reaches.size)
            diagnostics[f"own_history_reach_min_player_{player}"] = (
                float(np.min(reaches)) if reaches.size else np.nan
            )
            diagnostics[f"own_history_reach_mean_player_{player}"] = (
                float(np.mean(reaches)) if reaches.size else np.nan
            )
            diagnostics[f"own_history_reach_cv_player_{player}"] = (
                float(np.std(reaches) / np.mean(reaches))
                if reaches.size and np.mean(reaches) > 0.0
                else np.nan
            )
        return diagnostics

    def _record_sampling_coverage(self, player, infoset_key, sampling_reach):
        if not self._track_sampling_coverage:
            return
        self._sampling_coverage_counts[player][infoset_key] += 1
        reach = float(sampling_reach)
        if not np.isfinite(reach) or reach < 0.0:
            return
        stats = self._sampling_reach_stats[player]
        stats["count"] += 1
        stats["sum"] += reach
        stats["min"] = min(stats["min"], reach)
        stats["max"] = max(stats["max"], reach)

    def get_sampling_coverage_diagnostics(self, player):
        """Return cumulative empirical fixed-sampling coverage diagnostics."""
        counts = np.asarray(
            list(self._sampling_coverage_counts[player].values()),
            dtype=np.float64,
        )
        stats = self._sampling_reach_stats[player]
        return {
            "unique_infosets": int(counts.size),
            "visits_min": float(np.min(counts)) if counts.size else 0.0,
            "visits_mean": float(np.mean(counts)) if counts.size else 0.0,
            "visits_max": float(np.max(counts)) if counts.size else 0.0,
            "visits_cv": (
                float(np.std(counts) / np.mean(counts))
                if counts.size and np.mean(counts) > 0.0
                else 0.0
            ),
            "observed_own_reach_min": (
                float(stats["min"]) if stats["count"] else np.nan
            ),
            "observed_own_reach_mean": (
                float(stats["sum"] / stats["count"])
                if stats["count"]
                else np.nan
            ),
            "observed_own_reach_max": (
                float(stats["max"]) if stats["count"] else np.nan
            ),
        }

    def _traverse_game_tree(self, state, player, my_reach, opp_reach, sample_reach,
                            my_sample_reach, chance_reach, train_regret, train_value,
                            record_value=False,
                            on_policy_prob=0., track_mean_squares=True, expl=1.0, val_test=False, last_action=0):
        """Performs a traversal of the game tree using external sampling.

        Over a traversal the regret and average_policy memories are populated with
        computed regret values and matched regrets respectively if train_regret=True.
        If train_value=True then we use traversals to train the history value function.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index for this traversal.

        Returns:
          Recursively returns expected payoffs for each action.
        """
        self._nodes_visited += 1
        if state.is_terminal():
            # Terminal state get returns.
            return state.returns()[player], state.returns()[player]
        elif state.is_chance_node():
            # If this is a chance node, sample an action
            outcomes, probs = zip(*state.chance_outcomes())
            aidx = np.random.choice(range(len(outcomes)), p=probs)
            action = outcomes[aidx]
            new_state = state.child(action)
            return self._traverse_game_tree(new_state, player, my_reach,
                                            probs[aidx] * opp_reach, probs[aidx] * sample_reach, my_sample_reach,
                                            probs[aidx] * chance_reach,
                                            train_regret, train_value,
                                            record_value=record_value, expl=expl,
                                            track_mean_squares=track_mean_squares, val_test=val_test,
                                            last_action=action)

        # with probability equal to op_prob, we switch over to on-policy rollout for remainder of trajectory
        # used for value estimation to get coverage but not needing importance sampling
        if expl != 0.:
            if np.random.rand() < on_policy_prob:
                expl = 0.

        cur_player = state.current_player()
        legal_actions = state.legal_actions()
        num_legal_actions = len(legal_actions)
        num_actions = state.num_distinct_actions()
        _, policy = self._sample_action_from_regret(state, state.current_player())

        if cur_player == player or train_value:
            fixed_policy = self._fixed_sampling_policy(state)
            sample_policy = expl * fixed_policy + (1.0 - expl) * policy
        else:
            sample_policy = policy

        sample_policy /= sample_policy.sum()
        sampled_action = np.random.choice(range(state.num_distinct_actions()), p=sample_policy)
        orig_state = state.clone()
        new_state = state.child(sampled_action)

        child_value = self._estimate_value_from_hist(new_state.clone(), player, last_action=sampled_action)
        value_estimate = self._estimate_value_from_hist(state.clone(), player, last_action=last_action)

        if track_mean_squares:
            oracle_child_value = self._exact_value(new_state.clone(), player)
            oracle_value_estimate = self._exact_value(state.clone(), player)
            squared_error = (oracle_value_estimate - value_estimate) ** 2
            self._squared_errors.append(squared_error)
            squared_child_error = (oracle_child_value - child_value) ** 2
            self._squared_errors_child.append(squared_child_error)

        if cur_player == player:
            new_my_reach = my_reach * policy[sampled_action]
            new_opp_reach = opp_reach
            new_my_sample_reach = my_sample_reach * sample_policy[sampled_action]
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy[sampled_action]
            new_my_sample_reach = my_sample_reach
        new_sample_reach = sample_reach * sample_policy[sampled_action]

        iw_sampled_value, sampled_value = self._traverse_game_tree(new_state, player, new_my_reach,
                                                                   new_opp_reach, new_sample_reach, new_my_sample_reach,
                                                                   chance_reach, train_regret, train_value,
                                                                   record_value=record_value, expl=expl,
                                                                   track_mean_squares=track_mean_squares,
                                                                   val_test=val_test, last_action=sampled_action)
        importance_weighted_sampled_value = iw_sampled_value * policy[sampled_action] / sample_policy[sampled_action]

        # Compute each of the child estimated values.
        child_values = np.zeros(num_actions, dtype=np.float64)
        if self._all_actions:
            for aidx in range(num_legal_actions):
                cloned_state = orig_state.clone()
                action = legal_actions[aidx]
                new_cloned_state = cloned_state.child(action)
                child_values[action] = self._estimate_value_from_hist(new_cloned_state.clone(), player,
                                                                      last_action=action)
        else:
            child_values[sampled_action] = child_value / sample_policy[sampled_action]

        if train_regret:
            if cur_player == player:
                cf_action_values = 0 * policy
                for action in range(num_actions):
                    if self._importance_sampling:
                        action_sample_reach = my_sample_reach * sample_policy[sampled_action]
                        cf_value = value_estimate * min(1 / my_sample_reach, self._importance_sampling_threshold)
                        cf_action_value = child_values[action] * min(1 / action_sample_reach,
                                                                     self._importance_sampling_threshold)
                    else:
                        cf_action_value = child_values[action]
                        cf_value = value_estimate
                    cf_action_values[action] = cf_action_value

                legal_actions_mask = state.legal_actions_mask(player)
                target_result = compute_regret_target(
                    cf_action_values,
                    cf_value,
                    policy,
                    legal_actions_mask,
                    baseline_mode=self._regret_target_baseline,
                )
                samp_regret = target_result.target
                self._record_regret_target_consistency(
                    player,
                    target_result,
                    legal_actions_mask,
                )

                network_input = state.information_state_tensor()

                self._add_to_regret_memory(
                    player,
                    network_input,
                    self._iteration,
                    samp_regret,
                    state.legal_actions_mask(player),
                    counterfactual_reach=opp_reach,
                    sampling_reach=my_sample_reach,
                )
            else:
                obs_input = state.information_state_tensor(cur_player)
                actor_reach = opp_reach / max(chance_reach, 1e-12)
                actor_reach = float(np.clip(actor_reach, 0.0, 1.0))
                self._add_to_average_policy_memory(
                    obs_input,
                    self._iteration,
                    policy,
                    state.legal_actions_mask(cur_player),
                    reach_prob=actor_reach,
                )

        # value function predicts value for player 0
        if train_value or record_value:
            # if op_prob = 0 then we are doing importance weighted sampling
            # if op_prob > 0 then we need to wait until expl = 0 to get pure on-policy rollouts
            if on_policy_prob == 0 or expl == 0:
                hist_state = np.append(state.information_state_tensor(0), state.information_state_tensor(1))

                assert player == 0
                if self._val_bootstrap:
                    if self._all_actions:
                        target = policy @ child_values
                    else:
                        target = child_value * policy[sampled_action] / sample_policy[sampled_action]
                elif self._debug_val:
                    target = child_value * policy[sampled_action] / sample_policy[sampled_action]
                    print(target, 'value target')
                else:
                    target = iw_sampled_value
                if val_test:
                    self._value_memory_test.add(
                        self._serialize_value_memory(hist_state, self._iteration, target,
                                                     state.legal_actions_mask(cur_player)))
                else:
                    self._value_memory.add(
                        self._serialize_value_memory(hist_state, self._iteration, target,
                                                     state.legal_actions_mask(cur_player)))

        return importance_weighted_sampled_value, sampled_value

    def _traverse_game_tree_joint_on_policy(
        self,
        state,
        track_mean_squares=False,
        last_action=0,
    ):
        """Sample one joint-policy trajectory and train the acting player at each node."""
        self._nodes_visited += 1

        if state.is_terminal():
            return

        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = outcomes[np.random.choice(range(len(outcomes)), p=probs)]
            self._traverse_game_tree_joint_on_policy(
                state.child(action),
                track_mean_squares=track_mean_squares,
                last_action=action,
            )
            return

        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        num_actions = state.num_distinct_actions()
        legal_mask = np.asarray(
            state.legal_actions_mask(cur_player),
            dtype=np.float64,
        )

        _, policy = self._sample_action_from_regret(state, cur_player)
        policy = np.asarray(policy, dtype=np.float64) * legal_mask
        if policy.sum() <= 0.0 or not np.isfinite(policy.sum()):
            policy = legal_mask / legal_mask.sum()
        else:
            policy = policy / policy.sum()

        sampled_action = np.random.choice(range(num_actions), p=policy)
        value_estimate = self._estimate_value_from_hist(
            state.clone(),
            cur_player,
            last_action=last_action,
        )
        child_values = np.zeros(num_actions, dtype=np.float64)
        if self._all_actions:
            for action in legal_actions:
                child_values[action] = self._estimate_value_from_hist(
                    state.clone().child(action),
                    cur_player,
                    last_action=action,
                )
        else:
            child_values[sampled_action] = self._estimate_value_from_hist(
                state.clone().child(sampled_action),
                cur_player,
                last_action=sampled_action,
            )

        if track_mean_squares:
            oracle_value = self._exact_value(state.clone(), cur_player)
            oracle_child_value = self._exact_value(
                state.clone().child(sampled_action),
                cur_player,
            )
            self._squared_errors.append((oracle_value - value_estimate) ** 2)
            self._squared_errors_child.append(
                (oracle_child_value - child_values[sampled_action]) ** 2
            )

        target_result = compute_regret_target(
            child_values,
            value_estimate,
            policy,
            legal_mask,
            baseline_mode=self._regret_target_baseline,
        )
        sampled_regret = target_result.target
        self._record_regret_target_consistency(
            cur_player,
            target_result,
            legal_mask,
        )
        info_state = state.information_state_tensor(cur_player)
        self._add_to_regret_memory(
            cur_player,
            info_state,
            self._iteration,
            sampled_regret,
            state.legal_actions_mask(cur_player),
        )
        self._add_to_average_policy_memory(
            info_state,
            self._iteration,
            policy,
            state.legal_actions_mask(cur_player),
        )

        self._traverse_game_tree_joint_on_policy(
            state.child(sampled_action),
            track_mean_squares=track_mean_squares,
            last_action=sampled_action,
        )

    @tf.function
    def _init_main_regret_network(self, info_state, legal_actions_mask, player):
        """TF-Graph to calculate regret matching."""
        regrets = self._regret_networks[player](
            (tf.expand_dims(info_state, axis=0), legal_actions_mask),
            training=False)[0]

    @tf.function
    def _init_main_val_network(self, hist_state, legal_actions_mask):
        """TF-Graph to calculate regret matching."""
        estimated_val = \
            self._val_network((tf.expand_dims(hist_state, axis=0), legal_actions_mask), training=False)[0]

    @tf.function
    def _get_matched_regrets(self, info_state, legal_actions_mask, player):
        """TF-Graph to calculate regret matching."""
        regrets = self._regret_networks[player](
            (tf.expand_dims(info_state, axis=0), legal_actions_mask),
            training=False)[0]

        regrets = tf.maximum(regrets, 0)
        summed_regret = tf.reduce_sum(regrets)
        if summed_regret > 0:
            matched_regrets = regrets / summed_regret
        elif self._use_uniform_zero_regret_fallback:
            matched_regrets = legal_actions_mask / tf.reduce_sum(legal_actions_mask)
        else:
            matched_regrets = tf.one_hot(
                tf.argmax(tf.where(legal_actions_mask == 1, regrets, -10e20)),
                self._num_actions)
        return regrets, matched_regrets

    @tf.function
    def _get_estimated_value(self, hist_state, legal_actions_mask):
        """TF-Graph to calculate regret matching."""
        estimated_val = \
            self._val_network((tf.expand_dims(hist_state, axis=0), legal_actions_mask), training=False)[0]
        return estimated_val

    def _sample_action_from_regret(self, state, player):
        """Returns an info state policy by applying regret-matching.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index over which to compute regrets.

        Returns:
          1. (np-array) regret values for info state actions indexed by action.
          2. (np-array) Matched regrets, prob for actions indexed by action.
        """
        info_state = tf.constant(
            state.information_state_tensor(player), dtype=tf.float32)
            
        legal_actions_mask = tf.constant(
            state.legal_actions_mask(player), dtype=tf.float32)
        self._example_info_state[player] = info_state
        self._example_legal_actions_mask[player] = legal_actions_mask
        regrets, matched_regrets = self._get_matched_regrets(
            info_state, legal_actions_mask, player)
        return regrets.numpy(), matched_regrets.numpy()

    def _estimate_value_from_hist(self, state, player, last_action=0):
        """Returns an info state policy by applying regret-matching.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index over which to compute regrets.

        Returns:
          1. (np-array) regret values for info state actions indexed by action.
          2. (np-array) Matched regrets, prob for actions indexed by action.
        """
        state = state.clone()
        if state.is_terminal():
            return state.player_return(player)

        hist_state = np.append(state.information_state_tensor(0), state.information_state_tensor(1))

        self._example_hist_state = hist_state
        hist_state = tf.constant(hist_state, dtype=tf.float32)
        legal_actions_mask = tf.constant(
            state.legal_actions_mask(player), dtype=tf.float32)
        estimated_value = self._get_estimated_value(hist_state, legal_actions_mask)
        if player == 1:
            estimated_value = -estimated_value
        return estimated_value.numpy()

    def action_probabilities(self, state):
        """Returns a valid probability distribution over legal actions."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        legal_actions_mask = tf.constant(
            state.legal_actions_mask(cur_player), dtype=tf.float32)
        info_state_vector = tf.constant(
            state.information_state_tensor(cur_player), dtype=tf.float32)
        if len(info_state_vector.shape) == 1:
            info_state_vector = tf.expand_dims(info_state_vector, axis=0)
        probs = self._policy_network((info_state_vector, legal_actions_mask),
                                     training=False).numpy()[0]
        legal_probs = {action: float(probs[action]) for action in legal_actions}
        total = float(sum(legal_probs.values()))
        if (not np.isfinite(total)) or total <= 0.0:
            uniform = 1.0 / len(legal_actions)
            return {action: uniform for action in legal_actions}
        return {action: prob / total for action, prob in legal_probs.items()}

    def _get_regret_dataset(self, player):
        """Returns the collected regrets for the given player as a dataset."""
        if self._save_regret_memories:
            files = sorted(glob.glob(self._regret_tfrecord_pattern(player)))
            if not files:
                raise RuntimeError(f"No regret TFRecord shards found for player {player}.")
            data = tf.data.TFRecordDataset(
                files,
                compression_type=(
                    self._tfrecord_compression if self._tfrecord_compression else None
                ),
                num_parallel_reads=tf.data.experimental.AUTOTUNE,
            )
        else:
            stored_data = self.get_regret_memories(player)
            data = tf.data.Dataset.from_tensor_slices(stored_data)
            # A capped Experiment 28 reservoir is already smaller than the
            # standard shuffle window. The uncapped arm eventually is not, so
            # use its full population to avoid an insertion-order bias that
            # would exclude the newest samples from a fixed training budget.
            shuffle_size = max(REGRET_TRAIN_SHUFFLE_SIZE, len(stored_data))
        if self._save_regret_memories:
            shuffle_size = REGRET_TRAIN_SHUFFLE_SIZE
        data = data.shuffle(shuffle_size)
        data = data.repeat()
        data = data.batch(self._batch_size_regret)
        data = data.map(self._deserialize_regret_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    def _get_value_dataset(self):
        """Returns the collected value estimates for the given player as a dataset."""
        data = self.get_value_memory()
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.shuffle(VALUE_TRAIN_SHUFFLE_SIZE)
        data = data.repeat()
        data = data.batch(self._batch_size_value)
        data = data.map(self._deserialize_value_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    def _get_value_dataset_test(self):
        """Returns the collected value estimates for the given player as a dataset."""
        data = self.get_value_memory_test()
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.shuffle(VALUE_TRAIN_SHUFFLE_SIZE)
        data = data.repeat()
        data = data.batch(self._batch_size_value)
        data = data.map(self._deserialize_value_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    def _get_value_test_loss(self):
        with tf.device(self._train_device):
            tfit = tf.constant(self._iteration, dtype=tf.float32)
            data = self._get_value_dataset_test()
            for d in data.take(1):
                main_loss = self._value_test_step(*d, tfit)
                if self._debug_val:
                    print(main_loss, 'test loss')
        return main_loss

    def _get_regret_train_graph(self, player):
        """Return TF-Graph to perform regret network train step."""
        processing = self._regret_target_processing
        clip_value = tf.constant(self._regret_target_clip_value, dtype=tf.float32)
        standardize_epsilon = tf.constant(
            self._regret_target_standardize_epsilon,
            dtype=tf.float32,
        )
        fixed_scale = tf.constant(
            self._regret_target_fixed_scale,
            dtype=tf.float32,
        )
        ema_decay = tf.constant(
            self._regret_target_ema_decay,
            dtype=tf.float32,
        )
        ema_mean = self._regret_target_ema_mean[player]
        ema_second_moment = self._regret_target_ema_second_moment[player]
        ema_initialized = self._regret_target_ema_initialized[player]

        @tf.function
        def train_step(info_states, regrets, iterations, masks, iteration):
            model = self._regret_networks_train[player]
            target_mask = tf.cast(masks, tf.float32)
            raw_regrets = tf.cast(regrets, tf.float32)
            raw_regrets = tf.where(
                target_mask > 0.0,
                raw_regrets,
                tf.zeros_like(raw_regrets),
            )
            processed_regrets = raw_regrets
            legal_regrets = tf.boolean_mask(raw_regrets, target_mask > 0.0)

            def safe_mean(values):
                return tf.cond(
                    tf.size(values) > 0,
                    lambda: tf.reduce_mean(values),
                    lambda: tf.constant(0.0, dtype=tf.float32),
                )

            def safe_variance(values):
                return tf.cond(
                    tf.size(values) > 0,
                    lambda: tf.math.reduce_variance(values),
                    lambda: tf.constant(0.0, dtype=tf.float32),
                )

            raw_variance = safe_variance(legal_regrets)
            standardization_mean = tf.constant(0.0, dtype=tf.float32)
            standardization_scale = tf.constant(1.0, dtype=tf.float32)
            if processing in {BATCH_STANDARDIZE, BATCH_STANDARDIZE_CLIP}:
                standardization_mean = safe_mean(legal_regrets)
                standardization_scale = tf.maximum(
                    tf.sqrt(safe_variance(legal_regrets)),
                    standardize_epsilon,
                )
                processed_regrets = (
                    processed_regrets - standardization_mean
                ) / standardization_scale
                processed_regrets = tf.where(
                    target_mask > 0.0,
                    processed_regrets,
                    tf.zeros_like(processed_regrets),
                )
            elif processing == FIXED_UTILITY_SCALE:
                standardization_scale = fixed_scale
                processed_regrets = tf.where(
                    target_mask > 0.0,
                    raw_regrets / standardization_scale,
                    tf.zeros_like(raw_regrets),
                )
            elif processing == BATCH_RMS:
                standardization_scale = tf.maximum(
                    tf.sqrt(safe_mean(tf.square(legal_regrets))),
                    standardize_epsilon,
                )
                processed_regrets = tf.where(
                    target_mask > 0.0,
                    raw_regrets / standardization_scale,
                    tf.zeros_like(raw_regrets),
                )
            elif processing == EMA_STD:
                batch_mean = safe_mean(legal_regrets)
                batch_second_moment = safe_mean(tf.square(legal_regrets))
                updated_mean = tf.cond(
                    ema_initialized,
                    lambda: (
                        ema_decay * ema_mean
                        + (1.0 - ema_decay) * batch_mean
                    ),
                    lambda: batch_mean,
                )
                updated_second_moment = tf.cond(
                    ema_initialized,
                    lambda: (
                        ema_decay * ema_second_moment
                        + (1.0 - ema_decay) * batch_second_moment
                    ),
                    lambda: batch_second_moment,
                )
                ema_mean.assign(updated_mean)
                ema_second_moment.assign(updated_second_moment)
                ema_initialized.assign(True)
                persistent_variance = tf.maximum(
                    updated_second_moment - tf.square(updated_mean),
                    tf.square(standardize_epsilon),
                )
                standardization_scale = tf.sqrt(persistent_variance)
                processed_regrets = tf.where(
                    target_mask > 0.0,
                    raw_regrets / standardization_scale,
                    tf.zeros_like(raw_regrets),
                )

            clip_fraction = tf.constant(0.0, dtype=tf.float32)
            if processing in {CLIP, BATCH_STANDARDIZE_CLIP}:
                before_clip = processed_regrets
                processed_regrets = tf.clip_by_value(
                    processed_regrets,
                    -clip_value,
                    clip_value,
                )
                processed_regrets = tf.where(
                    target_mask > 0.0,
                    processed_regrets,
                    tf.zeros_like(processed_regrets),
                )
                changed = tf.logical_and(
                    target_mask > 0.0,
                    tf.not_equal(before_clip, processed_regrets),
                )
                legal_count = tf.maximum(
                    tf.reduce_sum(target_mask),
                    tf.constant(1.0, dtype=tf.float32),
                )
                clip_fraction = (
                    tf.reduce_sum(tf.cast(changed, tf.float32)) / legal_count
                )

            processed_legal_regrets = tf.boolean_mask(
                processed_regrets,
                target_mask > 0.0,
            )
            processed_variance = safe_variance(processed_legal_regrets)
            processed_abs_mean = safe_mean(tf.abs(processed_legal_regrets))
            legal_count = tf.maximum(
                tf.reduce_sum(target_mask),
                tf.constant(1.0, dtype=tf.float32),
            )
            raw_positive_fraction = (
                tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(target_mask > 0.0, raw_regrets > 0.0),
                        tf.float32,
                    )
                )
                / legal_count
            )
            processed_positive_fraction = (
                tf.reduce_sum(
                    tf.cast(
                        tf.logical_and(
                            target_mask > 0.0,
                            processed_regrets > 0.0,
                        ),
                        tf.float32,
                    )
                )
                / legal_count
            )
            sign_changed = tf.logical_and(
                target_mask > 0.0,
                tf.not_equal(tf.sign(raw_regrets), tf.sign(processed_regrets)),
            )
            sign_flip_fraction = (
                tf.reduce_sum(tf.cast(sign_changed, tf.float32)) / legal_count
            )
            with tf.GradientTape() as tape:
                preds = model((info_states, masks), training=True)
                main_loss = self._loss_regrets[player](
                    processed_regrets,
                    preds,
                    sample_weight=iterations * 2 / iteration,
                )
                loss = tf.add_n([main_loss], model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            self._optimizer_regrets[player].apply_gradients(
                zip(gradients, model.trainable_variables))

            return (
                main_loss,
                raw_variance,
                processed_variance,
                processed_abs_mean,
                standardization_mean,
                standardization_scale,
                clip_fraction,
                sign_flip_fraction,
                raw_positive_fraction,
                processed_positive_fraction,
            )

        return train_step

    def _get_value_train_graph(self):
        """Return TF-Graph to perform value network train step."""

        @tf.function
        def train_step(full_hist_states, values, iterations, masks, iteration):
            model = self._val_network_train
            with tf.GradientTape() as tape:
                preds = model((full_hist_states, masks), training=True)
                main_loss = self._loss_value(
                    values, preds, sample_weight=1)
                loss = tf.add_n([main_loss], model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            self._optimizer_value.apply_gradients(
                zip(gradients, model.trainable_variables))
            return main_loss

        return train_step

    def _get_value_test_graph(self):
        """Return TF-Graph to perform value network train step."""

        @tf.function
        def test_step(full_hist_states, values, iterations, masks, iteration):
            model = self._val_network
            with tf.GradientTape() as tape:
                preds = model((full_hist_states, masks), training=True)
                main_loss = self._loss_value(
                    values, preds, sample_weight=1)
                loss = tf.add_n([main_loss], model.losses)
            return main_loss

        return test_step

    def _learn_value_network(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Args:
          player: (int) player index.

        Returns:
          The average loss over the regret network of the last batch.
        """

        with tf.device(self._train_device):
            tfit = tf.constant(self._iteration, dtype=tf.float32)
            data = self._get_value_dataset()
            for d in data.take(self._value_network_train_steps):
                main_loss = self._value_train_step(*d, tfit)
                if self._debug_val:
                    print(main_loss, 'main val loss')

        self._val_network.set_weights(
            self._val_network_train.get_weights())
        return main_loss

    def _learn_regret_network(self, player):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Args:
          player: (int) player index.

        Returns:
          The average loss over the regret network of the last batch.
        """
        def _to_float(value):
            try:
                return float(value.numpy())
            except Exception:
                return float(np.asarray(value))

        sign_flip_fractions = []
        raw_positive_fractions = []
        processed_positive_fractions = []
        with tf.device(self._train_device):
            tfit = tf.constant(self._iteration, dtype=tf.float32)
            data = self._get_regret_dataset(player)
            for d in data.take(self._regret_network_train_steps):
                (
                    main_loss,
                    raw_variance,
                    processed_variance,
                    processed_abs_mean,
                    standardization_mean,
                    standardization_scale,
                    clip_fraction,
                    sign_flip_fraction,
                    raw_positive_fraction,
                    processed_positive_fraction,
                ) = self._regret_train_step[player](*d, tfit)
                self._last_raw_regret_target_variance[player] = _to_float(
                    raw_variance
                )
                self._last_processed_regret_target_variance[player] = _to_float(
                    processed_variance
                )
                self._last_processed_regret_target_abs_mean[player] = _to_float(
                    processed_abs_mean
                )
                self._last_regret_target_standardization_mean[player] = _to_float(
                    standardization_mean
                )
                self._last_regret_target_standardization_scale[player] = _to_float(
                    standardization_scale
                )
                self._last_regret_target_clip_fraction[player] = _to_float(
                    clip_fraction
                )
                sign_flip_fractions.append(_to_float(sign_flip_fraction))
                raw_positive_fractions.append(_to_float(raw_positive_fraction))
                processed_positive_fractions.append(
                    _to_float(processed_positive_fraction)
                )

        if sign_flip_fractions:
            self._last_regret_target_sign_flip_fraction[player] = float(
                np.mean(sign_flip_fractions)
            )
            self._last_raw_regret_target_positive_fraction[player] = float(
                np.mean(raw_positive_fractions)
            )
            self._last_processed_regret_target_positive_fraction[player] = float(
                np.mean(processed_positive_fractions)
            )

        self._regret_networks[player].set_weights(
            self._regret_networks_train[player].get_weights())
        return main_loss

    def _get_average_policy_dataset(self):
        """Returns the collected average_policy memories as a dataset."""
        if self._average_policy_tfrecord_dir:
            files = sorted(glob.glob(self._average_policy_tfrecord_pattern()))
            if not files:
                raise RuntimeError("No average-policy TFRecord shards found.")
            data = tf.data.TFRecordDataset(files)
        elif self._memories_tfrecordpath:
            data = tf.data.TFRecordDataset(self._memories_tfrecordpath)
        else:
            data = self.get_average_policy_memories()
            data = tf.data.Dataset.from_tensor_slices(data)
        data = data.shuffle(AVERAGE_POLICY_TRAIN_SHUFFLE_SIZE)
        data = data.repeat()
        data = data.batch(self._batch_size_average_policy)
        data = data.map(self._deserialize_average_policy_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        return data

    def _learn_average_policy_network(self):
        """Compute the loss over the average_policy network.

        Returns:
          The average loss obtained on the last training batch of transitions
          or `None`.
        """

        @tf.function
        def train_step(info_states, action_probs, iterations, masks, reach_probs, obs_indices):
            model = self._policy_network
            del obs_indices
            with tf.GradientTape() as tape:
                preds = model((info_states, masks), training=True)
                if self._average_policy_weighting == "linear":
                    base_weights = tf.squeeze(iterations, axis=-1) * (
                        2.0 / tf.cast(self._iteration, tf.float32)
                    )
                else:
                    base_weights = tf.ones_like(
                        tf.squeeze(iterations, axis=-1),
                        dtype=tf.float32,
                    )
                if self._use_reach_weighted_avg_policy_loss:
                    reach_weights = tf.squeeze(reach_probs, axis=-1)
                    reach_weights = tf.clip_by_value(reach_weights, 1e-8, 1.0)
                    reach_weights = reach_weights / (
                        tf.reduce_mean(reach_weights) + 1e-8
                    )
                    sample_weight = base_weights * reach_weights
                else:
                    sample_weight = base_weights
                main_loss = self._loss_policy(
                    action_probs,
                    preds,
                    sample_weight=sample_weight,
                )
                loss = tf.add_n([main_loss], model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            self._optimizer_policy.apply_gradients(
                zip(gradients, model.trainable_variables))
            return main_loss

        with tf.device(self._train_device):
            data = self._get_average_policy_dataset()
            for d in data.take(self._policy_network_train_steps):
                main_loss = train_step(*d)

        return main_loss
    
    def extract_full_model(self):
        """Extract a fully-serializable snapshot of the solver state.

        The returned dict is designed to be pickled and later fed to
        `load_full_model(...)` to resume training from the same weights and
        replay buffers.
        """
        ckpt = {
            # Training progress
            "iteration": int(getattr(self, "_iteration", 0)),

            # Network weights
            "policy_weights": self.get_policy_weights(),
            "regret_weights": self.get_weights(),      # list per player
            "value_weights": self.get_val_weights(),

            # Replay buffers (e.g., serialized tf.train.Example bytes)
            "avg_policy_data": list(self.get_average_policy_memories()),
            "regret_data": [list(self.get_regret_memories(p)) for p in range(self._num_players)],
            "value_data": list(self.get_value_memory()),
            "value_test_data": list(self.get_value_memory_test()),

            # Reservoir counters (so sampling continues consistently)
            "avg_policy_add_calls": int(self._average_policy_memories.get_num_calls()),
            "regret_add_calls": [int(self._regret_memories[p].get_num_calls()) for p in range(self._num_players)],
            "value_add_calls": int(self._value_memory.get_num_calls()),
            "value_test_add_calls": int(self._value_memory_test.get_num_calls()),
            "regret_replay_state": [
                self._regret_memories[p].state_dict()
                for p in range(self._num_players)
            ],
            "sampling_coverage_counts": [
                dict(counts) for counts in self._sampling_coverage_counts
            ],
            "sampling_reach_stats": [
                dict(stats) for stats in self._sampling_reach_stats
            ],

            # Persistent scale-only target-processing state.
            "regret_target_ema_mean": [
                float(value.numpy()) for value in self._regret_target_ema_mean
            ],
            "regret_target_ema_second_moment": [
                float(value.numpy())
                for value in self._regret_target_ema_second_moment
            ],
            "regret_target_ema_initialized": [
                bool(value.numpy())
                for value in self._regret_target_ema_initialized
            ],

            # Light metadata for sanity checks / debugging
            "meta": {
                "num_players": int(self._num_players),
                "num_actions": int(self._num_actions),
                "regret_network_output_mode": self._regret_network_output_mode,
                "regret_target_baseline": self._regret_target_baseline,
                "regret_target_processing": self._regret_target_processing,
                "regret_target_clip_value": float(self._regret_target_clip_value),
                "regret_target_standardize_epsilon": float(
                    self._regret_target_standardize_epsilon
                ),
                "regret_target_fixed_scale": float(
                    self._regret_target_fixed_scale
                ),
                "regret_target_ema_decay": float(self._regret_target_ema_decay),
                "regret_replay_mode": self._regret_replay_mode,
                "regret_replay_rare_history_quota": int(
                    self._regret_replay_rare_history_quota
                ),
                "regret_replay_weight_floor": float(
                    self._regret_replay_weight_floor
                ),
                "use_balanced_probs": bool(self._use_balanced_probs),
                "balanced_sampling_mix": float(self._balanced_sampling_mix),
                "track_sampling_coverage": bool(self._track_sampling_coverage),
                "average_policy_weighting": self._average_policy_weighting,
            },
        }
        return ckpt
    
    def load_full_model(self, ckpt):
        """Restore a snapshot produced by `extract_full_model`.

        Args:
          ckpt: dict returned by `extract_full_model` (or loaded from pickle).

        Returns:
          self (for chaining).
        """
        # Ensure Keras models have created variables before calling set_weights.
        # (Keras models sometimes have zero variables until they are called once.)
        def _ensure_built(model, x_dim, mask_dim):
            try:
                if len(getattr(model, "weights", [])) == 0:
                    dummy_x = tf.zeros((1, x_dim), dtype=tf.float32)
                    dummy_m = tf.ones((1, mask_dim), dtype=tf.float32)
                    _ = model((dummy_x, dummy_m), training=False)
            except Exception:
                # If building fails for any reason, we'll let set_weights raise.
                pass

        # Policy + value nets
        _ensure_built(self._policy_network, self._embedding_size, self._num_actions)
        _ensure_built(self._val_network, self._value_embedding_size, self._num_actions)
        if hasattr(self, "_val_network_train"):
            _ensure_built(self._val_network_train, self._value_embedding_size, self._num_actions)

        # Regret nets (infer + train)
        for p in range(getattr(self, "_num_players", 0)):
            _ensure_built(self._regret_networks[p], self._embedding_size, self._num_actions)
            if hasattr(self, "_regret_networks_train") and len(self._regret_networks_train) > p:
                _ensure_built(self._regret_networks_train[p], self._embedding_size, self._num_actions)

        # ----- restore iteration counter -----
        if "iteration" in ckpt:
            self.set_iteration(int(ckpt["iteration"]))

        # ----- restore weights (IMPORTANT: set both infer + train copies) -----
        if "regret_weights" in ckpt:
            rw = ckpt["regret_weights"]
            for p in range(min(self._num_players, len(rw))):
                self._regret_networks[p].set_weights(rw[p])
                if hasattr(self, "_regret_networks_train") and len(self._regret_networks_train) > p:
                    self._regret_networks_train[p].set_weights(rw[p])

        if "value_weights" in ckpt:
            self._val_network.set_weights(ckpt["value_weights"])
            if hasattr(self, "_val_network_train"):
                self._val_network_train.set_weights(ckpt["value_weights"])

        if "policy_weights" in ckpt:
            self._policy_network.set_weights(ckpt["policy_weights"])

        # ----- restore buffers (serialized tf.train.Example bytes) -----
        if "avg_policy_data" in ckpt:
            self._average_policy_memories._data = list(ckpt["avg_policy_data"])
            self._average_policy_memories._keys = [
                None for _ in self._average_policy_memories._data
            ]
            if "avg_policy_add_calls" in ckpt:
                self._average_policy_memories._add_calls = int(ckpt["avg_policy_add_calls"])

        if "regret_replay_state" in ckpt:
            for p, state in enumerate(ckpt["regret_replay_state"]):
                if p < self._num_players:
                    self._regret_memories[p].load_state_dict(state)
        elif "regret_data" in ckpt:
            for p in range(min(self._num_players, len(ckpt["regret_data"]))):
                state = {
                    "data": list(ckpt["regret_data"][p]),
                    "add_calls": (
                        int(ckpt["regret_add_calls"][p])
                        if "regret_add_calls" in ckpt
                        and p < len(ckpt["regret_add_calls"])
                        else len(ckpt["regret_data"][p])
                    ),
                }
                if self._regret_replay_mode == RESERVOIR:
                    self._regret_memories[p].load_state_dict(state)
                else:
                    self._regret_memories[p].clear()
                    for element in state["data"]:
                        self._regret_memories[p].add(element)

        if "value_data" in ckpt:
            self._value_memory._data = list(ckpt["value_data"])
            self._value_memory._keys = [None for _ in self._value_memory._data]
            if "value_add_calls" in ckpt:
                self._value_memory._add_calls = int(ckpt["value_add_calls"])

        if "value_test_data" in ckpt:
            self._value_memory_test._data = list(ckpt["value_test_data"])
            self._value_memory_test._keys = [
                None for _ in self._value_memory_test._data
            ]
            if "value_test_add_calls" in ckpt:
                self._value_memory_test._add_calls = int(ckpt["value_test_add_calls"])

        if "sampling_coverage_counts" in ckpt:
            self._sampling_coverage_counts = [
                collections.Counter(counts)
                for counts in ckpt["sampling_coverage_counts"]
            ]
        if "sampling_reach_stats" in ckpt:
            self._sampling_reach_stats = [
                dict(stats) for stats in ckpt["sampling_reach_stats"]
            ]

        for key, variables in [
            ("regret_target_ema_mean", self._regret_target_ema_mean),
            (
                "regret_target_ema_second_moment",
                self._regret_target_ema_second_moment,
            ),
            ("regret_target_ema_initialized", self._regret_target_ema_initialized),
        ]:
            for variable, value in zip(variables, ckpt.get(key, [])):
                variable.assign(value)

        return self
