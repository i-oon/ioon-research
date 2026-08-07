'''
This module defines a CoppeliaSim environment class that interfaces with the CoppeliaSim simulator using ZeroMQ.
It provides functionalities for normalizing and denormalizing observations and actions, controlling the robot's joints,
and retrieving simulation data such as joint angles, body position, orientation, forces, foot trajectory, and contact states.
Applicable for: expert/expert_66k_aug3c_fcontact.csv etc.
'''


import zmq
import msgpack
import numpy as np
import time
from collections import deque
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class ObsField:
    name: str               # The name of this field
    size: int               # The dimension of this field
    getter: str             # Get the function name in the environment class
    norm: str               # Normalization method: 'unified', 'separate', 'none'
    low: np.ndarray | float  #  The lower bound of this field (can be scalar or np array; binary can be None)
    high: np.ndarray | float  # The upper bound of this field (can be scalar or np array; binary can be None)
    include: bool = True    # whether to include this field in the observation

class CoppeliaSimEnv:

    _max_episode_steps = 1000
    _step_count = 0

    __leg_names = ['_FL','_ML','_HL','_FR','_MR','_HR']
    __joint_names = ['/m1', '/m2', '/m3']  # ThC, CTr, FTi
    __foot_names = ['/foot_FL', '/foot_ML', '/foot_HL', 
                    '/foot_FR', '/foot_MR', '/foot_HR']
    __IMU_names = ['/IMU_robot', '/IMU_ref']
    __forcesensor_names = ['/forceSensor_FL', '/forceSensor_ML', '/forceSensor_HL', 
                           '/forceSensor_FR', '/forceSensor_MR', '/forceSensor_HR']

    # ---- diagnostic switches (see reset()) -------------------------------------------
    # FIX_RESET_OBS_NOISE : correct the author's np.random.normal(-0.1, 0.1) reset-noise typo
    #                       (mean/std misread as low/high -> a -0.1 bias on every reset obs).
    # HARD_RESET_ALWAYS   : do a real stop/start on every reset instead of our teleport-in-place.
    # The reset-observation typo is fixed by default for new training. HARD_RESET_ALWAYS remains
    # off because it is a slow diagnostic mode.
    FIX_RESET_OBS_NOISE = True
    HARD_RESET_ALWAYS = False
    # RESUME_IF_PAUSED: recover a scene that errors -> CoppeliaSim "pause on script error" (e.g. the
    # uneven scene's simUI/missing-file init errors freeze it at step ~2). Default OFF (checking state
    # every step adds a ZMQ round-trip -> slows training); eval scripts flip it True. No effect on
    # scenes that never pause.
    RESUME_IF_PAUSED = False

    # INCLUDE_BODY_Z : True -> 28-dim state (Phase-A baseline). False -> 27-dim (thesis ยง4.2.5:
    # the real robot cannot sense body height without MoCap, so body_z was dropped and the stick
    # insect RETRAINED at 27-dim before cross-dynamics transfer to RedMirror).
    # MUST match expert.INCLUDE_BODY_Z, or expert states and env states differ in width.
    INCLUDE_BODY_Z = True

    __joint_handle = np.zeros((6, 3), dtype=int).astype(int)  # joint handle (leg l, joint j)
    __target_positions = np.zeros((6, 3), dtype=float).astype(float)  # joint target position (leg l, joint j)
    __initjoint_position = np.zeros((6, 3), dtype=float).astype(float)  # initial joint position (leg l, joint j)
    __init_pos_deg = np.array([[30, 9.5, -60], 
                                                        [ 0 ,  -2.5, -60],
                                                        [-40, 9.5,-60],
                                                        [30, 9.5, -60], 
                                                        [0, -2.5, -60],
                                                        [-40, 9.5, -60]], dtype=float).astype(float)  # initial joint position in degrees
    __init_pos_dirction = np.array([[-1, 1, 1],
                                                            [-1, 1, 1],
                                                            [-1, 1, 1],
                                                            [1, -1, -1],
                                                            [1, -1, -1],
                                                            [1, -1, -1]])  
    __init_pos_deg = __init_pos_deg * __init_pos_dirction # adjust the initial position direction
    __init_pos_rad = np.deg2rad(__init_pos_deg)  # initial joint position in radians
    __initjoint_position = __init_pos_rad

    OBS_SPEC: tuple[ObsField, ...] = (

                    ObsField('body_pos',      1,  'get_bodyposition',   'per_dim',
                                low=np.array([0.19342157]),  # np.array([-1.5743479, -0.13103247, 0.19342157]), 
                                high=np.array([0.2718982]),  # np.array([-0.01882718, 0.49992156, 0.2718982]), 
                                include=True), # True, False

                    ObsField('orientation',   3,  'get_bodyorientation','shared',
                                low=min([-0.1253066, -0.21079601, -0.14037536]),  
                                high=max([0.17421827, 0.03616637, 0.56608814]),
                                include=True),

                    # ObsField('orientation',   3,  'get_bodyorientation','shared',
                    #             low=min([-0.5, -0.5, -1.5]),  
                    #             high=max([0.5, 0.5, 1.5]),
                    #             include=True),               

                    ObsField('joint_angles', 18, 'get_jointangle', 'per_dim',
                                low=np.array([
                                    -1.3860602,  0.06034265, -2.4969175, 
                                    -0.9650939, -0.0351965 , -2.3240883,  
                                    0.28500196, 0.15376441, -2.507828, 
                                    0.5072578, -0.7540891 ,  0.5758628,  
                                    -0.6832747, -0.6967423 ,  0.64540935, 
                                    -1.2671623, -1.0010364 ,  0.58814275  
                                ]),
                                high=np.array([
                                    -0.5072578 ,  0.7540891 , -0.5758628, 
                                    0.6832747 ,  0.6967423 , -0.64540935, 
                                    1.2671623 ,  1.0010364 , -0.58814275, 
                                    1.3860602 , -0.06034265,  2.4969175,  
                                    0.9650939 ,  0.0351965 ,  2.3240883, 
                                    -0.28500196, -0.15376441,  2.507828    
                                ]),
                                include=True
                            ), 

                    ObsField('qvel_body', 6, 'get_qvel_body', 'shared',
                                low=min([-0.13889849, -0.41793427, -0.7163129, -1.5593499, -1.6832889, -1.5487039]),
                                high=max([0.86493146, 0.5213626, 0.32529727, 1.3493888, 1.0914965, 1.5461688]), 
                                include=False),

                    ObsField('qvel_joints', 18, 'get_qvel_joints', 'per_dim',
                            low=np.array([
                                -6.34015036, -5.73144054, -6.28769588, -6.40328217, -3.32265925, -6.28675842,
                                -6.27953339, -3.38050628, -6.29043961, -1.96826971, -2.68654227, -6.29233313,
                                -6.3885498,  -6.2946496,  -6.28978157, -4.32833147, -6.36295366, -4.85788155
                            ]),
                            high=np.array([
                                1.96826971, 2.68654227, 6.29233313, 6.3885498,  6.2946496,  6.28978157,
                                4.32833147, 6.36295366, 4.85788155, 6.34015036, 5.73144054, 6.28769588,
                                6.40328217, 3.32265925, 6.28675842, 6.27953339, 3.38050628, 6.29043961
                            ]),
                            include=False),

                    ObsField('forces',        6,  'get_force',          'shared',
                                low=0.0,
                                high=max([11.871004, 22.840376, 20.059353, 14.028709, 28.488878, 13.580413]),
                                include=False),  

                    ObsField('foot_traj',     6,  'get_foot_trajectory','shared',
                                low=min([0.00600649, 0.00653866, 0.00605668, 0.00703868, 0.00606308, 0.00643684]), 
                                high=max([0.30520386, 0.16491821, 0.10601486, 0.29555720, 0.08111896, 0.12053363]),
                                include=False),   
                            
                    ObsField('contact',       6,  'get_contact',        'binary',
                                low=None, high=None, 
                                include=True))

    # ------------------------------------------------------------------------------------
    # Action bounds recomputed for OUR expert file (expert/expert_66k_fcontact.csv) via
    # scripts/recompute_bounds.py, which applies the author's own expert.symmetric_lr_bounds()
    # to the actions returned by expert.load_expert_data() (i.e. offsets from the default pose).
    #
    # Why: the original literals were calibrated on the author's expert_66k_aug3c_fcontact.csv.
    # Against our regenerated data they were slightly too tight -> 0.87% of expert action values
    # (14.1% of frames) normalised outside [-1, 1]. The actor is tanh-squashed to [-1, 1], so
    # those expert actions are unreachable: log_pi(expert) blows up negative, which hands the
    # discriminator a trivial cue (contributes to acc_exp=1.0, logit_exp~+497) instead of it
    # judging gait quality. These bounds are the author's, widened only where our data exceeds
    # them, plus a uniform 0.02 margin -> 0.000% outside. Verify with scripts/recompute_bounds.py.
    #
    # These bounds MUST match expert.ACTION_SOURCE. They are currently calibrated for
    # ACTION_SOURCE = 'motor_pos' (achieved angles). If you set it back to 'motor_cmd',
    # swap in the motor_cmd block below, or the expert actions will fall outside [-1,1] again.
    #
    # To revert to the author's original literals, swap the blocks below.
    # ORIGINAL (author, aug3c dataset):
    #   high = [-0.08928384, 0.64018328, 0.73880163,  0.71728384, 0.53050838, 0.52528891,
    #            0.61509333, 0.76640703, 0.46057537,  0.87071587, 0.19994925, 1.43173578,
    #            0.90740824, 0.03776942, 1.30299309,  0.44833541, 0.00427082, 1.59938887]
    #   low  = [-0.87071587, -0.19994925, -1.43173578, -0.90740824, -0.03776942, -1.30299309,
    #           -0.44833541, -0.00427082, -1.59938887,  0.08928384, -0.64018328, -0.73880163,
    #           -0.71728384, -0.53050838, -0.52528891, -0.61509333, -0.76640703, -0.46057537]
    #
    # OURS for ACTION_SOURCE='motor_pos' (run 20260717-0108) โ€” TESTED AND REJECTED: duty_err 0.263
    # vs 0.207 for motor_cmd. See the ACTION_SOURCE comment in scripts/expert.py.
    #   high = [-0.00235992, 0.61483840, 0.45861237,  0.70776691, 0.50068348, 0.41869490,
    #            0.59009373, 1.14656952, 0.35914635,  0.87850932, 0.12236026, 1.43420463,
    #            1.00390025, 0.01027336, 1.24998312,  0.42348392, 0.03866417, 1.48260694]
    #   low  = [-0.87850932, -0.12236026, -1.43420463, -1.00390025, -0.01027336, -1.24998312,
    #           -0.42348392, -0.03866417, -1.48260694,  0.00235992, -0.61483840, -0.45861237,
    #           -0.70776691, -0.50068348, -0.41869490, -0.59009373, -1.14656952, -0.35914635]
    # ------------------------------------------------------------------------------------
    # ACTIVE: the AUTHOR'S ORIGINAL literals โ€” correct for expert_66k_aug3c_fcontact.csv.
    # 2026-07-17: the real aug3c file was obtained. symmetric_lr_bounds(aug3c) reproduces these
    # numbers to 6e-8, i.e. they were computed FROM aug3c and cover it exactly (0.000% outside).
    # They never fit our regenerated file because that file is a DIFFERENT augmentation
    # (30 rollouts x ~2185 frames vs aug3c's 1000 rollouts x 66 frames) -- so the earlier
    # "the author's bounds are buggy" conclusion was WRONG; we were simply using the wrong data.
    # Bounds are tight (zero margin): the extreme actions sit exactly at |a|=1, where atanh
    # diverges. Safe here only because networks/utils.py evaluate_log_pi clamps before atanh.
    # For the regenerated file use scripts/recompute_bounds.py instead (see block above).
    action_space_high = np.array([
                        -0.08928384, 0.64018328, 0.73880163,
                        0.71728384, 0.53050838, 0.52528891,
                        0.61509333, 0.76640703, 0.46057537,
                        0.87071587, 0.19994925, 1.43173578,
                        0.90740824, 0.03776942, 1.30299309,
                        0.44833541, 0.00427082, 1.59938887
                        ])

    action_space_low = np.array([
                        -0.87071587, -0.19994925, -1.43173578,
                        -0.90740824, -0.03776942, -1.30299309,
                        -0.44833541, -0.00427082, -1.59938887,
                        0.08928384, -0.64018328, -0.73880163,
                        -0.71728384, -0.53050838, -0.52528891,
                        -0.61509333, -0.76640703, -0.46057537
                        ])
    
    def __init__(self, port=23000, OnTimeStep=True, simulation = True, vx_coef=100.0,
                 reward_mode="speed", vx_target=0.45, track_sigma=0.15,
                 scale_target_by_leg=True, leg_ref=0.7717,
                 scale_body_z=True, stand_ref=0.254, track_window=25,
                 track_direction_mix=0.5):
        # build observation layout
        self._build_obs_layout()
        # env_reward modes (the AMP total is g(s')[~2/step] + lambda*env_reward, lambda in ppo):
        #  'speed' : vx * vx_coef        -- UNBOUNDED. vx*100 was ~20x the gait prior g, so g
        #                                   barely shaped the gait -> reared sprint. Legacy.
        #  'track' : a bounded blend of directional progress and Gaussian target precision.
        #            The old pure Gaussian was effectively sparse at rest: target/sigma=3 made
        #            r(0)=exp(-9)=0.00012, so PPO received almost no signal to start moving while
        #            g(s') paid 3-5 every step.  The directional half has a finite slope at rest,
        #            penalises reverse travel, and saturates at the target; the Gaussian half keeps
        #            the unique maximum at the commanded speed rather than rewarding a sprint.
        # vx_target is scaled by LEG LENGTH (measured at init): same gait rhythm => stride ∝ leg
        # => speed ∝ leg. So a 0.5x-leg body targets ~0.5x the expert's 0.45 m/s, not 0.45 (which
        # would force it to sprint at 2x its morphology's pace). leg_ref = long/expert leg (0.7717 m).
        self.vx_coef = vx_coef
        self.reward_mode = reward_mode
        self._vx_target_ref = vx_target                 # reference-body (long) target speed
        self.vx_target = vx_target                      # rescaled by leg length in the sim block
        self._track_sigma_ref = track_sigma             # reference tolerance; also scales by leg
        self.track_sigma = track_sigma
        self.track_direction_mix = float(track_direction_mix)
        if not 0.0 <= self.track_direction_mix <= 1.0:
            raise ValueError("track_direction_mix must be in [0, 1]")
        # 'track' reward uses AVERAGE forward velocity over a sliding window of this many steps
        # (net head displacement / elapsed time), NOT the instantaneous head vx. Instantaneous vx
        # is gameable: a body can ROCK back-and-forth so its head momentarily hits vx_target every
        # cycle while going nowhere (net x ~ 0). Averaging over a window longer than the rock period
        # cancels the back-and-forth -> only SUSTAINED travel earns reward. Steady walkers are
        # unaffected (their windowed avg == their instantaneous avg == target).
        self.track_window = int(track_window)
        self._xhist = deque(maxlen=self.track_window + 1)
        # Private training-only context for the asymmetric critic.  It is NOT part of the
        # environment observation: actor and frozen discriminator both remain exactly 28-D.
        self.critic_context_dim = 1
        self._critic_velocity_feature = 0.0
        self._last_vx_avg = 0.0
        self.dt = None                                  # sim timestep, measured in the sim block
        self.scale_target_by_leg = scale_target_by_leg
        self.leg_ref = leg_ref
        self.leg_length = leg_ref
        # body height scales ~linearly with leg length, so the long-calibrated body_z obs bounds
        # are wrong for shorter bodies (their natural stance falls outside -> discriminator penalises
        # their correct height). Measure standing height at init and scale the body_pos bounds so
        # every body's stance maps to the same normalized value. stand_ref = long standing head_z.
        self.scale_body_z = scale_body_z
        self.stand_ref = stand_ref
        self.stand_height = stand_ref

        if simulation:
            self._port = port
            self.client = RemoteAPIClient('localhost', port=port)
            self.sim = self.client.require('sim')
            try:
                self.sim.setBoolParam(self.sim.boolparam_display_enabled, False)  # rendering off = big speedup
            except Exception:
                pass
            self.OnTimeStep = OnTimeStep  # Set to True for stepping mode, False for continuous mode
            print('Ontime :', self.OnTimeStep)
            self.sim.setStepping(self.OnTimeStep)  # Enable stepping mode for the simulation

            # joint handle
            for leg in range(self.__joint_handle.shape[0]):
                for joint in range(self.__joint_handle.shape[1]):
                    # print(f'Getting joint handle for {self.__joint_names[joint]}{self.__leg_names[leg % 6]}')
                    self.__joint_handle[leg, joint] = self.sim.getObject(
                        self.__joint_names[joint] + self.__leg_names[leg % 6]
                    )
            self.IMU_robot = self.sim.getObject(self.__IMU_names[0])
            self.IMU_ref = self.sim.getObject(self.__IMU_names[1])

            self.dt = float(self.sim.getSimulationTimeStep())   # for windowed avg-velocity reward

            # measure this body's leg length (rigid links -> pose-invariant) and scale the speed
            # target by morphology: shorter legs => proportionally lower natural walking speed.
            def _seg(a, b):
                pa = np.array(self.sim.getObjectPosition(int(a)))
                pb = np.array(self.sim.getObjectPosition(int(b)))
                return float(np.linalg.norm(pa - pb))
            foot_FL = self.sim.getObject(self.__foot_names[0])
            self.leg_length = (_seg(self.__joint_handle[0, 0], self.__joint_handle[0, 1])
                               + _seg(self.__joint_handle[0, 1], self.__joint_handle[0, 2])
                               + _seg(self.__joint_handle[0, 2], foot_FL))
            if self.reward_mode == "track" and self.scale_target_by_leg:
                _s = self.leg_length / self.leg_ref                 # speed AND its tolerance both scale
                self.vx_target = self._vx_target_ref * _s
                self.track_sigma = self._track_sigma_ref * _s       # keep relative precision constant
            print(f"leg_length={self.leg_length:.4f} m  ->  vx_target={self.vx_target:.3f} m/s  "
                  f"sigma={self.track_sigma:.3f}  (ref {self._vx_target_ref}/{self._track_sigma_ref} @ {self.leg_ref} m)")

            self.set_robot_joint(np.zeros((18, 1)))
            self.update()

            # measure this body's STANDING height and scale the body_z obs bounds to it, so a
            # shorter body's natural stance maps to the same normalized value as the long expert's
            # (else its correct height reads as "collapsed" to the frozen discriminator). Done
            # BEFORE the final reset() so env init ends with the sim RUNNING -- a trailing stop()
            # here deadlocks training (it expects a running sim and the scene's auto-run /script,
            # TARGET_RUNS=1, then refuses to restart it).
            if self.scale_body_z:
                try:
                    self.start()
                    for _ in range(100):                     # hold standing pose, settle
                        self.set_robot_joint(np.zeros(18)); self.update()
                    self.stand_height = float(self.sim.getObjectPosition(self.sim.getObject('/head'))[2])
                    self.stop()
                    src = "measured"
                except Exception:
                    self._reconnect()
                    self.stand_height = self.stand_ref * (self.leg_length / self.leg_ref)  # leg-est fallback
                    src = "leg-est"
                hr = self.stand_height / self.stand_ref
                sl = self.slices.get('body_pos')
                if sl is not None:
                    self.observation_space_low[sl] = self.observation_space_low[sl] * hr
                    self.observation_space_high[sl] = self.observation_space_high[sl] * hr
                print(f"stand_height={self.stand_height:.3f} m ({src}) -> body_z obs bounds x{hr:.2f}")

            self.reset()                                     # LAST -> leaves the sim running for training
            print("INFO: VrepInterfaze is initialized successfully.")

        # normalization parameters for action space
        self._action_mid = (self.action_space_high + self.action_space_low) / 2.0
        self._action_scale = (self.action_space_high - self.action_space_low) / 2.0
        # print(f"Action space mid: {self._action_mid}, scale: {self._action_scale}")


    # ------------------- Build obs layout ------------------- #
    def _build_obs_layout(self):
        self.obs_fields = [f for f in self.OBS_SPEC
                           if f.include and not (f.name == 'body_pos' and not self.INCLUDE_BODY_Z)]
        # build slices and observation space bounds
        idx = 0
        self.slices = {}
        lows, highs = [], []
        for f in self.obs_fields:
            sl = slice(idx, idx + f.size)
            self.slices[f.name] = sl
            idx += f.size

            # expand low/high to array if needed
            low = np.full((f.size,), f.low)  if np.isscalar(f.low)  or f.low is None else np.asarray(f.low).reshape(-1)
            high= np.full((f.size,), f.high) if np.isscalar(f.high) or f.high is None else np.asarray(f.high).reshape(-1)
            # binary features are always in [0, 1]
            if f.norm != 'binary':
                lows.append(low)
                highs.append(high)
            else:
                lows.append(np.zeros((f.size,)))
                highs.append(np.ones((f.size,)))

        self.obs_dim = idx
        self.disc_observation_dim = idx
        self.observation_space_low  = np.concatenate(lows, axis=0).astype(float)
        self.observation_space_high = np.concatenate(highs, axis=0).astype(float)

        self.observation_space = np.zeros((self.obs_dim,), dtype=float)
        self.action_space = np.zeros((self.action_space_low.shape[0],), dtype=float)

    # ------------------- Normalization ------------------- #
    def normalize_observation(self, obs):
        obs = np.atleast_2d(obs).astype(float)  # (B, obs_dim)
        out = np.zeros_like(obs)

        for f in self.obs_fields:
            sl = self.slices[f.name]
            x = obs[:, sl]

            if f.norm == 'per_dim':
                low = self.observation_space_low[sl]
                high = self.observation_space_high[sl]
                out[:, sl] = 2.0 * (x - low) / (high - low) - 1.0

            elif f.norm == 'shared':
                low = float(np.min(self.observation_space_low[sl]))
                high = float(np.max(self.observation_space_high[sl]))
                out[:, sl] = 2.0 * (x - low) / (high - low) - 1.0

            elif f.norm == 'binary':
                out[:, sl] = x * 2.0 - 1.0 # 0 -> -1, 1 -> 1
            else:
                raise ValueError(f"Unknown norm: {f.norm}")
        return out[0] if out.shape[0] == 1 else out
    
    def denormalize_observation(self, norm_obs):
        norm_obs = np.atleast_2d(norm_obs).astype(float)
        out = np.zeros_like(norm_obs)

        for f in self.obs_fields:
            sl = self.slices[f.name]
            xn = norm_obs[:, sl]

            if f.norm == 'per_dim':
                low = self.observation_space_low[sl]
                high = self.observation_space_high[sl]
                out[:, sl] = (xn + 1.0) / 2.0 * (high - low) + low

            elif f.norm == 'shared':
                low = float(np.min(self.observation_space_low[sl]))
                high = float(np.max(self.observation_space_high[sl]))
                out[:, sl] = (xn + 1.0) / 2.0 * (high - low) + low

            elif f.norm == 'binary':
                out[:, sl] = (xn + 1.0) / 2.0 # -1 -> 0, 1 -> 1
            else:
                raise ValueError(f"Unknown norm: {f.norm}")
        return out[0] if out.shape[0] == 1 else out

    def normalize_action(self, action):
        return (action - self._action_mid) / self._action_scale

    def denormalize_action(self, norm_action):
        return norm_action * self._action_scale + self._action_mid

    def normalize_expert_data(self, expert_data):
        expert_data['state'] = self.normalize_observation(expert_data['state'])
        expert_data['action'] = self.normalize_action(expert_data['action'])
        return expert_data

    def _set_critic_velocity_context(self, vx):
        """Store a bounded, morphology-normalised velocity signal for the critic only.

        For track mode the command reward is exp(-error**2), with
        error=(vx_avg-vx_target)/track_sigma.  Keeping the signed error lets the critic
        distinguish too-slow from too-fast motion.  Values beyond four sigma map to the boundary;
        their command reward is already below 1.2e-7.  Actor/discriminator never receive this.
        """
        self._last_vx_avg = float(vx)
        if self.reward_mode == "track" and self.track_sigma:
            error = (self._last_vx_avg - self.vx_target) / self.track_sigma
            self._critic_velocity_feature = float(np.clip(error, -4.0, 4.0) / 4.0)
        else:
            scale = max(abs(self.vx_target), 1e-6)
            self._critic_velocity_feature = float(np.clip(self._last_vx_avg / scale, -4.0, 4.0) / 4.0)

    def _track_reward(self, vx):
        """Bounded speed-target reward with useful slope at zero velocity.

        direction is in [-1, 1] and saturates once forward speed reaches the target. precision is
        in [0, 1] with its unique maximum at the target.  Their blend remains bounded in [-m, 1],
        where m=track_direction_mix, and cannot be increased by unbounded forward velocity.
        """
        target = max(float(self.vx_target), 1e-6)
        sigma = max(float(self.track_sigma), 1e-6)
        direction = float(np.clip(vx / target, -1.0, 1.0))
        precision = float(np.exp(-((vx - target) ** 2) / (sigma ** 2)))
        m = self.track_direction_mix
        return m * direction + (1.0 - m) * precision

    def get_critic_context(self):
        """Training-only critic context; deliberately excluded from the public 28-D observation."""
        return np.array([self._critic_velocity_feature], dtype=np.float32)


    # ---------------------- actuation ------------------------ #
    def set_robot_joint(self, target_pos):
        target_pos = target_pos.reshape((6, 3))
        offset = self.__initjoint_position
        for leg in range(0, 6):
            target_pos[leg] += offset[leg]
        self.__target_positions = target_pos

    def set_zero(self):
        self.set_robot_joint(np.zeros(18))


    # ---------------------- get simulation data ------------------------ #
    def get_jointangle(self):
        positions = np.zeros((18))
        for l in range(self.__joint_handle.shape[0]):
            for j in range(self.__joint_handle.shape[1]):
                positions[3 * l + j] = self.sim.getJointPosition(int(self.__joint_handle[l][j]))
        # delete the CTr and FTi 
        # positions = np.delete(positions, [1,2,4,5,7,8,10,11,13,14,16,17])
        return positions
    
    def get_bodyposition(self):
        robot_pos = np.zeros((3))
        robot_pos = self.sim.getObjectPosition(self.sim.getObject('/head'))
        robot_z = robot_pos[2]
        robot_z = np.array([robot_z]).reshape((1,))
        return robot_z

    def get_bodyorientation(self):
        orientation = np.zeros((3))
        orientation = self.sim.getObjectOrientation(self.IMU_robot, self.IMU_ref)
        return orientation
    
    def get_qvel_body(self):
        qvel_body = np.zeros((6))
        qvel_body_get = self.sim.getObjectVelocity(self.sim.getObject('/head'))
        qvel_body = np.array(qvel_body_get[0] + qvel_body_get[1]).reshape((6,))
        return qvel_body

    def get_qvel_joints(self):
        qvel_joints = np.zeros((18))
        for l in range(self.__joint_handle.shape[0]):
            for j in range(self.__joint_handle.shape[1]):
                qvel_joints[3 * l + j] = self.sim.getJointVelocity(int(self.__joint_handle[l][j]))
        return qvel_joints
    
    def get_force(self):
        forces = np.zeros((6))
        for i in range(6):
            _, forceVector, _ = self.sim.readForceSensor(self.sim.getObject(self.__forcesensor_names[i]))
            forces[i] = max(0, np.sqrt((forceVector[0])**2 + (forceVector[1])**2 + (forceVector[2])**2) - 0.2)
        return forces
    
    def get_foot_trajectory(self):
        foot_traj = np.zeros((6))
        for i in range(6):
            # Get the foor trajectory z
            foot_traj[i] = self.sim.getObjectPosition(self.sim.getObject(self.__foot_names[i]))[2]
        return foot_traj
    
    def get_contact(self):
        # contact filtered by force sensor
        contact = np.zeros((6))
        forces = self.get_force()
        for i in range(6):
            contact[i] = 1 if forces[i] > 0.27 else 0
        return contact
    
    def get_states(self):
        parts = []
        for f in self.obs_fields:
            # Use the getter method to retrieve the field values
            vals = getattr(self, f.getter)()
            vals = np.asarray(vals, dtype=float).reshape(-1)
            if vals.size != f.size:
                raise RuntimeError(f"{f.name} size mismatch: got {vals.size}, expected {f.size}")
            parts.append(vals)
        return np.concatenate(parts, axis=0)

    # ---------------------- simulation control ------------------------ #
    def update(self):
        try:
            self._apply_and_step()
        except Exception:
            self._reconnect()
            self._apply_and_step()

    def _apply_and_step(self):
        for leg in range(self.__joint_handle.shape[0]):
            for joint in range(self.__joint_handle.shape[1]):
                self.sim.setJointTargetPosition(int(self.__joint_handle[leg][joint]),
                                                self.__target_positions[leg][joint])
        if self.OnTimeStep:
            self._sim_step()

    def _reconnect(self):
        # recreate the ZMQ client if the server restarted (e.g. CoppeliaSim auto-save can
        # restart it mid-run). Object handles persist server-side, so the still-running sim
        # resumes without losing state.
        self.client = RemoteAPIClient('localhost', port=self._port)
        self.sim = self.client.require('sim')
        self.sim.setStepping(self.OnTimeStep)
        try:
            self.sim.setBoolParam(self.sim.boolparam_display_enabled, False)
        except Exception:
            pass

    def reset(self, zero=True):
        # Normally teleport-in-place (fast). Every N resets do a real stop/start to refresh
        # the physics engine (a continuously-running sim eventually goes numerically unstable
        # / NaN). Reconnect the ZMQ client if the server restarted (auto-save can restart it).
        self._n_resets = getattr(self, '_n_resets', 0) + 1
        hard = (not hasattr(self, '_init_jpos')) or (self._n_resets % 40 == 0)
        if self.HARD_RESET_ALWAYS:
            # diagnostic: bypass reset-in-place entirely (our deviation from the author's env,
            # which did a real stop/start on EVERY reset). Slow (~1.7s vs 0.085s per reset).
            hard = True
        try:
            self._do_reset(hard)
        except Exception:
            self._reconnect()
            self._do_reset(hard=True)
        # reset the robot joints to zero or initial position
        if zero:
            self.set_zero()
            noise = np.random.uniform(-0.1, 0.1, size=(18, ))
            self.set_robot_joint(noise)
            self.update()
        # Start reward history AFTER reset actuation/settling.  Clearing it before the update
        # polluted the first track window with motion caused by reset rather than by the policy.
        head_pos = self.sim.getObjectPosition(self.sim.getObject('/head'))
        self._previous_x = head_pos[0]
        self._xhist.clear()
        self._xhist.append(head_pos[0])
        self._set_critic_velocity_context(0.0)
        # add noise to the initial states 
        # the reset state will be sent to the actor networks: need to normalize it first
        obs = self.get_states()
        obs = self.normalize_observation(obs)
        # NOTE the author's original is np.random.normal(-0.1, 0.1, ...). np.random.normal takes
        # (mean, std), NOT (low, high) -- so it adds noise with MEAN -0.1: a systematic negative
        # bias on every 28-dim reset observation, not the symmetric jitter the line above
        # (np.random.uniform(-0.1, 0.1)) clearly intends. It matters most early in training, when
        # is_healthy() terminates every ~5 steps, so a large fraction of states are reset states.
        # The expert states never carry this bias. Present in ALL SIX of the author's env files.
        # Set FIX_RESET_OBS_NOISE=False to restore the author's exact (buggy) behaviour.
        if self.FIX_RESET_OBS_NOISE:
            noise_obs = obs + np.random.uniform(-0.1, 0.1, size=obs.shape)
        else:
            noise_obs = obs + np.random.normal(-0.1, 0.1, size=obs.shape)
        # reset the step count
        self._step_count = 0
        return noise_obs

    def _do_reset(self, hard):
        if hard:
            # real stop/start refreshes the physics engine; infrequent -> won't crash the server
            self.stop()
            for _ in range(100):
                if self.sim.getSimulationState() == self.sim.simulation_stopped:
                    break
                time.sleep(0.02)
            self.start()
            self._sim_step()
            if not hasattr(self, '_init_jpos'):
                self._record_init_state()
        else:
            self._reset_in_place()

    def is_healthy(self):
        robot_height = self.sim.getObjectPosition(self.sim.getObject('/head'))[2]
        orientation = np.asarray(self.get_bodyorientation(), dtype=float)
        # The old test (head/feet merely above world z=0) treated a toppled, reared, or otherwise
        # unusable robot as healthy, making eval/ep_len=1000 nearly unconditional.  Use very loose,
        # morphology-scaled safety limits: normal expert motion is far inside these thresholds,
        # while a collapsed or overturned body now terminates.
        min_height = max(0.02, 0.25 * self.stand_height)
        return (np.isfinite(robot_height)
                and np.all(np.isfinite(orientation))
                and robot_height > min_height
                and abs(orientation[0]) < 1.2
                and abs(orientation[1]) < 1.2)

    def step(self, action):
        try:
            return self._step_impl(action)
        except Exception:
            self._reconnect()
            return self._step_impl(action)

    def _step_impl(self, action):
        # recieive the policy action and denormalize it
        action = self.denormalize_action(action)
        self.set_robot_joint(action)
        self.update() 
        # env_reward: 'speed' = vx*coef (unbounded, legacy) ; 'track' = Gaussian around vx_target
        # (bounded [0,1], no reward for exceeding expert speed -> no posture-wrecking sprint)
        if self.reward_mode == "track":
            # AVERAGE forward velocity over the sliding window: net head displacement / elapsed
            # time. Rocking in place -> net dx ~ 0 over the window -> vx_avg ~ 0 -> ~no reward.
            # Sustained walking -> vx_avg ~ vx_target -> full reward. (See track_window in __init__.)
            head_x = self.sim.getObjectPosition(self.sim.getObject('/head'))[0]
            self._xhist.append(head_x)
            n = len(self._xhist) - 1                          # number of steps spanned in the window
            if n >= 1 and self.dt:
                vx = (self._xhist[-1] - self._xhist[0]) / (n * self.dt)
            else:
                vx = 0.0
            reward = self._track_reward(vx)
        else:
            vx = self.sim.getObjectVelocity(self.sim.getObject('/head'))[0][0]
            reward = vx * self.vx_coef

        self._set_critic_velocity_context(vx)
        # Preserve the frozen discriminator/policy observation contract: exactly 28 dimensions.
        obs = self.normalize_observation(self.get_states())
        
        # # calculate the average contact gate
        # contacts6 = self.get_contact()
        # eps = 0.2
        # gate = eps + (1.0 - eps) * (float(sum(contacts6)) / 6.0)
        # reward = reward * gate

        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps
        # terminated = False
        terminated = not self.is_healthy()

        # NaN guard: a physics blow-up can make obs/reward NaN -> terminate & sanitize so it
        # can't poison training (the next reset teleports the robot to a clean state).
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            obs = np.nan_to_num(np.asarray(obs, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            reward = 0.0
            self._set_critic_velocity_context(0.0)
            terminated = True

        info = {
            'critic_context': self.get_critic_context(),
            'vx_avg': self._last_vx_avg,
        }
        return obs, reward, terminated, truncated, info

    def start(self):
        self.sim.setStepping(self.OnTimeStep)
        self.sim.startSimulation()

    def stop(self):
        self.sim.stopSimulation()

    def _sim_step(self):
        # advance one simulation step; if a scene errored and CoppeliaSim paused it (state 8),
        # resume first (gated by RESUME_IF_PAUSED so normal training pays no getSimulationState() cost).
        if self.RESUME_IF_PAUSED and self.sim.getSimulationState() == self.sim.simulation_paused:
            self.sim.startSimulation()
        self.sim.step()

    def _record_init_state(self):
        # capture the robot's clean init configuration once, for in-place resets
        self._robot_root = self.sim.getObject('/abdomen')
        tree = self.sim.getObjectsInTree(self._robot_root, self.sim.handle_all, 0)
        self._robot_joints = [o for o in tree
                              if self.sim.getObjectType(o) == self.sim.sceneobject_joint]
        self._robot_shapes = [o for o in ([self._robot_root] + tree)
                              if self.sim.getObjectType(o) == self.sim.sceneobject_shape]
        self._init_root_pose = self.sim.getObjectPose(self._robot_root, -1)
        self._init_jpos = {j: self.sim.getJointPosition(j) for j in self._robot_joints}

    def _reset_in_place(self):
        # teleport robot to recorded init pose + zero all velocities (no sim restart)
        for j, pos in self._init_jpos.items():
            self.sim.setJointPosition(j, pos)
            self.sim.setJointTargetPosition(j, pos)
        self.sim.setObjectPose(self._robot_root, -1, self._init_root_pose)
        for sh in self._robot_shapes:
            try:
                self.sim.resetDynamicObject(sh)
            except Exception:
                pass
        # let the teleport velocity artifact dissipate before the episode starts
        for _ in range(3):
            self._sim_step()


if __name__ == "__main__":
    env = CoppeliaSimEnv()
    env.reset()
    # env.start()
    for i in range(100):
        action = np.random.uniform(-1, 1, size=18)
        obs, reward, terminated, _, _ = env.step(action)
        next_obs = env.get_states()
        # print(f"Step {i+1}, Action: {action}, States: {next_obs}")
    env.stop()
