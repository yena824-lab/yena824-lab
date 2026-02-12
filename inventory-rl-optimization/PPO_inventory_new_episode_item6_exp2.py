"""
Weekly inventory RL experiment with PPO
Item specific:
- Demand: 342:57:171 train/valid/test split (6:1:3) + Poisson distribution
- Leadtime: ALL data (no split) + Lognormal distribution
→ Train/Valid/Test: 전체 124개 데이터로 적합
- Trajectory: train 342주, valid 57주, test 171주 각각 저장
"""

import os, argparse, re
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True
from matplotlib.ticker import FormatStrFormatter
from dataclasses import dataclass
import wandb
from collections import Counter, deque
import json
import math
from typing import Dict, Tuple
import torch
import torch.nn as nn

# Gymnasium & Stable-Baselines3
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

# ============================================================================
# Utils
# ============================================================================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def q95(x):
    return float(np.quantile(np.asarray(x), 0.95)) if len(x) else 1.0

def extract_item_key(csv_path: str) -> str:
    """CSV 파일명에서 item key 추출"""
    basename = os.path.basename(csv_path)
    match = re.search(r'item(\d+)', basename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract item key from filename: {csv_path}")

# ============================================================================
# ★★★ Leadtime Fit (LOGNORMAL) - 전체 데이터 사용 ★★★
# ============================================================================
def fit_lognormal_dist(data: np.ndarray, name: str = "data"):
    """
    Lognormal 분포 피팅 - 전체 데이터 사용 (split 없음)
    """
    data = np.array(data)
    data = data[data > 0]
    if len(data) < 2:
        print(f"[WARNING] Not enough data for {name}, using default params")
        return {'mu': 1.0, 'sigma': 0.5}
    
    # Lognormal fitting
    shape, loc, scale = st.lognorm.fit(data, floc=0)
    mu = np.log(scale)
    sigma = shape
    
    print(f"[INFO] {name.capitalize()} lognormal fit: mu={mu:.4f}, sigma={sigma:.4f}")
    print(f"[INFO] Data stats: mean={np.mean(data):.2f}, std={np.std(data):.2f}, n={len(data)}")
    
    return {'mu': float(mu), 'sigma': float(sigma)}

def fit_leadtime_all_data(leadtime_data: np.ndarray):
    """
    전체 leadtime 데이터로 lognormal 적합 (split 없음)
    Returns:
        dict: {
            'params': {'mu': ..., 'sigma': ...},
            'count': int
        }
    """
    leadtime_data = np.array(leadtime_data)
    leadtime_data = leadtime_data[leadtime_data > 0]
    n_total = len(leadtime_data)
    
    print(f"\n[INFO] Leadtime fitting (전체 데이터 사용 - NO SPLIT):")
    print(f"  - Total: {n_total} samples")
    print(f"  - Stats: mean={np.mean(leadtime_data):.2f}, std={np.std(leadtime_data):.2f}")
    
    # Lognormal 분포 피팅
    params = fit_lognormal_dist(leadtime_data, name="all_data")
    
    return {
        'params': params,
        'count': n_total
    }

def load_item_params_all_leadtime(
    params_master_path: str,
    item_key: str,
    leadtime_raw_data: np.ndarray = None
) -> dict:
    """
    params_master.json 로드 + leadtime 전체 데이터로 LOGNORMAL 피팅
    """
    with open(params_master_path, 'r') as f:
        master = json.load(f)
    
    # item_key 정규화
    if not item_key.startswith("item"):
        item_key = f"item{item_key}"
    
    if 'items' not in master or item_key not in master['items']:
        raise KeyError(f"Item '{item_key}' not found in {params_master_path}")
    
    item_data = master['items'][item_key]
    
    # Lognormal 피팅 - 전체 데이터
    if leadtime_raw_data is not None and len(leadtime_raw_data) > 0:
        print(f"[INFO] Using raw leadtime data for '{item_key}'")
        lt_result = fit_leadtime_all_data(leadtime_raw_data)
        leadtime_params = lt_result['params']
        count = lt_result['count']
        print(f"[INFO] Using ALL leadtime ({count} samples): lognormal(mu={leadtime_params['mu']:.4f}, sigma={leadtime_params['sigma']:.4f})")
        leadtime_data = {
            'family': 'lognorm',
            'mu': leadtime_params['mu'],
            'sigma': leadtime_params['sigma']
        }
    else:
        # JSON에서 로드 (폴백)
        lt_family = item_data['leadtime']['family']
        lt_params = item_data['leadtime']['params']
        
        if lt_family == "lognorm":
            leadtime_data = {
                'family': 'lognorm',
                'mu': lt_params['mu'],
                'sigma': lt_params['sigma']
            }
        elif lt_family == "gamma":
            # Gamma를 Lognormal로 변환
            print(f"[WARNING] Converting gamma to lognormal for {item_key}")
            k, theta = lt_params['k'], lt_params['theta']
            mean = k * theta
            var = k * theta**2
            mu = np.log(mean / np.sqrt(1 + var/mean**2))
            sigma = np.sqrt(np.log(1 + var/mean**2))
            leadtime_data = {
                'family': 'lognorm',
                'mu': mu,
                'sigma': sigma
            }
        else:
            raise ValueError(f"Unsupported leadtime family: {lt_family}")
    
    # ★★★ Demand: Poisson 직접 사용 (HurdleNB 근사 제거) ★★★
    demand_family = item_data['demand']['family']
    demand_params = item_data['demand']['params']
    
    if demand_family == "poisson":
        lam = demand_params['lam']
        demand_converted = {
            'family': 'poisson',
            'lam': lam
        }
        print(f"[INFO] Using Poisson demand directly: lam={lam:.2f}")
    elif demand_family == "nbinom":
        # NB는 그대로 사용
        demand_converted = {
            'family': 'nbinom',
            'r': demand_params['r'],
            'p': demand_params['p']
        }
        print(f"[INFO] Using Negative Binomial: r={demand_params['r']:.2f}, p={demand_params['p']:.4f}")
    else:
        raise ValueError(f"Unsupported demand family: {demand_family}")
    
    return {
        'leadtime': leadtime_data,
        'demand': demand_converted
    }

# ============================================================================
# Train/Valid/Test 분할 (6:1:3)
# ============================================================================
def split_dataframe_item6(df: pd.DataFrame):
    """
    Item5용 342:57:171 (train:valid:test) 순차 분할 - 6:1:3
    """
    df = df.sort_values("week_start").reset_index(drop=True)
    n = len(df)
    
    # 6:1:3 비율 설정
    train_ratio = 0.6
    valid_ratio = 0.1
    test_ratio = 0.3
    
    train_weeks = int(n * train_ratio)
    valid_weeks = int(n * valid_ratio)
    test_weeks = n - train_weeks - valid_weeks
    
    df_train = df.iloc[:train_weeks].reset_index(drop=True)
    df_valid = df.iloc[train_weeks:train_weeks+valid_weeks].reset_index(drop=True)
    df_test = df.iloc[train_weeks+valid_weeks:].reset_index(drop=True)
    
    print(f"\n[INFO] Item5 demand split (6:1:3):")
    print(f"  - Total: {n} weeks")
    print(f"  - Train: {len(df_train)} weeks ({df_train['week_start'].iloc[0]} ~ {df_train['week_start'].iloc[-1]})")
    print(f"  - Valid: {len(df_valid)} weeks ({df_valid['week_start'].iloc[0]} ~ {df_valid['week_start'].iloc[-1]})")
    print(f"  - Test: {len(df_test)} weeks ({df_test['week_start'].iloc[0]} ~ {df_test['week_start'].iloc[-1]})")
    
    return df_train, df_valid, df_test

def maybe_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """CSV 컬럼명 매핑"""
    w = df.copy()
    
    if "week_start" not in w:
        raise KeyError("CSV must include 'week_start'")
    
    w["week_start"] = pd.to_datetime(w["week_start"])
    w = w.sort_values("week_start").reset_index(drop=True)
    
    # demand_actual
    if "demand_actual" not in w:
        if "demand_net" in w:
            w["demand_actual"] = w["demand_net"].astype(float)
        elif "demand" in w:
            w["demand_actual"] = w["demand"].astype(float)
        else:
            raise KeyError("need 'demand_actual', 'demand_net', or 'demand'")
    else:
        w["demand_actual"] = w["demand_actual"].astype(float)
    
    # demand moving averages
    for window in [4, 8, 12]:
        col_name = f"demand_mean{window}w"
        if col_name not in w:
            w[col_name] = w["demand_actual"].rolling(window, min_periods=1).mean()
    
    if "planned_receipt" not in w:
        w["planned_receipt"] = 0.0
    
    if "end_on_hand" not in w and "end_onhand" not in w:
        w["end_onhand"] = 0.0
    elif "end_on_hand" in w and "end_onhand" not in w:
        w["end_onhand"] = w["end_on_hand"]
    
    if "end_backlog" not in w:
        w["end_backlog"] = 0.0
    
    for c in ["start_onhand", "start_backlog"]:
        if c not in w:
            w[c] = 0.0
    
    return w

@dataclass
class CostParams:
    h: float = 0.10
    b: float = 0.10
    c: float = 0.10
    K: float = 0.00

# ============================================================================
# Episode Recorder Wrapper
# ============================================================================
class EpisodeRecorder(gym.Wrapper):
    """에피소드별 trajectory 기록"""
    
    def __init__(self, env):
        super().__init__(env)
        self.current_episode_data = []
        self.all_complete_episodes = []
        self.episode_count = 0
        self.best_return = -np.inf
        self.best_episode = None
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_episode_data.append({
            "week_start": info.get("week_start", info.get("weekstart", "")),
            "issued_qty": info.get("issued_qty", info.get("orderqty", info.get("order_qty", 0))),
            "receipts_qty": info.get("receipts_qty", info.get("arrivals", 0)),
            "demand": info.get("demand", 0),
            "planned_receipt": info.get("planned_receipt", 0),
            "demand_cancel": info.get("demand_cancel", 0),
            "planned_receipt_cancel": info.get("planned_receipt_cancel", 0),
            "demand_net": info.get("demand_net", info.get("demand", 0)),
            "planned_receipt_net": info.get("planned_receipt_net", 0),
            "start_onhand": info.get("start_onhand", info.get("onhand", info.get("on_hand", 0))),
            "start_backlog": info.get("start_backlog", info.get("backlog", 0)),
            "end_onhand": info.get("end_onhand", info.get("onhand", info.get("on_hand", 0))),
            "end_backlog": info.get("end_backlog", info.get("backlog", 0)),
            "reward": reward
        })
        
        if terminated or truncated:
            self.episode_count += 1
            episode_return = sum(d["reward"] for d in self.current_episode_data)
            episode_info = {
                "episode_num": self.episode_count,
                "trajectory": self.current_episode_data.copy(),
                "return": episode_return,
                "length": len(self.current_episode_data)
            }
            self.all_complete_episodes.append(episode_info)
            
            if episode_return > self.best_return:
                self.best_return = episode_return
                self.best_episode = episode_info
            
            self.current_episode_data = []
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.current_episode_data = []
        return self.env.reset(**kwargs)
    
    def get_best_episode(self):
        return self.best_episode

# ============================================================================
# WeeklyInvEnv (Test용) - LOGNORMAL leadtime 샘플링
# ============================================================================
class WeeklyInvEnv(gym.Env):
    """CSV 기반 Test 환경 + LOGNORMAL leadtime"""
    
    def __init__(
        self,
        weekly_df: pd.DataFrame,
        leadtime_params: dict,
        k_past: int = 12,
        cost: CostParams = CostParams(),
        action_unit: int = 10,
        max_order: int = 300,
        seed: int = 42,
        pipeline_horizon: int = 64,
        scale_onhand: float = None,
        scale_backlog: float = None,
        scale_d: float = None,
        scale_planned: float = None,
        reward_scale: float = 10000.0
    ):
        super().__init__()
        
        self.df = maybe_map_columns(weekly_df.reset_index(drop=True))
        self.T = len(self.df)
        
        # ★ Lognormal leadtime ★
        self.lt_family = leadtime_params.get("family", "lognorm")
        if self.lt_family == "lognorm":
            self.lt_mu = leadtime_params["mu"]
            self.lt_sigma = leadtime_params["sigma"]
            print(f"[WeeklyInvEnv] Leadtime: lognorm(mu={self.lt_mu:.3f}, sigma={self.lt_sigma:.3f})")
        else:
            raise ValueError(f"Unsupported leadtime family: {self.lt_family}")
        
        self.pipeline_horizon = int(pipeline_horizon)
        self.lt_horizon_weeks = int(pipeline_horizon)  # WeeklyInvEnv
        self.reward_scale = reward_scale  # ★ reward scaling factor 저장도 lt_horizon_weeks 사용
        self.k = int(k_past)
        self.cost = cost
        self.rng = np.random.default_rng(seed)
        
        self.action_unit = int(action_unit)
        self.max_order = int(max_order)
        self.action_levels = np.arange(0, self.max_order + self.action_unit, self.action_unit, dtype=np.int64)
        
        # ★★★ 스케일 설정 ★★★
        if scale_planned is not None:
            self.scale_planned = float(scale_planned)
        else:
            self.scale_planned = max(1.0, self.df["planned_receipt"].max())
        
        if scale_d is not None:
            self.scale_d = float(scale_d)
        else:
            self.scale_d = max(1.0, self.df["demand_actual"].max())
        
        if scale_onhand is not None:
            self.scale_onhand = float(scale_onhand)
        else:
            self.scale_onhand = max(1.0, self.df["end_onhand"].max()) if "end_onhand" in self.df.columns else 1.0
        
        if scale_backlog is not None:
            self.scale_backlog = float(scale_backlog)
        else:
            self.scale_backlog = max(1.0, self.df["end_backlog"].max()) if "end_backlog" in self.df.columns else 1.0
        
        print(f"[INFO] WeeklyInvEnv Scaling: onhand={self.scale_onhand:.2f}, backlog={self.scale_backlog:.2f}, demand={self.scale_d:.2f}, planned={self.scale_planned:.2f}")
        
        self.action_space = spaces.Discrete(len(self.action_levels))
        obs_dim = 2 + (self.pipeline_horizon - 1) + self.k + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.t = 0
        self.on_hand = 0.0
        self.backlog = 0.0
        self.pending_orders = np.zeros((self.lt_horizon_weeks if hasattr(self, "lt_horizon_weeks") else self.pipeline_horizon) + 1, dtype=np.float64)
        self.order_remaining_lt = np.full(len(self.pending_orders), -1, dtype=np.int32)
        self.hist_d = np.zeros(self.k, dtype=np.float64)
    
    def sample_leadtime_weeks(self) -> int:
        """★ Lognormal leadtime 샘플링 ★"""
        lt = int(np.ceil(self.rng.lognormal(mean=self.lt_mu, sigma=self.lt_sigma)))
        return max(1, min(lt, self.lt_horizon_weeks))
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.t = 0
        start_onhand = float(self.df.loc[0, "start_onhand"]) if "start_onhand" in self.df.columns else 0.0
        start_backlog = float(self.df.loc[0, "start_backlog"]) if "start_backlog" in self.df.columns else 0.0
        
        self.on_hand = start_onhand
        self.backlog = start_backlog
        self.pending_orders = np.zeros((self.lt_horizon_weeks if hasattr(self, "lt_horizon_weeks") else self.pipeline_horizon) + 1, dtype=np.float64)
        self.order_remaining_lt = np.full(len(self.pending_orders), -1, dtype=np.int32)
        self.hist_d = np.zeros(self.k, dtype=np.float64)
        
        obs = self._compute_obs(self.t, self.on_hand, self.backlog, self.pending_orders, self.hist_d)
        return obs, {}
    
    def step(self, aidx):
        assert 0 <= aidx < len(self.action_levels)
        q = float(self.action_levels[aidx])
        t = self.t
        
        if t >= self.T:
            obs = self._compute_obs(self.t, self.on_hand, self.backlog, self.pending_orders, self.hist_d)
            return obs, 0.0, True, False, {}
        
        # Backlog 우선 처리
        if self.backlog > 0 and self.on_hand > 0:
            used = min(self.on_hand, self.backlog)
            self.on_hand -= used
            self.backlog -= used
        
        # Demand
        d = float(self.df.loc[t, "demand_actual"])
        served = min(self.on_hand, d)
        self.on_hand -= served
        unmet = d - served
        self.backlog += unmet
        
        # Arrivals
        arrivals = 0.0
        for i in range(len(self.pending_orders)):
            if self.order_remaining_lt[i] == 0:
                arrivals += self.pending_orders[i]
                self.pending_orders[i] = 0.0
                self.order_remaining_lt[i] = -1
        
        for i in range(len(self.pending_orders)):
            if self.order_remaining_lt[i] > 0:
                self.order_remaining_lt[i] -= 1
        
        self.on_hand += arrivals
        
        # ★ 새 주문: Lognormal leadtime 샘플링 ★
        self.pending_orders[1:] = self.pending_orders[:-1]
        self.order_remaining_lt[1:] = self.order_remaining_lt[:-1]
        self.pending_orders[0] = q
        self.order_remaining_lt[0] = self.sample_leadtime_weeks()
        
        # Cost
        holding = self.cost.h * self.on_hand
        back_cost = self.cost.b * self.backlog
        order_cost = (self.cost.c * q + self.cost.K) if q > 0 else 0.0
        reward = (-holding - back_cost - order_cost) / self.reward_scale
        
        # History
        if self.k > 0:
            self.hist_d = np.roll(self.hist_d, 1)
            self.hist_d[0] = d
        
        self.t += 1
        done = self.t >= self.T
        obs = self._compute_obs(min(self.t, self.T - 1), self.on_hand, self.backlog, self.pending_orders, self.hist_d)
        
        info = dict(
            t=t,
            week_start=str(self.df.loc[t, "week_start"]),
            demand=d,
            served=served,
            unmet=unmet,
            arrivals=arrivals,
            order_qty=q,
            holding_cost=holding,
            backlog_cost=back_cost,
            order_cost=order_cost,
            onhand=self.on_hand,
            backlog=self.backlog,
            pipeline=float(self.pending_orders.sum())
        )
        
        return obs, reward, done, False, info
    
    def _compute_obs(self, t, on_hand, backlog, pending_orders, hist_d):
        t = min(t, self.T - 1)
        
        core = np.array([on_hand / self.scale_onhand, backlog / self.scale_backlog], dtype=np.float32)
        
        pending_vec = (pending_orders[1:self.pipeline_horizon] / self.scale_planned).astype(np.float32)
        if len(pending_vec) < self.pipeline_horizon - 1:
            pending_vec = np.pad(pending_vec, (0, self.pipeline_horizon - 1 - len(pending_vec)), mode='constant', constant_values=0.0)
        pending_vec = pending_vec[:self.pipeline_horizon - 1]
        
        past_d = (hist_d / self.scale_d).astype(np.float32) if self.k > 0 else np.zeros(0, dtype=np.float32)
        
        dm4 = float(self.df.loc[t, "demand_mean4w"]) / self.scale_d
        dm8 = float(self.df.loc[t, "demand_mean8w"]) / self.scale_d
        dm12 = float(self.df.loc[t, "demand_mean12w"]) / self.scale_d
        ma_vec = np.array([dm4, dm8, dm12], dtype=np.float32)
        
        return np.concatenate([core, pending_vec, past_d, ma_vec], axis=0)

# ============================================================================
# GenerativeInvEnv (Training용) - LOGNORMAL + POISSON
# ============================================================================
class GenerativeInvEnv(gym.Env):
    """★ LOGNORMAL leadtime + Poisson demand 기반 생성적 환경 ★"""
    
    def __init__(
        self,
        item_params: dict,
        episode_len: int = 342,
        cost: CostParams = None,
        action_unit: float = 1.0,
        max_order: float = 1e9,
        initial_on_hand: float = 0.0,
        allow_backlog: bool = True,
        issue_before_receipt: bool = True,
        fulfill_backlog_same_week: bool = True,
        demand_history: int = 4,
        seed: int = 42,
        per_order_leadtime: bool = True,
        lt_horizon_weeks: int = 64,
        scale_onhand=None,
        scale_backlog=None,
        scale_d: float = None,
        scale_planned: float = None,
        reward_scale: float = 10000.0
    ):
        super().__init__()
        
        # ★ Lognormal Leadtime ★
        lt_data = item_params["leadtime"]
        self.lt_family = lt_data.get("family", "lognorm")
        if self.lt_family == "lognorm":
            self.lt_mu = lt_data["mu"]
            self.lt_sigma = lt_data["sigma"]
            print(f"[GenerativeInvEnv] Leadtime: lognorm(mu={self.lt_mu:.3f}, sigma={self.lt_sigma:.3f})")
        else:
            raise ValueError(f"Unsupported leadtime family: {self.lt_family}")
        
        # ★ Demand: Poisson 또는 NB ★
        self.demand_params = item_params["demand"]
        self.demand_family = self.demand_params["family"]
        print(f"[GenerativeInvEnv] Demand distribution: {self.demand_family}")
        
        self.episode_len = episode_len
        self.cost = cost if cost is not None else CostParams()
        self.reward_scale = reward_scale  # ★ reward scaling factor 저장
        self.action_unit = float(action_unit)
        self.max_order = float(max_order)
        self.initial_on_hand = float(initial_on_hand)
        self.allow_backlog = allow_backlog
        self.issue_before_receipt = issue_before_receipt
        self.fulfill_backlog_same_week = fulfill_backlog_same_week
        self.demand_history = demand_history
        self.per_order_leadtime = per_order_leadtime
        self.lt_horizon_weeks = lt_horizon_weeks
        
        self.scale_onhand = float(scale_onhand) if scale_onhand is not None else 1.0
        self.scale_backlog = float(scale_backlog) if scale_backlog is not None else 1.0
        self.scale_d = float(scale_d) if scale_d is not None else 1.0
        self.scale_planned = float(scale_planned) if scale_planned is not None else 1.0
        
        self.base_seed = seed if seed is not None else 42
        self.episode_count = 0
        self.rng = np.random.default_rng(seed if seed is not None else 42)
        
        self.t = 0
        self.current_date = pd.Timestamp("2020-01-01")
        self.start_date = None
        self.on_hand = float(self.initial_on_hand)
        self.backlog = 0.0
        self.leadtime_weeks = 1
        
        self.pending_orders = np.zeros((self.lt_horizon_weeks if hasattr(self, "lt_horizon_weeks") else self.pipeline_horizon) + 1, dtype=np.float64)
        self.order_remaining_lt = np.full(len(self.pending_orders), -1, dtype=np.int32)
        
        self.k = demand_history
        self.hist_max = max(12, demand_history)
        self.demand_hist = deque(maxlen=self.hist_max)
        
        self.action_space_n = max(1, int(math.floor(self.max_order / self.action_unit)) + 1)
        self.action_space = spaces.Discrete(self.action_space_n)
        
        obs_dim = 2 + (self.lt_horizon_weeks - 1) + self.k + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    def sample_leadtime_weeks(self) -> int:
        """★ Lognormal leadtime 샘플링 ★"""
        lt = int(np.ceil(self.rng.lognormal(mean=self.lt_mu, sigma=self.lt_sigma)))
        return max(1, min(lt, self.lt_horizon_weeks))
    
    def sample_demand(self) -> float:
        """★ Poisson 또는 NB demand 샘플링 ★"""
        cfg = self.demand_params
        fam = self.demand_family
        
        if fam == "poisson":
            lam = float(cfg["lam"])
            return float(self.rng.poisson(lam))
        elif fam == "nbinom":
            r = float(cfg["r"])
            p = float(cfg["p"])
            return float(self.rng.negative_binomial(r, p))
        else:
            raise ValueError(f"Unsupported demand family: {fam}")
    
    def push_incoming(self, qty: float):
        if qty <= 0.0:
            return
        
        self.pending_orders[1:] = self.pending_orders[:-1]
        self.order_remaining_lt[1:] = self.order_remaining_lt[:-1]
        self.pending_orders[0] = qty
        self.order_remaining_lt[0] = self.sample_leadtime_weeks() if self.per_order_leadtime else self.L
    
    def arrive(self) -> float:
        arrivals = 0.0
        for i in range(len(self.pending_orders)):
            if self.order_remaining_lt[i] == 0:
                arrivals += self.pending_orders[i]
                self.pending_orders[i] = 0.0
                self.order_remaining_lt[i] = -1
        
        for i in range(len(self.pending_orders)):
            if self.order_remaining_lt[i] > 0:
                self.order_remaining_lt[i] -= 1
        
        return arrivals
    
    def rolling_mean(self, n: int) -> float:
        if n <= 0 or len(self.demand_hist) == 0:
            return 0.0
        k = min(n, len(self.demand_hist))
        vals = list(self.demand_hist)[-k:]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0
    
    def get_obs(self) -> np.ndarray:
        state = [self.on_hand / self.scale_onhand, self.backlog / self.scale_backlog]
        
        if self.leadtime_weeks > 0:
            pending_vec = (self.pending_orders[1:64] / self.scale_planned).astype(np.float32)
            if len(pending_vec) < 63:
                pending_vec = np.pad(pending_vec, (0, 63 - len(pending_vec)), mode='constant', constant_values=0.0)
            pending_vec = pending_vec[:63]
            state += list(pending_vec)
        
        if self.demand_history > 0:
            hist = list(self.demand_hist)
            k = min(self.demand_history, len(hist))
            recent = hist[-k:] if k > 0 else []
            recent_scaled = [x / self.scale_d for x in recent]
            state += recent_scaled
        
        state += [
            self.rolling_mean(4) / self.scale_d,
            self.rolling_mean(8) / self.scale_d,
            self.rolling_mean(12) / self.scale_d
        ]
        
        obs_target = self.observation_space.shape[0]
        while len(state) < obs_target:
            state.append(0.0)
        
        return np.array(state[:obs_target], dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.base_seed + self.episode_count
        self.episode_count += 1
        self.rng = np.random.default_rng(seed)
        
        self.t = 0
        if self.start_date is not None:
            self.current_date = pd.Timestamp(self.start_date)
        else:
            self.current_date = pd.Timestamp("2020-01-01")
        
        self.on_hand = float(self.initial_on_hand)
        self.backlog = 0.0
        self.pending_orders = np.zeros((self.lt_horizon_weeks if hasattr(self, "lt_horizon_weeks") else self.pipeline_horizon) + 1, dtype=np.float64)
        self.order_remaining_lt = np.full(len(self.pending_orders), -1, dtype=np.int32)
        
        self.demand_hist.clear()
        for _ in range(self.hist_max):
            self.demand_hist.append(0.0)
        
        return self.get_obs(), {}
    
    def step(self, action: int):
        if self.t >= self.episode_len:
            return self.get_obs(), 0.0, True, False, {"msg": "episode_end"}
        
        action = max(0, min(int(action), self.action_space_n - 1))
        order_qty = min(self.max_order, self.action_unit * float(action))
        
        demand = self.sample_demand()
        
        if self.issue_before_receipt and self.backlog > 0 and self.on_hand > 0:
            repay1 = min(self.on_hand, self.backlog)
            self.on_hand -= repay1
            self.backlog -= repay1
        
        served = min(self.on_hand, demand) if demand > 0 else 0.0
        self.on_hand -= served
        unmet = float(demand - served)
        
        if self.allow_backlog and unmet > 0:
            self.backlog += unmet
        
        arrivals = self.arrive()
        self.on_hand += arrivals
        
        if self.fulfill_backlog_same_week and self.backlog > 0 and self.on_hand > 0:
            repay2 = min(self.on_hand, self.backlog)
            self.on_hand -= repay2
            self.backlog -= repay2
        
        if order_qty > 0:
            self.push_incoming(order_qty)
        
        holding_cost = self.cost.h * max(0.0, self.on_hand)
        backlog_cost = self.cost.b * self.backlog
        unit_order_cost = self.cost.c * order_qty
        fixed_order_cost = self.cost.K if order_qty > 0 else 0.0
        total_cost = holding_cost + backlog_cost + unit_order_cost + fixed_order_cost
        
        reward = -total_cost / self.reward_scale
        
        if self.demand_history > 0:
            self.demand_hist.append(float(demand))
        
        pipeline_qty = self.pending_orders.sum()
        
        info = {
            "t": self.t,
            "week_start": str(self.current_date),
            "demand": float(demand),
            "served": float(served),
            "unmet": float(unmet),
            "arrivals": float(arrivals),
            "order_qty": float(order_qty),
            "holding_cost": float(holding_cost),
            "backlog_cost": float(backlog_cost),
            "order_cost": float(unit_order_cost + fixed_order_cost),
            "onhand": self.on_hand,
            "backlog": self.backlog,
            "pipeline": float(pipeline_qty),
            "reward": float(reward),
        }
        
        self.t += 1
        self.current_date += pd.Timedelta(weeks=1)
        done = self.t >= self.episode_len
        
        return self.get_obs(), float(reward), bool(done), False, info

# ============================================================================
# Eval Policy
# ============================================================================
def eval_policy(env, policy, episodes: int = 1, seed: int = 123, deterministic: bool = True):
    traj, total = [], 0.0
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed+ep)
        done = False
        
        while not done:
            if hasattr(policy, "predict"):
                action, _ = policy.predict(obs, deterministic=deterministic)
                action = int(action)
            elif hasattr(policy, "act"):
                action = policy.act(obs)
            else:
                raise ValueError("Policy must have predict() or act() method")
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward
            
            row = {"episode": ep, "reward": reward, "action_idx": action}
            row.update(info)
            traj.append(row)
            
            obs = next_obs
    
    return total / episodes, pd.DataFrame(traj)

# ============================================================================
# WandB Callback
# ============================================================================
class WandbCallback(BaseCallback):
    """Best Episode snapshot 저장"""
    
    def __init__(self, env, episode_recorder, check_freq=10, snapshot_freq=20, total_episodes=None, output_dir=None, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.episode_recorder = episode_recorder
        self.check_freq = check_freq
        self.snapshot_freq = snapshot_freq
        self.total_episodes = total_episodes
        self.output_dir = output_dir
        self.last_visualized_episode = -1
        self.last_snapshot_episode = -1
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        current_episode = self.episode_recorder.episode_count
        episode_just_finished = (current_episode > self.episode_count)
        
        if episode_just_finished:
            self.episode_count = current_episode
            
            if self.episode_recorder.all_complete_episodes:
                last_episode = self.episode_recorder.all_complete_episodes[-1]
                traj = last_episode["trajectory"]
                
                on_hands = [d.get("start_onhand", d.get("end_onhand", 0)) for d in traj]
                backlogs = [d.get("start_backlog", d.get("end_backlog", 0)) for d in traj]
                order_qtys = [d.get("issued_qty", 0) for d in traj]
                holding_costs = [d.get("holding_cost", 0) for d in traj]
                backlog_costs = [d.get("backlog_cost", 0) for d in traj]
                order_costs = [d.get("order_cost", 0) for d in traj]
                
                wandb.log({
                    "Episode": current_episode,
                    "Train/EpisodeNum": current_episode,
                    "Train/LatestReturn": last_episode["return"],
                    "Train/BestReturn": self.episode_recorder.best_return,
                    "Train/AvgOnhand": np.mean(on_hands),
                    "Train/AvgBacklog": np.mean(backlogs),
                    "Train/AvgOrderQty": np.mean(order_qtys),
                    "Train/AvgHoldingCost": np.mean(holding_costs),
                    "Train/AvgBacklogCost": np.mean(backlog_costs),
                    "Train/AvgOrderCost": np.mean(order_costs),
                }, step=current_episode)
                
                # Best episode 시각화
                best_episode = self.episode_recorder.get_best_episode()
                if best_episode and best_episode["episode_num"] != self.last_visualized_episode:
                    self.last_visualized_episode = best_episode["episode_num"]
                    self._visualize_best_episode(best_episode)
        
        return True
    
    def _visualize_best_episode(self, best_episode):
        episode_num = best_episode["episode_num"]
        traj = best_episode["trajectory"]
        
        # Train Best Trajectory 저장
        if self.output_dir:
            traj_df = pd.DataFrame(traj)
            print(f"[INFO] Train best trajectory length: {len(traj_df)} weeks (expected: 342)")
            traj_path = os.path.join(self.output_dir, "best_episode_trajectory.csv")
            traj_df.to_csv(traj_path, index=False)
            print(f"[INFO] Train best trajectory saved to {traj_path}")

# ============================================================================
# Best Model Visualization Callback
# ============================================================================
class BestModelVisualizationCallback(EvalCallback):
    """Best model 시각화 + Valid trajectory 저장"""
    
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq, wandb_callback, output_dir, n_eval_episodes=1, deterministic=True, verbose=1):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            verbose=verbose
        )
        self.wandb_callback = wandb_callback
        self.output_dir = output_dir
        self.eval_count = 0
        self.best_mean_reward = -np.inf
        self.last_saved_step = -1
    
    def _on_step(self):
        continue_training = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            
            if self.last_mean_reward > self.best_mean_reward:
                print(f"\n[INFO] ✅ New best model detected! Mean reward: {self.last_mean_reward:.4f} (previous: {self.best_mean_reward:.4f})")
                self.best_mean_reward = self.last_mean_reward
                
                if self.best_model_save_path is not None:
                    best_model_path = os.path.join(self.best_model_save_path, "best_model")
                    
                    if os.path.exists(best_model_path + ".zip"):
                        try:
                            print(f"[INFO] Loading best model from {best_model_path}.zip")
                            best_model = PPO.load(best_model_path, device="cpu")
                            
                            print(f"[INFO] Collecting valid trajectory...")
                            _, best_traj = eval_policy(self.eval_env, best_model, episodes=1, deterministic=True)
                            
                            print(f"[INFO] Valid trajectory length: {len(best_traj)} weeks (expected: 57)")
                            valid_traj_path = os.path.join(self.output_dir, "valid_best_model_trajectory.csv")
                            best_traj.to_csv(valid_traj_path, index=False)
                            print(f"[INFO] ✅ Valid trajectory saved to {valid_traj_path}")
                            
                            if os.path.exists(valid_traj_path):
                                file_size = os.path.getsize(valid_traj_path)
                                print(f"[INFO] ✅ File verified: {file_size} bytes")
                        
                        except Exception as e:
                            print(f"Warning: Best model visualization failed: {e}")
                    else:
                        print(f"[WARNING] Best model file not found at {best_model_path}.zip")
            else:
                print(f"[INFO] No improvement. Current: {self.last_mean_reward:.4f}, Best: {self.best_mean_reward:.4f}")
        
        return continue_training

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly_csv", type=str, required=True)
    parser.add_argument("--params_master", type=str, default="./params_master.json")
    parser.add_argument("--leadtime_data", type=str, default=None, help="Path to leadtime raw data (CSV/npy)")
    parser.add_argument("--lt_horizon_weeks", type=int, default=87)
    parser.add_argument("--per_order_leadtime", action='store_true', default=True)
    parser.add_argument("--k_past", type=int, default=12)
    
    parser.add_argument("--cost_h", type=float, default=0.10)
    parser.add_argument("--cost_b", type=float, default=0.1789)
    parser.add_argument("--cost_c", type=float, default=0.10)
    parser.add_argument("--cost_K", type=float, default=0.00)
    parser.add_argument("--reward_scale", type=float, default=1000.0, help="Reward scaling factor (reward = -cost / reward_scale)")
    
    parser.add_argument("--action_unit", type=int, default=1)
    parser.add_argument("--max_order", type=int, default=30)
    
    # ★★★ DQN 파라미터를 PPO로 매핑 ★★★
    parser.add_argument("--episodes", type=int, default=37800)  # DQN episodes
    parser.add_argument("--eval_freq_episodes", type=int, default=5)  # DQN valid_interval
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")  # DQN log_dir
    
    # PPO 하이퍼파라미터 (DQN 파라미터로부터 매핑)
    parser.add_argument('--lr', type=float, default=0.0005)  # DQN lr
    parser.add_argument('--n_steps', type=int, default=2048)  # DQN target_update 대응
    parser.add_argument('--batch_size', type=int, default=512)  # DQN batch_size
    parser.add_argument('--n_epochs', type=int, default=10)  # DQN updates_per_step 대응
    parser.add_argument('--gamma', type=float, default=0.99)  # DQN gamma
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)  # DQN epsilon 대응 (탐색 장려)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)  # DQN grad_clip
    
    parser.add_argument("--wandb_check_freq", type=int, default=10)
    parser.add_argument("--wandb_snapshot_freq", type=int, default=20)
    
    args = parser.parse_args()
    
    # Item key 추출
    item_key = extract_item_key(args.weekly_csv)
    print(f"[INFO] Detected item key: {item_key}")
    
    # Leadtime 데이터 로드
    leadtime_weeks = None
    if args.leadtime_data is not None:
        if args.leadtime_data.endswith(".npy"):
            leadtime_weeks = np.load(args.leadtime_data)
        elif args.leadtime_data.endswith(".csv"):
            lt_df = pd.read_csv(args.leadtime_data)
            if "leadtime_weeks" in lt_df.columns:
                leadtime_weeks = lt_df["leadtime_weeks"].values
            elif "lead_weeks" in lt_df.columns:
                leadtime_weeks = lt_df["lead_weeks"].values
            else:
                leadtime_weeks = lt_df.iloc[:, 0].values
        print(f"[INFO] Loaded {len(leadtime_weeks)} leadtime data points")
    
    # ★★★ LOGNORMAL leadtime + POISSON demand 파라미터 로드 (전체 데이터) ★★★
    item_params = load_item_params_all_leadtime(
        params_master_path=args.params_master,
        item_key=item_key,
        leadtime_raw_data=leadtime_weeks
    )
    
    # Configuration 출력
    print(f"\n[INFO] ========== LEADTIME CONFIGURATION ==========")
    if item_params['leadtime'].get('family', 'lognorm') == 'lognorm':
        print(f"[INFO] Leadtime params (ALL DATA): lognormal(mu={item_params['leadtime']['mu']:.4f}, sigma={item_params['leadtime']['sigma']:.4f})")
    elif item_params['leadtime']['family'] == 'gamma':
        print(f"[INFO] Leadtime params (ALL DATA): gamma(k={item_params['leadtime']['k']:.4f}, theta={item_params['leadtime']['theta']:.4f})")
    else:
        print(f"[INFO] Leadtime params (ALL DATA): {item_params['leadtime']}")
    print(f"[INFO] Train/Valid/Test will ALL use SAME leadtime params")
    print(f"[INFO] ============================================\n")
    
    print(f"[INFO] ========== DEMAND CONFIGURATION ==========")
    if item_params['demand']['family'] == 'poisson':
        print(f"[INFO] Demand: Poisson(lam={item_params['demand']['lam']:.2f})")
    elif item_params['demand']['family'] == 'nbinom':
        print(f"[INFO] Demand: Negative Binomial(r={item_params['demand']['r']:.2f}, p={item_params['demand']['p']:.4f})")
    print(f"[INFO] ============================================\n")
    
    # Output 디렉토리
    output_dir = os.path.join(args.output_dir, f"item{item_key}_PPO_DQNParams")
    ensure_dir(output_dir)
    
    # Cost
    cost = CostParams(h=args.cost_h, b=args.cost_b, c=args.cost_c, K=args.cost_K)
    
    # Data split (6:1:3)
    df_weekly = pd.read_csv(args.weekly_csv)
    df_weekly = maybe_map_columns(df_weekly)
    df_train, df_valid, df_test = split_dataframe_item6(df_weekly)
    
    episode_len = len(df_train)
    print(f"\n[INFO] Episode length: {episode_len} weeks")
    
    # Scaling
    scale_planned = max(1.0, df_train["planned_receipt"].max())
    scale_d = max(1.0, df_train["demand_actual"].max())
    scale_onhand = max(1.0, df_train["end_onhand"].max())
    scale_backlog = max(1.0, df_train["end_backlog"].max())
    print(f"[INFO] Scaling: onhand={scale_onhand:.2f}, backlog={scale_backlog:.2f}, demand={scale_d:.2f}, planned={scale_planned:.2f}")
    
    # Train 환경
    train_env_base = GenerativeInvEnv(
        item_params=item_params,
        episode_len=episode_len,
        cost=cost,
        action_unit=args.action_unit,
        max_order=args.max_order,
        initial_on_hand=0.0,
        allow_backlog=True,
        issue_before_receipt=True,
        fulfill_backlog_same_week=True,
        demand_history=args.k_past,
        seed=args.seed,
        per_order_leadtime=args.per_order_leadtime,
        lt_horizon_weeks=args.lt_horizon_weeks,
        scale_onhand=scale_onhand,
        scale_backlog=scale_backlog,
        scale_d=scale_d,
        scale_planned=scale_planned,
        reward_scale=args.reward_scale
    )
    train_env_base.start_date = df_train['week_start'].iloc[0]
    train_env = EpisodeRecorder(train_env_base)
    
    # Valid 환경
    eval_env = WeeklyInvEnv(
        weekly_df=df_valid,
        leadtime_params=item_params["leadtime"],
        k_past=args.k_past,
        cost=cost,
        action_unit=args.action_unit,
        max_order=args.max_order,
        seed=args.seed + 1000,
        scale_onhand=scale_onhand,
        scale_backlog=scale_backlog,
        scale_d=scale_d,
        scale_planned=scale_planned,
        pipeline_horizon=args.lt_horizon_weeks,
        reward_scale=args.reward_scale
    )
    eval_env = EpisodeRecorder(eval_env)
    eval_env = Monitor(eval_env)
    
    # WandB
    run_name = f"item{item_key}_PPO_DQNParams_ep{args.episodes}"
    wandb.init(
        project="InventoryManagementRL_1201",
        name=run_name,
        config=vars(args),
        sync_tensorboard=False
    )
    
    # PPO 모델 (DQN 파라미터 매핑) 
    total_timesteps = args.episodes * episode_len
    eval_freq_steps = args.eval_freq_episodes * episode_len
    
    print(f"\n[INFO] ========== PPO Model Configuration (from DQN params) ==========")
    print(f"  - learning_rate: {args.lr} (from DQN lr)")
    print(f"  - n_steps: {args.n_steps} (from DQN target_update)")
    print(f"  - batch_size: {args.batch_size} (from DQN batch_size)")
    print(f"  - n_epochs: {args.n_epochs} (from DQN updates_per_step)")
    print(f"  - gamma: {args.gamma} (from DQN gamma)")
    print(f"  - max_grad_norm: {args.max_grad_norm} (from DQN grad_clip)")
    print(f"  - ent_coef: {args.ent_coef} (from DQN epsilon - exploration)")
    print(f"  - Network: [256, 256] with ReLU and LayerNorm (from DQN net config)")
    print(f"[INFO] ===========================================================\n")
    
    # ★ Network architecture (DQN 파라미터 적용: hidden_sizes=[256,256], activation=relu, layer_norm=true)
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.ReLU,
        normalize_images=False
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
        device="cpu"
    )
    
    # Callbacks
    wandb_callback = WandbCallback(
        env=train_env,
        episode_recorder=train_env,
        check_freq=args.wandb_check_freq,
        snapshot_freq=args.wandb_snapshot_freq,
        total_episodes=args.episodes,
        output_dir=output_dir,
        verbose=1
    )
    
    best_model_callback = BestModelVisualizationCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "eval_logs"),
        eval_freq=eval_freq_steps,
        wandb_callback=wandb_callback,
        output_dir=output_dir,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1
    )
    
    print(f"\n[INFO] Starting PPO training with DQN-mapped parameters...")
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Episodes: {args.episodes}")
    print(f"  - Episode length: {episode_len} weeks (Train)")
    print(f"  - Eval frequency: every {args.eval_freq_episodes} episodes")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[wandb_callback, best_model_callback],
        progress_bar=True
    )
    
    print(f"\n[INFO] Training completed!")
    
    best_model_path = os.path.join(output_dir, "best_model", "best_model")
    
    # Best 모델 로드
    if os.path.exists(best_model_path + ".zip"):
        print(f"[INFO] Loading BEST model for final evaluation...")
        final_model = PPO.load(best_model_path, device="cpu")
        model_type = "BEST"
    else:
        print(f"[WARNING] Best model not found! Using CURRENT model instead.")
        final_model = model
        model_type = "CURRENT"
    
    # ============================================
    # ★ 전체 데이터로 연속 평가 (Warm-up 방식) ★
    # ============================================
    print(f"\n[INFO] ========== Evaluating {model_type} Model with Warm-up ==========")
    print(f"[INFO] Strategy: Run full trajectory (570 weeks), compute Test metrics only")
    print(f"[INFO] - Train (0-341): Warm-up period")
    print(f"[INFO] - Valid (342-398): Warm-up period")
    print(f"[INFO] - Test (399-569): Performance evaluation")
    print(f"[INFO] - Running 10 independent evaluations with different seeds...")
    
    # 전체 데이터 합치기
    df_full = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    print(f"[INFO] Full dataset: {len(df_full)} weeks")
    
    # 10번 반복 평가
    test_returns = []
    test_backlogs = []
    test_order_qtys = []
    all_full_trajs = []  # 시각화를 위해 첫 번째 trajectory 저장
    
    for run_idx in range(10):
        print(f"\n[INFO] === Run {run_idx + 1}/10 ===")
        
        # 전체 데이터로 환경 생성 (매번 다른 seed)
        full_env = WeeklyInvEnv(
            weekly_df=df_full,
            leadtime_params=item_params["leadtime"],
            k_past=args.k_past,
            cost=cost,
            action_unit=args.action_unit,
            max_order=args.max_order,
            seed=args.seed + 7777 + run_idx,  # 매번 다른 seed
            scale_onhand=scale_onhand,
            scale_backlog=scale_backlog,
            scale_d=scale_d,
            scale_planned=scale_planned,
            pipeline_horizon=args.lt_horizon_weeks,
            reward_scale=args.reward_scale
        )
        full_env = EpisodeRecorder(full_env)
        
        # Best model로 전체 trajectory 생성
        full_mean_reward, full_traj_df = eval_policy(full_env, final_model, episodes=1, deterministic=True)
        
        # Trajectory 분할
        train_end_idx = len(df_train)  # 342
        valid_end_idx = train_end_idx + len(df_valid)  # 399
        
        train_traj_df = full_traj_df.iloc[:train_end_idx].copy()
        valid_traj_df = full_traj_df.iloc[train_end_idx:valid_end_idx].copy()
        test_traj_df = full_traj_df.iloc[valid_end_idx:].copy()
        
        # Test 구간 지표 계산 (실제값)
        test_return_raw = test_traj_df['reward'].sum()  # 정규화된 reward
        test_return = -test_return_raw * args.reward_scale  # 실제 비용 (양수)
        test_backlog_mean = test_traj_df['end_backlog'].mean()  # 실제 백로그
        test_order_qty_sum = test_traj_df['issued_qty'].sum()  # 실제 주문량 총합
        
        test_returns.append(test_return)
        test_backlogs.append(test_backlog_mean)
        test_order_qtys.append(test_order_qty_sum)
        
        print(f"  Test Total Cost: {test_return:.2f}")
        print(f"  Test Backlog (avg): {test_backlog_mean:.2f}")
        print(f"  Test Order Qty (sum): {test_order_qty_sum:.2f}")
        
        # 첫 번째 trajectory는 시각화를 위해 저장
        if run_idx == 0:
            all_full_trajs.append({
                'train': train_traj_df,
                'valid': valid_traj_df,
                'test': test_traj_df,
                'full': full_traj_df
            })
    
    # 10번 평균 계산
    avg_test_return = np.mean(test_returns)
    avg_test_backlog = np.mean(test_backlogs)
    avg_test_order_qty = np.mean(test_order_qtys)
    
    std_test_return = np.std(test_returns)
    std_test_backlog = np.std(test_backlogs)
    std_test_order_qty = np.std(test_order_qtys)
    
    print(f"\n[INFO] ========== 10-Run Test Results (Actual Values) ==========")
    print(f"[INFO] Test Total Cost:    {avg_test_return:.2f} ± {std_test_return:.2f}")
    print(f"[INFO] Test Backlog (avg): {avg_test_backlog:.2f} ± {std_test_backlog:.2f}")
    print(f"[INFO] Test Order Qty (sum): {avg_test_order_qty:.2f} ± {std_test_order_qty:.2f}")
    print(f"[INFO] =======================================================")
    print(f"[INFO] Note: Total Cost = -reward × {args.reward_scale:.0f} (actual cost in original scale)")
    
    # 첫 번째 trajectory 사용 (시각화 및 저장용)
    train_traj_df = all_full_trajs[0]['train']
    valid_traj_df = all_full_trajs[0]['valid']
    test_traj_df = all_full_trajs[0]['test']
    full_traj_df = all_full_trajs[0]['full']
    
    print(f"\n[INFO] Trajectory split (from Run 1):")
    print(f"  - Train: {len(train_traj_df)} weeks")
    print(f"  - Valid: {len(valid_traj_df)} weeks")
    print(f"  - Test: {len(test_traj_df)} weeks")
    
    # 각 구간별 reward 계산 (첫 번째 run 기준, 호환성 유지)
    train_mean_reward = train_traj_df['reward'].mean() if 'reward' in train_traj_df.columns else 0.0
    valid_mean_reward = valid_traj_df['reward'].mean() if 'reward' in valid_traj_df.columns else 0.0
    test_mean_reward = test_traj_df['reward'].mean() if 'reward' in test_traj_df.columns else 0.0
    
    train_total_reward = train_traj_df['reward'].sum() if 'reward' in train_traj_df.columns else 0.0
    valid_total_reward = valid_traj_df['reward'].sum() if 'reward' in valid_traj_df.columns else 0.0
    test_total_reward = test_traj_df['reward'].sum() if 'reward' in test_traj_df.columns else 0.0
    
    # Trajectory 저장
    train_traj_path = os.path.join(output_dir, "train_best_model_trajectory.csv")
    valid_traj_path = os.path.join(output_dir, "valid_best_model_trajectory.csv")
    test_traj_path = os.path.join(output_dir, "test_best_model_trajectory.csv")
    full_traj_path = os.path.join(output_dir, "full_trajectory.csv")
    
    train_traj_df.to_csv(train_traj_path, index=False)
    valid_traj_df.to_csv(valid_traj_path, index=False)
    test_traj_df.to_csv(test_traj_path, index=False)
    full_traj_df.to_csv(full_traj_path, index=False)
    
    print(f"\n[INFO] ✅ Trajectories saved:")
    print(f"  - Train: {train_traj_path}")
    print(f"  - Valid: {valid_traj_path}")
    print(f"  - Test: {test_traj_path}")
    print(f"  - Full: {full_traj_path}")
    
    # 결과 요약
    print(f"\n[INFO] ========== Evaluation Summary (Warm-up Method) ==========")
    print(f"[INFO] Model: {model_type}")
    print(f"[INFO] Train - Mean Reward: {train_mean_reward:.4f}, Total: {train_total_reward:.4f} ({len(train_traj_df)} weeks)")
    print(f"[INFO] Valid - Mean Reward: {valid_mean_reward:.4f}, Total: {valid_total_reward:.4f} ({len(valid_traj_df)} weeks)")
    print(f"[INFO] Test  - Mean Reward: {test_mean_reward:.4f}, Total: {test_total_reward:.4f} ({len(test_traj_df)} weeks) ⭐")
    print(f"[INFO] ========================================================")
    print(f"[INFO] ⭐ TEST performance is computed with warm-up from train+valid periods")
    print(f"[INFO] ⭐ Initial state for test period comes from continuous operation, not reset to 0")
    
    wandb.log({
        "Final/Train_Mean_Reward": train_mean_reward,
        "Final/Valid_Mean_Reward": valid_mean_reward,
        "Final/Test_Mean_Reward": test_mean_reward,
        "Final/Train_Total_Reward": train_total_reward,
        "Final/Valid_Total_Reward": valid_total_reward,
        "Final/Test_Total_Reward": test_total_reward,
    })


    # ============================================
    # ★ 최종 시각화 (20개) - 전체 trajectory 기반 ★
    # ============================================
    if os.path.exists(best_model_path + ".zip"):
        print(f"\n[INFO] ========== Creating 20 Visualizations (Full Trajectory) ==========")
        
        # Train/Valid/Test trajectory는 이미 위에서 분할됨
        # All trajectory는 전체 데이터
        train_df = train_traj_df
        valid_df = valid_traj_df
        test_df = test_traj_df
        all_df = full_traj_df
        
        # 경계 지점 계산
        train_end = len(train_df)  # 342
        valid_end = train_end + len(valid_df)  # 342 + 57 = 399
        
        # 컬럼명 매핑
        def get_column(df, candidates):
            for col in candidates:
                if col in df.columns:
                    return col
            raise KeyError(f"None of {candidates} found in DataFrame")
        
        onhand_col = get_column(train_df, ['end_onhand', 'on_hand', 'onhand'])
        backlog_col = get_column(train_df, ['end_backlog', 'backlog'])
        orderqty_col = get_column(train_df, ['issued_qty', 'order_qty', 'orderqty'])
        reward_col = 'reward'
        

        # ========== 시각화 범위 자동 계산 Helper 함수 ==========
        def get_ylim_with_margin(data_list, margin_ratio=0.1):
            """데이터 범위에 margin을 추가하여 y축 범위 반환"""
            if not data_list or len(data_list) == 0:
                return (0, 1)
            
            data_min = min(data_list)
            data_max = max(data_list)
            data_range = data_max - data_min
            
            if data_range == 0:
                # 모든 값이 동일한 경우
                if data_min == 0:
                    return (0, 1)
                else:
                    return (data_min * 0.9, data_min * 1.1)
            
            margin = data_range * margin_ratio
            y_min = data_min - margin
            y_max = data_max + margin
            
            # reward의 경우 음수 범위 처리
            if data_max <= 0:
                y_max = min(0, y_max)  # 0을 넘지 않도록
            
            return (y_min, y_max)
        
        def get_histogram_bins(data_list, max_order):
            """Order distribution을 위한 적절한 bins 생성"""
            if not data_list:
                return np.arange(0, max_order + 1, max(1, max_order // 20))
            
            data_max = max(data_list)
            bin_width = max(1, int((data_max + 1) / 20))  # 대략 20개 bins
            return np.arange(0, data_max + bin_width, bin_width)
        
        # 시각화 딕셔너리
        vis_log_dict = {}
        
        # ==============================================
        # 1. Reward per step (4개)
        # ==============================================
        datasets = {
            'Train': (train_df, None, None),
            'Valid': (valid_df, None, None),
            'Test': (test_df, None, None),
            'All': (all_df, train_end, valid_end)
        }
        
        for split_name, (df, boundary1, boundary2) in datasets.items():
            steps = range(len(df))
            rewards = df[reward_col].tolist()
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(steps, rewards, label='Reward', color='tab:green', linewidth=2)
            
            # All 데이터에만 경계선 추가
            if split_name == 'All' and boundary1 is not None and boundary2 is not None:
                ax.axvline(x=boundary1, color='red', linestyle='--', linewidth=2, label=f'Train|Valid (step {boundary1})')
                ax.axvline(x=boundary2, color='blue', linestyle='--', linewidth=2, label=f'Valid|Test (step {boundary2})')
            
            ax.set_title(f"Reward per Step - {split_name}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.legend()
            # 데이터 기반 y축 범위 자동 설정
            y_min, y_max = get_ylim_with_margin(rewards, margin_ratio=0.1)
            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.grid(True)
            plt.tight_layout()
            
            filename = f"reward_{split_name.lower()}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            vis_log_dict[f"Visualization/{split_name}_Reward"] = wandb.Image(fig)
            plt.close(fig)
            print(f"[INFO] ✅ Saved: {filename}")
        
        # ==============================================
        # 2. On-hand & Backlog (4개)
        # ==============================================
        for split_name, (df, boundary1, boundary2) in datasets.items():
            steps = range(len(df))
            onhands = df[onhand_col].tolist()
            backlogs = df[backlog_col].tolist()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(steps, onhands, label='On-hand', linewidth=2)
            ax.plot(steps, backlogs, label='Backlog', linewidth=2)
            
            if split_name == 'All' and boundary1 is not None and boundary2 is not None:
                ax.axvline(x=boundary1, color='red', linestyle='--', linewidth=2, label=f'Train|Valid (step {boundary1})')
                ax.axvline(x=boundary2, color='blue', linestyle='--', linewidth=2, label=f'Valid|Test (step {boundary2})')
            
            ax.set_title(f"On-hand & Backlog - {split_name}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Quantity")
            ax.legend()
            # 데이터 기반 y축 범위 자동 설정
            combined_data = onhands + backlogs
            y_min, y_max = get_ylim_with_margin(combined_data, margin_ratio=0.1)
            y_min = max(0, y_min)  # 재고는 0 이상
            ax.set_ylim(y_min, y_max)
            ax.grid(True)
            plt.tight_layout()
            
            filename = f"onhand_backlog_{split_name.lower()}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            vis_log_dict[f"Visualization/{split_name}_Onhand_Backlog"] = wandb.Image(fig)
            plt.close(fig)
            print(f"[INFO] ✅ Saved: {filename}")
        
        # ==============================================
        # 3. Order Quantity (4개)
        # ==============================================
        for split_name, (df, boundary1, boundary2) in datasets.items():
            steps = range(len(df))
            orderqtys = df[orderqty_col].tolist()
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(steps, orderqtys, label='Order Qty', color='tab:orange', linewidth=2)
            
            if split_name == 'All' and boundary1 is not None and boundary2 is not None:
                ax.axvline(x=boundary1, color='red', linestyle='--', linewidth=2, label=f'Train|Valid (step {boundary1})')
                ax.axvline(x=boundary2, color='blue', linestyle='--', linewidth=2, label=f'Valid|Test (step {boundary2})')
            
            ax.set_title(f"Order Quantity - {split_name}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Quantity")
            ax.legend()
            # 데이터 기반 y축 범위 자동 설정
            y_min, y_max = get_ylim_with_margin(orderqtys, margin_ratio=0.1)
            y_min = max(0, y_min)  # 주문량은 0 이상
            ax.set_ylim(y_min, y_max)
            ax.grid(True)
            plt.tight_layout()
            
            filename = f"order_qty_{split_name.lower()}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            vis_log_dict[f"Visualization/{split_name}_Order_Qty"] = wandb.Image(fig)
            plt.close(fig)
            print(f"[INFO] ✅ Saved: {filename}")
        
        # ==============================================
        # 4. Order Distribution (4개)
        # ==============================================
        for split_name, (df, boundary1, boundary2) in datasets.items():
            orderqtys = df[orderqty_col].tolist()
            
            # 데이터 기반 bins 자동 생성
            bins = get_histogram_bins(orderqtys, args.max_order)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            counts, _, _ = ax.hist(orderqtys, bins=bins, edgecolor='black')
            ax.set_title(f"Order Quantity Distribution - {split_name}")
            ax.set_xlabel("Order Quantity")
            ax.set_ylabel("Frequency")
            # 데이터 기반 y축 범위 자동 설정
            max_count = max(counts) if len(counts) > 0 else 1
            ax.set_ylim(0, max_count * 1.1)
            ax.set_xlim(0, args.max_order * 1.05)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            filename = f"order_dist_{split_name.lower()}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            vis_log_dict[f"Visualization/{split_name}_Order_Dist"] = wandb.Image(fig)
            plt.close(fig)
            print(f"[INFO] ✅ Saved: {filename}")

        # ==============================================
        # 5. Demand Plot (4개)
        # ==============================================
        demandcol = get_column(train_df, ['demand', 'demand_actual', 'demandactual'])

        for splitname, (df, boundary1, boundary2) in datasets.items():
            steps = range(len(df))
            demands = df[demandcol].tolist()
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(steps, demands, label='Demand', color='tab:purple', linewidth=2)
            
            if splitname == "All" and boundary1 is not None and boundary2 is not None:
                ax.axvline(x=boundary1, color='red', linestyle='--', linewidth=2, 
                        label=f'Train/Valid (step {boundary1})')
                ax.axvline(x=boundary2, color='blue', linestyle='--', linewidth=2,  
                        label=f'Valid/Test (step {boundary2})')
            
            ax.set_title(f'Demand per Step - {splitname}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Demand')
            ax.legend()
            ax.set_ylim(0, max(demands) * 1.1 if demands else 1)
            ax.grid(True)
            
            plt.tight_layout()
            filename = f'demand_{splitname.lower()}.png'
            fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            vis_log_dict[f'Visualizations/{splitname}/Demand'] = wandb.Image(fig)
            plt.close(fig)
            print(f"[INFO] Saved {filename}")
        
        # WandB 로깅
        wandb.log(vis_log_dict)
        print(f"\n[INFO] ========== 20 Visualizations Complete (Full Trajectory) ==========")
        print(f"[INFO] - Reward per step: 4 plots (Train/Valid/Test/All)")
        print(f"[INFO] - On-hand & Backlog: 4 plots (Train/Valid/Test/All)")
        print(f"[INFO] - Order Quantity: 4 plots (Train/Valid/Test/All)")
        print(f"[INFO] - Order Distribution: 4 plots (Train/Valid/Test/All)")
        print(f"[INFO] - Demand: 4 plots (Train/Valid/Test/All)")
        print(f"[INFO] - Total: 20 visualizations")
        print(f"[INFO] - All plots show CONTINUOUS trajectory with warm-up")
        print(f"[INFO] All visualizations saved to: {output_dir}")
    
    else:
        print(f"[WARNING] Best model not found at {best_model_path}")
        # Best model이 없어도 trajectory DataFrame은 이미 생성됨
        train_df = train_traj_df
        valid_df = valid_traj_df
        test_df = test_traj_df
        all_df = full_traj_df
    
    print(f"\n[INFO] ========== FINAL VERIFICATION ==========")
    print(f"[INFO] Evaluation Method: WARM-UP (continuous trajectory)")
    print(f"[INFO] - Full trajectory: 570 weeks (0→569)")
    print(f"[INFO] - Train (0-341): 342 weeks (warm-up)")
    print(f"[INFO] - Valid (342-398): 57 weeks (warm-up)")
    print(f"[INFO] - Test (399-569): 171 weeks (evaluation) ⭐")
    print(f"[INFO]")
    print(f"[INFO] Trajectory lengths:")
    print(f" - Train: {len(train_traj_df)} weeks")
    print(f" - Valid: {len(valid_traj_df)} weeks")
    print(f" - Test: {len(test_traj_df)} weeks")
    print(f" - Full: {len(full_traj_df)} weeks")
    print(f"\n[INFO] Leadtime configuration:")
    if item_params['leadtime'].get('family', 'lognorm') == 'lognorm':
        print(f" - All periods: lognormal(mu={item_params['leadtime']['mu']:.4f}, sigma={item_params['leadtime']['sigma']:.4f})")
    elif item_params['leadtime']['family'] == 'gamma':
        print(f" - All periods: gamma(k={item_params['leadtime']['k']:.4f}, theta={item_params['leadtime']['theta']:.4f})")
    else:
        print(f" - All periods: {item_params['leadtime']}")
    print(f"\n[INFO] Key insight:")
    print(f" - Test evaluation uses continuous state from train+valid periods")
    print(f" - No reset to initial_on_hand=0 at test start")
    print(f" - Reflects realistic continuous operation")
    print(f"[INFO] ============================================")
    
    wandb.finish()
    print(f"\n[INFO] All done! Results saved to {output_dir}")
  


if __name__ == "__main__":
    main()