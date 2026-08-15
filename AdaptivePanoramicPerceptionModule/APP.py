import numpy as np
from collections import deque
from typing import Tuple, List, Optional


# ============================================================================
# BacktrackController 类（软回退版本）
# ============================================================================

class BacktrackControllerV1:
    """
    Version 1: 软回退控制器
    只实现状态重置，不进行物理移动
    """

    def __init__(self, config):
        # ========== 基本配置 ==========
        self.enabled_threshold = 1  # 只对 ≥5 个子指令启用
        self.config = config

        # ========== 约束违反检测 ==========
        self.violation_window = 8  # 观察窗口
        self.violation_threshold = 6  # 触发阈值
        self.violation_history = deque(maxlen=self.violation_window)

        # ========== FMM距离停滞检测 ==========
        self.fmm_window = 5
        self.fmm_history = deque(maxlen=self.fmm_window)
        self.stagnation_ratio = 0.95

        # ========== 回退控制 ==========
        self.backtrack_cooldown = 10  # 回退后10步内不再触发
        self.steps_since_backtrack = 999

        # ========== 统计信息 ==========
        self.backtrack_count = 0
        self.total_violations = 0

        # ========== 定期回退配置（新增）==========
        self.periodic_interval = 30  # 终点搜索时期，每30步执行一次
        self.last_periodic_backtrack_step = 0  # 记录上次定期回退的步数

        # print("[BacktrackV1] Initialized with soft reset strategy")

    def is_long_instruction(self, num_sub_instructions: int) -> bool:
        """判断是否为长指令"""
        return num_sub_instructions >= self.enabled_threshold

    def record_step(self, fmm_dist: float):
        """
        记录当前步的信息
        fmm_dist: 当前到目标的FMM距离
        """
        self.fmm_history.append(fmm_dist)
        self.steps_since_backtrack += 1

    def check_constraint_violation(self, check_results: List[bool]) -> int:
        """检查约束违反数量"""
        if len(check_results) == 0:
            return 0
        violations = len(check_results) - sum(check_results)
        self.total_violations += violations
        return violations

    def update_violation_history(self, violations: int):
        """更新违反历史"""
        if violations == 0:
            self.violation_history.append(0)
        elif violations >= 2:
            self.violation_history.append(2)
        else:
            self.violation_history.append(1)

    def check_fmm_stagnation(self) -> bool:
        """检查FMM距离是否停滞"""
        if len(self.fmm_history) < self.fmm_window:
            return False

        recent_dists = list(self.fmm_history)
        no_progress_count = 0

        for i in range(1, len(recent_dists)):
            if recent_dists[i] >= recent_dists[i - 1] * self.stagnation_ratio:
                no_progress_count += 1

        return no_progress_count >= (self.fmm_window - 2)

    def compute_backtrack_score(self, check_results: List[bool]) -> Tuple[float, dict]:
        """计算综合回退分数"""
        violations = self.check_constraint_violation(check_results)
        self.update_violation_history(violations)

        score = 0.0
        debug_info = {}

        # 因素1: 约束违反累积 (权重 0.5)
        if len(self.violation_history) >= self.violation_window:
            violation_sum = sum(self.violation_history)
            violation_rate = violation_sum / (self.violation_window * 2)
            score += violation_rate * 0.5
            debug_info['violation_rate'] = f"{violation_rate:.2f}"
            debug_info['violation_sum'] = violation_sum

        # 因素2: FMM距离停滞 (权重 0.3)
        if self.check_fmm_stagnation():
            score += 0.3
            debug_info['fmm_stagnation'] = True
        else:
            debug_info['fmm_stagnation'] = False

        debug_info['total_score'] = f"{score:.3f}"
        return score, debug_info

    def should_backtrack(self,
                         num_sub_instructions: int,
                         check_results: List[bool],
                         constraint_steps: int) -> Tuple[bool, dict]:
        """
        判断是否应该执行回退
        """
        # 条件1: 只对长指令启用
        if not self.is_long_instruction(num_sub_instructions):
            return False, {'reason': 'short_instruction'}

        # 条件2: 冷却时间未到
        if self.steps_since_backtrack < self.backtrack_cooldown:
            return False, {
                'reason': 'cooldown',
                'steps_since': self.steps_since_backtrack
            }

        # 条件3: 至少执行了一定步数
        if constraint_steps < 5:
            return False, {
                'reason': 'too_few_steps',
                'steps': constraint_steps
            }

        # 计算回退分数
        score, debug_info = self.compute_backtrack_score(check_results)

        # 阈值判断
        BACKTRACK_THRESHOLD = 0.50 # 目前最高性能出现在：BACKTRACK_THRESHOLD = 0.50
        should_bt = score >= BACKTRACK_THRESHOLD

        debug_info['decision'] = 'BACKTRACK' if should_bt else 'continue'
        debug_info['threshold'] = BACKTRACK_THRESHOLD
        debug_info['num_sub_instr'] = num_sub_instructions
        debug_info['constraint_steps'] = constraint_steps

        return should_bt, debug_info

    def execute_soft_backtrack(self) -> dict:
        """
        执行软回退
        返回需要重置的信息
        """
        # 重置内部状态
        self.violation_history.clear()
        self.fmm_history.clear()
        self.steps_since_backtrack = 0
        self.backtrack_count += 1

        print("\n" + "=" * 60)
        print(f"[SOFT BACKTRACK] Soft Reset Triggered (#{self.backtrack_count})")
        print("=" * 60)
        print("Actions:")
        print("  → Clearing violation history")
        print("  → Clearing FMM history")
        print("  → Will perform look_around")
        print("=" * 60 + "\n")

        return {
            'backtrack_count': self.backtrack_count,
            'value_map_scale': 1,  # 保留100%
            'need_look_around': True,
            'reset_constraint_steps': False
        }

    def should_periodic_backtrack(self, current_step: int, is_searching_destination: bool,
                                  has_locked_target: bool) -> Tuple[bool, dict]:
        """
        判断是否应该执行定期回退（用于终点搜索时期）

        参数:
            current_step: 当前全局步数
            is_searching_destination: 是否在搜索最终目的地
            has_locked_target: 是否已锁定目标

        返回:
            (should_backtrack, debug_info)
        """
        # 条件1: 必须在终点搜索时期
        if not is_searching_destination:
            return False, {'reason': 'not_searching_destination'}

        # 条件2: 还没有锁定目标
        if has_locked_target:
            return False, {'reason': 'already_locked_target'}

        # 条件3: 距离上次定期回退超过指定间隔
        steps_since_last = current_step - self.last_periodic_backtrack_step

        if steps_since_last < self.periodic_interval:
            return False, {
                'reason': 'periodic_interval_not_reached',
                'steps_since_last': steps_since_last,
                'interval': self.periodic_interval
            }

        # 触发定期回退
        debug_info = {
            'decision': 'PERIODIC_BACKTRACK',
            'current_step': current_step,
            'steps_since_last': steps_since_last,
            'interval': self.periodic_interval
        }
        print("终点搜索时期，定期执行_look_around")

        return True, debug_info

    def execute_periodic_backtrack(self, current_step: int) -> dict:
        """
        执行定期回退

        参数:
            current_step: 当前全局步数
        """
        # 更新上次定期回退的步数
        self.last_periodic_backtrack_step = current_step
        self.backtrack_count += 1

        print("\n" + "=" * 60)
        print(f"[BACKTRACK V1] Periodic Reset (#{self.backtrack_count})")
        print("=" * 60)
        print("Reason: Searching destination without locked target")
        print(f"Step: {current_step}")
        print(f"Interval: {self.periodic_interval} steps")
        print("Actions:")
        print("  → Will perform look_around")
        print("  → Re-evaluate destination search")
        print("=" * 60 + "\n")

        return {
            'backtrack_count': self.backtrack_count,
            'type': 'periodic',
            'need_look_around': True,
            'reset_constraint_steps': False  # 终点搜索时期不需要重置
        }

    def reset(self):
        """重置控制器（新episode开始时）"""
        self.violation_history.clear()
        self.fmm_history.clear()
        self.steps_since_backtrack = 999
        self.backtrack_count = 0
        self.total_violations = 0
        self.last_periodic_backtrack_step = 0  # 新增：重置定期回退记录

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'backtrack_count': self.backtrack_count,
            'total_violations': self.total_violations
        }