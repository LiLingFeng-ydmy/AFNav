import numpy as np
from collections import deque
from typing import Tuple, List, Optional




class BacktrackControllerV1:
    """
    Version 1: 
    """

    def __init__(self, config):
        
        self.enabled_threshold = 1  
        self.config = config

        
        self.violation_window = 8  
        self.violation_threshold = 6  
        self.violation_history = deque(maxlen=self.violation_window)

        
        self.fmm_window = 5
        self.fmm_history = deque(maxlen=self.fmm_window)
        self.stagnation_ratio = 0.95

        
        self.backtrack_cooldown = 10  # 
        self.steps_since_backtrack = 999

        
        self.backtrack_count = 0
        self.total_violations = 0

        
        self.periodic_interval = 30  
        self.last_periodic_backtrack_step = 0  

        # print("[BacktrackV1] Initialized with soft reset strategy")

    def record_step(self, fmm_dist: float):

        self.fmm_history.append(fmm_dist)
        self.steps_since_backtrack += 1

    def check_constraint_violation(self, check_results: List[bool]) -> int:

        if len(check_results) == 0:
            return 0
        violations = len(check_results) - sum(check_results)
        self.total_violations += violations
        return violations

    def update_violation_history(self, violations: int):

        if violations == 0:
            self.violation_history.append(0)
        elif violations >= 2:
            self.violation_history.append(2)
        else:
            self.violation_history.append(1)

    def check_fmm_stagnation(self) -> bool:

        if len(self.fmm_history) < self.fmm_window:
            return False

        recent_dists = list(self.fmm_history)
        no_progress_count = 0

        for i in range(1, len(recent_dists)):
            if recent_dists[i] >= recent_dists[i - 1] * self.stagnation_ratio:
                no_progress_count += 1

        return no_progress_count >= (self.fmm_window - 2)

    def compute_backtrack_score(self, check_results: List[bool]) -> Tuple[float, dict]:

        violations = self.check_constraint_violation(check_results)
        self.update_violation_history(violations)

        score = 0.0
        debug_info = {}


        if len(self.violation_history) >= self.violation_window:
            violation_sum = sum(self.violation_history)
            violation_rate = violation_sum / (self.violation_window * 2)
            score += violation_rate * 0.5
            debug_info['violation_rate'] = f"{violation_rate:.2f}"
            debug_info['violation_sum'] = violation_sum

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

        if not self.is_long_instruction(num_sub_instructions):
            return False, {'reason': 'short_instruction'}

        if self.steps_since_backtrack < self.backtrack_cooldown:
            return False, {
                'reason': 'cooldown',
                'steps_since': self.steps_since_backtrack
            }


        if constraint_steps < 5:
            return False, {
                'reason': 'too_few_steps',
                'steps': constraint_steps
            }


        score, debug_info = self.compute_backtrack_score(check_results)


        BACKTRACK_THRESHOLD = 0.50 
        should_bt = score >= BACKTRACK_THRESHOLD

        debug_info['decision'] = 'BACKTRACK' if should_bt else 'continue'
        debug_info['threshold'] = BACKTRACK_THRESHOLD
        debug_info['num_sub_instr'] = num_sub_instructions
        debug_info['constraint_steps'] = constraint_steps

        return should_bt, debug_info

    def execute_soft_backtrack(self) -> dict:

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
            'value_map_scale': 0.7,  
            'need_look_around': True,
            'reset_constraint_steps': False
        }

    def should_periodic_backtrack(self, current_step: int, is_searching_destination: bool,
                                  has_locked_target: bool) -> Tuple[bool, dict]:

        if not is_searching_destination:
            return False, {'reason': 'not_searching_destination'}


        if has_locked_target:
            return False, {'reason': 'already_locked_target'}


        steps_since_last = current_step - self.last_periodic_backtrack_step

        if steps_since_last < self.periodic_interval:
            return False, {
                'reason': 'periodic_interval_not_reached',
                'steps_since_last': steps_since_last,
                'interval': self.periodic_interval
            }


        debug_info = {
            'decision': 'PERIODIC_BACKTRACK',
            'current_step': current_step,
            'steps_since_last': steps_since_last,
            'interval': self.periodic_interval
        }

        return True, debug_info

    def execute_periodic_backtrack(self, current_step: int) -> dict:

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
            'reset_constraint_steps': False  
        }

    def reset(self):

        self.violation_history.clear()
        self.fmm_history.clear()
        self.steps_since_backtrack = 999
        self.backtrack_count = 0
        self.total_violations = 0
        self.last_periodic_backtrack_step = 0  

    def get_statistics(self) -> dict:

        return {
            'backtrack_count': self.backtrack_count,
            'total_violations': self.total_violations
        }
