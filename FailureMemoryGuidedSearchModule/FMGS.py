"""
终点搜索专用失败区域记忆模块 (Destination Search FRM)
专门为终点搜索阶段设计,具有以下特点:

1. 循环检测: 只标记真正循环访问的位置
2. 轻量惩罚: 避免过度限制探索
3. 时间衰减: 旧的失败区域逐渐允许重访
4. 小范围影响: 精确避开问题点,不影响周边
5. 智能合并: 相近位置自动合并
6. 障碍物感知: 惩罚区域不穿过墙壁 (新增)!!!

Author: Claude & User
Date: 2025-01
Updated: 2025-02 (添加障碍物感知)
"""

import numpy as np
from collections import Counter, deque


class DestinationSearchFRM:
    """终点搜索专用失败区域记忆模块"""

    def __init__(self, config):
        """
        初始化模块

        参数:
            config: 配置字典
                - enabled: bool, 是否启用 (默认True)
                - detection_mode: str, 检测模式
                    'immediate': 立即标记 (软回退就标记)
                    'repeat': 重复访问检测 (推荐)
                    'stuck': 卡住检测 (严格)

                - repeat_threshold: int, 重复访问阈值 (默认3)
                - position_window: int, 位置历史窗口 (默认15)
                - position_precision: float, 位置精度/米 (默认0.5m)

                - base_radius: int, 基础半径 (默认5)
                - base_penalty: float, 基础惩罚系数 (默认0.8)

                - max_regions: int, 最大区域数 (默认15)
                - merge_threshold_ratio: float, 合并阈值比例 (默认1.3)

                - use_time_decay: bool, 是否使用时间衰减 (默认True)
                - decay_rate: float, 衰减率/步 (默认0.02)
                - min_penalty: float, 最小惩罚系数 (默认0.95)

                - over_penalty_threshold: float, 过度惩罚阈值 (默认0.05)

                - use_obstacle_aware: bool, 是否使用障碍物感知 (默认True, 新增)
        """
        # 基本配置
        self.enabled = config.get('enabled', True)
        self.detection_mode = config.get('detection_mode', 'repeat')

        # 循环检测参数
        self.repeat_threshold = config.get('repeat_threshold', 3)
        self.position_window = config.get('position_window', 15)
        self.position_precision = config.get('position_precision', 0.5)

        # 惩罚参数
        self.base_radius = config.get('base_radius', 5)
        self.base_penalty = config.get('base_penalty', 0.8)

        # 区域管理
        self.max_regions = config.get('max_regions', 15)
        self.merge_threshold_ratio = config.get('merge_threshold_ratio', 1.3)

        # 时间衰减
        self.use_time_decay = config.get('use_time_decay', True)
        self.decay_rate = config.get('decay_rate', 0.02)
        self.min_penalty = config.get('min_penalty', 0.95)

        # 过度惩罚保护
        self.over_penalty_threshold = config.get('over_penalty_threshold', 0.05)

        # ========== 新增: 障碍物感知 ==========
        self.use_obstacle_aware = config.get('use_obstacle_aware', True)
        # ========== 结束新增 ==========

        # 运行时状态
        self.failed_regions = []
        self.position_history = deque(maxlen=self.position_window)
        self.current_step = 0
        self.backtrack_count = 0

        # 打印配置
        if self.enabled:
            print("\n" + "=" * 70)
            print("Destination Search FRM - ENABLED")
            print("=" * 70)
            print(f"Detection mode: {self.detection_mode}")
            print(f"Repeat threshold: {self.repeat_threshold}")
            print(f"Base radius: {self.base_radius}")
            print(f"Base penalty: {self.base_penalty}")
            print(f"Obstacle aware: {self.use_obstacle_aware}")  # 新增
            print(f"Time decay: {self.use_time_decay}")
            if self.use_time_decay:
                print(f"  Decay rate: {self.decay_rate}/step")
                print(f"  Min penalty: {self.min_penalty}")
            print("=" * 70 + "\n")
        else:
            print("\n[Destination Search FRM] Module DISABLED\n")

    def _discretize_position(self, pose):
        """
        将连续位置离散化,用于检测重复访问

        参数:
            pose: [x, y, theta] 或 [x, y]

        返回:
            tuple: (discrete_x, discrete_y)
        """
        x = int(pose[0] / self.position_precision)
        y = int(pose[1] / self.position_precision)
        return (x, y)

    def update_position_history(self, pose):
        """
        更新位置历史

        参数:
            pose: [x, y, theta] 或 [x, y]
        """
        if not self.enabled:
            return

        discrete_pos = self._discretize_position(pose)
        self.position_history.append(discrete_pos)

    def check_and_mark(self, current_map_position, backtrack_triggered=False):
        """
        检查是否应该标记失败区域,并执行标记

        参数:
            current_map_position: np.array [y, x], 地图坐标
            backtrack_triggered: bool, 是否触发了软回退

        返回:
            bool: 是否标记了新区域
        """
        if not self.enabled:
            return False

        # 更新回退计数
        if backtrack_triggered:
            self.backtrack_count += 1

        should_mark = False

        # ========== 根据检测模式判断是否标记 ==========
        if self.detection_mode == 'immediate':
            # 立即标记模式: 软回退就标记
            should_mark = backtrack_triggered

        elif self.detection_mode == 'repeat':
            # 重复访问检测模式 (推荐)
            if len(self.position_history) >= 5:
                pos_counts = Counter(self.position_history)
                most_common_pos, count = pos_counts.most_common(1)[0]

                if count >= self.repeat_threshold:
                    should_mark = True
                    print(f"[Dest FRM] Loop detected: position {most_common_pos} "
                          f"visited {count} times")

        elif self.detection_mode == 'stuck':
            # 卡住检测模式 (严格)
            if self.backtrack_count >= 3 and len(self.position_history) >= 10:
                # 检查最近10步是否都在小范围内
                recent_positions = list(self.position_history)[-10:]
                unique_positions = len(set(recent_positions))

                if unique_positions <= 3:  # 只访问了3个或更少的不同位置
                    should_mark = True
                    print(f"[Dest FRM] Stuck detected: only {unique_positions} "
                          f"unique positions in last 10 steps")

        # ========== 执行标记 ==========
        if should_mark:
            self._mark_failed_region(current_map_position)
            return True

        return False

    def _mark_failed_region(self, position):
        """
        标记失败区域 (内部方法)

        参数:
            position: np.array [y, x], 地图坐标
        """
        # 检查是否与已有区域相邻
        merged = False
        merge_threshold = self.base_radius * self.merge_threshold_ratio

        for region in self.failed_regions:
            existing_pos = region['position']
            distance = np.linalg.norm(position - existing_pos)

            if distance < merge_threshold:
                # 合并: 保持位置,取较大半径
                old_radius = region['radius']
                region['radius'] = max(region['radius'], self.base_radius)
                region['visit_count'] += 1  # 增加访问计数
                merged = True

                print(f"\n{'=' * 60}")
                print(f"[Dest FRM] Merged with existing region")
                print(f"  Position: ({position[0]:.1f}, {position[1]:.1f})")
                print(f"  Radius: {old_radius} → {region['radius']}")
                print(f"  Visit count: {region['visit_count']}")
                print(f"{'=' * 60}\n")
                break

        if not merged:
            # 添加新区域
            new_region = {
                'position': position.copy(),
                'radius': self.base_radius,
                'penalty': self.base_penalty,
                'marked_step': self.current_step,
                'visit_count': 1
            }
            self.failed_regions.append(new_region)

            print(f"\n{'=' * 60}")
            print(f"[Dest FRM] Marked NEW region #{len(self.failed_regions)}")
            print(f"  Position: ({position[0]:.1f}, {position[1]:.1f})")
            print(f"  Radius: {self.base_radius}")
            print(f"  Penalty: {self.base_penalty}")
            print(f"  Total regions: {len(self.failed_regions)}/{self.max_regions}")
            print(f"{'=' * 60}\n")

            # 限制数量
            if len(self.failed_regions) > self.max_regions:
                removed = self.failed_regions.pop(0)
                print(f"[Dest FRM] Removed oldest region (max limit reached)")

    # ========== 新增函数: 障碍物感知的惩罚掩码 ==========
    def _get_obstacle_aware_mask(self, value_map, center_pos, radius, traversible):
        """
        获取障碍物感知的惩罚掩码

        使用洪水填充算法,从中心点开始扩散,遇到障碍物停止

        参数:
            value_map: np.ndarray, 价值地图
            center_pos: tuple, (y, x) 中心位置
            radius: int, 最大扩散半径
            traversible: np.ndarray, 可通行区域 (1=可通行, 0=障碍物)

        返回:
            mask: np.ndarray, 惩罚掩码 (1=应惩罚, 0=不惩罚)
        """
        mask = np.zeros_like(value_map, dtype=bool)

        cy, cx = int(center_pos[0]), int(center_pos[1])

        # 检查中心点是否在地图内
        if cy < 0 or cy >= value_map.shape[0] or cx < 0 or cx >= value_map.shape[1]:
            return mask

        # 使用BFS洪水填充
        visited = np.zeros_like(value_map, dtype=bool)
        queue = deque([(cy, cx, 0)])  # (y, x, distance)
        visited[cy, cx] = True
        mask[cy, cx] = True

        # 8方向扩散 (更平滑)
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),      # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)     # 对角线
        ]

        while queue:
            y, x, dist = queue.popleft()

            # 超过半径,停止扩散
            if dist >= radius:
                continue

            # 向8个方向扩散
            for dy, dx in directions:
                ny, nx = y + dy, x + dx

                # 检查边界
                if ny < 0 or ny >= value_map.shape[0] or nx < 0 or nx >= value_map.shape[1]:
                    continue

                # 已访问,跳过
                if visited[ny, nx]:
                    continue

                # 标记为已访问
                visited[ny, nx] = True

                # ========== 关键: 检查是否可通行 ==========
                # 如果是障碍物,不继续扩散
                if traversible[ny, nx] == 0:
                    continue
                # ========== 结束检查 ==========

                # 可通行,标记为惩罚区域,并加入队列
                mask[ny, nx] = True
                new_dist = dist + 1

                # 只有在未超过半径时才继续扩散
                if new_dist < radius:
                    queue.append((ny, nx, new_dist))

        return mask
    # ========== 结束新增函数 ==========

    def apply_penalty(self, value_map, current_step, traversible=None):
        """
        对value_map应用惩罚

        参数:
            value_map: np.ndarray, 价值地图
            current_step: int, 当前步数
            traversible: np.ndarray, 可通行区域 (新增参数,可选)

        返回:
            np.ndarray: 惩罚后的价值地图
        """
        if not self.enabled or len(self.failed_regions) == 0:
            return value_map

        self.current_step = current_step

        # ========== 修改: 判断是否使用障碍物感知 ==========
        use_obstacle = self.use_obstacle_aware and traversible is not None

        if use_obstacle:
            print(f"[Dest FRM] Using OBSTACLE-AWARE penalty")
        # ========== 结束判断 ==========

        # 应用每个失败区域的惩罚
        for i, region in enumerate(self.failed_regions):
            pos = region['position']
            radius = region['radius']
            base_penalty = region['penalty']
            marked_step = region['marked_step']

            # 计算时间衰减
            if self.use_time_decay:
                age = current_step - marked_step
                # 惩罚随时间线性衰减
                decay = 1.0 - (age * self.decay_rate)
                decay = max(0.0, decay)  # 不低于0

                # 计算有效惩罚
                effective_penalty = base_penalty + (1.0 - base_penalty) * (1.0 - decay)
                effective_penalty = max(effective_penalty, self.min_penalty)
            else:
                effective_penalty = base_penalty

            # ========== 修改: 选择惩罚方式 ==========
            if use_obstacle:
                # 使用障碍物感知的惩罚掩码
                penalty_mask = self._get_obstacle_aware_mask(
                    value_map, pos, radius, traversible
                )

                # 应用惩罚
                value_map[penalty_mask] *= effective_penalty

                # 统计(可选)
                penalty_cells = np.sum(penalty_mask)
                if penalty_cells > 0:
                    print(f"  Region {i+1}: penalized {penalty_cells} cells (obstacle-aware)")
            else:
                # 使用原有的标准矩形惩罚
                y, x = int(pos[0]), int(pos[1])
                y_min = max(0, y - radius)
                y_max = min(value_map.shape[0], y + radius)
                x_min = max(0, x - radius)
                x_max = min(value_map.shape[1], x + radius)

                value_map[y_min:y_max, x_min:x_max] *= effective_penalty
            # ========== 结束修改 ==========

        # 过度惩罚检测
        max_value = np.max(value_map)
        if max_value < self.over_penalty_threshold:
            print(f"\n{'!' * 60}")
            print(f"[Dest FRM WARNING] Over-penalized!")
            print(f"  Max value: {max_value:.4f} < {self.over_penalty_threshold}")
            print(f"  Clearing all failed regions to allow exploration")
            print(f"{'!' * 60}\n")

            # 清空所有失败区域
            self.failed_regions = []

        return value_map

    def reset(self):
        """重置模块 (新episode开始时调用)"""
        if not self.enabled:
            return

        num_regions = len(self.failed_regions)
        self.failed_regions = []
        self.position_history.clear()
        self.current_step = 0
        self.backtrack_count = 0

        if num_regions > 0:
            print(f"\n[Dest FRM] Reset - cleared {num_regions} regions\n")

    def get_stats(self):
        """获取统计信息"""
        if not self.enabled:
            return {'enabled': False}

        return {
            'enabled': True,
            'num_regions': len(self.failed_regions),
            'backtrack_count': self.backtrack_count,
            'detection_mode': self.detection_mode,
            'obstacle_aware': self.use_obstacle_aware,  # 新增
            'regions': [
                {
                    'position': region['position'].tolist(),
                    'radius': region['radius'],
                    'penalty': region['penalty'],
                    'age': self.current_step - region['marked_step'],
                    'visit_count': region['visit_count']
                }
                for region in self.failed_regions
            ]
        }


def create_destination_search_frm(enable=True, **kwargs):
    """
    工厂函数: 创建终点搜索FRM

    参数:
        enable: bool, 是否启用
        **kwargs: 其他配置参数

    返回:
        DestinationSearchFRM实例
    """
    config = {'enabled': enable}
    config.update(kwargs)
    return DestinationSearchFRM(config)


# ============================================================================
# 推荐配置方案
# ============================================================================

# 方案1: 保守模式 (推荐起点)
CONSERVATIVE_CONFIG = {
    'enabled': True,
    'detection_mode': 'repeat',  # 重复访问检测
    'repeat_threshold': 3,  # 访问3次才标记
    'base_radius': 5,  # 小半径
    'base_penalty': 0.8,  # 轻惩罚
    'use_time_decay': True,  # 启用时间衰减
    'decay_rate': 0.02,  # 每步衰减2%
    'use_obstacle_aware': True,  # 启用障碍物感知 (新增)
}

# 方案2: 激进模式 (如果循环严重)
AGGRESSIVE_CONFIG = {
    'enabled': True,
    'detection_mode': 'repeat',
    'repeat_threshold': 2,  # 访问2次就标记
    'base_radius': 6,  # 稍大半径
    'base_penalty': 0.7,  # 较重惩罚
    'use_time_decay': True,
    'decay_rate': 0.01,  # 衰减较慢
    'use_obstacle_aware': True,
}

# 方案3: 严格模式 (只标记严重卡住)
STRICT_CONFIG = {
    'enabled': True,
    'detection_mode': 'stuck',  # 卡住检测
    'base_radius': 4,  # 更小半径
    'base_penalty': 0.9,  # 非常轻的惩罚
    'use_time_decay': True,
    'decay_rate': 0.03,  # 快速衰减
    'use_obstacle_aware': True,
}

if __name__ == '__main__':
    # 测试代码
    print("Testing Destination Search FRM with Obstacle Awareness...\n")

    # 创建实例
    frm = create_destination_search_frm(**CONSERVATIVE_CONFIG)

    # 创建测试环境
    test_value_map = np.ones((50, 50))

    # 创建障碍物 (竖墙)
    traversible = np.ones((50, 50))
    traversible[20:30, 25] = 0  # 在x=25处有一堵墙

    print("测试场景:")
    print("  - 失败位置: (25, 20) - 墙的左边")
    print("  - 墙位置: x=25, y=20-30")
    print("  - 惩罚半径: 5")
    print()

    # 模拟位置序列 (触发循环检测)
    test_poses = [
        [20.0, 25.0, 0],
        [20.1, 25.1, 0],
        [20.0, 25.0, 0],  # 重复
        [20.1, 25.1, 0],  # 重复
        [20.0, 25.0, 0],  # 重复第3次 -> 应该触发标记
    ]

    for i, pose in enumerate(test_poses):
        frm.update_position_history(pose)

        # 模拟地图位置 (在墙左边)
        map_position = np.array([25, 20])  # (y, x)

        # 检查并标记
        marked = frm.check_and_mark(map_position, backtrack_triggered=(i >= 2))

        if marked:
            print(f"✓ Step {i}: Marked failed region\n")

    # 应用惩罚 (带障碍物感知)
    print("应用惩罚 (障碍物感知):")
    penalized_map = frm.apply_penalty(test_value_map, current_step=5, traversible=traversible)

    print(f"\n结果检查:")
    print(f"  墙左边 (应该被惩罚):")
    print(f"    value_map[25, 15] = {penalized_map[25, 15]:.3f}")
    print(f"  墙右边 (不应该被惩罚):")
    print(f"    value_map[25, 30] = {penalized_map[25, 30]:.3f}")
    print(f"  墙本身:")
    print(f"    traversible[25, 25] = {traversible[25, 25]}")

    # 对比: 不使用障碍物感知
    print(f"\n对比测试 (不使用障碍物感知):")
    test_value_map2 = np.ones((50, 50))
    penalized_map2 = frm.apply_penalty(test_value_map2, current_step=5, traversible=None)

    print(f"  墙左边: {penalized_map2[25, 15]:.3f}")
    print(f"  墙右边: {penalized_map2[25, 30]:.3f} (被错误惩罚)")

    # 获取统计
    stats = frm.get_stats()
    print(f"\nStats: {stats}")

    print("\nTest completed!")