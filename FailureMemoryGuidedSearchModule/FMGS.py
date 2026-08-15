

import numpy as np
from collections import Counter, deque


class DestinationSearchFRM:


    def __init__(self, config):

        self.enabled = config.get('enabled', True)
        self.detection_mode = config.get('detection_mode', 'repeat')


        self.repeat_threshold = config.get('repeat_threshold', 3)
        self.position_window = config.get('position_window', 15)
        self.position_precision = config.get('position_precision', 0.5)


        self.base_radius = config.get('base_radius', 5)
        self.base_penalty = config.get('base_penalty', 0.8)


        self.max_regions = config.get('max_regions', 15)
        self.merge_threshold_ratio = config.get('merge_threshold_ratio', 1.3)


        self.use_time_decay = config.get('use_time_decay', True)
        self.decay_rate = config.get('decay_rate', 0.02)
        self.min_penalty = config.get('min_penalty', 0.95)


        self.over_penalty_threshold = config.get('over_penalty_threshold', 0.05)


        self.use_obstacle_aware = config.get('use_obstacle_aware', True)



        self.failed_regions = []
        self.position_history = deque(maxlen=self.position_window)
        self.current_step = 0
        self.backtrack_count = 0


        if self.enabled:
            print("\n" + "=" * 70)
            print("Destination Search FRM - ENABLED")
            print("=" * 70)
            print(f"Detection mode: {self.detection_mode}")
            print(f"Repeat threshold: {self.repeat_threshold}")
            print(f"Base radius: {self.base_radius}")
            print(f"Base penalty: {self.base_penalty}")
            print(f"Obstacle aware: {self.use_obstacle_aware}")
            print(f"Time decay: {self.use_time_decay}")
            if self.use_time_decay:
                print(f"  Decay rate: {self.decay_rate}/step")
                print(f"  Min penalty: {self.min_penalty}")
            print("=" * 70 + "\n")
        else:
            print("\n[Destination Search FRM] Module DISABLED\n")

    def _discretize_position(self, pose):

        x = int(pose[0] / self.position_precision)
        y = int(pose[1] / self.position_precision)
        return (x, y)

    def update_position_history(self, pose):

        if not self.enabled:
            return

        discrete_pos = self._discretize_position(pose)
        self.position_history.append(discrete_pos)

    def check_and_mark(self, current_map_position, backtrack_triggered=False):

        if not self.enabled:
            return False


        if backtrack_triggered:
            self.backtrack_count += 1

        should_mark = False


        if self.detection_mode == 'immediate':

            should_mark = backtrack_triggered

        elif self.detection_mode == 'repeat':

            if len(self.position_history) >= 5:
                pos_counts = Counter(self.position_history)
                most_common_pos, count = pos_counts.most_common(1)[0]

                if count >= self.repeat_threshold:
                    should_mark = True
                    print(f"[Dest FRM] Loop detected: position {most_common_pos} "
                          f"visited {count} times")

        elif self.detection_mode == 'stuck':

            if self.backtrack_count >= 3 and len(self.position_history) >= 10:

                recent_positions = list(self.position_history)[-10:]
                unique_positions = len(set(recent_positions))

                if unique_positions <= 3:  
                    should_mark = True
                    print(f"[Dest FRM] Stuck detected: only {unique_positions} "
                          f"unique positions in last 10 steps")


        if should_mark:
            self._mark_failed_region(current_map_position)
            return True

        return False

    def _mark_failed_region(self, position):


        merged = False
        merge_threshold = self.base_radius * self.merge_threshold_ratio

        for region in self.failed_regions:
            existing_pos = region['position']
            distance = np.linalg.norm(position - existing_pos)

            if distance < merge_threshold:

                old_radius = region['radius']
                region['radius'] = max(region['radius'], self.base_radius)
                region['visit_count'] += 1  
                merged = True

                print(f"\n{'=' * 60}")
                print(f"[Dest FRM] Merged with existing region")
                print(f"  Position: ({position[0]:.1f}, {position[1]:.1f})")
                print(f"  Radius: {old_radius} → {region['radius']}")
                print(f"  Visit count: {region['visit_count']}")
                print(f"{'=' * 60}\n")
                break

        if not merged:

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


            if len(self.failed_regions) > self.max_regions:
                removed = self.failed_regions.pop(0)
                print(f"[Dest FRM] Removed oldest region (max limit reached)")


    def _get_obstacle_aware_mask(self, value_map, center_pos, radius, traversible):

        mask = np.zeros_like(value_map, dtype=bool)

        cy, cx = int(center_pos[0]), int(center_pos[1])

        if cy < 0 or cy >= value_map.shape[0] or cx < 0 or cx >= value_map.shape[1]:
            return mask


        visited = np.zeros_like(value_map, dtype=bool)
        queue = deque([(cy, cx, 0)])  # (y, x, distance)
        visited[cy, cx] = True
        mask[cy, cx] = True


        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),      
            (-1, -1), (-1, 1), (1, -1), (1, 1)     
        ]

        while queue:
            y, x, dist = queue.popleft()


            if dist >= radius:
                continue


            for dy, dx in directions:
                ny, nx = y + dy, x + dx


                if ny < 0 or ny >= value_map.shape[0] or nx < 0 or nx >= value_map.shape[1]:
                    continue


                if visited[ny, nx]:
                    continue


                visited[ny, nx] = True


                if traversible[ny, nx] == 0:
                    continue

                mask[ny, nx] = True
                new_dist = dist + 1


                if new_dist < radius:
                    queue.append((ny, nx, new_dist))

        return mask


    def apply_penalty(self, value_map, current_step, traversible=None):

        if not self.enabled or len(self.failed_regions) == 0:
            return value_map

        self.current_step = current_step


        use_obstacle = self.use_obstacle_aware and traversible is not None

        if use_obstacle:
            print(f"[Dest FRM] Using OBSTACLE-AWARE penalty")


        for i, region in enumerate(self.failed_regions):
            pos = region['position']
            radius = region['radius']
            base_penalty = region['penalty']
            marked_step = region['marked_step']


            if self.use_time_decay:
                age = current_step - marked_step

                decay = 1.0 - (age * self.decay_rate)
                decay = max(0.0, decay) 


                effective_penalty = base_penalty + (1.0 - base_penalty) * (1.0 - decay)
                effective_penalty = max(effective_penalty, self.min_penalty)
            else:
                effective_penalty = base_penalty


            if use_obstacle:

                penalty_mask = self._get_obstacle_aware_mask(
                    value_map, pos, radius, traversible
                )


                value_map[penalty_mask] *= effective_penalty


                penalty_cells = np.sum(penalty_mask)
                if penalty_cells > 0:
                    print(f"  Region {i+1}: penalized {penalty_cells} cells (obstacle-aware)")
            else:

                y, x = int(pos[0]), int(pos[1])
                y_min = max(0, y - radius)
                y_max = min(value_map.shape[0], y + radius)
                x_min = max(0, x - radius)
                x_max = min(value_map.shape[1], x + radius)

                value_map[y_min:y_max, x_min:x_max] *= effective_penalty



        max_value = np.max(value_map)
        if max_value < self.over_penalty_threshold:
            print(f"\n{'!' * 60}")
            print(f"[Dest FRM WARNING] Over-penalized!")
            print(f"  Max value: {max_value:.4f} < {self.over_penalty_threshold}")
            print(f"  Clearing all failed regions to allow exploration")
            print(f"{'!' * 60}\n")


            self.failed_regions = []

        return value_map

    def reset(self):

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

        if not self.enabled:
            return {'enabled': False}

        return {
            'enabled': True,
            'num_regions': len(self.failed_regions),
            'backtrack_count': self.backtrack_count,
            'detection_mode': self.detection_mode,
            'obstacle_aware': self.use_obstacle_aware, 
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

    config = {'enabled': enable}
    config.update(kwargs)
    return DestinationSearchFRM(config)








if __name__ == '__main__':
