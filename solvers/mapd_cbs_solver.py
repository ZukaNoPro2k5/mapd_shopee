"""
MAPD Solver based on Marginal Value Assignment and Dynamic Collision Avoidance.
This replaces the old heavy CBS tree with a highly optimized 1-step lookahead scoring system.
"""

import itertools
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set

from env import DeliveryEnv, Order, Shipper, delivery_reward, valid_next_pos
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]

INF = 10**9
MOVES = ("U", "D", "L", "R")


class MAPDCBSSolver(Solver):
    method_name = "MAPD-CBS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_maps: Dict[Position, Dict[Position, int]] = {}
        self._parent_maps: Dict[Position, Dict[Position, Tuple[Optional[Position], Move]]] = {}
        self._targets: Dict[int, Tuple[str, Any]] = {}
        self._stuck_ticks: Dict[int, int] = {}
        self._last_pos: Dict[int, Position] = {}
        self._rng = random.Random(42)
        self._pickup_history: Dict[Position, int] = {}
        self._seen_orders: set = set()

    def _neighbors(self, pos: Position):
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _distance_map_from(self, start: Position):
        if start in self._distance_maps:
            return self._distance_maps[start]

        distances = {start: 0}
        parents: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}
        queue: deque[Position] = deque([start])

        while queue:
            curr = queue.popleft()
            d = distances[curr]
            for move, nxt in self._neighbors(curr):
                if nxt not in distances:
                    distances[nxt] = d + 1
                    parents[nxt] = (curr, move)
                    queue.append(nxt)

        self._distance_maps[start] = distances
        self._parent_maps[start] = parents
        return distances

    def _distance(self, start: Position, goal: Position) -> int:
        if start not in self._distance_maps:
            self._distance_map_from(start)
        return self._distance_maps[start].get(goal, INF)

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        if start not in self._distance_maps:
            self._distance_map_from(start)

        parents = self._parent_maps[start]
        if goal not in parents:
            return "S"

        curr = goal
        while True:
            prev, move = parents[curr]
            if prev is None:
                return "S"
            if prev == start:
                return move
            curr = prev

    def _evaluate_route(self, start_pos: Position, start_t: int, orders_to_deliver: List[Order]) -> Tuple[float, Optional[Position]]:
        if not orders_to_deliver:
            return 0.0, None

        bag_groups: Dict[Position, List[Order]] = {}
        for o in orders_to_deliver:
            bag_groups.setdefault((o.ex, o.ey), []).append(o)

        best_reward = -INF
        best_first_dest = None

        for perm in itertools.permutations(bag_groups.keys()):
            curr_pos = start_pos
            curr_t = start_t
            total_reward = 0.0
            valid = True

            for dest in perm:
                dist = self._distance(curr_pos, dest)
                if dist == INF:
                    valid = False
                    break
                
                curr_t = curr_t + max(0, dist - 1)
                for o in bag_groups[dest]:
                    total_reward += delivery_reward(o, curr_t, self.env.T)
                curr_pos = dest
                curr_t += 1

            if valid and total_reward > best_reward:
                best_reward = total_reward
                best_first_dest = perm[0]

        return best_reward, best_first_dest

    def _assign_tasks(self, obs: dict):
        shippers: List[Shipper] = obs["shippers"]
        orders: Dict[int, Order] = obs["orders"]
        t = int(obs["t"])
        T = int(obs["T"])

        visible_orders = [o for o in orders.values() if not o.picked and not o.delivered]
        base_rewards = {}
        best_routes = {}

        for s in shippers:
            carried = [orders[oid] for oid in s.bag if oid in orders and not orders[oid].delivered]
            if carried:
                r, first_dest = self._evaluate_route(s.position, t, carried)
                base_rewards[s.id] = r
                best_routes[s.id] = first_dest
            else:
                base_rewards[s.id] = 0.0
                best_routes[s.id] = None

        pickup_candidates = []
        for s in shippers:
            carried = [orders[oid] for oid in s.bag if oid in orders and not orders[oid].delivered]
            carried_weight = sum(o.w for o in carried)

            if len(carried) < s.K_max:
                promising_orders = []
                # Use exact BFS distance map from current position to handle complex mazes
                dist_map = self._distance_map_from(s.position)
                for o in visible_orders:
                    if carried_weight + o.w <= s.W_max:
                        true_dist = dist_map.get((o.sx, o.sy), INF)
                        if true_dist != INF:
                            promising_orders.append((true_dist - o.p * 15, o, true_dist))
                
                # Only evaluate the top 30 most promising orders
                promising_orders.sort(key=lambda x: x[0])
                for _, o, dist_to_pickup in promising_orders[:30]:

                    pickup_t = t + max(1, dist_to_pickup)
                    new_r, _ = self._evaluate_route((o.sx, o.sy), pickup_t, carried + [o])

                    # 1. Marginal Reward: Độ chênh lệch điểm số thực tế khi thêm đơn này vào lộ trình.
                    # Hàm _evaluate_route đã giải bài toán TSP cực tiểu hóa độ trễ, nên marginal này 
                    # bao gồm cả penalty nếu việc nhặt đơn mới làm trễ các đơn đang có trong túi.
                    marginal = new_r - base_rewards[s.id]
                    
                    # 2. Capitalist Penalties (Không kích hoạt trực tiếp trong score nhưng có thể dùng để scale)
                    opp_cost = (o.w / s.W_max) * 4.0
                    trash_penalty = 0.0
                    if carried:
                        if o.p == 1: trash_penalty = 12.0 # Heavily penalize detouring for cheap orders
                        elif o.p == 2: trash_penalty = 5.0

                    # 3. Cluster Bonus: Khuyến khích Batching (gom đơn) cho Shipper rỗng.
                    # Bằng cách quét các đơn 'visible_orders' khác, ta tăng mạnh điểm số nếu đơn o
                    # có cùng đích đến (hoặc điểm lấy) với một đơn khác.
                    # Các hệ số 10.0 và 30.0 được tìm ra qua Grid Search, giúp Shipper rỗng nhận ra 
                    # tiềm năng giao 1 chuyến được nhiều đơn, từ đó tự động tối ưu hóa băng thông.
                    cluster_bonus = 0.0
                    for other in visible_orders:
                        if other.id != o.id:
                            d_pickup = abs(o.sx - other.sx) + abs(o.sy - other.sy)
                            if d_pickup == 0: cluster_bonus += 10.0 * other.p
                            elif d_pickup <= 2: cluster_bonus += 1.0 * other.p
                            
                            d_drop = abs(o.ex - other.ex) + abs(o.ey - other.ey)
                            if d_drop == 0: cluster_bonus += 30.0 * other.p
                            elif d_drop <= 2: cluster_bonus += 2.0 * other.p
                            
                    # 4. Stickiness: Chống Lắc Lư (Oscillation Avoidance).
                    # Giữ Shipper trung thành với mục tiêu hiện tại (5.0 điểm) để tránh đổi target liên tục
                    # khi các đơn hàng mới liên tiếp xuất hiện trên map.
                    stickiness = 5.0 if self._targets.get(s.id) == ('pickup', o.id) else 0.0
                    
                    # 5. Distance Penalty: Chi phí cơ hội (Opportunity Cost of Time).
                    # Không chia cho kích thước map (N) vì chi phí của việc đi 1 bước là mất 1 tick thời gian.
                    # Hệ số 1.2 đảm bảo shipper ưu tiên đơn gần nếu các phần thưởng xấp xỉ nhau, 
                    # nhưng vẫn sẵn sàng đi xa nếu đơn đó mang lại Reward vượt trội (p=3).
                    dist_penalty = dist_to_pickup * 1.2

                    # Tổng hợp điểm
                    score = marginal + cluster_bonus + stickiness - dist_penalty

                    threshold = 0.0 if carried else -INF
                        
                    if score > threshold:
                        pickup_candidates.append((score, s.id, o))

        pickup_candidates.sort(key=lambda x: x[0], reverse=True)
        assigned_orders = set()
        assigned_shippers = set()

        for score, sid, o in pickup_candidates:
            if sid in assigned_shippers or o.id in assigned_orders:
                continue
            self._targets[sid] = ('pickup', o.id)
            assigned_shippers.add(sid)
            assigned_orders.add(o.id)

        for oid, o in orders.items():
            if oid not in self._seen_orders:
                self._seen_orders.add(oid)
                pos = (o.sx, o.sy)
                self._pickup_history[pos] = self._pickup_history.get(pos, 0) + 1

        assigned_hotspots = set()
        hotspots = sorted(self._pickup_history.items(), key=lambda x: x[1], reverse=True)

        for s in shippers:
            if s.id not in assigned_shippers:
                if best_routes[s.id] is not None:
                    self._targets[s.id] = ('deliver', best_routes[s.id])
                else:
                    best_hs = s.position
                    best_hs_score = -INF
                    for hs_pos, count in hotspots:
                        if hs_pos not in assigned_hotspots:
                            # Distance-weighted hotspot: High frequency is good, but far away is bad
                            dist = self._distance(s.position, hs_pos)
                            if dist == INF: continue
                            score = count * 10 - dist
                            if score > best_hs_score:
                                best_hs_score = score
                                best_hs = hs_pos
                    
                    if best_hs != s.position:
                        assigned_hotspots.add(best_hs)
                    self._targets[s.id] = ('idle', best_hs)

    def _get_action(self, s: Shipper, target: Tuple[str, Any], orders: Dict[int, Order], t: int) -> Action:
        carried = [orders[oid] for oid in s.bag if oid in orders and not orders[oid].delivered]
        
        if any((o.ex, o.ey) == s.position for o in carried):
            return ("S", 2)

        if target[0] == 'pickup':
            order_id = target[1]
            o = orders[order_id]
            goal = (o.sx, o.sy)
            if s.position == goal:
                return ("S", 1)
        elif target[0] == 'deliver':
            goal = target[1]
            if s.position == goal:
                return ("S", 2)
        else:
            goal = s.position

        if s.position == goal:
            return ("S", 0)

        # Basic pathing
        move = self._next_move(s.position, goal)
        
        # Deadlock breaking: if stuck for > 1 ticks, pick a random valid move
        if self._stuck_ticks.get(s.id, 0) > 1:
            valid_m = []
            for m in MOVES:
                nxt = valid_next_pos(s.position, m, self.grid)
                if nxt != s.position:
                    valid_m.append(m)
            if valid_m:
                move = self._rng.choice(valid_m)

        next_pos = valid_next_pos(s.position, move, self.grid)

        if any((o.ex, o.ey) == next_pos for o in carried):
            op = 2
        elif next_pos == goal and target[0] == 'pickup':
            op = 1
        elif next_pos == goal and target[0] == 'deliver':
            op = 2
        else:
            op = 0

        return (move, op)

    def _shipper_importance(self, s: Shipper, orders: Dict[int, Order]) -> float:
        carried = [orders[oid] for oid in s.bag if oid in orders and not orders[oid].delivered]
        if not carried: return 0.0
        return max(o.p for o in carried)

    def _resolve_conflicts(self, actions: Dict[int, Action], shippers: List[Shipper], orders: Dict[int, Order]) -> Dict[int, Action]:
        desired_pos = {}
        for s in shippers:
            move, op = actions[s.id]
            desired_pos[s.id] = valid_next_pos(s.position, move, self.grid)

        # 1. Idle blockers yielding
        for s in shippers:
            t_pos = desired_pos[s.id]
            if t_pos != s.position:
                blocker = next((x for x in shippers if x.position == t_pos), None)
                if blocker and blocker.id != s.id:
                    b_target = self._targets.get(blocker.id, ('idle', None))
                    if desired_pos[blocker.id] == blocker.position and not blocker.bag and b_target[0] == 'idle':
                        forbidden = set(desired_pos.values())
                        forbidden.add(s.position)
                        yield_move = None
                        for move in MOVES:
                            nxt = valid_next_pos(blocker.position, move, self.grid)
                            if nxt != blocker.position and nxt not in forbidden:
                                yield_move = move
                                break
                        if yield_move:
                            actions[blocker.id] = (yield_move, 0)
                            desired_pos[blocker.id] = valid_next_pos(blocker.position, yield_move, self.grid)

        # 2. Swap breaking with Priority
        shippers_sorted = sorted(shippers, key=lambda s: (self._shipper_importance(s, orders), -s.id), reverse=True)
        for i in range(len(shippers_sorted)):
            for j in range(i + 1, len(shippers_sorted)):
                s1 = shippers_sorted[i]
                s2 = shippers_sorted[j]
                if desired_pos[s1.id] == s2.position and desired_pos[s2.id] == s1.position:
                    # Try to make s2 yield
                        forbidden = set(desired_pos.values())
                        forbidden.add(s1.position)
                        yield_move = None
                        for move in MOVES:
                            nxt = valid_next_pos(s2.position, move, self.grid)
                            if nxt != s2.position and nxt not in forbidden:
                                yield_move = move
                                break
                        if yield_move:
                            actions[s2.id] = (yield_move, 0)
                            desired_pos[s2.id] = valid_next_pos(s2.position, yield_move, self.grid)
                        else:
                            # s2 is trapped. Force s1 to yield
                            for move in MOVES:
                                nxt = valid_next_pos(s1.position, move, self.grid)
                                if nxt != s1.position and nxt not in forbidden and nxt != s2.position:
                                    yield_move = move
                                    break
                            if yield_move:
                                actions[s1.id] = (yield_move, 0)
                                desired_pos[s1.id] = valid_next_pos(s1.position, yield_move, self.grid)
                                actions[s2.id] = ("S", 0)
                                desired_pos[s2.id] = s2.position
                            else:
                                # Both trapped. Break the swap by stopping the lower priority one
                                actions[s2.id] = ("S", 0)
                                desired_pos[s2.id] = s2.position

        # 3. Simulate Environment Collision Rules
        old_positions = {s.id: s.position for s in shippers}
        occupied = set(old_positions.values())
        actual_pos = {}

        for s in sorted(shippers, key=lambda x: x.id):
            old = old_positions[s.id]
            target = desired_pos[s.id]
            occupied.discard(old)
            if target in occupied:
                target = old
            occupied.add(target)
            actual_pos[s.id] = target

        for s in shippers:
            if actual_pos[s.id] != desired_pos[s.id]:
                actions[s.id] = ("S", 0)

        return actions

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            # Update stuck ticks
            for s in obs["shippers"]:
                if s.position == self._last_pos.get(s.id):
                    self._stuck_ticks[s.id] = self._stuck_ticks.get(s.id, 0) + 1
                else:
                    self._stuck_ticks[s.id] = 0
                self._last_pos[s.id] = s.position

            self._assign_tasks(obs)

            raw_actions = {}
            t = int(obs["t"])
            for s in obs["shippers"]:
                target = self._targets.get(s.id, ('idle', None))
                raw_actions[s.id] = self._get_action(s, target, obs["orders"], t)

            actions = self._resolve_conflicts(raw_actions, obs["shippers"], obs["orders"])
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
