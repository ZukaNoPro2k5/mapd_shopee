from __future__ import annotations
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from math import inf
import sys

sys.path.append("/kaggle/input/datasets/minhhhtrann/graph-shopee")


from env import (
    DeliveryEnv,
    Order,
    Shipper,
    is_valid_cell,
    manhattan,
    move_cost,
    delivery_reward,
)
from solvers.solver import Solver

ALPHA = 0.8
BETA = 3  # trong so heuristic (eta)
RHO = 0.08  # toc do bay hoi pheromone
N_ANTS = 40  # số kiến mỗi iteration (sweet spot)
N_ITER = 15  # số iteration ACO (sweet spot)
TAU_MIN = 0.1
TAU_MAX = 25.0

DEADLINE_BUFFER = 8
DETOUR_LIMIT = 6
NEAREST_K = 8
DEPOSIT_FACTOR = 0.12  # he so reward -> pheromone real-time
CAP_W_THRESHOLD = 0.85  # nguong phat khi gan day weight
CAP_PEN_W = 0.25
CAP_PEN_K = 0.4


def a_star(
    start: Tuple[int, int], goal: Tuple[int, int], grid, avoid=None
) -> List[Tuple[int, int]]:
    if not is_valid_cell(goal, grid):
        return []

    if avoid:
        avoid = avoid - {start, goal}
    else:
        avoid = set()

    from heapq import heappush, heappop

    open_set = []  # Priority Queue
    closed_set = set()  # các ô quét xong
    came_from = {}  # truy vết đường đi
    g_score = {start: 0}  # chi phí thực tế từ start

    heappush(open_set, (manhattan(*start, *goal), 0, start))

    while open_set:
        _, current_g, current = heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if current in closed_set:
            continue
        closed_set.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in closed_set:
                continue
            if not is_valid_cell(neighbor, grid):
                continue
            if neighbor in avoid:
                continue
            tentative_g = current_g + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = manhattan(*neighbor, *goal)
                heappush(open_set, (tentative_g + h, tentative_g, neighbor))

    if avoid:
        return a_star(start, goal, grid, avoid=None)
    return []


import itertools


def choose_best_order_to_deliver(
    shipper: Shipper,
    orders: Dict[int, Order],
    t: int,
    T: int,
) -> Optional[Order]:
    candidates = [orders[oid] for oid in shipper.bag if oid in orders]
    if not candidates:
        return None

    if len(candidates) <= 6:
        best_first_order = None
        max_total_reward = -1

        for sequence in itertools.permutations(candidates):
            current_t = t
            current_r, current_c = shipper.r, shipper.c
            total_path_reward = 0

            for o in sequence:
                dist = manhattan(current_r, current_c, o.ex, o.ey)
                current_t += dist

                total_path_reward += delivery_reward(o, current_t, T)

                current_r, current_c = o.ex, o.ey

            if total_path_reward > max_total_reward:
                max_total_reward = total_path_reward
                best_first_order = sequence[0]

        return best_first_order

    else:

        def urgency_score(o: Order) -> float:
            dist = manhattan(shipper.r, shipper.c, o.ex, o.ey)
            est_delivery = t + dist
            reward_now = delivery_reward(o, est_delivery, T)
            if reward_now <= 0:
                return -999.0
            return reward_now - delivery_reward(o, est_delivery + 1, T)

        return max(candidates, key=urgency_score)


class ACOSystem:
    def __init__(self):
        self.pheromone: Dict[int, Dict[int, float]] = defaultdict(
            lambda: defaultdict(lambda: 1.0)
        )
        self.total_reward = 0.0
        self.on_time_delivered = 0
        self.total_orders_appeared = 0  # để tính % giao
        self.deadline_buffer = DEADLINE_BUFFER

    # tối ưu heuristic

    def _heuristic_with_weight(
        self,
        shipper: Shipper,
        order: Order,
        w_carried: float,
        orders: Dict[int, Order],
        t: int,
        T: int,
        num_pending: int = 20,
    ) -> float:
        dist_pick = manhattan(shipper.r, shipper.c, order.sx, order.sy)
        dist_del = manhattan(order.sx, order.sy, order.ex, order.ey)
        dist_total = dist_pick + dist_del

        cost_before = abs(move_cost(w_carried, shipper.W_max))
        cost_after = abs(move_cost(w_carried + order.w, shipper.W_max))
        total_cost = dist_pick * cost_before + dist_del * cost_after + 1

        time_to_deadline = order.et - t
        slack = time_to_deadline - dist_total
        if slack < 0:
            return 0.001

        est_t = t + dist_total
        r_now = delivery_reward(order, est_t, T)

        delta_t = 5 + (num_pending / 5)
        r_delay = delivery_reward(order, int(est_t + delta_t), T)

        opp_cost = max(0.1, r_now - r_delay)

        drop_rate = opp_cost / (r_now + 1)
        urgency = 1.0 + (drop_rate * 15.0)

        if slack < 5:
            urgency *= 2.0

        cap_pen = 1.0
        if w_carried + order.w > shipper.W_max * CAP_W_THRESHOLD:
            cap_pen = CAP_PEN_W
        if len(shipper.bag) >= shipper.K_max - 1:
            cap_pen *= CAP_PEN_K

        return max(0.01, (r_now * (urgency**2) * cap_pen) / total_cost)

    # xây dựng bộ nhớ đệm cho heuristic
    # tính trước hết tất cả các điểm heuristic cho tất cả shipper × tất cả pending_orders -> lưu vào dictionary

    def _build_eta_cache(
        self,
        shippers: List[Shipper],
        pending_orders: List[Order],
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> Dict:
        cache = {}
        num_pending = len(pending_orders)
        for s in shippers:
            w_carried = sum(orders[oid].w for oid in s.bag if oid in orders)
            for o in pending_orders:
                cache[(s.id, o.id)] = self._heuristic_with_weight(
                    s, o, w_carried, orders, t, T, num_pending
                )
        return cache

    # xây kiến cho nó gán đơn hàng cho shipper theo xác suất dựa trên pheromone và heurisitc
    def _build_solution(
        self,
        shippers: List[Shipper],
        pending_orders: List[Order],
        orders: Dict[int, Order],
        eta_cache: Dict,
    ) -> Dict[int, List[Order]]:
        # kết quả cuối shiper nào nhận đơn nào
        assignment = {s.id: [] for s in shippers}
        unassigned = list(pending_orders)  # danh sách đơn chưa gán
        # bản sao mô tả việc giao đơn mà không làm thay đổi dữ liệu gốc
        temp_bag = {s.id: list(s.bag) for s in shippers}
        temp_weight = {
            s.id: sum(orders[oid].w for oid in s.bag if oid in orders) for s in shippers
        }

        # round-robin: mỗi vòng mỗi shipper được chọn 1 đơn (giảm bias)
        # dùng set id để discard O(1) thay vì list.remove O(n)
        shipper_order = random.sample(shippers, len(shippers))
        unassigned_ids = {o.id for o in unassigned}
        any_assigned = True
        while unassigned_ids and any_assigned:
            any_assigned = False
            for s in shipper_order:
                w_room = s.W_max - temp_weight[s.id]
                k_room = s.K_max - len(temp_bag[s.id])
                if k_room <= 0 or w_room <= 0:
                    continue
                feasible = [
                    o
                    for o in unassigned
                    if o.id in unassigned_ids
                    and not o.picked
                    and not o.delivered
                    and o.w <= w_room
                ]
                if not feasible:
                    continue

                scores = [
                    (self.pheromone[s.id][o.id] ** ALPHA)
                    * (eta_cache.get((s.id, o.id), 0.01) ** BETA)
                    for o in feasible
                ]
                total = sum(scores)
                chosen = (
                    random.choices(
                        feasible, weights=[sc / total for sc in scores], k=1
                    )[0]
                    if total > 0
                    else random.choice(feasible)
                )
                assignment[s.id].append(chosen)
                unassigned_ids.discard(chosen.id)
                temp_bag[s.id].append(chosen.id)
                temp_weight[s.id] += chosen.w
                any_assigned = True

        return assignment

    # hàm đánh giá chất lượng của 1 solution ở hàm trước
    # mục đích: tổng reward ước tính mà các shipper sẽ nhận được nếu họ giao theo đúng thứ tự đơn đã được gán trong solution
    def _evaluate(
        self,
        solution: Dict[int, List[Order]],
        shippers_map: Dict[int, Shipper],
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> Tuple[float, int]:
        total_reward = 0.0
        on_time_count = 0

        for sid, assigned in solution.items():
            s = shippers_map[sid]
            sr, sc = s.r, s.c
            curr_t = t

            for o in assigned:
                dist_to_pick = manhattan(sr, sc, o.sx, o.sy)
                dist_to_del = manhattan(o.sx, o.sy, o.ex, o.ey)
                travel_time = dist_to_pick + dist_to_del

                arrival_t = curr_t + travel_time

                reward = delivery_reward(o, arrival_t, T)
                total_reward += reward

                if arrival_t <= o.et:
                    on_time_count += 1

                sr, sc = o.ex, o.ey
                curr_t = arrival_t

        return total_reward, on_time_count

    # def _two_opt_swap(
    #     self,
    #     solution: Dict[int, List[Order]],
    #     score: float,
    #     shippers_map: Dict[int, Shipper],
    #     orders: Dict[int, Order],
    #     t: int,
    #     T: int,
    #     max_passes: int = 3,
    # ) -> Tuple[Dict[int, List[Order]], float]:

    #     current = {sid: list(chain) for sid, chain in solution.items()}
    #     current_score = score
    #     chain_weight = {sid: sum(o.w for o in chain) for sid, chain in current.items()}

    #     improved = True
    #     pass_count = 0
    #     while improved and pass_count < max_passes:
    #         improved = False
    #         pass_count += 1
    #         sids = list(current.keys())

    #         for i in range(len(sids)):
    #             for j in range(i + 1, len(sids)):
    #                 s1_id, s2_id = sids[i], sids[j]
    #                 s1, s2 = shippers_map[s1_id], shippers_map[s2_id]
    #                 if s1.bag or s2.bag:
    #                     continue
    #                 s1_chain = current[s1_id]
    #                 s2_chain = current[s2_id]
    #                 if not s1_chain or not s2_chain:
    #                     continue

    #                 for idx1 in range(len(s1_chain)):
    #                     for idx2 in range(len(s2_chain)):
    #                         o1 = s1_chain[idx1]
    #                         o2 = s2_chain[idx2]
    #                         w1_new = chain_weight[s1_id] - o1.w + o2.w
    #                         w2_new = chain_weight[s2_id] - o2.w + o1.w
    #                         if w1_new > s1.W_max or w2_new > s2.W_max:
    #                             continue

    #                         s1_chain[idx1] = o2
    #                         s2_chain[idx2] = o1

    #                         new_score = self._evaluate(
    #                             current, shippers_map, orders, t, T
    #                         )
    #                         if new_score > current_score + 0.01:
    #                             current_score = new_score
    #                             chain_weight[s1_id] = w1_new
    #                             chain_weight[s2_id] = w2_new
    #                             improved = True
    #                         else:
    #                             s1_chain[idx1] = o1
    #                             s2_chain[idx2] = o2

    #     return current, current_score

    def _update_pheromone_aco(
        self, solutions: List[Dict], scores: List[float], on_time_counts: List[int]
    ):
        if not scores or max(scores) <= 0:
            return

        best_idx = scores.index(max(scores))
        best_sol = solutions[best_idx]

        for sid in self.pheromone:
            for oid in self.pheromone[sid]:
                self.pheromone[sid][oid] *= 1 - RHO

        total_orders = sum(len(v) for v in best_sol.values())
        if total_orders == 0:
            return

        on_time_ratio = on_time_counts[best_idx] / total_orders
        base_dep = (scores[best_idx] / total_orders) * (1 + on_time_ratio)

        for sid, assigned in best_sol.items():
            for o in assigned:

                specific_dep = base_dep
                if hasattr(o, "service_type") and o.service_type == "URGENT":
                    specific_dep *= 1.5
                self.pheromone[sid][o.id] = min(
                    TAU_MAX, self.pheromone[sid][o.id] + specific_dep
                )

    def deposit(self, shipper_id: int, order_id: int, reward: float):
        if reward <= 0:
            return
        self.pheromone[shipper_id][order_id] = min(
            TAU_MAX,
            self.pheromone[shipper_id][order_id] + reward * DEPOSIT_FACTOR,
        )

    def evaporate(self):
        for sid in self.pheromone:
            for oid in list(self.pheromone[sid].keys()):
                self.pheromone[sid][oid] = max(
                    TAU_MIN, self.pheromone[sid][oid] * (1 - RHO)
                )

    def cleanup(self, delivered_ids: List[int]):
        for sid in self.pheromone:
            for oid in delivered_ids:
                self.pheromone[sid].pop(oid, None)

    def choose_order_for_shipper(
        self,
        shipper: Shipper,
        pending_orders: List[Order],
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> Optional[Order]:

        feasible = [
            o
            for o in pending_orders
            if not o.picked
            and not o.delivered
            and o.et > t + self.deadline_buffer
            and shipper.can_carry(o, orders)
        ]
        if not feasible:
            return None

        # lọc đơn hợp lệ mà ship có thể nhận
        w_carried = sum(orders[oid].w for oid in shipper.bag if oid in orders)

        scores = [
            (self.pheromone[shipper.id][o.id] ** ALPHA)
            * (self._heuristic_with_weight(shipper, o, w_carried, orders, t, T) ** BETA)
            for o in feasible
        ]

        total = sum(scores)
        if total == 0:
            return random.choice(feasible)

        return random.choices(feasible, weights=[s / total for s in scores], k=1)[0]

    def run(self, shippers, pending_orders, orders, t, T, grid):
        if not pending_orders or not shippers:
            return {s.id: [] for s in shippers}

        shippers_map = {s.id: s for s in shippers}
        global_best_sol = None
        global_best_score = -inf
        global_best_on_time = 0

        eta_cache = self._build_eta_cache(shippers, pending_orders, orders, t, T)

        for _ in range(N_ITER):
            solutions, scores, on_time_counts = [], [], []

            for _ in range(N_ANTS):
                sol = self._build_solution(shippers, pending_orders, orders, eta_cache)
                score, on_time_count = self._evaluate(sol, shippers_map, orders, t, T)

                solutions.append(sol)
                scores.append(score)
                on_time_counts.append(on_time_count)

                if score > global_best_score:
                    global_best_score = score
                    global_best_sol = sol
                    global_best_on_time = on_time_count

            self._update_pheromone_aco(solutions, scores, on_time_counts)

        return global_best_sol or {s.id: [] for s in shippers}


class ACOAgent:

    def __init__(self, shipper_id, aco_system):
        self.shipper_id = shipper_id
        self.aco_system = aco_system
        self.target_order_id: Optional[int] = None
        self.current_path: List[Tuple[int, int]] = []
        self.path_target: Optional[Tuple[int, int]] = None
        self.target_orders: List[int] = []

    def get_next_move_cached(
        self, current: Tuple[int, int], target: Tuple[int, int], grid, avoid=None
    ) -> str:

        cache_valid = (
            self.path_target == target
            and self.current_path
            and self.current_path[0] == current
            and (
                not avoid
                or len(self.current_path) < 2
                or self.current_path[1] not in avoid
            )
        )
        if not cache_valid:
            self.current_path = a_star(current, target, grid, avoid)
            self.path_target = target
        # không có đường
        if len(self.current_path) < 2:
            self.current_path = []
            return "S"

        next_pos = self.current_path[1]
        self.current_path.pop(0)  # bỏ vị trí hiện ại

        cx, cy = current
        nx, ny = next_pos

        if not is_valid_cell(next_pos, grid):
            self.current_path = []
            return "S"

        if nx == cx - 1:
            return "U"
        if nx == cx + 1:
            return "D"
        if ny == cy - 1:
            return "L"
        if ny == cy + 1:
            return "R"

        return "S"

    def get_action(self, env: DeliveryEnv, t: int, T: int, avoid_positions=None):

        try:
            shipper = env.shippers[self.shipper_id]
            orders = env.orders
            grid = env.grid

            avoid = avoid_positions or {
                s.position for s in env.shippers if s.id != self.shipper_id
            }

            for oid in shipper.bag:
                order = orders[oid]
                if shipper.can_deliver(order):
                    self.target_order_id = None
                    self.current_path = []  # reset path
                    return "S", 2, oid

            if len(shipper.bag) < shipper.K_max:
                for order in orders.values():
                    if shipper.can_pickup(order, orders):
                        self.target_order_id = None
                        self.current_path = []
                        return "S", 1, order.id

            if shipper.bag:
                target = choose_best_order_to_deliver(shipper, orders, t, T)
                if target:
                    cur = shipper.position
                    goal = (target.ex, target.ey)
                    if len(shipper.bag) < shipper.K_max:
                        base_dist = manhattan(*cur, *goal)
                        w_carried = sum(
                            orders[oid].w for oid in shipper.bag if oid in orders
                        )
                        buf = self.aco_system.deadline_buffer
                        best_pickup = None
                        best_detour = DETOUR_LIMIT + 1
                        for o in orders.values():
                            if o.picked or o.delivered or o.id == target.id:
                                continue
                            if w_carried + o.w > shipper.W_max:
                                continue
                            if o.et <= t + buf:
                                continue
                            detour = (
                                manhattan(*cur, o.sx, o.sy)
                                + manhattan(o.sx, o.sy, *goal)
                                - base_dist
                            )
                            if detour < best_detour:
                                best_detour = detour
                                best_pickup = o
                        if best_pickup is not None:
                            move = self.get_next_move_cached(
                                cur, (best_pickup.sx, best_pickup.sy), grid, avoid
                            )
                            return move, 0, 0
                    move = self.get_next_move_cached(cur, goal, grid, avoid)
                    return move, 0, 0

            buf = self.aco_system.deadline_buffer
            while self.target_orders:
                next_oid = self.target_orders[0]
                target_order = orders.get(next_oid)

                if (
                    target_order
                    and not target_order.picked
                    and not target_order.delivered
                    and target_order.et > t + buf
                    and shipper.can_carry(target_order, orders)
                ):
                    move = self.get_next_move_cached(
                        shipper.position,
                        (target_order.sx, target_order.sy),
                        grid,
                        avoid,
                    )
                    return move, 0, 0

                self.target_orders.pop(0)

            pending_orders = [
                o
                for o in orders.values()
                if not o.picked
                and not o.delivered
                and o.et > t + self.aco_system.deadline_buffer
                and delivery_reward(
                    o,
                    t
                    + manhattan(*shipper.position, o.sx, o.sy)
                    + manhattan(o.sx, o.sy, o.ex, o.ey),
                    T,
                )
                > 0
            ]

            if pending_orders:
                w_carried = sum(orders[oid].w for oid in shipper.bag if oid in orders)
                best_order = None
                best_score = -inf

                close_orders = sorted(
                    pending_orders,
                    key=lambda o: manhattan(*shipper.position, o.sx, o.sy),
                )[:15]

                for o in close_orders:
                    if not shipper.can_carry(o, orders):
                        continue

                    dist = manhattan(*shipper.position, o.sx, o.sy)
                    eta = self.aco_system._heuristic_with_weight(
                        shipper, o, w_carried, orders, t, T
                    )
                    pheromone = self.aco_system.pheromone[shipper.id][o.id]
                    aco_score = (pheromone**ALPHA) * (eta**BETA)

                    distance_factor = 1.0 / (1.0 + dist * 0.12)
                    blended = aco_score * distance_factor

                    if blended > best_score:
                        best_score = blended
                        best_order = o

                if best_order:
                    self.target_order_id = best_order.id
                    self.current_path = []
                    move = self.get_next_move_cached(
                        shipper.position, (best_order.sx, best_order.sy), grid, avoid
                    )
                    return move, 0, 0

            N = len(grid)
            center = (N // 2, N // 2)
            if shipper.position != center:
                move = self.get_next_move_cached(shipper.position, center, grid, avoid)
                return move, 0, 0

            return "S", 0, 0

        except Exception as e:
            print(f"[ACOAgent {self.shipper_id}] Error at t={t}: {e}")
            return "S", 0, 0


class ACOSolver(Solver):
    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.method_name = "ACO_Adaptive_v2"
        self.aco_system = ACOSystem()
        random.seed(42)
        self.aco_system.deadline_buffer = DEADLINE_BUFFER

        # mỗi shipper có 1 ACOAgents riêng quyết định hành động
        self.agents: Dict[int, ACOAgent] = {
            s.id: ACOAgent(s.id, self.aco_system) for s in env.shippers
        }

        self.last_global_assign = -999
        self.current_interval = 9  # tần suất chạy
        self.prev_pending = 0
        self.total_reward = 0.0
        self.on_time_delivered = 0

    def _calculate_dynamic_interval(
        self, num_shippers: int, num_pending: int, delta_pending: int
    ) -> int:

        is_high_pressure = (num_pending / max(1, num_shippers)) > 4.0

        if num_shippers <= 5:
            base = 6 if is_high_pressure else 11
        elif num_shippers <= 10:
            base = 8 if is_high_pressure else 12
        else:
            base = 12

        if num_pending > 35:
            base -= 3
        elif num_pending < 10:
            base += 3

        if delta_pending >= 5:
            base -= 2

        return max(3, min(15, base))

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            t = obs.get("t", 0)
            T = obs.get("T", 1000)

            pending_orders_list = [
                o
                for o in self.env.orders.values()
                if not o.picked and not o.delivered and o.et > t
            ]
            num_pending = len(pending_orders_list)
            num_shippers = len(self.env.shippers)
            delta_pending = max(0, num_pending - self.prev_pending)

            self.current_interval = self._calculate_dynamic_interval(
                num_shippers, num_pending, delta_pending
            )

            run_global = False
            if (t - self.last_global_assign >= self.current_interval) or t <= 10:
                if num_pending >= 1:
                    run_global = True

            if run_global:
                pending_for_aco = [
                    o
                    for o in pending_orders_list
                    if o.et > t + self.aco_system.deadline_buffer
                ]
                if pending_for_aco:
                    global_assignment = self.aco_system.run(
                        shippers=list(self.env.shippers),
                        pending_orders=pending_for_aco,
                        orders=self.env.orders,
                        t=t,
                        T=T,
                        grid=self.env.grid,
                    )
                    for sid, assigned_list in global_assignment.items():
                        if assigned_list and sid in self.agents:
                            agent = self.agents[sid]
                            chain_len = 4 if num_pending > 15 else 3

                            new_targets = [o.id for o in assigned_list[:chain_len]]

                            if not agent.target_orders:
                                agent.target_orders = new_targets
                            else:
                                current = agent.target_orders[0]
                                current_order = self.env.orders.get(current)
                                if (
                                    current_order
                                    and not current_order.picked
                                    and not current_order.delivered
                                ):
                                    agent.target_orders = [current] + [
                                        oid for oid in new_targets if oid != current
                                    ][: chain_len - 1]
                                else:
                                    agent.target_orders = new_targets

                            agent.target_order_id = (
                                agent.target_orders[0] if agent.target_orders else None
                            )

                    self.last_global_assign = t

            avoid_positions = {s.position for s in self.env.shippers}
            actions = {
                sid: agent.get_action(self.env, t, T)
                for sid, agent in self.agents.items()
            }

            obs, rewards, done, _ = self.env.step(actions)

            just_delivered = [
                o for o in self.env.orders.values() if o.delivered and o.deliver_t == t
            ]

            for order in just_delivered:
                reward = delivery_reward(order, order.deliver_t, T)
                self.total_reward += reward
                if order.deliver_t <= order.et:
                    self.on_time_delivered += 1
                if order.carrier >= 0:
                    self.aco_system.deposit(order.carrier, order.id, reward)

            self.aco_system.evaporate()
            self.aco_system.cleanup([o.id for o in just_delivered])

            total_delivered_count = sum(
                1 for o in self.env.orders.values() if o.delivered
            )
            total_orders = len(self.env.orders)
            delivery_rate = (
                (total_delivered_count / total_orders * 100) if total_orders > 0 else 0
            )
            on_time_rate = (
                (self.on_time_delivered / total_delivered_count * 100)
                if total_delivered_count > 0
                else 0
            )

            self.prev_pending = num_pending

            # ====================== LOG ======================
            if t % 25 == 0 or run_global or len(just_delivered) > 0:
                status = "ON" if run_global else "off"
                print(
                    f"t={t:4d} | Reward={self.total_reward:8.1f} | "
                    f"Delivered={total_delivered_count:3d}/{total_orders} ({delivery_rate:5.1f}%) | "
                    f"OnTime={on_time_rate:5.1f}% | Pending={num_pending:3d} | "
                    f"Interval={self.current_interval} | Global={status}"
                )

            if done:
                break

        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
