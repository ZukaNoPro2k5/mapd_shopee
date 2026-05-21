from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from env import (
    DeliveryEnv, Order, Shipper,
    delivery_reward, is_valid_cell, valid_next_pos,
)
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]

INF = 10 ** 9

_MOVE_DELTAS: Dict[str, Tuple[int, int]] = {
    "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1),
}
_MOVES = tuple(_MOVE_DELTAS.keys())

# Scoring weights
_W_DIST    = 0.10   # penalty mỗi bước khoảng cách
_W_PRIO    = 5.0    # bonus theo priority
_W_CLUSTER = 6.0    # bonus nếu cùng điểm giao với đơn đang mang
_URGENT_SLACK = 5   # bước còn trước deadline → coi là urgent


class VRPOrToolsSolver(Solver):
    """
    Rolling-horizon VRP — thuần Python standard library, không cần OR-Tools.

    Mỗi timestep:
      1. Global assignment (stateless — reset mỗi bước): phân công đơn chưa nhặt
         cho shipper, không trùng lặp, ưu tiên cặp (shipper, đơn) có điểm cao nhất.
      2. Quyết định action từng shipper theo thứ tự ưu tiên.
      3. Giải quyết va chạm + phá deadlock: shipper bị chặn thử ô bên cạnh.
    """

    method_name = "VRPOrToolsSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._dist_cache: Dict[Tuple[Position, Position], int] = {}
        self._move_cache: Dict[Tuple[Position, Position], Move] = {}

    # ------------------------------------------------------------------
    # BFS — tự viết, không import từ greedy_bfs
    # ------------------------------------------------------------------

    def _bfs(self, start: Position, goal: Position) -> Tuple[int, Move]:
        """
        Tìm đường ngắn nhất trên lưới (BFS).
        Trả (khoảng_cách, bước_đầu_tiên_từ_start). INF nếu không đi được.
        """
        if start == goal:
            return 0, "S"
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return INF, "S"

        queue: deque[Position] = deque([start])
        # parent[node] = (node_trước_đó, hướng_đi_để_đến_node)
        parent: Dict[Position, Tuple[Position, Move]] = {}

        while queue:
            cur = queue.popleft()
            for mv, (dr, dc) in _MOVE_DELTAS.items():
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt == start or nxt in parent or not is_valid_cell(nxt, self.grid):
                    continue
                parent[nxt] = (cur, mv)
                if nxt == goal:
                    # Trace ngược từ goal để lấy bước đầu tiên và khoảng cách
                    dist, node, first_move = 0, goal, mv
                    while node != start:
                        prev, m = parent[node]
                        first_move = m
                        dist += 1
                        node = prev
                    return dist, first_move
                queue.append(nxt)

        return INF, "S"

    def _distance(self, a: Position, b: Position) -> int:
        if a == b:
            return 0
        key = (a, b)
        if key not in self._dist_cache:
            d, mv = self._bfs(a, b)
            self._dist_cache[key] = d
            self._move_cache[key] = mv
        return self._dist_cache[key]

    def _next_move(self, a: Position, b: Position) -> Move:
        if a == b:
            return "S"
        key = (a, b)
        if key not in self._move_cache:
            d, mv = self._bfs(a, b)
            self._dist_cache[key] = d
            self._move_cache[key] = mv
        return self._move_cache[key]

    # ------------------------------------------------------------------
    # Tính điểm
    # ------------------------------------------------------------------

    def _pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> float:
        """
        Điểm khi phân công đơn (chưa nhặt) cho shipper.
        Dương = đáng làm; âm = không nên. -INF = không thể.
        """
        if not shipper.can_carry(order, orders):
            return -INF

        d_pickup = self._distance(shipper.position, (order.sx, order.sy))
        d_trip   = self._distance((order.sx, order.sy), (order.ex, order.ey))
        if d_pickup >= INF or d_trip >= INF:
            return -INF

        est_delivery_t = t + d_pickup + d_trip
        reward = delivery_reward(order, est_delivery_t, T)
        if reward <= 0:
            return -INF  # giao cũng không có lãi

        # Bonus: cùng điểm giao với đơn đang mang → giao được 1 lượt
        cluster_bonus = _W_CLUSTER * sum(
            1 for oid in shipper.bag
            if oid in orders and (orders[oid].ex, orders[oid].ey) == (order.ex, order.ey)
        )

        return (
            reward
            + _W_PRIO   * order.p
            + cluster_bonus
            - _W_DIST   * (d_pickup + d_trip)
        )

    def _delivery_score(
        self,
        shipper: Shipper,
        order: Order,
        t: int,
        T: int,
    ) -> float:
        """Điểm của việc chọn đơn trong túi làm mục tiêu giao tiếp theo."""
        d = self._distance(shipper.position, (order.ex, order.ey))
        if d >= INF:
            return -INF
        est_t   = t + max(d - 1, 0)
        reward  = delivery_reward(order, est_t, T)
        # Ưu tiên đơn deadline gần (urgency score)
        urgency = max(1, order.et - t)
        return reward + _W_PRIO * order.p - _W_DIST * d + 10.0 / urgency

    # ------------------------------------------------------------------
    # Global assignment (stateless — reset mỗi bước)
    # ------------------------------------------------------------------

    def _global_assign(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> Dict[int, int]:
        """
        Phân công đơn chưa nhặt cho shipper, không trùng lặp.
        Trả dict {shipper_id -> order_id}.

        Shippers đang có túi đầy hoặc có đơn urgent sẽ không nhận assignment mới
        (họ nên tập trung giao hàng trước).
        """
        shipper_map = {s.id: s for s in shippers}

        # Tính điểm mọi cặp (shipper chưa quá tải, đơn chưa nhặt)
        candidates: List[Tuple[float, int, int]] = []
        for order in orders.values():
            if order.picked or order.delivered:
                continue
            for sid, shipper in shipper_map.items():
                # Không gán cho shipper đang bị urgent delivery
                if self._is_urgent(shipper, orders, t):
                    continue
                score = self._pickup_score(shipper, order, orders, t, T)
                if score > -INF / 2:
                    candidates.append((score, sid, order.id))

        candidates.sort(key=lambda x: -x[0])

        assignment: Dict[int, int] = {}
        used_sids: Set[int] = set()
        used_oids: Set[int] = set()

        for score, sid, oid in candidates:
            if sid in used_sids or oid in used_oids:
                continue
            if shipper_map[sid].can_carry(orders[oid], orders):
                assignment[sid] = oid
                used_sids.add(sid)
                used_oids.add(oid)

        return assignment

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_delivery_target(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> Optional[Order]:
        """Chọn đơn trong túi để giao tiếp theo (điểm cao nhất)."""
        best_order, best_score = None, -INF
        for oid in shipper.bag:
            if oid not in orders or orders[oid].delivered:
                continue
            score = self._delivery_score(shipper, orders[oid], t, T)
            if score > best_score:
                best_score = score
                best_order = orders[oid]
        return best_order

    def _is_urgent(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
    ) -> bool:
        """True nếu có đơn trong túi sắp trễ hạn cần giao ngay."""
        for oid in shipper.bag:
            if oid not in orders:
                continue
            d = self._distance(shipper.position, (orders[oid].ex, orders[oid].ey))
            if d < INF and t + d >= orders[oid].et - _URGENT_SLACK:
                return True
        return False

    # ------------------------------------------------------------------
    # Quyết định action từng shipper
    # ------------------------------------------------------------------

    def _action_for(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        assignment: Dict[int, int],
        t: int,
        T: int,
        all_assigned_oids: Set[int],
    ) -> Action:
        pos = shipper.position

        # 1. Đang đứng đúng điểm giao → giao ngay
        for oid in shipper.bag:
            if oid in orders and not orders[oid].delivered:
                if (orders[oid].ex, orders[oid].ey) == pos:
                    return ("S", 2)

        # 2. Đơn trong túi sắp trễ hạn → ưu tiên giao
        if self._is_urgent(shipper, orders, t):
            target = self._best_delivery_target(shipper, orders, t, T)
            if target:
                goal = (target.ex, target.ey)
                mv   = self._next_move(pos, goal)
                nxt  = valid_next_pos(pos, mv, self.grid)
                return (mv, 2) if nxt == goal else (mv, 0)

        # 3. Túi đầy → phải đi giao
        if len(shipper.bag) >= shipper.K_max:
            target = self._best_delivery_target(shipper, orders, t, T)
            if target:
                goal = (target.ex, target.ey)
                mv   = self._next_move(pos, goal)
                nxt  = valid_next_pos(pos, mv, self.grid)
                return (mv, 2) if nxt == goal else (mv, 0)

        # 4. Có đơn được phân công → đi nhặt
        assigned_oid = assignment.get(shipper.id)
        if assigned_oid is not None:
            order = orders.get(assigned_oid)
            if order and not order.picked and not order.delivered:
                goal = (order.sx, order.sy)
                mv   = self._next_move(pos, goal)
                nxt  = valid_next_pos(pos, mv, self.grid)
                return (mv, 1) if nxt == goal else (mv, 0)

        # 5. Còn đơn trong túi → đi giao
        if shipper.bag:
            target = self._best_delivery_target(shipper, orders, t, T)
            if target:
                goal = (target.ex, target.ey)
                mv   = self._next_move(pos, goal)
                nxt  = valid_next_pos(pos, mv, self.grid)
                return (mv, 2) if nxt == goal else (mv, 0)

        # 6. Rảnh → tìm đơn chưa ai nhận
        best_order, best_score = None, -INF
        for order in orders.values():
            if order.picked or order.delivered or order.id in all_assigned_oids:
                continue
            score = self._pickup_score(shipper, order, orders, t, T)
            if score > best_score:
                best_score = score
                best_order = order

        if best_order is not None:
            goal = (best_order.sx, best_order.sy)
            mv   = self._next_move(pos, goal)
            nxt  = valid_next_pos(pos, mv, self.grid)
            return (mv, 1) if nxt == goal else (mv, 0)

        return ("S", 0)

    # ------------------------------------------------------------------
    # Giải quyết va chạm + phá deadlock
    # ------------------------------------------------------------------

    def _resolve(
        self,
        actions: Dict[int, Action],
        shippers: List[Shipper],
    ) -> Dict[int, Action]:
        """
        Shipper id nhỏ hơn được ưu tiên giữ ô (giống env._apply_moves).
        Khi bị chặn, thử ô bên cạnh để tránh deadlock thay vì đứng yên.
        """
        shipper_by_id = {s.id: s for s in shippers}
        old_pos = {s.id: s.position for s in shippers}
        desired: Dict[int, Position] = {
            sid: valid_next_pos(shipper_by_id[sid].position, mv, self.grid)
            for sid, (mv, _) in actions.items()
        }

        occupied: Set[Position] = set(old_pos.values())
        resolved = dict(actions)

        for sid in sorted(shipper_by_id):
            old = old_pos[sid]
            tgt = desired.get(sid, old)
            occupied.discard(old)

            if tgt in occupied:
                # Bị chặn: thử ô bên cạnh (tránh deadlock)
                # Ưu tiên: ô không bị chiếm, không phải ô hiện tại của shipper khác
                alt_found = False
                for mv, (dr, dc) in _MOVE_DELTAS.items():
                    alt = (old[0] + dr, old[1] + dc)
                    if is_valid_cell(alt, self.grid) and alt not in occupied:
                        resolved[sid] = (mv, 0)  # di chuyển, không làm cargo_op
                        tgt = alt
                        alt_found = True
                        break
                if not alt_found:
                    resolved[sid] = ("S", actions[sid][1])
                    tgt = old

            occupied.add(tgt)

        return resolved

    # ------------------------------------------------------------------
    # Vòng lặp chính
    # ------------------------------------------------------------------

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            orders: Dict[int, Order]    = obs["orders"]
            shippers: List[Shipper]     = obs["shippers"]
            t = int(obs["t"])
            T = int(obs["T"])

            # Bước 1: Global assignment (stateless — reset mỗi bước)
            assignment = self._global_assign(shippers, orders, t, T)
            all_assigned_oids: Set[int] = set(assignment.values())

            # Bước 2: Quyết định action từng shipper
            actions: Dict[int, Action] = {}
            for shipper in sorted(shippers, key=lambda s: s.id):
                actions[shipper.id] = self._action_for(
                    shipper, orders, assignment, t, T, all_assigned_oids
                )

            # Bước 3: Giải quyết va chạm + phá deadlock
            actions = self._resolve(actions, shippers)

            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
