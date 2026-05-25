from __future__ import annotations

import time
from collections import OrderedDict, deque
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
_OPPOSITE: Dict[str, str] = {"U": "D", "D": "U", "L": "R", "R": "L", "S": "S"}

# Scoring weights
_W_DIST    = 0.10   # penalty mỗi bước khoảng cách
_W_PRIO    = 5.0    # bonus theo priority
_W_CLUSTER = 10.0   # bonus nếu cùng điểm giao với đơn đang mang (sweet spot local)
_URGENT_SLACK = 8   # bước còn trước deadline → coi là urgent (bump để giảm late)
_URGENCY_WEIGHT = 30.0  # bonus urgency trong delivery_score
_ENDGAME_FRAC = 0.10  # phần cuối T → ưu tiên giao hàng
_BFS_CACHE_CAP = 256  # số cell-gốc tối đa cache BFS toàn cây


def _est_delivery_t(t: int, d_to_pickup: int, d_trip: int) -> int:
    """Ước tính thời điểm giao hàng: pickup + trip, trừ 1 bước vì move+pickup cùng timestep."""
    return t + max(d_to_pickup - 1, 0) + d_trip


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
        # BFS cache theo START cell — preserve exact behavior của old per-pair BFS:
        # cùng neighbor expansion order, cùng first_move trên đường đi.
        # Mỗi lần BFS từ start sinh ra toàn bộ cây shortest-path; mọi câu hỏi
        # _distance(start, goal_i) cho nhiều goal_i đều free.
        self._bfs_cache: "OrderedDict[Position, Tuple[Dict[Position, int], Dict[Position, Move]]]" = OrderedDict()

    # ------------------------------------------------------------------
    # BFS rooted at start — tự viết, không import từ greedy_bfs
    # ------------------------------------------------------------------

    def _bfs_from(self, start: Position) -> Tuple[Dict[Position, int], Dict[Position, Move]]:
        """BFS rooted at `start`; trả (dist tới mọi cell, first_move từ start đến mỗi cell)."""
        cached = self._bfs_cache.get(start)
        if cached is not None:
            self._bfs_cache.move_to_end(start)
            return cached
        if not is_valid_cell(start, self.grid):
            empty: Tuple[Dict[Position, int], Dict[Position, Move]] = ({}, {})
            self._bfs_cache[start] = empty
            self._evict_bfs_cache()
            return empty

        dist: Dict[Position, int] = {start: 0}
        first_move: Dict[Position, Move] = {start: "S"}
        queue: deque[Position] = deque([start])
        grid = self.grid
        n_rows = len(grid)
        n_cols = len(grid[0]) if n_rows else 0
        while queue:
            cur = queue.popleft()
            d_next = dist[cur] + 1
            cur_first = first_move[cur]
            cr, cc = cur
            for mv, (dr, dc) in _MOVE_DELTAS.items():
                nr, nc = cr + dr, cc + dc
                if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                    continue
                if grid[nr][nc] != 0:
                    continue
                nxt = (nr, nc)
                if nxt in dist:
                    continue
                dist[nxt] = d_next
                # Propagate first move: nếu cur == start thì first move là mv,
                # ngược lại kế thừa từ cur. Khớp đúng tie-breaking BFS từ start.
                first_move[nxt] = mv if cur == start else cur_first
                queue.append(nxt)

        self._bfs_cache[start] = (dist, first_move)
        self._evict_bfs_cache()
        return self._bfs_cache[start]

    def _evict_bfs_cache(self) -> None:
        while len(self._bfs_cache) > _BFS_CACHE_CAP:
            self._bfs_cache.popitem(last=False)

    def _distance(self, a: Position, b: Position) -> int:
        if a == b:
            return 0
        dist, _ = self._bfs_from(a)
        return dist.get(b, INF)

    def _next_move(self, a: Position, b: Position) -> Move:
        if a == b:
            return "S"
        _, first_move = self._bfs_from(a)
        return first_move.get(b, "S")

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

        # Fix: tính đúng thời điểm giao — move+pickup cùng timestep nên trừ 1
        est_delivery_t = _est_delivery_t(t, d_pickup, d_trip)
        reward = delivery_reward(order, est_delivery_t, T)
        if reward <= 0:
            return -INF

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
        return reward + _W_PRIO * order.p - _W_DIST * d + _URGENCY_WEIGHT / urgency

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
            # Fix: dùng max(d-1,0) cho đúng với env (move+deliver cùng bước)
            if d < INF and t + max(d - 1, 0) >= orders[oid].et - _URGENT_SLACK:
                return True
        return False

    def _is_endgame(self, t: int, T: int) -> bool:
        """True nếu đang ở giai đoạn cuối episode."""
        return T - t <= max(10, int(T * _ENDGAME_FRAC))

    def _idle_reposition(self, shipper: Shipper, orders: Dict[int, Order]) -> Action:
        """Di chuyển về centroid các đơn chưa nhặt khi rảnh."""
        pending = [o for o in orders.values() if not o.picked and not o.delivered]
        if not pending:
            return ("S", 0)
        cr = round(sum(o.sx for o in pending) / len(pending))
        cc = round(sum(o.sy for o in pending) / len(pending))
        goal: Position = (cr, cc)
        if goal == shipper.position:
            return ("S", 0)
        mv = self._next_move(shipper.position, goal)
        return (mv, 0) if mv != "S" else ("S", 0)

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

        # 4. Endgame: gần cuối T → giao hết hàng đang mang, không nhặt thêm
        if self._is_endgame(t, T) and shipper.bag:
            target = self._best_delivery_target(shipper, orders, t, T)
            if target:
                goal = (target.ex, target.ey)
                mv   = self._next_move(pos, goal)
                nxt  = valid_next_pos(pos, mv, self.grid)
                return (mv, 2) if nxt == goal else (mv, 0)

        # 5. Có đơn được phân công → so sánh với delivery trước khi quyết
        assigned_oid = assignment.get(shipper.id)
        if assigned_oid is not None:
            order = orders.get(assigned_oid)
            if order and not order.picked and not order.delivered:
                pickup_score = self._pickup_score(shipper, order, orders, t, T)
                # Nếu bag có hàng và delivery score tốt hơn rõ rệt thì giao trước
                if shipper.bag:
                    target = self._best_delivery_target(shipper, orders, t, T)
                    if target:
                        d_score = self._delivery_score(shipper, target, t, T)
                        if d_score > pickup_score + 5.0:
                            goal = (target.ex, target.ey)
                            mv   = self._next_move(pos, goal)
                            nxt  = valid_next_pos(pos, mv, self.grid)
                            return (mv, 2) if nxt == goal else (mv, 0)
                goal = (order.sx, order.sy)
                mv   = self._next_move(pos, goal)
                nxt  = valid_next_pos(pos, mv, self.grid)
                return (mv, 1) if nxt == goal else (mv, 0)

        # 6. Còn đơn trong túi → đi giao
        if shipper.bag:
            target = self._best_delivery_target(shipper, orders, t, T)
            if target:
                goal = (target.ex, target.ey)
                mv   = self._next_move(pos, goal)
                nxt  = valid_next_pos(pos, mv, self.grid)
                return (mv, 2) if nxt == goal else (mv, 0)

        # 7. Rảnh + không endgame → tìm đơn chưa ai nhận
        if not self._is_endgame(t, T):
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

        # 8. Hoàn toàn rảnh → reposition về vùng có đơn
        return self._idle_reposition(shipper, orders)

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
                alt_found = False
                for mv, (dr, dc) in _MOVE_DELTAS.items():
                    alt = (old[0] + dr, old[1] + dc)
                    if is_valid_cell(alt, self.grid) and alt not in occupied:
                        resolved[sid] = (mv, 0)
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
