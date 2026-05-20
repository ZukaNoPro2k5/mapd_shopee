from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]
ScoredOrder = Tuple[Order, float]

INF = 10**9
NEG_INF = -10**18
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")

MIN_PICKUP_SCORE = 0.0
PICKUP_OVER_DELIVERY_MARGIN = 4.0
URGENT_SLACK_BUFFER = 4

DISTANCE_PENALTY = 0.08
LATE_PENALTY = 18.0
PRIORITY_BONUS = 3.0
SAME_DESTINATION_BONUS = 4.0
PICKUP_WAIT_PENALTY = 0.03


class GreedyBFS(Solver):
    """
    Self-contained Greedy BFS dispatcher for the online MAPD environment.

    This file deliberately owns its pathfinding and scoring logic. Other methods
    may copy ideas, but should not import this solver's internals, so Phase 2 can
    safely submit/run one method without coupling algorithm files together.
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}

    # ------------------------------------------------------------------
    # Local BFS utilities: intentionally not shared across solver files.
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}

        while queue:
            current = queue.popleft()
            if current == goal:
                return parent

            for move, nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                parent[nxt] = (current, move)
                queue.append(nxt)

        return None

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0

        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF

        distance = 0
        current = goal
        while current != start:
            previous, _ = parent[current]
            if previous is None:
                self._distance_cache[key] = INF
                return INF
            current = previous
            distance += 1

        self._distance_cache[key] = distance
        return distance

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"

        key = (start, goal)
        if key in self._next_move_cache:
            return self._next_move_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._next_move_cache[key] = "S"
            return "S"

        current = goal
        while True:
            previous, move = parent[current]
            if previous is None:
                self._next_move_cache[key] = "S"
                return "S"
            if previous == start:
                self._next_move_cache[key] = move
                return move
            current = previous

    # ------------------------------------------------------------------
    # Local scoring helpers: follow env.py/code reward and move semantics.
    # ------------------------------------------------------------------
    def _delivery_time_after_distance(self, t: int, distance: int) -> int:
        # In env.step, final move and delivery can happen in the same timestep.
        if distance >= INF:
            return INF
        return t + max(distance - 1, 0)

    def _carried_orders(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

    def _same_destination_count(
        self,
        shipper: Shipper,
        target: Position,
        orders: Dict[int, Order],
    ) -> int:
        return sum(
            1
            for order in self._carried_orders(shipper, orders)
            if (order.ex, order.ey) == target
        )

    def _delivery_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> float:
        if order.id not in shipper.bag or order.delivered:
            return NEG_INF

        destination = (order.ex, order.ey)
        distance = self._distance(shipper.position, destination)
        if distance >= INF:
            return NEG_INF

        delivery_t = self._delivery_time_after_distance(t, distance)
        reward = delivery_reward(order, delivery_t, T)
        lateness = max(0, delivery_t - order.et)
        same_destination = self._same_destination_count(shipper, destination, orders)

        return (
            reward
            + PRIORITY_BONUS * order.p
            + SAME_DESTINATION_BONUS * max(0, same_destination - 1)
            - DISTANCE_PENALTY * distance
            - LATE_PENALTY * lateness
        )

    def _pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> float:
        if not shipper.can_carry(order, orders):
            return NEG_INF

        pickup = (order.sx, order.sy)
        destination = (order.ex, order.ey)
        distance_to_pickup = self._distance(shipper.position, pickup)
        trip_distance = self._distance(pickup, destination)
        if distance_to_pickup >= INF or trip_distance >= INF:
            return NEG_INF

        delivery_t = t + max(distance_to_pickup - 1, 0) + trip_distance
        reward = delivery_reward(order, delivery_t, T)
        lateness = max(0, delivery_t - order.et)
        same_destination = self._same_destination_count(shipper, destination, orders)

        return (
            reward
            + PRIORITY_BONUS * order.p
            + SAME_DESTINATION_BONUS * same_destination
            - DISTANCE_PENALTY * (distance_to_pickup + trip_distance)
            - LATE_PENALTY * lateness
            - PICKUP_WAIT_PENALTY * max(0, t - order.appear_t)
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _has_delivery_at_position(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        position: Position,
    ) -> bool:
        return any((order.ex, order.ey) == position for order in self._carried_orders(shipper, orders))

    def _bag_is_full(self, shipper: Shipper) -> bool:
        return len(shipper.bag) >= shipper.K_max

    def _has_urgent_carried_order(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
    ) -> bool:
        for order in self._carried_orders(shipper, orders):
            distance = self._distance(shipper.position, (order.ex, order.ey))
            if distance >= INF:
                continue
            estimated_delivery_t = self._delivery_time_after_distance(t, distance)
            if estimated_delivery_t + URGENT_SLACK_BUFFER >= order.et:
                return True
        return False

    def _can_pick_more(self, shipper: Shipper, orders: Dict[int, Order]) -> bool:
        return any(shipper.can_carry(order, orders) for order in orders.values())

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _select_delivery(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        T: int,
    ) -> Optional[ScoredOrder]:
        best_order: Optional[Order] = None
        best_score = NEG_INF

        for order in self._carried_orders(shipper, orders):
            score = self._delivery_score(shipper, order, orders, t, T)
            if score > best_score or (
                score == best_score
                and best_order is not None
                and (order.et, -order.p, order.id) < (best_order.et, -best_order.p, best_order.id)
            ):
                best_order = order
                best_score = score

        if best_order is None:
            return None
        return best_order, best_score

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
        reserved_pickup_cells: Set[Position],
        t: int,
        T: int,
    ) -> Optional[ScoredOrder]:
        best_order: Optional[Order] = None
        best_score = NEG_INF

        for order in orders.values():
            pickup = (order.sx, order.sy)
            if order.id in reserved_order_ids or pickup in reserved_pickup_cells:
                continue
            score = self._pickup_score(shipper, order, orders, t, T)
            if score <= NEG_INF / 2:
                continue
            if score > best_score or (
                score == best_score
                and best_order is not None
                and (-order.p, order.et, order.id) < (-best_order.p, best_order.et, best_order.id)
            ):
                best_order = order
                best_score = score

        if best_order is None:
            return None
        return best_order, best_score

    # ------------------------------------------------------------------
    # Action construction
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        return move, next_position

    def _delivery_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 1) if next_position == goal else (move, 0)

    def _blocked_action(self, shipper: Shipper, orders: Dict[int, Order]) -> Action:
        if self._has_delivery_at_position(shipper, orders, shipper.position):
            return ("S", 2)
        return ("S", 0)

    # ------------------------------------------------------------------
    # One-step conflict handling
    # ------------------------------------------------------------------
    def _find_yield_move(
        self,
        blocker: Shipper,
        old_positions: Dict[int, Position],
        desired_positions: Dict[int, Position],
        extra_forbidden: Set[Position],
    ) -> Optional[Move]:
        forbidden = set(old_positions.values())
        forbidden.discard(blocker.position)
        forbidden.update(pos for sid, pos in desired_positions.items() if sid != blocker.id)
        forbidden.update(extra_forbidden)

        for move, next_position in self._neighbors(blocker.position):
            if next_position not in forbidden:
                return move
        return None

    def _yield_idle_blockers(
        self,
        actions: Dict[int, Action],
        shippers: List[Shipper],
    ) -> Dict[int, Action]:
        """
        Ask empty non-delivering shippers to step away from a carried delivery path.
        This is local to Greedy and avoids permanent blocking without shared CBS code.
        """
        resolved = dict(actions)
        shipper_by_id = {shipper.id: shipper for shipper in shippers}
        old_positions = {shipper.id: shipper.position for shipper in shippers}
        occupant_by_pos = {shipper.position: shipper.id for shipper in shippers}

        desired_positions = {
            sid: valid_next_pos(shipper.position, resolved.get(sid, ("S", 0))[0], self.grid)
            for sid, shipper in shipper_by_id.items()
        }

        for sid in sorted(shipper_by_id):
            shipper = shipper_by_id[sid]
            target = desired_positions[sid]
            if target == shipper.position:
                continue

            blocker_id = occupant_by_pos.get(target)
            if blocker_id is None or blocker_id == sid:
                continue

            blocker_action = resolved.get(blocker_id, ("S", 0))
            _, blocker_op = blocker_action
            blocker = shipper_by_id[blocker_id]
            if blocker.bag or blocker_op == 2:
                continue

            yield_move = self._find_yield_move(
                blocker,
                old_positions,
                desired_positions,
                extra_forbidden={shipper.position, target},
            )
            if yield_move is None:
                continue

            resolved[blocker_id] = (yield_move, 0)
            desired_positions[blocker_id] = valid_next_pos(blocker.position, yield_move, self.grid)

        return resolved

    def _resolve_conflicts(
        self,
        actions: Dict[int, Action],
        shippers: List[Shipper],
        orders: Dict[int, Order],
    ) -> Dict[int, Action]:
        actions = self._yield_idle_blockers(actions, shippers)
        shipper_by_id = {shipper.id: shipper for shipper in shippers}
        old_positions = {shipper.id: shipper.position for shipper in shippers}
        occupied = set(old_positions.values())
        desired_positions: Dict[int, Position] = {}
        actual_positions: Dict[int, Position] = {}

        for sid, shipper in shipper_by_id.items():
            move, _ = actions.get(sid, ("S", 0))
            desired_positions[sid] = valid_next_pos(shipper.position, move, self.grid)

        for sid in sorted(shipper_by_id):
            shipper = shipper_by_id[sid]
            old = old_positions[sid]
            target = desired_positions[sid]
            occupied.discard(old)
            if target in occupied:
                target = old
            occupied.add(target)
            actual_positions[sid] = target

        resolved = dict(actions)
        for sid, desired in desired_positions.items():
            if actual_positions[sid] != desired:
                resolved[sid] = self._blocked_action(shipper_by_id[sid], orders)
        return resolved

    # ------------------------------------------------------------------
    # Greedy policy
    # ------------------------------------------------------------------
    def _decide_shipper_action(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
        reserved_pickup_cells: Set[Position],
        t: int,
        T: int,
    ) -> Action:
        if self._has_delivery_at_position(shipper, orders, shipper.position):
            return ("S", 2)

        delivery_candidate = self._select_delivery(shipper, orders, t, T)
        if delivery_candidate is not None:
            delivery_order, delivery_score = delivery_candidate
        else:
            delivery_order, delivery_score = None, NEG_INF

        must_deliver = delivery_order is not None and (
            self._bag_is_full(shipper) or self._has_urgent_carried_order(shipper, orders, t)
        )
        if must_deliver:
            return self._delivery_action(shipper, delivery_order)

        pickup_candidate = None
        if self._can_pick_more(shipper, orders):
            pickup_candidate = self._select_pickup(
                shipper,
                orders,
                reserved_order_ids,
                reserved_pickup_cells,
                t,
                T,
            )

        if pickup_candidate is not None:
            pickup_order, pickup_score = pickup_candidate
            pickup_is_worth_it = pickup_score >= MIN_PICKUP_SCORE and (
                delivery_order is None or pickup_score >= delivery_score + PICKUP_OVER_DELIVERY_MARGIN
            )
            if pickup_is_worth_it:
                reserved_order_ids.add(pickup_order.id)
                reserved_pickup_cells.add((pickup_order.sx, pickup_order.sy))
                return self._pickup_action(shipper, pickup_order)

        if delivery_order is not None:
            return self._delivery_action(shipper, delivery_order)

        return ("S", 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs["t"])
        T = int(obs["T"])

        actions: Dict[int, Action] = {}
        reserved_order_ids: Set[int] = set()
        reserved_pickup_cells: Set[Position] = set()

        for shipper in sorted(shippers, key=lambda s: s.id):
            actions[shipper.id] = self._decide_shipper_action(
                shipper,
                orders,
                reserved_order_ids,
                reserved_pickup_cells,
                t,
                T,
            )

        return self._resolve_conflicts(actions, shippers, orders)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
