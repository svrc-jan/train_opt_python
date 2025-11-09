#!.venv/bin/python3

import sys

from collections import defaultdict
from typing import List, Dict, Tuple, NamedTuple

from solver import Event, Solution, Solver
from instance import Instance, Op, Op_idx


DEFAULT_DATA = 'data/nor1_critical_0.json'


class Res_col(NamedTuple):
	res: int
	op1: Op_idx
	op2: Op_idx


class Heuristic:
	inst: Instance
	groups: List[Solution]
	train_to_group: Dict[int, int]
	collisions: List[Res_col]
	group_col_count: Dict[Tuple[int, int], int]

	def __init__(self, inst: Instance):
		self.inst = inst

	
	def solve(self):
		self.groups = []

		for t in range(self.inst.n_trains):
			solver = Solver(self.inst, None, free=[t])
			solver.add_time_obj()
			solver.solve()
			self.groups.append(solver.get_solution())


		self.make_collisions()
		self.count_collisions()

		while self.group_col_count:
			g1, g2 = max(self.group_col_count.keys(), key=lambda x: self.group_col_count[x])

			print(self.groups[g1].events.keys(), self.groups[g1].events.keys())

			self.group_col_count = { 
				(x1, x2): v 
				for (x1, x2), v 
				in self.group_col_count.items()
				if x1 != g1 and x2 != g1 and x1 != g2 and x2 != g2
			}


	def make_collisions(self):
		events: List[Event] = []
		self.train_to_group = {}

		for g, sol in enumerate(self.groups):
			for t, evs in sol.events.items():
				self.train_to_group[t] = g
				events.extend(evs)

		events.sort(key=lambda x: (x.start, x.end))

		self.collisions = []

		prev_op: Dict[int, Op] = {}
		res_locks = defaultdict(list)
		
		for ev in events:
			t, o = ev.idx

			train = self.inst.trains[t]
			op = train.ops[o]

			prev = prev_op.get(t, None)
			
			if prev:
				for r in prev.res:
					res_locks[r].remove(prev.idx)

			for r in op.res:
				self.collisions.extend(Res_col(r, other, op.idx) for other in res_locks[r])
				res_locks[r].append(op.idx)
			
			prev_op[t] = op


	def count_collisions(self):
		self.group_col_count = defaultdict(lambda: 0)

		for col in self.collisions:
			g1 = self.train_to_group[col.op1.train] 
			g2 = self.train_to_group[col.op2.train]

			k = (g1, g2) if (g1 < g2) else (g2, g1)

			self.group_col_count[k] += 1



if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	heur = Heuristic(inst)

	heur.solve()
