#!.venv/bin/python3

import sys
import random

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple


from instance import Instance, Train, Op, Op_idx


DEFAULT_DATA = 'data/nor1_critical_0.json'


class Graph:
	inst: Instance

	nodes: List[Op_idx]

	prev: Dict[Op_idx, List[Op_idx]]
	succ: Dict[Op_idx, List[Op_idx]]
	dur: Dict[Tuple[Op_idx, Op_idx], int]

	res_uses: Dict[int, List[Tuple[Op_idx, Op_idx, int]]]

	def __init__(self, inst):
		self.inst = inst

		self.nodes = []

		self.prev = defaultdict(list)
		self.succ = defaultdict(list)
		self.dur = {}
		
		self.res_uses = defaultdict(list)


	def add_edge(self, i, j, d=0):
		self.succ[i].append(j)
		self.prev[j].append(i)
		self.dur[i, j] = d


	def remove_edge(self, i, j):
		self.succ[i].remove(j)
		self.prev[j].remove(i)
		del self.dur[i, j]


	def add_path(self, path):
		
		for i, j in zip(path, path[1:] + [None]):
			self.nodes.append(i)
			op = self.inst.op(i)
			
			if j:
				self.add_edge(i, j, op.dur)

			for r in op.res:
				if r in self.res_uses:
					last = self.res_uses[r][-1]
					if last[1] == i:
						self.res_uses[r].pop()
						self.res_uses[r].append((last[0], j, self.inst.res_time(r, i, 1)))

						continue

				self.res_uses[r].append((i, j, self.inst.res_time(r, i, 1)))

		
	def make_order(self):
		in_order = { o: len(self.prev.get(o, [])) for o in self.nodes }
		start = { o: self.inst.op(o).start_lb for o in self.nodes }

		order = []
		q = deque(o for o in self.nodes if in_order[o] == 0)

		while q:
			o = q.popleft()
			order.append(o)

			for s in self.succ[o]:
				start[s] = max(start[s], start[o] + self.dur[o, s])
				in_order[s] -= 1

				if in_order[s] == 0:
					q.append(s)

		return order, start


	def find_branch(self, start: Dict[Op_idx, int]):
		overlap = 0
		col = None

		for r, res_uses in self.res_uses.items():
			for i, ru1 in enumerate(res_uses):
				for ru2 in res_uses[i+1:]:
					s1 = start[ru1[0]]
					e1 = start[ru1[1]] if ru1[1] else s1 + self.inst.op(ru1[0]).dur

					s2 = start[ru2[0]]
					e2 = start[ru2[1]] if ru2[1] else s2 + self.inst.op(ru2[0]).dur

					ol = min(e1, e2) - max(s1, s2)

					if ol > overlap:
						col = (ru1, ru2)
		
		return col
	
	
	def resolve_col(self, depth=0):
		order, start = g.make_order()

		if len(order) < len(self.nodes):
			print(f'{'  '*depth}cycle')
			return False

		col = self.find_branch(start)
		if not col:
			return True
		
		(s1, e1, t1), (s2, e2, t2) = col

		if start[s1] > start[s2]:
			(s2, e2, t2), (s1, e1, t1) = col
		
		print(f'{'  '*depth}{e1} -> {s2}')
		self.add_edge(e1, s2, t1)
		if self.resolve_col(depth+1):
			return True
		self.remove_edge(e1, s2)

		print(f'{'  '*depth}{e2} -> {s1}')
		self.add_edge(e2, s1, t2)
		if self.resolve_col(depth+1):
			return True
		self.remove_edge(e2, s1)

		return False

class Heur:
	inst: Instance
	graphs: List[Graph]


	def __init__(self, inst):
		self.inst = inst
		pass


	def make_graphs(self):
		g = Graph(self.inst)

		for train in self.inst.trains:
			g.add_path(self.make_random_path(train))

		return g


	def make_random_path(self, train: Train):
		path = [train.ops[0].idx]

		while True:
			op = self.inst.op(path[-1])

			if op.n_succ == 0:
				break

			path.append(Op_idx(train.idx, random.choice(op.succ)))

		return path


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	heur = Heur(inst)

	g = heur.make_graphs()
	# print(g.res_uses)

	g.resolve_col()