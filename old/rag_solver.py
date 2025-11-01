#!.venv/bin/python3

import sys
import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict
from collections import namedtuple

from instance import Instance, Op


class Solver:
	def __init__(self, inst):
		self.inst : Instance = inst 


	def make_random_paths(self):
		self.paths : List[List[Tuple[Op, Op]]] = []

		self.next_op: Dict[Op, Op] = {}
		self.prev_op: Dict[Op, Op] = {}

		for t in range(self.inst.n_trains):
			path : List[Op] = []

			while True:
				if not path:
					first = self.inst.ops[t][0]

				else:
					first = path[-1][1]
					if first.n_succ == 0:
						break

					else:
						second = random.choice(first.succ)
						path.append(second)
			
			self.paths.append(path)

		return self.paths


	def get_order(self):
		self.make_earliest_start()

		self.order = list(itertools.chain(*self.paths))
		self.order.sort(key=lambda x: self.earliest_start[x])

		return self.order


	def make_earliest_start(self):
		self.earliest_start = {}

		for path in self.paths:
			self.earliest_start[path[0]] = path[0].start_lb

			for first, second in self.get_transitions(path):
				self.earliest_start[second] = max(
					self.earliest_start[first] + first.dur,
					second.start_lb)
				

	def prio_func(self, x):
		op1, op2 = x[0], x[1]
		if op1.n_prev == 0:
			return -1
		
		if op2 is None:
			return self.inst.max_dur
		
		return self.earliest_start[op1] + self.earliest_start[op2]

	def order_res_uses(self, res_idx):
		self.res_uses[res_idx].sort(key=self.prio_func)
		

	def add_res_uses(self, path: List[Tuple[Op, Op]]):
		for first, second in path:
			for res in op.res:
				if not res.idx in self.res_uses:
					self.res_uses[res.idx] = [(op, self.next_op[op])]
					continue

				ru = self.res_uses[res.idx]
				if ru[-1][1] == op:
					ru[-1] = (ru[-1][0], self.next_op[op])
				else:
					ru.append((op, self.next_op[op]))



	@staticmethod
	def get_transitions(ops: List):
		return zip(ops[:-1], ops[1:])

if __name__ == '__main__':
	DEFAULT_DATA = 'data/phase1/line3_5.json'

	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
	sol = Solver(inst)
	sol.make_random_paths()
	sol.make_earliest_start()
	sol.make_graph()

	for k, v in sol.res_uses.items():
		def path_dist(x):
			tr = x[0].train_idx
			p = sol.paths[tr]
			if x[1] is None:
				return len(p) - p.index(x[0])
			
			return p.index(x[1]) - p.index(x[0])
		 
		print(f'res: {k}', [f'{x[0].train_idx}:{path_dist(x)}' for x in v])

	# n_scc = len(sol.get_scc())

	# for c in nx.simple_cycles(sol.g):
	# 	print(c)
	
	# print('---')
	
	for tr in range(inst.n_trains):
		sol.push_train(tr)

	# for c in nx.simple_cycles(sol.g):
	# 	print(c)

	# inst.verify_order(sol.get_order_from_graph())
	


