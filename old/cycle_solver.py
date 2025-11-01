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
		self.g = nx.Graph()


	def make_random_paths(self):
		self.paths : List[List[Op]] = []

		self.next_op: Dict[Op, Op] = {}
		self.prev_op: Dict[Op, Op] = {}

		for t in range(self.inst.n_trains):
			path : List[Op] = []

			while True:
				if not path:
					path.append(self.inst.ops[t][0])
					self.prev_op[self.inst.ops[t][0]] = None

				else:
					last_op = path[-1]
					if last_op.n_succ == 0:
						self.next_op[last_op] = None
						break

					else:
						next_op = random.choice(last_op.succ)
						path.append(next_op)
						self.next_op[last_op] = next_op
						self.prev_op[next_op] = last_op
			
			self.paths.append(path)

		return self.paths

	def add_paths_edges(self):
		for path in self.paths:
			self.g.add_edges_from(self.get_transitions(path))

	def make_res_uses(self):
		self.res_uses = []

		for path in self.paths:
			res_locks = {}

			for op in path:
				unlocks = self.inst.get_res_diff(self.prev_op[op], op)
				locks = self.inst.get_res_diff(op, self.prev_op[op])

				for r in unlocks:
					assert(r in res_locks)
					self.res_uses.append((r, res_locks[r], op))
					del res_locks[r]

				for r in locks:
					res_locks[r] = op

			assert(len(res_locks) == 0)

		return self.res_uses

	def sort_res_uses(self):
		self.make_earliest_start()
		self.res_uses.sort(key=self.prio_func)

		return self.res_uses

	def make_earliest_start(self):
		self.earliest_start = {}

		for path in self.paths:
			self.earliest_start[path[0]] = path[0].start_lb

			for first, second in self.get_transitions(path):
				self.earliest_start[second] = max(
					self.earliest_start[first] + first.dur,
					second.start_lb)
		
		return self.earliest_start

	def prio_func(self, x):
		op1, op2 = x[1], x[2]
		if op1.n_prev == 0:
			return -1
		
		if op2.n_succ == 0:
			return self.inst.total_dur
		
		return self.earliest_start[op1] + self.earliest_start[op2]				


	def get_order_from_graph(self):
		return list(nx.lexicographical_topological_sort(self.g, lambda x: self.earliest_start[x]))


	def add_res_uses_to_graph(self):
		prev_use = {}

		change = True
		while change:
			change = False
			for i, ru in enumerate(self.res_uses):
				res_idx, op1, op2 = ru
				if res_idx in prev_use:
					self.g.add_edge(prev_use[res_idx], op1)

					if self.has_cycle():
						self.g.remove_edge(prev_use[res_idx], op1)
						continue
				
					else:
						print(f'adding {prev_use[res_idx]} -> {op1}, res: {res_idx}')

				prev_use[res_idx] = op2
				
				ret_ru = self.res_uses.pop(i)
				assert(ru == ret_ru)

				change = True

				break
		
		print(self.res_uses)

	
	def has_cycle(self, op=None):
		try:
			c = nx.find_cycle(self.g, op)
		except nx.exception.NetworkXNoCycle:
			return False
		else:
			return True


	@staticmethod
	def get_transitions(ops: List):
		return zip(ops[:-1], ops[1:])

if __name__ == '__main__':
	DEFAULT_DATA = 'data/phase1/line1_critical_0.json'

	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
	sol = Solver(inst)
	sol.make_random_paths()
	sol.add_paths_edges()
	sol.make_res_uses()
	sol.sort_res_uses()

	sol.add_res_uses_to_graph()

	# sol.make_graph()

	# for k, v in sol.res_uses.items():
	# 	def path_dist(x):
	# 		tr = x[0].train_idx
	# 		p = sol.paths[tr]
	# 		if x[1] is None:
	# 			return len(p) - p.index(x[0])
			
	# 		return p.index(x[1]) - p.index(x[0])
		 
	# 	print(f'res: {k}', [f'{x[0].train_idx}:{path_dist(x)}' for x in v])

	# n_scc = len(sol.get_scc())

	# for c in nx.simple_cycles(sol.g):
	# 	print(c)
	
	# print('---')
	
	# for tr in range(inst.n_trains):
	# 	sol.push_train(tr)

	# for c in nx.simple_cycles(sol.g):
	# 	print(c)

	if sol.has_cycle():
		inst.verify_order(sol.get_order_from_graph())
	else:
		print('cycles found!')


