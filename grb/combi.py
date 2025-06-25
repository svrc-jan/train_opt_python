#!.venv/bin/python3

import sys
import random

from typing import List, Tuple, Dict

import numpy as np
import gurobipy as gp

from gurobipy import GRB

from instance import Instance, Op, Res_use
from utils import round_dict


DEFAULT_DATA = 'data/testing/headway1.json'


class Solver:
	inst: Instance
	
	def __init__(self, inst):
		self.inst = inst

		self.make_res_vectors()

	
	def generate_random_path(self, train):
		path = [self.inst.ops[train][0]]

		while True:
			op = path[-1]
			if op.n_succ == 0:
				break

			succ = random.choice(op.succ)
			path.append(succ)

		return path

	
	def make_res_vectors(self):
		self.res_vec = {}

		for op in self.inst.all_ops:
			self.res_vec[op] = np.zeros((self.inst.n_res, ), dtype=int)
			for res in op.res:
				self.res_vec[op][res.idx] = 1

	
	def make_random_sol(self):

		last_train_op = [self.inst.ops[train][0] for train in range(self.inst.n_trains)]

		min_start = {}
	
		def start_func(op: Op):
			if op in min_start:
				return
			
			s = op.start_lb
			for prev in op.prev:
				if not prev in min_start:
					return

				s = max(min_start[prev] + op.dur, s)

			min_start[op] = s
			
			for succ in op.succ:
				start_func(succ)

		for op in last_train_op:
			start_func(op)

		res_state = np.ones((self.inst.n_res, ), dtype=int)
		for op in last_train_op:
			res_state -= self.res_vec[op]
			
		sol = []
		def backtrack(sol, last_train_op, res_state):
			poss_ops = []
			
			finished = True
			for op in last_train_op:
				if op.n_succ == 0:
					continue

				finished = False
				for succ in op.succ:
					if np.all(res_state >= self.res_vec[succ]):
						poss_ops.append(succ)

			if finished:
				return True

			poss_ops.sort(key=lambda x: min_start[x])
			print(len(sol), len(poss_ops))

			for op in poss_ops:
				train = op.train_idx
				prev_op = last_train_op[train]

				res_diff = self.res_vec[prev_op] - self.res_vec[op]
				
				sol.append(op)
				last_train_op[train] = op
				res_state += res_diff

				if backtrack(sol, last_train_op, res_state):
					return True

				sol.pop()
				last_train_op[train] = prev_op
				res_state -= res_diff

			return False
		
		backtrack(sol, last_train_op, res_state)

		return sol



				








if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)

	sys.setrecursionlimit(inst.n_ops)

	sol = Solver(inst)
	sol.make_random_sol()

		