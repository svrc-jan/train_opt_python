#!.venv/bin/python3

import sys
import random

from collections import deque
from typing import List

from instance import Instance, Op


class Solution:
	def __init__(self, inst):
		self.inst : Instance = inst
		self.ops = []
		self.obj_ops = []
		self.pred = {}
		self.succ = {}

	def make_random_train_path(self, train):
		prev_op = self.inst.ops[train][0]
		self.ops.append(prev_op)

		if prev_op.obj is not None:
			self.obj_ops.append(prev_op)

		while prev_op.n_succ > 0:
			next_op = random.choice(prev_op.succ)

			self.ops.append(next_op)
			if next_op.obj is not None:
				self.obj_ops.append(next_op)

			self.pred.setdefault(next_op, []).append((prev_op, prev_op.dur))
			self.succ.setdefault(prev_op, []).append((next_op, prev_op.dur))

			prev_op = next_op

	def make_random_paths(self):
		for t in range(self.inst.n_trains):
			self.make_random_train_path(t)

	
	def make_order(self):
		indegree = { op : 0 for op in self.ops}
		self.start = { op : op.start_lb for op in self.ops }
		
		self.order = []
		q = deque()

		for op in self.ops:
			if len(self.pred.setdefault(op, [])) == 0:
				q.append(op)
				continue

			for _ in self.pred[op]:
				indegree[op] += 1

		while q:
			op = q.popleft()
			self.order.append(op)

			for s, dur in self.succ.setdefault(op, []):
				indegree[s] -= 1
				self.start[s] = max(self.start[s], self.start[op] + dur)
				if indegree[s] == 0:
					q.append(s)

		return len(self.order) == len(self.ops)

	def calc_obj(self):
		obj = 0
		slack = 0
		for op in self.obj_ops:
			op_delay = min(self.start[op] - op.obj.threshold, 0)
			op_slack = min(op.obj.threshold - self.start[op], 0)

			obj += op_delay*op.obj.coeff + (op_delay > 0)*op.obj.increment
			slack += op_slack
	
		return obj

	def find_res_col(self):
		self.col = []
		res_lock = {}

		for op in self.order:
			if op.n_prev > 0:
				prev, _ = self.pred[op][0]

				for r in prev.res:
					res_lock[r.idx].remove((prev, r.time))

			for r in op.res:
				for col_op, col_op_time in res_lock.setdefault(r.idx, []):
					self.col.append((col_op, op, col_op_time, col_op_time, r.idx))

				res_lock[r.idx].append((op, r.time))
		
		return self.col
	
	def resolve_col(self):
		best_obj = float('inf')
		best_col = None

		for col in self.col:
			op1, op2, t1, t2, r = col
			succ1, _ = self.succ[op1][0]
			succ2, _ = self.succ[op2][0]

			# succ2 -> op1
			self.pred[op1].append((succ2, t2))
			self.succ[succ2].append((op1, t2))

			if self.make_order():
				obj = self.calc_obj()
				if obj < best_obj:
					best_obj = obj
					best_col = (succ2, op1, t2)


			self.pred[op1].pop()
			self.succ[succ2].pop()

	
			# succ1 -> op2
			self.pred[op2].append((succ1, t1))
			self.succ[succ1].append((op2, t1))

			if self.make_order():
				obj = self.calc_obj()
				if obj < best_obj:
					best_obj = obj
					best_col = (succ1, op2, t1)

			self.pred[op2].pop()
			self.succ[succ1].pop()
		
		assert(best_col is not None)
		
		op1, op2, t = best_col
		self.pred[op2].append((op1, t))
		self.succ[op1].append((op2, t))
			


if __name__ == '__main__':	
	DEFAULT_DATA = 'data/phase1/line3_2.json'

	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
	sol = Solution(inst)
	sol.make_random_paths()
	sol.make_order()

	while sol.find_res_col():
		print(sol.col)
		sol.resolve_col()

	print(sol.order, sol.calc_obj())