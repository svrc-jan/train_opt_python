#!.venv/bin/python3

import sys

import itertools as it
import numpy as np
import gurobipy as gp

from typing import List, Tuple, Dict, Set
from instance import Instance, Op, Res_use
from dataclasses import dataclass, field
from gurobipy import GRB

from utils import round_dict


DEFAULT_DATA = 'data/testing/headway1.json'
# DEFAUL_DATA = 'data/phase1/line1_critical_0.json'

class Model:
	gm: gp.Model
	inst: Instance

	def __init__(self, inst):
		self.inst = inst
		self.create_model()

	def create_op_vars(self):
		m = self.gm

		keys = []
		for op in self.inst.all_ops:
			keys.extend((op, succ) for succ in op.succ)

		time_lbs = [succ.start_lb for op, succ in keys]
		time_ubs = list(map(
			lambda x: float('inf') if (x.start_ub is None) else x.start_ub, 
			(succ for op, succ in keys)))
		
		self.var_time = m.addVars(keys, name='time', vtype='C',
			lb=time_lbs, ub=time_ubs)

		# self.var_used = m.addVars(self.inst.all_ops, name='used', vtype='C')
		self.var_flow = m.addVars(keys, name='flow', vtype='B')

	def create_used_conss(self):
		m = self.gm
		u = self.var_used
		f = self.var_flow

		self.cons_used = {}
		for op in self.inst.all_ops:
			if op.n_prev == 1 or op.n_succ == 1:
				cons = u[op] == 1
			else:
				cons = u[op] == f.sum(op, '*')

			self.cons_used[op] = m.addConstr(cons, name=f'used{op}')
	
	def create_dur_conss(self):
		m = self.gm
		t = self.var_time
		f = self.var_flow
		# u = self.var_used

		self.cons_dur = {}
		for op in self.inst.all_ops:
			M = op.dur

			for prev in op.prev:
				for succ in op.succ:
					cons = t[prev, op] + op.dur <= t[op, succ] + M*(2 - f[prev, op] - f[op, succ])
					self.cons_dur[prev, op, succ] = m.addConstr(cons, name=f'dur{prev},{op},{succ}')

	def create_flow_conss(self):
		m = self.gm
		f = self.var_flow
		
		self.cons_flow = {}
		for op in self.inst.all_ops:
			if op.n_prev == 0:
				cons = f.sum(op, '*') == 1
			elif op.n_succ == 0:
				cons = f.sum('*', op) == 1
			else:
				cons = f.sum('*', op) == f.sum(op, '*')

			self.cons_flow[op] = m.addConstr(cons, name=f'flow{op}')


	def create_obj(self):
		m = self.gm
		t = self.var_time

		obj_ops = [op for op in self.inst.all_ops if op.obj is not None]
		obj_vtypes = ['C' if op.obj.coeff > 0 else 'B' for op in obj_ops]
		obj_coeffs = [op.obj.coeff if op.obj.coeff > 0 else op.obj.increment for op in obj_ops]


		self.var_obj = m.addVars(obj_ops, name='obj', vtype=obj_vtypes, obj=obj_coeffs)

		M = self.inst.max_dur
		
		self.cons_obj = {}
		for op in obj_ops:
			for prev in op.prev:
				if op.obj.coeff > 0:
					cons = t[prev, op] - op.obj.threshold <= self.var_obj[op]
				else:
					cons = t[prev, op] - op.obj.threshold <= M*self.var_obj[op]
				
				self.cons_obj[prev, op] = m.addConstr(cons, name=f'obj{prev}{op}')
	
	def create_resource_vars(self):
		pass

	def create_model(self):
		self.gm = gp.Model('train opt')
		
		self.create_op_vars()
		self.create_dur_conss()
		self.create_flow_conss()
		self.create_obj()
		# self.create_diff_vars()
		# self.create_diff_conss()

	def optimize(self):
		self.gm.optimize()


	def get_collisions(self):
		m = self.gm

		t = round_dict(m.getAttr('x', self.var_time))
		f = round_dict(m.getAttr('x', self.var_flow))

		collisions = []

		used = {}
		for op in self.inst.all_ops:
			used[op] = sum(f[op, succ] for succ in op.succ) if op.n_succ > 0 else 1

		for res_idx, res_uses in self.inst.res_uses.items():
			for i, ru1 in enumerate(res_uses):
				op1 = ru1.op
				if not used[op1]:
					continue

				str1 = [(prev, op1) for prev in op1.prev if f[prev, op1] == 1][0]
				etr1 = [(op1, succ) for succ in op1.succ if f[op1, succ] == 1][0]

				for ru2 in res_uses[i+1:]:
					op2 = ru2.op

					if op1.train_idx == op2.train_idx:
						continue
				
					if not used[op2]:
						continue

					str2 = [(prev, op1) for prev in op1.prev if f[prev, op1] == 1][0]
					etr1 = [(op1, succ) for succ in op1.succ if f[op1, succ] == 1][0]

					if (s1 < e2) and (s2 < e1):
						col = (res_idx, ru1, ru2) if op1.idx < op2.idx else (res_idx, ru2, ru1)
						collisions.append(col)

		return collisions


	def create_res_conss(self, collisions: List[Tuple[int, Res_use, Res_use]]):
		if not hasattr(self, 'var_res'):
			self.var_res = {}

		if not hasattr(self, 'cons_res'):
			self.cons_res = {}

		if not hasattr(self, 'cons_res_sum'):
			self.cons_res_sum = {}

		m = self.gm
		s = self.var_start
		e = self.var_end
		f = self.var_flow
		r = self.var_res

		for res_idx, ru1, ru2 in collisions:
			op1 = ru1.op
			op2 = ru2.op

			assert(((op1, op2) in r) == ((op2, op1) in r))

			if not (op1, op2) in r:
				r[op1, op2] = m.addVar(name=f'res{op1},{op2}', vtype='B')
				r[op2, op1] = m.addVar(name=f'res{op2},{op1}', vtype='B')


				use1 = f.sum(op1, '*') if op1.n_succ > 0 else 1
				use2 = f.sum(op2, '*') if op2.n_succ > 0 else 1

				cons = 2*(r[op1, op2] + r[op2, op1]) == use1 + use2
				self.cons_res_sum[op1, op2] = m.addConstr(cons, f'res_sum{res_idx},{op1},{op2}')

			
			cons = (r[op1, op2] == 1) >> (e[op1] + ru1.time <= s[op2])
			self.cons_res[res_idx, op1, op2] = m.addConstr(cons, name=f'res{res_idx},{op1},{op2}')
			
			cons = (r[op2, op1] == 1) >> (e[op2] + ru2.time <= s[op1])

			self.cons_res[res_idx, op2, op1] = m.addConstr(cons, name=f'res{res_idx},{op2},{op1}')
			

		m.update()

class Solver:
	inst: Instance
	def __init__(self, inst):
		self.inst = inst


	def solve(self):
		
		m = Model(inst)

		m.optimize()
		# print(m.get_collisions())
		# m.gm.Params.SolutionLimit = 1
		# m.gm.Params.ImproveStartGap = 1.0

		# while True:
		# 	m.optimize()
			
		# 	if m.gm.Status == GRB.INFEASIBLE:
		# 		m.gm.write('model.mps')
		# 		break
			
		# 	# print(m.gm.getAttr('x', m.var_start))
		# 	collisions = m.get_collisions()
		# 	if len(collisions) == 0:
		# 		break
			
		# 	print(collisions)
		# 	m.create_res_conss(collisions)		
		

if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
	sol = Solver(inst)
	sol.solve()
