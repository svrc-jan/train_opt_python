#!.venv/bin/python3

import sys

import pyscipopt as scip
import itertools as it
import numpy as np

from typing import List, Tuple, Dict, Set
from instance import Instance, Op, Res_use


class Model(scip.Model):
	def __init__(self, inst):
		super().__init__()
		self.make_init_model(self, inst)

	@staticmethod
	def get_values(m, var, to_int=False):
		if isinstance(var, (list, tuple)):
			return tuple(Model.get_values(m, v, to_int) for v in var)
		
		best_sol = m.getBestSol()

		var = m.data[var]

		if isinstance(var, dict):
			if to_int:
				rv =  { k: int(round(m.getSolVal(best_sol, v))) for k, v in var.items() }
			else:
				rv =  { k: m.getSolVal(best_sol, v) for k, v in var.items() }
		else:
			if to_int:
				rv = int(round(m.getVal(var)))
			else:
				rv = m.getVal(var)

		return rv 


	@staticmethod
	def make_op_vars(m, op: Op):
		
		for var in ['s', 'e', 'f']:
			if not var in m.data:
				m.data[var] = {}

		s = m.data['s']
		e = m.data['e']
		# o = m.data['o']
		f = m.data['f']

		s[op] = m.addVar(name=f's{op}', vtype='C', lb=op.start_lb, ub=op.start_ub)
		e[op] = m.addVar(name=f'e{op}', vtype='C', lb=0, ub=None)
		# o[op] = m.addVar(name=f'e{op}', vtype='C', lb=0, ub=None)

		for succ in op.succ:
			vtype = 'C' if op.n_succ == 1 else 'B'
			# vtype = 'C' if (op.n_succ == 1 or is_heur) else 'B'
			f[op, succ] = m.addVar(name=f'f{op},{succ}', vtype=vtype, lb=0, ub=1)
		
		# if op.n_succ > 1:
		# 	m.addConsSOS1(name=f'sos1{op}', vars=[f[op, succ] for succ in op.succ],
		# 		initial=False, enforce=False, check=False)

	@staticmethod
	def make_dur_cons(m, op: Op):
		s = m.data['s']
		e = m.data['e']
		f = m.data['f']

		# M = self.inst.max_train_dur[op.train_idx]
		# M = self.inst.max_dur*100

		M = op.dur

		if op.n_succ == 0:
			return

		use = scip.quicksum(f[op, succ] for succ in op.succ)

		cons = s[op] + op.dur <= e[op] + M*(1 - use)
		m.addCons(name=f'dur{op}', cons=cons)
		

	@staticmethod
	def make_end_cons(m, op: Op):
		s = m.data['s']
		e = m.data['e']

		for succ in op.succ:
			cons = e[op] == s[succ]
			m.addCons(name=f'end{op},{succ}', cons=cons)


	@staticmethod
	def make_flow_cons(m, op):
		f = m.data['f']
		if op.n_prev == 0:
			cons = scip.quicksum(f[op, succ] for succ in op.succ) == 1
		elif op.n_succ == 0:
			cons = scip.quicksum(f[prev, op] for prev in op.prev) == 1
		else:
			cons = scip.quicksum(f[op, succ] for succ in op.succ) == \
				   scip.quicksum(f[prev, op] for prev in op.prev)

		m.addCons(name=f'flow{op}', cons=cons)


	@staticmethod
	def make_obj_vars(m, ops: List[Op]):
		s = m.data['s']

		u = {} # over threshold
		v = {} # under threshold
		c = {}

		for op in ops:
			if op.obj:
				u[op] = m.addVar(name=f'u{op}', vtype='C', lb=0, ub=None)
				v[op] = m.addVar(name=f'v{op}', vtype='C', lb=0, ub=None)

				c[op] = op.obj.coeff

				cons = s[op] - op.obj.threshold == u[op] - v[op]
				m.addCons(name=f'obj{op}', cons=cons)

		m.data['u'] = u
		m.data['v'] = v
		m.data['obj_coef'] = c


	@staticmethod
	def set_obj(m):
		u = m.data['u']
		c = m.data['obj_coef']

		m.setObjective(scip.quicksum(c[k]*u[k] for k in u.keys()), sense='minimize')


	@staticmethod
	def make_init_model(m, inst: Instance):
		m.data = {}

		for op in inst.all_ops:
			Model.make_op_vars(m, op)

		for op in inst.all_ops:
			Model.make_dur_cons(m, op)

		for op in inst.all_ops:
			Model.make_end_cons(m, op)
	
		for op in inst.all_ops:
			Model.make_flow_cons(m, op)

		Model.make_obj_vars(m, inst.all_ops)

		Model.set_obj(m)
		
		return m