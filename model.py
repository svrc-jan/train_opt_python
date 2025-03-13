#!.venv/bin/python3

import sys

import pyscipopt as scip
import itertools as it
import numpy as np

from typing import List, Tuple, Dict, Set
from instance import Instance, Op, Res_use
from dataclasses import dataclass, field

class Model:
	def __init__(self):
		pass
	
	@staticmethod
	def set_obj(m):
		u = m.data['u']
		c = m.data['obj_coef']

		m.setObjective(scip.quicksum(c[k]*u[k] for k in u.keys()), sense='minimize')
		
	@staticmethod
	def get_values(m, var, to_int=False):
		if isinstance(var, (list, tuple)):
			return tuple(Model.get_values(m, v, to_int) for v in var)
		

		var = m.data[var]

		if isinstance(var, dict):
			if to_int:
				rv =  { k: int(round(m.getVal(v))) for k, v in var.items() }
			else:
				rv =  { k: m.getVal(v) for k, v in var.items() }
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
			vtype = 'B' if op.n_succ > 1 else 'C'
			f[op, succ] = m.addVar(name=f'f{op},{succ}', vtype=vtype, lb=0, ub=1)

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