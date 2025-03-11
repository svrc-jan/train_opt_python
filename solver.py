#!.venv/bin/python3

import sys

import pyscipopt as scip
import itertools as it
import numpy as np

from typing import List, Tuple, Dict, Set
from instance import Instance, Op, Res_use
from dataclasses import dataclass, field


# DEFAUL_DATA = 'data/testing/headway1.json'
DEFAUL_DATA = 'data/phase1/line1_critical_0.json'


class Solver:
	inst: Instance

	def __init__(self, inst):
		self.inst = inst 
	
	@staticmethod
	def get_values(m, var, to_int=True):
		if isinstance(var, (list, tuple)):
			return tuple(Solver.get_values(m, v, to_int) for v in var)
		
		if to_int:
			rv =  { k: int(round(m.getVal(v))) for k, v in var.items() }
		else:
			rv =  { k: m.getVal(v) for k, v in var.items() }

		return rv 

	def make_op_vars(self, m, s, e, o, f, op: Op):
		s[op] = m.addVar(name=f's{op}', vtype='C', lb=op.start_lb, ub=op.start_ub)
		e[op] = m.addVar(name=f'e{op}', vtype='C', lb=0, ub=None)
		o[op] = m.addVar(name=f'e{op}', vtype='C', lb=0, ub=None)

		for succ in op.succ:
			vtype = 'B' if op.n_succ > 1 else 'C'
			f[op, succ] = m.addVar(name=f'f{op},{succ}', vtype=vtype, lb=0, ub=1)

	def make_dur_cons(self, m, s, e, op: Op):
		cons = s[op] + op.dur <= e[op]
		m.addCons(name=f'dur{op}', cons=cons)

	def make_end_cons(self, m, s, e, o, f, op: Op):
		M = op.dur

		if op.n_succ == 1:
			succ = op.succ[0]

			cons = e[op] == s[succ]
			m.addCons(name=f'end{op},{succ}', cons=cons)

		elif op.n_succ > 1:
			for succ in op.succ:
				cons1 = e[op] <= s[succ]
				cons2 = e[op] >= s[succ]

				m.addConsIndicator(name=f'end1{op},{succ}', cons=cons1, binvar=f[op, succ])
				m.addConsIndicator(name=f'end2{op},{succ}', cons=cons2, binvar=f[op, succ], initial=False)


				m.addCons(name=f'ord{op},{succ}', cons=o[op] + 1 <= o[succ])

	def make_flow_cons(self, m, f, op):
		if op.n_prev == 0:
			cons = scip.quicksum(f[op, succ] for succ in op.succ) == 1
		elif op.n_succ == 0:
			cons = scip.quicksum(f[prev, op] for prev in op.prev) == 1
		else:
			cons = scip.quicksum(f[op, succ] for succ in op.succ) == \
				   scip.quicksum(f[prev, op] for prev in op.prev)

		m.addCons(name=f'flow{op}', cons=cons)

	
	def make_res_cons(self, m, s, e, o, f, r):
		for res, uses in self.inst.res_uses.items():
			for i, u1 in enumerate(uses):
				for u2 in uses[i+1:]:
					op1 = u1.op
					op2 = u2.op

					if op1.train_idx == op2.train_idx:
						continue

					M1 = inst.total_dur/2
					M2 = inst.total_dur/2
					
					v1 = m.addVar(name=f'r{res},{op1},{op2}', vtype='B', lb=0, ub=1)
					v2 = m.addVar(name=f'r{res},{op2},{op1}', vtype='B', lb=0, ub=1)

					r[res, op1, op2] = v1
					r[res, op1, op2] = v2

					cons1 = e[op1] + u1.time <= s[op2]
					cons2 = e[op2] + u2.time <= s[op1]

					m.addConsIndicator(name=f'res{res},{op1},{op2}', cons=cons1, binvar=v1, initial=False)
					m.addConsIndicator(name=f'res{res},{op2},{op1}', cons=cons2, binvar=v2, initial=False)

					if u1.time == 0:
						cons1 = o[op1] + 1 <= o[op2]
						m.addConsIndicator(name=f'res_ord{res},{op1},{op2}', cons=cons1, binvar=v1, initial=False)

					if u2.time == 0:
						cons2 = o[op2] + 1 <= o[op1]
						m.addConsIndicator(name=f'res_ord{res},{op2},{op1}', cons=cons2, binvar=v2, initial=False)

					use_op1 = scip.quicksum(f[prev1, op1] for prev1 in op1.prev) if op1.n_prev > 0 else 1
					use_op2 = scip.quicksum(f[prev2, op2] for prev2 in op2.prev) if op2.n_prev > 0 else 1

					cons = v1 + v2 == (use_op1 + use_op2)/2
					m.addCons(name=f'ch{res},{op1},{op2}', cons=cons)

				
		


	def make_obj_vars(self, m, s, ops: List[Op]):
		u = {} # over threshold
		v = {} # under threshold
		c = {}

		v_min = m.addVar(name='v_min', vtype='C', lb=0, ub=None)

		for op in ops:
			if op.obj:
				u[op] = m.addVar(name=f'u{op}', vtype='C', lb=0, ub=None)
				v[op] = m.addVar(name=f'v{op}', vtype='C', lb=0, ub=None)

				c[op] = op.obj.coeff

				cons = s[op] - op.obj.threshold == u[op] - v[op]
				m.addCons(name=f'obj{op}', cons=cons)

				cons = v[op] >= v_min
				m.addCons(name=f'v_min{op}', cons=cons)

		z = m.addVar(name='z', vtype='C', lb=0, ub=None)
		cons = scip.quicksum(c[k]*u[k] for k in u.keys()) == z
		m.addCons(name='obj', cons=cons)

		return z, v_min


	def make_init_model(self):
		m = scip.Model()

		s = {}
		e = {}
		o = {}
		f = {}
		r = {}

		for op in it.chain(*self.inst.ops):
			self.make_op_vars(m, s, e, o, f, op)

		for op in it.chain(*self.inst.ops):
			self.make_dur_cons(m, s, e, op)

		for op in it.chain(*self.inst.ops):
			self.make_end_cons(m, s, e, o, f, op)
	
		for op in it.chain(*self.inst.ops):
			self.make_flow_cons(m, f, op)

		self.make_res_cons(m, s, e, o, f, r)

		z, v_min = self.make_obj_vars(m, s, it.chain(*self.inst.ops))
		
		return m, (s, e, o, f, r), (z, v_min)

	def lock_max_flow(self, m, f, f_v, locked: Set[Op], thr=0.5):
		m.freeTransform()
		
		locks = []
		eps = 1e-6
		
		for op in it.chain(*self.inst.ops):
			if op.n_prev > 0:
				u_op = sum(f_v[prev, op] for prev in op.prev) > 1 - eps
			else:
				u_op = True

			if u_op and op.n_succ > 0:
				max_idx = np.argmax([f_v[op, succ] for succ in op.succ])
				
				max_succ = op.succ[max_idx]
				if not max_succ in locked:
					print(max_idx, end=' ')
					locked.add(max_succ)
					m.addCons(name=f'lock{op}{max_succ}', cons=(f[op, max_succ] == 1))

					locks.append(max_succ)

		return locks

	def get_op_use(self, f):
		u = {}
		for op in it.chain(*self.inst.ops):
			if op.n_prev > 0:
				u[op] = sum(f[prev, op] for prev in op.prev)
			else:
				u[op] = 1.0



	def change_flow_to_bin(self, m, f):
		for op in it.chain(*self.inst.ops):
			if op.n_succ > 1:
				for succ in op.succ:
					m.chgVarType(f[op, succ], vtype='B')

	def change_res_to_bin(self, m, r_var):
		for v in r_var.values():
			m.chgVarType(v, vtype='B')

	def solve(self):
		m, (s_var, e_var, o_var, f_var, r_var), (z_var, v_min) = self.make_init_model()
		m.setObjective(z_var, sense='minimize')

		# self.change_flow_to_bin(m, f_var)
		# self.change_res_to_bin(m, r_var)

		m.writeProblem()
		m.optimize()

		z1 = m.getVal(z_var)
		
		# print(z1, z2)





if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAUL_DATA)
	for u in inst.res_uses.values():
		print(u)

	# exit()
	sol = Solver(inst)
	sol.solve()

	# for op in sol.get_start_ops():
	#	 print(op)

	
	# for op in sol.get_end_ops():
	#	 print(op)