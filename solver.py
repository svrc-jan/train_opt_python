#!.venv/bin/python3

import sys

import pyscipopt as scip
import itertools as it
import numpy as np

from typing import List, Tuple, Dict, Set
from instance import Instance, Op, Res_use
from dataclasses import dataclass, field


DEFAUL_DATA = 'data/testing/headway1.json'
# DEFAUL_DATA = 'data/phase1/line1_critical_0.json'

class Res_conshdlr(scip.Conshdlr):
	pass

class Solver:
	inst: Instance

	def __init__(self, inst):
		self.inst = inst 
	
	@staticmethod
	def get_values(m, var, to_int=False):
		if isinstance(var, (list, tuple)):
			return tuple(Solver.get_values(m, v, to_int) for v in var)
		

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
		
		for var in ['s', 'e', 'o', 'f']:
			if not var in m.data:
				m.data[var] = {}

		s = m.data['s']
		e = m.data['e']
		o = m.data['o']
		f = m.data['f']

		s[op] = m.addVar(name=f's{op}', vtype='C', lb=op.start_lb, ub=op.start_ub)
		o[op] = m.addVar(name=f'e{op}', vtype='C', lb=0, ub=None)

		for succ in op.succ:
			# vtype = 'B' if op.n_succ > 1 else 'C'
			f[op, succ] = m.addVar(name=f'f{op},{succ}', vtype='C', lb=0, ub=1)

	@staticmethod
	def make_dur_cons(m, op: Op):
		s = m.data['s']
		e = m.data['e']

		cons = s[op] + op.dur <= e[op]
		m.addCons(name=f'dur{op}', cons=cons)
		

	@staticmethod
	def make_end_cons(m, op: Op):
		s = m.data['s']
		e = m.data['e']
		o = m.data['o']
		f = m.data['f']

		M = op.dur

		if op.n_succ == 1:
			succ = op.succ[0]

			cons = e[op] == s[succ]
			m.addCons(name=f'end{op},{succ}', cons=cons)

		elif op.n_succ > 1:
			for succ in op.succ:
				

				frac = succ.frac_flow
				aux = m.addVar(name=f'end_aux{op},{succ}', vtype='C', lb=0, ub=1)
				
				aux_cons = aux <= frac - f[op, succ]


				cons1 = e[op] <= s[succ] + M*(1 - f[op, succ])
				cons2 = e[op] >= s[succ] - M*(1 - f[op, succ])

				# m.addConsIndicator(name=f'end1{op},{succ}', cons=cons1, binvar=f[op, succ])
				# m.addConsIndicator(name=f'end2{op},{succ}', cons=cons2, binvar=f[op, succ], initial=False)

				m.addCons(name=f'end1{op},{succ}', cons=cons1)
				m.addCons(name=f'end2{op},{succ}', cons=cons2)

				cons_o = cons=o[op] + 1 <= o[succ]
				m.addCons(name=f'ord{op},{succ}', cons=cons_o)


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

	def make_init_model(self):
		m = scip.Model()

		m.data = {}

		for op in it.chain(*self.inst.ops):
			self.make_op_vars(m, op)

		for op in it.chain(*self.inst.ops):
			self.make_dur_cons(m, op)

		# for op in it.chain(*self.inst.ops):
		# 	self.make_end_cons(m, op)
	
		for op in it.chain(*self.inst.ops):
			self.make_flow_cons(m, op)

		self.make_res_vars(m)

		self.make_obj_vars(m, it.chain(*self.inst.ops))

		m.data['obj1_cons'] = None
		
		return m

	@staticmethod
	def find_non_bin_var(m, var):
		val = Solver.get_values(m, var)

		keys = []

		eps = 1e-6
		for k, v in val.items():
			if abs(v) > eps and abs(v - 1) > eps:
				keys.append(k)

		
		keys.sort(key=lambda x: val[x], reverse=True)
	
		return keys

	@staticmethod
	def fix_var_to_one(m, var, keys):
		if not 'fix_cons' in m.data:
			m.data['fix_cons'] = {}

		if not var in m.data['fix_cons']:
			m.data['fix_cons'][var] = {}

		fix_cons = m.data['fix_cons'][var]
		m.freeTransform()
		for k in keys:
			fix_cons[k] = m.addCons(name=f'fix_{var},{k}', cons=(m.data[var][k] == 1))

	@staticmethod
	def set_var_to_binary(m, var, keys):
		m.freeTransform()
		for k in keys:
			m.chgVarType(m.data[var][k], vtype='B')

	@staticmethod
	def optim_obj1(m):
		m.freeTransform()
		if m.data['obj1_cons'] is not None:
			m.delCons(m.data['obj1_cons'])
			m.data['obj1_cons'] = None

		u = m.data['u']
		c = m.data['obj_coef']

		m.setObjective(scip.quicksum(c[k]*u[k] for k in u.keys()), sense='minimize')
		m.optimize()

		if m.getStatus() == 'optimal':
			return m.getObjVal()
		else:
			return None

	@staticmethod
	def optim_obj2(m, obj1_limit=None):
		if m.getStatus() == "infeasible":
			return None

		if obj1_limit is None:
			obj1_limit = m.getObjVal()

		eps = 1e-6

		v_val = Solver.get_values(m, 'v')
		
		if len([x for x in v_val.values() if x > 0]) == 0:
			return 0
		
		u = m.data['u']
		v = m.data['v']
		c = m.data['obj_coef']

		
		m.freeTransform()

		cons = scip.quicksum(c[k]*u[k] for k in u.keys()) <= obj1_limit
		m.data['obj1_cons'] = m.addCons(name='obj1_limit', cons=cons)

		m.setObjective(scip.quicksum(c[k]*v[k] for k in v.keys()), sense='maximize')
		m.optimize()

		if m.getStatus() == 'optimal':
			return m.getObjVal()
		else:
			return None


	def make_res_vars(self, m):
		if not 'r' in m.data:
			m.data['r'] = {}

		f = m.data['f']
		r = m.data['r']

		for res, uses in self.inst.res_uses.items():
			for i, u1 in enumerate(uses):
				for u2 in uses[i+1:]:
					op1 = u1.op
					op2 = u2.op

				if op1.train_idx == op2.train_idx:
					continue
				
				if op1.n_succ == 0 or op2.n_succ == 0:
					continue

				vars1 = []
				vars2 = []

				for succ1 in op1.succ:
					v1 = m.addVar(name=f'r{res},{op1},{succ1},{op2}', vtype='C', lb=0, ub=1)
					
					r[res, op1, succ1, op2] = v1
					vars1.append(v1)

				for succ2 in op2.succ:
					v2 = m.addVar(name=f'r{res},{op2},{succ2},{op1}', vtype='C', lb=0, ub=1)
					
					r[res, op2, succ2, op1] = v2
					vars2.append(v2)


				use_op1 = scip.quicksum(f[op1, succ1] for succ1 in op1.succ)
				use_op2 = scip.quicksum(f[op2, succ2] for succ2 in op2.succ)

				for v1, succ1 in zip(vars1, op1.succ):
					cons = 2.0*(v1 + scip.quicksum(vars2)) == f[op1, succ1] + use_op2
					m.addCons(name=f'ch{res},{op1},{succ1},{op2}', cons=cons)

				for v2, succ2 in zip(vars2, op2.succ):
					cons = 2.0*(v2 + scip.quicksum(vars1)) == f[op2, succ2] + use_op1
					m.addCons(name=f'ch{res},{op2},{succ2},{op1}', cons=cons) 


	def make_res_cons(self, m, s_val=None, mult=1):
		m.freeTransform()

		if 'res_conss' in m.data:
			for cons in m.data['res_conss']:
				m.delCons(cons)

		m.data['res_conss'] = []

		s = m.data['s']
		# e = m.data['e']
		o = m.data['o']
		f = m.data['f']
		r = m.data['r']

		res_conss = m.data['res_conss']

		for res, uses in self.inst.res_uses.items():
			for i, u1 in enumerate(uses):
				for u2 in uses[i+1:]:
					op1 = u1.op
					op2 = u2.op

				if op1.train_idx == op2.train_idx:
					continue

				assert(op1.n_succ > 0 or op2.n_succ > 0)

				if s_val:
					M = mult*(abs(s_val[op1] - s_val[op2]) + op1.dur + op2.dur)
				else:
					M = mult*self.inst.avg_dur

				if op1.n_succ == 0:
					for succ2 in op2.succ:
						cons = s[succ2] + u2.time <= s[op1] + M*(1 - f[op2, succ2])
						res_conss.append(m.addCons(name=f'res{res},{op2},{succ2},{op2}', cons=cons))

				elif op2.n_succ == 0:
					for succ1 in op1.succ:
						cons = s[succ1] + u1.time <= s[op2] + M*(1 - f[op1, succ1])
						res_conss.append(m.addCons(name=f'res{res},{op1},{succ1},{op2}', cons=cons))
					
				else:
					for succ1 in op1.succ:
						v = r[res, op1, succ1, op2]

						cons = s[succ1] + u1.time <= s[op2] + M*(1 - v)
						res_conss.append(m.addCons(name=f'res{res},{op1},{succ1},{op2}', cons=cons))

					for succ2 in op2.succ:
						v = r[res, op2, succ2, op1]

						cons = s[succ2] + u2.time <= s[op1] + M*(1 - v)
						res_conss.append(m.addCons(name=f'res{res},{op2},{succ2},{op2}', cons=cons))


	def solve(self):
		m = self.make_init_model()
		
		m.hideOutput()

		i = 0
		s_val = None
		while True:
			mult = 1
			while True:

				self.make_res_cons(m, s_val, mult)
				
				z1 = self.optim_obj1(m)
				# z2 = self.optim_obj2(m)
				
				if m.getStatus() == 'optimal' :
					break
				
				mult = 1.5*mult


			s_val = self.get_values(m, 's')
			non_bin = self.find_non_bin_var(m, 'r')
			if len(non_bin) > 0:
				self.fix_var_to_one(m, 'r', non_bin[:1])
				print('r', i, mult, z1)

			else:
				break

			
			
			i += 1

		i = 0

		non_bin_r = self.find_non_bin_var(m, 'r')
		self.make_res_cons(m, s_val, mult=10)
		m.hideOutput(quiet=False)

		while True:
			if len(non_bin_r) > 0:
				self.set_var_to_binary(m, 'r', non_bin_r)

				z1 = self.optim_obj1(m)
				print('r', i, z1)

			else:
				break
			
			non_bin_r = self.find_non_bin_var(m, 'r')
			i += 1 


if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAUL_DATA)

	# exit()
	sol = Solver(inst)
	sol.solve()

	# for op in sol.get_start_ops():
	#	 print(op)

	
	# f`o`r op in sol.get_end_ops():
	#	 print(op)