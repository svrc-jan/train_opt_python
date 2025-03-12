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
	def get_res_collisions(self, solution):
		eps = 1e-5

		model = self.model
		solver = self.solver

		s = model.data['s']
		e = model.data['e']
		f = model.data['f']

		collisions = []

		used = {}

		for res_idx, res_uses in solver.inst.res_uses.items():
			for i, ru1 in enumerate(res_uses):
				op1 = ru1.op
				if not op1 in used:
					used[op1] = op1.n_succ == 0 or \
						sum(model.getSolVal(solution, f[op1, succ1]) for succ1 in op1.succ) > eps
				
				if not used[op1]:
					continue

				s1 = int(round(model.getSolVal(solution, s[op1])))
				e1 = int(round(model.getSolVal(solution, e[op1]) + ru1.time))

				for ru2 in res_uses[i+1:]:
					op2 = ru2.op

					if op1.train_idx == op2.train_idx:
						continue

					if not op2 in used:
						used[op2] = op2.n_succ == 0 or \
							sum(model.getSolVal(solution, f[op2, succ2]) for succ2 in op2.succ) > eps
				
					if not used[op2]:
						continue

					s2 = int(round(model.getSolVal(solution, s[op2])))
					e2 = int(round(model.getSolVal(solution, e[op2]) + ru2.time))

					if (s1 < e2) and (s2 < e1):
						collisions.append((res_idx, ru1, ru2))

		return collisions

	def make_conss_from_collisions(self, collisions):

		model = self.model
		if not 'r' in model.data:
			model.data['r'] = {}

		s = model.data['s']
		e = model.data['e']
		# o = model.data['o']
		f = model.data['f']
		r = model.data['r']

		for res_idx, ru1, ru2 in collisions:
			op1 = ru1.op
			op2 = ru2.op
			
			v1 = model.addVar(name=f'r{res_idx},{op1},{op2}', vtype='B')
			v2 = model.addVar(name=f'r{res_idx},{op2},{op1}', vtype='B')

			r[res_idx, op1, op2] = v1
			r[res_idx, op2, op1] = v2

			use_op1 = scip.quicksum(f[op1, succ1] for succ1 in op1.succ) if op1.n_succ > 0 else 1
			use_op2 = scip.quicksum(f[op2, succ2] for succ2 in op2.succ) if op2.n_succ > 0 else 1

			cons = 2*(v1 + v2) == use_op1 + use_op2
			model.addCons(name=f'ch{res_idx},{op1},{op2}', cons=cons)

			M1 = self.solver.inst.max_train_dur[op1.train_idx] + \
				 self.solver.inst.max_train_dur[op2.train_idx]

			cons1 = e[op1] + ru1.time <= s[op2] + M1*(1 - v1)
			cons2 = e[op2] + ru2.time <= s[op1] + M1*(1 - v2)

			# model.addConsIndicator(name=f'res{res_idx},{op1},{op2}', cons=cons1, binvar=v1, removable=True)
			# model.addConsIndicator(name=f'res{res_idx},{op2},{op1}', cons=cons2, binvar=v2, removable=True)

			model.addCons(name=f'res{res_idx},{op1},{op2}', cons=cons1, removable=True)
			model.addCons(name=f'res{res_idx},{op2},{op1}', cons=cons2, removable=True)

			# cons1 = o[op1] + 1 <= o[op2] + M2*(1 - v1)
			# cons2 = o[op2] + 1 <= o[op1] + M2*(1 - v2)

			# model.addCons(name=f'res_ord{res_idx},{op1},{op2}', cons=cons1, removable=True)
			# model.addCons(name=f'res_ord{res_idx},{op2},{op1}', cons=cons2, removable=True)

	def conscheck(self, constraints, solution, checkintegrality, checklprows, printreason, completely):
		collisions = self.get_res_collisions(solution)

		if len(collisions) == 0:
			return {"result": scip.SCIP_RESULT.FEASIBLE}
		
		return {"result": scip.SCIP_RESULT.INFEASIBLE}

	def consenfolp(self, constraints, nusefulconss, solinfeasible):
		collisions = self.get_res_collisions(solution=None)
		# print(collisions)
		if len(collisions) == 0:
			return {"result": scip.SCIP_RESULT.FEASIBLE}
		
		self.make_conss_from_collisions(collisions)
		return {"result": scip.SCIP_RESULT.CONSADDED}
			

	def conslock(self, constraint, locktype, nlockspos, nlocksneg):
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

	def make_dur_cons(self, m, op: Op):
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


	def make_init_model(self):
		m = scip.Model()

		m.data = {}

		for op in it.chain(*self.inst.ops):
			self.make_op_vars(m, op)

		for op in it.chain(*self.inst.ops):
			self.make_dur_cons(m, op)

		for op in it.chain(*self.inst.ops):
			self.make_end_cons(m, op)
	
		for op in it.chain(*self.inst.ops):
			self.make_flow_cons(m, op)

		self.make_obj_vars(m, it.chain(*self.inst.ops))

		m.data['obj1_cons'] = None
		
		return m


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

	@staticmethod
	def set_obj1(m):
		u = m.data['u']
		c = m.data['obj_coef']

		m.setObjective(scip.quicksum(c[k]*u[k] for k in u.keys()), sense='minimize')

	def solve(self):
		model = self.make_init_model()
		self.set_obj1(model)
		
		conshdlr = Res_conshdlr()
		conshdlr.model = model
		conshdlr.solver = self

		model.includeConshdlr(conshdlr, "Train_opt", "Constraint handler resource constrains",
			sepapriority=0, enfopriority=-1, chckpriority=-1, sepafreq=-1, propfreq=-1,
			eagerfreq=-1, maxprerounds=0, delaysepa=False, delayprop=False, needscons=False,
			presoltiming=scip.SCIP_PRESOLTIMING.FAST, proptiming=scip.SCIP_PROPTIMING.BEFORELP)


		model.setParam("misc/allowstrongdualreds", False)
		# model.setParam("misc/allowweakdualreds", False)
		
		model.setParam('parallel/mode', 0)
		model.optimize()

		# m.hideOutput()
		print(model.getObjVal())

		s_v = self.get_values(model, 's', to_int=True)
		e_v = self.get_values(model, 'e', to_int=True)
		f_v = self.get_values(model, 'f')

		eps = 1e-5
		used = {}

		for op in it.chain(*self.inst.ops):
			u = sum(f_v[op, succ] for succ in op.succ)
			assert(abs(u) < eps or abs(u - 1) < eps)
			used[op] = int(round(u))

		for op in it.chain(*self.inst.ops):			
			if used[op] > eps:
				s = [x for x in op.succ if used[x] > eps]
				print(f'{op}, {s_v[op] + op.dur <= e_v[op]} {used[op]:.2f}', op.n_succ, s)

		# for res_idx, res_uses in self.inst.res_uses.items():
		# 	l = [x for x in res_uses if used[x.op] > eps]
		# 	l.sort(key=lambda x: s_v[x.op])

		# 	if len(l) > 0:
		# 		print(res_idx, '- res uses')
		# 	for u in l:
		# 		print(u.op, s_v[u.op], e_v[u.op] + u.time)

if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAUL_DATA)

	# exit()
	sol = Solver(inst)
	sol.solve()


	# for op in sol.get_start_ops():
	#	 print(op)

	
	# f`o`r op in sol.get_end_ops():
	#	 print(op)