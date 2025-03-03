#!.venv/bin/python3

import collections.abc
import sys

import itertools
from copy import deepcopy, copy
from functools import cmp_to_key
from typing import List, Tuple, Dict
from instance import Instance, Op
from dataclasses import dataclass, field
from pyscipopt import Model, quicksum, Variable, Constraint

from utils import model_and

# DEFAULT_DATA = 'data/testing/headway1.json'
DEFAULT_DATA = 'data/phase1/line1_critical_0.json'

Var = Dict[Tuple[int, int], Variable]
Values = Dict[Tuple[int, int], int]

@dataclass
class Res_use:
	op: Op
	time: int

class Solution:
	inst: Instance
	paths: List[List[Op]]
	res_uses: Dict[int, List[Res_use]]
	obj_val: int

	def __init__(self, inst: Instance, paths=None, res_uses=None, obj_val=None):
		self.inst = inst
		self.paths = paths
		self.res_uses = res_uses
		self.obj_val = obj_val

		if paths is None:
			self.get_fork_sol()

		if obj_val is None:
			self.get_base_sol()

	@staticmethod
	def get_values(m: Model, var: Var|Tuple[Var], to_int=True) -> Values|Tuple[Values]:
		if isinstance(var, (list, tuple)):
			return tuple(Solution.get_values(m, v, to_int) for v in var)
		
		if to_int:
			rv =  { k: int(round(m.getVal(v))) for k, v in var.items() }
		else:
			rv =  { k: m.getVal(v) for k, v in var.items() }

		return rv 


	def make_obj(self, m: Model, s: Var) -> None:
		o = {}
		c = {}

		for t_ops in self.inst.ops:
			for op in t_ops:
				if op.obj and op in s:
					o[op] = m.addVar(name=f'o{op.idx}', vtype='C', lb=0, ub=None)
					c[op] = op.obj.coeff

					m.addCons(name=f'obj{op.idx}', cons=o[op] >= s[op] - op.obj.threshold)

		m.setObjective(quicksum(c[k]*o[k] for k in o.keys()), sense='minimize')

		return o
	
	def make_col_obj(self, m: Model, u: Var):
		o = {}

		for res, ops in inst.res_ops.items():
			o[res] = m.addVar(name=f'o{res}', vtype='C', lb=0, ub=None)
			cons = quicksum(u[op] for op in ops) - len(ops)/2 <= o[res]
			m.addCons(name=f'obj{res}', cons=cons)

		m.setObjective(quicksum(o[k] for k in o.keys()), sense='minimize')



	@staticmethod
	def make_ops_var(m: Model, ops: List[Op], var_name: str='s', force_use=True, var=None) -> Var:
		assert(var_name in ['s', 'u', 'e'])

		if var is None:
			var = {}

		for op in ops:
			if var_name == 's':
				var[op] = m.addVar(name=f's{op.idx}', vtype='C', lb=op.start_lb, ub=op.start_ub)
			
			elif var_name == 'e':
				var[op] = m.addVar(name=f'e{op.idx}', vtype='C', lb=0, ub=None)
			
			elif var_name == 'u':
				var[op] = m.addVar(name=f'u{op.idx}', vtype='B')
				if force_use and (op.n_prev == 0 or op.n_succ == 0 or op.obj):
					m.addCons(name=f'force_u{op.idx}', cons=(var[op] == 1))

		return var

	@staticmethod
	def make_dur_cons(m: Model, s: Var, e: Var, op: Op) -> Constraint:
		cons = s[op] + op.dur <= e[op]
		
		return m.addCons(name=f'dur{op.idx}', cons=cons)


	def make_fork_cons(self, m: Model, u: Var, s: Var, e: Var, op: Op) -> List[Constraint]:
		rv = []

		# if op.n_succ == 1:
		# 	succ = op.succ[0]

		# 	if succ.n_prev == 1:
		# 		cons = u[op] == u[succ]
		# 	else:
		# 		cons = u[op] <= u[succ]
	
		# 	rv.append(m.addCons(name=f'cont{op.idx}', cons=cons))
		
		if op.n_succ == 1:
			succ = op.succ[0]

			if succ.n_prev == 1:
				cons = u[op] == u[succ]
			else:
				cons = u[op] <= u[succ]
	
			rv.append(m.addCons(name=f'cont{op.idx}', cons=cons))

			cons = e[op] == s[succ]
			rv.append(m.addCons(name=f'end{op.idx}', cons=cons))
	
		
		elif op.n_succ > 1:
			succ_vars = [u[succ] for succ in op.succ]
			rv.append(m.addConsSOS1(name='cont_sos', vars=succ_vars))
			rv.append(m.addCons(name='cont_sum', cons=quicksum(succ_vars) >= u[op]))
			

			for succ in op.succ:
				cons1 = e[op] <= s[succ]
				cons2 = e[op] >= s[succ]
				
				rv.append(m.addConsIndicator(name=f'end1{op.idx}{succ.idx}', cons=cons1, binvar=u[succ]))
				rv.append(m.addConsIndicator(name=f'end2{op.idx}{succ.idx}', cons=cons2, binvar=u[succ]))


		return rv

	def make_forks_model(self):
		m = Model()

		u = self.make_ops_var(m, itertools.chain(*self.inst.ops), 'u')
		s = self.make_ops_var(m, itertools.chain(*self.inst.ops), 's')
		e = self.make_ops_var(m, itertools.chain(*self.inst.ops), 'e')
		
		for op in itertools.chain(*self.inst.ops):
			self.make_dur_cons(m, s, e, op)
			self.make_fork_cons(m, u, s, e, op)

		self.make_col_obj(m, u)

		return m, (u, s, e)


	def make_path(self, u: Values, train: int|None=None) -> List[Op]|List[List[Op]]:
		if train is None:
			return [self.make_path(u, i) for i in range(self.inst.n_trains)]
			
		op = self.inst.ops[train][0]
		assert(op.n_prev == 0)
		path = []

		while op.n_succ > 0:
			path.append(op)

			succ = [x for x in op.succ if u[x] == 1]
			assert(len(succ) == 1)

			op = succ[0]

		path.append(op)

		return path

	@staticmethod
	def get_transitions(path: List):
		n = len(path)

		for i in range(n-1):
			yield path[i], path[i+1]

	def make_res_uses(self, path: List, res_uses: Dict[int, List[Res_use]]):
		for op in path:
			for res in op.res:
				if not res.idx in res_uses:
					res_uses[res.idx] = []

				res_uses[res.idx].append(Res_use(
					op=op,
					time=res.time
				))


	def sort_res_uses(self, s: Values, e: Values, res_uses: Dict[int, List[Res_use]]|None=None):
		if res_uses is None:
			res_uses = self.res_uses

		for v in res_uses.values():
			v.sort(key=lambda u: (s[u.op], e[u.op]))

	@staticmethod
	def make_path_cons(m: Model, s: Var, e: Var, path: List[Op]):
		rv = []

		for op, succ in zip(path[:-1], path[1:]):
			cons = e[op] == s[succ]
			rv.append(m.addCons(name=f'end{op.idx}', cons=cons))

		return rv

	@staticmethod
	def make_res_use_cons(m: Model, s: Var, e: Var, res_uses: List[Res_use]):
		rv = []

		for first, second in zip(res_uses[:-1], res_uses[1:]):
			cons = e[first.op] + first.time <= s[second.op]
			rv.append(m.addCons(name=f'r{first.op.idx},{second.op.idx}',cons=cons))

		return rv

	def make_base_model(self):
		m = Model()
		
		s = self.make_ops_var(m, itertools.chain(*self.paths), 's')
		e = self.make_ops_var(m, itertools.chain(*self.paths), 'e')

		for op in itertools.chain(*self.paths):
			self.make_dur_cons(m, s, e, op)

		for p in self.paths:
			self.make_path_cons(m, s, e, p)

		for ru in self.res_uses.values():
			# continue
			self.make_res_use_cons(m, s, e, ru)

		self.make_obj(m, s)

		return m, (s, e)
	
	def get_fork_sol(self):
		mf, (u, s, e) = self.make_forks_model()
		
		mf.hideOutput()
		print('forks - optimizing')
		# mf.setParam('presolving/maxrounds', 20)
		# mf.setParam('limits/stallnodes', 1000)

		# mf.writeProblem()

		mf.optimize()
		print('forks', mf.getObjVal())

		u_v, s_v, e_v = self.get_values(mf, (u, s, e))

		# for k in u_v:
		# 	print(k, s_v[k], e_v[k], '' if u_v[k] == 1 else ' - unused')

		self.paths = self.make_path(u_v)
		self.res_uses = {}
		for p in self.paths:
			self.make_res_uses(p, self.res_uses)
		self.sort_res_uses(s_v, e_v)


	def get_base_sol(self):

		mb, (s, e) = self.make_base_model()

		# mb.hideOutput()
		print('base - optimizing')
		# mb.writeProblem()
		mb.optimize()
		print('base', mb.getObjVal())
		self.obj_val = mb.getObjVal()
		
		# s_v, e_v = self.get_values(mb, (s, e))

		# for k in s_v:
		# 	print(k, s_v[k], e_v[k])

		

	def make_res_improve_cons(self, m: Model, u: Var, s: Var, e: Var, r: Var,
						   	  res_uses: Dict[int, List[Res_use]], op: Op):
		
		for res in op.res:
			if not res.idx in res_uses or len(res_uses[res.idx]) == 0:
				continue

			choices = []
			for i, (first, second) in enumerate(zip([None] + res_uses[res.idx], res_uses[res.idx] + [None])):
				
				var = m.addVar(name=f'r{res.idx},{op.idx},{i}', vtype='B')
				r[res.idx, op.idx, i] = var
				choices.append(var)

				if first:
					cons = e[first.op] + first.time <= s[op]
					m.addConsIndicator(name=f'r1_{res.idx},{first.op.idx},{op.idx}', cons=cons, binvar=var)

				if second:
					cons = e[op] + res.time <= s[second.op]
					m.addConsIndicator(name=f'r2_{res.idx},{op.idx},{second.op.idx}', cons=cons, binvar=var)
				
			m.addCons(name=f'ch{op.idx},{res.idx}', cons=quicksum(choices) == u[op])
			# m.addConsSOS1(name='ch_sos', vars=choices)
	

	def get_filtered_res_uses(self, train: int):
		return { k: [u for u in v if u.op.train_idx != train]
			for k, v in self.res_uses.items() }

	
	def make_improve_model(self, train: int, res_uses: Dict[int, List[Res_use]]|None=None):
		m = Model()
		
		if res_uses is None:
			res_uses = self.get_filtered_res_uses(train)
		
		u = {}
		s = {}
		e = {}
		r = {}

		for i, p in enumerate(self.paths):
			if i == train:
				continue

			self.make_ops_var(m, p, 's', var=s)
			self.make_ops_var(m, p, 'e', var=e)

			for op in p:
				self.make_dur_cons(m, s, e, op)

			self.make_path_cons(m, s, e, p)

		for ru in res_uses.values():
			self.make_res_use_cons(m, s, e, ru)

		self.make_ops_var(m, self.inst.ops[train], 's', var=s)
		self.make_ops_var(m, self.inst.ops[train], 'e', var=e)
		self.make_ops_var(m, self.inst.ops[train], 'u', var=u)

		for op in self.inst.ops[train]:
			self.make_dur_cons(m, s, e, op)
			self.make_fork_cons(m=m, u=u, s=s, e=e, op=op)
			self.make_res_improve_cons(m=m, u=u, s=s, e=e, r=r, 
							  		   res_uses=res_uses, op=op)

		self.make_obj(m, s)

		return m, (u, s, e, r)
	

	def get_improve_sol(self, train: int):
		filt_res_uses = self.get_filtered_res_uses(train)
		m, (u, s, e, r) = self.make_improve_model(train, filt_res_uses)

		m.hideOutput()
		print(f'impr {train} - optimizing')
		m.setParam('limits/stallnodes', 1000)
		m.writeProblem()
		m.optimize()
		print(f'impr {train}', m.getObjVal())
		# print(filt_res_uses)

		u_v, s_v, e_v, r_v = self.get_values(m, (u, s, e, r))

		# for k, v in r_v.items():
		# 	if u_v[self.inst.ops[k[1][0]][k[1][1]]] == 1:
		# 		print(k, v)

		new_path = self.make_path(u_v, train)

		imp_paths = [[op for op in p] if i != train else new_path for i, p in enumerate(self.paths)]
		
		imp_res_uses = {}
		for p in imp_paths:
			self.make_res_uses(p, imp_res_uses)
		self.sort_res_uses(s_v, e_v, imp_res_uses)
		
		imp_sol = Solution(self.inst, imp_paths, imp_res_uses, m.getObjVal())
	
		# print('old path', self.paths[train])
		# print('new path', new_path)

		# print(new_res_uses)

		return imp_sol
	
if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
	sol = Solution(inst)

	
	# while True:
	# 	best = sol
	for tr in range(inst.n_trains):
		ns = sol.get_improve_sol(tr)
	# 		if ns.obj_val < best.obj_val:
	# 			best = ns
		
	# 	if best.obj_val >= sol.obj_val:
	# 		break
		
	# 	sol = best
			



