#!.venv/bin/python3

import collections.abc
import sys

import itertools
from functools import cmp_to_key
from typing import List, Tuple, Dict
from instance import Instance, Op
from dataclasses import dataclass, field
from pyscipopt import Model, quicksum, Variable, Constraint

# DEFAUL_DATA = 'data/testing/headway1.json'
DEFAUL_DATA = 'data/phase1/line1_critical_0.json'

Var = Dict[Tuple[int, int], Variable]
Values = Dict[Tuple[int, int], int]

@dataclass
class Res_use:
	op: Op
	succ: Op|None
	time: int

class Solution:
	inst: Instance
	paths: List[List[Op]]
	res_uses: Dict[int, List[Res_use]]

	def __init__(self, inst: Instance):
		self.inst = inst


	@staticmethod
	def get_values(m: Model, var: Var|Tuple[Var], to_int=True) -> Values|Tuple[Values]:
		if isinstance(var, (list, tuple)):
			print('iter')
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


	@staticmethod
	def make_ops_var(m: Model, ops: List[Op], var_name: str='s', force_use=True) -> Var:
		assert(var_name in ['s', 'u'])

		var = {}

		for op in ops:
			if var_name == 's':
				var[op] = m.addVar(name=f's{op.idx}', vtype='C', lb=op.start_lb, ub=op.start_ub)
			elif var_name == 'u':
				var[op] = m.addVar(name=f'u{op.idx}', vtype='B')
				if force_use and (op.n_prev == 0 or op.n_succ == 0 or op.obj):
					m.addCons(name=f'force_u{op.idx}', cons=(var[op] == 1))

		return var


	def make_fork_cons(self, m: Model, u: Var, s: Var, op: Op) -> List[Constraint]:

		M = self.inst.total_dur

		rv = []

		if op.n_succ == 1:
			succ = op.succ[0]

			if succ.n_prev == 1:
				cons = u[op] == u[succ]
			else:
				cons = u[op] <= u[succ]
	
			rv.append(m.addCons(name=f'cont{op.idx}', cons=cons))
	
		
		elif op.n_succ > 1:
			cons_lb = quicksum(u[succ] for succ in op.succ) >= u[op]
			cons_ub = quicksum(u[succ] for succ in op.succ) <= 1

			rv.append(m.addCons(name=f'fork_lb{op.idx}', cons=cons_lb))
			rv.append(m.addCons(name=f'fork_ub{op.idx}', cons=cons_ub))

		for succ in op.succ:
			cons = s[op] + op.dur <= s[succ] + \
				   M*(2 - u[op] - u[succ])

			rv.append(m.addCons(name=f'dur{op.idx},{succ.idx}', cons=cons))

		return rv
	

	def make_forks_model(self):
		m = Model()

		u = self.make_ops_var(m, itertools.chain(*self.inst.ops), 'u')
		s = self.make_ops_var(m, itertools.chain(*self.inst.ops), 's')
		
		for op in itertools.chain(*self.inst.ops):
			print(op)
			self.make_fork_cons(m, u, s, op)

		self.make_obj(m, s)

		return m, (u, s)


	def make_paths(self, u: Values) -> List[List[Op]]:
		paths = []
		for t_ops in self.inst.ops:
			
			op = t_ops[0]
			path = []

			while op.n_succ > 0:
				path.append(op)

				succ = [x for x in op.succ if u[x] == 1]
				assert(len(succ) == 1)

				op = succ[0]

			path.append(op)

			paths.append(path)

		return paths


	def make_res_uses(self, u: Values, s: Values):
		res_uses = {}
		
		for t_ops in self.inst.ops:
			for op in t_ops:
				succ = [x for x in op.succ if u[x] == 1]
				assert(len(succ) == 1 or op.n_succ == 0)

				succ = succ[0] if op.n_succ > 0 else None

				for res in op.res:
					if not res.idx in res_uses:
						res_uses[res.idx] = []

					res_uses[res.idx].append(Res_use(
						op=op,
						succ=succ,
						time=res.time
					))

		def res_use_cmp(x: Res_use, y: Res_use):
			assert(not (x.succ is None and y.succ is None))
			if x.succ is None:
				return False
			
			if y.succ is None:
				return True
			
			if s[x.op] == s[y.op]:
				return s[x.succ] < s[y.succ]
			
			return s[x.op] < s[y.op]

		for v in res_uses.values():
			v.sort(key=cmp_to_key(res_use_cmp))


		return res_uses


if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAUL_DATA)
	sol = Solution(inst)

	m, (u, s) = sol.make_forks_model()
	m.writeProblem()
	m.optimize()

	u_v, s_v = sol.get_values(m, (u, s))

	print(sol.make_paths(u_v))
	print(sol.make_res_uses(u_v, s_v))