#!.venv/bin/python3

import sys

import pyscipopt as scip
import itertools as it
import numpy as np

from typing import List, Tuple, Dict, Set
from instance import Instance, Op, Res_use
from dataclasses import dataclass, field

from model import Model
from conshdlr import Res_conshdlr

DEFAULT_DATA = 'data/testing/headway1.json'
# DEFAUL_DATA = 'data/phase1/line1_critical_0.json'


class Solver:
	inst: Instance

	def __init__(self, inst):
		self.inst = inst 
	

	@staticmethod
	def fix_var_to_one(m, var, keys):
		if not 'fix_cons' in m.data:
			m.data['fix_cons'] = {}

		if not var in m.data['fix_cons']:
			m.data['fix_cons'][var] = {}

		fix_cons = m.data['fix_cons'][var]
		for k in keys:
			fix_cons[k] = m.addCons(name=f'fix_{var},{k}', cons=(m.data[var][k] == 1))

	@staticmethod
	def set_var_to_binary(m, var, keys):
		m.freeTransform()
		for k in keys:
			m.chgVarType(m.data[var][k], vtype='B')


	def solve(self):
		model = Model(self.inst)
		# model.hideOutput()
		
		conshdlr = Res_conshdlr(model, self.inst.res_uses, self.inst.max_train_dur)

		model.includeConshdlr(conshdlr, "Train_opt", "Constraint handler resource constrains",
			sepapriority=0, enfopriority=-1, chckpriority=-1, sepafreq=-1, propfreq=-1,
			eagerfreq=-1, maxprerounds=0, delaysepa=False, delayprop=False, needscons=False,
			presoltiming=scip.SCIP_PRESOLTIMING.FAST, proptiming=scip.SCIP_PROPTIMING.BEFORELP)

		# heuristic = Flow_max_heuristic()
		# heuristic.model = model
		# heuristic.solver = self
		# model.includeHeur(heuristic, "SimpleRounding", "custom heuristic implemented in python", "Y",
		# 		 timingmask=scip.SCIP_HEURTIMING.DURINGLPLOOP)


		model.setParam("misc/allowstrongdualreds", False)
		# model.setParam("misc/allowweakdualreds", False)
		# model.setParam('lp/initalgorithm', 'p')
		# model.setParam('lp/resolvealgorithm', 'p')
		
		# model.setParam('parallel/mode', 0)


		model.setParam('limits/time', 120)

		model.setParam('parallel/mode', 0)
		# model.solveConcurrent()
		model.optimize()
		model.writeProblem()

		# m.hideOutput()
		print(model.getObjVal())

		s_v = model.get_values(model, 's', to_int=True)
		e_v = model.get_values(model, 'e', to_int=True)
		f_v = model.get_values(model, 'f')

		eps = 1e-5
		used = {}

		print(conshdlr.conscheck(None, model.getBestSol(), None , None , None, None))

		# for op in self.inst.all_ops:
		# 	u = sum(f_v[op, succ] for succ in op.succ) if op.n_succ > 0 else 1
		# 	assert(abs(u) < eps or abs(u - 1) < eps)
		# 	used[op] = int(round(u))

		# for op in it.chain(*self.inst.ops):			
		# 	if used[op] > eps:
		# 		s = [x for x in op.succ if used[x] > eps]
		# 		assert(len(s) == 1 or op.n_succ == 0)
		# 		print(f'{op}, {s_v[op] + op.dur <= e_v[op]} {used[op]:.2f}', op.n_succ, s)



if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)

	# exit()
	sol = Solver(inst)
	sol.solve()

