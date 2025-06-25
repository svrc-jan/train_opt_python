#!.venv/bin/python3

import sys

import pyscipopt as scip
import itertools as it
import numpy as np

from instance import Instance, Op
from model import Model
from conshdlr import Res_conshdlr
from utils import is_bin, round_to_int

DEFAULT_DATA = 'data/testing/headway1.json'

class Heur_solver:
	def __init__(self, inst):
		self.inst = inst

	@staticmethod
	def fix_var(m, var, k, v, eps=1e-6):
		if not 'fix_conss' in m.data:
			m.data['fix_conss'] = {}

		fix_conss = m.data['fix_conss']
		if not var in fix_conss:
			fix_conss[var] = {}

		if not k in fix_conss[var]:
			fix_conss[var][k] = m.addCons(name=f'fix{k},{v}', cons=(m.data[var][k] == v))

		else:
			rhs = m.getRhs(fix_conss[var][k])
			if abs(rhs - v) > eps:
				m.chgRhs(fix_conss[var][k], v)
		
	def solve(self):
		model = Model(self.inst, is_heur=True)

		conshdlr = Res_conshdlr(model, self.inst.res_uses, self.inst.max_train_dur)

		model.includeConshdlr(conshdlr, "Train_opt", "Constraint handler resource constrains",
			sepapriority=0, enfopriority=-1, chckpriority=-1, sepafreq=-1, propfreq=-1,
			eagerfreq=-1, maxprerounds=0, delaysepa=False, delayprop=False, needscons=False,
			presoltiming=scip.SCIP_PRESOLTIMING.FAST, proptiming=scip.SCIP_PROPTIMING.BEFORELP)



		model.setParam("misc/allowstrongdualreds", False)

		
		while True:
			model.optimize()
			assert(model.getStatus() == "optimal")

			f_val = Model.get_values(model, 'f')

			non_bin_ks = [ k for k, v in f_val.items() if not is_bin(v) ]
			if len(non_bin_ks) == 0:
				break
			
			model.freeTransform()
			for k, v in f_val.items():
				if abs(v - 1) < 1e-06:
					self.fix_var(model, 'f', k, 1)
			
			non_bin_ks.sort(key=lambda x: f_val[x], reverse=True)
			self.fix_var(model, 'f', non_bin_ks[0], 1)

		print(conshdlr.conscheck(None, model.getBestSol(), None , None , None, None))
			
class Flow_max_heuristic(scip.Heur):

	def heurexec(self, heurtiming, nodeinfeasible):
		solver = self.solver
		model = self.model
		

		e = model.data['e']
		f = model.data['f']
	
		e_val = { k : model.getSolVal(None, e[k]) for k in e.keys() }
		f_val = { k : model.getSolVal(None, f[k]) for k in f.keys() }

		eps = 1e-06

		for k, v in f.items():
			c = model.getVarRedcost(v)
			x = f_val[k]
			if (c > eps and not is_bin(x)):
				print(k, x, c)

		
		heur_solv = Heur_solver(solver.inst)
		heur_solv.get_flow(f_val)

		
		return {"result": scip.SCIP_RESULT.DIDNOTFIND}
	

if __name__ == '__main__':
	inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA)
	sol = Heur_solver(inst)
	sol.solve()


