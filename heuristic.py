#!.venv/bin/python3

import sys

import pyscipopt as scip
import itertools as it
import numpy as np

from instance import Op

class Heur_solver:
	def __init__(self):
		pass

class Flow_max_heuristic(scip.Heur):

	def heurexec(self, heurtiming, nodeinfeasible):
		solver = self.solver
		model = self.model
		result = scip.SCIP_RESULT.DIDNOTRUN

		# Create a solution that is initialised to the LP values
		par_sol = model.createPartialSol(self)
		
		s = model.data['s']
		f = model.data['f']
	
		s_val = { k : model.getSolVal(None, s[k]) for k in s.keys() }
		f_val = { k : model.getSolVal(None, f[k]) for k in f.keys() }
		
		eps = 1e-6

		def rec_func(op: Op):
			used = 0 if op.n_prev > 0 else 1
			for prev in op.prev:
				v = f_val[prev, op]
				if abs(v) > eps and abs(v - 1) > eps:
					rec_func(prev)
					v = f_val[prev, op]
					print(prev, op, v)
					assert(abs(v) < eps or abs(v - 1) < eps)

				used += v

			if op.n_succ == 0:
				return

			if used < eps:
				for succ in op.succ:
					f_val[op, succ] = 0
			
			else:
				assert(abs(used - 1) < eps)

				succs = sorted(op.succ, key=lambda x: (f_val[op, x], -s_val[x]), reverse=True)
				
				f_val[op, succs[0]] = 1
				for succ in succs[1:]:
					f_val[op, succ] = 0

		for op in solver.inst.all_ops:
			rec_func(op)

		for k in f_val.keys():
			model.setSolVal(par_sol, f[k], f_val[k])

		# Now try the solution. Note: This will free the solution afterwards by default.
		stored = model.addSol(par_sol, free=False)
		print(stored)

		if stored:
			return {"result": scip.SCIP_RESULT.FOUNDSOL}
		else:
			return {"result": scip.SCIP_RESULT.DIDNOTFIND}