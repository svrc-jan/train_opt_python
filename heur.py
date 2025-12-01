#!.venv/bin/python3
import sys

from gurobipy import GRB

from instance import Instance
from train_interval import Model

DEFAULT_DATA = 'data/nor1_critical_0.json'


class Heur():
	inst: Instance
	model: Model

	def __init__(self, inst):
		self.inst = inst
		self.model = Model(self.inst)
		self.model.build()
		self.model.set_inst_obj()

	
	def solve(self):
		self.model.gm.Params.OutputFlag = 0

		it = 0

		while True:
			it += 1
			self.model.gm.update()
			self.model.gm.optimize()
			self.model.gm.write('model.lp')

			if (self.model.gm.Status == GRB.INFEASIBLE):
				print(f'it {it} infeasible')
				break

			collisions = self.get_col()

			if len(collisions) == 0:
				print(f'it {it} solved')
				break

			_, r, t1, t2 = min(collisions)

			print(f'it {it} adding:', r, t1, t2)

			self.model.add_cons_res_overlap(r, t1, t2)
	
	def get_col(self):
		res_uses = self.model.get_result_res_uses()

		collisions = []

		for r, ru in res_uses.items():
			for i, (s1, e1, t1) in enumerate(ru):
				for (s2, e2, t2) in ru[i+1:]:
					if (e1 > s2):
						collisions.append((s1, r, t1, t2))

		return collisions



if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	heur = Heur(inst)
	heur.solve()
