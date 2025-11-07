#!.venv/bin/python3

import sys

from solver import Solution, Solver
from instance import Instance, Op, Op_idx

class Heuristic:
	inst: Instance
	
	def __init__(self, inst: Instance):
		self.inst = inst
