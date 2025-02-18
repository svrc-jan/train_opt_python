#!.venv/bin/python3

import sys

from instance import Instance, Op
from pyscipopt import Model

DEFAUL_DATA = 'data/testing/headway1.json'

class Solver:
    inst: Instance

    def __init__(self, inst):
        self.inst = inst 
    
    def get_res_overlap(self):
        self.res_op = {}
        for train_ops in self.inst.ops:
            for op in train_ops:
                for res in op.res:
                    if not res.idx in self.res_op:
                        self.res_op[res.idx] = []
                    self.res_op[res.idx].append(op)

    def make_model(self):
        m = Model()

        x = {}
        y = {}

        
        for train_ops in self.inst.ops:
            for op in train_ops:
                x[op.idx] = m.addVar(name=f'x{op}', vtype='C', lb=op.start_lb, ub=op.start_ub)
                
                if op.n_prev == 0 or op.n_succ == 0 or
                y[op.idx] = m.addVar(name=f'y{op}', vtype='B')

        for train_ops in self.inst.ops:
            for op in train_ops:
                start = op
        

if __name__ == '__main__':
    inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAUL_DATA)
    sol = Solver(inst)
    sol.get_res_overlap()
    for k, v in sol.res_op.items():
        print(k, len(v))