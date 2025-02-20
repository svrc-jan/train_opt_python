#!.venv/bin/python3

import sys

from instance import Instance, Op
from pyscipopt import Model, quicksum

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
                    self.res_op[res.idx].append((op, res.time))

        for v in self.res_op.values():
            v.sort(key=lambda op_time: op_time[0].start_obj)

    def get_end_ops(self):
        return [[op for op in train_ops if op.n_succ == 0] for train_ops in self.inst.ops]
    
    def get_start_ops(self):
        return [[op for op in train_ops if op.n_prev == 0] for train_ops in self.inst.ops]

    def make_model(self) -> Model:
        m = Model()

        x = {}
        o = {}
        y = {}
        z = {}
        c = {}

        omega = self.inst.total_dur
        
        for train_ops in self.inst.ops:
            for op in train_ops:
                x[op.idx] = m.addVar(name=f'x{op}', vtype='C', lb=op.start_lb, ub=op.start_ub)
                o[op.idx] = m.addVar(name=f'o{op}', vtype='C', lb=0, ub=None)
                
                y[op.idx] = m.addVar(name=f'y{op.idx}', vtype='B')

                if op.obj is not None:
                    z[op.idx] = m.addVar(name=f'z{op.idx}', vtype='C', lb=0, ub=None)
                    m.addCons(z[op.idx] >= x[op.idx] - op.obj.threshold)

                    c[op.idx] = op.obj.coeff

        m.setObjective(quicksum(c[k]*z[k] for k in z.keys()), sense='minimize')

        for train_ops in self.inst.ops:
            for op in train_ops:
                if op.n_succ == 0:
                    m.addCons(quicksum(y[prev.idx] for prev in op.prev) == 1)
                    m.addCons(y[op.idx] == 1)
                elif op.n_prev == 0:
                    m.addCons(quicksum(y[succ.idx] for succ in op.succ) == 1)
                    m.addCons(y[op.idx] == 1)
                else:
                    m.addCons(quicksum(y[prev.idx] for prev in op.prev) == quicksum(y[succ.idx] for succ in op.succ))

                for succ in op.succ:
                    m.addCons(x[op.idx] + op.dur <= x[succ.idx] + omega*(1-y[succ.idx]))
        

        for res, ops in self.res_op.items():
            for i, (prev, time) in enumerate(ops):
                for (succ, _) in ops[i+1:]:
                    for prev_end in prev.succ:
                        m.addCons(x[prev_end.idx] + time <= x[succ.idx] + omega*(1 - y[prev_end.idx]) + omega*(1 - y[succ.idx]))


        return m
    
if __name__ == '__main__':
    inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAUL_DATA)
    sol = Solver(inst)
    sol.get_res_overlap()
    m = sol.make_model()
    m.writeProblem()
    m.optimize()

    m.get

    # for op in sol.get_start_ops():
    #     print(op)

    
    # for op in sol.get_end_ops():
    #     print(op)