#!.venv/bin/python3

import sys

from typing import List, Tuple, Dict
from itertools import chain
from instance import Instance, Op
from dataclasses import dataclass
from pyscipopt import Model, quicksum

DEFAUL_DATA = 'data/testing/headway1.json'

@dataclass
class Res_use:
    op: Op
    succ: Op|None
    time: int

    start: int
    end: int

class Res_col:
    u1: Res_use
    u2: Res_use

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
            # v.sort(key=lambda op_time: op_time[0].start_obj)
            pass

    def get_end_ops(self):
        return [[op for op in train_ops if op.n_succ == 0] for train_ops in self.inst.ops]
    
    def get_start_ops(self):
        return [[op for op in train_ops if op.n_prev == 0] for train_ops in self.inst.ops]

    def make_model_obj(self, m, ops, start):
        obj = {}
        obj_c = {}
        for train_ops in ops:
            for op in train_ops:
                if op.obj is not None:
                    obj[op.idx] = m.addVar(name=f'obj{op.idx}', vtype='C', lb=0, ub=None)
                    m.addCons(name=f'obj{op.idx}', 
                        cons=obj[op.idx] >= start[op.idx] - op.obj.threshold)

                    obj_c[op.idx] = op.obj.coeff
        
        m.setObjective(quicksum(obj_c[k]*obj[k] for k in obj.keys()), sense='minimize')

    def make_forks_model(self):
        m = Model()

        start = {}
        used = {}

        omega = self.inst.total_dur
        
        for train_ops in self.inst.ops:
            for op in train_ops:
                start[op.idx] = m.addVar(name=f'start{op.idx}', vtype='C', 
                                         lb=op.start_lb, ub=op.start_ub)
                
                used[op.idx] = m.addVar(name=f'used{op.idx}', vtype='B')

                if op.n_prev == 0 or op.n_succ == 0 or op.obj is not None:
                    m.addCons(used[op.idx] == 1)

        for train_ops in self.inst.ops:
            for op in train_ops:
                if op.n_succ > 0:
                    m.addCons(name=f'fork_lb{op.idx}',
                        cons=quicksum(used[succ.idx] for succ in op.succ) >= used[op.idx])
                    m.addCons(name=f'fork_ub{op.idx}',
                        cons=quicksum(used[succ.idx] for succ in op.succ) <= 1)

                for succ in op.succ:
                    m.addCons(name=f'dur{op.idx},{succ.idx}',
                        cons=start[op.idx] + op.dur <= start[succ.idx] + omega*(1-used[succ.idx]))
    
        self.make_model_obj(m, self.inst.ops, start)

        return m, (used, start)
    
    def solve_forks_model(self):
        m, (used, start) = self.make_forks_model()
        
        m.optimize()
        
        used_val = { op.idx: m.getVal(used[op.idx]) for op in chain(*self.inst.ops)}
        start_val = { op.idx: m.getVal(start[op.idx]) for op in chain(*self.inst.ops)}

        return used_val, start_val

    def get_paths(self, used) -> List[List[Op]]:
        paths = []
        for train_ops in self.inst.ops:
            train_path = []

            op = train_ops[0]
            train_path.append(op)

            while op.n_succ > 0:
                op = train_path[-1]
                
                succ = [x for x in op.succ if used[x.idx] == 1]

                op = succ[0]
                train_path.append(op)

            paths.append(train_path)

        return paths
    

    def get_collisions(self, paths, start):
        res_uses = {}

        for path in paths:
            for op, succ in zip(path, path[1:] + [None]):
                for res in op.res:
                    if not res.idx in res_uses:
                        res_uses[res.idx] = []

                    res_uses[res.idx].append(Res_use(
                        op=op,
                        succ=succ if succ else None,
                        time=res.time,
                        start=start[op.idx],
                        end=start[succ.idx] + res.time if succ else None,
                    ))

        collisions = []
        for res, uses in res_uses.items():
            for i, u1 in enumerate(uses):
                for u2 in uses[i+1:]:
                    if u1.start < u2.end and u2.start < u1.end:
                        collisions.append((u1, u2))


        return collisions

    def make_res_model(self, paths: List[List[Op]], 
        collisions: List[Tuple[Res_use, Res_use]]):

        omega = self.inst.total_dur

        m = Model()

        start = {}
        res = {}
        
        for path in paths:
            for op in path:
                start[op.idx] = m.addVar(name=f'start{op.idx}', 
                    vtype='C', lb=op.start_lb, ub=op.start_ub)
                
        for path in paths:
            for op, succ in zip(path[:-1], path[1:]):
                m.addCons(name=f'dur{op.idx},{succ.idx}', cons=start[op.idx] + op.dur <= start[succ.idx])

        for u1, u2 in collisions:
            # u1 -> u2 : succ1 + time1 < op2
            # u2 -> u1 : succ2 + time2 < op1

            if u2.succ is None:
                m.addCons(start[u1.succ.idx] + u1.time <= start[u2.op.idx])
            elif u2.succ is None:
                m.addCons(start[u2.succ.idx] + u2.time <= start[u1.op.idx])
            else:
                res[(u1.succ.idx, u2.op.idx)] = m.addVar(name=f'order{u1.succ.idx},{u2.op.idx}', vtype='B')
                res[(u2.succ.idx, u1.op.idx)] = m.addVar(name=f'order{u2.succ.idx},{u1.op.idx}', vtype='B')

                m.addCons(name=f'order{u1.succ.idx},{u2.op.idx}',
                    cons=start[u1.succ.idx] + u1.time <= start[u2.op.idx] + 
                        omega*(1 - res[(u1.succ.idx, u2.op.idx)]))
                
                m.addCons(name=f'order{u1.succ.idx},{u2.op.idx}',
                    cons=start[u2.succ.idx] + u2.time <= start[u1.op.idx] + 
                        omega*(1 - res[(u2.succ.idx, u1.op.idx)]))

                # m.addCons(res[(u1.succ.idx, u2.op.idx)] + res[(u1.succ.idx, u2.op.idx)] == 1)
        
        self.make_model_obj(m, paths, start)

        return m, (res, start)

    def solve_res_model(self, used, start):
        paths = self.get_paths(used)
        collisions = self.get_collisions(paths, start)

        m, (res, start) = self.make_res_model(paths, collisions)
        
        m.writeProblem()
        m.optimize()
        
        # res_val = { op.idx: m.getVal(res[op.idx]) for op in chain(*paths)}
        res_val = {}
        start_val = { op.idx: m.getVal(start[op.idx]) for op in chain(*paths)}
        

        return res_val, start_val
        

if __name__ == '__main__':
    inst = Instance(sys.argv[1] if len(sys.argv) > 1 else DEFAUL_DATA)
    sol = Solver(inst)
    sol.get_res_overlap()
    used, start = sol.solve_forks_model()
    res, start = sol.solve_res_model(used, start)

    # for op in sol.get_start_ops():
    #     print(op)

    
    # for op in sol.get_end_ops():
    #     print(op)