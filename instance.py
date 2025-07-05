#!.venv/bin/python3

import json
import sys
import itertools

from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import List, Tuple, Dict
from functools import cached_property, cache

DEFAULT_DATA = 'data/testing/headway1.json'

@dataclass
class Res:
	idx: int
	time: int

@dataclass
class Obj:
	threshold: int	= 0
	coeff: int		= 0
	increment: int	= 0


@dataclass
class Point:
	train_idx: int = -1
	point_idx: int = -1

	start_lb: int 		= 0
	start_ub: int|None 	= None

	obj = None

	ops_in: List['Op']	= field(default_factory=list)
	ops_out: List['Op'] = field(default_factory=list)

	res_lock: Dict[int, List['Op']] 	= field(default_factory=dict)
	res_unlock: Dict[int, List['Op']] 	= field(default_factory=dict)


	@cached_property
	def idx(self) -> int:
		return (self.train_idx, self.point_idx)

	def __hash__(self) -> int:
		return hash(self.idx)

	def __repr__(self) -> str:
		return f'Point({self.train_idx}, {self.point_idx})'


@dataclass
class Op:
	train_idx: int	= -1
	op_idx: int		= -1

	dur: int		= 0
	start_lb: int	= 0
	start_ub: int|None	= None

	succ: List['Op']	= field(default_factory=list)
	prev: List['Op']	= field(default_factory=list)
	res: List[Res]		= field(default_factory=list)

	obj: Obj|None		= None

	p_start: Point|None = None
	p_end: Point|None   = None


	@cached_property
	def n_succ(self) -> int:
		return len(self.succ)
	
	@cached_property
	def n_prev(self) -> int:
		return len(self.prev)

	@cached_property
	def n_res(self) -> int:
		return len(self.res)

	@cached_property
	def idx(self) -> int:
		return (self.train_idx, self.op_idx)

	def __hash__(self) -> int:
		return hash(self.idx)

	def __repr__(self) -> str:
		return f'Op({self.train_idx}, {self.op_idx})'

	def __lt__(self, other):
		return self.idx < other.idx

@dataclass
class Res_use:
	op: Op
	time: int

class Instance:
	ops: List[List[Op]]
	points: List[List[Point]]

	res_idx: Dict[str, int]
	res_uses: Dict[int, List[Res_use]]
	
	max_dur: int
	max_train_dur: List[int]

	n_train_ops: List[int]


	def __init__(self, json_file: str):
		self.parse_json(json_file)
		self.generate_points()

	def parse_json(self, json_file: str) -> None:
		with open(json_file, 'r') as fd:
			jsn = json.load(fd)

		self.ops = []
		self.res_idx = {}
		self.res_uses = {}
		self.total_dur = 0

		for train_idx, train_jsn in enumerate(jsn['trains']):
			train_ops = []

			succ_l = []

			for op_idx, op_jsn in enumerate(train_jsn):
				op = Op(
					train_idx	=train_idx,
					op_idx		=op_idx,
					dur			=op_jsn['min_duration'],
					start_lb	=op_jsn.get('start_lb', 0),
					start_ub	=op_jsn.get('start_ub', None)
				)

				op.start_obj = op.start_ub
				self.total_dur += op.dur

				succ_l.append(op_jsn['successors'])

				for res_jsn in op_jsn.get('resources', []):
					res_name = res_jsn['resource']
					res_time = res_jsn.get('release_time', 0)
					
					if not res_name in self.res_idx:
						res_idx = len(self.res_idx)
						self.res_idx[res_name] = res_idx
					else:
						res_idx = self.res_idx[res_name]

					op.res.append(Res(res_idx, res_time))
					
					if not res_idx in self.res_uses:
						self.res_uses[res_idx] = []

					self.res_uses[res_idx].append(Res_use(op=op, time=res_time))

				train_ops.append(op)

			for op, succ_idx in zip(train_ops, succ_l):
				for succ in [train_ops[i] for i in succ_idx]:
					op.succ.append(succ)
					succ.prev.append(op)

			self.ops.append(train_ops)

		self.n_train_ops = [len(train_ops) for train_ops in self.ops]

		for obj_jsn in jsn['objective']:
			if obj_jsn['type'] != 'op_delay':
				continue

			self.ops[obj_jsn['train']][obj_jsn['operation']].obj = Obj(
				threshold	=obj_jsn.get('threshold', 0),
				coeff		=obj_jsn.get('coeff', 0),
				increment	=obj_jsn.get('increment', 0)
			)

		for op in self.all_ops:
			if op.obj:
				assert((op.obj.coeff == 0) != (op.obj.increment == 0))


	def generate_points(self):
		self.points = []
		for t_idx, t_ops in enumerate(self.ops):
			t_points = [Point(
				train_idx=t_idx,
				point_idx=0,
				ops_out=[t_ops[0]],
				ops_in=[]
			)]

			for op in t_ops:
				p = next((x for x in t_points if x.ops_out == op.succ), None)
				if p:
					op.p_end = p
					p.ops_in.append(op)
				else:
					t_points.append(Point(
						train_idx=t_idx,
						point_idx=len(t_points),
						ops_out=op.succ,
						ops_in=[op]
					))

					op.p_end = t_points[-1]
			
			for p in t_points:
				for op in p.ops_out:
					assert(op.p_start is None)
					op.p_start = p


			for op in t_ops:
				for res in op.res:
					op.p_start.res_lock.setdefault(res.idx, []).append(op)
					op.p_end.res_unlock.setdefault(res.idx, []).append(op)
				

			for p in t_points:
				p.ops_in.sort()
				p.ops_out.sort()

				for lock_ops in p.res_lock.values():
					lock_ops.sort()

				for unlock_ops in p.res_unlock.values():
					unlock_ops.sort()

				for res_idx, lock_ops in list(p.res_lock.items()):
					if res_idx in p.res_unlock:
						unlock_ops = p.res_lock[res_idx]
						if lock_ops == p.ops_in and unlock_ops == p.ops_out:
							del p.res_lock[res_idx]
							del p.res_unlock[res_idx]


			self.points.append(t_points)				
	

			for p in itertools.chain(*self.points):
				if p.ops_out:
					assert(all(x.start_ub == p.ops_out[0].start_ub for x in p.ops_out))

					assert(all(x.start_lb == p.ops_out[0].start_lb for x in p.ops_out))
				
					assert(all(x.obj == p.ops_out[0].obj for x in p.ops_out))
	
				for op in p.ops_out:
					assert(p.point_idx < op.p_end.point_idx)


	@cached_property
	def n_trains(self) -> int:
		return len(self.ops)
	
	@cached_property
	def n_ops(self) -> int:
		return sum(self.n_train_ops)
	
	@cached_property
	def max_dur(self) -> int:
		return sum(self.max_train_dur)

	@cached_property
	def n_res(self) -> int:
		return len(self.res_idx)
	
	@cached_property
	def avg_dur(self) -> float:
		return self.total_dur/sum(self.n_ops)
	
	@property
	def all_ops(self):
		return itertools.chain(*self.ops)

	@property
	def all_points(self):
		return itertools.chain(*self.points)

	@staticmethod
	@cache
	def get_res_inter(op1, op2):
		r1 = [x.idx for x in op1.res]
		r2 = [x.idx for x in op2.res]

		return [x for x in r1 if x in r2]
	
	@staticmethod
	@cache
	def get_res_diff(op1, op2):
		if not op1:
			return []
		
		r1 = [x.idx for x in op1.res]
		if not op2:
			return r1

		r2 = [x.idx for x in op2.res]

		return [x for x in r1 if x not in r2]
		
	
	def verify_order(self, order: List[Op]):
		prev_op = [None]*self.n_trains
		res_vec = [1]*self.n_res

		for op in order:
			if op.n_prev > 0:
				assert(not prev_op[op.train_idx] is None)
				assert(prev_op[op.train_idx] in op.prev)

				for res in prev_op[op.train_idx].res:
					assert(res_vec[res.idx] == 0)
					res_vec[res.idx] += 1
			
			else:
				assert(prev_op[op.train_idx] is None)
			
			for res in op.res:
				if res_vec[res.idx] != 1:
					print(f'locked resource {res.idx} for {op}')
					return False
				res_vec[res.idx] -= 1
			
			prev_op[op.train_idx] = op


if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)

	for t_idx, t_points in enumerate(inst.points):
		print(f'train {t_idx},  points: {len(t_points)}, ops: {inst.n_train_ops[t_idx]}')
		for p in t_points:
			print(f'  {p}, in: {p.ops_in}, out: {p.ops_out}, lock: {p.res_lock}, unlock: {p.res_unlock}')

	# for t_idx in range(inst.n_trains):
	# 	print('tr')

	
		