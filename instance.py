import sys
import json

from dataclasses import dataclass, field
from typing import List, Dict

DEFAULT_DATA = 'data/nor1_critical_0.json'
MAX_DUR = 100000

@dataclass
class Res:
	idx: int = -1
	time: int = 0


@dataclass
class Obj:
	threshold: int = 0
	coeff: int = 0
	increment: int = 0


@dataclass
class Op:
	idx: int = -1
	train: int = -1

	dur: int = 0
	start_lb: int = 0
	start_ub: int = MAX_DUR

	succ: List[int] = field(default_factory=list)
	prev: List[int] = field(default_factory=list)

	res: List[Res] = field(default_factory=list)

	obj: Obj|None = None


@dataclass
class Train:
	idx: int = -1

	ops: List[Op] = field(default_factory=list)

class Instance:
	trains: List[Train]
	points_valid: bool

	__res_name_idx: Dict[str, int]

	def __init__(self, jsn_file: str):
		self.parse_json_file(jsn_file)
		self.make_prev_ops()


	def parse_json_file(self, jsn_file: str):
		self.trains = []
		self.__res_name_idx = {}

		with open(jsn_file, 'r') as fd:
			jsn = json.load(fd)
		
		for jsn_train in jsn['trains']:
			self.parse_json_train(jsn_train)

		for jsn_obj in jsn['objective']:
			self.parse_json_obj(jsn_obj)


	def parse_json_train(self, jsn_train: dict):
		train = Train(idx=len(self.trains))
		
		for jsn_op in jsn_train:
			self.parse_json_op(jsn_op, train)

		self.trains.append(train)


	def parse_json_op(self, jsn_op: dict, train: Train):
		op = Op(idx=len(train.ops), train=train.idx)

		op.dur = jsn_op['min_duration']
		op.start_lb = jsn_op.get('start_lb', 0)
		op.start_ub = jsn_op.get('start_ub', MAX_DUR)

		op.succ = jsn_op['successors']

		for jsn_res in jsn_op.get('resources', []):
			res = Res(
				idx=self.get_res_idx(jsn_res['resource']),
				time=jsn_res.get('release_time', 0)
			)
			
			op.res.append(res)

		train.ops.append(op)


	def parse_json_obj(self, jsn_obj):
		if jsn_obj['type'] != 'op_delay':
			return
		
		obj = Obj(
			threshold	=jsn_obj.get('threshold', 0),
			coeff		=jsn_obj.get('coeff', 0),
			increment	=jsn_obj.get('increment', 0)
		)

		self.trains[jsn_obj['train']].obs[jsn_obj['operation']].obj = obj


	def make_prev_ops(self):
		for train in self.trains:
			for op in train.ops:
				for succ in op.succ:
					train.ops[succ].prev.append(op.idx)


	def get_res_idx(self, name):
		idx = self.__res_name_idx.get(name, -1)
		
		if idx == -1:
			idx = len(self.__res_name_idx)
			self.__res_name_idx[name] = idx

		return idx

if __name__ == '__main__':
	data = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
	print(data)
	inst = Instance(data)
	pass
