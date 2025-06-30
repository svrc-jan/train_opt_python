
def model_and(m, vars):
    resvar = m.addVar(name='and', vtype='B')
    m.addConsAnd(vars=vars, resvar=resvar)

    return resvar

def is_bin(x: float, eps=1e-6):
    return abs(x) < eps or abs(x - 1) < eps

def round_dict(d):
    return { k: round_to_int(v) for k, v in d.items() }

def round_to_int(x: float):
    return int(round(x))