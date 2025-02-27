
def model_and(m, vars):
    resvar = m.addVar(name='and', vtype='B')
    m.addConsAnd(vars=vars, resvar=resvar)

    return resvar