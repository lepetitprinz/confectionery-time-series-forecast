import gurobipy as gp
from gurobipy import GRB
from itertools import product, combinations

try:
    # create a new model
    m = gp.Model('job_schedule')

    MACH = [1,2,3]
    PAPER = [1,2,3]
    # RANK = [1,2,3]
    REQ = {(1,1):45, (2,1) : 0, (3,1):10, (1,2):20, (2,2):10, (3,2):34, (1,3):12, (2,3):17, (3,3):28}  #(MACH, PAPER) : TIME
    SEQ = {1: [1,3,2], 2: [2,1,3], 3:[3,1,2]} # PAPER : [SEQ] paper1

    # create variables
    # RNK = {}
    # for i in RANK:
    #     RNK[i] = m.addVars(product(MACH, PAPER), vtype=GRB.BINARY, name='RANK')
    START = m.addVars(product(MACH, PAPER), name='START')
    MSEQ = {}
    for i in PAPER:
        MSEQ[i] = m.addVars(combinations(PAPER,2), vtype=GRB.BINARY, name='MRANK' + str(i))
    SUM_REQ = sum(REQ.values())
    FIN = m.addVar(vtype=GRB.CONTINUOUS, name='finish')
    # set objective function
    obj = 0
    obj = FIN
    m.setObjective(obj, GRB.MINIMIZE)


    # Add constraint
    for p in PAPER:
        for i in range(len(SEQ[p]) - 1):
            m.addConstr(START[SEQ[p][i], p] + REQ[SEQ[p][i], p] <= START[SEQ[p][i + 1], p], 'constr1'+ str(p) + str(i))

    for mc in MACH:
        for f,s in combinations(PAPER, 2):
            m.addConstr(START[mc,f] + REQ[mc,f] <= START[mc, s] + SUM_REQ * (1 - MSEQ[mc][f,s]), 'constr2' + str(mc) + str(f) + str(s))
            m.addConstr(START[mc,s] + REQ[mc,s] <= START[mc, f] + SUM_REQ * MSEQ[mc][f,s], 'constr3' + str(mc) + str(f) + str(s))
    for p in PAPER:
        m.addConstr(FIN >= START[SEQ[p][2],p] + REQ[SEQ[p][2],p], 'constr_fin' + str(p))



    # Optimize Model
    m.optimize()
    m.write("OPT_6.lp")  # 진행 기록가능
    print('목적식 결과 : %g' % m.objval)
except gp.GurobiError as e:
    print('error : ' + e)