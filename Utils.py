# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:08:03 2022

@author: erfan
"""
import xmltodict
from itertools import combinations
import numpy as np

def load_problem(file):
    # load file
    with open(file, 'r') as f:
        data_dict = xmltodict.parse(f.read())
        f.close()
    n_teams = len(data_dict['Instance']['Resources']['Teams']['team'])
    n_slots = (n_teams - 1 ) * 2
    gameMode = data_dict['Instance']['Structure']['Format']['gameMode']
    #Capacity constraints
    ca = data_dict['Instance']['Constraints']['CapacityConstraints']
    ca_hard, ca_soft = [[] for i in range(4)],[[] for i in range(4)]
    cas = ['CA1', 'CA2', 'CA3', 'CA4']
    if ca:
        for i, c in enumerate(cas):
            if c not in ca.keys():
                continue
            elif type(ca[c]) == list:
                cc = ca[c]
            else:
                cc = [ca[c]]
            for num in cc:
                const = {'max': int(num['@max']),
                         'min': int(num['@min']),
                         'penalty': int(num['@penalty'])}
                if '@teams' in num.keys() :
                    const['teams'] = [int(x) for x in num['@teams'].split(';')]
                else:
                    const['teams1'] = [int(x) for x in num['@teams1'].split(';')]
                    const['teams2'] = [int(x) for x in num['@teams2'].split(';')]
                if '@mode' in num.keys() :
                    const['mode']= num['@mode']
                else:
                    const['mode1']= num['@mode1']
                    const['mode2']= num['@mode2']
                if '@slots' in num.keys() :
                    const['slots'] = [int(x) for x in num['@slots'].split(';')]
                if '@intp' in num.keys() :
                    const['intp'] = int(num['@intp'])
                if num['@type'] == 'HARD':
                    ca_hard[i].append(const)
                else:
                    ca_soft[i].append(const)
    
    #Game Constraints
    ga = data_dict['Instance']['Constraints']['GameConstraints']
    ga_hard, ga_soft = [[] for i in range(1)],[[] for i in range(1)]
    gas = ['GA1']
    if ga:
        for i, g in enumerate(gas):
            if g not in ga.keys():
                continue
            elif type(ga[g]) == list:
                ga1 = ga[g]
            else:
                ga1 = [ga[g]]
            for num in ga1:
                const = {'meetings': [[int(y),int(z)] for y,z in (x.split(',') for x in num['@meetings'].split(';') if x)],
                         'slots': [int(x) for x in num['@slots'].split(';')],
                         'max': int(num['@max']),
                         'min': int(num['@min']),
                         'penalty': int(num['@penalty'])}
                if num['@type'] == 'HARD':
                    ga_hard[i].append(const)
                else:
                    ga_soft[i].append(const)
    #Break Constraints
    ba = data_dict['Instance']['Constraints']['BreakConstraints']
    ba_hard, ba_soft = [[] for i in range(2)],[[] for i in range(2)]
    bas = ['BR1', 'BR2']
    if ba:
        for i, b in enumerate(bas):
            if b not in ba.keys():
                continue
            elif type(ba[b]) == list:
                bc = ba[b]
            else:
                bc = [ba[b]]
            for num in bc:
                const = {'teams': [int(x) for x in num['@teams'].split(';')],
                         'slots': [int(x) for x in num['@slots'].split(';')],
                         'intp': int(num['@intp']),
                         'mode2': num['@mode2'],
                         'penalty': int(num['@penalty'])}
                if '@mode1' in num.keys():
                    const['mode1'] = num['@mode1']
                elif '@homeMode' in num.keys():
                    const['homeMode'] = num['@homeMode']
                if num['@type'] == 'HARD':
                    ba_hard[i].append(const)
                else:
                    ba_soft[i].append(const)
    #Fairness Constraints
    fa = data_dict['Instance']['Constraints']['FairnessConstraints']
    fa_hard, fa_soft = [[] for i in range(1)],[[] for i in range(1)]
    fas = ['FA2']
    if fa:
        
        for i, f in enumerate(fas):
            if f not in fa.keys():
                continue
            elif type(fa[f]) == list:
                fc = fa[f]
            else:
                fc = [fa[f]]
            for num in fc:
                const = {'teams': [int(x) for x in num['@teams'].split(';')],
                         'slots': [int(x) for x in num['@slots'].split(';')],
                         'intp': int(num['@intp']),
                         'mode': num['@mode'],
                         'penalty': int(num['@penalty'])}
                if num['@type'] == 'HARD':
                    fa_hard[i].append(const)
                else:
                    fa_soft[i].append(const)
    #Separation Constraints
    sa = data_dict['Instance']['Constraints']['SeparationConstraints'] 
    sa_hard, sa_soft = [[] for i in range(1)],[[] for i in range(1)]
    sas = ['SE1']
    if sa:
        for i, s in enumerate(sas):
            if s not in sa.keys():
                continue
            elif type(sa[s]) == list:
                sc = sa[s]
            else:
                sc = [sa[s]]
            for num in sc:
                const = {'teams': list(combinations([int(x) for x in num['@teams'].split(';')],2)),
                         'min': int(num['@min']),
                         'mode1': num['@mode1'],
                         'penalty': int(num['@penalty'])}
                if num['@type'] == 'HARD':
                    sa_hard[i].append(const)
                else:
                    sa_soft[i].append(const)
    output = {
        'feas_constr':{'ca': ca_hard,
                      'ga': ga_hard,
                      'ba': ba_hard,
                      'fa': fa_hard,
                      'sa': sa_hard},
        'obj_constr':{'ca': ca_soft,
                      'ga': ga_soft,
                      'ba': ba_soft,
                      'fa': fa_soft,
                      'sa': sa_soft},
        'n_teams': n_teams,
        'n_slots': n_slots,
        'gameMode': gameMode
    }
    return output

def cost_function(Solution, problem):
    ca, ga, ba, fa, sa = problem['obj_constr'].values()
    obj = 0
    for i, cc in enumerate(ca):
        if len(cc) == 0:
            continue
        if i == 0: #CA1 constraints
            for c in cc:
                for team in c['teams']:
                    p = 0
                    if c['mode'] == 'H':
                        p += np.sum(np.transpose(Solution[team:team+1,:]) == c['slots'])
                    else:
                        p += np.sum(Solution[:,team:team+1] == c['slots'])
                    if p > c['max']:
                        obj += (p - c['max'])*c['penalty']
        elif i == 1:#CA2 constraints
            for c in cc:
                if c['mode1'] == 'HA':
                    for team1 in c['teams1']:
                        p = 0
                        for team2 in c['teams2']:
                            p += np.sum(Solution[team1, team2]== c['slots'])
                            p += np.sum(Solution[team2, team1]== c['slots'])
                        if p > c['max']:
                            obj += (p - c['max'])*c['penalty']
                elif c['mode1'] == 'H':
                    for team1 in c['teams1']:
                        p = 0
                        for team2 in c['teams2']:
                            p += np.sum(Solution[team1, team2]== c['slots'])
                        if p > c['max']:
                            obj += (p - c['max'])*c['penalty']
                else:
                    for team1 in c['teams1']:
                        p = 0
                        for team2 in c['teams2']:
                            p += np.sum(Solution[team2, team1]== c['slots'])
                        if p > c['max']:
                            obj += (p - c['max'])*c['penalty']
        elif i == 2:#CA3 constraints
            for c in cc:
                if c['mode1'] == 'HA':
                    for team in c['teams1']:
                        slots_H = Solution[team, c['teams2']]
                        slots_A = Solution[c['teams2'], team]
                        slots = np.concatenate([slots_A, slots_H])
                        for s in range(c['intp'],problem['n_slots'] + 1):
                            p = np.sum(np.logical_and((slots < s) ,(slots >= s - c['intp'])))
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
                elif c['mode1'] == 'H':
                    for team in c['teams1']:
                        slots_H = Solution[team, c['teams2']]
                        for s in range(c['intp'],problem['n_slots'] + 1):
                            p = np.sum(np.logical_and((slots_H < s) ,(slots_H >= s - c['intp'])))
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
                else:
                    for team in c['teams1']:
                        slots_A = Solution[c['teams2'], team]
                        for s in range(c['intp'],problem['n_slots'] + 1):
                            p = np.sum(np.logical_and((slots_A < s) ,(slots_A >= s - c['intp'])))
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
        else:#CA4 constraints
            for c in cc:
                if c['mode1'] == 'HA':
                    if c['mode2'] == 'GLOBAL':
                        p = 0
                        slots_H = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        slots_A = Solution[np.ix_(c['teams2'], c['teams1'])].flatten()
                        slots = np.concatenate([slots_A, slots_H])
                        for slot in c['slots']:
                            p += np.sum( slots == slot)
                        if p > c['max']:
                            obj += (p - c['max'])*c['penalty']
                    else:
                        slots = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        for slot in c['slots']:
                            p = np.sum( slots == slot)
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
                elif c['mode1'] == 'H':
                    if c['mode2'] == 'GLOBAL':
                        p = 0
                        slots = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        for slot in c['slots']:
                            p += np.sum( slots == slot)
                        if p > c['max']:
                            obj += (p - c['max'])*c['penalty']
                    else:
                        slots = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        for slot in c['slots']:
                            p = np.sum( slots == slot)
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
                else:
                    if c['mode2'] == 'GLOBAL':
                        p = 0
                        slots = Solution[np.ix_(c['teams2'], c['teams1'])].flatten()
                        for slot in c['slots']:
                            p += np.sum( slots == slot)
                        if p > c['max']:
                            obj += (p - c['max'])*c['penalty']
                    else:
                        slots = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        for slot in c['slots']:
                            p = np.sum( slots == slot)
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
    for i, gc in enumerate(ga):
        if len(gc) == 0:
            continue
        for c in gc:#GA1 constraints
            p = 0
            for meeting in c['meetings']:
                p += np.sum(Solution[tuple(meeting)]== c['slots'])
            if p < c['min']:
                obj += (c['min'] - p)*c['penalty']
            elif p > c['max']:
                obj += (p - c['max'])*c['penalty']
    for i, bc in enumerate(ba):
        if len(bc) == 0:
            continue
        if i == 0:#BR1 constraints
            for c in bc:
                for team in c['teams']:
                    p = 0
                    for slot in c['slots']:
                        if slot == 0:
                            continue
                        cur = (Solution[team, :] == slot).any()
                        prev = (Solution[team, :] == slot -1).any()
                        if c['mode2'] == 'HA':
                            p += (cur == prev)
                        elif c['mode2'] == 'H':
                            p += (cur == prev and cur == True)
                        else:
                            p += (cur == prev and cur == False)
                    if p > c['intp']:
                        obj += (p - c['intp'])*c['penalty']
        elif i == 1:#BR2 constraints
            for c in bc:
                p = 0
                for team in c['teams']:
                    for slot in c['slots']:
                        if slot == 0:
                            continue
                        cur = (Solution[team, :] == slot).any()
                        prev = (Solution[team, :] == slot -1).any()
                        p += (cur == prev)
                if p > c['intp']:
                    obj += (p - c['intp'])*c['penalty']
    for i, fc in enumerate(fa):
        if len(fc) == 0:
            continue
        for c in fc:#FA1 constraints
            diff = np.zeros([len(c['teams']),len(c['teams'])], dtype = int)
            for s in c['slots']:
                p = 0
                home_count = np.zeros_like(c['teams'])
                for team in c['teams']:
                    home_count[team] = np.sum(Solution[team,:] <= s) - 1 # excluding the column = team
                for i, j in combinations(c['teams'], 2):
                    diff[i,j] = max(abs(home_count[i] - home_count[j]),diff[i,j])
                    # if diff[i,j] > c['intp']:
                    #     p += (diff - c['intp'])
            diff -= c['intp']
            diff[diff < 0] = 0
            obj += np.sum(diff)*c['penalty']
    for i, sc in enumerate(sa):
        if len(sc) == 0:
            continue
        for c in sc:#SE1 constraints
            for team1, team2 in c['teams']:
                first = Solution[team1, team2]
                second = Solution[team2, team1]
                diff = abs(second - first) - 1
                if diff < c['min']:
                    obj += (c['min'] - diff)* c['penalty']
    return obj
                
def feasibility_check(Solution, problem):
    ca, ga, ba, fa, sa = problem['feas_constr'].values()
    status, feasibility = 'Feasible', True
    for i, cc in enumerate(ca):
        if len(cc) == 0:
            continue
        if i == 0: #CA1 constraints
            for c in cc:
                if feasibility:
                    for team in c['teams']:
                        p = 0
                        if c['mode'] == 'H':
                            p += np.sum(np.transpose(Solution[team:team+1,:]) == c['slots'])
                        else:
                            p += np.sum(Solution[:,team:team+1] == c['slots'])
                        if p > c['max']:
                            break
                feasibility = False
                status = 'Team {} has {} more {} games than max= {} during time slots {}'.format(team, p -c['max'], c['mode'], c['max'], c['slots'])
        elif i == 1:#CA2 constraints
            for c in cc:
                if feasibility:
                    if c['mode1'] == 'HA':
                        for team1 in c['teams1']:
                            p = 0
                            for team2 in c['teams2']:
                                p += np.sum(Solution[team1, team2]== c['slots'])
                                p += np.sum(Solution[team2, team1]== c['slots'])
                            if p > c['max']:
                               break 
                    elif c['mode1'] == 'H':
                        for team1 in c['teams1']:
                            p = 0
                            for team2 in c['teams2']:
                                p += np.sum(Solution[team1, team2]== c['slots'])
                            if p > c['max']:
                                break
                    else:
                        for team1 in c['teams1']:
                            p = 0
                            for team2 in c['teams2']:
                                p += np.sum(Solution[team2, team1]== c['slots'])
                            if p > c['max']:
                                break
                    feasibility = False
                    status = 'Team {} has {} more {} games than max= {} during time slots {} agains teams: {}'.format(team1, p -c['max'], c['mode'], c['max'], c['slots'], c['teams2'])
        elif i == 2:#CA3 constraints
            for c in cc:
                if c['mode1'] == 'HA':
                    for team in c['teams1']:
                        slots_H = Solution[team, c['teams2']]
                        slots_A = Solution[c['teams2'], team]
                        slots = np.concatenate([slots_A, slots_H])
                        for s in range(c['intp'],problem['n_slots'] + 1):
                            p = np.sum(np.logical_and((slots < s) ,(slots >= s - c['intp'])))
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
                elif c['mode1'] == 'H':
                    for team in c['teams1']:
                        slots_H = Solution[team, c['teams2']]
                        for s in range(c['intp'],problem['n_slots'] + 1):
                            p = np.sum(np.logical_and((slots_H < s) ,(slots_H >= s - c['intp'])))
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
                else:
                    for team in c['teams1']:
                        slots_A = Solution[c['teams2'], team]
                        for s in range(c['intp'],problem['n_slots'] + 1):
                            p = np.sum(np.logical_and((slots_A < s) ,(slots_A >= s - c['intp'])))
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
        else:#CA4 constraints
            for c in cc:
                if c['mode1'] == 'HA':
                    if c['mode2'] == 'GLOBAL':
                        p = 0
                        slots_H = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        slots_A = Solution[np.ix_(c['teams2'], c['teams1'])].flatten()
                        slots = np.concatenate([slots_A, slots_H])
                        for slot in c['slots']:
                            p += np.sum( slots == slot)
                        if p > c['max']:
                            obj += (p - c['max'])*c['penalty']
                    else:
                        slots = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        for slot in c['slots']:
                            p = np.sum( slots == slot)
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
                elif c['mode1'] == 'H':
                    if c['mode2'] == 'GLOBAL':
                        p = 0
                        slots = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        for slot in c['slots']:
                            p += np.sum( slots == slot)
                        if p > c['max']:
                            obj += (p - c['max'])*c['penalty']
                    else:
                        slots = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        for slot in c['slots']:
                            p = np.sum( slots == slot)
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
                else:
                    if c['mode2'] == 'GLOBAL':
                        p = 0
                        slots = Solution[np.ix_(c['teams2'], c['teams1'])].flatten()
                        for slot in c['slots']:
                            p += np.sum( slots == slot)
                        if p > c['max']:
                            obj += (p - c['max'])*c['penalty']
                    else:
                        slots = Solution[np.ix_(c['teams1'], c['teams2'])].flatten()
                        for slot in c['slots']:
                            p = np.sum( slots == slot)
                            if p > c['max']:
                                obj += (p - c['max'])*c['penalty']
    for i, gc in enumerate(ga):
        if len(gc) == 0:
            continue
        for c in gc:#GA1 constraints
            p = 0
            for meeting in c['meetings']:
                p += np.sum(Solution[tuple(meeting)]== c['slots'])
            if p < c['min']:
                obj += (c['min'] - p)*c['penalty']
            elif p > c['max']:
                obj += (p - c['max'])*c['penalty']
    for i, bc in enumerate(ba):
        if len(bc) == 0:
            continue
        if i == 0:#BR1 constraints
            for c in bc:
                for team in c['teams']:
                    p = 0
                    for slot in c['slots']:
                        if slot == 0:
                            continue
                        cur = (Solution[team, :] == slot).any()
                        prev = (Solution[team, :] == slot -1).any()
                        if c['mode2'] == 'HA':
                            p += (cur == prev)
                        elif c['mode2'] == 'H':
                            p += (cur == prev and cur == True)
                        else:
                            p += (cur == prev and cur == False)
                    if p > c['intp']:
                        obj += (p - c['intp'])*c['penalty']
        elif i == 1:#BR2 constraints
            for c in bc:
                p = 0
                for team in c['teams']:
                    for slot in c['slots']:
                        if slot == 0:
                            continue
                        cur = (Solution[team, :] == slot).any()
                        prev = (Solution[team, :] == slot -1).any()
                        p += (cur == prev)
                if p > c['intp']:
                    obj += (p - c['intp'])*c['penalty']
    for i, fc in enumerate(fa):
        if len(fc) == 0:
            continue
        for c in fc:#FA1 constraints
            diff = np.zeros([len(c['teams']),len(c['teams'])], dtype = int)
            for s in c['slots']:
                p = 0
                home_count = np.zeros_like(c['teams'])
                for team in c['teams']:
                    home_count[team] = np.sum(Solution[team,:] <= s) - 1 # excluding the column = team
                for i, j in combinations(c['teams'], 2):
                    diff[i,j] = max(abs(home_count[i] - home_count[j]),diff[i,j])
                    # if diff[i,j] > c['intp']:
                    #     p += (diff - c['intp'])
            diff -= c['intp']
            diff[diff < 0] = 0
            obj += np.sum(diff)*c['penalty']
    for i, sc in enumerate(sa):
        if len(sc) == 0:
            continue
        for c in sc:#SE1 constraints
            for team1, team2 in c['teams']:
                first = Solution[team1, team2]
                second = Solution[team2, team1]
                diff = abs(second - first) - 1
                if diff < c['min']:
                    obj += (c['min'] - diff)* c['penalty']
    return obj
                
def load_solution(file, sol):
    # load file
    with open(file, 'r') as f:
        data_dict = xmltodict.parse(f.read())
        f.close()
    Games = data_dict['Solution']['Games']['ScheduledMatch']
    objective_value = int(data_dict['Solution']['MetaData']['ObjectiveValue']['@objective'])
    for game in Games:
        sol[int(game['@home']), int(game['@away'])] = int(game['@slot'])
    return sol, objective_value
problem = load_problem('Instances//EarlyInstances_V3//ITC2021_Early_3.xml')
sol = np.ones((problem['n_teams'],problem['n_teams']), dtype = int)*(-1)
solution, objective_value = load_solution('..//Appendix_Files//Final_Solutions//Early_instances//E3.xml', sol)
obj = cost_function(solution, problem)
