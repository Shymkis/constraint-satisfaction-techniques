# Joe Shymanski
# Project 2: Constraint Satisfaction Techniques (Backtracking)
import itertools
import json
import math
import matplotlib.pyplot as plt
import os
import pprint
import random
import time

############################## RELATIONS ###############################

def all_diff(vals):
    """Relation function to ensure all provided values are unique

    Args:
        vals (list): List of variable values.

    Returns:
        bool: True if all values in list are unique, False otherwise.
    """
    return len(set(vals)) == len(vals)

######################### COLORING CONSTRAINTS #########################

def coloring_constraints(points, edges):
    """Create all the necessary constraints for a graph coloring problem
    (GCP)

    Args:
        points (list): List of points.
        edges (list): List of edges in the form of [point1, point2].

    Returns:
        list: List of dictionaries containing "scope" and "rel" keys.
        Scope contains the edge.
        Rel contains the relation function all_diff
    """
    C = []
    for point in points:
        for edge in edges:
            if point in edge:
                C.append({
                    "scope": edge,
                    "rel": all_diff
                })
    return C

########################## SUDOKU CONSTRAINTS ##########################

def cells_in_row(r):
    """Obtain all sudoku cells in row r

    Args:
        b (int): Row index.

    Returns:
        list: List of cells in row r
    """
    return list(range(9*r, 9*(r + 1)))

def cells_in_col(c):
    """Obtain all sudoku cells in column c

    Args:
        c (int): Column index.

    Returns:
        list: List of cells in column c
    """
    return list(range(c, 81, 9))

def cells_in_box(b):
    """Obtain all sudoku cells in box b

    Args:
        b (int): Box index.

    Returns:
        list: List of cells in box b
    """
    b_r = b // 3
    b_c = b % 3
    return list(itertools.chain(
        range(3*(9*b_r + b_c + 0), 3*(9*b_r + b_c + 1)),
        range(3*(9*b_r + b_c + 3), 3*(9*b_r + b_c + 4)),
        range(3*(9*b_r + b_c + 6), 3*(9*b_r + b_c + 7))
    ))

def sudoku_constraints():
    """Create all the necessary constraints for Sudoku

    Returns:
        list: List of dictionaries containing "scope" and "rel" keys.
        Scope contains a row, column, or box of cells.
        Rel contains the relation function all_diff
    """
    C = []
    for row in range(9):
        cells = cells_in_row(row)
        C.append({
            "scope": cells,
            "rel": all_diff
        })
    for col in range(9):
        cells = cells_in_col(col)
        C.append({
            "scope": cells,
            "rel": all_diff
        })
    for box in range(9):
        cells = cells_in_box(box)
        C.append({
            "scope": cells,
            "rel": all_diff
        })
    return C

############################## CSP FORMAT ##############################

def gen_csp(X, D, C):
    """Generate CSP dictionary

    Args:
        X (list): Variables.
        D (list): Domain.
        C (list): Constraints.

    Returns:
        {
            "variables": X,
            "domains": dictionary of domains for every x in X,
            "constraints": C
        }
    """
    return {
        "variables": X,
        "domains": {x: D.copy() for x in X},
        "constraints": C
    }

############################### HELPERS ################################

runs = {}
times = {}

runs["RC"] = times["RC"] = 0

def relevant_constraints(csp, var):
    """Obtain the constraints containing var

    Args:
        csp (dict): CSP.
        var: Variable.

    Returns:
        list: List of constraints containing var
    """
    global runs, times
    runs["RC"] += 1
    start = time.time()

    ret = [c for c in csp["constraints"] if var in c["scope"]]

    end = time.time()
    times["RC"] += end - start
    return ret

runs["IC"] = times["IC"] = 0

def is_consistent(csp, var, value, assignment):
    """Checks var: value against each constraint given in csp and the
    current assignment

    Args:
        csp (dict): CSP.
        var: Variable.
        value: Value.
        assignment (dict): Assignment.

    Returns:
        bool: True if var: value is consistent, False otherwise.
    """
    global runs, times
    runs["IC"] += 1
    start = time.time()

    constraints = relevant_constraints(csp, var)
    for constraint in constraints:
        scope = constraint["scope"]
        rel = constraint["rel"]
        vals = [assignment[x] for x in scope if x in assignment]
        # Append value since var is not currently in assignment
        vals.append(value)
        if not rel(vals):
            end = time.time()
            times["IC"] += end - start
            return False

    end = time.time()
    times["IC"] += end - start
    return True

runs["GN"] = times["GN"] = 0

def get_neighbors(csp, var, assignment):
    """Obtain all neighboring variables to var not yet in assignment

    Args:
        csp (dict): CSP.
        var: Variable.
        assignment (dict): Assignment.

    Returns:
        set: Set of neighboring variables
    """
    global runs, times
    runs["GN"] += 1
    start = time.time()

    neighbors = set()
    constraints = relevant_constraints(csp, var)
    for constraint in constraints:
        n = [x for x in constraint["scope"] if x != var and x not in assignment]
        neighbors.update(n)

    end = time.time()
    times["GN"] += end - start
    return neighbors

############################ CSP COMPLETION ############################

def is_complete(csp, assignment):
    """Checks if assignment is complete

    Args:
        csp (dict): CSP.
        assignment (dict): Assignment.

    Returns:
        bool: True if assignment is complete, False otherwise.
    """
    return len(assignment) == len(csp["variables"])

########################## VARIABLE SELECTION ##########################

runs["MR"] = times["MR"] = 0

def mrv(csp, unassigned_variables, degree=False):
    """Minimum-remaining-value (MRV)

    Args:
        csp (dict): CSP.
        unassigned_variables (list): Unassigned variables.
        degree (bool, optional): When True, adds degree heuristic
        tiebreaker.
        Defaults to False.

    Returns:
        Optimal variable
    """
    global runs, times
    runs["MR"] += 1
    start = time.time()

    min_remaining = math.inf
    min_var = None
    ties = []
    for var in unassigned_variables:
        remaining = len(csp["domains"][var])
        if remaining < min_remaining:
            min_remaining = remaining
            min_var = var
            if degree:
                ties = [var]
        elif degree and remaining == min_remaining:
            ties.append(var)

    # Degree tiebreaker
    if degree and len(ties) > 1:
        max_branching = -math.inf
        max_var = None
        for var in ties:
            branching = 0
            constraints = relevant_constraints(csp, var)
            for constraint in constraints:
                u = [x for x in constraint["scope"] if x in unassigned_variables]
                # Substract one since var is in u
                branching += len(u) - 1
            if branching > max_branching:
                max_branching = branching
                max_var = var
        min_var = max_var

    end = time.time()
    times["MR"] += end - start
    return min_var

runs["MD"] = times["MD"] = 0

def mrv_degree(csp, unassigned_variables):
    """Minimum-remaining-value (MRV) with degree heuristic tiebreaker

    Args:
        csp (dict): CSP.
        unassigned_variables (list): Unassigned variables.

    Returns:
        Optimal variable
    """
    global runs, times
    runs["MD"] += 1
    start = time.time()

    ret = mrv(csp, unassigned_variables, degree=True)

    end = time.time()
    times["MD"] += end - start
    return ret

runs["SU"] = times["SU"] = 0

def select_unassigned_variable(csp, assignment, heuristic=None):
    """Select unassigned variables

    Args:
        csp (dict): CSP.
        assignment (dict): Assignment.
        heuristic (func, optional): Which heuristic to use in selection.
        Defaults to None, meaning random selection.

    Returns:
        Optimal variable
    """
    global runs, times
    runs["SU"] += 1
    start = time.time()

    selection = None
    unassigned_variables = [x for x in csp["variables"] if x not in assignment]
    if heuristic == None:
        selection = random.choice(unassigned_variables)
    else:
        selection = heuristic(csp, unassigned_variables)

    end = time.time()
    times["SU"] += end - start
    return selection

############################ VALUE ORDERING ############################

runs["OD"] = times["OD"] = 0

def order_domain_values(csp, var, assignment):
    """Order domain values according to least-constraining-value (LCV)
    heuristic

    Args:
        csp (dict): CSP.
        var: Variable.
        assignment (dict): Assignment.

    Returns:
        list: List of values sorted in ascending order according to LCV
    """
    global runs, times
    runs["OD"] += 1
    start = time.time()

    lcv = {}
    values = csp["domains"][var]
    neighbors = get_neighbors(csp, var, assignment)
    for value in values:
        num_conflicts = 0
        for neighbor in neighbors:
            if value in csp["domains"][neighbor]:
                num_conflicts += 1
        lcv[value] = num_conflicts
    ret = dict(sorted(lcv.items(), key=lambda item: item[1]))

    end = time.time()
    times["OD"] += end - start
    return ret

############################## INFERENCES ##############################

runs["FC"] = times["FC"] = 0

def forward_checking(csp, var, assignment):
    """Forward checking

    Args:
        csp (dict): CSP.
        var: Variable.
        assignment (dict): Assignment.

    Returns:
        dict: Updated domains dictionary
    """
    global runs, times
    runs["FC"] += 1
    start = time.time()

    domains = csp["domains"].copy()
    neighbors = get_neighbors(csp, var, assignment)
    for x, domain in domains.items():
        if x in neighbors:
            d = domain.copy()
            for value in domain:
                if not is_consistent(csp, x, value, assignment):
                    d.remove(value)
            domains[x] = d

    end = time.time()
    times["FC"] += end - start
    return domains

runs["AC"] = times["AC"] = 0

def ac3(csp, var, assignment):
    """AC-3

    Args:
        csp (dict): CSP.
        var: Variable.
        assignment (dict): Assignment.

    Returns:
        Updated domains dictionary or failure
    """
    global runs, times
    runs["AC"] += 1
    start = time.time()

    domains = csp["domains"].copy()
    neighbors = get_neighbors(csp, var, assignment)
    while neighbors:
        for x, domain in domains.items():
            if x in neighbors:
                neighbors.remove(x)
                revised = False
                d = domain.copy()
                for value in domain:
                    if not is_consistent(csp, x, value, assignment):
                        d.remove(value)
                        revised = True
                domains[x] = d
                if revised:
                    if len(domains[x]) == 0:
                        end = time.time()
                        times["AC"] += end - start
                        return "failure"
                    neighbors |= get_neighbors(csp, x, assignment)

    end = time.time()
    times["AC"] += end - start
    return domains

runs["IN"] = times["IN"] = 0

def inference(csp, var, assignment, method=forward_checking):
    """Make inferences on each variable's domain using the specified
    method

    Args:
        csp (dict): CSP.
        var: Variable.
        assignment (dict): Assignment.
        method (func, optional): Which method to use in making
        inferences.
        Defaults to forward checking.

    Returns:
        Updated domains dictionary or failure
    """
    global runs, times
    runs["IN"] += 1
    start = time.time()

    ret = method(csp, var, assignment)

    end = time.time()
    times["IN"] += end - start
    return ret

############################# BACKTRACKING #############################

def backtrack(csp, assignment):
    """Recursive backtracking algorithm for constraint satisfaction
    problems

    Args:
        csp (dict): CSP.
        assignment (dict): Assignment.

    Returns:
        Solution assignment or failure
    """
    if is_complete(csp, assignment): return assignment
    var = select_unassigned_variable(csp, assignment, heuristic=mrv)
    for value in order_domain_values(csp, var, assignment):
        if is_consistent(csp, var, value, assignment):
            domains_before_assignment = csp["domains"].copy()
            assignment[var] = value
            csp["domains"][var] = [value]
            inferences = inference(csp, var, assignment, method=forward_checking)
            if inferences != "failure":
                domains_before_inference = csp["domains"].copy()
                csp["domains"] = inferences
                result = backtrack(csp, assignment)
                if result != "failure": return result
                csp["domains"] = domains_before_inference
            del assignment[var]
            csp["domains"] = domains_before_assignment
    return "failure"

def backtracking_search(csp):
    """Recursive backtracking search for constraint satisfaction problems
    with blank initial assignment

    Args:
        csp (dict): CSP.

    Returns:
        Solution assignment or failure
    """
    return backtrack(csp, {})

############################# READ INPUTS ##############################

runs["CO"] = times["CO"] = 0

def coloring(plot_graph=False):
    """Run coloring CSP

    Args:
        plot_graph (bool, optional): If True, plots solution graph.
        Defaults to False.
    """
    global runs, times

    with open('gcp.json', 'r') as f:
        data = json.load(f)

        # Convert data point keys into integers
        data["points"] = {int(k): v for k, v in data["points"].items()}

        # Create Coloring csp
        edges = data["edges"]
        X = list(data["points"].keys())
        D = ["red", "blue", "green", "orange"]
        C = coloring_constraints(X, edges)
        csp = gen_csp(X, D, C)

        # Calculate result
        runs["CO"] += 1
        start = time.time()

        result = backtracking_search(csp)

        end = time.time()
        times["CO"] += end - start

        if result == "failure":
            print(result)
            print()
        elif plot_graph:
            for edge in data["edges"]:
                x = [data["points"][p][0] for p in edge]
                y = [data["points"][p][1] for p in edge]
                plt.plot(x, y, color="black", alpha=0.1, zorder=0)
            colors = D
            for val in colors:
                x = [data["points"][k][0] for k, v in result.items() if v == val]
                y = [data["points"][k][1] for k, v in result.items() if v == val]
                plt.scatter(x, y, color=val, s=50, zorder=1)
            plt.title("Num points = " + str(len(data["points"])))
            plt.show()

runs["SK"] = times["SK"] = 0

def sudoku(print_table=False):
    """Run Sudoku CSP

    Args:
        print_table (bool, optional): If True, prints beginning and
        solution tables. Defaults to False.
    """
    global runs, times

    with open('sudoku.json', 'r') as f:
        board = json.load(f)

        if print_table:
            pprint.pprint(board)
            print()

        # Flatten 2D board into 1D list
        flat_board = [x for row in board for x in row]
        # Convert list into assignment dictionary
        assignment = {i: v for i, v in enumerate(flat_board) if v != 0}

        # Create Sudoku csp
        X = list(range(81))
        D = list(range(1, 10))
        C = sudoku_constraints()
        csp = gen_csp(X, D, C)

        # Calculate result
        runs["SK"] += 1
        start = time.time()

        result = backtrack(csp, assignment)

        end = time.time()
        times["SK"] += end - start

        if result == "failure":
            print(result)
        elif print_table:
            # Convert result into list
            result_list = [result[k] for k in range(81)]
            # Convert 1D list into 2D board
            result_board = [[result_list[9*r + c] for c in range(9)] for r in range(9)]
            pprint.pprint(result_board)

if __name__ == '__main__':
    # for n in range(100):
    #     print(n)
    #     os.system("python .\sudoku-generator.py 60")
    #     sudoku()
    # print(times["SK"]/runs["SK"])

    coloring(plot_graph=True)
    sudoku(print_table=True)
