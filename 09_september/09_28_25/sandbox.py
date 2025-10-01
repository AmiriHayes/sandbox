# Pyomo Optimization Tutorial
# Complete guide to mathematical optimization with Pyomo

# Installation and imports
# !apt-get install -y glpk-utils
import pyomo.environ as pe
import pyomo.opt as po
import numpy as np
import pandas as pd
from pyomo.util.infeasible import log_infeasible_constraints

# ## Problem 1: Linear Programming - Production Planning
# 
# **Scenario**: A factory produces two products (A and B) with limited resources.
# - Product A: profit $40/unit, requires 2 hours labor, 1 kg material
# - Product B: profit $30/unit, requires 1 hour labor, 2 kg material  
# - Available: 100 hours labor, 80 kg material
# **Goal**: Maximize profit

def production_planning_lp():
    # Create model
    model = pe.ConcreteModel()
    
    # Decision variables
    model.x_A = pe.Var(within=pe.NonNegativeReals, doc='Units of Product A')
    model.x_B = pe.Var(within=pe.NonNegativeReals, doc='Units of Product B')
    
    # Objective function - maximize profit
    model.profit = pe.Objective(
        expr=40*model.x_A + 30*model.x_B,
        sense=pe.maximize,
        doc='Total profit to maximize'
    )
    
    # Constraints
    model.labor_constraint = pe.Constraint(
        expr=2*model.x_A + 1*model.x_B <= 100,
        doc='Labor hours limitation'
    )
    
    model.material_constraint = pe.Constraint(
        expr=1*model.x_A + 2*model.x_B <= 80,
        doc='Material kg limitation'
    )
    
    # Solve
    solver = pe.SolverFactory('glpk')
    results = solver.solve(model, tee=True)
    
    # Display results
    print(f"\nOptimal Solution:")
    print(f"Product A: {pe.value(model.x_A):.2f} units")
    print(f"Product B: {pe.value(model.x_B):.2f} units")
    print(f"Maximum Profit: ${pe.value(model.profit):.2f}")
    
    return model

model1 = production_planning_lp()

# ## Problem 2: Mixed Integer Programming - Facility Location
#
# **Scenario**: Company wants to open warehouses to serve 4 cities
# - 3 potential warehouse locations with different fixed costs and capacities
# - Each city has specific demand that must be satisfied
# - Transportation costs vary by distance
# **Goal**: Minimize total cost (fixed + transportation)

def facility_location_mip():
    model = pe.ConcreteModel()
    
    # Sets
    warehouses = ['W1', 'W2', 'W3']
    cities = ['C1', 'C2', 'C3', 'C4']
    
    # Parameters
    fixed_costs = {'W1': 1000, 'W2': 1200, 'W3': 900}
    capacities = {'W1': 150, 'W2': 200, 'W3': 180}
    demands = {'C1': 80, 'C2': 60, 'C3': 70, 'C4': 90}
    
    # Transportation costs (warehouse to city)
    transport_costs = {
        ('W1', 'C1'): 5, ('W1', 'C2'): 7, ('W1', 'C3'): 6, ('W1', 'C4'): 8,
        ('W2', 'C1'): 6, ('W2', 'C2'): 4, ('W2', 'C3'): 8, ('W2', 'C4'): 5,
        ('W3', 'C1'): 9, ('W3', 'C2'): 8, ('W3', 'C3'): 3, ('W3', 'C4'): 4
    }
    
    # Binary variables - whether to open warehouse
    model.y = pe.Var(warehouses, within=pe.Binary, doc='Warehouse open decision')
    
    # Continuous variables - shipment quantities
    model.x = pe.Var(warehouses, cities, within=pe.NonNegativeReals, 
                    doc='Shipment from warehouse to city')
    
    # Objective - minimize total cost
    model.total_cost = pe.Objective(
        expr=sum(fixed_costs[w] * model.y[w] for w in warehouses) + 
             sum(transport_costs[w,c] * model.x[w,c] 
                 for w in warehouses for c in cities),
        sense=pe.minimize
    )
    
    # Constraints
    # Demand satisfaction
    model.demand_constraints = pe.ConstraintList()
    for c in cities:
        model.demand_constraints.add(
            sum(model.x[w,c] for w in warehouses) == demands[c]
        )
    
    # Capacity constraints
    model.capacity_constraints = pe.ConstraintList()
    for w in warehouses:
        model.capacity_constraints.add(
            sum(model.x[w,c] for c in cities) <= capacities[w] * model.y[w]
        )
    
    # Solve
    solver = pe.SolverFactory('glpk')
    results = solver.solve(model, tee=True)
    
    # Results
    print(f"\nFacility Location Solution:")
    for w in warehouses:
        if pe.value(model.y[w]) > 0.5:
            print(f"Open warehouse {w} (Fixed cost: ${fixed_costs[w]})")
            total_shipped = sum(pe.value(model.x[w,c]) for c in cities)
            print(f"  Total shipment: {total_shipped:.1f} units")
            
    print(f"\nShipment Plan:")
    for w in warehouses:
        for c in cities:
            if pe.value(model.x[w,c]) > 0.01:
                print(f"  {w} -> {c}: {pe.value(model.x[w,c]):.1f} units")
                
    print(f"\nMinimum Total Cost: ${pe.value(model.total_cost):.2f}")
    
    return model

model2 = facility_location_mip()

# ## Problem 3: Nonlinear Programming - Portfolio Optimization
#
# **Scenario**: Invest in 4 stocks with different expected returns and risks
# - Each stock has expected return and standard deviation
# - Portfolio variance depends on correlations between stocks
# - Must invest all capital, limit individual stock weights
# **Goal**: Maximize return for given risk level (Markowitz model)

def portfolio_optimization_nlp():
    model = pe.ConcreteModel()
    
    # Data
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    expected_returns = {'AAPL': 0.12, 'GOOGL': 0.15, 'MSFT': 0.11, 'AMZN': 0.14}
    
    # Simplified correlation matrix (symmetric)
    correlations = {
        ('AAPL', 'AAPL'): 1.0, ('AAPL', 'GOOGL'): 0.3, ('AAPL', 'MSFT'): 0.4, ('AAPL', 'AMZN'): 0.2,
        ('GOOGL', 'AAPL'): 0.3, ('GOOGL', 'GOOGL'): 1.0, ('GOOGL', 'MSFT'): 0.5, ('GOOGL', 'AMZN'): 0.6,
        ('MSFT', 'AAPL'): 0.4, ('MSFT', 'GOOGL'): 0.5, ('MSFT', 'MSFT'): 1.0, ('MSFT', 'AMZN'): 0.3,
        ('AMZN', 'AAPL'): 0.2, ('AMZN', 'GOOGL'): 0.6, ('AMZN', 'MSFT'): 0.3, ('AMZN', 'AMZN'): 1.0
    }
    
    std_devs = {'AAPL': 0.20, 'GOOGL': 0.25, 'MSFT': 0.18, 'AMZN': 0.28}
    
    # Variables - portfolio weights
    model.w = pe.Var(stocks, within=pe.NonNegativeReals, bounds=(0, 0.4), 
                    doc='Portfolio weights')
    
    # Portfolio return
    portfolio_return = sum(expected_returns[s] * model.w[s] for s in stocks)
    
    # Portfolio variance (quadratic term)
    portfolio_variance = sum(
        model.w[i] * model.w[j] * std_devs[i] * std_devs[j] * correlations[i,j]
        for i in stocks for j in stocks
    )
    
    # Objective - maximize return minus risk penalty
    risk_aversion = 2.0  # Risk aversion parameter
    model.utility = pe.Objective(
        expr=portfolio_return - 0.5 * risk_aversion * portfolio_variance,
        sense=pe.maximize
    )
    
    # Constraints
    model.budget_constraint = pe.Constraint(
        expr=sum(model.w[s] for s in stocks) == 1.0,
        doc='Must invest all capital'
    )
    
    # Solve with nonlinear solver
    solver = pe.SolverFactory('ipopt', executable='/usr/bin/ipopt')
    if solver.available():
        results = solver.solve(model, tee=True)
    else:
        # Fallback to linear approximation
        print("IPOPT not available, using GLPK with linear approximation")
        solver = pe.SolverFactory('glpk')
        results = solver.solve(model, tee=True)
    
    # Results
    print(f"\nOptimal Portfolio:")
    total_return = 0
    total_risk = 0
    for s in stocks:
        weight = pe.value(model.w[s])
        if weight > 0.001:
            print(f"{s}: {weight:.1%}")
            total_return += weight * expected_returns[s]
    
    print(f"\nExpected Portfolio Return: {total_return:.2%}")
    print(f"Portfolio Utility: {pe.value(model.utility):.4f}")
    
    return model

model3 = portfolio_optimization_nlp()

# ## Problem 4: Multi-Period Planning with Inventory
#
# **Scenario**: Production planning over 6 months with inventory management
# - Variable demand and production costs over time
# - Limited production capacity and inventory storage
# - Setup costs for production in each period
# **Goal**: Minimize total cost while meeting demand

def multiperiod_inventory():
    model = pe.ConcreteModel()
    
    # Time periods
    periods = list(range(1, 7))  # 6 months
    
    # Data
    demand = {1: 100, 2: 150, 3: 180, 4: 140, 5: 200, 6: 160}
    production_cost = {1: 10, 2: 12, 3: 11, 4: 13, 5: 10, 6: 11}
    setup_cost = {1: 200, 2: 220, 3: 210, 4: 240, 5: 200, 6: 215}
    holding_cost = 2  # per unit per period
    production_capacity = 250
    max_inventory = 100
    
    # Variables
    model.x = pe.Var(periods, within=pe.NonNegativeReals, 
                    doc='Production quantity')
    model.y = pe.Var(periods, within=pe.Binary, 
                    doc='Setup decision (1 if produce)')
    model.inv = pe.Var(periods, within=pe.NonNegativeReals, 
                      bounds=(0, max_inventory), doc='Inventory level')
    
    # Objective
    model.total_cost = pe.Objective(
        expr=sum(production_cost[t] * model.x[t] + 
                setup_cost[t] * model.y[t] +
                holding_cost * model.inv[t] for t in periods),
        sense=pe.minimize
    )
    
    # Constraints
    model.inventory_balance = pe.ConstraintList()
    initial_inventory = 0
    
    for t in periods:
        if t == 1:
            prev_inv = initial_inventory
        else:
            prev_inv = model.inv[t-1]
            
        model.inventory_balance.add(
            prev_inv + model.x[t] == demand[t] + model.inv[t]
        )
    
    # Production capacity constraints
    model.capacity_constraints = pe.ConstraintList()
    for t in periods:
        model.capacity_constraints.add(model.x[t] <= production_capacity * model.y[t])
    
    # Solve
    solver = pe.SolverFactory('glpk')
    results = solver.solve(model, tee=True)
    
    # Results
    print(f"\nMulti-Period Production Plan:")
    print(f"{'Period':<8}{'Demand':<8}{'Produce':<8}{'Inventory':<10}{'Setup':<8}")
    print("-" * 50)
    
    for t in periods:
        print(f"{t:<8}{demand[t]:<8}{pe.value(model.x[t]):<8.0f}"
              f"{pe.value(model.inv[t]):<10.0f}{'Yes' if pe.value(model.y[t]) > 0.5 else 'No':<8}")
    
    print(f"\nTotal Cost: ${pe.value(model.total_cost):.2f}")
    
    # Cost breakdown
    prod_cost = sum(production_cost[t] * pe.value(model.x[t]) for t in periods)
    setup_cost_total = sum(setup_cost[t] * pe.value(model.y[t]) for t in periods)
    hold_cost = sum(holding_cost * pe.value(model.inv[t]) for t in periods)
    
    print(f"Production Cost: ${prod_cost:.2f}")
    print(f"Setup Cost: ${setup_cost_total:.2f}")  
    print(f"Holding Cost: ${hold_cost:.2f}")
    
    return model

model4 = multiperiod_inventory()

# ## Problem 5: Transportation Network Optimization
#
# **Scenario**: Multi-commodity flow on transportation network
# - Multiple suppliers, distribution centers, and customers
# - Different products with varying transportation costs
# - Capacity constraints on links and storage limits
# **Goal**: Minimize transportation costs while satisfying demands

def transportation_network():
    model = pe.ConcreteModel()
    
    # Network nodes
    suppliers = ['S1', 'S2', 'S3']
    distribution_centers = ['DC1', 'DC2']  
    customers = ['C1', 'C2', 'C3', 'C4']
    products = ['P1', 'P2']
    
    # Supply and demand
    supply = {('S1', 'P1'): 200, ('S1', 'P2'): 150,
              ('S2', 'P1'): 180, ('S2', 'P2'): 120, 
              ('S3', 'P1'): 160, ('S3', 'P2'): 200}
    
    demand = {('C1', 'P1'): 80, ('C1', 'P2'): 60,
              ('C2', 'P1'): 90, ('C2', 'P2'): 70,
              ('C3', 'P1'): 100, ('C3', 'P2'): 80,
              ('C4', 'P1'): 70, ('C4', 'P2'): 90}
    
    # Transportation costs (per unit)
    transport_cost = {}
    # Supplier to DC costs
    for s in suppliers:
        for dc in distribution_centers:
            for p in products:
                transport_cost[s, dc, p] = np.random.uniform(2, 5)
    
    # DC to customer costs  
    for dc in distribution_centers:
        for c in customers:
            for p in products:
                transport_cost[dc, c, p] = np.random.uniform(1, 4)
    
    # Decision variables
    model.flow_s_dc = pe.Var(suppliers, distribution_centers, products,
                            within=pe.NonNegativeReals, 
                            doc='Flow from supplier to DC')
    
    model.flow_dc_c = pe.Var(distribution_centers, customers, products,
                            within=pe.NonNegativeReals,
                            doc='Flow from DC to customer')
    
    # Objective - minimize transportation cost
    model.total_transport_cost = pe.Objective(
        expr=sum(transport_cost[s,dc,p] * model.flow_s_dc[s,dc,p] 
                for s in suppliers for dc in distribution_centers for p in products) +
             sum(transport_cost[dc,c,p] * model.flow_dc_c[dc,c,p]
                for dc in distribution_centers for c in customers for p in products),
        sense=pe.minimize
    )
    
    # Supply constraints
    model.supply_constraints = pe.ConstraintList()
    for s in suppliers:
        for p in products:
            model.supply_constraints.add(
                sum(model.flow_s_dc[s,dc,p] for dc in distribution_centers) <= supply[s,p]
            )
    
    # Demand constraints  
    model.demand_constraints = pe.ConstraintList()
    for c in customers:
        for p in products:
            model.demand_constraints.add(
                sum(model.flow_dc_c[dc,c,p] for dc in distribution_centers) >= demand[c,p]
            )
    
    # Flow balance at distribution centers
    model.flow_balance = pe.ConstraintList()
    for dc in distribution_centers:
        for p in products:
            model.flow_balance.add(
                sum(model.flow_s_dc[s,dc,p] for s in suppliers) ==
                sum(model.flow_dc_c[dc,c,p] for c in customers)
            )
    
    # Solve
    solver = pe.SolverFactory('glpk')
    results = solver.solve(model, tee=False)
    
    print(f"\nTransportation Network Solution:")
    print(f"Total Transportation Cost: ${pe.value(model.total_transport_cost):.2f}")
    
    print(f"\nSupplier to DC Flows:")
    for s in suppliers:
        for dc in distribution_centers:
            for p in products:
                flow = pe.value(model.flow_s_dc[s,dc,p])
                if flow > 0.1:
                    print(f"  {s} -> {dc} ({p}): {flow:.0f} units")
    
    print(f"\nDC to Customer Flows:")
    for dc in distribution_centers:  
        for c in customers:
            for p in products:
                flow = pe.value(model.flow_dc_c[dc,c,p])
                if flow > 0.1:
                    print(f"  {dc} -> {c} ({p}): {flow:.0f} units")
    
    return model

model5 = transportation_network()

# ## Advanced Features and Model Analysis

# Model inspection and debugging
def analyze_model(model, model_name):
    print(f"\n=== {model_name} Analysis ===")
    print(f"Number of variables: {model.nvariables()}")
    print(f"Number of constraints: {model.nconstraints()}")
    print(f"Number of objectives: {model.nobjectives()}")
    
    # Display model structure
    print(f"\nModel components:")
    for component in model.component_objects():
        print(f"  {component.name}: {type(component).__name__}")

# Analyze our models
analyze_model(model1, "Production Planning LP")
analyze_model(model2, "Facility Location MIP") 
analyze_model(model3, "Portfolio Optimization NLP")

# ## Sensitivity Analysis Example
def sensitivity_analysis():
    # Create a simple model for sensitivity analysis
    model = pe.ConcreteModel()
    
    model.x = pe.Var(within=pe.NonNegativeReals)
    model.y = pe.Var(within=pe.NonNegativeReals)
    
    # Parameter we'll change
    model.rhs_param = pe.Param(initialize=100, mutable=True)
    
    model.obj = pe.Objective(expr=3*model.x + 2*model.y, sense=pe.maximize)
    model.c1 = pe.Constraint(expr=model.x + model.y <= model.rhs_param)
    model.c2 = pe.Constraint(expr=2*model.x + model.y <= 150)
    
    solver = pe.SolverFactory('glpk')
    
    print(f"\nSensitivity Analysis:")
    print(f"{'RHS Value':<10}{'Obj Value':<12}{'x*':<8}{'y*':<8}")
    print("-" * 40)
    
    for rhs_val in [80, 90, 100, 110, 120]:
        model.rhs_param.set_value(rhs_val)
        results = solver.solve(model, tee=False)
        
        if results.solver.termination_condition == pe.TerminationCondition.optimal:
            print(f"{rhs_val:<10}{pe.value(model.obj):<12.1f}"
                  f"{pe.value(model.x):<8.1f}{pe.value(model.y):<8.1f}")

sensitivity_analysis()

# ## Stochastic Programming Example - Two-Stage Model
def two_stage_stochastic():
    model = pe.ConcreteModel()
    
    # Scenarios for uncertain demand
    scenarios = ['Low', 'Medium', 'High']
    probabilities = {'Low': 0.3, 'Medium': 0.4, 'High': 0.3}
    demand_scenarios = {'Low': 80, 'Medium': 120, 'High': 160}
    
    # First-stage variables (capacity decisions)
    model.capacity = pe.Var(within=pe.NonNegativeReals, doc='Production capacity')
    
    # Second-stage variables (production decisions by scenario)
    model.production = pe.Var(scenarios, within=pe.NonNegativeReals, 
                             doc='Production level by scenario')
    model.shortage = pe.Var(scenarios, within=pe.NonNegativeReals,
                           doc='Demand shortage by scenario')
    
    # Costs
    capacity_cost = 50  # per unit of capacity
    production_cost = 10  # per unit produced
    shortage_penalty = 100  # per unit of unmet demand
    
    # Objective - minimize expected cost
    model.expected_cost = pe.Objective(
        expr=capacity_cost * model.capacity +
             sum(probabilities[s] * (production_cost * model.production[s] + 
                                   shortage_penalty * model.shortage[s]) 
                 for s in scenarios),
        sense=pe.minimize
    )
    
    # Constraints
    model.capacity_constraints = pe.ConstraintList()
    model.demand_constraints = pe.ConstraintList()
    
    for s in scenarios:
        # Production limited by capacity
        model.capacity_constraints.add(model.production[s] <= model.capacity)
        
        # Demand satisfaction with shortage
        model.demand_constraints.add(
            model.production[s] + model.shortage[s] >= demand_scenarios[s]
        )
    
    # Solve
    solver = pe.SolverFactory('glpk')
    results = solver.solve(model, tee=False)
    
    print(f"\nTwo-Stage Stochastic Programming Solution:")
    print(f"Optimal Capacity: {pe.value(model.capacity):.1f} units")
    print(f"Expected Total Cost: ${pe.value(model.expected_cost):.2f}")
    
    print(f"\nScenario Analysis:")
    print(f"{'Scenario':<10}{'Probability':<12}{'Demand':<8}{'Production':<12}{'Shortage':<10}")
    print("-" * 65)
    
    for s in scenarios:
        print(f"{s:<10}{probabilities[s]:<12.1%}{demand_scenarios[s]:<8}"
              f"{pe.value(model.production[s]):<12.1f}{pe.value(model.shortage[s]):<10.1f}")

two_stage_stochastic()

print("\n" + "="*60)
print("PYOMO TUTORIAL COMPLETE")
print("="*60)
print("You've learned:")
print("• Linear Programming (Production Planning)")
print("• Mixed Integer Programming (Facility Location)")  
print("• Nonlinear Programming (Portfolio Optimization)")
print("• Multi-period Models (Inventory Management)")
print("• Network Flow Models (Transportation)")
print("• Sensitivity Analysis")
print("• Stochastic Programming")
print("\nNext steps: Explore domain-specific applications!")