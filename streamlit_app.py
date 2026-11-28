import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


st.set_page_config(page_title="Smart Fleet Dispatch IDSS", layout="wide")

FIXED_COST = 50
FUEL_PRICE = 1.79
BOTTLE_WEIGHT = 10
UNLOADING_SECONDS = 10
MAX_ROUTE_MINUTES = 540

TRUCKS = {
    "Large": {"capacity": 6000, "efficiency": 5.5},
    "Medium": {"capacity": 4000, "efficiency": 5.0},
    "Small 1": {"capacity": 3000, "efficiency": 4.5},
    "Small 2": {"capacity": 3000, "efficiency": 4.5},
}


@st.cache_data
def load_files(dataset_file, matrix_file):
    dataset = pd.read_excel(dataset_file)
    matrix = pd.read_csv(matrix_file, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.astype(float)

    distances = (
        dataset[["Origin", "Destination", "Distance_m"]]
        .drop_duplicates(["Origin", "Destination"])
        .set_index(["Origin", "Destination"])["Distance_m"]
        .to_dict()
    )
    distances = {(i, j): value / 1000 for (i, j), value in distances.items()}
    return matrix, distances


def create_routes(customers, demand, matrix, distances):
    routes = []
    for size in range(1, len(customers) + 1):
        for subset in itertools.combinations(customers, size):
            for order in itertools.permutations(subset):
                sequence = ("Depot",) + order + ("Depot",)
                travel_seconds = sum(
                    matrix.loc[sequence[i], sequence[i + 1]]
                    for i in range(len(sequence) - 1)
                )
                bottles = sum(demand[customer] for customer in order)
                unloading_minutes = bottles * UNLOADING_SECONDS / 60
                delivery_travel_seconds = sum(
                    matrix.loc[sequence[i], sequence[i + 1]]
                    for i in range(len(sequence) - 2)
                )
                delivery_time = delivery_travel_seconds / 60 + unloading_minutes
                return_time = travel_seconds / 60 + unloading_minutes
                distance = sum(
                    distances[(sequence[i], sequence[i + 1])]
                    for i in range(len(sequence) - 1)
                )
                if return_time <= MAX_ROUTE_MINUTES:
                    routes.append({
                        "sequence": sequence,
                        "customers": order,
                        "bottles": bottles,
                        "delivery_time": delivery_time,
                        "return_time": return_time,
                        "travel": delivery_travel_seconds / 60,
                        "distance": distance,
                    })
    return routes


def solve(routes, customers, demand, cost_weight, time_weight, mode="weighted"):
    variables = []
    for truck_name, truck in TRUCKS.items():
        for route in routes:
            if route["bottles"] * BOTTLE_WEIGHT <= truck["capacity"]:
                variables.append((truck_name, route))

    n = len(variables)
    costs = np.array([
        FIXED_COST + route["distance"] / TRUCKS[name]["efficiency"] * FUEL_PRICE
        for name, route in variables
    ])
    times = np.array([route["delivery_time"] for _, route in variables])

    if mode == "cost":
        objective = costs
    elif mode == "time":
        objective = times
    else:
        objective = cost_weight * costs / max(costs) + time_weight * times / max(times)

    row_count = len(customers) + len(TRUCKS)
    matrix = lil_matrix((row_count, n), dtype=float)
    lower = np.full(row_count, -np.inf)
    upper = np.full(row_count, np.inf)

    for row, customer in enumerate(customers):
        for col, (_, route) in enumerate(variables):
            if customer in route["customers"]:
                matrix[row, col] = 1
        lower[row] = 1
        upper[row] = 1

    for truck_row, truck_name in enumerate(TRUCKS, start=len(customers)):
        for col, (name, _) in enumerate(variables):
            if name == truck_name:
                matrix[truck_row, col] = 1
        upper[truck_row] = 1

    result = milp(
        c=objective,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 120},
    )

    if not result.success:
        return None

    selected = [
        variables[i]
        for i, value in enumerate(result.x)
        if value > 0.5
    ]

    rows = []
    total_cost = 0
    makespan = 0

    for name, route in selected:
        truck = TRUCKS[name]
        fuel_cost = route["distance"] / truck["efficiency"] * FUEL_PRICE
        route_cost = FIXED_COST + fuel_cost
        total_cost += route_cost
        makespan = max(makespan, route["delivery_time"])
        truck_icon = {"Large": "🚛", "Medium": "🚚", "Small 1": "🚐", "Small 2": "🚐"}[name]
        rows.append({
            "Truck": f"{truck_icon} {name}",
            "Route": " -> ".join(route["sequence"]),
            "Bottles": route["bottles"],
            "Load (kg)": route["bottles"] * BOTTLE_WEIGHT,
            "Capacity (kg)": truck["capacity"],
            "Utilization (%)": round(route["bottles"] * BOTTLE_WEIGHT / truck["capacity"] * 100, 1),
            "Travel (min)": round(route["travel"], 2),
            "Delivery Time (min)": round(route["delivery_time"], 2),
            "Return Time (min)": round(route["return_time"], 2),
            "Cost (SAR)": round(route_cost, 2),
        })

    return pd.DataFrame(rows), total_cost, makespan


def explanation_text(table, total_cost, makespan, demand, cost_percent):
    used_trucks = len(table)
    total_bottles = sum(demand.values())
    return (
        f"The model selected {used_trucks} truck(s) to deliver {total_bottles} bottles "
        f"across {len(demand)} customer stations. The transportation cost is "
        f"{total_cost:.2f} SAR and the latest Delivery Time is {makespan:.2f} minutes. "
        f"The current priorities are {cost_percent}% cost and {100 - cost_percent}% time."
    )


def recommendations(table, total_cost, makespan):
    items = []
    if table.empty:
        return ["R7 - Infeasible Distribution Plan: No feasible distribution plan was found under the current constraints."]

    average_utilization = table["Utilization (%)"].mean()
    if total_cost > 250:
        items.append(
            f"R1 - High Transportation Cost: The current cost is {total_cost:.2f} SAR. "
            "Consider increasing the cost priority and testing the revised plan."
        )
    if makespan > 300:
        items.append(
            f"R2 - Long Delivery Time: The current Delivery Time is {makespan:.2f} minutes. "
            "Consider increasing the time priority and testing the revised plan."
        )
    low_utilization = table[table["Utilization (%)"] < 50]
    if not low_utilization.empty:
        for _, row in low_utilization.iterrows():
            items.append(
                f"R3 - Low Vehicle Utilization: {row['Truck']} is operating at "
                f"{row['Utilization (%)']}% capacity utilization. Consider testing whether "
                "its deliveries can be reassigned."
            )
    if average_utilization < 60:
        items.append(
            f"R3 note: Average vehicle utilization is {average_utilization:.1f}%. "
            "Consider testing a scenario with fewer trucks or different weights."
        )
    total_load = table["Load (kg)"].sum()
    fleet_capacity = sum(truck["capacity"] for truck in TRUCKS.values())
    if total_load > fleet_capacity:
        items.append(
            "R4 - Limited Fleet Capacity: Scheduled demand exceeds available fleet capacity."
        )
    if any(value > 600 for value in st.session_state.get("demand", {}).values()):
        items.append(
            "R5 - Carried-over Customer Demand: A station exceeds the daily scheduling limit "
            "of 600 bottles and may require carry-over to the next day."
        )
    if (table["Return Time (min)"] >= 510).any():
        items.append(
            "R6 - Limited Operating Time: A truck has 30 minutes or less before the "
            "5:00 PM return deadline. Consider testing a higher time priority."
        )
    if not items:
        items.append(
            "No additional recommendations were activated under the current evaluation criteria."
        )
    return items


st.title("Smart Fleet Dispatch Decision Support System")
st.caption("An intelligent system for selecting vehicles, assigning customer stations, and planning efficient delivery routes.")

with st.sidebar:
    st.header("Inputs")
    dataset_upload = st.file_uploader("Upload dataset", type=["xlsx"])
    matrix_upload = st.file_uploader("Upload travel-time matrix", type=["csv", "xls"])
    st.subheader("Optimization Priorities")
    cost_percent = st.slider("Cost weight (%)", 0, 100, 50)
    time_percent = 100 - cost_percent
    st.write(f"Time weight: {time_percent}%")

default_dataset = Path("full_dataset_large(1).xlsx")
default_matrix = Path("travel_time_matrix_seconds.xls")

if dataset_upload is not None and matrix_upload is not None:
    dataset_source = dataset_upload
    matrix_source = matrix_upload
elif default_dataset.exists() and default_matrix.exists():
    dataset_source = str(default_dataset)
    matrix_source = str(default_matrix)
else:
    st.info("Upload the dataset and travel-time matrix from the sidebar.")
    st.stop()

try:
    travel_matrix, distances = load_files(dataset_source, matrix_source)
except Exception as error:
    st.error(f"Unable to read input files: {error}")
    st.stop()

customers = [location for location in travel_matrix.index if location != "Depot"]

st.subheader("Scheduled Demand")
cols = st.columns(len(customers))
demand = {}
for col, customer in zip(cols, customers):
    demand[customer] = col.number_input(
        customer,
        min_value=0,
        max_value=600,
        value={"S1": 150, "S2": 200, "S3": 100, "S4": 250, "S5": 180, "S6": 220}.get(customer, 0),
        step=10,
    )

if st.button("Run MILP Optimization", type="primary"):
    active_customers = [customer for customer in customers if demand[customer] > 0]
    if not active_customers:
        st.warning("Enter demand for at least one customer.")
        st.stop()

    total_weight = sum(demand.values()) * BOTTLE_WEIGHT
    fleet_capacity = sum(truck["capacity"] for truck in TRUCKS.values())
    if total_weight > fleet_capacity:
        st.error("The total demand exceeds the available fleet capacity.")
        st.stop()

    with st.spinner("Solving MILP..."):
        routes = create_routes(active_customers, demand, travel_matrix, distances)
        solution = solve(
            routes,
            active_customers,
            demand,
            cost_percent / 100,
            time_percent / 100,
        )

    if solution is None:
        st.error("Solver status: Infeasible")
    else:
        solution_table, total_cost, makespan = solution
        st.session_state["solution"] = solution
        st.session_state["demand"] = demand.copy()
        st.session_state["active_customers"] = active_customers.copy()
        st.session_state["cost_percent"] = cost_percent
        st.success("Solver status: Optimal")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total Cost", f"{total_cost:.2f} SAR")
        metric2.metric("Delivery Time", f"{makespan:.2f} min")
        metric3.metric("Trucks Used", len(solution_table))

        st.subheader("Recommended Distribution Plan")
        st.dataframe(solution_table, use_container_width=True, hide_index=True)

        st.download_button(
            "Download MILP Solution",
            solution_table.to_csv(index=False).encode("utf-8"),
            "milp_solution.csv",
            "text/csv",
        )

        st.subheader("Automatic Explanation")
        st.write(
            explanation_text(solution_table, total_cost, makespan, demand, cost_percent)
        )


if "solution" in st.session_state:
    saved_table, saved_cost, saved_makespan = st.session_state["solution"]
    saved_demand = st.session_state["demand"]
    saved_cost_percent = st.session_state["cost_percent"]

    st.divider()
    st.header("Decision Support")

    explanation_tab, recommendation_tab, what_if_tab = st.tabs(
        ["Detailed Explanations", "Recommendations", "What-if Scenarios"]
    )

    with explanation_tab:
        question = st.selectbox(
            "Select a question",
            [
                "Why were some available trucks not selected?",
                "Why was this truck selected for these deliveries?",
                "Why were multiple customer stations assigned to the same truck?",
                "Why was this delivery route selected?",
                "Why is the utilization of a truck low?",
                "Why does a truck return later than its last delivery completion?",
                "What determines the delivery time?",
                "How do the selected cost and time weights influence the plan?",
                "Why does the system not minimize cost and time independently?",
                "Does giving more priority to delivery time mean the system will use more trucks?",
                "Is the recommended distribution plan guaranteed to be optimal?",
                "Why was no feasible distribution plan generated?",
                "Why does the system show a feasible plan instead of an optimal plan?",
                "Why might actual delivery times differ from predicted times?",
                "Why was part of a station's demand moved to the next day?",
            ],
        )

        if question == "Why were some available trucks not selected?":
            st.write(
                f"The model selected {len(saved_table)} truck(s) because the selected fleet "
                f"can serve the scheduled demand while respecting capacity and return-time "
                f"constraints. Using another truck would add a fixed operating cost of 50 SAR."
            )
        elif question == "Why was this truck selected for these deliveries?":
            st.write(
                "The truck was selected because its capacity can carry the assigned load and "
                "its route can be completed within the daily operating limit."
            )
        elif question == "Why were multiple customer stations assigned to the same truck?":
            st.write(
                "Customers were assigned to the same truck because the combined load "
                "fits within the truck capacity and the resulting route remains feasible "
                "within the operating-time limit."
            )
        elif question == "Why was this delivery route selected?":
            st.write(
                "The route was selected because it is feasible and provides a favorable "
                "cost-time result under the selected optimization weights."
            )
        elif question == "Why is the utilization of a truck low?":
            st.write(
                "Truck utilization may be low because the model must satisfy customer "
                "assignments, vehicle capacities, route feasibility, and return-time limits."
            )
        elif question == "Why does a truck return later than its last delivery completion?":
            st.write(
                "The truck may need additional travel from its final customer back to the "
                "depot. This return time is separate from Delivery Time."
            )
        elif question == "What determines the delivery time?":
            st.write(
                f"Delivery Time is the travel time from the depot through the final customer "
                f"plus unloading time. For this plan, the latest Delivery Time is "
                f"{saved_makespan:.2f} minutes. Return travel is not included."
            )
        elif question == "How do the selected cost and time weights influence the plan?":
            st.write(
                f"The current priorities are {saved_cost_percent}% cost and "
                f"{100 - saved_cost_percent}% time. Increasing the cost weight favors "
                "lower-cost plans, while increasing the time weight favors faster plans."
            )
        elif question == "Why does the system not minimize cost and time independently?":
            st.write(
                "Cost and time can conflict. The system uses the selected weights to produce "
                "one balanced plan instead of optimizing each objective separately."
            )
        elif question == "Does giving more priority to delivery time mean the system will use more trucks?":
            st.write(
                "Not necessarily. Increasing time priority may change truck selection or routes, "
                "but the actual result depends on capacity, route times, and fixed truck costs. "
                "A what-if run is required to confirm the impact."
            )
        elif question == "Is the recommended distribution plan guaranteed to be optimal?":
            st.write(
                "The plan is reported as optimal only when the solver returns an optimal status "
                "for the implemented model. It is optimal for the provided inputs and assumptions."
            )
        elif question == "Why was no feasible distribution plan generated?":
            st.write(
                "A plan may be infeasible when demand cannot fit within fleet capacity or the "
                "operating-time constraints. Review demand, fleet availability, and capacities."
            )
        elif question == "Why does the system show a feasible plan instead of an optimal plan?":
            st.write(
                "A feasible plan satisfies all constraints, but the solver may not have proved "
                "that no better solution exists. The system should report the actual solver status."
            )
        elif question == "Why might actual delivery times differ from predicted times?":
            st.write(
                "Actual times may differ because traffic, weather, service conditions, and "
                "other operational events can change after planning."
            )
        elif question == "Why was part of a station's demand moved to the next day?":
            st.write(
                "Demand above the daily scheduling limit of 600 bottles may be carried over "
                "to the next day under the demand-preprocessing rule."
            )
        else:
            st.write(
                "Total cost = fixed cost for each used truck + diesel fuel cost. "
                f"The resulting total is {saved_cost:.2f} SAR."
            )

    with recommendation_tab:
        st.subheader("Operational Recommendations")
        for item in recommendations(saved_table, saved_cost, saved_makespan):
            st.info(item)

    with what_if_tab:
        st.subheader("Test an Alternative Scenario")
        demand_change = st.slider(
            "Change all customer demand by (%)",
            min_value=-50,
            max_value=100,
            value=0,
            step=10,
        )
        scenario_cost_percent = st.slider(
            "Scenario cost weight (%)",
            min_value=0,
            max_value=100,
            value=saved_cost_percent,
        )

        if st.button("Run What-if Scenario"):
            scenario_demand = {
                customer: max(0, round(value * (1 + demand_change / 100)))
                for customer, value in saved_demand.items()
            }
            scenario_customers = [
                customer for customer, value in scenario_demand.items() if value > 0
            ]
            scenario_routes = create_routes(
                scenario_customers,
                scenario_demand,
                travel_matrix,
                distances,
            )
            scenario_solution = solve(
                scenario_routes,
                scenario_customers,
                scenario_demand,
                scenario_cost_percent / 100,
                (100 - scenario_cost_percent) / 100,
            )

            if scenario_solution is None:
                st.error("The what-if scenario is infeasible.")
            else:
                scenario_table, scenario_cost, scenario_makespan = scenario_solution
                comparison = pd.DataFrame({
                    "Measure": [
                        "Total Cost (SAR)",
                        "Delivery Time (min)",
                        "Trucks Used",
                    ],
                    "Original Plan": [
                        round(saved_cost, 2),
                        round(saved_makespan, 2),
                        len(saved_table),
                    ],
                    "What-if Plan": [
                        round(scenario_cost, 2),
                        round(scenario_makespan, 2),
                        len(scenario_table),
                    ],
                })
                st.success("What-if scenario solved successfully.")
                st.dataframe(comparison, hide_index=True, use_container_width=True)
                st.subheader("What-if Routes")
                st.dataframe(scenario_table, hide_index=True, use_container_width=True)
