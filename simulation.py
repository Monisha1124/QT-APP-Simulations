"""
QT-APP Corrected Congestion Analysis
=====================================

Purpose
-------
Experimental simulation of Quantum-Tunneling-Based Adaptive Path Planning
(QT-APP) against Dijkstra under dynamic network conditions.

This version corrects several issues identified in the earlier implementation:

1. Path cost is additive rather than a mean of edge costs.
2. Hop penalty is explicitly included.
3. Dijkstra and QT-APP use the same base route-quality objective.
4. Tunneling is based on a route's maximum edge barrier, making it a
   distinct mechanism from ordinary thermal route selection.
5. Tunneling influence is bounded so that it cannot overwhelm route quality.
6. The "without tunneling" ablation genuinely removes the tunneling term.
7. The "without adaptation" ablation keeps tunneling but fixes energy/kappa.
8. Candidate selection and tunneling diagnostics are recorded.
9. Candidate/probability array lengths are guaranteed to match.
10. Dynamic-regime analysis is retained.

Outputs
-------
CSV:
    results/corrected_congestion_analysis_results.csv
    results/corrected_congestion_period_summary.csv
    results/corrected_congestion_improvement_vs_dijkstra.csv
    results/corrected_candidate_diagnostics.csv
    results/corrected_regime_summary.csv
    results/corrected_regime_improvement_vs_dijkstra.csv

PNG:
    results/corrected_congestion_path_cost.png
    results/corrected_congestion_latency.png
    results/corrected_congestion_packet_loss.png
    results/corrected_congestion_bandwidth.png
    results/corrected_congestion_reliability.png
    results/corrected_congestion_congestion.png
    results/corrected_congestion_hops.png
    results/corrected_regime_path_cost.png
    results/corrected_regime_latency.png
    results/corrected_regime_packet_loss.png
    results/corrected_regime_bandwidth.png
    results/corrected_regime_reliability.png
    results/corrected_regime_congestion.png
    results/corrected_regime_hops.png
    results/corrected_congestion_relative_performance.png
    results/corrected_algorithm_comparison.png
    results/corrected_tunneling_probability.png
    results/corrected_energy_kappa.png
    results/corrected_selected_rank.png

Requirements
------------
    pip install numpy pandas networkx matplotlib

Run
---
    python qt_app_congestion_corrected.py
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

NUM_NODES = 20
EDGE_PROBABILITY = 0.18

SOURCE = 0
DESTINATION = 19

TIME_STEPS = 40
SEEDS = [42, 43, 44, 45, 46]

# Dynamic congestion period
CONGESTION_START = 10
CONGESTION_END = 14

RECOVERY_START = 20
RECOVERY_END = 22

# Candidate routes
NUM_CANDIDATE_PATHS = 12

# QT-APP parameters
INITIAL_ENERGY = 0.45
INITIAL_KAPPA = 1.20

MIN_ENERGY = 0.05
MAX_ENERGY = 1.00

MIN_KAPPA = 0.20
MAX_KAPPA = 2.00

TEMPERATURE = 0.50

# Maximum contribution of tunneling to route-selection weight.
# score = thermal_weight * (1 + RHO * tunneling_probability)
RHO = 0.75

# Adaptation step sizes
ENERGY_UP_STEP = 0.035
ENERGY_DOWN_STEP = 0.020

KAPPA_DOWN_STEP = 0.025
KAPPA_UP_STEP = 0.020

# Stress thresholds
HIGH_STRESS_THRESHOLD = 0.55
LOW_STRESS_THRESHOLD = 0.35

HIGH_VOLATILITY_THRESHOLD = 0.15
LOW_VOLATILITY_THRESHOLD = 0.08

# Path-quality weights
W_LATENCY = 0.25
W_CONGESTION = 0.25
W_PACKET_LOSS = 0.15
W_BANDWIDTH = 0.15
W_RELIABILITY = 0.15
W_HOPS = 0.05

# Numerical safety
EPSILON = 1e-12


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# RANDOMNESS
# ============================================================

def set_seed(seed):
    """Set both Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# NETWORK CREATION
# ============================================================

def create_dynamic_network(seed):
    """
    Create the initial random network.

    Each undirected edge contains:
        latency
        congestion
        packet_loss
        bandwidth
        reliability
    """

    set_seed(seed)

    G = nx.Graph()

    for node in range(NUM_NODES):
        G.add_node(node)

    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):

            if np.random.random() < EDGE_PROBABILITY:

                latency = np.random.uniform(10.0, 50.0)
                congestion = np.random.uniform(0.10, 0.50)
                packet_loss = np.random.uniform(0.01, 0.08)
                bandwidth = np.random.uniform(20.0, 100.0)
                reliability = np.random.uniform(0.75, 0.99)

                G.add_edge(
                    i,
                    j,
                    latency=latency,
                    congestion=congestion,
                    packet_loss=packet_loss,
                    bandwidth=bandwidth,
                    reliability=reliability,
                )

    # Guarantee connectivity if the random graph happens to be disconnected.
    if not nx.is_connected(G):

        components = list(nx.connected_components(G))

        for index in range(len(components) - 1):

            a = list(components[index])[0]
            b = list(components[index + 1])[0]

            G.add_edge(
                a,
                b,
                latency=np.random.uniform(10.0, 50.0),
                congestion=np.random.uniform(0.10, 0.50),
                packet_loss=np.random.uniform(0.01, 0.08),
                bandwidth=np.random.uniform(20.0, 100.0),
                reliability=np.random.uniform(0.75, 0.99),
            )

    return G


# ============================================================
# DYNAMIC NETWORK UPDATE
# ============================================================

def update_network(G, time_step):
    """
    Apply dynamic changes to the network.

    Congestion period:
        t = 10 ... 14

    Recovery:
        t = 20 ... 22
    """

    for u, v, data in G.edges(data=True):

        # Small ordinary random movement
        latency_change = np.random.normal(0.0, 1.5)
        congestion_change = np.random.normal(0.0, 0.025)
        loss_change = np.random.normal(0.0, 0.004)
        bandwidth_change = np.random.normal(0.0, 3.0)
        reliability_change = np.random.normal(0.0, 0.008)

        # Artificial congestion event
        if CONGESTION_START <= time_step <= CONGESTION_END:

            latency_change += np.random.uniform(5.0, 15.0)
            congestion_change += np.random.uniform(0.15, 0.35)
            loss_change += np.random.uniform(0.02, 0.05)
            bandwidth_change -= np.random.uniform(5.0, 20.0)
            reliability_change -= np.random.uniform(0.03, 0.08)

        # Recovery after congestion
        elif RECOVERY_START <= time_step <= RECOVERY_END:

            latency_change -= np.random.uniform(2.0, 6.0)
            congestion_change -= np.random.uniform(0.05, 0.15)
            loss_change -= np.random.uniform(0.005, 0.015)
            bandwidth_change += np.random.uniform(3.0, 10.0)
            reliability_change += np.random.uniform(0.01, 0.03)

        data["latency"] = max(
            5.0,
            data["latency"] + latency_change
        )

        data["congestion"] = np.clip(
            data["congestion"] + congestion_change,
            0.0,
            1.0
        )

        data["packet_loss"] = np.clip(
            data["packet_loss"] + loss_change,
            0.0,
            0.50
        )

        data["bandwidth"] = np.clip(
            data["bandwidth"] + bandwidth_change,
            5.0,
            150.0
        )

        data["reliability"] = np.clip(
            data["reliability"] + reliability_change,
            0.40,
            0.999
        )


# ============================================================
# NORMALIZATION
# ============================================================

def calculate_edge_normalization(G):
    """
    Calculate per-network-state min/max values for normalization.
    """

    attributes = [
        "latency",
        "congestion",
        "packet_loss",
        "bandwidth",
        "reliability",
    ]

    normalization = {}

    for attribute in attributes:

        values = [
            data[attribute]
            for _, _, data in G.edges(data=True)
        ]

        if not values:
            normalization[attribute] = (0.0, 1.0)
            continue

        minimum = min(values)
        maximum = max(values)

        if abs(maximum - minimum) < EPSILON:
            maximum = minimum + 1.0

        normalization[attribute] = (minimum, maximum)

    return normalization


def normalize_value(value, minimum, maximum):
    """Normalize to [0, 1]."""
    return np.clip(
        (value - minimum) / (maximum - minimum),
        0.0,
        1.0
    )


# ============================================================
# EDGE COST
# ============================================================

def calculate_edge_potential(G, u, v, normalization):
    """
    Calculate the normalized base route-quality cost for one edge.

    Lower is better.

    Components:
        latency
        congestion
        packet loss
        inverse bandwidth
        inverse reliability
        hop penalty
    """

    data = G[u][v]

    latency_norm = normalize_value(
        data["latency"],
        *normalization["latency"]
    )

    congestion_norm = normalize_value(
        data["congestion"],
        *normalization["congestion"]
    )

    loss_norm = normalize_value(
        data["packet_loss"],
        *normalization["packet_loss"]
    )

    bandwidth_norm = normalize_value(
        data["bandwidth"],
        *normalization["bandwidth"]
    )

    reliability_norm = normalize_value(
        data["reliability"],
        *normalization["reliability"]
    )

    inverse_bandwidth = 1.0 - bandwidth_norm
    inverse_reliability = 1.0 - reliability_norm

    potential = (
        W_LATENCY * latency_norm
        + W_CONGESTION * congestion_norm
        + W_PACKET_LOSS * loss_norm
        + W_BANDWIDTH * inverse_bandwidth
        + W_RELIABILITY * inverse_reliability
        + W_HOPS
    )

    return float(potential)


# ============================================================
# PATH METRICS
# ============================================================

def calculate_path_metrics(G, path, normalization):
    """
    Calculate complete metrics for a route.

    IMPORTANT:
    path_cost is the SUM of edge potentials, not their mean.
    """

    if path is None or len(path) < 2:
        return None

    edge_potentials = []

    latencies = []
    congestions = []
    packet_losses = []
    bandwidths = []
    reliabilities = []

    for u, v in zip(path[:-1], path[1:]):

        data = G[u][v]

        potential = calculate_edge_potential(
            G,
            u,
            v,
            normalization
        )

        edge_potentials.append(potential)

        latencies.append(data["latency"])
        congestions.append(data["congestion"])
        packet_losses.append(data["packet_loss"])
        bandwidths.append(data["bandwidth"])
        reliabilities.append(data["reliability"])

    hops = len(path) - 1

    # Additive route cost.
    path_cost = float(np.sum(edge_potentials))

    # Total latency is additive.
    latency = float(np.sum(latencies))

    # Average route-level conditions.
    congestion = float(np.mean(congestions))
    packet_loss = float(np.mean(packet_losses))
    bandwidth = float(np.mean(bandwidths))
    reliability = float(np.mean(reliabilities))

    # Maximum edge potential is used as the quantum barrier.
    max_edge_barrier = float(np.max(edge_potentials))

    # Mean edge potential is useful diagnostically.
    mean_edge_potential = float(np.mean(edge_potentials))

    return {
        "path": list(path),
        "path_cost": path_cost,
        "latency": latency,
        "hops": hops,
        "packet_loss": packet_loss,
        "bandwidth": bandwidth,
        "reliability": reliability,
        "congestion": congestion,
        "max_edge_barrier": max_edge_barrier,
        "mean_edge_potential": mean_edge_potential,
    }


# ============================================================
# DIJKSTRA ROUTE
# ============================================================

def dijkstra_route(G):
    """
    Dijkstra baseline using exactly the same base edge potential
    used by QT-APP.
    """

    normalization = calculate_edge_normalization(G)

    for u, v, data in G.edges(data=True):

        data["base_cost"] = calculate_edge_potential(
            G,
            u,
            v,
            normalization
        )

    try:
        path = nx.shortest_path(
            G,
            SOURCE,
            DESTINATION,
            weight="base_cost"
        )
    except nx.NetworkXNoPath:
        return None

    return calculate_path_metrics(
        G,
        path,
        normalization
    )


# ============================================================
# CANDIDATE PATH GENERATION
# ============================================================

def generate_candidate_paths(G):
    """
    Generate multiple high-quality simple paths using the common
    base route-quality objective.
    """

    normalization = calculate_edge_normalization(G)

    for u, v, data in G.edges(data=True):

        data["candidate_cost"] = calculate_edge_potential(
            G,
            u,
            v,
            normalization
        )

    try:

        generator = nx.shortest_simple_paths(
            G,
            SOURCE,
            DESTINATION,
            weight="candidate_cost"
        )

        candidates = []

        for path in generator:

            metrics = calculate_path_metrics(
                G,
                path,
                normalization
            )

            if metrics is not None:
                candidates.append(metrics)

            if len(candidates) >= NUM_CANDIDATE_PATHS:
                break

    except nx.NetworkXNoPath:
        return []

    return candidates


# ============================================================
# TUNNELING PROBABILITY
# ============================================================

def calculate_tunneling_probability(
    candidate,
    best_candidate_cost,
    energy,
    kappa
):
    """
    Quantum-inspired tunneling probability.

    The barrier is based on the route's maximum edge potential.

    The energy level is scaled relative to the best candidate route.

    This makes tunneling distinct from ordinary thermal selection:
    a route can have a somewhat higher total cost while possessing a
    lower maximum barrier.

    P_tunnel = exp(-2 * kappa * effective_barrier)

    where effective_barrier is the positive difference between the
    route's maximum edge barrier and the adaptive energy threshold.
    """

    best_cost = max(best_candidate_cost, EPSILON)

    # Relative route cost factor.
    relative_cost = candidate["path_cost"] / best_cost

    # The energy threshold is related to route-quality scale.
    #
    # Energy in [0, 1].
    # At low energy, only routes with relatively small barriers tunnel.
    # At high energy, more candidate barriers become accessible.
    energy_threshold = (
        0.25
        + 0.50 * energy
    )

    # Convert barrier into a relative scale.
    barrier = (
        candidate["max_edge_barrier"]
        / max(candidate["mean_edge_potential"], EPSILON)
    )

    # Small route-cost correction prevents very expensive paths from
    # receiving the same tunneling probability merely because they have
    # a favorable maximum-edge barrier.
    cost_penalty = max(
        0.0,
        relative_cost - 1.0
    ) * 0.15

    effective_barrier = max(
        0.0,
        barrier - energy_threshold + cost_penalty
    )

    probability = np.exp(
        -2.0 * kappa * effective_barrier
    )

    return float(
        np.clip(probability, 0.0, 1.0)
    )


# ============================================================
# THERMAL ROUTE SELECTION
# ============================================================

def calculate_selection_probabilities(
    candidates,
    tunneling_enabled,
    energy,
    kappa
):
    """
    Calculate route-selection probabilities.

    Base:
        thermal_weight = exp(-cost / T)

    With tunneling:
        score = thermal_weight * (1 + RHO * P_tunnel)

    Without tunneling:
        score = thermal_weight

    The tunneling multiplier is deliberately bounded between:
        1 and 1 + RHO

    This prevents tunneling from completely overpowering route quality.
    """

    if not candidates:
        return [], []

    costs = np.array(
        [candidate["path_cost"] for candidate in candidates],
        dtype=float
    )

    best_cost = float(np.min(costs))

    # Stabilize the thermal calculation by subtracting best cost.
    relative_costs = costs - best_cost

    thermal_weights = np.exp(
        -relative_costs / max(TEMPERATURE, EPSILON)
    )

    tunneling_values = []

    if tunneling_enabled:

        for candidate in candidates:

            probability = calculate_tunneling_probability(
                candidate,
                best_cost,
                energy,
                kappa
            )

            tunneling_values.append(probability)

        tunneling_values = np.array(
            tunneling_values,
            dtype=float
        )

        scores = (
            thermal_weights
            * (
                1.0
                + RHO * tunneling_values
            )
        )

    else:

        tunneling_values = np.zeros(
            len(candidates),
            dtype=float
        )

        scores = thermal_weights.copy()

    score_sum = float(np.sum(scores))

    if score_sum <= EPSILON:

        probabilities = np.ones(
            len(candidates),
            dtype=float
        ) / len(candidates)

    else:

        probabilities = (
            scores / score_sum
        )

    return (
        probabilities.tolist(),
        tunneling_values.tolist()
    )


# ============================================================
# ROUTE SELECTION
# ============================================================

def select_qt_route(
    G,
    energy,
    kappa,
    tunneling_enabled=True
):
    """
    Select one route using QT-APP logic.

    Returns:
        selected route metrics
        diagnostics dictionary
    """

    candidates = generate_candidate_paths(G)

    if not candidates:
        return None, None

    probabilities, tunneling_values = (
        calculate_selection_probabilities(
            candidates,
            tunneling_enabled,
            energy,
            kappa
        )
    )

    # Guarantee equal lengths.
    assert len(candidates) == len(probabilities)
    assert len(candidates) == len(tunneling_values)

    selected_index = int(
        np.random.choice(
            len(candidates),
            p=np.array(probabilities)
        )
    )

    selected = candidates[selected_index]

    sorted_indices = np.argsort(
        [
            candidate["path_cost"]
            for candidate in candidates
        ]
    )

    selected_rank = int(
        np.where(
            sorted_indices == selected_index
        )[0][0]
    ) + 1

    # Base thermal weight for diagnostics.
    candidate_costs = np.array(
        [candidate["path_cost"] for candidate in candidates]
    )

    best_cost = float(np.min(candidate_costs))

    relative_costs = candidate_costs - best_cost

    thermal_weights = np.exp(
        -relative_costs / max(TEMPERATURE, EPSILON)
    )

    if tunneling_enabled:

        scores = (
            thermal_weights
            * (
                1.0
                + RHO * np.array(tunneling_values)
            )
        )

    else:

        scores = thermal_weights

    diagnostics = {
        "candidate_count": len(candidates),
        "selected_index": selected_index,
        "selected_rank": selected_rank,
        "best_candidate_cost": best_cost,
        "selected_base_cost": selected["path_cost"],
        "mean_candidate_cost": float(
            np.mean(candidate_costs)
        ),
        "selected_tunneling_probability": float(
            tunneling_values[selected_index]
        ),
        "mean_tunneling_probability": float(
            np.mean(tunneling_values)
        ),
        "std_tunneling_probability": float(
            np.std(tunneling_values)
        ),
        "selection_probability": float(
            probabilities[selected_index]
        ),
        "selected_thermal_weight": float(
            thermal_weights[selected_index]
        ),
        "selected_final_score": float(
            scores[selected_index]
        ),
        "candidate_cost_std": float(
            np.std(candidate_costs)
        ),
    }

    return selected, diagnostics


# ============================================================
# ADAPTATION
# ============================================================

def calculate_route_stress(route):
    """
    Calculate a normalized stress score.

    Higher = more network stress.
    """

    if route is None:
        return 1.0

    stress = (
        0.50 * route["congestion"]
        + 0.30 * route["packet_loss"]
        + 0.20 * (1.0 - route["reliability"])
    )

    return float(
        np.clip(stress, 0.0, 1.0)
    )


def adapt_parameters(
    energy,
    kappa,
    current_route,
    previous_route
):
    """
    Adapt energy and kappa based on network stress and route volatility.

    High stress / high volatility:
        increase energy
        decrease kappa

    Low stress / low volatility:
        decrease energy
        increase kappa

    This allows tunneling to become more permissive during instability
    without directly replacing the route-quality objective.
    """

    if current_route is None:
        return energy, kappa

    current_stress = calculate_route_stress(
        current_route
    )

    if previous_route is None:

        volatility = 0.0

    else:

        previous_cost = max(
            previous_route["path_cost"],
            EPSILON
        )

        volatility = abs(
            current_route["path_cost"]
            - previous_route["path_cost"]
        ) / previous_cost

    high_stress = (
        current_stress > HIGH_STRESS_THRESHOLD
    )

    high_volatility = (
        volatility > HIGH_VOLATILITY_THRESHOLD
    )

    low_stress = (
        current_stress < LOW_STRESS_THRESHOLD
    )

    low_volatility = (
        volatility < LOW_VOLATILITY_THRESHOLD
    )

    if high_stress or high_volatility:

        energy += ENERGY_UP_STEP
        kappa -= KAPPA_DOWN_STEP

    elif low_stress and low_volatility:

        energy -= ENERGY_DOWN_STEP
        kappa += KAPPA_UP_STEP

    energy = float(
        np.clip(
            energy,
            MIN_ENERGY,
            MAX_ENERGY
        )
    )

    kappa = float(
        np.clip(
            kappa,
            MIN_KAPPA,
            MAX_KAPPA
        )
    )

    return energy, kappa


# ============================================================
# SINGLE ALGORITHM RUN
# ============================================================

def run_algorithm(
    algorithm,
    seed
):
    """
    Run one algorithm on one dynamic network realization.

    Algorithms:
        Dijkstra
        QT-APP
        QT-APP without adaptation
        QT-APP without tunneling
    """

    set_seed(seed)

    G = create_dynamic_network(seed)

    records = []
    diagnostics_records = []

    energy = INITIAL_ENERGY
    kappa = INITIAL_KAPPA

    previous_route = None

    previous_path = None

    for time_step in range(TIME_STEPS):

        update_network(
            G,
            time_step
        )

        # ----------------------------------------------------
        # DIJKSTRA
        # ----------------------------------------------------

        if algorithm == "Dijkstra":

            route = dijkstra_route(G)

            if route is None:
                continue

            diagnostic = {
                "candidate_count": 1,
                "selected_index": 0,
                "selected_rank": 1,
                "best_candidate_cost": route["path_cost"],
                "selected_base_cost": route["path_cost"],
                "mean_candidate_cost": route["path_cost"],
                "selected_tunneling_probability": 0.0,
                "mean_tunneling_probability": 0.0,
                "std_tunneling_probability": 0.0,
                "selection_probability": 1.0,
                "selected_thermal_weight": 1.0,
                "selected_final_score": 1.0,
                "candidate_cost_std": 0.0,
            }

        # ----------------------------------------------------
        # QT-APP
        # ----------------------------------------------------

        elif algorithm == "QT-APP":

            route, diagnostic = select_qt_route(
                G,
                energy,
                kappa,
                tunneling_enabled=True
            )

            if route is None:
                continue

        # ----------------------------------------------------
        # QT-APP WITHOUT ADAPTATION
        # ----------------------------------------------------

        elif algorithm == "QT-APP without adaptation":

            route, diagnostic = select_qt_route(
                G,
                INITIAL_ENERGY,
                INITIAL_KAPPA,
                tunneling_enabled=True
            )

            if route is None:
                continue

        # ----------------------------------------------------
        # QT-APP WITHOUT TUNNELING
        # ----------------------------------------------------

        elif algorithm == "QT-APP without tunneling":

            route, diagnostic = select_qt_route(
                G,
                energy,
                kappa,
                tunneling_enabled=False
            )

            if route is None:
                continue

        else:

            raise ValueError(
                f"Unknown algorithm: {algorithm}"
            )

        # ----------------------------------------------------
        # Route-change diagnostic
        # ----------------------------------------------------

        current_path = tuple(
            route["path"]
        )

        if previous_path is None:

            route_changed = False

        else:

            route_changed = (
                current_path != previous_path
            )

        # ----------------------------------------------------
        # Record metrics
        # ----------------------------------------------------

        record = {
            "seed": seed,
            "time_step": time_step,
            "algorithm": algorithm,
            "path_cost": route["path_cost"],
            "latency": route["latency"],
            "hops": route["hops"],
            "packet_loss": route["packet_loss"],
            "bandwidth": route["bandwidth"],
            "reliability": route["reliability"],
            "congestion": route["congestion"],
            "max_edge_barrier": route["max_edge_barrier"],
            "mean_edge_potential": route[
                "mean_edge_potential"
            ],
            "energy": energy,
            "kappa": kappa,
            "route_changed": int(route_changed),
            "path": "->".join(
                map(str, route["path"])
            ),
        }

        records.append(record)

        diagnostic_record = {
            "seed": seed,
            "time_step": time_step,
            "algorithm": algorithm,
            "energy": energy,
            "kappa": kappa,
            "route_changed": int(route_changed),
            **diagnostic,
        }

        diagnostics_records.append(
            diagnostic_record
        )

        # ----------------------------------------------------
        # Adaptation
        # ----------------------------------------------------

        if algorithm == "QT-APP":

            energy, kappa = adapt_parameters(
                energy,
                kappa,
                route,
                previous_route
            )

        elif algorithm == "QT-APP without tunneling":

            energy, kappa = adapt_parameters(
                energy,
                kappa,
                route,
                previous_route
            )

        # No adaptation for:
        #     Dijkstra
        #     QT-APP without adaptation

        previous_route = route
        previous_path = current_path

    return records, diagnostics_records


# ============================================================
# FULL EXPERIMENT
# ============================================================

def run_full_experiment():
    """
    Run all algorithms over all seeds.
    """

    algorithms = [
        "Dijkstra",
        "QT-APP",
        "QT-APP without adaptation",
        "QT-APP without tunneling",
    ]

    all_records = []
    all_diagnostics = []

    print("\n" + "=" * 70)
    print("QT-APP CORRECTED CONGESTION ANALYSIS")
    print("=" * 70)

    print("\nAlgorithms:")
    for algorithm in algorithms:
        print(f"  - {algorithm}")

    print(f"\nSeeds: {SEEDS}")
    print(f"Time steps: {TIME_STEPS}")
    print(
        f"Congestion period: "
        f"t={CONGESTION_START}..{CONGESTION_END}"
    )

    for algorithm in algorithms:

        print(
            f"\nRunning {algorithm}..."
        )

        for seed in SEEDS:

            records, diagnostics = run_algorithm(
                algorithm,
                seed
            )

            all_records.extend(records)
            all_diagnostics.extend(diagnostics)

            print(
                f"  seed={seed}: "
                f"{len(records)} time steps"
            )

    results = pd.DataFrame(
        all_records
    )

    diagnostics = pd.DataFrame(
        all_diagnostics
    )

    return results, diagnostics


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(results):
    """
    Calculate mean/std summary for every algorithm.
    """

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    summary = (
        results
        .groupby("algorithm")[metrics]
        .agg(["mean", "std"])
    )

    return summary


# ============================================================
# PERCENTAGE CHANGE VS DIJKSTRA
# ============================================================

def calculate_improvement_vs_dijkstra(
    results,
    period_name=None
):
    """
    Calculate percentage change relative to Dijkstra.

    Positive means the QT-APP value is higher than Dijkstra.

    For:
        path cost
        latency
        hops
        packet loss
        congestion

    lower is generally better.

    For:
        bandwidth
        reliability

    higher is generally better.
    """

    if period_name is not None:

        if period_name == "Congestion":

            data = results[
                results["time_step"].between(
                    CONGESTION_START,
                    CONGESTION_END
                )
            ].copy()

        elif period_name == "Normal":

            data = results[
                (
                    (results["time_step"] >= 0)
                    & (
                        results["time_step"]
                        < CONGESTION_START
                    )
                )
            ].copy()

        elif period_name == "Post-congestion":

            data = results[
                results["time_step"].between(
                    CONGESTION_END + 1,
                    RECOVERY_START - 1
                )
            ].copy()

        elif period_name == "Recovery":

            data = results[
                results["time_step"].between(
                    RECOVERY_START,
                    RECOVERY_END
                )
            ].copy()

        elif period_name == "Stable":

            data = results[
                results["time_step"] > RECOVERY_END
            ].copy()

        else:

            raise ValueError(
                f"Unknown period: {period_name}"
            )

    else:

        data = results.copy()

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    means = (
        data
        .groupby("algorithm")[metrics]
        .mean()
    )

    if "Dijkstra" not in means.index:

        return pd.DataFrame()

    baseline = means.loc[
        "Dijkstra"
    ]

    rows = []

    for algorithm in means.index:

        row = {
            "algorithm": algorithm
        }

        for metric in metrics:

            baseline_value = float(
                baseline[metric]
            )

            current_value = float(
                means.loc[
                    algorithm,
                    metric
                ]
            )

            if abs(baseline_value) < EPSILON:

                change = np.nan

            else:

                change = (
                    (
                        current_value
                        - baseline_value
                    )
                    / baseline_value
                ) * 100.0

            row[
                f"{metric}_change_percent"
            ] = change

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# DYNAMIC REGIMES
# ============================================================

def assign_regime(time_step):
    """
    Assign each time step to a dynamic regime.
    """

    if 0 <= time_step < CONGESTION_START:
        return "Normal"

    if (
        CONGESTION_START
        <= time_step
        <= CONGESTION_END
    ):
        return "Congestion"

    if (
        CONGESTION_END < time_step
        < RECOVERY_START
    ):
        return "Post-congestion"

    if (
        RECOVERY_START
        <= time_step
        <= RECOVERY_END
    ):
        return "Recovery"

    return "Stable"


def calculate_regime_summary(results):
    """
    Calculate mean/std metrics for each dynamic regime.
    """

    data = results.copy()

    data["regime"] = data[
        "time_step"
    ].apply(assign_regime)

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    summary = (
        data
        .groupby(
            ["regime", "algorithm"]
        )[metrics]
        .agg(["mean", "std"])
    )

    return data, summary


def calculate_regime_improvement(results):
    """
    Calculate percentage change against Dijkstra for every regime.
    """

    data, _ = calculate_regime_summary(
        results
    )

    regimes = [
        "Normal",
        "Congestion",
        "Post-congestion",
        "Recovery",
        "Stable",
    ]

    frames = []

    for regime in regimes:

        regime_data = data[
            data["regime"] == regime
        ]

        frame = calculate_improvement_vs_dijkstra(
            regime_data
        )

        if not frame.empty:

            frame.insert(
                0,
                "regime",
                regime
            )

            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_metric_over_time(
    results,
    metric,
    title,
    filename
):
    """
    Plot average metric over time.
    """

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in results[
        "algorithm"
    ].unique():

        subset = (
            results[
                results["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[metric]
            .mean()
        )

        plt.plot(
            subset.index,
            subset.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.axvspan(
        RECOVERY_START,
        RECOVERY_END,
        alpha=0.10,
        label="Recovery"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        metric.replace("_", " ").title()
    )

    plt.title(title)

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


def plot_regime_metric(
    results,
    metric,
    title,
    filename
):
    """
    Plot mean metric by dynamic regime.
    """

    data = results.copy()

    data["regime"] = data[
        "time_step"
    ].apply(assign_regime)

    regimes = [
        "Normal",
        "Congestion",
        "Post-congestion",
        "Recovery",
        "Stable",
    ]

    algorithms = list(
        data["algorithm"].unique()
    )

    x = np.arange(
        len(regimes)
    )

    width = 0.8 / max(
        len(algorithms),
        1
    )

    plt.figure(
        figsize=(12, 6)
    )

    for index, algorithm in enumerate(
        algorithms
    ):

        values = []

        for regime in regimes:

            subset = data[
                (
                    data["algorithm"]
                    == algorithm
                )
                & (
                    data["regime"]
                    == regime
                )
            ]

            values.append(
                subset[metric].mean()
                if not subset.empty
                else np.nan
            )

        plt.bar(
            x
            + (
                index
                - (len(algorithms) - 1) / 2
            ) * width,
            values,
            width,
            label=algorithm
        )

    plt.xticks(
        x,
        regimes,
        rotation=15
    )

    plt.ylabel(
        metric.replace("_", " ").title()
    )

    plt.title(title)

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# RELATIVE PERFORMANCE PLOT
# ============================================================

def plot_relative_congestion_performance(
    results
):
    """
    Plot QT-APP variants relative to Dijkstra during congestion.
    """

    improvement = (
        calculate_improvement_vs_dijkstra(
            results,
            "Congestion"
        )
    )

    if improvement.empty:
        return

    metrics = [
        "path_cost_change_percent",
        "latency_change_percent",
        "hops_change_percent",
        "packet_loss_change_percent",
        "bandwidth_change_percent",
        "reliability_change_percent",
        "congestion_change_percent",
    ]

    labels = [
        "Path cost",
        "Latency",
        "Hops",
        "Packet loss",
        "Bandwidth",
        "Reliability",
        "Congestion",
    ]

    plt.figure(
        figsize=(13, 7)
    )

    for _, row in improvement.iterrows():

        if row["algorithm"] == "Dijkstra":
            continue

        values = [
            row[metric]
            for metric in metrics
        ]

        plt.plot(
            labels,
            values,
            marker="o",
            label=row["algorithm"]
        )

    plt.axhline(
        0.0,
        linewidth=1
    )

    plt.ylabel(
        "Change vs Dijkstra (%)"
    )

    plt.title(
        "QT-APP Relative Performance During Congestion"
    )

    plt.xticks(
        rotation=20
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_congestion_relative_performance.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# ALGORITHM COMPARISON
# ============================================================

def plot_algorithm_comparison(
    results
):
    """
    Bar comparison during congestion.
    """

    congestion = results[
        results["time_step"].between(
            CONGESTION_START,
            CONGESTION_END
        )
    ]

    metrics = [
        "path_cost",
        "latency",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    labels = [
        "Path cost",
        "Latency",
        "Packet loss",
        "Bandwidth",
        "Reliability",
        "Congestion",
    ]

    algorithms = list(
        congestion["algorithm"].unique()
    )

    means = (
        congestion
        .groupby("algorithm")[metrics]
        .mean()
    )

    # Normalize each metric against the maximum
    # solely for visualization.
    normalized = means.copy()

    for metric in metrics:

        maximum = normalized[
            metric
        ].max()

        if maximum > EPSILON:

            normalized[
                metric
            ] = (
                normalized[metric]
                / maximum
            )

    x = np.arange(
        len(metrics)
    )

    width = 0.8 / max(
        len(algorithms),
        1
    )

    plt.figure(
        figsize=(13, 7)
    )

    for index, algorithm in enumerate(
        algorithms
    ):

        values = normalized.loc[
            algorithm,
            metrics
        ].values

        plt.bar(
            x
            + (
                index
                - (len(algorithms) - 1) / 2
            ) * width,
            values,
            width,
            label=algorithm
        )

    plt.xticks(
        x,
        labels,
        rotation=20
    )

    plt.ylabel(
        "Normalized value"
    )

    plt.title(
        "Algorithm Comparison During Congestion"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_algorithm_comparison.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# TUNNELING DIAGNOSTICS
# ============================================================

def plot_tunneling_probability(
    diagnostics
):
    """
    Plot tunneling probability over time.
    """

    subset = diagnostics[
        diagnostics["algorithm"]
        != "Dijkstra"
    ]

    if subset.empty:
        return

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in subset[
        "algorithm"
    ].unique():

        data = (
            subset[
                subset["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[
                "selected_tunneling_probability"
            ]
            .mean()
        )

        plt.plot(
            data.index,
            data.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Selected tunneling probability"
    )

    plt.title(
        "Selected Route Tunneling Probability"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_tunneling_probability.png"
        ),
        dpi=300
    )

    plt.close()


def plot_energy_kappa(
    diagnostics
):
    """
    Plot adaptive energy and kappa over time.

    Energy and kappa use separate figures so the scales are clear.
    """

    subset = diagnostics[
        diagnostics["algorithm"].isin(
            [
                "QT-APP",
                "QT-APP without tunneling",
            ]
        )
    ]

    if subset.empty:
        return

    # Energy
    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in subset[
        "algorithm"
    ].unique():

        data = (
            subset[
                subset["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[
                "energy"
            ]
            .mean()
        )

        plt.plot(
            data.index,
            data.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Energy"
    )

    plt.title(
        "Adaptive Energy"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_energy_kappa.png"
        ),
        dpi=300
    )

    plt.close()

    # Kappa
    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in subset[
        "algorithm"
    ].unique():

        data = (
            subset[
                subset["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[
                "kappa"
            ]
            .mean()
        )

        plt.plot(
            data.index,
            data.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Kappa"
    )

    plt.title(
        "Adaptive Kappa"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_energy_kappa.png"
        ),
        dpi=300
    )

    plt.close()


def plot_selected_rank(
    diagnostics
):
    """
    Plot selected candidate rank over time.

    Rank 1 means the lowest-base-cost candidate.
    Higher ranks indicate greater deviation from the
    ordinary best route.
    """

    subset = diagnostics[
        diagnostics["algorithm"]
        != "Dijkstra"
    ]

    if subset.empty:
        return

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in subset[
        "algorithm"
    ].unique():

        data = (
            subset[
                subset["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[
                "selected_rank"
            ]
            .mean()
        )

        plt.plot(
            data.index,
            data.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="Best candidate"
    )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Selected candidate rank"
    )

    plt.title(
        "Selected Candidate Rank"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_selected_rank.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    diagnostics
):
    """
    Save all CSV outputs.
    """

    # Main results
    results_path = os.path.join(
        RESULTS_DIR,
        "corrected_congestion_analysis_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    # Summary
    summary = summarize_results(
        results
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "corrected_congestion_period_summary.csv"
    )

    summary.to_csv(
        summary_path
    )

    # Overall improvement
    improvement = (
        calculate_improvement_vs_dijkstra(
            results
        )
    )

    improvement_path = os.path.join(
        RESULTS_DIR,
        "corrected_congestion_improvement_vs_dijkstra.csv"
    )

    improvement.to_csv(
        improvement_path,
        index=False
    )

    # Candidate diagnostics
    diagnostics_path = os.path.join(
        RESULTS_DIR,
        "corrected_candidate_diagnostics.csv"
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False
    )

    # Regime summary
    _, regime_summary = (
        calculate_regime_summary(
            results
        )
    )

    regime_summary_path = os.path.join(
        RESULTS_DIR,
        "corrected_regime_summary.csv"
    )

    regime_summary.to_csv(
        regime_summary_path
    )

    # Regime improvement
    regime_improvement = (
        calculate_regime_improvement(
            results
        )
    )

    regime_improvement_path = os.path.join(
        RESULTS_DIR,
        "corrected_regime_improvement_vs_dijkstra.csv"
    )

    regime_improvement.to_csv(
        regime_improvement_path,
        index=False
    )

    return {
        "results": results_path,
        "summary": summary_path,
        "improvement": improvement_path,
        "diagnostics": diagnostics_path,
        "regime_summary": regime_summary_path,
        "regime_improvement": regime_improvement_path,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_main_results(
    results
):
    """
    Print the key overall comparison.
    """

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    overall = (
        results
        .groupby("algorithm")[metrics]
        .mean()
    )

    print("\n")
    print("=" * 90)
    print("OVERALL MEAN PERFORMANCE")
    print("=" * 90)

    print(
        overall.round(4).to_string()
    )

    print("\n")
    print("=" * 90)
    print("PERCENTAGE CHANGE VS DIJKSTRA")
    print("=" * 90)

    improvement = (
        calculate_improvement_vs_dijkstra(
            results
        )
    )

    print(
        improvement.round(3).to_string(
            index=False
        )
    )


def print_congestion_results(
    results
):
    """
    Print congestion-period comparison.
    """

    congestion = results[
        results["time_step"].between(
            CONGESTION_START,
            CONGESTION_END
        )
    ]

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    means = (
        congestion
        .groupby("algorithm")[metrics]
        .mean()
    )

    print("\n")
    print("=" * 90)
    print(
        "CONGESTION PERIOD "
        f"(t={CONGESTION_START}..{CONGESTION_END})"
    )
    print("=" * 90)

    print(
        means.round(4).to_string()
    )

    print("\n")
    print("=" * 90)
    print("CONGESTION PERIOD CHANGE VS DIJKSTRA")
    print("=" * 90)

    improvement = (
        calculate_improvement_vs_dijkstra(
            results,
            "Congestion"
        )
    )

    print(
        improvement.round(3).to_string(
            index=False
        )
    )


def print_diagnostics_summary(
    diagnostics
):
    """
    Print diagnostic information that helps determine whether
    tunneling is actually influencing route selection.
    """

    if diagnostics.empty:
        return

    subset = diagnostics[
        diagnostics["algorithm"] != "Dijkstra"
    ]

    summary = (
        subset
        .groupby("algorithm")[
            [
                "candidate_count",
                "selected_rank",
                "selected_tunneling_probability",
                "mean_tunneling_probability",
                "std_tunneling_probability",
                "selection_probability",
                "route_changed",
            ]
        ]
        .mean()
    )

    print("\n")
    print("=" * 90)
    print("QT-APP SELECTION DIAGNOSTICS")
    print("=" * 90)

    print(
        summary.round(4).to_string()
    )

    print("\nInterpretation:")
    print(
        "  selected_rank = 1 means the lowest-base-cost "
        "candidate was selected."
    )
    print(
        "  selected_tunneling_probability shows the "
        "tunneling term actually used for the selected route."
    )
    print(
        "  std_tunneling_probability indicates whether "
        "tunneling probabilities differ across candidates."
    )
    print(
        "  route_changed indicates how often the selected "
        "path differs from the previous time step."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nStarting corrected QT-APP experiment...")

    results, diagnostics = (
        run_full_experiment()
    )

    if results.empty:

        print(
            "\nERROR: No simulation results were generated."
        )

        return

    # Save CSV files
    paths = save_results(
        results,
        diagnostics
    )

    # Create plots
    plot_metric_over_time(
        results,
        "path_cost",
        "Path Cost Over Time",
        "corrected_congestion_path_cost.png"
    )

    plot_metric_over_time(
        results,
        "latency",
        "Latency Over Time",
        "corrected_congestion_latency.png"
    )

    plot_metric_over_time(
        results,
        "packet_loss",
        "Packet Loss Over Time",
        "corrected_congestion_packet_loss.png"
    )

    plot_metric_over_time(
        results,
        "bandwidth",
        "Bandwidth Over Time",
        "corrected_congestion_bandwidth.png"
    )

    plot_metric_over_time(
        results,
        "reliability",
        "Reliability Over Time",
        "corrected_congestion_reliability.png"
    )

    plot_metric_over_time(
        results,
        "congestion",
        "Congestion Over Time",
        "corrected_congestion_congestion.png"
    )

    plot_metric_over_time(
        results,
        "hops",
        "Hop Count Over Time",
        "corrected_congestion_hops.png"
    )

    plot_regime_metric(
        results,
        "path_cost",
        "Path Cost by Dynamic Regime",
        "corrected_regime_path_cost.png"
    )

    plot_regime_metric(
        results,
        "latency",
        "Latency by Dynamic Regime",
        "corrected_regime_latency.png"
    )

    plot_regime_metric(
        results,
        "packet_loss",
        "Packet Loss by Dynamic Regime",
        "corrected_regime_packet_loss.png"
    )

    plot_regime_metric(
        results,
        "bandwidth",
        "Bandwidth by Dynamic Regime",
        "corrected_regime_bandwidth.png"
    )

    plot_regime_metric(
        results,
        "reliability",
        "Reliability by Dynamic Regime",
        "corrected_regime_reliability.png"
    )

    plot_regime_metric(
        results,
        "congestion",
        "Congestion by Dynamic Regime",
        "corrected_regime_congestion.png"
    )

    plot_regime_metric(
        results,
        "hops",
        "Hop Count by Dynamic Regime",
        "corrected_regime_hops.png"
    )

    plot_relative_congestion_performance(
        results
    )

    plot_algorithm_comparison(
        results
    )

    plot_tunneling_probability(
        diagnostics
    )

    plot_energy_kappa(
        diagnostics
    )

    plot_selected_rank(
        diagnostics
    )

    # Print results
    print_main_results(
        results
    )

    print_congestion_results(
        results
    )

    print_diagnostics_summary(
        diagnostics
    )

    print("\n")
    print("=" * 90)
    print("FILES CREATED")
    print("=" * 90)

    for name, path in paths.items():

        print(
            f"{name:25s}: {path}"
        )

    print("\nPlots created in:")
    print(
        f"  {os.path.abspath(RESULTS_DIR)}"
    )

    print("\nExperiment completed successfully.")
    print(
        "\nIMPORTANT: These simulation results are experimental "
        "evidence only; they do not by themselves establish "
        "patent novelty, inventive step, or patentability."
    )


if __name__ == "__main__":
    main()
    """
QT-APP Corrected Congestion Analysis
=====================================

Purpose
-------
Experimental simulation of Quantum-Tunneling-Based Adaptive Path Planning
(QT-APP) against Dijkstra under dynamic network conditions.

This version corrects several issues identified in the earlier implementation:

1. Path cost is additive rather than a mean of edge costs.
2. Hop penalty is explicitly included.
3. Dijkstra and QT-APP use the same base route-quality objective.
4. Tunneling is based on a route's maximum edge barrier, making it a
   distinct mechanism from ordinary thermal route selection.
5. Tunneling influence is bounded so that it cannot overwhelm route quality.
6. The "without tunneling" ablation genuinely removes the tunneling term.
7. The "without adaptation" ablation keeps tunneling but fixes energy/kappa.
8. Candidate selection and tunneling diagnostics are recorded.
9. Candidate/probability array lengths are guaranteed to match.
10. Dynamic-regime analysis is retained.

Outputs
-------
CSV:
    results/corrected_congestion_analysis_results.csv
    results/corrected_congestion_period_summary.csv
    results/corrected_congestion_improvement_vs_dijkstra.csv
    results/corrected_candidate_diagnostics.csv
    results/corrected_regime_summary.csv
    results/corrected_regime_improvement_vs_dijkstra.csv

PNG:
    results/corrected_congestion_path_cost.png
    results/corrected_congestion_latency.png
    results/corrected_congestion_packet_loss.png
    results/corrected_congestion_bandwidth.png
    results/corrected_congestion_reliability.png
    results/corrected_congestion_congestion.png
    results/corrected_congestion_hops.png
    results/corrected_regime_path_cost.png
    results/corrected_regime_latency.png
    results/corrected_regime_packet_loss.png
    results/corrected_regime_bandwidth.png
    results/corrected_regime_reliability.png
    results/corrected_regime_congestion.png
    results/corrected_regime_hops.png
    results/corrected_congestion_relative_performance.png
    results/corrected_algorithm_comparison.png
    results/corrected_tunneling_probability.png
    results/corrected_energy_kappa.png
    results/corrected_selected_rank.png

Requirements
------------
    pip install numpy pandas networkx matplotlib

Run
---
    python qt_app_congestion_corrected.py
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

NUM_NODES = 20
EDGE_PROBABILITY = 0.18

SOURCE = 0
DESTINATION = 19

TIME_STEPS = 40
SEEDS = [42, 43, 44, 45, 46]

# Dynamic congestion period
CONGESTION_START = 10
CONGESTION_END = 14

RECOVERY_START = 20
RECOVERY_END = 22

# Candidate routes
NUM_CANDIDATE_PATHS = 12

# QT-APP parameters
INITIAL_ENERGY = 0.45
INITIAL_KAPPA = 1.20

MIN_ENERGY = 0.05
MAX_ENERGY = 1.00

MIN_KAPPA = 0.20
MAX_KAPPA = 2.00

TEMPERATURE = 0.50

# Maximum contribution of tunneling to route-selection weight.
# score = thermal_weight * (1 + RHO * tunneling_probability)
RHO = 0.75

# Adaptation step sizes
ENERGY_UP_STEP = 0.035
ENERGY_DOWN_STEP = 0.020

KAPPA_DOWN_STEP = 0.025
KAPPA_UP_STEP = 0.020

# Stress thresholds
HIGH_STRESS_THRESHOLD = 0.55
LOW_STRESS_THRESHOLD = 0.35

HIGH_VOLATILITY_THRESHOLD = 0.15
LOW_VOLATILITY_THRESHOLD = 0.08

# Path-quality weights
W_LATENCY = 0.25
W_CONGESTION = 0.25
W_PACKET_LOSS = 0.15
W_BANDWIDTH = 0.15
W_RELIABILITY = 0.15
W_HOPS = 0.05

# Numerical safety
EPSILON = 1e-12


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# RANDOMNESS
# ============================================================

def set_seed(seed):
    """Set both Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# NETWORK CREATION
# ============================================================

def create_dynamic_network(seed):
    """
    Create the initial random network.

    Each undirected edge contains:
        latency
        congestion
        packet_loss
        bandwidth
        reliability
    """

    set_seed(seed)

    G = nx.Graph()

    for node in range(NUM_NODES):
        G.add_node(node)

    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):

            if np.random.random() < EDGE_PROBABILITY:

                latency = np.random.uniform(10.0, 50.0)
                congestion = np.random.uniform(0.10, 0.50)
                packet_loss = np.random.uniform(0.01, 0.08)
                bandwidth = np.random.uniform(20.0, 100.0)
                reliability = np.random.uniform(0.75, 0.99)

                G.add_edge(
                    i,
                    j,
                    latency=latency,
                    congestion=congestion,
                    packet_loss=packet_loss,
                    bandwidth=bandwidth,
                    reliability=reliability,
                )

    # Guarantee connectivity if the random graph happens to be disconnected.
    if not nx.is_connected(G):

        components = list(nx.connected_components(G))

        for index in range(len(components) - 1):

            a = list(components[index])[0]
            b = list(components[index + 1])[0]

            G.add_edge(
                a,
                b,
                latency=np.random.uniform(10.0, 50.0),
                congestion=np.random.uniform(0.10, 0.50),
                packet_loss=np.random.uniform(0.01, 0.08),
                bandwidth=np.random.uniform(20.0, 100.0),
                reliability=np.random.uniform(0.75, 0.99),
            )

    return G


# ============================================================
# DYNAMIC NETWORK UPDATE
# ============================================================

def update_network(G, time_step):
    """
    Apply dynamic changes to the network.

    Congestion period:
        t = 10 ... 14

    Recovery:
        t = 20 ... 22
    """

    for u, v, data in G.edges(data=True):

        # Small ordinary random movement
        latency_change = np.random.normal(0.0, 1.5)
        congestion_change = np.random.normal(0.0, 0.025)
        loss_change = np.random.normal(0.0, 0.004)
        bandwidth_change = np.random.normal(0.0, 3.0)
        reliability_change = np.random.normal(0.0, 0.008)

        # Artificial congestion event
        if CONGESTION_START <= time_step <= CONGESTION_END:

            latency_change += np.random.uniform(5.0, 15.0)
            congestion_change += np.random.uniform(0.15, 0.35)
            loss_change += np.random.uniform(0.02, 0.05)
            bandwidth_change -= np.random.uniform(5.0, 20.0)
            reliability_change -= np.random.uniform(0.03, 0.08)

        # Recovery after congestion
        elif RECOVERY_START <= time_step <= RECOVERY_END:

            latency_change -= np.random.uniform(2.0, 6.0)
            congestion_change -= np.random.uniform(0.05, 0.15)
            loss_change -= np.random.uniform(0.005, 0.015)
            bandwidth_change += np.random.uniform(3.0, 10.0)
            reliability_change += np.random.uniform(0.01, 0.03)

        data["latency"] = max(
            5.0,
            data["latency"] + latency_change
        )

        data["congestion"] = np.clip(
            data["congestion"] + congestion_change,
            0.0,
            1.0
        )

        data["packet_loss"] = np.clip(
            data["packet_loss"] + loss_change,
            0.0,
            0.50
        )

        data["bandwidth"] = np.clip(
            data["bandwidth"] + bandwidth_change,
            5.0,
            150.0
        )

        data["reliability"] = np.clip(
            data["reliability"] + reliability_change,
            0.40,
            0.999
        )


# ============================================================
# NORMALIZATION
# ============================================================

def calculate_edge_normalization(G):
    """
    Calculate per-network-state min/max values for normalization.
    """

    attributes = [
        "latency",
        "congestion",
        "packet_loss",
        "bandwidth",
        "reliability",
    ]

    normalization = {}

    for attribute in attributes:

        values = [
            data[attribute]
            for _, _, data in G.edges(data=True)
        ]

        if not values:
            normalization[attribute] = (0.0, 1.0)
            continue

        minimum = min(values)
        maximum = max(values)

        if abs(maximum - minimum) < EPSILON:
            maximum = minimum + 1.0

        normalization[attribute] = (minimum, maximum)

    return normalization


def normalize_value(value, minimum, maximum):
    """Normalize to [0, 1]."""
    return np.clip(
        (value - minimum) / (maximum - minimum),
        0.0,
        1.0
    )


# ============================================================
# EDGE COST
# ============================================================

def calculate_edge_potential(G, u, v, normalization):
    """
    Calculate the normalized base route-quality cost for one edge.

    Lower is better.

    Components:
        latency
        congestion
        packet loss
        inverse bandwidth
        inverse reliability
        hop penalty
    """

    data = G[u][v]

    latency_norm = normalize_value(
        data["latency"],
        *normalization["latency"]
    )

    congestion_norm = normalize_value(
        data["congestion"],
        *normalization["congestion"]
    )

    loss_norm = normalize_value(
        data["packet_loss"],
        *normalization["packet_loss"]
    )

    bandwidth_norm = normalize_value(
        data["bandwidth"],
        *normalization["bandwidth"]
    )

    reliability_norm = normalize_value(
        data["reliability"],
        *normalization["reliability"]
    )

    inverse_bandwidth = 1.0 - bandwidth_norm
    inverse_reliability = 1.0 - reliability_norm

    potential = (
        W_LATENCY * latency_norm
        + W_CONGESTION * congestion_norm
        + W_PACKET_LOSS * loss_norm
        + W_BANDWIDTH * inverse_bandwidth
        + W_RELIABILITY * inverse_reliability
        + W_HOPS
    )

    return float(potential)


# ============================================================
# PATH METRICS
# ============================================================

def calculate_path_metrics(G, path, normalization):
    """
    Calculate complete metrics for a route.

    IMPORTANT:
    path_cost is the SUM of edge potentials, not their mean.
    """

    if path is None or len(path) < 2:
        return None

    edge_potentials = []

    latencies = []
    congestions = []
    packet_losses = []
    bandwidths = []
    reliabilities = []

    for u, v in zip(path[:-1], path[1:]):

        data = G[u][v]

        potential = calculate_edge_potential(
            G,
            u,
            v,
            normalization
        )

        edge_potentials.append(potential)

        latencies.append(data["latency"])
        congestions.append(data["congestion"])
        packet_losses.append(data["packet_loss"])
        bandwidths.append(data["bandwidth"])
        reliabilities.append(data["reliability"])

    hops = len(path) - 1

    # Additive route cost.
    path_cost = float(np.sum(edge_potentials))

    # Total latency is additive.
    latency = float(np.sum(latencies))

    # Average route-level conditions.
    congestion = float(np.mean(congestions))
    packet_loss = float(np.mean(packet_losses))
    bandwidth = float(np.mean(bandwidths))
    reliability = float(np.mean(reliabilities))

    # Maximum edge potential is used as the quantum barrier.
    max_edge_barrier = float(np.max(edge_potentials))

    # Mean edge potential is useful diagnostically.
    mean_edge_potential = float(np.mean(edge_potentials))

    return {
        "path": list(path),
        "path_cost": path_cost,
        "latency": latency,
        "hops": hops,
        "packet_loss": packet_loss,
        "bandwidth": bandwidth,
        "reliability": reliability,
        "congestion": congestion,
        "max_edge_barrier": max_edge_barrier,
        "mean_edge_potential": mean_edge_potential,
    }


# ============================================================
# DIJKSTRA ROUTE
# ============================================================

def dijkstra_route(G):
    """
    Dijkstra baseline using exactly the same base edge potential
    used by QT-APP.
    """

    normalization = calculate_edge_normalization(G)

    for u, v, data in G.edges(data=True):

        data["base_cost"] = calculate_edge_potential(
            G,
            u,
            v,
            normalization
        )

    try:
        path = nx.shortest_path(
            G,
            SOURCE,
            DESTINATION,
            weight="base_cost"
        )
    except nx.NetworkXNoPath:
        return None

    return calculate_path_metrics(
        G,
        path,
        normalization
    )


# ============================================================
# CANDIDATE PATH GENERATION
# ============================================================

def generate_candidate_paths(G):
    """
    Generate multiple high-quality simple paths using the common
    base route-quality objective.
    """

    normalization = calculate_edge_normalization(G)

    for u, v, data in G.edges(data=True):

        data["candidate_cost"] = calculate_edge_potential(
            G,
            u,
            v,
            normalization
        )

    try:

        generator = nx.shortest_simple_paths(
            G,
            SOURCE,
            DESTINATION,
            weight="candidate_cost"
        )

        candidates = []

        for path in generator:

            metrics = calculate_path_metrics(
                G,
                path,
                normalization
            )

            if metrics is not None:
                candidates.append(metrics)

            if len(candidates) >= NUM_CANDIDATE_PATHS:
                break

    except nx.NetworkXNoPath:
        return []

    return candidates


# ============================================================
# TUNNELING PROBABILITY
# ============================================================

def calculate_tunneling_probability(
    candidate,
    best_candidate_cost,
    energy,
    kappa
):
    """
    Quantum-inspired tunneling probability.

    The barrier is based on the route's maximum edge potential.

    The energy level is scaled relative to the best candidate route.

    This makes tunneling distinct from ordinary thermal selection:
    a route can have a somewhat higher total cost while possessing a
    lower maximum barrier.

    P_tunnel = exp(-2 * kappa * effective_barrier)

    where effective_barrier is the positive difference between the
    route's maximum edge barrier and the adaptive energy threshold.
    """

    best_cost = max(best_candidate_cost, EPSILON)

    # Relative route cost factor.
    relative_cost = candidate["path_cost"] / best_cost

    # The energy threshold is related to route-quality scale.
    #
    # Energy in [0, 1].
    # At low energy, only routes with relatively small barriers tunnel.
    # At high energy, more candidate barriers become accessible.
    energy_threshold = (
        0.25
        + 0.50 * energy
    )

    # Convert barrier into a relative scale.
    barrier = (
        candidate["max_edge_barrier"]
        / max(candidate["mean_edge_potential"], EPSILON)
    )

    # Small route-cost correction prevents very expensive paths from
    # receiving the same tunneling probability merely because they have
    # a favorable maximum-edge barrier.
    cost_penalty = max(
        0.0,
        relative_cost - 1.0
    ) * 0.15

    effective_barrier = max(
        0.0,
        barrier - energy_threshold + cost_penalty
    )

    probability = np.exp(
        -2.0 * kappa * effective_barrier
    )

    return float(
        np.clip(probability, 0.0, 1.0)
    )


# ============================================================
# THERMAL ROUTE SELECTION
# ============================================================

def calculate_selection_probabilities(
    candidates,
    tunneling_enabled,
    energy,
    kappa
):
    """
    Calculate route-selection probabilities.

    Base:
        thermal_weight = exp(-cost / T)

    With tunneling:
        score = thermal_weight * (1 + RHO * P_tunnel)

    Without tunneling:
        score = thermal_weight

    The tunneling multiplier is deliberately bounded between:
        1 and 1 + RHO

    This prevents tunneling from completely overpowering route quality.
    """

    if not candidates:
        return [], []

    costs = np.array(
        [candidate["path_cost"] for candidate in candidates],
        dtype=float
    )

    best_cost = float(np.min(costs))

    # Stabilize the thermal calculation by subtracting best cost.
    relative_costs = costs - best_cost

    thermal_weights = np.exp(
        -relative_costs / max(TEMPERATURE, EPSILON)
    )

    tunneling_values = []

    if tunneling_enabled:

        for candidate in candidates:

            probability = calculate_tunneling_probability(
                candidate,
                best_cost,
                energy,
                kappa
            )

            tunneling_values.append(probability)

        tunneling_values = np.array(
            tunneling_values,
            dtype=float
        )

        scores = (
            thermal_weights
            * (
                1.0
                + RHO * tunneling_values
            )
        )

    else:

        tunneling_values = np.zeros(
            len(candidates),
            dtype=float
        )

        scores = thermal_weights.copy()

    score_sum = float(np.sum(scores))

    if score_sum <= EPSILON:

        probabilities = np.ones(
            len(candidates),
            dtype=float
        ) / len(candidates)

    else:

        probabilities = (
            scores / score_sum
        )

    return (
        probabilities.tolist(),
        tunneling_values.tolist()
    )


# ============================================================
# ROUTE SELECTION
# ============================================================

def select_qt_route(
    G,
    energy,
    kappa,
    tunneling_enabled=True
):
    """
    Select one route using QT-APP logic.

    Returns:
        selected route metrics
        diagnostics dictionary
    """

    candidates = generate_candidate_paths(G)

    if not candidates:
        return None, None

    probabilities, tunneling_values = (
        calculate_selection_probabilities(
            candidates,
            tunneling_enabled,
            energy,
            kappa
        )
    )

    # Guarantee equal lengths.
    assert len(candidates) == len(probabilities)
    assert len(candidates) == len(tunneling_values)

    selected_index = int(
        np.random.choice(
            len(candidates),
            p=np.array(probabilities)
        )
    )

    selected = candidates[selected_index]

    sorted_indices = np.argsort(
        [
            candidate["path_cost"]
            for candidate in candidates
        ]
    )

    selected_rank = int(
        np.where(
            sorted_indices == selected_index
        )[0][0]
    ) + 1

    # Base thermal weight for diagnostics.
    candidate_costs = np.array(
        [candidate["path_cost"] for candidate in candidates]
    )

    best_cost = float(np.min(candidate_costs))

    relative_costs = candidate_costs - best_cost

    thermal_weights = np.exp(
        -relative_costs / max(TEMPERATURE, EPSILON)
    )

    if tunneling_enabled:

        scores = (
            thermal_weights
            * (
                1.0
                + RHO * np.array(tunneling_values)
            )
        )

    else:

        scores = thermal_weights

    diagnostics = {
        "candidate_count": len(candidates),
        "selected_index": selected_index,
        "selected_rank": selected_rank,
        "best_candidate_cost": best_cost,
        "selected_base_cost": selected["path_cost"],
        "mean_candidate_cost": float(
            np.mean(candidate_costs)
        ),
        "selected_tunneling_probability": float(
            tunneling_values[selected_index]
        ),
        "mean_tunneling_probability": float(
            np.mean(tunneling_values)
        ),
        "std_tunneling_probability": float(
            np.std(tunneling_values)
        ),
        "selection_probability": float(
            probabilities[selected_index]
        ),
        "selected_thermal_weight": float(
            thermal_weights[selected_index]
        ),
        "selected_final_score": float(
            scores[selected_index]
        ),
        "candidate_cost_std": float(
            np.std(candidate_costs)
        ),
    }

    return selected, diagnostics


# ============================================================
# ADAPTATION
# ============================================================

def calculate_route_stress(route):
    """
    Calculate a normalized stress score.

    Higher = more network stress.
    """

    if route is None:
        return 1.0

    stress = (
        0.50 * route["congestion"]
        + 0.30 * route["packet_loss"]
        + 0.20 * (1.0 - route["reliability"])
    )

    return float(
        np.clip(stress, 0.0, 1.0)
    )


def adapt_parameters(
    energy,
    kappa,
    current_route,
    previous_route
):
    """
    Adapt energy and kappa based on network stress and route volatility.

    High stress / high volatility:
        increase energy
        decrease kappa

    Low stress / low volatility:
        decrease energy
        increase kappa

    This allows tunneling to become more permissive during instability
    without directly replacing the route-quality objective.
    """

    if current_route is None:
        return energy, kappa

    current_stress = calculate_route_stress(
        current_route
    )

    if previous_route is None:

        volatility = 0.0

    else:

        previous_cost = max(
            previous_route["path_cost"],
            EPSILON
        )

        volatility = abs(
            current_route["path_cost"]
            - previous_route["path_cost"]
        ) / previous_cost

    high_stress = (
        current_stress > HIGH_STRESS_THRESHOLD
    )

    high_volatility = (
        volatility > HIGH_VOLATILITY_THRESHOLD
    )

    low_stress = (
        current_stress < LOW_STRESS_THRESHOLD
    )

    low_volatility = (
        volatility < LOW_VOLATILITY_THRESHOLD
    )

    if high_stress or high_volatility:

        energy += ENERGY_UP_STEP
        kappa -= KAPPA_DOWN_STEP

    elif low_stress and low_volatility:

        energy -= ENERGY_DOWN_STEP
        kappa += KAPPA_UP_STEP

    energy = float(
        np.clip(
            energy,
            MIN_ENERGY,
            MAX_ENERGY
        )
    )

    kappa = float(
        np.clip(
            kappa,
            MIN_KAPPA,
            MAX_KAPPA
        )
    )

    return energy, kappa


# ============================================================
# SINGLE ALGORITHM RUN
# ============================================================

def run_algorithm(
    algorithm,
    seed
):
    """
    Run one algorithm on one dynamic network realization.

    Algorithms:
        Dijkstra
        QT-APP
        QT-APP without adaptation
        QT-APP without tunneling
    """

    set_seed(seed)

    G = create_dynamic_network(seed)

    records = []
    diagnostics_records = []

    energy = INITIAL_ENERGY
    kappa = INITIAL_KAPPA

    previous_route = None

    previous_path = None

    for time_step in range(TIME_STEPS):

        update_network(
            G,
            time_step
        )

        # ----------------------------------------------------
        # DIJKSTRA
        # ----------------------------------------------------

        if algorithm == "Dijkstra":

            route = dijkstra_route(G)

            if route is None:
                continue

            diagnostic = {
                "candidate_count": 1,
                "selected_index": 0,
                "selected_rank": 1,
                "best_candidate_cost": route["path_cost"],
                "selected_base_cost": route["path_cost"],
                "mean_candidate_cost": route["path_cost"],
                "selected_tunneling_probability": 0.0,
                "mean_tunneling_probability": 0.0,
                "std_tunneling_probability": 0.0,
                "selection_probability": 1.0,
                "selected_thermal_weight": 1.0,
                "selected_final_score": 1.0,
                "candidate_cost_std": 0.0,
            }

        # ----------------------------------------------------
        # QT-APP
        # ----------------------------------------------------

        elif algorithm == "QT-APP":

            route, diagnostic = select_qt_route(
                G,
                energy,
                kappa,
                tunneling_enabled=True
            )

            if route is None:
                continue

        # ----------------------------------------------------
        # QT-APP WITHOUT ADAPTATION
        # ----------------------------------------------------

        elif algorithm == "QT-APP without adaptation":

            route, diagnostic = select_qt_route(
                G,
                INITIAL_ENERGY,
                INITIAL_KAPPA,
                tunneling_enabled=True
            )

            if route is None:
                continue

        # ----------------------------------------------------
        # QT-APP WITHOUT TUNNELING
        # ----------------------------------------------------

        elif algorithm == "QT-APP without tunneling":

            route, diagnostic = select_qt_route(
                G,
                energy,
                kappa,
                tunneling_enabled=False
            )

            if route is None:
                continue

        else:

            raise ValueError(
                f"Unknown algorithm: {algorithm}"
            )

        # ----------------------------------------------------
        # Route-change diagnostic
        # ----------------------------------------------------

        current_path = tuple(
            route["path"]
        )

        if previous_path is None:

            route_changed = False

        else:

            route_changed = (
                current_path != previous_path
            )

        # ----------------------------------------------------
        # Record metrics
        # ----------------------------------------------------

        record = {
            "seed": seed,
            "time_step": time_step,
            "algorithm": algorithm,
            "path_cost": route["path_cost"],
            "latency": route["latency"],
            "hops": route["hops"],
            "packet_loss": route["packet_loss"],
            "bandwidth": route["bandwidth"],
            "reliability": route["reliability"],
            "congestion": route["congestion"],
            "max_edge_barrier": route["max_edge_barrier"],
            "mean_edge_potential": route[
                "mean_edge_potential"
            ],
            "energy": energy,
            "kappa": kappa,
            "route_changed": int(route_changed),
            "path": "->".join(
                map(str, route["path"])
            ),
        }

        records.append(record)

        diagnostic_record = {
            "seed": seed,
            "time_step": time_step,
            "algorithm": algorithm,
            "energy": energy,
            "kappa": kappa,
            "route_changed": int(route_changed),
            **diagnostic,
        }

        diagnostics_records.append(
            diagnostic_record
        )

        # ----------------------------------------------------
        # Adaptation
        # ----------------------------------------------------

        if algorithm == "QT-APP":

            energy, kappa = adapt_parameters(
                energy,
                kappa,
                route,
                previous_route
            )

        elif algorithm == "QT-APP without tunneling":

            energy, kappa = adapt_parameters(
                energy,
                kappa,
                route,
                previous_route
            )

        # No adaptation for:
        #     Dijkstra
        #     QT-APP without adaptation

        previous_route = route
        previous_path = current_path

    return records, diagnostics_records


# ============================================================
# FULL EXPERIMENT
# ============================================================

def run_full_experiment():
    """
    Run all algorithms over all seeds.
    """

    algorithms = [
        "Dijkstra",
        "QT-APP",
        "QT-APP without adaptation",
        "QT-APP without tunneling",
    ]

    all_records = []
    all_diagnostics = []

    print("\n" + "=" * 70)
    print("QT-APP CORRECTED CONGESTION ANALYSIS")
    print("=" * 70)

    print("\nAlgorithms:")
    for algorithm in algorithms:
        print(f"  - {algorithm}")

    print(f"\nSeeds: {SEEDS}")
    print(f"Time steps: {TIME_STEPS}")
    print(
        f"Congestion period: "
        f"t={CONGESTION_START}..{CONGESTION_END}"
    )

    for algorithm in algorithms:

        print(
            f"\nRunning {algorithm}..."
        )

        for seed in SEEDS:

            records, diagnostics = run_algorithm(
                algorithm,
                seed
            )

            all_records.extend(records)
            all_diagnostics.extend(diagnostics)

            print(
                f"  seed={seed}: "
                f"{len(records)} time steps"
            )

    results = pd.DataFrame(
        all_records
    )

    diagnostics = pd.DataFrame(
        all_diagnostics
    )

    return results, diagnostics


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(results):
    """
    Calculate mean/std summary for every algorithm.
    """

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    summary = (
        results
        .groupby("algorithm")[metrics]
        .agg(["mean", "std"])
    )

    return summary


# ============================================================
# PERCENTAGE CHANGE VS DIJKSTRA
# ============================================================

def calculate_improvement_vs_dijkstra(
    results,
    period_name=None
):
    """
    Calculate percentage change relative to Dijkstra.

    Positive means the QT-APP value is higher than Dijkstra.

    For:
        path cost
        latency
        hops
        packet loss
        congestion

    lower is generally better.

    For:
        bandwidth
        reliability

    higher is generally better.
    """

    if period_name is not None:

        if period_name == "Congestion":

            data = results[
                results["time_step"].between(
                    CONGESTION_START,
                    CONGESTION_END
                )
            ].copy()

        elif period_name == "Normal":

            data = results[
                (
                    (results["time_step"] >= 0)
                    & (
                        results["time_step"]
                        < CONGESTION_START
                    )
                )
            ].copy()

        elif period_name == "Post-congestion":

            data = results[
                results["time_step"].between(
                    CONGESTION_END + 1,
                    RECOVERY_START - 1
                )
            ].copy()

        elif period_name == "Recovery":

            data = results[
                results["time_step"].between(
                    RECOVERY_START,
                    RECOVERY_END
                )
            ].copy()

        elif period_name == "Stable":

            data = results[
                results["time_step"] > RECOVERY_END
            ].copy()

        else:

            raise ValueError(
                f"Unknown period: {period_name}"
            )

    else:

        data = results.copy()

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    means = (
        data
        .groupby("algorithm")[metrics]
        .mean()
    )

    if "Dijkstra" not in means.index:

        return pd.DataFrame()

    baseline = means.loc[
        "Dijkstra"
    ]

    rows = []

    for algorithm in means.index:

        row = {
            "algorithm": algorithm
        }

        for metric in metrics:

            baseline_value = float(
                baseline[metric]
            )

            current_value = float(
                means.loc[
                    algorithm,
                    metric
                ]
            )

            if abs(baseline_value) < EPSILON:

                change = np.nan

            else:

                change = (
                    (
                        current_value
                        - baseline_value
                    )
                    / baseline_value
                ) * 100.0

            row[
                f"{metric}_change_percent"
            ] = change

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# DYNAMIC REGIMES
# ============================================================

def assign_regime(time_step):
    """
    Assign each time step to a dynamic regime.
    """

    if 0 <= time_step < CONGESTION_START:
        return "Normal"

    if (
        CONGESTION_START
        <= time_step
        <= CONGESTION_END
    ):
        return "Congestion"

    if (
        CONGESTION_END < time_step
        < RECOVERY_START
    ):
        return "Post-congestion"

    if (
        RECOVERY_START
        <= time_step
        <= RECOVERY_END
    ):
        return "Recovery"

    return "Stable"


def calculate_regime_summary(results):
    """
    Calculate mean/std metrics for each dynamic regime.
    """

    data = results.copy()

    data["regime"] = data[
        "time_step"
    ].apply(assign_regime)

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    summary = (
        data
        .groupby(
            ["regime", "algorithm"]
        )[metrics]
        .agg(["mean", "std"])
    )

    return data, summary


def calculate_regime_improvement(results):
    """
    Calculate percentage change against Dijkstra for every regime.
    """

    data, _ = calculate_regime_summary(
        results
    )

    regimes = [
        "Normal",
        "Congestion",
        "Post-congestion",
        "Recovery",
        "Stable",
    ]

    frames = []

    for regime in regimes:

        regime_data = data[
            data["regime"] == regime
        ]

        frame = calculate_improvement_vs_dijkstra(
            regime_data
        )

        if not frame.empty:

            frame.insert(
                0,
                "regime",
                regime
            )

            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# PLOTTING HELPERS
# ============================================================

def plot_metric_over_time(
    results,
    metric,
    title,
    filename
):
    """
    Plot average metric over time.
    """

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in results[
        "algorithm"
    ].unique():

        subset = (
            results[
                results["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[metric]
            .mean()
        )

        plt.plot(
            subset.index,
            subset.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.axvspan(
        RECOVERY_START,
        RECOVERY_END,
        alpha=0.10,
        label="Recovery"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        metric.replace("_", " ").title()
    )

    plt.title(title)

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


def plot_regime_metric(
    results,
    metric,
    title,
    filename
):
    """
    Plot mean metric by dynamic regime.
    """

    data = results.copy()

    data["regime"] = data[
        "time_step"
    ].apply(assign_regime)

    regimes = [
        "Normal",
        "Congestion",
        "Post-congestion",
        "Recovery",
        "Stable",
    ]

    algorithms = list(
        data["algorithm"].unique()
    )

    x = np.arange(
        len(regimes)
    )

    width = 0.8 / max(
        len(algorithms),
        1
    )

    plt.figure(
        figsize=(12, 6)
    )

    for index, algorithm in enumerate(
        algorithms
    ):

        values = []

        for regime in regimes:

            subset = data[
                (
                    data["algorithm"]
                    == algorithm
                )
                & (
                    data["regime"]
                    == regime
                )
            ]

            values.append(
                subset[metric].mean()
                if not subset.empty
                else np.nan
            )

        plt.bar(
            x
            + (
                index
                - (len(algorithms) - 1) / 2
            ) * width,
            values,
            width,
            label=algorithm
        )

    plt.xticks(
        x,
        regimes,
        rotation=15
    )

    plt.ylabel(
        metric.replace("_", " ").title()
    )

    plt.title(title)

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# RELATIVE PERFORMANCE PLOT
# ============================================================

def plot_relative_congestion_performance(
    results
):
    """
    Plot QT-APP variants relative to Dijkstra during congestion.
    """

    improvement = (
        calculate_improvement_vs_dijkstra(
            results,
            "Congestion"
        )
    )

    if improvement.empty:
        return

    metrics = [
        "path_cost_change_percent",
        "latency_change_percent",
        "hops_change_percent",
        "packet_loss_change_percent",
        "bandwidth_change_percent",
        "reliability_change_percent",
        "congestion_change_percent",
    ]

    labels = [
        "Path cost",
        "Latency",
        "Hops",
        "Packet loss",
        "Bandwidth",
        "Reliability",
        "Congestion",
    ]

    plt.figure(
        figsize=(13, 7)
    )

    for _, row in improvement.iterrows():

        if row["algorithm"] == "Dijkstra":
            continue

        values = [
            row[metric]
            for metric in metrics
        ]

        plt.plot(
            labels,
            values,
            marker="o",
            label=row["algorithm"]
        )

    plt.axhline(
        0.0,
        linewidth=1
    )

    plt.ylabel(
        "Change vs Dijkstra (%)"
    )

    plt.title(
        "QT-APP Relative Performance During Congestion"
    )

    plt.xticks(
        rotation=20
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_congestion_relative_performance.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# ALGORITHM COMPARISON
# ============================================================

def plot_algorithm_comparison(
    results
):
    """
    Bar comparison during congestion.
    """

    congestion = results[
        results["time_step"].between(
            CONGESTION_START,
            CONGESTION_END
        )
    ]

    metrics = [
        "path_cost",
        "latency",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    labels = [
        "Path cost",
        "Latency",
        "Packet loss",
        "Bandwidth",
        "Reliability",
        "Congestion",
    ]

    algorithms = list(
        congestion["algorithm"].unique()
    )

    means = (
        congestion
        .groupby("algorithm")[metrics]
        .mean()
    )

    # Normalize each metric against the maximum
    # solely for visualization.
    normalized = means.copy()

    for metric in metrics:

        maximum = normalized[
            metric
        ].max()

        if maximum > EPSILON:

            normalized[
                metric
            ] = (
                normalized[metric]
                / maximum
            )

    x = np.arange(
        len(metrics)
    )

    width = 0.8 / max(
        len(algorithms),
        1
    )

    plt.figure(
        figsize=(13, 7)
    )

    for index, algorithm in enumerate(
        algorithms
    ):

        values = normalized.loc[
            algorithm,
            metrics
        ].values

        plt.bar(
            x
            + (
                index
                - (len(algorithms) - 1) / 2
            ) * width,
            values,
            width,
            label=algorithm
        )

    plt.xticks(
        x,
        labels,
        rotation=20
    )

    plt.ylabel(
        "Normalized value"
    )

    plt.title(
        "Algorithm Comparison During Congestion"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_algorithm_comparison.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# TUNNELING DIAGNOSTICS
# ============================================================

def plot_tunneling_probability(
    diagnostics
):
    """
    Plot tunneling probability over time.
    """

    subset = diagnostics[
        diagnostics["algorithm"]
        != "Dijkstra"
    ]

    if subset.empty:
        return

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in subset[
        "algorithm"
    ].unique():

        data = (
            subset[
                subset["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[
                "selected_tunneling_probability"
            ]
            .mean()
        )

        plt.plot(
            data.index,
            data.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Selected tunneling probability"
    )

    plt.title(
        "Selected Route Tunneling Probability"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_tunneling_probability.png"
        ),
        dpi=300
    )

    plt.close()


def plot_energy_kappa(
    diagnostics
):
    """
    Plot adaptive energy and kappa over time.

    Energy and kappa use separate figures so the scales are clear.
    """

    subset = diagnostics[
        diagnostics["algorithm"].isin(
            [
                "QT-APP",
                "QT-APP without tunneling",
            ]
        )
    ]

    if subset.empty:
        return

    # Energy
    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in subset[
        "algorithm"
    ].unique():

        data = (
            subset[
                subset["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[
                "energy"
            ]
            .mean()
        )

        plt.plot(
            data.index,
            data.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Energy"
    )

    plt.title(
        "Adaptive Energy"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_energy_kappa.png"
        ),
        dpi=300
    )

    plt.close()

    # Kappa
    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in subset[
        "algorithm"
    ].unique():

        data = (
            subset[
                subset["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[
                "kappa"
            ]
            .mean()
        )

        plt.plot(
            data.index,
            data.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Kappa"
    )

    plt.title(
        "Adaptive Kappa"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_energy_kappa.png"
        ),
        dpi=300
    )

    plt.close()


def plot_selected_rank(
    diagnostics
):
    """
    Plot selected candidate rank over time.

    Rank 1 means the lowest-base-cost candidate.
    Higher ranks indicate greater deviation from the
    ordinary best route.
    """

    subset = diagnostics[
        diagnostics["algorithm"]
        != "Dijkstra"
    ]

    if subset.empty:
        return

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm in subset[
        "algorithm"
    ].unique():

        data = (
            subset[
                subset["algorithm"]
                == algorithm
            ]
            .groupby("time_step")[
                "selected_rank"
            ]
            .mean()
        )

        plt.plot(
            data.index,
            data.values,
            marker="o",
            markersize=3,
            label=algorithm
        )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="Best candidate"
    )

    plt.axvspan(
        CONGESTION_START,
        CONGESTION_END,
        alpha=0.15,
        label="Congestion"
    )

    plt.xlabel(
        "Time step"
    )

    plt.ylabel(
        "Selected candidate rank"
    )

    plt.title(
        "Selected Candidate Rank"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "corrected_selected_rank.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    diagnostics
):
    """
    Save all CSV outputs.
    """

    # Main results
    results_path = os.path.join(
        RESULTS_DIR,
        "corrected_congestion_analysis_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    # Summary
    summary = summarize_results(
        results
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "corrected_congestion_period_summary.csv"
    )

    summary.to_csv(
        summary_path
    )

    # Overall improvement
    improvement = (
        calculate_improvement_vs_dijkstra(
            results
        )
    )

    improvement_path = os.path.join(
        RESULTS_DIR,
        "corrected_congestion_improvement_vs_dijkstra.csv"
    )

    improvement.to_csv(
        improvement_path,
        index=False
    )

    # Candidate diagnostics
    diagnostics_path = os.path.join(
        RESULTS_DIR,
        "corrected_candidate_diagnostics.csv"
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False
    )

    # Regime summary
    _, regime_summary = (
        calculate_regime_summary(
            results
        )
    )

    regime_summary_path = os.path.join(
        RESULTS_DIR,
        "corrected_regime_summary.csv"
    )

    regime_summary.to_csv(
        regime_summary_path
    )

    # Regime improvement
    regime_improvement = (
        calculate_regime_improvement(
            results
        )
    )

    regime_improvement_path = os.path.join(
        RESULTS_DIR,
        "corrected_regime_improvement_vs_dijkstra.csv"
    )

    regime_improvement.to_csv(
        regime_improvement_path,
        index=False
    )

    return {
        "results": results_path,
        "summary": summary_path,
        "improvement": improvement_path,
        "diagnostics": diagnostics_path,
        "regime_summary": regime_summary_path,
        "regime_improvement": regime_improvement_path,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_main_results(
    results
):
    """
    Print the key overall comparison.
    """

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    overall = (
        results
        .groupby("algorithm")[metrics]
        .mean()
    )

    print("\n")
    print("=" * 90)
    print("OVERALL MEAN PERFORMANCE")
    print("=" * 90)

    print(
        overall.round(4).to_string()
    )

    print("\n")
    print("=" * 90)
    print("PERCENTAGE CHANGE VS DIJKSTRA")
    print("=" * 90)

    improvement = (
        calculate_improvement_vs_dijkstra(
            results
        )
    )

    print(
        improvement.round(3).to_string(
            index=False
        )
    )


def print_congestion_results(
    results
):
    """
    Print congestion-period comparison.
    """

    congestion = results[
        results["time_step"].between(
            CONGESTION_START,
            CONGESTION_END
        )
    ]

    metrics = [
        "path_cost",
        "latency",
        "hops",
        "packet_loss",
        "bandwidth",
        "reliability",
        "congestion",
    ]

    means = (
        congestion
        .groupby("algorithm")[metrics]
        .mean()
    )

    print("\n")
    print("=" * 90)
    print(
        "CONGESTION PERIOD "
        f"(t={CONGESTION_START}..{CONGESTION_END})"
    )
    print("=" * 90)

    print(
        means.round(4).to_string()
    )

    print("\n")
    print("=" * 90)
    print("CONGESTION PERIOD CHANGE VS DIJKSTRA")
    print("=" * 90)

    improvement = (
        calculate_improvement_vs_dijkstra(
            results,
            "Congestion"
        )
    )

    print(
        improvement.round(3).to_string(
            index=False
        )
    )


def print_diagnostics_summary(
    diagnostics
):
    """
    Print diagnostic information that helps determine whether
    tunneling is actually influencing route selection.
    """

    if diagnostics.empty:
        return

    subset = diagnostics[
        diagnostics["algorithm"] != "Dijkstra"
    ]

    summary = (
        subset
        .groupby("algorithm")[
            [
                "candidate_count",
                "selected_rank",
                "selected_tunneling_probability",
                "mean_tunneling_probability",
                "std_tunneling_probability",
                "selection_probability",
                "route_changed",
            ]
        ]
        .mean()
    )

    print("\n")
    print("=" * 90)
    print("QT-APP SELECTION DIAGNOSTICS")
    print("=" * 90)

    print(
        summary.round(4).to_string()
    )

    print("\nInterpretation:")
    print(
        "  selected_rank = 1 means the lowest-base-cost "
        "candidate was selected."
    )
    print(
        "  selected_tunneling_probability shows the "
        "tunneling term actually used for the selected route."
    )
    print(
        "  std_tunneling_probability indicates whether "
        "tunneling probabilities differ across candidates."
    )
    print(
        "  route_changed indicates how often the selected "
        "path differs from the previous time step."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nStarting corrected QT-APP experiment...")

    results, diagnostics = (
        run_full_experiment()
    )

    if results.empty:

        print(
            "\nERROR: No simulation results were generated."
        )

        return

    # Save CSV files
    paths = save_results(
        results,
        diagnostics
    )

    # Create plots
    plot_metric_over_time(
        results,
        "path_cost",
        "Path Cost Over Time",
        "corrected_congestion_path_cost.png"
    )

    plot_metric_over_time(
        results,
        "latency",
        "Latency Over Time",
        "corrected_congestion_latency.png"
    )

    plot_metric_over_time(
        results,
        "packet_loss",
        "Packet Loss Over Time",
        "corrected_congestion_packet_loss.png"
    )

    plot_metric_over_time(
        results,
        "bandwidth",
        "Bandwidth Over Time",
        "corrected_congestion_bandwidth.png"
    )

    plot_metric_over_time(
        results,
        "reliability",
        "Reliability Over Time",
        "corrected_congestion_reliability.png"
    )

    plot_metric_over_time(
        results,
        "congestion",
        "Congestion Over Time",
        "corrected_congestion_congestion.png"
    )

    plot_metric_over_time(
        results,
        "hops",
        "Hop Count Over Time",
        "corrected_congestion_hops.png"
    )

    plot_regime_metric(
        results,
        "path_cost",
        "Path Cost by Dynamic Regime",
        "corrected_regime_path_cost.png"
    )

    plot_regime_metric(
        results,
        "latency",
        "Latency by Dynamic Regime",
        "corrected_regime_latency.png"
    )

    plot_regime_metric(
        results,
        "packet_loss",
        "Packet Loss by Dynamic Regime",
        "corrected_regime_packet_loss.png"
    )

    plot_regime_metric(
        results,
        "bandwidth",
        "Bandwidth by Dynamic Regime",
        "corrected_regime_bandwidth.png"
    )

    plot_regime_metric(
        results,
        "reliability",
        "Reliability by Dynamic Regime",
        "corrected_regime_reliability.png"
    )

    plot_regime_metric(
        results,
        "congestion",
        "Congestion by Dynamic Regime",
        "corrected_regime_congestion.png"
    )

    plot_regime_metric(
        results,
        "hops",
        "Hop Count by Dynamic Regime",
        "corrected_regime_hops.png"
    )

    plot_relative_congestion_performance(
        results
    )

    plot_algorithm_comparison(
        results
    )

    plot_tunneling_probability(
        diagnostics
    )

    plot_energy_kappa(
        diagnostics
    )

    plot_selected_rank(
        diagnostics
    )

    # Print results
    print_main_results(
        results
    )

    print_congestion_results(
        results
    )

    print_diagnostics_summary(
        diagnostics
    )

    print("\n")
    print("=" * 90)
    print("FILES CREATED")
    print("=" * 90)

    for name, path in paths.items():

        print(
            f"{name:25s}: {path}"
        )

    print("\nPlots created in:")
    print(
        f"  {os.path.abspath(RESULTS_DIR)}"
    )

    print("\nExperiment completed successfully.")
    print(
        "\nIMPORTANT: These simulation results are experimental "
        "evidence only; they do not by themselves establish "
        "patent novelty, inventive step, or patentability."
    )


if __name__ == "__main__":
   main()