"""
Global Twin — Scenario Engine (v2.0)

Pre-built macro shock scenarios and custom scenario builder.
Each scenario defines multi-variable simultaneous shocks.
"""


# ─────────────────────────────────────────────
#  Pre-Built Scenario Registry
# ─────────────────────────────────────────────

SCENARIOS = {
    "oil_embargo": {
        "name": "Oil Embargo",
        "description": "Major oil-producing nations reduce output. Energy prices spike, shipping costs soar, inflation cascades globally.",
        "icon": "🛢️",
        "severity": "high",
        "shocks": {
            "CRUDE_OIL":        0.60,   # +60%
            "NATURAL_GAS":      0.40,   # +40%
            "BALTIC_DRY_INDEX": 0.30,   # +30% (shipping costs)
        },
    },
    "us_china_trade_war": {
        "name": "US-China Trade War",
        "description": "Escalating tariffs between US and China. Manufacturing slows, trade balances shift, currencies adjust.",
        "icon": "⚔️",
        "severity": "high",
        "shocks": {
            "CN_GDP_GROWTH":    -0.05,  # -5%
            "US_TRADE_BALANCE": -0.20,  # -20% (deficit widens)
            "CNY_USD":           0.08,  # +8% (CNY depreciates vs USD)
            "COPPER":           -0.15,  # -15% (manufacturing proxy)
            "SEMICONDUCTOR_IDX":-0.10,  # -10% (supply chain disruption)
        },
    },
    "global_pandemic": {
        "name": "Global Pandemic",
        "description": "Worldwide health crisis. GDP contracts, unemployment surges, supply chains break, safe-haven assets rise.",
        "icon": "🦠",
        "severity": "critical",
        "shocks": {
            "US_GDP_GROWTH":    -0.08,  # -8%
            "EU_GDP_GROWTH":    -0.07,  # -7%
            "CN_GDP_GROWTH":    -0.03,  # -3%
            "IN_GDP_GROWTH":    -0.10,  # -10%
            "JP_GDP_GROWTH":    -0.05,  # -5%
            "US_UNEMPLOYMENT":   0.80,  # +80% (doubles)
            "CRUDE_OIL":        -0.40,  # -40% (demand collapse)
            "BALTIC_DRY_INDEX": -0.30,  # -30%
            "VIX":               1.50,  # +150% (fear spikes)
            "GOLD":              0.15,  # +15% (safe haven)
            "SP500":            -0.25,  # -25%
        },
    },
    "interest_rate_hike": {
        "name": "Fed Rate Hike (+200bps)",
        "description": "Federal Reserve raises rates aggressively. Dollar strengthens, emerging market currencies weaken, equities dip.",
        "icon": "📈",
        "severity": "medium",
        "shocks": {
            "US_FED_RATE":       0.40,  # +40% (represents ~200bps hike)
            "INR_USD":           0.06,  # +6% (INR depreciates)
            "CNY_USD":           0.03,  # +3%
            "JPY_USD":           0.04,  # +4%
            "EUR_USD":          -0.03,  # -3% (EUR weakens)
            "SP500":            -0.08,  # -8%
            "GOLD":             -0.05,  # -5%
        },
    },
    "energy_transition": {
        "name": "Green Energy Transition",
        "description": "Rapid shift to renewables. Fossil fuel demand drops, green metals surge, traditional energy producers suffer.",
        "icon": "🌱",
        "severity": "medium",
        "shocks": {
            "CRUDE_OIL":        -0.30,  # -30%
            "NATURAL_GAS":      -0.20,  # -20%
            "COPPER":            0.40,  # +40% (green infrastructure)
            "GOLD":              0.05,  # +5%
        },
    },
    "china_slowdown": {
        "name": "China Economic Slowdown",
        "description": "China's growth stalls. Global commodity demand drops, trade partners affected, safe-haven flows increase.",
        "icon": "🇨🇳",
        "severity": "high",
        "shocks": {
            "CN_GDP_GROWTH":    -0.10,  # -10%
            "CN_CPI_INFLATION": -0.05,  # -5% (deflation risk)
            "COPPER":           -0.25,  # -25%
            "CRUDE_OIL":        -0.15,  # -15%
            "CNY_USD":           0.05,  # +5% (CNY weakens)
            "WHEAT":            -0.10,  # -10%
            "BALTIC_DRY_INDEX": -0.20,  # -20%
        },
    },
}


def list_scenarios():
    """Return a summary list of all available scenarios."""
    return [
        {
            "id": sid,
            "name": s["name"],
            "icon": s["icon"],
            "description": s["description"],
            "severity": s["severity"],
            "num_shocks": len(s["shocks"]),
        }
        for sid, s in SCENARIOS.items()
    ]


def get_scenario(scenario_id):
    """
    Get a scenario by ID.
    
    Args:
        scenario_id: e.g., 'oil_embargo', 'global_pandemic'
    
    Returns:
        dict with name, description, and shocks mapping.
    
    Raises:
        ValueError if scenario not found.
    """
    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {scenario_id}. "
            f"Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[scenario_id]


def build_custom_scenario(name, shocks_dict, description="Custom user-defined scenario"):
    """
    Build a custom scenario from user-provided shocks.
    
    Args:
        name: Human-readable scenario name.
        shocks_dict: dict mapping indicator_name → percentage shock (e.g., {'CRUDE_OIL': 0.30})
        description: Optional description.
    
    Returns:
        dict in the same format as pre-built scenarios.
    """
    return {
        "name": name,
        "description": description,
        "icon": "🔧",
        "severity": "custom",
        "shocks": shocks_dict,
    }
