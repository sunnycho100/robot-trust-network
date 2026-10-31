"""Economic assumptions with sources (single place; used by s5 for the cost objective and by s7 for ROI).

Values that come from config/demo.yaml (discount rate, horizon, fringe multiplier, shifts) are read from cfg.
Everything else is listed here with its source; items without a published source are labelled 'assumption'.
"""
from __future__ import annotations

import math

BLS_URL = "https://www.bls.gov/oes/2025/may/oes512090.htm"
BLS_OOH_URL = "https://www.bls.gov/ooh/production/assemblers-and-fabricators.htm#tab-5"
NIST_INTEGRATION_URL = "https://www.nist.gov/node/703906"
HOWTOROBOT_URL = "https://howtorobot.com/expert-insight/robot-price-how-much-does-robot-solution-really-cost"

# BLS OEWS May 2025, SOC 51-2090 Miscellaneous Assemblers and Fabricators (the SOC that replaced 51-2092 Team
# Assemblers): national median annual wage $44,650 (BLS OOH pay table, May 2025). BLS converts annual to hourly
# with 2,080 h/yr, so median hourly = 44,650 / 2,080.
WAGE_ANNUAL_MEDIAN = 44650.0
WAGE_ANNUAL_TRANSPORT_EQUIP = 50790.0  # same OOH table, industry row "Transportation equipment manufacturing"
BLS_HOURS_PER_YEAR = 2080.0

# End-of-arm tooling / feeders per robot task category (USD). Labelled assumptions: typical list prices of an
# electric screwdriving end-effector with screw presenter, a parallel/vacuum gripper, and a 2D vision kit.
EOAT_USD = {
    "screw": 15000.0,
    "fasten": 15000.0,
    "pick": 6000.0,
    "place": 6000.0,
    "insert": 8000.0,
    "inspect": 9000.0,
    "vision": 9000.0,
    "default": 8000.0,
}
SAFETY_USD = 6000.0          # collaborative-cell risk assessment (ISO 10218-2 / ISO/TS 15066) + area scanner
INTEGRATION_MULT = 1.0       # integration (engineering, programming, commissioning) as multiple of robot price
INTEGRATION_MULT_RANGE = (0.5, 2.0)
MAINT_PCT = 0.05             # annual maintenance as fraction of robot + EOAT capex
MAINT_PCT_RANGE = (0.02, 0.12)
ROBOT_AVAILABILITY = 0.95    # robot uptime for rough-cut capacity (assumption)
ROBOT_EFFICIENCY = 1.00      # rate efficiency for RCA (simulated cycle already includes motion limits)
HANDOVER_S = 2.0             # human<->robot synchronisation per robot-assigned element (assumption)
ROBOT_TIME_CV = 0.03         # robot cycle-time coefficient of variation in DES (assumption)
# Layout / lean levers for s3 'layout_change' elements (not robot-arm work). Assumptions, shown with sensitivity.
LAYOUT_LEVERS = {
    "amr_kitting": {"reduction": 0.50, "line_capex_usd": 45000.0, "station_capex_usd": 3000.0,
                    "desc": "AMR kitting / point-of-use supply: parts delivered kitted to the station, operator walking "
                            "to line-side racks halved (assumption); one AMR shared by the line + a kit rack per station"},
    "fixture_or_layout": {"reduction": 0.50, "line_capex_usd": 0.0, "station_capex_usd": 5000.0,
                          "desc": "quick-setup fixture / tool presented at point of use, setup time halved (assumption)"},
    "default": {"reduction": 0.30, "line_capex_usd": 0.0, "station_capex_usd": 5000.0,
                "desc": "generic re-layout of the station (assumption)"},
}
# Placement-implied hidden costs (s4 mounting = "linear_track": conveyor-synchronised 7th axis because the operator
# stands on the moving belt and a fixed floor base cannot reach the door over the tracking window).
TRACK_URL = "https://vention.io/designs/2m-large-cobot-range-extender-track-only-168270"
TRACK_URL_HIGH = "https://vention.io/designs/84388"
TRACK_ONLY_USD = 11168.46     # Vention 2 m large cobot range extender, track only (list price)
TRACK_DRIVE_SYNC_USD = 7000.0  # servo drive + 7th-axis/conveyor-sync integration (assumption)
TRACK_USD = TRACK_ONLY_USD + TRACK_DRIVE_SYNC_USD
TRACK_USD_RANGE = (TRACK_ONLY_USD, 24390.57 + TRACK_DRIVE_SYNC_USD)   # low: track only; high: Vention 3 m overhead
# Automatic tool changer when one arm serves >1 task category (e.g. gripper + screwdriver)
TOOL_CHANGER_URL = "https://www.automationworld.com/factory/robotics/product/13319313/onrobot-light-and-compact-robotics-tool-changer"
TOOL_CHANGER_PRICE_URL = "https://grabarobot.com/blog/robot-tool-changing-system-2026/"
TOOL_CHANGER_USD = 7500.0     # automatic changer (robot side + tool sides, ATI QC-11 class USD 3.5-6k) + tool stand
TOOL_CHANGER_USD_RANGE = (5000.0, 10000.0)
TOOL_CHANGE_S = 5.0           # per exchange; OnRobot Quick Changer: "tool change in under 5 seconds" (vendor claim)
HELPER_WALK_S = 6.0          # relief operator walk per station served per cycle (assumption)
CELL_TRAVEL_S = 8.0          # robot repositioning between two adjacent stations on a 7th-axis rail (assumption)


def hourly_wage() -> float:
    return WAGE_ANNUAL_MEDIAN / BLS_HOURS_PER_YEAR


def loaded_wage(cfg: dict, wage: float | None = None) -> float:
    return (wage if wage is not None else hourly_wage()) * cfg["economics"]["labor_fringe_multiplier"]


def annual_hours(cfg: dict, shifts: float | None = None) -> float:
    ln = cfg["line"]
    return (shifts if shifts is not None else ln["shifts_per_day"]) * ln["hours_per_shift"] * ln["days_per_year"]


def annual_cycles(cfg: dict, shifts: float | None = None) -> float:
    return annual_hours(cfg, shifts) * 3600.0 / cfg["line"]["takt_s"]


def crf(i: float, n: int) -> float:
    """Capital recovery factor A/P = i(1+i)^n / ((1+i)^n - 1)  (ISyE 510 lecture03; Exam1 study guide sec. 4)."""
    if i == 0:
        return 1.0 / n
    g = (1 + i) ** n
    return i * g / (g - 1)


def eoat_cost(category: str) -> float:
    c = (category or "").lower()
    for k, v in EOAT_USD.items():
        if k != "default" and k in c:
            return v
    return EOAT_USD["default"]


def robot_capex(price_mid: float, category: str, integration_mult: float = INTEGRATION_MULT) -> dict:
    """Installed cost of one robot cell = robot + EOAT/feeder + safety + integration (multiple of robot price)."""
    eoat = eoat_cost(category)
    integ = integration_mult * price_mid
    total = price_mid + eoat + SAFETY_USD + integ
    return {"robot_usd": price_mid, "eoat_usd": eoat, "safety_usd": SAFETY_USD, "integration_usd": integ,
            "total_usd": total}


def assumption_rows(cfg: dict) -> list[dict]:
    ec, ln = cfg["economics"], cfg["line"]
    return [
        {"name": "Hourly wage (assemblers, SOC 51-2090, national median)", "value": round(hourly_wage(), 4),
         "unit": "USD/h",
         "source": f"BLS OEWS May 2025, SOC 51-2090 Miscellaneous Assemblers and Fabricators (successor of 51-2092 "
                   f"Team Assemblers), median annual wage $44,650 / 2,080 h; {BLS_URL} ; pay table: {BLS_OOH_URL}"},
        {"name": "Labour fringe multiplier", "value": ec["labor_fringe_multiplier"], "unit": "x",
         "source": "config/demo.yaml economics.labor_fringe_multiplier (assumption)"},
        {"name": "Loaded labour cost", "value": round(loaded_wage(cfg), 4), "unit": "USD/h",
         "source": "derived: wage x fringe multiplier"},
        {"name": "Shifts per day", "value": ln["shifts_per_day"], "unit": "shifts",
         "source": "config/demo.yaml line.shifts_per_day (assumption)"},
        {"name": "Operating hours per year", "value": annual_hours(cfg), "unit": "h/yr",
         "source": "derived: shifts x hours/shift x days/yr (config/demo.yaml)"},
        {"name": "Takt time", "value": ln["takt_s"], "unit": "s",
         "source": "config/demo.yaml line.takt_s (CarDA record: ~4 min cycle)"},
        {"name": "Units per year", "value": round(annual_cycles(cfg), 1), "unit": "doors/yr",
         "source": "derived: operating hours x 3600 / takt"},
        {"name": "Discount rate (MARR)", "value": ec["discount_rate"], "unit": "1/yr",
         "source": "config/demo.yaml economics.discount_rate (assumption)"},
        {"name": "Horizon", "value": ec["horizon_years"], "unit": "yr",
         "source": "config/demo.yaml economics.horizon_years (assumption; no salvage value)"},
        {"name": "Robot price", "value": "midpoint of s4 price_usd_range", "unit": "USD",
         "source": "out/s4/robot_catalog.json price_usd_range (catalog source URLs per model)"},
        {"name": "Integration cost multiple of robot price", "value": INTEGRATION_MULT, "unit": "x robot price",
         "source": f"assumption for a collaborative cell without fencing; NIST reports 2-4x robot cost for "
                   f"industrial-robot integration ({NIST_INTEGRATION_URL}), used as the high end of the range"},
        {"name": "End-effector / feeder cost (screwdriving)", "value": EOAT_USD["screw"], "unit": "USD",
         "source": "assumption (typical list price of electric screwdriving EOAT + screw presenter)"},
        {"name": "End-effector cost (gripper)", "value": EOAT_USD["pick"], "unit": "USD",
         "source": "assumption (typical list price of an electric parallel gripper)"},
        {"name": "Safety (risk assessment + area scanner)", "value": SAFETY_USD, "unit": "USD/cell",
         "source": "assumption (ISO 10218-2 / ISO/TS 15066 collaborative operation)"},
        {"name": "Annual maintenance", "value": MAINT_PCT, "unit": "fraction of robot+EOAT capex",
         "source": f"assumption; howtorobot reports operating+maintenance up to 10-12%/yr of equipment cost "
                   f"({HOWTOROBOT_URL}), used as the high end"},
        {"name": "Layout lever: AMR kitting / point-of-use supply", "value": LAYOUT_LEVERS["amr_kitting"]["reduction"],
         "unit": "fraction of WALK time removed",
         "source": "assumption; capex: AMR shared by line USD %d + kit rack USD %d per station (assumption)" % (
             LAYOUT_LEVERS["amr_kitting"]["line_capex_usd"], LAYOUT_LEVERS["amr_kitting"]["station_capex_usd"])},
        {"name": "Relief (roaming) operator", "value": HELPER_WALK_S, "unit": "s walk per station served",
         "source": "assumption; ISyE 510 lecture06 'other ways to improve the line balance: a roaming helper'"},
        {"name": "7th-axis linear track (conveyor-synchronised)", "value": TRACK_USD, "unit": "USD/robot",
         "range": list(TRACK_USD_RANGE),
         "source": f"Vention 2 m large cobot range extender, track only USD {TRACK_ONLY_USD:,.2f} ({TRACK_URL}) + servo "
                   f"drive and conveyor-sync integration USD {TRACK_DRIVE_SYNC_USD:,.0f} (assumption); high end: Vention "
                   f"3 m overhead track ({TRACK_URL_HIGH}). Needed where s4 mounting = linear_track."},
        {"name": "Automatic tool changer (arm serving >1 task category)", "value": TOOL_CHANGER_USD, "unit": "USD/robot",
         "range": list(TOOL_CHANGER_USD_RANGE),
         "source": f"ATI QC-11-class cobot changer USD 3,500-6,000 (secondary source {TOOL_CHANGER_PRICE_URL}) + tool "
                   f"stand (assumption)"},
        {"name": "Tool exchange time", "value": TOOL_CHANGE_S, "unit": "s per exchange (k tools -> k exchanges/cycle)",
         "source": f"OnRobot Quick Changer, vendor claim 'tool change in under 5 seconds' ({TOOL_CHANGER_URL})"},
        {"name": "Human-robot handover per robot element", "value": HANDOVER_S, "unit": "s",
         "source": "assumption (synchronisation wait in a collaborative station)"},
    ]


def npv(capex: float, annual: float, i: float, n: int) -> float:
    return -capex + sum(annual / (1 + i) ** t for t in range(1, n + 1))


def irr(capex: float, annual: float, n: int) -> float | None:
    if capex <= 0 or annual <= 0:
        return None
    from scipy.optimize import brentq
    f = lambda r: npv(capex, annual, r, n)  # noqa: E731
    lo, hi = -0.99, 10.0
    if f(lo) * f(hi) > 0:
        return None
    return float(brentq(f, lo, hi))


def p_over_a(i: float, n: int) -> float:
    return 1.0 / crf(i, n) if n > 0 else 0.0


assert math.isclose(crf(0.08, 5), 0.25045645, rel_tol=1e-6)
