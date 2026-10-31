"""Part facts from the customer intake (mode.parts = intake): mass, positioning tolerance, deformability, basis.

Used by s2 (MTM-UAS weight class, ILO use-of-force allowance) and s3 (payload, tolerance, handling) instead of the
built-in part table. The SOP step name / text that an element belongs to is matched against the station's declared
parts by keyword (rcd.common.intake.part_for); unmatched steps get the intake's default part. Every returned record
carries source = "intake".
"""
from __future__ import annotations

from rcd.common.intake import load_intake, mode, part_for, station


class IntakeParts:
    def __init__(self, cfg: dict):
        self.intake = load_intake(cfg)
        if self.intake is None:
            raise ValueError("mode.parts = intake requires an `intake:` file in the config")
        self._text: dict[tuple[str, str], str] = {}
        for st in self.intake["stations"]:
            for s in st.get("sop", []):
                self._text[(st["name"], s["step"])] = " ".join([s["step"], s.get("text", "")])

    def __call__(self, station_name: str, subgoal: str | None) -> dict:
        text = self._text.get((station_name, subgoal or ""), subgoal or "")
        p = part_for(self.intake, station_name, text if subgoal else None)
        return {"part": p["name"], "mass_kg": float(p["mass_kg"]), "precision_mm": float(p["precision_mm"]),
                "basis": f"intake: {p.get('basis', '')}", "flexible": bool(p.get("flexible", False)), "source": "intake"}

    def line(self) -> dict:
        return dict(self.intake.get("line", {}))

    def has_station(self, name: str) -> bool:
        return station(self.intake, name) is not None


def resolver(cfg: dict):
    """(station, subgoal) -> part record. Intake-driven in parts=intake mode, otherwise None (caller keeps the
    original behaviour)."""
    if mode(cfg, "parts", "builtin") == "intake":
        return IntakeParts(cfg)
    return None
