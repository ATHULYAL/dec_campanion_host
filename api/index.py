from flask import Flask, render_template, request, jsonify
import math
import random
import copy
import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)


class DecisionEngine:

    def __init__(self):
        self.qualitative_map = {
            "very low": 1, "low": 3, "medium": 5,
            "high": 7, "very high": 9
        }
        self.value_label_map = {
            1: "very low", 2: "very low",
            3: "low",      4: "low",
            5: "medium",   6: "medium",
            7: "high",     8: "high",
            9: "very high"
        }

    def value_to_label(self, value):
        rounded = max(1, min(9, round(float(value))))
        return self.value_label_map.get(rounded, "medium")

    def calculate_roc_weights(self, n):
        """Standard ROC weights for strict priority order."""
        weights = []
        for i in range(1, n + 1):
            w = sum(1 / j for j in range(i, n + 1)) / n
            weights.append(w)
        return weights

    def calculate_roc_weights_with_ties(self, priorities):
        """
        ROC weights that handle tied priorities.

        Tied criteria occupy the same positions in the ordering.
        Their weights are averaged so tied criteria are treated equally.

        Example: priorities = [1, 1, 2]
          - Positions 1 and 2 are both occupied by the tied pair.
          - Each tied criterion gets avg(ROC_weight_1, ROC_weight_2).
          - The third criterion gets ROC_weight_3 normally.
        """
        n = len(priorities)
        base = self.calculate_roc_weights(n)

        sorted_unique = sorted(set(priorities))
        slot_cursor = 0
        priority_to_slots = {}
        for p in sorted_unique:
            count = priorities.count(p)
            priority_to_slots[p] = list(range(slot_cursor, slot_cursor + count))
            slot_cursor += count

        priority_to_weight = {
            p: sum(base[s] for s in slots) / len(slots)
            for p, slots in priority_to_slots.items()
        }

        return [priority_to_weight[p] for p in priorities]

    def _to_float(self, v):
        try:
            return float(v)
        except:
            return float(self.qualitative_map.get(str(v).lower().strip(), 5))

    def run_topsis(self, options, criteria):
        norm = {}
        for c in criteria:
            cid = c['id']
            sq = sum(o['values'][cid] ** 2 for o in options)
            den = math.sqrt(sq) if sq > 0 else 1
            for o in options:
                norm.setdefault(o['name'], {})
                norm[o['name']][cid] = (o['values'][cid] / den) * c['weight']

        best = {}
        worst = {}
        for c in criteria:
            cid = c['id']
            vals = [norm[o['name']][cid] for o in options]
            if c['type'] == "benefit":
                best[cid] = max(vals)
                worst[cid] = min(vals)
            else:
                best[cid] = min(vals)
                worst[cid] = max(vals)

        results = []
        for o in options:
            name = o['name']
            d1 = math.sqrt(sum(
                (norm[name][c['id']] - best[c['id']]) ** 2 for c in criteria
            ))
            d2 = math.sqrt(sum(
                (norm[name][c['id']] - worst[c['id']]) ** 2 for c in criteria
            ))
            score = d2 / (d1 + d2) if d1 + d2 > 0 else 0.5
            results.append({"name": name, "score": score})

        return (
            sorted(results, key=lambda x: x['score'], reverse=True),
            best, worst, norm
        )

    def simulate(self, options, criteria):
        counts = {o['name']: 0 for o in options}
        for _ in range(300):
            sim = []
            for o in options:
                vals = o['values'].copy()
                for c in criteria:
                    if c['dynamic']:
                        vals[c['id']] = max(1, min(9, vals[c['id']] + random.gauss(0, 1)))
                sim.append({"name": o['name'], "values": vals})
            res, _, _, _ = self.run_topsis(sim, criteria)
            counts[res[0]['name']] += 1

        return sorted(
            [{"name": n, "confidence": round(c / 3, 1)} for n, c in counts.items()],
            key=lambda x: x['confidence'], reverse=True
        )

    def explain_all(self, options, criteria):
        results, _, _, norm = self.run_topsis(options, criteria)

        ideal_raw = {}
        worst_raw = {}
        for c in criteria:
            cid = c['id']
            raw_vals = [o['values'][cid] for o in options]
            if c['type'] == 'benefit':
                ideal_raw[cid] = max(raw_vals)
                worst_raw[cid] = min(raw_vals)
            else:
                ideal_raw[cid] = min(raw_vals)
                worst_raw[cid] = max(raw_vals)

        all_explanations = {}
        for o in options:
            name = o['name']
            explanation = []
            for c in criteria:
                cid = c['id']
                actual = o['values'][cid]
                ideal_val = ideal_raw[cid]
                worst_val = worst_raw[cid]
                raw_range = abs(ideal_val - worst_val)
                gap_pct = abs(actual - ideal_val) / raw_range * 100 if raw_range > 0 else 0.0
                explanation.append({
                    "name":    c['name'],
                    "actual":  actual,
                    "gap_pct": round(gap_pct, 1),
                    "weight":  c['weight'],
                })
            explanation.sort(key=lambda x: x['gap_pct'])
            all_explanations[name] = explanation

        return all_explanations, results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.json
    engine = DecisionEngine()

    goal = data['goal']
    criteria_data = data['criteria']
    options_data = data['options']

    # Extract priorities — default to position order if not provided
    priorities = [int(c.get('priority', i + 1)) for i, c in enumerate(criteria_data)]
    priority_counts = Counter(priorities)

    # Assign weights using tied-aware ROC
    weights = engine.calculate_roc_weights_with_ties(priorities)

    criteria = []
    for i, c in enumerate(criteria_data):
        criteria.append({
            "id":       f"c{i}",
            "name":     c['name'],
            "type":     c['type'],
            "dynamic":  c['dynamic'],
            "priority": priorities[i],
            "tied":     priority_counts[priorities[i]] > 1,
            "weight":   weights[i]
        })

    options = []
    for o in options_data:
        vals = {}
        for i, v in enumerate(o['values']):
            vals[f"c{i}"] = engine._to_float(v)
        options.append({"name": o['name'], "values": vals})

    original_options = copy.deepcopy(options)

    sim = engine.simulate(options, criteria)
    all_explanations, _ = engine.explain_all(original_options, criteria)

# Winner reasoning
winner = sim[0]['name']
winner_expl = all_explanations[winner]

best_crit = min(
    winner_expl,
    key=lambda e: (e['gap_pct'], -e['weight'])
)

# Find criterion type
crit_type = next(
    c['type'] for c in criteria
    if c['name'] == best_crit['name']
)

value_label = engine.value_to_label(best_crit['actual'])

# Correct wording depending on type
if crit_type == "benefit":
    reasoning = f"'{winner}' is selected due to its {value_label} {best_crit['name']}."
else:
    reasoning = f"'{winner}' is selected due to its low {best_crit['name']}."
    # Per-option breakdown
    breakdown = []
    for o in original_options:
        name = o['name']
        expl = all_explanations[name]
        confidence = next(r['confidence'] for r in sim if r['name'] == name)
        rank = next(i + 1 for i, r in enumerate(sim) if r['name'] == name)

        opt_best = min(expl, key=lambda e: (e['gap_pct'], -e['weight']))
        opt_label = engine.value_to_label(opt_best['actual'])
        selection_note = f"'{name}' is notable for its {opt_label} {opt_best['name']}."

        strengths  = [e['name'] for e in expl if e['gap_pct'] <= 40]
        weaknesses = [e['name'] for e in expl if e['gap_pct'] > 40]

        breakdown.append({
            "name":           name,
            "rank":           rank,
            "confidence":     confidence,
            "selection_note": selection_note,
            "strengths":      strengths,
            "weaknesses":     weaknesses,
        })

    return jsonify({
        "goal": goal,
        "criteria": [
            {
                "name":     c['name'],
                "type":     c['type'],
                "dynamic":  c['dynamic'],
                "priority": c['priority'],
                "tied":     c['tied'],
                "weight":   round(c['weight'] * 100, 2)
            }
            for c in criteria
        ],
        "simulation_results": sim,
        "reasoning":          reasoning,
        "option_breakdown":   breakdown,
    })


app = app

