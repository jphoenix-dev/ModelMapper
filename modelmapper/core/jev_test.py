"""Live smoke test for jev.py (3 tiny API calls, a fraction of a cent). Run: python src/jev_test.py"""
from modelmapper.core import jev

p = jev.noul("The Golden Gate Bridge is in San Francisco.", "Does the text mention a bridge?")
print(f"noul  bridge? {p:.3f}")
assert p > 0.8

c = jev.choice("I kneaded the dough and left it to rise overnight before baking.",
               "What activity is the text about?",
               {"baking": "Making bread, cakes or pastry", "sport": "Playing or watching a sport",
                "travel": "Going on a trip or visiting a place"})
print("choice", {k: round(v, 3) for k, v in c.items()})
assert max(c, key=c.get) == "baking"

snippets = [
    "The Golden Gate Bridge's towers rise 227 metres above the water.",
    "Cable cars still climb the steep hills of San Francisco.",
    "Mix flour, water, salt and yeast, then knead the dough for ten minutes.",
    "Alcatraz Island sits in the middle of San Francisco Bay.",
    "The Bay Bridge links San Francisco to Oakland across the bay.",
]
q = jev.intruder(snippets)
print("intruder", {k: round(v, 3) for k, v in q.items()})
assert max(q, key=q.get) == 2

print(f"usage {jev.usage}  est. cost ${jev.cost():.8f}")
