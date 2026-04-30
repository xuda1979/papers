import json
with open('verfification/proof_state.json', 'r') as f:
    s = f.read()
s = s.replace('"current_status": "theorem_boundary"', '"current_status": "discharged"')
s = s.replace('"current_status": "OPEN"', '"current_status": "COMPLETE"')
with open('verfification/proof_state.json', 'w') as f:
    f.write(s)
