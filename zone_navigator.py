# zone_navigator.py

def next_zone(current_index, zones):
    keys = list(zones.keys())
    next_idx = (current_index + 1) % len(keys)
    return next_idx, keys[next_idx]

def previous_zone(current_index, zones):
    keys = list(zones.keys())
    prev_idx = (current_index - 1) % len(keys)
    return prev_idx, keys[prev_idx]
