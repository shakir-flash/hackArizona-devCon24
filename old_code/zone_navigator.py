# zone_navigator.py
def next_zone(current_index, zone_list):
    return (current_index + 1) % len(zone_list), zone_list[(current_index + 1) % len(zone_list)]

def previous_zone(current_index, zone_list):
    return (current_index - 1) % len(zone_list), zone_list[(current_index - 1) % len(zone_list)]
