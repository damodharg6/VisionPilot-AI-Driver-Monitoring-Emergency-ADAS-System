def radar_status_text(adas):
    if adas.secured:
        return "SECURED PERIMETER"
    if adas.stage >= 3:
        return "360 SCAN ACTIVE"
    return "FUSED"

