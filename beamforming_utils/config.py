# Grid and direction settings
GRID_MODE = "fibonacci"   # "latlong", "fibonacci", or "beamformer_default"
AZ_STEP_DEG = 30
EL_PLANES_DEG = (-45, 0, 45)
INCLUDE_POLES = True
FIBONACCI_COUNT = 128

# Selection constraints
USE_CORRELATION_FILTER = True
CORRELATION_THRESHOLD = 0.70
USE_MIN_ANGLE_FILTER = True
MIN_ANGULAR_SEPARATION_DEG = 30.0

# Export/analysis defaults
DEFAULT_DURATION_SECONDS = 30
DEFAULT_START_TIME = 0
MAX_EXPORTS = 5
MIN_UNIQUENESS_THRESHOLD = 0.30
GENERATE_HTML_REPORT = False
EXPORT_ALL_DIRECTIONS = False

# Uniqueness score weights
W_ACTIVITY = 0.20
W_FREQDIV = 0.20
W_TEMP = 0.20
W_ACI = 0.20
W_SPATIAL = 0.20

ADI_AEI_DB_THRESHOLD = -50

# --- Profile registry for geophony tuning ---
PROFILE_WEIGHTS = {
    "wind":              dict(Hf=0.15, ADI=0.10, TEMP=0.30, ACI=0.15, SPATIAL=0.30),
    "rain":              dict(Hf=0.10, ADI=0.15, TEMP=0.30, ACI=0.10, SPATIAL=0.35),
    "surf_river":        dict(Hf=0.10, ADI=0.30, TEMP=0.10, ACI=0.20, SPATIAL=0.30),
    "thunder":           dict(Hf=0.05, ADI=0.05, TEMP=0.40, ACI=0.15, SPATIAL=0.35),
    "geophony_general":  dict(Hf=0.10, ADI=0.20, TEMP=0.30, ACI=0.10, SPATIAL=0.30),
    "none":              dict(Hf=W_ACTIVITY, ADI=W_FREQDIV, TEMP=W_TEMP, ACI=W_ACI, SPATIAL=W_SPATIAL),
}

PROFILE_PARAMS = {
    "wind":             dict(ADI_dB=-30, corr=0.60, min_angle=45.0, hpf_hz=120),
    "rain":             dict(ADI_dB=-25, corr=0.60, min_angle=45.0, hpf_hz=None, envelope_median_ms=15),
    "surf_river":       dict(ADI_dB=-30, corr=0.65, min_angle=40.0, hpf_hz=60),
    "thunder":          dict(ADI_dB=-35, corr=0.55, min_angle=50.0, hpf_hz=None),
    "geophony_general": dict(ADI_dB=-30, corr=0.60, min_angle=45.0, hpf_hz=90),
    "none":             dict(ADI_dB=ADI_AEI_DB_THRESHOLD, corr=CORRELATION_THRESHOLD,
                                 min_angle=MIN_ANGULAR_SEPARATION_DEG, hpf_hz=None),
}

def get_profile(profile_name: str):
    p = (profile_name or "none").lower()
    weights = PROFILE_WEIGHTS.get(p, PROFILE_WEIGHTS["none"])
    params  = PROFILE_PARAMS.get(p,  PROFILE_PARAMS["none"])
    return weights, params