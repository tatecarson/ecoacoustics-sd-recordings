import math

# =========================
# Direction grid helpers
# =========================
def latlong_grid(az_step_deg=30, el_planes_deg=(-45, 0, 45), include_poles=True):
    """
    Build a lat/long direction grid as (az_deg, el_deg) pairs.
    azimuth: 0..360-az_step
    elevation planes: list in degrees
    """
    dirs = []
    for el in el_planes_deg:
        for az in range(0, 360, int(az_step_deg)):
            dirs.append((float(az), float(el)))
    if include_poles:
        dirs.append((0.0, 90.0))
        dirs.append((0.0, -90.0))
    return dirs


def fibonacci_sphere_grid(n_points=128):
    """
    Generate ~uniform points on a sphere using Fibonacci lattice.
    Returns list of (az_deg, el_deg) where az is [0,360), el is [-90,90].
    """
    points = []
    phi = (1 + 5 ** 0.5) / 2
    for i in range(n_points):
        z = 1 - (2 * i + 1) / n_points
        r = math.sqrt(max(0.0, 1 - z * z))
        theta = 2 * math.pi * (i / phi % 1.0)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        # Convert to az/el
        az = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        el = math.degrees(math.asin(z))
        points.append((az, el))
    return points


def unit_vector_from_azel(az_deg, el_deg):
    """
    Convert azimuth/elevation in degrees to a 3D unit vector.
    Azimuth 0° = +X by this math; adjust if your convention differs.
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)
    v = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def angular_separation_deg(v1, v2):
    """Angle between two 3D unit vectors in degrees."""
    c = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(math.acos(c))