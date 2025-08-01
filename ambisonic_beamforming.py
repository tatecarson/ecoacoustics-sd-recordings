import numpy as np
from typing import List, Tuple

class AmbisonicBeamformer:
    """
    2nd-order ambisonic beamformer for 9-channel (B-format) audio.
    Provides methods to beamform to arbitrary directions and output mono signals.
    """
    SQRT3 = np.sqrt(3)
    SQRT5 = np.sqrt(5)
    SQRT15 = np.sqrt(15)

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    @staticmethod
    def intercardinal_directions() -> Tuple[List[str], List[int]]:
        """
        Returns intercardinal direction names and azimuth angles (degrees).
        """
        names = [
            'north', 'northeast', 'east', 'southeast',
            'south', 'southwest', 'west', 'northwest'
        ]
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        return names, angles

    @staticmethod
    def directions_3d() -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Returns 3D direction names and (azimuth, elevation) angles in degrees.
        """
        names = [
            'north', 'northeast', 'east', 'southeast',
            'south', 'southwest', 'west', 'northwest',
            'up', 'down',
            'up-north', 'up-east', 'up-south', 'up-west',
            'down-north', 'down-east', 'down-south', 'down-west'
        ]
        angles = [
            (0, 0), (45, 0), (90, 0), (135, 0),
            (180, 0), (225, 0), (270, 0), (315, 0),
            (0, 90), (0, -90),
            (0, 45), (90, 45), (180, 45), (270, 45),
            (0, -45), (90, -45), (180, -45), (270, -45)
        ]
        return names, angles

    @classmethod
    def spherical_harmonics_2nd_order(cls, azimuth: float, elevation: float) -> np.ndarray:
        """
        Real spherical harmonics coefficients for 2nd-order ambisonics (N3D normalization).
        Channel order: W, Y, Z, X, V, T, R, S, U
        """
        sqrt3 = cls.SQRT3
        sqrt5 = cls.SQRT5
        sqrt15 = cls.SQRT15
        phi = azimuth  # azimuth (radians)
        theta = np.pi/2 - elevation  # colatitude (radians)
        Y = np.zeros(9)
        Y[0] = 1  # W (omni)
        Y[1] = sqrt3 * np.sin(theta) * np.sin(phi)  # Y
        Y[2] = sqrt3 * np.cos(theta)                # Z
        Y[3] = sqrt3 * np.sin(theta) * np.cos(phi)  # X
        Y[4] = 0.5 * sqrt15 * np.sin(theta)**2 * np.sin(2*phi)  # V
        Y[5] = 0.5 * sqrt15 * np.sin(2*theta) * np.sin(phi)      # T
        Y[6] = 0.5 * sqrt5 * (3 * np.cos(theta)**2 - 1)          # R
        Y[7] = 0.5 * sqrt15 * np.sin(2*theta) * np.cos(phi)      # S
        Y[8] = 0.5 * sqrt15 * np.sin(theta)**2 * np.cos(2*phi)   # U
        return Y

    def beamform(self, audio: np.ndarray, angles: List[Tuple[int, int]]) -> np.ndarray:
        """
        Beamform input 9-channel audio to arbitrary directions.
        Args:
            audio: (samples, 9)
            angles: list of (azimuth, elevation) in degrees
        Returns:
            (samples, num_directions)
        """
        azimuths = np.deg2rad([a[0] for a in angles])
        elevations = np.deg2rad([a[1] for a in angles])
        weights = np.stack(
            [self.spherical_harmonics_2nd_order(az, el) for az, el in zip(azimuths, elevations)],
            axis=1
        )  # (9, num_directions)
        return np.dot(audio, weights)

    def beamform_intercardinal(self, audio: np.ndarray) -> np.ndarray:
        """
        Beamform to 8 intercardinal directions (horizontal plane).
        """
        names, azs = self.intercardinal_directions()
        angles = [(az, 0) for az in azs]
        return self.beamform(audio, angles)

    def beamform_3d_directions(self, audio: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Beamform to all 3D directions.
        Returns:
            (samples, num_directions), direction_names
        """
        names, angles = self.directions_3d()
        return self.beamform(audio, angles), names
    
    def beamform_to_directions(self, audio: np.ndarray, angles: list) -> Tuple[np.ndarray, list]:
        """
        Beamform to arbitrary directions.
        Args:
            audio: (samples, 9)
            angles: list of (azimuth, elevation) in degrees
        Returns:
            (samples, num_directions), direction_names
        """
        out = self.beamform(audio, angles)
        # Generate direction names as "az<az>_el<el>"
        names = [f"az{int(az)}_el{int(el)}" for az, el in angles]
        return out, names