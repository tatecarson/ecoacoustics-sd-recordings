from beamforming_utils.config import get_profile

def test_profile_weights_exist():
    for p in ["wind","rain","surf_river","thunder","geophony_general","none"]:
        w, params = get_profile(p)
        assert set(w.keys()) == {"Hf","ADI","TEMP","ACI","SPATIAL"}
        assert round(sum(w.values()), 6) == 1.0
        assert {"ADI_dB","corr","min_angle"}.issubset(params.keys())
