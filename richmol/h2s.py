from keo_jax import com, bisector
import keo_jax
import jax.numpy as np
import poten_h2s_Tyuterev
from prim import numerov, legcos, herm, laguerre
import sys

@com
@bisector('zxy')
def internal_to_cartesian(coords):
    r1, r2, alpha = coords
    xyz = np.array([[0.0, 0.0, 0.0], \
                    [ r1 * np.sin(alpha/2), 0.0, r1 * np.cos(alpha/2)], \
                    [-r2 * np.sin(alpha/2), 0.0, r2 * np.cos(alpha/2)]], \
                    dtype=np.float64)
    return xyz


if __name__ == "__main__":

    # H2S, using valence-bond coordinates and Tyuterev potential

    # setup kinetic energy
    keo_jax.init(masses=[31.97207070, 1.00782505, 1.00782505], \
                 internal_to_cartesian=internal_to_cartesian)

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    # test Numerov basis

    _, num_enr_str, _ = numerov(0, ref_coords, 1000, 30, [0.86, 3.0], \
                                poten_h2s_Tyuterev.poten, keo_jax.Gmat, \
                                pseudo=keo_jax.pseudo, dgmat=keo_jax.dGmat, verbose=True)

    _, num_enr_bnd, _ = numerov(2, ref_coords, 1000, 30, [30*np.pi/180.0, 170*np.pi/180.0], \
                                poten_h2s_Tyuterev.poten, keo_jax.Gmat, \
                                dgmat=keo_jax.dGmat, verbose=True)

    # test Hermite basis
    r, her_enr_str, psi, dpsi = herm(0, ref_coords, 100, 30, [0.5, 10.0], \
                                poten_h2s_Tyuterev.poten, keo_jax.Gmat, \
                                verbose=False)
    # test Legendre(cos) basis
    _, leg_enr_bnd, _, _ = legcos(2, ref_coords, 100, 30, [0, np.pi], \
                                  poten_h2s_Tyuterev.poten, keo_jax.Gmat, \
                                  verbose=True)


    # reference Numerov bending energies for H2S from TROVE
    trove_bend_enr = [0.00000000, 1209.51837915, 2413.11694104, 3610.38836754, 4800.89613073, \
                      5984.17417895, 7159.74903030, 8327.19623484, 9486.23461816,10636.84908026, \
                      11779.41405000,12914.77206889,14044.22229672,15169.39699217,16292.01473383, \
                      17413.45867039,18534.10106278,19652.46977498,20765.06867992,21869.61533575, \
                      22975.24449419]

    # reference Numerov stretching energies for H2S from TROVE
    trove_str_enr = [0.00000000, 2631.91316250, 5168.60694305, 7610.21707943, 9956.62949349, \
                     12207.53640553,14362.48285216,16420.90544929,18382.16446603,20245.56943697, \
                     22010.39748652,23675.90513652,25241.42657687,26708.10237789,28092.75163810, \
                     29462.97307758,30918.23065498,32507.23053835,34229.22373259,36072.32685441, \
                     38026.38406259]
    print("Hermite-TROVE for stretching")
    print(" ".join("  %12.4f"%e1 + "  %12.4f"%e2 + "  %12.4f"%(e1-e2) + "\n" \
        for e1,e2 in zip(her_enr_str, trove_str_enr)))

    print("Numerov-TROVE for stretching")
    print(" ".join("  %12.4f"%e1 + "  %12.4f"%e2 + "  %12.4f"%(e1-e2) + "\n" \
        for e1,e2 in zip(num_enr_str, trove_str_enr)))

    print("Numerov-TROVE for bending")
    print(" ".join("  %12.4f"%e1 + "  %12.4f"%e2 + "  %12.4f"%(e1-e2) + "\n" \
        for e1,e2 in zip(num_enr_bnd, trove_bend_enr)))

    print("Hermite-TROVE for stretching")
    print(" ".join("  %12.4f"%e1 + "  %12.4f"%e2 + "  %12.4f"%(e1-e2) + "\n" \
        for e1,e2 in zip(her_enr_str, trove_str_enr)))

    print("Legendre-TROVE for bending")
    print(" ".join("  %12.4f"%e1 + "  %12.4f"%e2 + "  %12.4f"%(e1-e2) + "\n" \
        for e1,e2 in zip(leg_enr_bnd, trove_bend_enr)))
