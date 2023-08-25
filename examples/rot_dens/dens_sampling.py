"""
This is an example of computing of the alignment functions, including cos(theta),
cos^2(theta), cos(theta_2D), and cos^2(theta_2D), by sampling the rotational densiy.
"""
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from typing import Tuple, List


def alignment(rotmat: List[np.ndarray],
              mol_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
              lab_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
              lab_plane: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
              ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Computes expectation values of alignment functions.

    Calculates the expectation values of alignment functions, including cos(theta),
    cos^2(theta), cos(theta_2D), and cos^2(theta_2D), where theta is the angle between
    an arbitrary molecular-frame vector defined by `mol_axis` and a laboratory-frame
    vector defined by `lab_axis`. The angle theta_2D is the projection of theta onto
    a plane defined by `lab_plane`. The rotational distribution is provided as an array
    of rotation matrices in the `rotmat` parameter.

    Args:
        rotmat (List[np.ndarray]): List of rotation matrices representing the
            rotational distribution for different time steps.
        mol_axis (Tuple[float, float, float]): Arbitrary molecular-frame axis
            Default: (0.0, 0.0, 1.0).
        lab_axis (Tuple[float, float, float]): Arbitrary laboratory-frame axis
            Default: (0.0, 0.0, 1.0).
        lab_plane (Tuple[Tuple[float, float, float], Tuple[float, float, float]]):
            Tuple of two vectors defining the laboratory plane
            Default: ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)).

    Returns:
        A tuple containing the following lists of numpy arrays:
        costheta: List of cos(theta) expectation values for each time step.
        cos2theta: List of cos^2(theta) expectation values for each time step.
        costheta2d: List of cos(theta_2D) expectation values for each time step.
        cos2theta2d: List of cos^2(theta_2D) expectation values for each time step.
    """

    # distribution of the molecular-frame axis `mol_axis` in laboratory frame
    mol_axis = np.array(mol_axis) / np.linalg.norm(mol_axis)
    mol_axis_distr = [np.dot(mat, mol_axis) for mat in rotmat]

    # normalize lab axis and lab plane
    lab_axis = np.array(lab_axis) / np.linalg.norm(lab_axis)
    lab_plane = np.array(lab_plane) / np.linalg.norm(lab_plane, axis=-1)[:, None]

    # vector perpendicular to the laboratory plane `lab_plane`
    lab_plane_norm = np.cross(*lab_plane)
    lab_plane_norm = lab_plane_norm / np.linalg.norm(lab_plane_norm)

    # projection of the laboratory axis `lab_axis` onto the laboratory plane `lab_plane`
    lab_axis_lab_plane = np.cross(lab_plane_norm, np.cross(lab_axis, lab_plane_norm))

    # projection of the molecular axis distribution onto the laboratory axis `lab_axis`
    mol_axis_lab_axis = [np.dot(ax, lab_axis) for ax in mol_axis_distr]

    # projection of the molecular axis distribution onto the laboratory plane `lab_plane`
    mol_axis_lab_plane = [np.cross(lab_plane_norm, np.cross(ax, lab_plane_norm)) for ax in mol_axis_distr]
    mol_axis_lab_plane = [ax / np.linalg.norm(ax, axis=-1)[:, None] for ax in mol_axis_lab_plane]

    # projection of `mol_axis_lab_plane` onto the `lab_axis_lab_plane`
    mol_axis_lab_axis_plane = [np.dot(ax, lab_axis_lab_plane) for ax in mol_axis_lab_plane]

    # cos of theta between molecular axis `mol_axis` and laboratory axis `lab_axis`
    # and its projection onto laboratory plane `lab_plane`
    costheta = [np.mean(elem) for elem in mol_axis_lab_axis]
    cos2theta = [np.mean(elem**2) for elem in mol_axis_lab_axis]
    costheta2d = [np.mean(elem) for elem in mol_axis_lab_axis_plane]
    cos2theta2d = [np.mean(elem**2) for elem in mol_axis_lab_axis_plane]

    return costheta, cos2theta, costheta2d, cos2theta2d


if __name__ == '__main__':

    npoints = 1000000

    # read rotational density from file

    with h5py.File("density.h5", 'r') as fl:
        dens = fl['density'][()]
        alpha = fl['alpha'][()]
        beta = fl['beta'][()]
        gamma = fl['gamma'][()]
        times = fl['times'][()]
        field = fl['field'][()]
        cos2theta = fl['cos2theta'][()]
        costheta = fl['costheta'][()]
        cos2theta_wig = fl['cos2theta_wig'][()]
        cos2theta2d_wig = fl['cos2theta2d_wig'][()]

    # integration volume element
    dens *= np.sin(beta)[None, :, None, None]

    # rejections sampling
    fdens = RegularGridInterpolator((alpha, beta, gamma), dens)
    max_dens = np.max(dens, axis=(0,1,2))
    pts = np.random.uniform(low=[0,0,0], high=[2*np.pi,np.pi,2*np.pi], size=(npoints,3))
    w = fdens(pts) / max_dens
    eta = np.random.uniform(0.0, 1.0, size=len(w))
    points = [pts[np.where(ww > eta)] for ww in w.T]

    # rotation matrix for Euler angle samples
    rotmat = [R.from_euler('zyz', pts).as_matrix() for pts in points]

    # expectation values of alignment functions`
    costheta_, cos2theta_, costheta2d_, cos2theta2d_ = alignment(rotmat)

    # plot results

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Time in ps")
    ax1.set_ylabel("Field in V/m")
    ax1.plot(times, field, color='gray')
    ax1.fill_between(times, field[:,-1], 0, color='gray', alpha=.1)

    ax2 = ax1.twinx()

    ax2.set_ylabel("Alignment")
    ax2.plot(times, cos2theta, '-r', linewidth=2, label="$\cos^2\\theta$ analytic")
    ax2.plot(times, cos2theta_wig, '--g', linewidth=2, label="$\cos^2\\theta$ Wigner")
    ax2.plot(times, cos2theta_, ':b', linewidth=2, label="$\cos^2\\theta$ Monte-Carlo")

    ax2.plot(times, cos2theta2d_wig, '--b', linewidth=2, label="$\cos^2\\theta_{\\rm 2D}$ Wigner")
    ax2.plot(times, cos2theta2d_, ':r', linewidth=2, label="$\cos^2\\theta_{\\rm 2D}$ Monte-Carlo")

    ax2.plot(times, costheta, '-b', linewidth=2, label="$\cos\\theta$ analytic")
    ax2.plot(times, costheta_, ':g', linewidth=2, label="$\cos\\theta$ Monte-Carlo")

    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.draw()
    plt.pause(0.0001)
    plt.savefig(f"ocs_alignment_test.png", format = "png", dpi = 300, bbox_inches = "tight")
