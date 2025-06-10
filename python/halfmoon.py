import numpy as np
import matplotlib.pyplot as plt

from vdc import Pupil, HalfmoonOrientation


def main(
    pupil_size: tuple[int, int] = (256, 256),
    psf_size: tuple[int, int, int] = (256, 256, 256),
    numerical_aperture: float = 1.2,
    refractive_index: float = 1.33,
    wavelength: float = 561e-9,
    psf_pitch: tuple[float, float, float] = (12.5e-9, 12.5e-9, 12.5e-9),
) -> None:
    sim_params = {
        "pupil_size": np.array(pupil_size),
        "psf_size": np.array(psf_size),
        "numerical_aperture": numerical_aperture,
        "refractive_index": refractive_index,
        "wavelength": wavelength,
        "psf_pitch": np.array(psf_pitch)
    }

    pupil = Pupil(sim_params)
    pupil.set_halfmoon_pupil(HalfmoonOrientation.HORIZONTAL, phase=np.pi)
    pupil.apply_polarisation("vertical")
    _, intensity = pupil.propagate(0, sim_params)

    _, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(pupil.amp)
    ax[0, 0].title.set_text("Pupil amplitude")
    ax[0, 1].imshow(np.angle(pupil.phase))
    ax[0, 1].title.set_text("Pupil phase")
    ax[1, 0].imshow(intensity)
    ax[1, 0].title.set_text("PSF")
    ax[1, 1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(intensity))))
    ax[1, 1].title.set_text("MTF")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
