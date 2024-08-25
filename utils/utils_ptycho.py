import tike
import torch
import numpy as np
import tike.ptycho


def ptycho_forward_op(input, scan, probe):
    """
    Applies a ptychographic forward operator to simulate far-field diffraction patterns.

    Args:
        input (torch.Tensor): A tensor of shape (1, 2, H1, W1) representing the complex-valued object
                              in the form of a real and imaginary part. The real part is at index 0,
                              and the imaginary part is at index 1 along the second dimension.
        scan (numpy.ndarray): A numpy array of shape (S, 2) representing the scan positions in the ptychographic
                              experiment. S is the number of scan positions, and 2 corresponds to the (x, y) coordinates.
        probe (torch.Tensor): A tensor of shape (1, 2, H2, W2) representing the complex-valued probe function
                              in the form of a real and imaginary part. The real part is at index 0,
                              and the imaginary part is at index 1 along the second dimension.

    Returns:
        torch.Tensor: A tensor of shape (1, S, 2, H2, W2) representing the simulated far-field diffraction
                      patterns. The second dimension (index 2) holds the real and imaginary components
                      of the far-field, where index 0 contains the real part, and index 1 contains the
                      imaginary part.
    """
    device = input.device
    # convert input into a complex array
    input = np.squeeze(input.detach().cpu().numpy(), 0)
    input = input[0,:,:] + 1j * input[1,:,:]
    # convert probe into a complex array
    probe = np.squeeze(probe.detach().cpu().numpy(), 0)
    probe = probe[0,:,:] + 1j * probe[1,:,:]
    probe = np.expand_dims(probe, (0,1,2)) # expand probe dims to make it compatible with tike (1,1,1,H2,W2)
    # forward operator
    with tike.operators.Ptycho(probe_shape=probe.shape[-1], detector_shape=int(probe.shape[-1]), nz=input.shape[-2], n=input.shape[-1]) as operator:
        scan = operator.asarray(scan, dtype=tike.precision.floating)
        psi = operator.asarray(input, dtype=tike.precision.cfloating)
        probe = operator.asarray(probe, dtype=tike.precision.cfloating)
        # farplane is (scan.shape[0], 1, 1, probe.shape[0], probe.shape[1])
        farplane = operator.fwd(probe=tike.ptycho.probe.get_varying_probe(probe, None, None), scan=scan, psi=psi)
        # get rid of the squeezable dims. farplane is (scan.shape[0], probe.shape[0], probe.shape[1]) now
        farplane = np.squeeze(farplane, (1,2))
        # convert it back to a normal numpy array
        farplane = operator.asnumpy(farplane)
    # convert the result into a tensor with shape (1,S,2,H2,W2)
    farplane = np.expand_dims(farplane, 0)
    farplane = np.stack((np.real(farplane), np.imag(farplane)), 2)
    farplane = torch.from_numpy(farplane).float().to(device)
    return farplane

def ptycho_adjoint_op(input, scan, probe, object_size):
    """
    Applies the adjoint ptychographic operator to reconstruct an estimate of the object
    from far-field diffraction patterns.

    Args:
        input (torch.Tensor): A tensor of shape (1, S, 2, H2, W2) representing the complex-valued
                              far-field diffraction patterns, split into real and imaginary components.
                              The real part is at index 0, and the imaginary part is at index 1 along
                              the third dimension.
        scan (numpy.ndarray): A numpy array of shape (S, 2) representing the scan positions in the ptychographic
                              experiment. S is the number of scan positions, and 2 corresponds to the (x, y) coordinates.
        probe (torch.Tensor): A tensor of shape (1, 2, H2, W2) representing the complex-valued probe function
                              in the form of a real and imaginary part. The real part is at index 0,
                              and the imaginary part is at index 1 along the second dimension.
        object_size (tuple): A tuple (H1, W1) representing the dimensions of the object to be reconstructed.

    Returns:
        torch.Tensor: A tensor of shape (1, 2, H1, W1) representing the reconstructed object, with the
                      real and imaginary components split along the second dimension. Index 0 contains
                      the real part, and index 1 contains the imaginary part.
    """
    device = input.device
    # convert input into a complex array (S,H2,W2)
    input = np.squeeze(input.detach().cpu().numpy(), 0)
    input = input[:,0,:,:] + 1j * input[:,1,:,:]
    # convert probe into a complex array
    probe = np.squeeze(probe.detach().cpu().numpy(), 0)
    probe = probe[0,:,:] + 1j * probe[1,:,:]
    probe = np.expand_dims(probe, (0,1,2)) # expand probe dims to make it compatible with tike (1,1,1,H2,W2)
    with tike.operators.Ptycho(probe_shape=probe.shape[-1], detector_shape=int(probe.shape[-1]), nz=input.shape[-2], n=input.shape[-1]) as operator:
        farplane = operator.asarray(np.expand_dims(input, (1,2)), dtype=tike.precision.cfloating) #cfloating??
        scan = operator.asarray(scan, dtype=tike.precision.floating)
        probe = operator.asarray(probe, dtype=tike.precision.cfloating)
        psi = operator.asarray(np.zeros(object_size), dtype=tike.precision.cfloating)
        output = operator.adj(farplane=farplane, probe=probe, scan=scan, psi=psi)
        output = operator.asnumpy(output)
    # convert the result into a tensor with shape (1,2,H1,W1)
    output = np.expand_dims(output, 0)
    output = np.stack((np.real(output), np.imag(output)), 1)
    output = torch.from_numpy(output).float().to(device)
    return output

def cartesian_scan_pattern(object_size, probe_shape, step_size = 4, sigma = 0.5):
    """
    Generates a Cartesian scan pattern with optional random perturbations for ptychographic imaging.

    Args:
        object_size (tuple): A tuple (H1, W1) representing the height and width of the object to be scanned.
        probe_shape (tuple): A tuple (H2, W2) representing the height and width of the probe function.
        step_size (int, optional): The step size for the Cartesian grid in pixels. Default is 4.
        sigma (float, optional): The standard deviation for the Gaussian noise added as perturbation to the
                                 scan positions. Default is 0.5.

    Returns:
        numpy.ndarray: A numpy array of shape (N, 2) where N is the number of scan positions. Each row
                       contains the (x, y) coordinates of a scan position, possibly perturbed by a small
                       random value.
    """
    scan = []
    for y in range(0, object_size[0] - probe_shape[2] - 1, step_size):
        for x in range(0, object_size[1] - probe_shape[3] - 1, step_size):
            y_perturbation = sigma * np.random.randn()
            x_perturbation = sigma * np.random.randn()
            y_new = 1 + y + y_perturbation
            x_new = 1 + x + x_perturbation
            if x_new <= 1:
                x_new = 1 + x + np.abs(x_perturbation)
            if y_new <= 1:
                y_new = 1 + y + np.abs(y_perturbation)
            if x_new >= object_size[1] - probe_shape[3] + 1:
                x_new = 1 + x - x_perturbation
            if y_new >= object_size[0] - probe_shape[2] + 1:
                y_new = 1 + y - y_perturbation
            scan.append((y_new, x_new))
    scan = np.array(scan, dtype=np.float32)
    return scan

def create_disk_probe(size = (16, 16), width = 8.0, magnitude = 100.0):
    """
    Generates a synthetic disk-shaped complex probe for ptychographic imaging.

    Args:
        size (tuple, optional): A tuple (H2, W2) representing the height and width of the probe array.
                                Default is (16, 16).
        width (float, optional): The diameter of the disk-shaped probe in pixels. Default is 16.0.
        magnitude (float, optional): The magnitude of the complex values within the disk-shaped probe.
                                     Default is 100.0.

    Returns:
        numpy.ndarray: A 2D numpy array of shape `size` representing the complex-valued disk-shaped probe.
                       The values inside the disk have a magnitude specified by the `magnitude` parameter
                       and a phase shift of π/2. The values outside the disk are zero.
    """
    # Construct a grid
    x = np.linspace(-size[1]//2, size[1]//2, size[1])
    y = np.linspace(-size[0]//2, size[0]//2, size[0])
    xx, yy = np.meshgrid(x, y)
    # Create a circular disk
    r_squared = xx**2 + yy**2
    flat_top_region = r_squared <= (width/2)**2
    # Generate the synthetic probe
    complex_probe = magnitude * flat_top_region * np.exp(1j * np.pi / 2 * flat_top_region)
    return complex_probe

def l2_error(true_object, reconstructed_object):
    """
    Computes the L2 error (Euclidean distance) between the true object and the reconstructed object.

    Args:
        true_object (numpy.ndarray): A numpy array of shape (1, 2, H1, W1) representing the true object.
                                     The array contains complex values split into real and imaginary parts,
                                     with the real part at index 0 and the imaginary part at index 1 along
                                     the second dimension.
        reconstructed_object (numpy.ndarray): A numpy array of shape (1, 2, H1, W1) representing the reconstructed object.
                                              The array contains complex values split into real and imaginary parts,
                                              with the real part at index 0 and the imaginary part at index 1 along
                                              the second dimension.

    Returns:
        float: The L2 error, which is a non-negative real number representing the Euclidean distance between
               the true object and the reconstructed object.
    """
    # true object is (1,2,H1,W1)
    # reconstructed_object object is (1,2,H1,W1)
    term1 = np.vdot(reconstructed_object, reconstructed_object)
    term2 = np.vdot(true_object, true_object)
    term3 = - 2 * np.abs(np.vdot(reconstructed_object, true_object))
    l2_error = np.real(np.sqrt(term1 + term2 + term3))
    return l2_error


# Perform the adjoint test here.
if __name__ == '__main__':

    object_size = (512, 512)
    probe_size = (1, 2, 128, 128)

    probe = torch.randn(*probe_size)
    scan = cartesian_scan_pattern(object_size, probe_size, step_size = 32, sigma = 1)

    x = torch.randn(1, 2, *object_size)
    y_tilde = ptycho_forward_op(x, scan, probe)
    y_1 = torch.randn_like(y_tilde)
    x_tilde = ptycho_adjoint_op(y_1, scan, probe, object_size)

    # Convert everything back to complex tensors
    y_tilde = torch.complex(y_tilde[0,:,0,:,:], y_tilde[0,:,1,:,:])
    y_1 = torch.complex(y_1[0,:,0,:,:], y_1[0,:,1,:,:])
    x = torch.complex(x[0,0,:,:], x[0,1,:,:])
    x_tilde = torch.complex(x_tilde[0,0,:,:], x_tilde[0,1,:,:])

    print(torch.sum(torch.conj(y_1) * y_tilde))
    print(torch.sum(torch.conj(x) * x_tilde))
