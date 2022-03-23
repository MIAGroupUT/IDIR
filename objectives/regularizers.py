import torch


def compute_hyper_elastic_loss(
    input_coords, output, batch_size=None, alpha_l=1, alpha_a=1, alpha_v=1
):
    """Compute the hyper-elastic regularization loss."""

    grad_u = compute_jacobian_matrix(input_coords, output, add_identity=False)
    grad_y = compute_jacobian_matrix(
        input_coords, output, add_identity=True
    )  # This is slow, faster to infer from grad_u

    # Compute length loss
    length_loss = torch.linalg.norm(grad_u, dim=(1, 2))
    length_loss = torch.pow(length_loss, 2)
    length_loss = torch.sum(length_loss)
    length_loss = 0.5 * alpha_l * length_loss

    # Compute cofactor matrices for the area loss
    cofactors = torch.zeros(batch_size, 3, 3)

    # Compute elements of cofactor matrices one by one (Ugliest solution ever?)
    cofactors[:, 0, 0] = torch.det(grad_y[:, 1:, 1:])
    cofactors[:, 0, 1] = torch.det(grad_y[:, 1:, 0::2])
    cofactors[:, 0, 2] = torch.det(grad_y[:, 1:, :2])
    cofactors[:, 1, 0] = torch.det(grad_y[:, 0::2, 1:])
    cofactors[:, 1, 1] = torch.det(grad_y[:, 0::2, 0::2])
    cofactors[:, 1, 2] = torch.det(grad_y[:, 0::2, :2])
    cofactors[:, 2, 0] = torch.det(grad_y[:, :2, 1:])
    cofactors[:, 2, 1] = torch.det(grad_y[:, :2, 0::2])
    cofactors[:, 2, 2] = torch.det(grad_y[:, :2, :2])

    # Compute area loss
    area_loss = torch.pow(cofactors, 2)
    area_loss = torch.sum(area_loss, dim=1)
    area_loss = area_loss - 1
    area_loss = torch.maximum(area_loss, torch.zeros_like(area_loss))
    area_loss = torch.pow(area_loss, 2)
    area_loss = torch.sum(area_loss)  # sum over dimension 1 and then 0
    area_loss = alpha_a * area_loss

    # Compute volume loss
    volume_loss = torch.det(grad_y)
    volume_loss = torch.mul(torch.pow(volume_loss - 1, 4), torch.pow(volume_loss, -2))
    volume_loss = torch.sum(volume_loss)
    volume_loss = alpha_v * volume_loss

    # Compute total loss
    loss = length_loss + area_loss + volume_loss

    return loss / batch_size


def compute_bending_energy(input_coords, output, batch_size=None):
    """Compute the bending energy."""

    jacobian_matrix = compute_jacobian_matrix(input_coords, output, add_identity=False)

    dx_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    dy_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    dz_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        dx_xyz[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 0])
        dy_xyz[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 1])
        dz_xyz[:, i, :] = gradient(input_coords, jacobian_matrix[:, i, 2])

    dx_xyz = torch.square(dx_xyz)
    dy_xyz = torch.square(dy_xyz)
    dz_xyz = torch.square(dz_xyz)

    loss = (
        torch.mean(dx_xyz[:, :, 0])
        + torch.mean(dy_xyz[:, :, 1])
        + torch.mean(dz_xyz[:, :, 2])
    )
    loss += (
        2 * torch.mean(dx_xyz[:, :, 1])
        + 2 * torch.mean(dx_xyz[:, :, 2])
        + torch.mean(dy_xyz[:, :, 2])
    )

    return loss / batch_size


def compute_jacobian_loss(input_coords, output, batch_size=None):
    """Compute the jacobian regularization loss."""

    # Compute Jacobian matrices
    jac = compute_jacobian_matrix(input_coords, output)

    # Compute determinants and take norm
    loss = torch.det(jac) - 1
    loss = torch.linalg.norm(loss, 1)

    return loss / batch_size


def compute_jacobian_matrix(input_coords, output, add_identity=True):
    """Compute the Jacobian matrix of the output wrt the input."""

    jacobian_matrix = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        jacobian_matrix[:, i, :] = gradient(input_coords, output[:, i])
        if add_identity:
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i])
    return jacobian_matrix


def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input."""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True
    )[0]
    return grad
