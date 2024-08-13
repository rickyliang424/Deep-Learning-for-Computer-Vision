import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    loss = None

    loss = None
    real_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)

    real_loss = bce_loss(logits_real,real_labels)
    fake_loss = bce_loss(logits_fake,fake_labels)

    loss = (real_loss + fake_loss) / 2

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########

    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss.

    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    loss = None
    labels = torch.ones_like(logits_fake)
    loss = bce_loss(logits_fake,labels)

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    loss_real = 0.5 * torch.mean((scores_real - 1) ** 2)
    loss_fake = 0.5 * torch.mean(scores_fake ** 2)

    loss = loss_real + loss_fake


    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########

    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None
    loss = 0.5 * torch.mean((scores_fake - 1) ** 2)

    ####################################
    #          YOUR CODE HERE          #
    ####################################

    ##########       END      ##########

    return loss
