import torch

def anneal_dsm_score_estimation(scorenet, samples, sigmas, anneal_power = 2.):
    labels = torch.randint(0, len(sigmas), (samples.shape[0],), device = samples.device)
    sigmas_labels = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * sigmas_labels

    perturbed_samples = samples + noise

    target = -1 / (sigmas_labels ** 2) * noise
    target = target.view(target.shape[0], -1)

    scores = scorenet(perturbed_samples, labels)
    scores = scores.view(scores.shape[0], -1)

    loss = 0.5 * ((scores - target)**2).sum(dim = -1) * sigmas_labels.squeeze()**anneal_power

    return loss.mean(dim = 0)
