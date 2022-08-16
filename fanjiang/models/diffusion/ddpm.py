import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["linear_schedule_name", "cosine_schedule_name", "GaussianDiffusion", "DDPM"]


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def extract_lerp(arr, timesteps, broadcast_shape):
    timesteps = timesteps.float()
    frac = timesteps.frac()
    while len(frac.shape) < len(broadcast_shape):
        frac = frac[..., None]
    res_1 = extract(arr, timesteps.floor().long(), broadcast_shape)
    res_2 = extract(arr, timesteps.ceil().long(), broadcast_shape)
    return torch.lerp(res_1, res_2, frac)


def cosine_schedule_name(timesteps, s = 0.008, thres = 0.999):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, thres)

def linear_schedule_name(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        elif section_counts == "fast27":
            steps = space_timesteps(num_timesteps, "10,10,3,2,2")
            # Help reduce DDIM artifacts from noisiest timesteps.
            steps.remove(num_timesteps - 1)
            steps.add(num_timesteps - 3)
            return steps
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def update_beta(betas, timesteps):
    new_betas = []
    alphas_cumprod = torch.cumprod(1 - betas, axis = 0)
    last_alpha_cumprod = 1.0
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
    betas = torch.tensor(new_betas)
    return betas


def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def get_time_schedule(n_timestep):
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t

def get_sigma_schedule(n_timestep, beta_min=0.01, beta_max=20, use_geometric=False):
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas))
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas


def plot_schedule(N=27):
    import matplotlib.pyplot as plt
    linear_betas = linear_schedule_name(N)
    cosine_betas = cosine_schedule_name(N)
    _, a_s, betas = get_sigma_schedule(N)
    a_s_cum = torch.cumprod(a_s, dim=0)

    x = np.arange(N) / N

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(x, torch.cumprod(1-linear_betas, axis=0).numpy(), 'b', label="linear")
    ax[0].plot(x, torch.cumprod(1-cosine_betas, axis=0).numpy(), 'g', label="cosine")
    ax[0].plot(x, a_s_cum[:N].numpy(), 'r', label="vp")
    ax[0].set_ylabel(r'$\bar{\alpha_{t}}$')
    ax[0].set_xlabel("diffusion step (t/T)")

    ax[1].plot(x, linear_betas.numpy(), 'b', label="linear")
    ax[1].plot(x, cosine_betas.numpy(), 'g', label="cosine")
    ax[1].plot(x, betas[1:].numpy(), 'r', label="vp")
    ax[1].set_ylabel(r'$\beta_{t}$')
    ax[1].set_xlabel("diffusion step (t/T)")

    plt.legend()
    plt.show()



class FastDDPM(nn.Module):
    def __init__(self,
        N=4,
        beta_min=0.1,
        beta_max=20,
        use_geometric=False,
        *,
        unet,
    ):
        super().__init__()

        self.N = N
        self.unet = unet

        _, a_s, betas = get_sigma_schedule(N, beta_min, beta_max, use_geometric)
        a_s_cum = torch.cumprod(a_s, dim=0)
        sigmas_cum = (1 - a_s_cum ** 2).sqrt()
        a_s_prev = a_s.clone()
        a_s_prev[-1] = 1

        #we don't need the zeros
        betas = betas.type(torch.float32)[1:]
        alphas = 1 - betas

        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod))
        posterior_mean_coef2 = ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod))
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

        self.register_buffer('a_s_cum', a_s_cum, persistent=False)
        self.register_buffer('sigmas_cum', sigmas_cum, persistent=False)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1, persistent=False)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2, persistent=False)
        self.register_buffer('posterior_variance', posterior_variance, persistent=False)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped, persistent=False)


    def q_sample(self, x_start, t, *, noise=None):
        """
        Diffuse the data (t == 0 means diffused for t step)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = extract(self.a_s_cum, t, x_start.shape) * x_start + \
            extract(self.sigmas_cum, t, x_start.shape) * noise

        return x_t


    def p_sample(self, x_0, x_t, t):

        def q_posterior(x_0, x_t, t):
            mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            )
            var = extract(self.posterior_variance, t, x_t.shape)
            log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
            return mean, var, log_var_clipped


        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = rearrange(1 - (t == 0).float(), 'n -> n 1 1 1')
        sample = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return sample


    @torch.no_grad()
    def p_sample_loop(self, x_init, lowres_x=None, lowres_t=None, hist_x=None, continous=False):
        samples = []
        x = x_init
        for i in reversed(range(self.N)):
            t = torch.full((x.size(0),), i, dtype=torch.int64, device=x.device)
            t_time = t
            x_0 = self.unet(x, t_time, lowres_x=lowres_x, lowres_t=lowres_t, hist_x=hist_x)
            x_new = self.p_sample(x_0, x, t)
            x = x_new.detach()
            samples.append(x)

        if continous:
            return samples

        return x


    def forward(self, x, lowres_x=None, lowres_t=None, hist_x=None):

        bs, _, img_h, img_w = x.shape

        t = torch.randint(0, self.N, (bs,), device=x.device)

        noise = torch.randn_like(x)

        xt = self.q_sample(x, t, noise=noise)

        pred = self.unet(xt, t, lowres_x=lowres_x, lowres_t=lowres_t, hist_x=hist_x)

        return pred


class GaussianDiffusion(nn.Module):
    def __init__(self, betas):
        super().__init__()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.num_timesteps = len(betas)

        # register buffer helper function to cast double back to float

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent = False)

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


    def get_times(self, batch_size, noise_level):
        device = self.betas.device
        return torch.full((batch_size,), int(self.num_timesteps * noise_level), device = device, dtype = torch.long)

    def sample_random_times(self, batch_size):
        device = self.betas.device
        return torch.randint(0, self.num_timesteps, (batch_size,), device = device, dtype = torch.long)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):

        noise = torch.randn_like(x_start) if noise is None else noise

        alphas_cumprod = extract(self.alphas_cumprod, t, t.shape)
        snr = 1. / alphas_cumprod - 1
        log_snr = -torch.log(snr.clamp(min=1e-12))

        noised = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return noised

    def q_sample_pair(self, x_start, t, noise=None):

        noise = torch.randn_like(x_start) if noise is None else noise

        n1 = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        n2 = (
            extract(self.sqrt_alphas_cumprod, t+1, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t+1, x_start.shape) * noise
        )

        return torch.cat([n1, n2], dim=1)


    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_eps_from_xstart(self, x_t, t, x_start):
       return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - x_start
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)



class Sampler(GaussianDiffusion):
    def __init__(self, betas, sample_name):
        timesteps = space_timesteps(len(betas), sample_name)
        timesteps = sorted(list(timesteps))
        betas = update_beta(betas, timesteps)
        super().__init__(betas)
        self.timesteps = timesteps



class DDPM(nn.Module):
    def __init__(
        self,
        N=1000,
        use_ddim=False,
        sample_name="fast27",
        schedule_name="cosine",
        clip_denoised=False,
        dynamic_threshold=0.9,
        *,
        unet,
    ):
        super().__init__()
        self.use_ddim = use_ddim
        self.clip_denoised = clip_denoised
        self.dynamic_threshold = dynamic_threshold

        self.unet = unet

        if schedule_name == "cosine":
            betas = cosine_schedule_name(N)
        elif schedule_name == "linear":
            betas = linear_schedule_name(N)
        else:
            raise NotImplementedError()

        self.schedule = GaussianDiffusion(betas)
        self.sampler = Sampler(betas, sample_name)


    def mapping(self, ts):
        ts = ts.float()
        frac = ts.frac()
        timesteps = ts.new_tensor(self.sampler.timesteps)
        ts_1 = timesteps[ts.floor().long()]
        ts_2 = timesteps[ts.ceil().long()]
        return torch.lerp(ts_1, ts_2, frac)


    def p_mean_variance(
        self,
        x,
        t,
        lowres_x=None,
        lowres_t=None,
        hist_x=None,
    ):

        x_pred = self.unet(x, self.mapping(t), lowres_x=lowres_x, lowres_t=lowres_t, hist_x=hist_x)

        model_mean, posterior_variance, posterior_log_variance = self.sampler.q_posterior(
            x_start=x_pred, x_t=x, t=t
        )

        return {
            "mean": model_mean,
            "var": posterior_variance,
            "log_var": posterior_log_variance,
            "x_pred": x_pred,
        }


    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        lowres_x=None,
        lowres_t=None,
        hist_x=None,
    ):
        out = self.p_mean_variance(
            x,
            t,
            lowres_x=lowres_x,
            lowres_t=lowres_t,
            hist_x=hist_x,
        )
        noise = torch.randn_like(x)

        nonzero_mask = rearrange((t != 0).float(), 'n -> n 1 1 1')

        sample = out["mean"] + nonzero_mask * (0.5 * out["log_var"]).exp() * noise

        return sample


    @torch.no_grad()
    def ddim_sample(
        self,
        x,
        t,
        lowres_x=None,
        lowres_t=None,
        hist_x=None,
        eta=1.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            x,
            t,
            lowres_x=lowres_x,
            lowres_t=lowres_t,
            hist_x=hist_x,
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self.sampler.predict_eps_from_xstart(x, t, out["x_pred"])

        alpha_bar =  extract(self.sampler.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.sampler.alphas_cumprod_prev, t, x.shape)

        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Equation 12.
        noise = torch.randn_like(x)

        mean_pred = (
            out["x_pred"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        nonzero_mask = rearrange((t != 0).float(), 'n -> n 1 1 1')

        sample = mean_pred + nonzero_mask * sigma * noise

        return sample

    def get_eps(
        self,
        x,
        t,
        lowres_x=None,
        lowres_t=None,
        hist_x=None
    ):
        x_pred = self.unet(x, self.mapping(t), lowres_x=lowres_x, lowres_t=lowres_t, hist_x=hist_x)
        eps = self.pred_eps_from_xstart(x, t, x_pred)
        return eps


    def pred_xstart_from_eps(
        self,
        x,
        t,
        eps,
    ):
        alpha_bar = extract_lerp(self.sampler.alphas_cumprod, t, x.shape)
        return (x - eps * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)

    def pred_eps_from_xstart(self, x_t, t, x_start):
        alpha_bar = extract_lerp(self.sampler.alphas_cumprod, t, x_t.shape)
        return (x_t / torch.sqrt(alpha_bar) - x_start) / torch.sqrt( 1 / alpha_bar - 1)

    def pndm_transfer(
        self,
        x,
        eps,
        t_1,
        t_2,
    ):
        pred_xstart = self.pred_xstart_from_eps(x, t_1, eps)
        alpha_bar_prev = extract_lerp(self.sampler.alphas_cumprod, t_2, x.shape)
        return pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev) * eps


    @torch.no_grad()
    def prk_sample(
        self,
        x,
        t,
        lowres_x=None,
        lowres_t=None,
        hist_x=None,
    ):
        t_mid = t.float() - 0.5
        t_prev = t - 1
        eps_1 = self.get_eps(x, t, lowres_x, lowres_t, hist_x)
        x_1 = self.pndm_transfer(x, eps_1, t, t_mid)
        eps_2 = self.get_eps(x_1, t_mid, lowres_x, lowres_t, hist_x)
        x_2 = self.pndm_transfer(x, eps_2, t, t_mid)
        eps_3 = self.get_eps(x_2, t_mid, lowres_x, lowres_t, hist_x)
        x_3 = self.pndm_transfer(x, eps_3, t, t_prev)
        eps_4 = self.get_eps(x_3, t_prev, lowres_x, lowres_t, hist_x)
        eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6
        sample = self.pndm_transfer(x, eps_prime, t, t_prev)

        return dict(
            sample=sample,
            eps=eps_prime,
        )


    @torch.no_grad()
    def plms_sample(
        self,
        x,
        t,
        eps_buff,
        lowres_x=None,
        lowres_t=None,
        hist_x=None,
    ):
        eps = self.get_eps(x, t, lowres_x, lowres_t, hist_x)
        eps_prime = (55 * eps - 59 * eps_buff[-1] + 37 * eps_buff[-2] - 9 * eps_buff[-3]) / 24
        sample = self.pndm_transfer(x, eps_prime, t, t - 1)

        return dict(
            sample=sample,
            eps=eps,
        )


    @torch.no_grad()
    def prk_sample_loop(self, x, lowres_x, lowres_t=None, hist_x=None):
        bs = x.shape[0]
        device = x.device

        N = self.sampler.num_timesteps
        indices = list(range(N))[::-1][1:-1]

        for i in indices:
            t = torch.tensor([i] * bs, device=device)
            out = self.prk_sample(
                x,
                t,
                lowres_x=lowres_x,
                lowres_t=lowres_t,
                hist_x=hist_x,
            )
            x = out["sample"]
        return x


    @torch.no_grad()
    def plms_sample_loop(self, x, lowres_x, lowres_t=None, hist_x=None):
        bs = x.shape[0]
        device = x.device

        N = self.sampler.num_timesteps
        indices = list(range(N))[::-1][1:-1]

        eps_buff = []
        for i in indices:
            t = torch.tensor([i] * bs, device=device)
            if len(eps_buff) < 3:
                out = self.prk_sample(
                    x,
                    t,
                    lowres_x=lowres_x,
                    lowres_t=lowres_t,
                    hist_x=hist_x,
                )
            else:
                out = self.plms_sample(
                    x,
                    t,
                    eps_buff,
                    lowres_x=lowres_x,
                    lowres_t=lowres_t,
                    hist_x=hist_x,
                )
                eps_buff.pop(0)

            eps_buff.append(out["eps"])
            x = out["sample"]

        return x


    @torch.no_grad()
    def p_sample_loop(self, x, lowres_x, lowres_t=None, hist_x=None, continous=False):
        bs = x.shape[0]
        device = x.device

        N = self.sampler.num_timesteps
        interval = max(N // 25, 1)

        imgs = []
        for i in reversed(range(N)):
            t = torch.tensor([i] * bs, device=device)

            if self.use_ddim:
                x = self.ddim_sample(x, t, lowres_x=lowres_x, lowres_t=lowres_t, hist_x=hist_x)
            else:
                x = self.p_sample(x, t, lowres_x=lowres_x, lowres_t=lowres_t, hist_x=hist_x)

            if i % interval == 0:
                imgs.append(x)

        if continous:
            return imgs

        return x


    @torch.no_grad()
    def q_sample_loop(self, x_start, interval=10):
        num_timesteps = self.schedule.num_timesteps
        imgs = []
        for i in range(num_timesteps):
            t = x_start.new_tensor([i], dtype=torch.long)
            xt = self.schedule.q_sample(x_start, t)

            if i % interval == 0:
                imgs.append(xt)

        return imgs


    def p2_weight(self, log_snr, gamma=0.0):
        weight = (1 + log_snr.exp()) ** -gamma
        return weight


    def forward(self, x, lowres_x=None, lowres_t=None, hist_x=None):

        if not self.training:
            return self.p_sample_loop(
                x,
                lowres_x=lowres_x,
                lowres_t=lowres_t,
                hist_x=hist_x
            )

        bs, _, img_h, img_w = x.shape

        t = self.schedule.sample_random_times(bs)

        noise = torch.randn_like(x)
        xt = self.schedule.q_sample(x, t, noise)


        if (t > 100):
            import matplotlib.pyplot as plt
            img = xt[0, 65].cpu().numpy()
            print(t, img.shape)
            plt.imshow(img)
            plt.savefig("noise.png", bbox_inches='tight', pad_inches=0.0, transparent='true', dpi=300)
            exit(0)

        pred = self.unet(xt, t, lowres_x=lowres_x, lowres_t=lowres_t, hist_x=hist_x)
        return pred



