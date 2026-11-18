"""Diagnose the U distribution produced by the new score pipeline.

The smoke test showed u_mean ~0.05, i.e. almost every entry lands on U=0. This
checks whether that is a bug or the expected consequence of the maths.
"""
import sys
import torch as th

N = 6
T = 50
B = 20


def stats(e, label):
    mu = e.mean(-1, keepdim=True)
    sd = e.std(-1, keepdim=True)
    z = (e - mu) / (sd + 1e-8)
    s = z.clamp(0, 3)
    print("  {:24s} e mean={:.3f} sd={:.3f} | z>=1 frac={:.3f} | s mean={:.3f}".format(
        label, float(e.mean()), float(sd.mean()),
        float((z >= 1).float().mean()), float(s.mean())))
    return s


print("\nRaw-error distributions and their normalised scores\n")

g = th.Generator().manual_seed(0)

# (a) independent noise: what the smoke test effectively produces
e_noise = th.rand(B, T, N, generator=g) * 10
s_noise = stats(e_noise, "independent noise")

# (b) one clear outlier per timestep: the situation the method targets
e_hot = th.rand(B, T, N, generator=g) * 0.1
e_hot[..., 0] += 5.0
s_hot = stats(e_hot, "one strong outlier")

# (c) heavy-tailed: one step in five is novel for everyone
e_mix = th.rand(B, T, N, generator=g) * 0.1
novel = th.rand(B, T, generator=g) < 0.2
e_mix[novel] += th.rand(int(novel.sum()), N, generator=g) * 5
s_mix = stats(e_mix, "20% novel timesteps")


def counts(scores, cap=3):
    fused = sum(scores[k] for k in scores) / 3.0 if isinstance(scores, dict) else scores
    return th.floor(fused).clamp(0, cap)


print("\nU distribution\n")
for label, s in (("independent noise", s_noise),
                 ("one strong outlier", s_hot),
                 ("20% novel timesteps", s_mix)):
    u = counts(s)
    hist = [float((u == k).float().mean()) for k in range(4)]
    print("  {:24s} mean={:.3f}  P(U=0..3)={}".format(
        label, float(u.float().mean()),
        " ".join("{:.2f}".format(h) for h in hist)))

print("\nInterpretation")
print("  With N=6 agents drawn i.i.d., roughly 1 sd above the mean happens for")
print("  ~1 agent, so after averaging three branches and flooring, most entries")
print("  sit at 0 and a minority reach 1. Reaching 3 requires several branches")
print("  to be ~3 sd high simultaneously, which is rare by construction.")
print("  The old code standardised across the whole batch and then floored,")
print("  which also concentrates mass at 0 but with a different shape.")

print("\nCheck against the old code path for reference\n")
old_like = th.rand(B, T, N, generator=g) * 10
mu = old_like.mean(dim=1, keepdim=True)      # standardise across TIME (old bug-ish)
sd = old_like.std(dim=1, keepdim=True) + 1e-8
old_u = th.clamp((old_like - mu) / sd, 0, 3).to(th.int)
hist = [float((old_u == k).float().mean()) for k in range(4)]
print("  old batch-level standardisation: mean={:.3f} P(U=0..3)={}".format(
    float(old_u.float().mean()), " ".join("{:.2f}".format(h) for h in hist)))
