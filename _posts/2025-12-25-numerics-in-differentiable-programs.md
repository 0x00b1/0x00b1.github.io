---
layout: post
title: "Numerics in Differentiable Programs"
date: 2025-12-25
---

## Numerics in Differentiable Programs

Numerical accuracy is an unfashionable subject in contemporary machine learning. In many contexts it is also, in a narrow sense, rationally unfashionable: the marginal utility of one more bit of floating-point fidelity is often dominated by the utility of more throughput, less latency, or more memory headroom. Rendering pipelines and graphics hardware institutionalized this tradeoff decades ago. Modern inference does the same thing with quantization, knowingly discarding information in order to meet deployment constraints.

The trouble begins when one forgets that training is not inference. Training is not a single evaluation; it is an iterative numerical procedure in which finite-precision error is not merely present, but coupled into the trajectory of the parameters and therefore able to be amplified. In program-like models, *ulp*-scale changes—where an **ulp** is the *unit in the last place*, the spacing between adjacent representable floating-point numbers at a given magnitude—can flip a solver iteration or event timing, which is a discrete change with long-horizon consequences.

The regimes I care about here are **differentiable world models**: training setups where the forward pass includes a *stateful, environment-like program*—a simulator, planner, renderer, or solver-heavy pipeline unrolled over time and differentiated end-to-end. I am **not** using “world model” in the narrower sense of a purely learned latent dynamics predictor; I mean “the world” as an executed program, potentially with learned components embedded inside it.

Two terms will recur:

* An **event** is a discrete control-flow change triggered by crossing a threshold (contact activates, visibility toggles, a neighbor interaction enters/leaves a set, etc.).
* A **termination boundary** is a threshold at which a stopping rule changes (a residual drops below `tol`, an iteration cap is hit, a line search accepts/rejects), changing which iterations execute.

Stated plainly:

> In differentiable world models, numerical settings are not backend trivia. They change the program, and therefore they change the gradient field you are optimizing.

---

## A running example: a bouncing ball that is “a program,” not just a function

To keep the discussion concrete without turning it into a numerical analysis textbook, I will use a small running example throughout: a differentiable bouncing ball with ground contact, unrolled over a horizon $T$, with a **learnable parameter inside the simulator**.

For concreteness, take a scalar parameter $k$ to be learnable. In the penalty contact variant below it plays the role of stiffness; in the projection variant it can be interpreted as a solver gain / compliance / relaxation parameter that controls how aggressively constraint violations are corrected. The forward computation is a rollout. The "model" is a program: it integrates dynamics, detects contact, and enforces constraints. The backward computation differentiates through that program.

At step $t$ the state is position $y_t$ and velocity $v_t$. Gravity is $g$. Step size is $h$. We roll out for $t=0,\ldots,T-1$.

There are two plausible contact implementations that look similar in forward simulation under many settings but behave very differently once you differentiate through them, especially over long horizons and near thresholds.

In a penalty (soft) contact model, define penetration

$$
p_t = \max(0, -y_t)
$$

and apply a contact force

$$
f^c_t = k \cdot p_t
$$

(optionally with damping). Large stiffness $k$ makes the dynamics stiff.

In a projection / impulse model, after an unconstrained step you enforce $y \ge 0$ (and optionally restitution) using an iterative projection. You iterate until a residual (for example, total penetration) is below a tolerance `tol` or you hit a cap `max_iter`. The iteration count is therefore part of the computation.

Because this post is about how numerical details change the *differentiated* program, I will assume the iterative projection is differentiated by **unrolling the loop** and backpropagating through the iterations that actually ran. Later, when I discuss implicit solves, I will return to the alternative: treating the projection as a fixed point and applying implicit differentiation rather than unrolling.

A minimal “world step” sketch that makes the control flow explicit is below. It is not meant to be faithful physics; it is meant to show where branches, loops, and reductions enter.

```python
def world_step(y, v, u, *, h, g, k,
               contact="projection", tol=1e-4, max_iter=25, restitution=0.8):
    # semi-implicit Euler (one possible choice)
    v_free = v + h * (u - g)
    y_free = y + h * v_free

    if contact == "penalty":
        p = relu(-y_free)                   # penetration (nonsmooth at 0)
        f_c = k * p                         # k acts like stiffness here
        v = v_free + h * f_c                # stiff when k is large
        return y_free, v, p, 0

    # projection / impulse "solver" (assume this loop is unrolled for autograd)
    # 1D contact could project in one step; iteration stands in for coupled multi-constraint solves.
    y = y_free
    it = 0
    alpha = 0.5                             # damped correction; coupled projections converge iteratively
    p = relu(-y)                            # ensure p is defined even if the loop doesn't run

    while it < max_iter:
        p = relu(-y)

        # If batched, this sum is a reduction on a termination boundary.
        #
        # UNSAFE under autocast / reduced-precision accumulation:
        # - reduction is performed in the current autocast dtype
        # - small rounding differences can flip S < tol, changing the number of iterations (a different program path)
        # if p.sum() < tol:
        #     break
        #
        # Safer: promote the termination reduction to FP32 to reduce branch flips.
        if p.float().sum() < tol:
            break

        # Here k plays the role of a solver gain / compliance-like parameter:
        # it controls how aggressively penetration is corrected per iteration.
        y = y + alpha * k * p
        it += 1

    y = relu(y)                             # enforce y >= 0 (schematic)
    v = where(y_free < 0, -restitution * clamp(v_free, max=0.0), v_free)
    return y, v, p, it
```

Even in this toy, the structural ingredients that make differentiable world model numerics different from "just a deeper network" are already present: long-horizon statefulness (outputs become inputs), nonsmooth events (contact activates at $y=0$), algorithmic inner loops (projection iterations and stopping criteria), and reductions that can sit on termination boundaries (the summed penetration in the stopping rule).

---

## Training is a discrete dynamical system (and gradients live on its trajectory)

If you work in PyTorch long enough, you encounter failures that are simultaneously mundane and hard to reason about: a run that is stable in full precision and unstable under mixed precision; a model that trains on one GPU but not another; a refactor that is algebraically equivalent but changes convergence; a loss curve that looks well behaved until the first step that produces NaNs. These incidents are often treated as folklore—seed everything, lower the learning rate, add gradient clipping, try a different optimizer—and those interventions sometimes help. These are often compensatory measures that treat symptoms rather than the numerical program.

The pattern underneath is less mystical. Training is an iterative numerical procedure. A small perturbation in one step changes the state for the next. In a long rollout, a perturbation can move an event across a threshold (contact happens one step earlier), which changes the computation path, which changes all subsequent states and gradients.

If you want a mental picture, imagine plotting $y_t$ versus $t$ for two runs that are identical except for a tiny numerical perturbation. The curves lie on top of each other until the moment one trajectory crosses the contact threshold a single step earlier. At that instant the **first-contact index differs by one step**, and from there the bounce phase drifts; subsequent contacts occur at different steps, and by the end of the horizon the histories are visibly different. The “diagram” is two nearly coincident trajectories that fork at the first threshold flip.

In the bouncing-ball rollout, this sensitivity is visible even before talking about floating point. Changing $h$ shifts the bounce timing; changing $k$ changes stiffness and stability margins; changing the projection tolerance `tol` changes the number of inner iterations. These are not implementation details; they are definitional parameters of the computation. Differentiable world models force this point into the foreground because the forward pass contains more parameters that define an algorithm, not merely parameters that define a smooth function.

---

## The numerical program

In conventional discussions, numerical accuracy is often framed as the gap between an ideal real-number computation and the finite-precision result of evaluating the same mathematical expression. In differentiable world models, that framing is incomplete because the forward computation is itself a numerical method.

A simulator is not a single function; it is an approximation procedure. An integrator is a discretization of a continuous system, parameterized by a step size. An iterative solver is a convergence process, parameterized by a tolerance and a stopping criterion. A differentiable renderer is often an estimator, parameterized by sample counts and variance-reduction choices, and it is usually made differentiable by replacing nonsmooth events (visibility, occlusion) with surrogates.

When you backpropagate through such pipelines, the derivative you obtain is the derivative of the implemented algorithm—the discretized, finite-precision program with its tolerances and surrogates—not the derivative of an idealized continuous world. The object of differentiation is the discretized map $F_{\theta,h,\mathrm{tol},\ldots}$, not an underlying continuous-time idealization.

This observation has a classical name in numerical analysis. There is a real distinction between “differentiate then discretize” (derive sensitivities of a continuous-time or continuous-space model, then discretize those sensitivity equations) and “discretize then differentiate” (first choose a discrete algorithm, then differentiate the discrete computation). Autodiff frameworks, by design, implement the latter: they take the program you wrote—your discretization, your tolerance-based loop, your branch logic—and differentiate *that*.

In the bouncing-ball toy, even in exact arithmetic, the program depends on choices: explicit Euler versus semi-implicit Euler defines a different discrete map; penalty contact versus projection defines a different map near $y=0$; and a projection with `tol` and `max_iter` defines a mapping that changes when termination changes. Changing these is not “just changing rounding error.” It is changing the function being differentiated.

To speak precisely without turning this into a numerical analysis textbook, it helps to keep a few notions separate. The following is not deep theory; it is a vocabulary for talking about failure modes without collapsing everything into “precision issues.”

> **Precision** is the granularity and dynamic range of the number system (FP32 vs FP16 vs bfloat16; reduced-mantissa internal modes). **Accuracy** is closeness to an intended reference semantics. **Stability** is whether small perturbations remain controlled or are amplified by the computation. **Conditioning** is sensitivity of the underlying problem; even a stable algorithm cannot overcome severe ill-conditioning. **Discretization/modeling error** is the gap between a continuous or idealized process and the discrete approximation you simulate. Instability can come from either the method or the representation; they often co-occur.

Differentiable world models force discretization error into the foreground because it exists even in exact arithmetic; yet it interacts with floating-point error, because discretization stability governs whether finite-precision perturbations remain benign.

---

## When tiny rounding becomes a different program

The mechanism that turns “tiny numerical differences” into qualitatively meaningful behavior in differentiable world models is not that the rounding error is larger. It is that the program is more sensitive: many programs make decisions based on quantities computed by reductions, and those decisions change the computation path.

A clarifier up front: **not all rounding changes the “program.”** In smooth regions of a computation, rounding typically perturbs outputs continuously. The qualitative change happens when those perturbations interact with **discrete control flow**: events, stopping rules, acceptance tests—anything that makes a piecewise-defined program path.

Floating-point arithmetic is not associative. The order in which a reduction accumulates terms affects the final rounded result. In scalar code this can often be ignored. In parallel code it cannot, because parallel reductions impose a reduction tree, and that tree depends on tiling, vectorization, warps, blocks, kernel selection, and sometimes nondeterministic scheduling.

In a conventional neural network, a low-order change in a reduction often manifests as a low-order change in an activation. Over many training steps, that can still lead to trajectory divergence, but the divergence often looks like noise. In differentiable world models, reductions frequently sit on decision boundaries: force accumulation, residual norms, constraint penalty sums, Monte Carlo integrals. These are the quantities that determine solver termination, event triggering, and stability.

A minimal mechanism is:

* termination is based on a scalar reduction $S$ and a threshold $\texttt{tol}$,
* two numerically valid executions produce $S_A$ and $S_B$ close to $\texttt{tol}$,
* one sees $S < \texttt{tol}$ and exits; the other does one more iteration.

Concretely, suppose the projection variant terminates when total penetration is below a tolerance:

$$
\text{stop if } \sum_{i=1}^N \max(0, -y^{(i)}) < \texttt{tol}.
$$

Near $\texttt{tol}$, ulp-scale differences can matter. If, due to different reduction trees or accumulation pathways, two executions produce slightly different rounded totals near the boundary:

$$
S_A = 9.96\times 10^{-4} \quad \text{and} \quad S_B = 1.004\times 10^{-3},
$$

then one run sees $S_A < \texttt{tol}$ and exits the loop; the other sees $S_B \ge \texttt{tol}$ and performs one more projection iteration. That extra iteration is not a “low-order digit” change. It is a different program path: the state after the step changes discontinuously with respect to the reduction outcome, and over a horizon it can shift the entire event history.

This connects directly to differentiation-through-unrolled-solvers: as a function of inputs and parameters, the **iteration count is piecewise constant**. If you unroll “the iterations that actually ran,” the overall mapping is **piecewise smooth**, and the gradient you get is the derivative of the executed branch of the loop. Along iteration-count boundaries, that derivative can **jump**.

If termination is based on a global reduction, it is often worth asking whether a single global sum should be the branch point at all. Per-instance termination, a small amount of hysteresis in the stopping rule, or computing the reduction in FP32/FP64 (ideally with a deterministic accumulation order) can all make the program less knife-edge without changing the high-level intent.

This is also where determinism controls become valuable, and where they are often misunderstood. Determinism is a property of execution, not a guarantee of mathematical accuracy. PyTorch’s `torch.use_deterministic_algorithms(True)` is described as forcing operations to use deterministic algorithms when available and raising an error when only nondeterministic implementations exist; “deterministic” here means producing the same output given the same input when run on the same software, driver/runtime, and hardware stack.  As always, this reproducibility is scoped to that fixed configuration; it is an observability tool, not a mathematical certificate.

The value of determinism is attribution. It reduces the confounding variable of schedule noise and atomic interleavings so you can ask controlled questions about the numerical program: if I change TF32 policy, does the first contact step change? If I change `tol`, does the iteration count distribution change? With determinism off, quantities like “first contact step” and “projection iteration count per step” can vary run-to-run when computation sits near a termination boundary; with determinism on, they often stabilize into exact reproducibility on a fixed configuration. That does not mean the computation is more accurate. It means you can now observe causality.

---

## Micro-demo 1: reduction order flips a termination decision

The following is a deliberately minimal, *deterministic* reproduction of the mechanism above. It is not meant to model a specific GPU reduction tree; it shows that **different accumulation orders in finite precision can cross a termination boundary**, changing control flow.

The trick is to make a sum "near $\texttt{tol}$" out of one term just below $\texttt{tol}$ plus several sub-ulp terms that are either retained (if accumulated while the running sum is small) or lost (if accumulated after the running sum is large).

```python
import torch

# Choose a convenient tol near 1e-3 but exactly representable in binary.
tol = torch.tensor(2**-10, dtype=torch.float16)   # ~9.765625e-4

# A value one ulp below tol at this scale (in float16 here, ulp is ~4.768e-7).
L = torch.tensor(float(tol) - 2**-21, dtype=torch.float16)

# A sub-ulp increment (quarter-ulp at this scale).
s = torch.tensor(2**-23, dtype=torch.float16)

p = torch.stack([L, s, s, s, s])                  # total is exactly tol in real arithmetic

def fp16_seq_sum(x):
    acc = torch.zeros((), dtype=torch.float16)
    for xi in x:
        acc = (acc + xi).half()                   # force rounding to float16 after each add
    return acc

S_large_first = fp16_seq_sum(p)                   # add big term first, then tiny terms
S_small_first = fp16_seq_sum(torch.cat([p[1:], p[:1]]))  # add tiny terms first

print("tol         =", float(tol))
print("S_large_first=", float(S_large_first), " stop?", bool(S_large_first < tol))
print("S_small_first=", float(S_small_first), " stop?", bool(S_small_first < tol))

# Typical output:
# tol          = 0.0009765625
# S_large_first= 0.0009760856628   stop? True
# S_small_first= 0.0009765625      stop? False
```

Both executions are “numerically reasonable,” but they do different things at the termination boundary. In a solver, that translates into a different iteration count. In an unrolled solver, a different iteration count means a different executed program and, consequently, a potentially different gradient.

---

## Long-horizon sensitivity: integration, solves, and nonsmooth events

At this point it is tempting to treat backend differences as the main story. They matter, but they are still secondary to the central amplifier in differentiable world models: long-horizon, stateful algorithms.

Consider a parameterized discrete dynamical system

$$
z_{t+1} = F_\theta(z_t, u_t), \qquad t = 0,\ldots,T-1,
$$

with a loss accumulated along the trajectory. Reverse-mode differentiation of an unrolled program propagates sensitivities backward via Jacobian products along the rollout. Even before considering floating-point effects, the numerical difficulty is visible: if the relevant Jacobians have singular values greater than one in important directions, sensitivities grow; if less than one, they decay. In stiff systems both regimes can occur across different modes; in chaotic regimes sensitivity can grow rapidly with horizon.

The computed gradient is therefore a fragile object even in exact arithmetic. Finite-precision effects, reduction order, discretization perturbations, and branch flips are injected into this propagation and can become dominant over long horizons. This is why it is misleading to conceptualize differentiable world model rollouts as “just deeper networks.” The depth is algorithmic depth: repeated application of a transition operator that embodies physics and event logic.

Time integration makes this concrete. An ODE integrator with step size $h$ is a discretization of a continuous system. Its stability properties are governed by the method and by $h$. If the system is stiff, explicit methods may require very small $h$ to be stable. If $h$ is too large, the discretized program can diverge even in exact arithmetic; floating-point precision merely shifts the boundary of failure.

When you differentiate through the integrator, stability can become even more delicate: forward trajectories may remain bounded while adjoint or sensitivity variables blow up, because backward propagation reflects the sensitivity of the discretized mapping. This is one of the most common sources of confusion in practice: the forward simulation “looks fine,” but the first NaN appears during backpropagation. In differentiable world models, this is not an oddity; it is a predictable consequence of long-horizon sensitivity propagation through a discretized method.

Penalty contact makes stiffness explicit. Increasing $k$ sharpens the contact response but also increases stiffness. With an explicit integrator, large $k$ can be forward-stable only for sufficiently small $h$. It is common to find regimes where the forward rollout appears stable and physical enough, yet gradients spike near contact boundaries and eventually overflow or produce NaNs in backward.

A similar phenomenon occurs for implicit methods and constraint projections, which are ubiquitous in realistic simulators. Many physically constrained systems compute a state by solving an implicit equation $g(x,\theta)=0$, either exactly or approximately via an iterative solver. Differentiating through an iterative solver by unrolling iterations is conceptually straightforward but can be numerically punishing: memory costs grow with iteration count, and the gradient is sensitive to the solver's convergence path. The alternative is implicit differentiation: under appropriate regularity assumptions, $\partial x/\partial \theta$ can be expressed via a linear system involving $\partial g/\partial x$. The important practical consequence is that the backward computation now contains its own numerical method: it requires solving linear systems (often repeatedly), and conditioning and solver tolerances become explicit components of the gradient.

A caveat matters in exactly the regimes differentiable world models inhabit: contact, friction, and complementarity formulations are often **nonsmooth**, and the regularity conditions behind implicit differentiation (differentiability of $g$ and invertibility / well-conditioning of $\partial g/\partial x$ at the solution) can fail without smoothing, regularization, or a reformulation. In practice, many systems that use implicit sensitivities do so on a regularized surrogate of the physical constraint set.

Nonsmoothness is the other structural difference. Many differentiable world model programs are inherently nonsmooth: contact and friction laws, collision handling, visibility and occlusion, branch-based event logic, thresholding and clamping used for stability. Classical derivatives often do not exist at these events. Differentiable implementations therefore employ surrogates: smooth approximations to contact, softened visibility, differentiable relaxations of discrete decisions, stochastic estimators with reparameterization tricks. These surrogates are not mere implementation details; they define what gradient means and therefore define the effective optimization problem.

In the bouncing-ball toy, the hard event is "contact activates at $y=0$." Even in penalty contact, activation is thresholded by $p_t=\max(0,-y_t)$, which is nonsmooth at zero. Replacing $\max(0,\cdot)$ with a smooth approximation (for example, a softplus with width $\varepsilon$) does not merely make gradients "nicer." It defines a different optimization problem: you are now optimizing a system with softened contact near the boundary. This is often the right move, and the right framing is to treat it as a modeling choice, not a numerical patch.

A useful aside that appears in many real simulators is event logic that flips topology, not merely a branch. A canonical example is a cutoff-based neighbor list in molecular dynamics: include pair $(i,j)$ only when $d_{ij} < r_c$. That membership test is a branch. If a pair sits near the cutoff, a tiny numerical perturbation flips whether the interaction exists. Over a long rollout, the force set changes, positions change, future neighbor lists change, and trajectories diverge. Two robustness moves in MD—switching functions (soft cutoffs) and Verlet lists with a skin—are not merely performance tricks; they are ways of making the numerical program less knife-edge.

Stochasticity complicates diagnosis in the same way. Estimator variance (for example in differentiable rendering) is conceptually distinct from floating-point perturbations, but in a sensitive program the two can interact: a small change in sampling sequence and a small change in reduction order can both flip the same event. This is the practical justification for deterministic debugging modes and aggressive control of randomness during diagnosis. The goal is not to eliminate stochasticity from the method; it is to isolate sources of variation so instability can be attributed and mitigated.

---

## Micro-demo 2: reduced precision flips solver termination (and deletes a gradient)

The previous demo isolates reduction-order sensitivity in the abstract. This one shows the same mechanism in a tiny unrolled "solver" where **a learnable parameter $k$** lives *inside* the iterative correction step.

The logging is:

* first-contact index (here it will be 0 by construction; the demo starts in penetration)
* per-step iteration count `it`
* trajectory deltas (here: deltas in the per-step penetration history)
* gradient norm deltas (w.r.t. the learnable parameter $k$)

The point is not that AMP “breaks training,” but that **if the termination decision is made in reduced precision**, you can cross a termination boundary and execute a different number of iterations. If the loop runs zero iterations, the learnable parameter inside the loop can become *unused*, and the gradient w.r.t. that parameter can disappear entirely.

To make this easy to reproduce, the script below constructs a batch whose penetration sum is **exactly on the termination boundary** in real arithmetic, but can fall below it under a deliberately fragile low-precision accumulation. It then compares:

* **FP32** with an unsafe termination reduction,
* **“AMP”** (here meaning: execute in reduced precision, float16 on CUDA or bfloat16 on CPU) with the same unsafe termination reduction,
* **AMP with a mitigation**: compute the termination reduction in FP32 (`p.float().sum()`).

```python
import torch

def fragile_sum(x: torch.Tensor) -> torch.Tensor:
    # Deliberately fragile reduction: sequentially accumulate in x.dtype,
    # rounding each add. This models "termination reduction in reduced precision."
    acc = torch.zeros((), dtype=x.dtype, device=x.device)
    for xi in x.reshape(-1):
        acc = (acc + xi).to(x.dtype)
    return acc


def make_boundary_batch(*, device: str, low_dtype: torch.dtype, m_small: int = 8, tol_value: float = 1e-3):
    # Construct p0 so that in real arithmetic sum(p0) == tol,
    # but under low-precision sequential accumulation the sum can drop below tol.
    tol_low = torch.tensor(tol_value, device=device, dtype=low_dtype)
    inf_low = torch.tensor(float("inf"), device=device, dtype=low_dtype)
    ulp_low = torch.nextafter(tol_low, inf_low) - tol_low
    if ulp_low.item() == 0.0:
        raise RuntimeError("ulp_low is zero; pick a larger tol_value or different dtype.")

    L_low = tol_low - ulp_low
    s_low = (ulp_low / m_small).to(low_dtype)
    if s_low.item() == 0.0:
        raise RuntimeError("s_low rounded to 0; reduce m_small or increase tol_value.")

    p_low = torch.cat([L_low.view(1), s_low.repeat(m_small)], dim=0)  # length = m_small + 1
    p0 = p_low.float()  # keep float32 master copy
    return p0, float(tol_low.float())


def projection_step(y: torch.Tensor, *, k: torch.Tensor, tol: float, max_iter: int,
                    alpha: float = 0.5, unsafe_termination: bool = True):
    """
    Iteratively reduce penetration p = relu(-y) by updating:
        y <- y + alpha * k * p

    Termination boundary:
        stop if sum(p) < tol

    unsafe_termination=True uses fragile_sum(p) in the active dtype.
    unsafe_termination=False uses p.float().sum() (FP32 termination reduction).
    """
    it = 0
    k_eff = k.to(y.dtype)
    p = torch.relu(-y)

    while it < max_iter:
        p = torch.relu(-y)

        S = fragile_sum(p) if unsafe_termination else p.float().sum()

        # Python branch: iteration-count boundaries are explicit (piecewise program).
        if (S < tol).item():
            break

        y = y + alpha * k_eff * p
        it += 1

    return y, p, it


def rollout(k: torch.Tensor, *, y0: torch.Tensor, T: int, tol: float, max_iter: int,
           dtype: torch.dtype, unsafe_termination: bool):
    y = y0.to(dtype)
    it_hist = []
    p_hist = []
    first_contact = None

    for t in range(T):
        # Define "contact event" as penetration present at step start.
        if first_contact is None and (y.detach() < 0).any().item():
            first_contact = t

        y, p, it = projection_step(
            y, k=k, tol=tol, max_iter=max_iter,
            alpha=0.5, unsafe_termination=unsafe_termination
        )

        it_hist.append(int(it))
        p_hist.append(p.detach().float())

    p_hist = torch.stack(p_hist, dim=0)  # [T, N]
    loss = p.float().pow(2).mean()
    return loss, first_contact, it_hist, p_hist


def run_once(*, use_amp: bool, unsafe_termination: bool,
             y0: torch.Tensor, T: int, tol: float, max_iter: int,
             device: str, low_dtype: torch.dtype):
    k = torch.tensor(0.5, device=device, requires_grad=True)  # learnable solver gain / compliance parameter

    dtype = low_dtype if use_amp else torch.float32

    loss, first_contact, it_hist, p_hist = rollout(
        k, y0=y0, T=T, tol=tol, max_iter=max_iter,
        dtype=dtype, unsafe_termination=unsafe_termination,
    )

    # If the loop exits before ever using k, loss may not depend on k.
    loss_requires_grad = bool(loss.requires_grad)
    if not loss_requires_grad:
        grad = torch.zeros_like(k)
    else:
        grad = torch.autograd.grad(loss, k, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(k)

    return {
        "loss": float(loss.detach().cpu()),
        "first_contact": first_contact,
        "it_hist": it_hist,
        "p_hist": p_hist.cpu(),
        "grad_k": float(grad.detach().cpu()),
        "grad_k_abs": float(grad.detach().abs().cpu()),
        "loss_requires_grad": loss_requires_grad,
    }


def summarize(a, b, name_a="A", name_b="B"):
    dp = (a["p_hist"] - b["p_hist"]).abs()
    print(f"{name_a}: loss={a['loss']:.6e} first_contact={a['first_contact']} "
          f"|grad_k|={a['grad_k_abs']:.6e} loss_requires_grad={a['loss_requires_grad']}")
    print(f"{name_b}: loss={b['loss']:.6e} first_contact={b['first_contact']} "
          f"|grad_k|={b['grad_k_abs']:.6e} loss_requires_grad={b['loss_requires_grad']}")
    print("it_hist A:", a["it_hist"])
    print("it_hist B:", b["it_hist"])
    print("max |Δp_t| over horizon:", float(dp.max()))
    print("grad_k (A vs B):", a["grad_k"], b["grad_k"])
    print()


device = "cuda" if torch.cuda.is_available() else "cpu"
low_dtype = torch.float16 if device == "cuda" else torch.bfloat16

# Construct an initial penetration batch sitting on a termination boundary in low precision.
p0, tol = make_boundary_batch(device=device, low_dtype=low_dtype, m_small=8, tol_value=1e-3)
y0 = (-p0).to(torch.float32).to(device)  # y < 0 => p = relu(-y) = p0

T = 4
max_iter = 25

r_fp32_unsafe = run_once(use_amp=False, unsafe_termination=True,
                         y0=y0, T=T, tol=tol, max_iter=max_iter,
                         device=device, low_dtype=low_dtype)

r_amp_unsafe  = run_once(use_amp=True, unsafe_termination=True,
                         y0=y0, T=T, tol=tol, max_iter=max_iter,
                         device=device, low_dtype=low_dtype)

r_amp_safe    = run_once(use_amp=True, unsafe_termination=False,
                         y0=y0, T=T, tol=tol, max_iter=max_iter,
                         device=device, low_dtype=low_dtype)

print("=== FP32 vs AMP (UNSAFE termination reduction) ===")
summarize(r_fp32_unsafe, r_amp_unsafe, "FP32", "AMP-unsafe")

print("=== AMP unsafe vs AMP safe (mitigation: FP32 termination reduction) ===")
summarize(r_amp_unsafe, r_amp_safe, "AMP-unsafe", "AMP-safe")

# Typical output shape (numbers depend on device/dtype, but the pattern is the point):
# - FP32 does 1 iteration on step 0; AMP-unsafe does 0
# - AMP-unsafe loss does not depend on k (loss_requires_grad=False; grad_k=0)
# - Promoting the termination reduction to FP32 restores FP32 behavior and gradients
```

What to look for:

* `it_hist` differs (often on the first step): one execution does an iteration, the other exits immediately.
* The penetration history differs (`max |Δp_t| ...` is nonzero), because one run applied a correction and the other did not.
* Most importantly: if the reduced-precision run exits before ever executing `y = y + alpha * k * p`, then the learnable parameter (k) is *unused* and the loss may not require grad w.r.t. (k) at all. Promoting the termination reduction to FP32 typically restores the FP32 iteration behavior and therefore restores gradients.

---

## In PyTorch, the numerical program includes backend pathways

In ordinary deep learning, it is tempting to treat numerical behavior as a property of an operator and a dtype. PyTorch makes that abstraction difficult to sustain, because PyTorch is not a single arithmetic model. It is a heterogeneous tensor system: multiple devices and backends, multiple kernel families, optional compilation and fusion, sparse representations, complex dtypes, mixed precision and quantization.

The numerical behavior you observe is therefore an emergent property of an execution pathway. A minimal abstraction is the tuple “operator, dtype policy, device/backend, algorithm/kernel variant.” In differentiable world models, that tuple is not enough. You also have discretization choices (step sizes), solver tolerances and stopping criteria, event handling and surrogate smoothing, and estimator randomness and seeding. This enlarged object—the numerical program—is part of what you are experimenting with.

PyTorch’s internal precision modes matter in ways that surprise practitioners because user-visible dtypes do not fully specify internal arithmetic. On NVIDIA GPUs, TensorFloat32 (TF32) tensor cores can accelerate float32 matmuls and convolutions by rounding inputs to a reduced mantissa while maintaining float32 dynamic range via float32 accumulation. PyTorch’s CUDA semantics notes describe TF32 in these terms, and they emphasize that matmuls and convolutions are controlled separately by corresponding flags.  PyTorch also exposes `torch.set_float32_matmul_precision` to control float32 matrix multiplication internal precision on CUDA; some settings permit TF32-style pathways for float32 matmuls while others more strongly favor full float32 internal computation, and the documentation describes the intended tradeoff between speed and numerical fidelity. 
In many deep learning workloads, TF32 is acceptable and materially improves throughput. The sharper issue in differentiable world models is where linear algebra sits inside iterative methods: least squares steps, Gauss–Newton updates, implicit constraint solves, or backward passes that solve linear systems. In those contexts, reduced mantissa precision can influence residual reduction and iteration counts—and because solver termination is part of the program, those differences propagate into gradients. In practice it can be useful to treat “matmul inside a solver or termination-critical path” differently from “matmul inside a feature extractor,” even when both are nominally float32.

Mixed precision illustrates the same “numerical program” lens. PyTorch’s automatic mixed precision facilities are best understood as dtype policies rather than global dtype switches. Autocast chooses the precision for operations within a region to improve performance while maintaining accuracy; the recommended training pattern combines autocast with `torch.amp.GradScaler`.  The documentation also notes that older AMP interfaces have been consolidated in newer `torch.amp` APIs. 
In conventional neural training, AMP often “just works” because many networks tolerate small perturbations and key primitives have been engineered for mixed precision. In differentiable world models, the dominant failure mode is frequently representability rather than mere rounding. Programs contain quantities that can be legitimately outside FP16’s comfortable dynamic range: accumulated energies, long sums of forces, tiny residuals, penalty terms with large coefficients, and inverse scales that explode when denominators approach zero. Mixed precision also interacts with solver logic. Residual-based termination criteria can behave differently if residual norms are computed or accumulated in reduced precision; a solver may terminate early or fail to terminate. Because the solver path is part of the differentiated program, this changes gradients.

One pattern that often helps is to treat precision not as a global switch but as a localized policy: allow reduced precision in high-throughput arithmetic, while explicitly enforcing FP32 computation (or at least FP32 accumulation) around **decision and termination boundaries**—termination checks, event detection, and residual computations. A minimal sketch looks like:

```python
scaler = torch.amp.GradScaler("cuda")

for batch in data:
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        z = encode(batch.obs0)
        for t in range(T):
            z = world_step(z, batch.u[t])

    # keep decision/termination-critical computations out of autocast
    with torch.amp.autocast("cuda", enabled=False):
        # e.g., compute residual norms / termination margins / event flags / loss in FP32
        loss = loss_fn(z.float(), batch.targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

This does not make an unstable discretization stable, and it does not remove nonsmoothness pathologies. It increases representational margin and reduces precision-induced perturbations in the places most likely to flip events or termination.

Quantization is a more explicit example of numerical policy as program modification. In differentiable world models it is attractive for memory and throughput reasons, especially when a hybrid system contains learned components one wishes to deploy on constrained hardware. But the right conceptual framing is that quantization changes the program, and quantization-aware training optimizes a surrogate.

PyTorch's fake quantization modules make this explicit. The documentation for `torch.ao.quantization.fake_quantize.FakeQuantize` describes it as simulating quantize and dequantize operations during training time, and it specifies the forward computation as a clamp-round-scale transformation:

$$
x_{\text{out}} =
\big(\mathrm{clamp}(\mathrm{round}(x/\mathrm{scale} + \mathrm{zero\_point}),
\mathrm{quant\_min}, \mathrm{quant\_max}) - \mathrm{zero\_point}\big)\cdot \mathrm{scale}.
$$


The presence of `round` and `clamp` makes the forward nonsmooth; quantization-aware training relies on surrogate gradient conventions. In differentiable world models, the question is not “how close is the quantized computation to the real-number one,” but whether the surrogate-trained system behaves acceptably under quantized inference at the level that matters: event timing, termination behavior, and long-horizon stability. Quantizing a perception encoder may be benign; quantizing state variables that determine contact activation or solver termination may not be, unless you intentionally redesign the event logic to be robust under that representation.

Two final corners are worth noting because they show that “numerics” includes conventions and representations, not only floating-point formats, and because in differentiable world models these conventions can become decision-critical near singularities or event thresholds.

PyTorch supports autograd for complex tensors, and its documentation states that the gradient computed is the conjugate Wirtinger derivative, which is the convention that makes standard optimizers behave sensibly for real-valued losses on complex parameters.  In practice, complex pipelines often become numerically delicate where complex-to-real mappings introduce singular behavior: magnitudes $|z|$ and normalizations $z/|z|$ are ill-behaved near $z=0$, and phases $\arg(z)$ introduce discontinuities across branch cuts. Finite precision can drive small magnitudes to exact zero, and divide-by-magnitude patterns can then produce explosive sensitivities. When gradient checking such pipelines, PyTorch’s `torch.autograd.gradcheck` notes that for most complex functions considered for optimization, no notion of Jacobian exists in the usual sense; gradcheck verifies consistency of Wirtinger and conjugate Wirtinger derivatives under the assumption that the overall function has a real-valued output. 
Sparse and irregular computation is another case where representation semantics matter. Sparsity is pervasive in differentiable world models: neighbor interactions induce sparse patterns, contact graphs are sparse, mesh adjacency is sparse, constraint Jacobians are sparse. When one moves from dense to sparse representations, one often moves from smooth, regular kernels to irregular accumulation that may involve reductions and atomics, whose order and scheduling can matter for floating point.

PyTorch's sparse COO representation makes this concrete. The documentation for `torch.sparse_coo_tensor` notes that it returns an uncoalesced tensor when `is_coalesced` is unspecified or `None`. The sparse documentation warns about coalescing semantics and how values are accessed in a way compatible with autograd. The `Tensor.values()` documentation states that it can only be called on a coalesced sparse tensor. Coalescing combines duplicates by summation; summation is a reduction; reduction order matters; and sparse kernels may accumulate through atomics depending on the operation. In differentiable world models, where sparse structures often control event logic (contacts, neighbors), these details can be amplified into observable behavioral differences over long horizons.

---

## Practical checklist: making the numerical program observable and less knife-edge

The individual points above compound because the model is a stateful program with events, termination boundaries, and long-horizon sensitivity. A short checklist that matches that reality:

**Instrumentation (make the numerical program visible):**

* Log **event indices** (e.g., first contact step; visibility toggles; neighbor list size changes).
* Log **solver iteration counts** per step and distributions over batches.
* Log **termination margins**, e.g. $S - \texttt{tol}$ for any stopping rule based on a reduction.
* Log **residual norms** and whether termination was due to tolerance vs `max_iter`.
* Log **range checks** (max/min of key state variables; denominators; penalty terms) to catch representability issues before NaNs.

**Precision policy (localize reduced precision):**

* Keep high-throughput arithmetic under autocast where it is not decision-critical.
* Force FP32 (or FP32 accumulation) for:

  * termination checks and residual norms,
  * event detection tests,
  * reductions that decide control flow.

**Event and termination robustness (reduce branch flips):**

* Avoid global reductions as single branch points when possible (prefer per-instance termination).
* Add hysteresis to thresholds when semantics allow (separate enter/exit tolerances).
* If unrolling solvers, recognize that iteration-count boundaries imply **gradient jumps**; reduce sensitivity at those boundaries via smoothing/regularization when appropriate.

**Solver differentiation choices (make assumptions explicit):**

* If using implicit differentiation, state the regularity assumptions and what surrogate/regularization makes them plausible in nonsmooth regimes.

**Debugging discipline:**

* Use determinism controls to stabilize confounders and attribute causality; treat determinism as observability, not accuracy.

---

## Closing

Most of the individual facts in this post are not new. Reductions are non-associative; determinism is for debugging; TF32 and internal precision modes can change numerical behavior; mixed precision requires scaling; quantization uses surrogate gradients; complex autograd follows Wirtinger calculus conventions; sparse representations have coalescing semantics.

What changes in the differentiable world model regime is the way these facts compound. Differentiable world models put long-horizon sensitivity, iterative numerical methods, nonsmooth events, and termination boundaries at the core of the computation. The numerical program—the discretized, finite-precision, tolerance-parameterized, branch-containing algorithm you actually run—becomes part of the definition of the objective you are optimizing.

Backend heterogeneity and precision policies matter not because we have become obsessed with ulps, but because we are differentiating programs whose qualitative behavior can change when small perturbations alter event sequences or solver paths. That is also why "accuracy" is not a single notion here: you are not differentiating an idealized continuous world; you are differentiating $F_{\theta,h,\mathrm{tol},\ldots}$, the implemented map with all of its numerical choices.

When the model is a world, numerics are not the fine print. They are part of the experiment.
