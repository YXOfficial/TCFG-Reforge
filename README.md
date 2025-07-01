# TCFG for ReForge

*modifies the unconditional score `ε_θ(x_t)` to `ε_θ'(x_t)`.*

Unlike ComfyUI which has a `pre-cfg` hook, ReForge does not. This script uses the available `post-cfg` hook to achieve the same result:

1.  It intercepts the sampler arguments *after* the standard CFG calculation is complete.
2.  It uses the provided `cond_denoised` and `uncond_denoised` predictions to calculate the original conditional and unconditional scores (`ε_cond` and `ε_uncond`).
3.  It applies the Tangential Damping formula to modify `ε_uncond`, creating a new `ε_uncond_final`.
4.  Finally, it manually recalculates the final guided result using the standard CFG formula: `denoised = uncond_final + cfg_scale * (cond - uncond_final)`.
5.  This new `denoised` tensor is returned to the sampler, effectively replacing the original CFG result.

This ensures the logic of TCFG is correctly implemented within the constraints of the ReForge extension system.
