"""Data-aware calibration. Deliberately empty — here is why, and what would fill it.

The scale this package computes is::

    s = max(|W_group|) / Qmax

There is no activation term in that expression. Calibration data cannot change
a single scale, so a ``calibrate(model, batches)`` function here would be
theatre: it would consume data, run forward passes, and produce a checkpoint
bit-identical to one built without it. That is worse than an empty file.

It stops being inert under either of two changes, neither of which is
implemented:

**Data-aware weight quantization.** GPTQ orders and error-compensates using a
Hessian estimated from real activations; AWQ searches per-channel scales by how
much each input channel actually matters. Both make the *weight* scale depend
on data, so both need this module.

**Activation quantization (W8A8).** Activations need ranges of their own, which
can only come from observed data.

Is it worth building? The measurement now says something concrete
(``RESULTS.md`` sections 3 and 4):

===================  ==================  ===================
scheme               weight rel. L2      GSM8K vs BF16
===================  ==================  ===================
INT8 g128            0.0065              -2.0 pt (p=0.585)
INT4-MSE g128        0.1011              -6.0 pt (p=0.179)
===================  ==================  ===================

INT4 carries ~15x INT8's weight error and the accuracy loss tracks that error,
so error reduction should convert into accuracy. MSE clipping search recovered
13%. AWQ-class methods typically recover 2-4x on INT4 — enough to narrow the
gap, not to close it.

The honest reading: this is worth building **only if INT4 specifically is
needed**. INT8 already costs no measurable accuracy and, fused, runs ~2x faster
than BF16. INT4's extra 2x memory saving matters when memory is the binding
constraint; on a 48 GB card holding a 14 GB model, it is not.

Fill this in when a measurement says INT4 is required, not before.
"""
