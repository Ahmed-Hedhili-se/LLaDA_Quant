"""Outlier handling. Mostly already done, in two other places — see below.

This module is empty because weight-outlier remediation for the current scheme
exists elsewhere, not because nothing handles outliers.

**Groupwise scaling contains them structurally.** With ``group_size=128`` every
group carries its own scale, so a single large weight inflates the step size
for its own 128 neighbours and nobody else. Per-tensor scaling would let one
outlier degrade the whole matrix; grouping is what stops that, and it is the
default.

**Clipping search handles the rest**, and it lives in
:func:`~LLaDA_Quant.algorithms.symmetric.search_group_scale`. ``s = amax/Qmax``
sizes the grid around the largest weight in the group, which at 4 bits spends
most of 16 levels on one value. The search deliberately clips outliers whenever
the group's total squared error improves — that *is* outlier remediation, just
expressed as a scale choice rather than a separate pass. Measured: 12-14% lower
INT4 weight error at zero extra bytes, and ~0% at INT8, where 256 levels absorb
an outlier without help.

So what would actually go here?

**Per-channel remediation for activation quantization.** SmoothQuant-style
migration of activation outliers into the weights matters when activations are
quantized too (W8A8). It does nothing for weight-only quantization, which is
all this package does today.

**Mixed precision at the outlier level.** Keeping a small number of outlier
channels in BF16 (the LLM.int8() approach). This is the plausible next step for
INT4, and ``RESULTS.md`` section 4 now motivates it: the accuracy loss tracks
weight precision rather than a routing threshold, so spending bits where the
error concentrates should pay. It needs a sensitivity measurement first — which
layers and channels actually carry the damage — and that measurement does not
exist yet.

Do not add a generic ``handle_outliers()`` here. It would duplicate
``search_group_scale`` under a name that suggests it does something more.
"""
