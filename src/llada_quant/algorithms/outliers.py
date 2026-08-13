"""Outlier handling strategies (planned for v0.4).

INT8 groupwise with group_size=128 already contains most outliers; larger
group sizes or INT4 will need per-channel outlier remediation here.
"""