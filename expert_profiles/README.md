# Expert usage profiles

SAGER persists the hot-expert frequency table to a JSON profile so a
restarting server resumes warm instead of ramping cold.

- Location: `$SAGER_PROFILE_PATH` (no default; set it to something durable,
  e.g. `/var/lib/sager/profile.json`)
- Written: every 10 minutes and at engine exit (`cache_.save_profile`)
- Contents: decayed per-(layer, expert) frequency scores, layer/expert
  geometry for validation, and the residency snapshot

Fresh deployments start with all scores at zero; the first rebalance fills
the pool with an arbitrary (deterministic) selection and the table converges
within ~10 minutes of traffic. Keep the profile on local disk - it is small
(a few hundred KB) and written rarely.
