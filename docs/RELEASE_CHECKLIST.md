# Author release checklist

- Confirm coauthor approval, institutional restrictions, journal disclosure policy and code/data redistribution rights. No license has been assigned by this packaging step; select a license only for materials the authors may license. Keep the repository private until clearance.
- Review all derived traffic CSVs, archived predictions, private network identifiers, source metadata, binary model files and third-party components before staging for public release. The complete datasets, manuscript, editorial correspondence and original ZIPs are not included here.
- Recover the missing originals listed in REPRODUCTION.md, or document them explicitly as unavailable. Preserve historic hashes and original source bytes. Do not bypass checks to make a rerun pass.
- Export the actual environment on the workstation (`python -m pip freeze > requirements-lock.txt`); review for local paths/private URLs before sharing. Use the appropriate CUDA build instructions.
- Test `verify_release.py`, the checksum manifest, synthetic self-tests and actual replay in an isolated copy; archive outputs and compare against the original metrics. A partial archive check is not independent reproduction.
- Perform a complete secret and privacy scan on the workstation, including any prior Git history; inspect what is staged with `git status --short` and `git diff --cached --stat`. The packaging environment performed a limited pattern scan only, not a comprehensive audit.
- After clearance, create a repository on GitHub, push the reviewed files, and verify the public URL and exact commit from a logged-out browser. Record the commit or a release tag in the manuscript; do not assert public availability before verification.
- A new release/commit is a new record, not proof of when older experiments were frozen. Retain original archives privately.
