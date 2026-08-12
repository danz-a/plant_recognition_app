# Migration guide — how to apply this scaffold

For: Christoph & Alexander · Branch: `work` · Time needed: ~30–45 min (plus notebook downloads)
Delete this file once migration is complete.

## Step 1 — Copy the scaffold in

Unzip this archive **into the repo root** so the new folders sit next to `streamlit/`.
Say yes to merging/overwriting — the new `README.md`, `requirements.txt` and `.gitignore`
intentionally replace the old root versions.

⚠ `.gitignore` starts with a dot — make sure it actually arrived (in VS Code you'll see it;
in Windows Explorer it's visible too, just easy to overlook).

## Step 2 — Delete old files

Work through `DELETE_CHECKLIST.md`, ticking as you go. Do this **before** filling folders,
so nothing old gets mixed in.

## Step 3 — Fill the folders

Every folder contains a `PLACE_FILES_HERE.md` saying exactly which file goes in from
Drive or Colab. Order doesn't matter, except: `models/` needs the export-notebook run
first (Christoph).

**Delete each `PLACE_FILES_HERE.md` once its folder is filled** — they must not be in
the submitted repo.

## Step 4 — Pin the requirements

After the export notebook has run, replace the `tensorflow-cpu` line in
`requirements.txt` with the exact printed version. One line, root file only.

## Step 5 — Commit, push, verify

Commit on `work`, push, then open the repo on github.com and check:
- folder tree matches the structure in `README.md`
- no `PLACE_FILES_HERE.md` left
- no `data/PlantVillage/` folder
- `models/` shows the two `.keras` files with sizes ~30 MB and ~3 MB

Merge `work` → `main` only once, at the end, before the Google-form submission.
