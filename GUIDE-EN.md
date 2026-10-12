# PSE J-Lab — guide

Ordered by importance. The first five sections matter before you travel.

---

## 1. The first launch must happen at home

The Guardian learns each server's TLS fingerprint on first contact. Learn them
on a hotel's wifi and it learns whatever that network shows it — and trusts it
afterwards.

```
cd ~/Desktop/PSE/PSE\ J-Lab/app
./setup.sh                 # builds the venv from the bundled wheels, offline
./run.sh --selftest        # 20 checks, ~5 s
./run.sh                   # first run: wizard -> "Learn TLS pins now"
```

If `--selftest` fails anywhere, **do not trust a number this app prints** until
you know what failed.

## 2. What this app actually claims

That a network's Jacobian is exactly computable without an autodiff library:

```
J(x) = W_L · D_{L-1}(x) · W_{L-1} · … · D_1(x) · W_1,   D_l = diag(φ'(z_l))
```

and that everything else follows from it. That claim is checked four ways:
against finite differences, by comparing forward and reverse accumulation,
against answers known in advance (Fuss–Catalan moments, log|det J| in coupling
flows), and — if you install torch — directly against `torch.func.jacrev`.
The last agrees to **1e-16**, including the hand-derived gradient of the
Jacobian penalty.

Torch is required by no lab. It is only an independent witness:

```
./setup.sh --with-torch     # ~1 GB, offline from the wheelhouse
./run.sh --selftest         # now includes 30 comparisons against torch
```

## 3. The five laboratories, and how each one can mislead you

| Lab | Question | The trap |
|---|---|---|
| **L1 Isometry** | how the spectrum of J depends on depth, init, activation | χ=1 does not make a deep net trainable — the condition number still grows exponentially |
| **L2 NTK** | spectrum of Θ=J_θJ_θᵀ, label alignment, kernel drift | the learning rate **must** stay below 2n/λ_max, or you are comparing two divergences (hence `lr:"auto"`) |
| **L3 Robustness** | ‖J‖₂ and margin/gradient vs a real PGD attack | measure robust accuracy at **several** attack strengths; at one you get 100% or 0% and learn nothing |
| **L4 Dimension** | autoencoder Jacobian rank vs known intrinsic dimension | the estimator must ignore the numerical tail below the bottleneck rank, or it reports the bottleneck width every time |
| **L5 log\|det J\|** | analytic vs assembled vs numeric determinant | the strictest engine check; if it fails, no other lab's numbers are worth anything |

From the GUI: **Laboratory** tab → pick on the left → edit parameters (JSON) →
Run. From a terminal:

```
./run.sh --list
./run.sh --lab l1_depth_sweep --params '{"depths":[2,4,8,16,32],"width":64}'
```

## 4. One gate — why the app refuses your claims

Nothing enters the vault without backing:

- **COMPUTED** must point at a run directory whose sha256 still matches. Edit
  `result.json` afterwards and the check catches it; the card falls.
- **CITED** needs a reference that **resolves over the network**. A title that
  comes back different means the id was misremembered — the proposal is
  rejected, not quietly corrected.
- **SPECULATION** may not carry evidence. If it has evidence it is not speculation.
- **REFUTED** is never deleted.

A refusal is the app working, not failing. A vault that refuses nothing is a
notebook.

The model **never** writes a card. It proposes; you admit.

## 5. The Guardian — because you are on someone else's network

All traffic goes through one function. Everything else is refused: https only,
allowlisted hosts only, redirects re-checked at every hop, an API key sent only
to its own provider's endpoint, and every fetched document scanned for prompt
injection before a model sees it.

The **Guardian** tab shows the state. If nothing has been blocked or flagged for
a long session, the app says so — a Guardian that notices nothing is asleep.

---

## 6. Where things live

```
PSE J-Lab/
  app/            code, run.sh, setup.sh
  datasets/       MNIST + FashionMNIST + MANIFEST.json with hashes
  wheels/         311 MB of packages for offline installation
~/.local/share/pse-jlab/
  runs/           per experiment: params, result.json, sha256, meta, plots
  vault/          cards (SQLite) + topics + reference proposals
  state/          journal.sqlite (append-only), theta.json, keys.json (0600)
```

## 7. Models (optional)

No lab needs the network. Keys go in the wizard or the Chat tab. The router
picks by role and sends the **skeptic to a different provider** than the author —
a model grading its own work agrees with itself.

Pricing: only Anthropic has a real per-token price here (opus-5 $5/$25 per
million). Others are recorded at $0 and **labelled as such**, because an
invented number in a report looks like knowledge.

## 8. The report

**Report** tab → Generate → Save. You get markdown in which every COMPUTED
claim carries its run id, its sha256, and the result of re-hashing it. That is
what you bring home.

## 9. When something breaks

```
./run.sh --selftest              # this first, always
```

- **`no interpreter with numpy found`** → you have not run `./setup.sh`.
- **`datasets present…` fails** → check `datasets/MANIFEST.json`; the data
  changed or is missing. Do not re-download it on hotel wifi.
- **theta out of band after an upgrade** → the app resets theta when the stored
  state was written under different constants; look for `THETA_RESET` in the journal.
- **Guardian reports `TLS pin mismatch`** → both the key and the certificate
  issuer changed. On someone else's network: do not work around this.
- **Display at 200%, content falls off the window** → windows size themselves to
  the available screen and the wizard scrolls with its buttons pinned at the
  bottom. If you still want more on screen than the desktop scaling allows,
  override it **for this app only**:
  ```
  PSE_JLAB_SCALE=1     ./run.sh     # as if 100% (smaller, but everything fits)
  PSE_JLAB_SCALE=1.5   ./run.sh     # halfway
  ```
  This does not touch your system setting.

## 10. Installing on another machine

```
sudo apt install ./pse-jlab_0.1.0_all.deb    # 345 MB, carries data and wheels
/opt/pse-jlab/app/setup.sh
pse-jlab --selftest
```
