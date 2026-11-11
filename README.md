# AX-AGENT
```markdown
# AX-AGENT: Contract-Driven Closed-Loop Framework for Error-Bounded PPA-Aware Approximate RTL Generation

[![Paper](https://img.shields.io/badge/Paper-ASP-DAC%202027-blue)](https://doi.org/10.1145/3840404.3848303)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official implementation of **AX-AGENT**, a contract-driven multi-agent framework for generating error-bounded, PPA-aware approximate RTL. Given a natural-language request (e.g., *"Design an 8-bit unsigned approximate multiplier with MAE <= 16 and minimized PDP"*), AX-AGENT parses it into a persistent contract, plans an approximation strategy, generates Verilog, verifies it via simulation, evaluates PPA via synthesis, and iteratively refines the design through a four-type Critic.

> **Paper:** *AX-AGENT: Contract-Driven Closed-Loop Framework for Error-Bounded PPA-Aware Approximate RTL Generation*, ASP-DAC 2027.

---

## Pipeline

```
Natural-language request
   │
   ▼
[Architect]      → persistent contract (circuit type, bit-width, signedness, MAE/WCE bounds, PPA objectives)
   │
   ▼
[Catalog Router] → filter candidate approximation strategies from catalogs/
   │
   ▼
[Planner]        → select strategy (primitive) or adder/multiplier/skeleton (composite)
   │
   ▼
[Designer]       → generate Verilog (instantiate / repair / micro-tune)
   │
   ▼
[Simulation]     → Icarus Verilog + structured MAE/WCE/XZ metrics
   │
   ▼
[Synthesis]      → Design Compiler + area/delay/power
   │
   ▼
[Critic]         → classify A/B/C/D → repair / micro-tune / reselect / accept
   │
   └──► loop until accept or max iterations
```

Supported families: **adders, multipliers, MAC units, processing elements (PEs)**, primitive and composite (skeleton-based).

---

## Repository Structure

```
.
├── ax_agent_auto.py          # Main orchestration script (batch runner)
├── agent_config.example.json # LLM API config template
├── requirements.txt
├── prompts/                  # Agent system prompts
├── catalogs/                 # Approximate strategy catalog (per family)
├── skeletons/                # Composite circuit skeletons
├── templates/                # RTL templates per strategy
├── testbench/                # Handwritten testbenches (tb_<signedness>_<family>.v)
├── critic/critic_policy.txt  # Critic policy text
├── dc/run_dc.tcl             # Design Compiler synthesis script
├── experiments/
│   └── prompt_suite.jsonl    # Benchmark prompts (one JSON per line)
├── examples/                 # Example run snapshots
└── runs/                     # Per-request artifacts (auto-generated)
```

---

## Requirements

- Python >= 3.9
- [Icarus Verilog](http://iverilog.icarus.com/) (`iverilog`, `vvp`)
- Synopsys Design Compiler (`dc_shell`) + TSMC 28nm library (optional; PPA is skipped if unavailable)
- An OpenAI-compatible LLM endpoint

```bash
pip install -r requirements.txt
```

---

## Configuration

```bash
cp agent_config.example.json agent_config.json
```

Then fill in your API keys:

```json
{
  "parser":       {"api_key": "YOUR_KEY", "base_url": "https://api.example.com/v1", "model": "gpt-5.2", "temperature": 0.1},
  "planner":      {"api_key": "YOUR_KEY", "base_url": "https://api.example.com/v1", "model": "gpt-5.2", "temperature": 0.1},
  "generator":    {"api_key": "YOUR_KEY", "base_url": "https://api.example.com/v1", "model": "gpt-5.2", "temperature": 0.1},
  "critic":       {"api_key": "YOUR_KEY", "base_url": "https://api.example.com/v1", "model": "gpt-5.2", "temperature": 0.1},
  "tb_generator": {"api_key": "YOUR_KEY", "base_url": "https://api.example.com/v1", "model": "gpt-5.2", "temperature": 0.1},
  "system": {
    "max_iterations_cap": 6,
    "exhaustive_threshold": 8,
    "random_tests": 4000,
    "vvp_timeout": 300,
    "dc_timeout": 1800,
    "require_ppa_for_accept": true
  },
  "dc": {"script_path": "dc/run_dc.tcl"}
}
```

`parser`, `planner`, `generator`, `critic` are required. `tb_generator` is only needed for families without handwritten testbenches.

---

## Quick Start

### Single request (interactive)

```bash
python ax_agent_auto.py
```

Input example:

```
Design an 8-bit unsigned approximate multiplier with MAE <= 16 and WCE <= 64. Minimize PDP.
```

Press Enter twice to submit. Artifacts are written to `runs/run_<id>_<timestamp>/`.

### Batch benchmark (paper reproduction)

```bash
python ax_agent_auto.py --prompt-suite experiments/prompt_suite.jsonl
```

Filter by family:

```bash
python ax_agent_auto.py --families adder multiplier
```

Results:

- `experiments/results_<family>.jsonl` — one record per task
- `experiments/last_batch_manifest.json` — batch summary

---

## Output Artifacts

Each run directory contains:

| Path | Content |
|---|---|
| `user_request.txt` | Raw natural-language input |
| `persistent_contract.json` | Parsed contract (hard/soft constraints) |
| `catalog_router/` | Candidate filtering result |
| `iteration_i/planner/` | Strategy selection |
| `iteration_i/template_loader/` | Loaded templates / skeletons |
| `iteration_i/generator/` | Generated RTL (`.v`) + design notes |
| `iteration_i/verification/` | Icarus logs + structured MAE/WCE metrics |
| `iteration_i/dc/` | DC logs + area/delay/power |
| `iteration_i/compliance_report.json` | Hard/soft constraint evaluation |
| `iteration_i/critic/` | Evidence bundle + rule-based + LLM critic output |
| `run_status.json` | Iteration history |
| `final_result.json` | Final generator / critic / metrics |
| `experiment_result.json` | Compact record used for batch tables |

---

## Evaluation Protocol

- **Validity** — final task success rate and normalized contract utilization `ρ = max(MAE/τ_MAE, WCE/τ_WCE)`.
- **Quality** — post-synthesis area / power / PDP under the same TSMC 28nm flow.
- **Robustness** — cross-backbone consistency across CodeGen-16B, Claude-4.5, ChatGPT-5.2.

Benchmark contracts (MAE/WCE bounds per task) are in `experiments/prompt_suite.jsonl` and Table 2 of the paper.

---

## Known Limitations

- **Sampled WCE**: for larger designs, WCE is the maximum observed error, not a formally certified bound. A SAT/SMT or partitioned-exhaustive checker is required for certification.
- **PPA availability**: if `dc_shell` is not installed, PPA is skipped and `effective_ppa_required` is set to false; designs may still be accepted on hard constraints alone.
- **LLM cost**: ~5.5 LLM calls and ~43.3k tokens per task vs. 1.9 calls / 2.9k tokens for the direct baseline (~11× cost per valid design).
- **Statistical coverage**: 22 tasks and three backbones; prompt-paraphrase and independent-seed studies are future work.

---

## Citation

```bibtex
@inproceedings{dai2027axagent,
  title     = {AX-AGENT: Contract-Driven Closed-Loop Framework for Error-Bounded PPA-Aware Approximate RTL Generation},
  author    = {Dai, Chenyang and Zhou, Tongxin and Xiang, Meng and Liu, Yijin and Waris, Haroon and Li, Li and Fu, Yuxiang},
  booktitle = {32nd Asia and South Pacific Design Automation Conference (ASP-DAC '27)},
  year      = {2027},
  doi       = {10.1145/3840404.3848303}
}
```

---

## License

MIT License (see `LICENSE`). Parts of `catalogs/`, `templates/`, and `skeletons/` are adapted from prior work (EvoApproxLib, SALSA, BLASYS, etc.); please cite the corresponding papers when reusing them.

---

## Contact

- Chenyang Dai — 602024730002@smail.nju.edu.cn
- Yuxiang Fu (corresponding) — yuxiangfu@nju.edu.cn
```
