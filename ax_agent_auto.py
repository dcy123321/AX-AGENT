from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import combinations
import requests


# =========================================================
# Paths and defaults
# =========================================================
ROOT = Path.cwd()
RUNS_DIR = ROOT / "runs"
PROMPTS_DIR = ROOT / "prompts"
TESTBENCH_DIR = ROOT / "testbench"
CATALOGS_DIR = ROOT / "catalogs"
TEMPLATES_DIR = ROOT / "templates"
CRITIC_DIR = ROOT / "critic"
AGENT_CONFIG_PATH = ROOT / "agent_config.json"
EXPERIMENTS_DIR = ROOT / "experiments"
PROMPT_SUITE_PATH = EXPERIMENTS_DIR / "prompt_suite.jsonl"
DEFAULT_DC_SCRIPT_CANDIDATES = ROOT / "dc" / "run_dc.tcl"

SKELETONS_DIR = ROOT / "skeletons"
COMPOSITE_FAMILIES = {"mac", "pe", "systolic_array"}

DEFAULT_SYSTEM_MAX_ITER = 6
DEFAULT_EXHAUSTIVE_BITWIDTH_THRESHOLD = 8
DEFAULT_RANDOM_TESTS = 4000
DEFAULT_VVP_TIMEOUT = 300
DEFAULT_DC_TIMEOUT = 1800
DEFAULT_HUGE_ERROR_RATIO = 1.5
DEFAULT_SUPPORTED_FAMILIES = ["adder", "multiplier", "mac", "pe", "systolic_array"]


# =========================================================
# Utility
# =========================================================
def print_text_preview(path: Path, title: str, max_lines: int = 80) -> None:
    print_sub(title)
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if len(lines) <= max_lines:
        print(text)
    else:
        print("\n".join(lines[:max_lines]))
        print(f"\n... [truncated: showing first {max_lines} / {len(lines)} lines]")

def canonical_module_name_for_contract(contract: Dict[str, Any]) -> Optional[str]:
    circuit_type = str(contract.get("circuit_type", "")).strip().lower()
    signedness = str(contract.get("signedness", "unsigned")).strip().lower()
    W = coerce_int(contract.get("bit_width"), 8)

    suffix = "u" if signedness == "unsigned" else "s"

    if circuit_type == "adder":
        return f"ax_adder_w{W}_{suffix}"
    if circuit_type == "multiplier":
        return f"ax_multiplier_w{W}_{suffix}"
    return None


def canonical_output_width_for_contract(contract: Dict[str, Any]) -> Optional[int]:
    requested_ow = coerce_int(contract.get("output_width"), 0)
    return requested_ow if requested_ow > 0 else None


def build_exact_top_interface_spec(contract: Dict[str, Any]) -> str:
    circuit_type = str(contract.get("circuit_type", "")).strip().lower()
    W = coerce_int(contract.get("bit_width"), 8)
    module_name = canonical_module_name_for_contract(contract)
    OW = canonical_output_width_for_contract(contract)

    if circuit_type in {"adder", "multiplier"} and module_name and OW is not None:
        return (
            f"module {module_name}(A, B, O);\n"
            f"  input  [{W-1}:0] A;\n"
            f"  input  [{W-1}:0] B;\n"
            f"  output [{OW-1}:0] O;"
        )

    return ""


def build_generator_family_overlay(contract: Dict[str, Any]) -> str:
    circuit_type = str(contract.get("circuit_type", "")).strip().lower()
    module_name = canonical_module_name_for_contract(contract)
    W = coerce_int(contract.get("bit_width"), 8)
    requested_ow = coerce_int(contract.get("output_width"), 0)

    if circuit_type in {"adder", "multiplier"} and module_name:
        ow_rule = (
            f'- The top-level output port "O" must have width exactly {requested_ow} bits.'
            if requested_ow > 0 else
            '- The top-level output port "O" width does NOT need to be fixed in advance. '
            'You may keep the template-native output width or widen it if needed, '
            'but you must declare it explicitly and keep the RTL self-consistent.'
        )

        return textwrap.dedent(
            f"""
            Family-specific mandatory requirement:

            You are designing an approximate {circuit_type}.

            The final returned RTL MUST contain a top-level module with:
            - module name exactly "{module_name}"
            - top-level ports exactly: A, B, O
            - input A width exactly {W} bits
            - input B width exactly {W} bits
            - output port name exactly O

            {ow_rule}

            Important:
            - If the source template uses different module names or port names, you MUST adapt them.
            - If the source template uses a different output width, you may keep that width unless an explicit output_width is required by the contract.
            - The generated RTL must make the top-level interface declarations explicit and unambiguous.
            - Prefer classic non-ANSI Verilog style for the top-level module if convenient.
            """
        ).strip()

    if circuit_type in {"mac", "pe", "systolic_array"}:
        return textwrap.dedent(
            f"""
            Family-specific mandatory requirement:

            You are designing a composite approximate {circuit_type}.

            You must preserve the intended skeleton-oriented structure.
            Do not replace the whole design with an unrelated architecture.
            If interface specification or skeleton interface text is provided in the bundle, you must follow it exactly.
            Prefer stable, testbench-friendly top-level wiring and explicit port naming.
            Protect accumulation-sensitive paths conservatively when the contract is accuracy-sensitive.
            """
        ).strip()

    return ""

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: Path, default: Optional[str] = None) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    if default is None:
        raise FileNotFoundError(f"File not found: {path}")
    return default


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name))
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "module"


def print_banner(title: str) -> None:
    line = "=" * 96
    print(f"\n{line}\n{title}\n{line}")


def print_sub(title: str) -> None:
    line = "-" * 96
    print(f"\n{line}\n{title}\n{line}")


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def lower_no_space(s: Any) -> str:
    return re.sub(r"\s+", "", str(s).strip().lower())


def parse_listish(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, (int, float)):
        return [str(value)]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    for sep in ["|", ",", ";"]:
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip()]
    return [text]


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "pass", "passed"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed"}:
        return False
    return default


def coerce_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def ceil_log2(n: int) -> int:
    if n <= 1:
        return 0
    return int(math.ceil(math.log2(n)))


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def sanitize_agent_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if lower_no_space(k) in {"api_key", "apikey", "token", "secret"}:
                out[k] = "***REDACTED***"
            else:
                out[k] = sanitize_agent_config(v)
        return out
    if isinstance(obj, list):
        return [sanitize_agent_config(x) for x in obj]
    return obj


def normalize_name_token(s: str) -> str:
    s = lower_no_space(s)
    replacements = {
        "mae%": "mae_percent",
        "wce%": "wce_percent",
        "ep%": "ep_percent",
        "mre%": "mre_percent",
        "wcre%": "wcre_percent",
        "mse%": "mse_percent",
        "power": "power_mw",
        "pwr": "power_mw",
        "area": "area_um2",
        "delay": "delay_ns",
        "latency": "delay_ns",
    }
    if s in replacements:
        return replacements[s]
    s = s.replace("%", "_percent")
    s = s.replace("pdk45_", "")
    s = s.replace("pdk45", "")
    s = s.replace("um^2", "um2")
    s = s.replace("μm2", "um2")
    return s


def extract_first_json_block(text: str) -> str:
    if not text:
        raise ValueError("Empty text while extracting JSON")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    raise ValueError("JSON object appears incomplete")


def parse_json_from_text(text: str) -> Any:
    return json.loads(extract_first_json_block(text))


def gather_text_file(path: Path, missing: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else missing


def resolve_path_maybe_relative(path_str: str, base_dir: Optional[Path] = None) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p if p.exists() else None
    if base_dir:
        pp = base_dir / p
        if pp.exists():
            return pp
    pp = ROOT / p
    if pp.exists():
        return pp
    return None


def run_command(
    cmd: List[str],
    cwd: Optional[Path],
    stdout_path: Path,
    stderr_path: Path,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    ensure_dir(stdout_path.parent)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        write_text(stdout_path, proc.stdout or "")
        write_text(stderr_path, proc.stderr or "")
        return {
            "cmd": cmd,
            "cmd_str": " ".join(shlex.quote(c) for c in cmd),
            "cwd": str(cwd) if cwd else None,
            "returncode": proc.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
    except subprocess.TimeoutExpired as e:
        write_text(stdout_path, e.stdout or "")
        write_text(stderr_path, (e.stderr or "") + f"\n[TIMEOUT] after {timeout} seconds")
        return {
            "cmd": cmd,
            "cmd_str": " ".join(shlex.quote(c) for c in cmd),
            "cwd": str(cwd) if cwd else None,
            "returncode": -999,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "timeout": timeout,
        }
    except FileNotFoundError as e:
        write_text(stdout_path, "")
        write_text(stderr_path, f"[FILE_NOT_FOUND] {e}")
        return {
            "cmd": cmd,
            "cmd_str": " ".join(shlex.quote(c) for c in cmd),
            "cwd": str(cwd) if cwd else None,
            "returncode": -998,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error": str(e),
        }


# =========================================================
# Config
# =========================================================
@dataclass
class AgentSpec:
    name: str
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.1


@dataclass
class SystemConfig:
    max_iterations_cap: int = DEFAULT_SYSTEM_MAX_ITER
    exhaustive_threshold: int = DEFAULT_EXHAUSTIVE_BITWIDTH_THRESHOLD
    random_tests: int = DEFAULT_RANDOM_TESTS
    vvp_timeout: int = DEFAULT_VVP_TIMEOUT
    dc_timeout: int = DEFAULT_DC_TIMEOUT
    require_ppa_for_accept: bool = True
    huge_error_ratio: float = DEFAULT_HUGE_ERROR_RATIO
    supported_families: List[str] = field(default_factory=lambda: DEFAULT_SUPPORTED_FAMILIES[:])
    known_family_tb_mandatory: bool = True
    critic_use_llm: bool = True
    critic_rule_first: bool = True
    max_micro_tune_rounds_per_strategy: int = 2

    composite_exact_bootstrap_first_iter: bool = True
    composite_replan_after_repeated_b1: int = 2
    role_aware_mac_diagnosis: bool = True


class AppConfig:
    def __init__(self, path: Path):
        raw = load_json(path)
        self.raw = raw
        self.agents: Dict[str, AgentSpec] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and {"api_key", "base_url", "model"}.issubset(set(v.keys())):
                self.agents[k] = AgentSpec(
                    name=k,
                    api_key=v["api_key"],
                    base_url=str(v["base_url"]).rstrip("/"),
                    model=v["model"],
                    temperature=float(v.get("temperature", 0.1)),
                )

        system = raw.get("system", {}) if isinstance(raw.get("system"), dict) else {}
        self.system = SystemConfig(
            max_iterations_cap=coerce_int(system.get("max_iterations_cap"), DEFAULT_SYSTEM_MAX_ITER),
            exhaustive_threshold=coerce_int(system.get("exhaustive_threshold"), DEFAULT_EXHAUSTIVE_BITWIDTH_THRESHOLD),
            random_tests=coerce_int(system.get("random_tests"), DEFAULT_RANDOM_TESTS),
            vvp_timeout=coerce_int(system.get("vvp_timeout"), DEFAULT_VVP_TIMEOUT),
            dc_timeout=coerce_int(system.get("dc_timeout"), DEFAULT_DC_TIMEOUT),
            require_ppa_for_accept=parse_bool(system.get("require_ppa_for_accept"), True),
            huge_error_ratio=coerce_float(system.get("huge_error_ratio"), DEFAULT_HUGE_ERROR_RATIO),
            supported_families=[str(x).strip().lower() for x in parse_listish(system.get("supported_families") or DEFAULT_SUPPORTED_FAMILIES)],
            known_family_tb_mandatory=parse_bool(system.get("known_family_tb_mandatory"), True),
            critic_use_llm=parse_bool(system.get("critic_use_llm"), True),
            critic_rule_first=parse_bool(system.get("critic_rule_first"), True),
            max_micro_tune_rounds_per_strategy=coerce_int(
                system.get("max_micro_tune_rounds_per_strategy"), 2
            ),

            # NEW
            composite_exact_bootstrap_first_iter=parse_bool(
                system.get("composite_exact_bootstrap_first_iter"), True
            ),
            composite_replan_after_repeated_b1=coerce_int(
                system.get("composite_replan_after_repeated_b1"), 2
            ),
            role_aware_mac_diagnosis=parse_bool(
                system.get("role_aware_mac_diagnosis"), True
            ),
        )

    def has_agent(self, name: str) -> bool:
        return name in self.agents


# =========================================================
# Prompt loading
# =========================================================
def load_prompt(name: str) -> str:
    p = find_first_existing([PROMPTS_DIR / f"{name}.txt"])
    if p is None:
        raise FileNotFoundError(f"Prompt file not found: {PROMPTS_DIR / f'{name}.txt'}")
    return p.read_text(encoding="utf-8")


def load_optional_text(paths: List[Path], default: str = "") -> str:
    p = find_first_existing(paths)
    if p is None:
        return default
    return p.read_text(encoding="utf-8")


# =========================================================
# LLM client
# =========================================================
class LLMClient:
    def __init__(self, config: AppConfig):
        self.config = config

    def chat(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        save_dir: Path,
    ) -> str:
        if agent_name not in self.config.agents:
            raise KeyError(f"Agent '{agent_name}' not found in agent_config.json")

        spec = self.config.agents[agent_name]
        endpoint = spec.base_url + "/chat/completions"

        headers = {
            "Authorization": f"Bearer {spec.api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        }

        payload = {
            "model": spec.model,
            "temperature": spec.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        ensure_dir(save_dir)
        write_text(save_dir / "system_prompt.txt", system_prompt)
        write_text(save_dir / "user_prompt.txt", user_prompt)
        write_json(save_dir / "request.json", payload)

        payload_str = json.dumps(payload, ensure_ascii=False)
        payload_bytes = len(payload_str.encode("utf-8"))

        print_sub(f"LLM Request Debug / {agent_name}")
        print(f"[INFO] endpoint            : {endpoint}")
        print(f"[INFO] model               : {spec.model}")
        print(f"[INFO] system_prompt chars : {len(system_prompt)}")
        print(f"[INFO] user_prompt chars   : {len(user_prompt)}")
        print(f"[INFO] request bytes       : {payload_bytes}")

        try:
            with requests.Session() as s:
                resp = s.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(30, 300),
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            write_text(save_dir / "request_size.txt", f"{payload_bytes}\n")
            write_text(save_dir / "request_preview.txt", payload_str[:20000])
            raise

        write_json(save_dir / "response.json", data)
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response format for agent '{agent_name}': {e}\nRaw: {data}")

        write_text(save_dir / "response_content.txt", content)
        return content

    def chat_json_strict(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        save_dir: Path,
    ) -> Dict[str, Any]:
        content = self.chat(agent_name, system_prompt, user_prompt, save_dir)
        try:
            obj = parse_json_from_text(content)
        except Exception as e:
            raise RuntimeError(
                f"Malformed JSON from agent '{agent_name}'. Aborting to avoid wasting compute. Error: {e}"
            )
        if not isinstance(obj, dict):
            raise RuntimeError(
                f"Agent '{agent_name}' must return a JSON object. Got type={type(obj).__name__}."
            )
        return obj


# =========================================================
# Contract normalization
# =========================================================
def normalize_error_constraints(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    allowed_metrics = {
        "mae", "wce", "mse", "ep", "ep_percent",
        "mre", "mre_percent", "wce_percent",
        "mae_percent", "wcre_percent"
    }

    def _canon_metric(metric_name: Any) -> Optional[str]:
        if metric_name is None:
            return None
        text = str(metric_name).strip()
        if not text:
            return None

        text = re.sub(r"\s+", "_", text.strip())
        text = text.replace("%", "_percent")
        text = re.sub(r"_+", "_", text)
        m = normalize_name_token(text)

        if m in allowed_metrics:
            return m
        return None

    def _append_constraint(metric_name: Any, operator: Any, value: Any) -> None:
        metric = _canon_metric(metric_name)
        if metric is None:
            return

        if value is None or value == "":
            return

        try:
            value_f = float(value)
        except Exception:
            return

        op = str(operator or "<=").strip()
        if op not in {"<=", "<", ">=", ">", "=", "=="}:
            op = "<="

        item = {
            "metric": metric,
            "operator": op,
            "value": value_f,
        }

        for existing in out:
            if (
                existing.get("metric") == item["metric"]
                and existing.get("operator") == item["operator"]
                and float(existing.get("value")) == item["value"]
            ):
                return
        out.append(item)

    def _extract_from_text(text: Any) -> None:
        if text is None:
            return
        s = str(text).strip()
        if not s:
            return

        metric_pat = (
            r"(MAE(?:[_ ]percent|%)?"
            r"|WCE(?:[_ ]percent|%)?"
            r"|MSE"
            r"|EP(?:[_ ]percent|%)?"
            r"|MRE(?:[_ ]percent|%)?"
            r"|WCRE(?:[_ ]percent|%)?)"
        )
        full_pat = re.compile(
            rf"(?i)\b{metric_pat}\b\s*(<=|>=|<|>|==|=)\s*([-+]?\d+(?:\.\d+)?)"
        )

        for m in full_pat.finditer(s):
            metric_raw = m.group(1)
            op = m.group(2)
            value = m.group(3)
            _append_constraint(metric_raw, op, value)

    if raw is None:
        return out
    
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric") or item.get("name")
            operator = item.get("operator") or item.get("op") or "<="
            value = item.get("value")
            _append_constraint(metric, operator, value)

            _extract_from_text(item.get("notes"))
            _extract_from_text(item.get("text"))
        return out

    if isinstance(raw, dict):

        for k, v in raw.items():
            metric = _canon_metric(k)
            if metric is None:
                continue

            if isinstance(v, dict):
                if "max" in v:
                    _append_constraint(metric, "<=", v["max"])
                elif "min" in v:
                    _append_constraint(metric, ">=", v["min"])
                elif "value" in v:
                    _append_constraint(metric, v.get("operator", "<="), v["value"])
            else:
                _append_constraint(metric, "<=", v)

        primary_metric = raw.get("primary_metric")
        max_error_value = raw.get("max_error_value")
        max_error_percent = raw.get("max_error_percent")

        if primary_metric is not None:
            if max_error_value not in (None, ""):
                _append_constraint(primary_metric, "<=", max_error_value)

            if max_error_percent not in (None, ""):
                primary_metric_norm = _canon_metric(primary_metric)
                if primary_metric_norm is not None:
                    if not primary_metric_norm.endswith("_percent"):
                        primary_metric_norm = f"{primary_metric_norm}_percent"
                    _append_constraint(primary_metric_norm, "<=", max_error_percent)

        secondary_metric = raw.get("secondary_metric")
        secondary_value = raw.get("secondary_value")
        if secondary_metric is not None and secondary_value not in (None, ""):
            _append_constraint(secondary_metric, raw.get("secondary_operator", "<="), secondary_value)

        secondary_metrics = raw.get("secondary_metrics")
        if isinstance(secondary_metrics, list):
            for item in secondary_metrics:
                if isinstance(item, dict):
                    _append_constraint(
                        item.get("metric") or item.get("name"),
                        item.get("operator", "<="),
                        item.get("value"),
                    )

        _extract_from_text(raw.get("notes"))
        _extract_from_text(raw.get("text"))
        _extract_from_text(raw.get("description"))

        return out

    _extract_from_text(raw)
    return out


def normalize_soft_objectives(raw: Any) -> Dict[str, Any]:
    result = {"priority_order": [], "targets": {}}
    if raw is None:
        return result
    if isinstance(raw, str):
        result["priority_order"] = [normalize_name_token(x) for x in parse_listish(raw)]
        return result
    if isinstance(raw, list):
        order: List[str] = []
        for item in raw:
            if isinstance(item, str):
                order.append(normalize_name_token(item))
            elif isinstance(item, dict):
                metric = item.get("metric") or item.get("name")
                if metric:
                    m = normalize_name_token(metric)
                    order.append(m)
                    if item.get("value") is not None:
                        result["targets"][m] = {
                            "operator": str(item.get("operator", "<=")),
                            "value": item["value"],
                        }
        result["priority_order"] = order
        return result
    if isinstance(raw, dict):
        order = raw.get("priority_order") or raw.get("prefer") or raw.get("optimize_for") or []
        result["priority_order"] = [normalize_name_token(x) for x in parse_listish(order)]
        targets = raw.get("targets", {})
        if isinstance(targets, dict):
            for k, v in targets.items():
                if isinstance(v, dict) and v.get("value") is not None:
                    result["targets"][normalize_name_token(k)] = {
                        "operator": str(v.get("operator", "<=")),
                        "value": v["value"],
                    }
                elif isinstance(v, dict):
                    if "max" in v:
                        result["targets"][normalize_name_token(k)] = {"operator": "<=", "value": v["max"]}
                    elif "min" in v:
                        result["targets"][normalize_name_token(k)] = {"operator": ">=", "value": v["min"]}
        return result
    return result

def _dedupe_error_constraints(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        metric = normalize_name_token(item.get("metric"))
        operator = str(item.get("operator", "<=")).strip()
        value = item.get("value")
        key = (metric, operator, str(value))
        if metric and key not in seen:
            seen.add(key)
            out.append({
                "metric": metric,
                "operator": operator,
                "value": value,
            })
    return out


def _extract_constraints_from_text(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    allowed_metrics = {
        "mae", "wce", "mse", "ep", "ep_percent",
        "mre", "mre_percent", "wce_percent",
        "mae_percent", "wcre_percent"
    }

    pattern = re.compile(
        r"\b(MAE|WCE|MSE|EP|EP_percent|MRE|MRE_percent|WCE_percent|MAE_percent|WCRE_percent)\b"
        r"\s*(<=|>=|<|>|==|=)\s*"
        r"([-+]?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE
    )

    out: List[Dict[str, Any]] = []
    for m in pattern.finditer(text):
        metric = normalize_name_token(m.group(1))
        if metric not in allowed_metrics:
            continue
        out.append({
            "metric": metric,
            "operator": m.group(2),
            "value": m.group(3),
        })
    return _dedupe_error_constraints(out)


def _extract_error_constraints_from_contract(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # 1) already-structured fields
    out.extend(normalize_error_constraints(contract.get("error_constraints")))
    out.extend(normalize_error_constraints(contract.get("hard_error_constraints")))

    # 2) parser-style error_contract
    error_contract = contract.get("error_contract")
    if isinstance(error_contract, dict):
        primary_metric = error_contract.get("primary_metric")
        max_error_value = error_contract.get("max_error_value")
        max_error_percent = error_contract.get("max_error_percent")

        if primary_metric is not None and max_error_value not in (None, ""):
            out.append({
                "metric": normalize_name_token(str(primary_metric)),
                "operator": "<=",
                "value": max_error_value,
            })

        if primary_metric is not None and max_error_percent not in (None, ""):
            out.append({
                "metric": normalize_name_token(f"{primary_metric}_percent"),
                "operator": "<=",
                "value": max_error_percent,
            })

        notes_blob = "\n".join([
            str(error_contract.get("notes", "") or ""),
            "\n".join(parse_listish(contract.get("parser_notes"))),
            str(contract.get("refined_spec", "") or ""),
        ]).strip()
        out.extend(_extract_constraints_from_text(notes_blob))

    # 3) fallback: parse from parser notes / other textual fields
    fallback_blob = "\n".join([
        "\n".join(parse_listish(contract.get("parser_notes"))),
        str(contract.get("architecture_hint", "") or ""),
        str(contract.get("user_request", "") or ""),
    ]).strip()
    out.extend(_extract_constraints_from_text(fallback_blob))

    return _dedupe_error_constraints(out)


def _merge_soft_objectives(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "priority_order": list(dst.get("priority_order", [])),
        "targets": dict(dst.get("targets", {})),
    }
    for item in src.get("priority_order", []):
        if item not in out["priority_order"]:
            out["priority_order"].append(item)
    for k, v in src.get("targets", {}).items():
        out["targets"][k] = v
    return out


def _extract_soft_objectives_from_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    result = {"priority_order": [], "targets": {}}

    candidates = [
        contract.get("soft_objectives"),
        contract.get("ppa_objectives"),
        contract.get("optimize_for"),
        contract.get("ppa_contract"),
    ]

    for raw in candidates:
        parsed = normalize_soft_objectives(raw)
        result = _merge_soft_objectives(result, parsed)

    # parser-style ppa_contract often carries optimize_for / priority_order
    ppa_contract = contract.get("ppa_contract")
    if isinstance(ppa_contract, dict):
        parsed = normalize_soft_objectives(ppa_contract)
        result = _merge_soft_objectives(result, parsed)

    return result


def sanitize_rtl_code(text: Any) -> str:
    code = str(text or "")
    code = code.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip()

    if not code:
        return ""

    # Prefer fenced verilog blocks if present
    fenced_blocks = re.findall(
        r"```(?:verilog|systemverilog|sv)?\s*(.*?)```",
        code,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_blocks:
        code = "\n\n".join(block.strip() for block in fenced_blocks if block.strip()).strip()

    # If the model added prose before/after RTL, clip to the first module ... last endmodule
    module_match = re.search(r"\bmodule\b", code)
    endmodule_matches = list(re.finditer(r"\bendmodule\b", code))
    if module_match and endmodule_matches:
        start = module_match.start()
        end = endmodule_matches[-1].end()
        if end > start:
            code = code[start:end].strip()

    return code + "\n" if code else ""

def normalize_contract(contract: Dict[str, Any], cfg: SystemConfig) -> Dict[str, Any]:
    c = deepcopy(contract)
    c["circuit_type"] = str(c.get("circuit_type") or c.get("family") or c.get("target_circuit") or "").strip().lower()
    c["bit_width"] = coerce_int(c.get("bit_width") or c.get("width"), 8)
    c["signedness"] = str(c.get("signedness", "unsigned")).strip().lower() or "unsigned"
    c["architecture_hint"] = str(c.get("architecture_hint", "")).strip()

    c["error_constraints"] = normalize_error_constraints(
        c.get("error_constraints") or c.get("error_contract") or c.get("hard_error_constraints") or {}
    )
    c["soft_objectives"] = normalize_soft_objectives(
        c.get("soft_objectives") or c.get("ppa_objectives") or c.get("optimize_for") or {}
    )

    iter_budget = c.get("iteration_budget", {}) if isinstance(c.get("iteration_budget"), dict) else {}
    requested_iter = max(1, coerce_int(iter_budget.get("max_iterations"), 3))
    c["iteration_budget"] = {"max_iterations": min(requested_iter, cfg.max_iterations_cap)}

    test_req = c.get("test_requirement", {}) if isinstance(c.get("test_requirement"), dict) else {}
    c["test_requirement"] = {
        "prefer_exhaustive_upto_bits": coerce_int(
            test_req.get("prefer_exhaustive_upto_bits"), cfg.exhaustive_threshold
        ),
        "num_random_tests": coerce_int(test_req.get("num_random_tests"), cfg.random_tests),
        "enable_corner_cases": parse_bool(test_req.get("enable_corner_cases"), True),
        "enable_directed_cases": parse_bool(test_req.get("enable_directed_cases"), True),
    }

    c["pipeline_depth"] = coerce_int(c.get("pipeline_depth"), 1)
    c["array_size"] = coerce_int(c.get("array_size"), 4)
    c["acc_width"] = coerce_int(c.get("acc_width"), max(2 * c["bit_width"] + 4, 16))
    c["latency"] = coerce_int(c.get("latency"), 1)
    c["require_ppa"] = parse_bool(c.get("require_ppa"), cfg.require_ppa_for_accept)

    c["top_k_strategies"] = max(1, coerce_int(c.get("top_k_strategies"), 2))
    c["top_k_adders"] = max(1, coerce_int(c.get("top_k_adders"), c["top_k_strategies"]))
    c["top_k_multipliers"] = max(1, coerce_int(c.get("top_k_multipliers"), c["top_k_strategies"]))

    requested_ow = coerce_int(c.get("output_width"), 0)
    if requested_ow > 0:
        c["output_width"] = requested_ow
    else:
        c.pop("output_width", None)

    c["planner_mode"] = "composite" if is_composite_circuit(c["circuit_type"]) else "primitive"
    c["component_families"] = []
    if c["planner_mode"] == "composite":
        c["component_families"] = ["adder", "multiplier"]
        c["skeleton_type"] = c["circuit_type"]
    else:
        c["skeleton_type"] = ""

    if not c["circuit_type"]:
        raise ValueError("Contract missing circuit_type/family")
    return c


# =========================================================
# Catalog loading and routing
# =========================================================
def load_family_desc(circuit_type: str) -> str:
    family_dir = CATALOGS_DIR / circuit_type
    return load_optional_text(
        [
            family_dir / "family_desc.txt",
            family_dir / "description.txt",
            family_dir / "planning_knowledge.txt",
        ],
        default="",
    )

def is_composite_circuit(circuit_type: str) -> bool:
    return str(circuit_type).strip().lower() in COMPOSITE_FAMILIES


def pick_metric_value(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", "null"):
            return d[k]
    return None

def _extract_banned_composite_plan_keys(banned_values: List[str]) -> set:
    out = set()
    for x in banned_values or []:
        s = str(x).strip()
        if "::A[" in s and "::M[" in s and "::S[" in s:
            out.add(s)
    return out


def _fallback_non_banned_composite_plan(
    base_plan: Dict[str, Any],
    contract: Dict[str, Any],
    adder_candidates: List[Dict[str, Any]],
    multiplier_candidates: List[Dict[str, Any]],
    skeleton_candidates: List[Dict[str, Any]],
    banned_plan_keys: set,
) -> Dict[str, Any]:
    preferred_adder_count = max(1, len(base_plan.get("selected_adder_ids") or []))
    preferred_multiplier_count = max(1, len(base_plan.get("selected_multiplier_ids") or []))

    adder_ids = [x["strategy_id"] for x in adder_candidates]
    multiplier_ids = [x["strategy_id"] for x in multiplier_candidates]
    skeleton_ids = [x["skeleton_id"] for x in skeleton_candidates]

    max_adders = min(coerce_int(contract.get("top_k_adders"), 2), len(adder_ids))
    max_multipliers = min(coerce_int(contract.get("top_k_multipliers"), 2), len(multiplier_ids))

    adder_count_options: List[int] = []
    for n in [preferred_adder_count] + list(range(max_adders, 0, -1)):
        if 1 <= n <= len(adder_ids) and n not in adder_count_options:
            adder_count_options.append(n)

    multiplier_count_options: List[int] = []
    for n in [preferred_multiplier_count] + list(range(max_multipliers, 0, -1)):
        if 1 <= n <= len(multiplier_ids) and n not in multiplier_count_options:
            multiplier_count_options.append(n)

    for a_count in adder_count_options:
        for m_count in multiplier_count_options:
            for a_subset in combinations(adder_ids, a_count):
                for m_subset in combinations(multiplier_ids, m_count):
                    for sk in skeleton_ids:
                        plan_key = make_composite_plan_key(
                            contract["circuit_type"],
                            list(a_subset),
                            list(m_subset),
                            sk,
                        )
                        if plan_key in banned_plan_keys:
                            continue

                        repaired = deepcopy(base_plan)
                        repaired["selected_adder_ids"] = list(a_subset)
                        repaired["selected_multiplier_ids"] = list(m_subset)
                        repaired["selected_skeleton_id"] = sk

                        component_mapping = repaired.get("component_mapping", {})
                        if not isinstance(component_mapping, dict):
                            component_mapping = {}

                        if component_mapping.get("product_multiplier") not in m_subset:
                            component_mapping["product_multiplier"] = list(m_subset)[0]
                        if component_mapping.get("accumulator_adder") not in a_subset:
                            component_mapping["accumulator_adder"] = list(a_subset)[0]

                        repaired["component_mapping"] = component_mapping
                        repaired["plan_key"] = plan_key

                        reason = str(repaired.get("selection_reason", "")).strip()
                        suffix = (
                            "[AUTO-FALLBACK] Planner returned a banned composite plan_key, "
                            "so it was deterministically rewritten to the first non-banned combination."
                        )
                        repaired["selection_reason"] = f"{reason} {suffix}".strip()

                        return repaired

    raise RuntimeError(
        "All composite plan combinations are banned or exhausted. "
        "No non-banned composite plan can be constructed."
    )

def make_planner_candidate_view(entry: Dict[str, Any]) -> Dict[str, Any]:
    ref = entry.get("reference_metrics", {}) or {}
    return {
        "strategy_id": entry.get("strategy_id"),
        "family": entry.get("family"),
        "strategy_name": entry.get("strategy_name"),
        "description": entry.get("description", ""),
        "supported_signedness": entry.get("supported_signedness", []),
        "bitwidth_min": entry.get("bitwidth_min"),
        "bitwidth_max": entry.get("bitwidth_max"),
        "supported_metrics": entry.get("supported_metrics", []),
        "supported_architecture": entry.get("supported_architecture", []),
        "approximation_style": entry.get("approximation_style", ""),
        "application_scenarios": entry.get("application_scenarios", []),
        "reference_metrics": {
            "MAE": pick_metric_value(ref, ["MAE", "mae"]),
            "WCE": pick_metric_value(ref, ["WCE", "wce"]),
            "MSE": pick_metric_value(ref, ["MSE", "mse"]),
            "EP_percent": pick_metric_value(ref, ["EP_percent", "ep_percent"]),
            "area_um2": pick_metric_value(ref, ["PDK45_AREA_um2", "area_um2", "PDK45_AREA"]),
            "delay_ns": pick_metric_value(ref, ["PDK45_DELAY_ns", "delay_ns", "PDK45_DELAY"]),
            "power_mw": pick_metric_value(ref, ["PDK45_PWR_mW", "power_mw", "PDK45_PWR"]),
        },
        "risk_notes": entry.get("risk_notes", []),
    }


def render_planner_candidate_table(entries: List[Dict[str, Any]], max_rows: int = 100) -> str:
    rows = []
    header = [
        "strategy_id", "name", "bw_range", "signedness", "metrics",
        "MAE", "WCE", "MSE", "AREA", "DELAY", "POWER"
    ]
    rows.append(" | ".join(header))
    rows.append(" | ".join(["---"] * len(header)))

    for raw in entries[:max_rows]:
        entry = make_planner_candidate_view(raw)
        ref = entry.get("reference_metrics", {}) or {}
        rows.append(" | ".join([
            str(entry.get("strategy_id", "")),
            str(entry.get("strategy_name", "")),
            f"{entry.get('bitwidth_min', '')}-{entry.get('bitwidth_max', '')}",
            ",".join(entry.get("supported_signedness", [])),
            ",".join(entry.get("supported_metrics", [])),
            str(ref.get("MAE", "")),
            str(ref.get("WCE", "")),
            str(ref.get("MSE", "")),
            str(ref.get("area_um2", "")),
            str(ref.get("delay_ns", "")),
            str(ref.get("power_mw", "")),
        ]))
    if len(entries) > max_rows:
        rows.append(f"... truncated, showing first {max_rows} rows of {len(entries)} candidates")
    return "\n".join(rows)


def normalize_skeleton_entry(raw: Dict[str, Any], target_circuit: str) -> Dict[str, Any]:
    entry = deepcopy(raw)
    skeleton_id = str(entry.get("skeleton_id") or entry.get("id") or entry.get("name") or "").strip()
    if not skeleton_id:
        raise ValueError(f"Skeleton entry missing skeleton_id: {raw}")

    entry["skeleton_id"] = skeleton_id
    entry["target_circuit"] = str(entry.get("target_circuit") or target_circuit).strip()
    entry["skeleton_name"] = str(entry.get("skeleton_name") or entry.get("name") or skeleton_id).strip()
    entry["description"] = str(entry.get("description", "")).strip()
    entry["supported_signedness"] = [x.lower() for x in parse_listish(
        entry.get("supported_signedness") or ["unsigned", "signed"]
    )]
    entry["supported_architecture"] = [str(x).strip() for x in parse_listish(
        entry.get("supported_architecture") or entry.get("architecture")
    )]
    entry["skeleton_dir"] = str(entry.get("skeleton_dir", "")).strip()
    entry["skeleton_path"] = str(entry.get("skeleton_path", "")).strip()
    entry["guide_path"] = str(entry.get("guide_path", "")).strip()
    entry["interface_spec_path"] = str(entry.get("interface_spec_path", "")).strip()
    return entry


def load_skeleton_entries(target_circuit: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    catalog_path = SKELETONS_DIR / target_circuit / "catalog.json"

    if catalog_path.exists():
        raw_all = load_json(catalog_path)
        if not isinstance(raw_all, list):
            raise ValueError(f"Skeleton catalog JSON must be a list: {catalog_path}")
        return [normalize_skeleton_entry(item, target_circuit) for item in raw_all]

    roots = [
        SKELETONS_DIR / target_circuit,
        ROOT / "templates" / target_circuit / "skeletons",
    ]

    for root in roots:
        if not root.exists():
            continue

        for child in sorted(root.iterdir()):
            if child.is_dir():
                meta_path = find_first_existing([child / "metadata.json"])
                meta = load_json(meta_path) if meta_path else {}
                skeleton_id = str(meta.get("skeleton_id") or child.name)
                candidates.append(normalize_skeleton_entry({
                    "skeleton_id": skeleton_id,
                    "target_circuit": target_circuit,
                    "skeleton_name": meta.get("skeleton_name", skeleton_id),
                    "description": meta.get("description", ""),
                    "supported_signedness": meta.get("supported_signedness", ["unsigned", "signed"]),
                    "supported_architecture": meta.get("supported_architecture", []),
                    "skeleton_dir": str(child),
                    "guide_path": str(child / "guide.txt") if (child / "guide.txt").exists() else "",
                    "interface_spec_path": str(child / "interface.txt") if (child / "interface.txt").exists() else "",
                }, target_circuit))
            elif child.is_file() and child.suffix.lower() == ".v":
                candidates.append(normalize_skeleton_entry({
                    "skeleton_id": child.stem,
                    "target_circuit": target_circuit,
                    "skeleton_name": child.stem,
                    "description": "",
                    "supported_signedness": ["unsigned", "signed"],
                    "supported_architecture": [],
                    "skeleton_path": str(child),
                }, target_circuit))

    if not candidates:
        raise FileNotFoundError(
            f"No skeleton candidates found for target_circuit='{target_circuit}'. "
            f"Checked under {SKELETONS_DIR / target_circuit} and templates/{target_circuit}/skeletons"
        )
    return candidates


def make_composite_plan_key(
    target_circuit: str,
    adder_ids: List[str],
    multiplier_ids: List[str],
    skeleton_id: str,
) -> str:
    a = ",".join(adder_ids)
    m = ",".join(multiplier_ids)
    return f"{target_circuit}::A[{a}]::M[{m}]::S[{skeleton_id}]"


def normalize_planner_output(
    raw: Dict[str, Any],
    contract: Dict[str, Any],
    catalog_router_result: Dict[str, Any],
) -> Dict[str, Any]:
    out = deepcopy(raw)
    router_mode = catalog_router_result.get("router_mode")

    if router_mode == "primitive":
        valid_ids = {x["strategy_id"] for x in catalog_router_result["kept_candidates"]}
        selected_ids = parse_listish(out.get("selected_strategy_ids") or out.get("selected_strategy_id"))
        if not selected_ids:
            raise ValueError("Planner output missing selected_strategy_ids / selected_strategy_id")
        selected_ids = [x for x in selected_ids if x in valid_ids]
        if not selected_ids:
            raise ValueError("Planner selected no valid primitive strategy IDs")
        primary_id = str(out.get("primary_strategy_id") or selected_ids[0]).strip()
        if primary_id not in selected_ids:
            primary_id = selected_ids[0]

        out["plan_type"] = "primitive"
        out["selected_strategy_ids"] = selected_ids
        out["primary_strategy_id"] = primary_id
        out["fallback_strategy_ids"] = [x for x in parse_listish(out.get("fallback_strategy_ids")) if x in valid_ids and x != primary_id]
        out["selection_reason"] = str(out.get("selection_reason", "")).strip()
        out["expected_risks"] = parse_listish(out.get("expected_risks"))
        out["initial_tuning_hint"] = out.get("initial_tuning_hint", {}) if isinstance(out.get("initial_tuning_hint"), dict) else {}
        out["plan_key"] = primary_id if len(selected_ids) == 1 else "__OR__".join(selected_ids)
        return out

    if router_mode == "composite":
        valid_adder_ids = {x["strategy_id"] for x in catalog_router_result["adder_candidates"]}
        valid_multiplier_ids = {x["strategy_id"] for x in catalog_router_result["multiplier_candidates"]}
        valid_skeleton_ids = {x["skeleton_id"] for x in catalog_router_result["skeleton_candidates"]}

        adder_ids = [x for x in parse_listish(out.get("selected_adder_ids")) if x in valid_adder_ids]
        multiplier_ids = [x for x in parse_listish(out.get("selected_multiplier_ids")) if x in valid_multiplier_ids]
        skeleton_id = str(out.get("selected_skeleton_id", "")).strip()

        if not adder_ids:
            raise ValueError("Planner output missing valid selected_adder_ids")
        if not multiplier_ids:
            raise ValueError("Planner output missing valid selected_multiplier_ids")
        if skeleton_id not in valid_skeleton_ids:
            raise ValueError("Planner output missing valid selected_skeleton_id")

        out["plan_type"] = "composite"
        out["target_circuit"] = contract["circuit_type"]
        out["selected_adder_ids"] = adder_ids
        out["selected_multiplier_ids"] = multiplier_ids
        out["selected_skeleton_id"] = skeleton_id
        out["component_mapping"] = out.get("component_mapping", {}) if isinstance(out.get("component_mapping"), dict) else {}
        out["selection_reason"] = str(out.get("selection_reason", "")).strip()
        out["expected_risks"] = parse_listish(out.get("expected_risks"))
        out["initial_tuning_hint"] = out.get("initial_tuning_hint", {}) if isinstance(out.get("initial_tuning_hint"), dict) else {}
        out["plan_key"] = make_composite_plan_key(contract["circuit_type"], adder_ids, multiplier_ids, skeleton_id)
        return out

    raise ValueError(f"Unsupported router_mode: {router_mode}")


def resolve_skeleton_dir(entry: Dict[str, Any]) -> Path:
    direct = resolve_path_maybe_relative(entry.get("skeleton_dir", ""))
    if direct:
        return direct
    skeleton_path = resolve_path_maybe_relative(entry.get("skeleton_path", ""))
    if skeleton_path:
        return skeleton_path.parent
    raise FileNotFoundError(f"Skeleton directory/path not found for skeleton_id={entry.get('skeleton_id')}")


def load_single_template_payload(entry: Dict[str, Any], run_dir: Path, iteration_idx: int, role: str) -> Dict[str, Any]:
    stage_dir = ensure_dir(
        run_dir / f"iteration_{iteration_idx}" / "template_loader" / f"{role}__{entry.get('strategy_id')}"
    )
    template_dir = resolve_template_dir(entry)

    template_file = template_dir / "rtl_template.v"
    interface_path = template_dir / "interface.txt"
    guide_path = template_dir / "guide.txt"
    metadata_path = template_dir / "metadata.json"
    knobs_path = template_dir / "knobs.json"
    test_policy_path = template_dir / "test_policy.json"

    if not template_file.exists():
        raise FileNotFoundError(f"RTL template file not found: {template_file}")

    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    knobs = load_json(knobs_path) if knobs_path.exists() else entry.get("tunable_knobs", {})
    interface_spec = read_text(interface_path, default="") if interface_path.exists() else ""
    guide_text = read_text(guide_path, default="") if guide_path.exists() else ""
    rtl_template = read_text(template_file)
    test_policy = load_json(test_policy_path) if test_policy_path.exists() else {}

    payload = {
        "strategy_entry": entry,
        "template_dir": str(template_dir),
        "template_file": str(template_file),
        "metadata": metadata,
        "knobs": knobs if isinstance(knobs, dict) else {},
        "interface_spec": interface_spec,
        "guide_text": guide_text,
        "rtl_template": rtl_template,
        "test_policy": test_policy if isinstance(test_policy, dict) else {},
    }

    write_json(
        stage_dir / "template_payload.json",
        {
            "strategy_entry": entry,
            "template_dir": str(template_dir),
            "template_file": str(template_file),
            "metadata": metadata,
            "knobs": payload["knobs"],
            "interface_spec_exists": bool(interface_spec),
            "guide_text_exists": bool(guide_text),
            "test_policy": payload["test_policy"],
        },
    )
    write_text(stage_dir / "rtl_template.v", rtl_template)
    if interface_spec:
        write_text(stage_dir / "interface.txt", interface_spec)
    if guide_text:
        write_text(stage_dir / "guide.txt", guide_text)

    return payload



def load_single_skeleton_payload(
    entry: Dict[str, Any],
    run_dir: Path,
    iteration_idx: int,
) -> Dict[str, Any]:
    stage_dir = ensure_dir(run_dir / f"iteration_{iteration_idx}" / "template_loader" / f"skeleton__{entry.get('skeleton_id')}")
    skeleton_dir = resolve_skeleton_dir(entry)
    skeleton_file = resolve_path_maybe_relative(entry.get("skeleton_path", ""), skeleton_dir) or find_first_existing([
        skeleton_dir / "rtl_template.v",
        skeleton_dir / "skeleton.v",
        skeleton_dir / "template.v",
        skeleton_dir / "rtl.v",
    ])
    if skeleton_file is None:
        raise FileNotFoundError(f"No skeleton RTL file found under {skeleton_dir}")

    metadata_path = find_first_existing([skeleton_dir / "metadata.json"])
    guide_path = resolve_path_maybe_relative(entry.get("guide_path", ""), skeleton_dir) or find_first_existing([
        skeleton_dir / "guide.txt",
        skeleton_dir / "generator_guideline.txt",
    ])
    interface_path = resolve_path_maybe_relative(entry.get("interface_spec_path", ""), skeleton_dir) or find_first_existing([
        skeleton_dir / "interface.txt",
        skeleton_dir / "interface_spec.txt",
    ])
    test_policy_path = find_first_existing([skeleton_dir / "test_policy.json"])

    metadata = load_json(metadata_path) if metadata_path else {}
    guide_text = read_text(guide_path, default="") if guide_path else ""
    interface_spec = read_text(interface_path, default="") if interface_path else ""
    rtl_skeleton = read_text(skeleton_file)
    test_policy = load_json(test_policy_path) if test_policy_path else {}

    payload = {
        "skeleton_entry": entry,
        "skeleton_dir": str(skeleton_dir),
        "skeleton_file": str(skeleton_file),
        "metadata": metadata,
        "guide_text": guide_text,
        "interface_spec": interface_spec,
        "rtl_skeleton": rtl_skeleton,
        "test_policy": test_policy if isinstance(test_policy, dict) else {},
    }

    write_json(stage_dir / "skeleton_payload.json", {
        "skeleton_entry": entry,
        "skeleton_dir": str(skeleton_dir),
        "skeleton_file": str(skeleton_file),
        "metadata": metadata,
        "interface_spec_exists": bool(interface_spec),
        "guide_text_exists": bool(guide_text),
        "test_policy": payload["test_policy"],
    })
    write_text(stage_dir / "rtl_skeleton.v", rtl_skeleton)
    if interface_spec:
        write_text(stage_dir / "interface.txt", interface_spec)
    if guide_text:
        write_text(stage_dir / "guide.txt", guide_text)
    return payload

def normalize_catalog_entry(raw: Dict[str, Any], family_hint: str) -> Dict[str, Any]:
    entry = deepcopy(raw)
    strategy_id = str(entry.get("strategy_id") or entry.get("id") or entry.get("name") or "").strip()
    if not strategy_id:
        raise ValueError(f"Catalog entry missing strategy_id: {raw}")

    entry["strategy_id"] = strategy_id
    entry["family"] = str(entry.get("family") or family_hint).strip()
    entry["strategy_name"] = str(entry.get("strategy_name") or entry.get("name") or strategy_id).strip()
    entry["description"] = str(entry.get("description", "")).strip()
    entry["supported_signedness"] = [x.lower() for x in parse_listish(
        entry.get("supported_signedness") or ["unsigned", "signed"]
    )]
    entry["supported_metrics"] = [normalize_name_token(x) for x in parse_listish(
        entry.get("supported_metrics") or entry.get("target_metrics")
    )]
    entry["supported_architecture"] = [str(x).strip() for x in parse_listish(
        entry.get("supported_architecture") or entry.get("architecture")
    )]
    entry["bitwidth_min"] = coerce_int(entry.get("bitwidth_min") or entry.get("width_min"), 1)
    entry["bitwidth_max"] = coerce_int(entry.get("bitwidth_max") or entry.get("width_max"), 4096)
    entry["template_dir"] = str(entry.get("template_dir", "")).strip()
    entry["template_path"] = str(entry.get("template_path", "")).strip()
    entry["interface_spec_path"] = str(entry.get("interface_spec_path", "")).strip()
    entry["guide_path"] = str(entry.get("guide_path", "")).strip()
    entry["knobs_path"] = str(entry.get("knobs_path", "")).strip()

    knobs = entry.get("tunable_knobs") or entry.get("knobs") or {}
    if isinstance(knobs, str):
        try:
            knobs = json.loads(knobs)
        except Exception:
            knobs = {}
    entry["tunable_knobs"] = knobs if isinstance(knobs, dict) else {}

    ref_metrics = entry.get("reference_metrics", {})
    if isinstance(ref_metrics, str):
        try:
            ref_metrics = json.loads(ref_metrics)
        except Exception:
            ref_metrics = {}
    if not isinstance(ref_metrics, dict):
        ref_metrics = {}
    entry["reference_metrics"] = ref_metrics

    entry["risk_notes"] = entry.get("risk_notes", [])
    if isinstance(entry["risk_notes"], str):
        entry["risk_notes"] = parse_listish(entry["risk_notes"])

    refinement_knobs = entry.get("refinement_knobs", [])
    if isinstance(refinement_knobs, str):
        refinement_knobs = parse_listish(refinement_knobs)
    entry["refinement_knobs"] = refinement_knobs if isinstance(refinement_knobs, list) else []

    return entry


def load_catalog_entries(circuit_type: str) -> List[Dict[str, Any]]:
    family_dir = CATALOGS_DIR / circuit_type
    p = find_first_existing([
        family_dir / "catalog.jsonl",
        family_dir / "catalog.json",
        family_dir / "catalog.csv",
    ])
    if p is None:
        raise FileNotFoundError(f"No catalog file found in {family_dir}")

    entries: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".jsonl":
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(normalize_catalog_entry(json.loads(line), circuit_type))
    elif p.suffix.lower() == ".json":
        raw_all = load_json(p)
        if not isinstance(raw_all, list):
            raise ValueError(f"Catalog JSON must be a list: {p}")
        for item in raw_all:
            entries.append(normalize_catalog_entry(item, circuit_type))
    elif p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(normalize_catalog_entry(dict(row), circuit_type))
    else:
        raise ValueError(f"Unsupported catalog file type: {p}")

    if not entries:
        raise RuntimeError(f"Catalog is empty: {p}")
    return entries


def render_candidate_table(entries: List[Dict[str, Any]], max_rows: int = 100) -> str:
    rows = []
    header = [
        "strategy_id", "name", "bw_range", "signedness", "metrics",
        "ref_MAE", "ref_WCE", "ref_MSE", "ref_AREA", "ref_DELAY", "ref_POWER"
    ]
    rows.append(" | ".join(header))
    rows.append(" | ".join(["---"] * len(header)))
    for entry in entries[:max_rows]:
        ref = entry.get("reference_metrics", {}) or {}
        rows.append(" | ".join([
            entry.get("strategy_id", ""),
            entry.get("strategy_name", ""),
            f"{entry.get('bitwidth_min', '')}-{entry.get('bitwidth_max', '')}",
            ",".join(entry.get("supported_signedness", [])),
            ",".join(entry.get("supported_metrics", [])),
            str(ref.get("MAE", ref.get("mae", ""))),
            str(ref.get("WCE", ref.get("wce", ""))),
            str(ref.get("MSE", ref.get("mse", ""))),
            str(ref.get("PDK45_AREA_um2", ref.get("area_um2", ref.get("PDK45_AREA", "")))),
            str(ref.get("PDK45_DELAY_ns", ref.get("delay_ns", ref.get("PDK45_DELAY", "")))),
            str(ref.get("PDK45_PWR_mW", ref.get("power_mw", ref.get("PDK45_PWR", "")))),
        ]))
    if len(entries) > max_rows:
        rows.append(f"... truncated, showing first {max_rows} rows of {len(entries)} candidates")
    return "\n".join(rows)

def entry_matches_signedness(entry: Dict[str, Any], contract: Dict[str, Any]) -> bool:
    signedness = str(contract.get("signedness", "unsigned")).strip().lower()
    supported_signedness = [
        str(x).strip().lower()
        for x in entry.get("supported_signedness", [])
    ]
    return (not supported_signedness) or (signedness in supported_signedness)


def adder_entry_matches_contract(entry: Dict[str, Any], contract: Dict[str, Any]) -> bool:
    return entry_matches_signedness(entry, contract)


def multiplier_entry_matches_contract(entry: Dict[str, Any], contract: Dict[str, Any]) -> bool:
    return entry_matches_signedness(entry, contract)

def run_catalog_router(contract: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    print_banner("STEP 2 / Catalog Router")
    stage_dir = ensure_dir(run_dir / "catalog_router")
    circuit_type = contract["circuit_type"]

    if not is_composite_circuit(circuit_type):
        family_desc = load_family_desc(circuit_type)
        all_entries = load_catalog_entries(circuit_type)

        if circuit_type == "adder":
            kept_entries = [x for x in all_entries if adder_entry_matches_contract(x, contract)]
            filter_note = "Applied signedness-only filtering for adder catalog."
        elif circuit_type == "multiplier":
            kept_entries = [x for x in all_entries if multiplier_entry_matches_contract(x, contract)]
            filter_note = "Applied bit-width + signedness filtering for multiplier catalog."
        else:
            kept_entries = all_entries
            filter_note = "No family-specific catalog filtering applied."

        rejected_entries = [x for x in all_entries if x not in kept_entries]

        if not kept_entries:
            raise RuntimeError(
                f"No catalog candidates left for primitive family='{circuit_type}' "
                f"after applying the requested filtering rules."
            )

        result = {
            "router_mode": "primitive",
            "contract_snapshot": contract,
            "family_desc": family_desc,
            "all_candidates_count": len(all_entries),
            "kept_candidates_count": len(kept_entries),
            "rejected_candidates_count": len(rejected_entries),
            "kept_candidates": kept_entries,
            "rejected_candidates": rejected_entries,
            "planner_candidates": [make_planner_candidate_view(x) for x in kept_entries],
            "candidate_table_for_prompt": render_planner_candidate_table(kept_entries),
        }
        write_text(stage_dir / "family_desc.txt", family_desc)
        write_json(stage_dir / "catalog_router_result.json", result)

        print_sub("Catalog router summary")
        print(pretty_json({
            "router_mode": "primitive",
            "all_candidates_count": len(all_entries),
            "kept_candidates_count": len(kept_entries),
            "rejected_candidates_count": len(rejected_entries),
            "note": filter_note,
        }))
        return result

    # composite: keep current behavior, but filter skeletons by signedness when metadata exists
    adder_desc = load_family_desc("adder")
    multiplier_desc = load_family_desc("multiplier")
    p = CATALOGS_DIR / circuit_type / "family_desc.txt"
    target_desc = p.read_text(encoding="utf-8") if p.exists() else ""

    # composite: adder 只按 signedness 过滤；multiplier 按 bit-width + signedness 过滤；
    # skeleton 继续只按 signedness 过滤
    adder_all = load_catalog_entries("adder")
    multiplier_all = load_catalog_entries("multiplier")
    skeleton_all = load_skeleton_entries(circuit_type)

    adder_entries = [x for x in adder_all if adder_entry_matches_contract(x, contract)]
    multiplier_entries = [x for x in multiplier_all if multiplier_entry_matches_contract(x, contract)]

    signedness = str(contract.get("signedness", "unsigned")).strip().lower()
    skeleton_entries = []
    for x in skeleton_all:
        supported = [str(s).strip().lower() for s in x.get("supported_signedness", [])]
        if not supported or signedness in supported:
            skeleton_entries.append(x)

    adder_rejected = [x for x in adder_all if x not in adder_entries]
    multiplier_rejected = [x for x in multiplier_all if x not in multiplier_entries]
    skeleton_rejected = [x for x in skeleton_all if x not in skeleton_entries]

    if not adder_entries:
        raise RuntimeError(
            "Composite router found 0 adder candidates after signedness filtering"
        )
    if not multiplier_entries:
        raise RuntimeError(
            "Composite router found 0 multiplier candidates after bit-width + signedness filtering"
        )
    if not skeleton_entries:
        raise RuntimeError(
            f"Composite router found 0 skeleton candidates for target_circuit='{circuit_type}' "
            "after signedness filtering"
        )

    result = {
        "router_mode": "composite",
        "contract_snapshot": contract,
        "target_circuit": circuit_type,
        "target_desc": target_desc,
        "adder_family_desc": adder_desc,
        "multiplier_family_desc": multiplier_desc,

        "adder_candidates": adder_entries,
        "multiplier_candidates": multiplier_entries,
        "skeleton_candidates": skeleton_entries,

        "adder_rejected": adder_rejected,
        "multiplier_rejected": multiplier_rejected,
        "skeleton_rejected": skeleton_rejected,

        "adder_table_for_prompt": render_planner_candidate_table(adder_entries),
        "multiplier_table_for_prompt": render_planner_candidate_table(multiplier_entries),
        "skeleton_table_for_prompt": json.dumps([
            {
                "skeleton_id": x.get("skeleton_id"),
                "skeleton_name": x.get("skeleton_name"),
                "description": x.get("description", ""),
                "supported_signedness": x.get("supported_signedness", []),
                "supported_architecture": x.get("supported_architecture", []),
            }
            for x in skeleton_entries
        ], ensure_ascii=False, indent=2),
    }

    write_json(stage_dir / "catalog_router_result.json", result)

    print_sub("Catalog router summary")
    print(pretty_json({
        "router_mode": "composite",
        "adder_all_count": len(adder_all),
        "adder_candidates_count": len(adder_entries),
        "adder_rejected_count": len(adder_rejected),
        "multiplier_all_count": len(multiplier_all),
        "multiplier_candidates_count": len(multiplier_entries),
        "multiplier_rejected_count": len(multiplier_rejected),
        "skeleton_all_count": len(skeleton_all),
        "skeleton_candidates_count": len(skeleton_entries),
        "skeleton_rejected_count": len(skeleton_rejected),
        "note": "Composite routing: adder uses signedness-only filtering; multiplier uses bit-width + signedness filtering; skeleton uses signedness-only filtering.",
    }))
    return result


# =========================================================
# Template loading
# =========================================================
def resolve_template_dir(entry: Dict[str, Any]) -> Path:
    family = entry.get("family", "")
    sid = entry.get("strategy_id", "")

    direct = resolve_path_maybe_relative(entry.get("template_dir", ""))
    if direct:
        return direct

    template_path = resolve_path_maybe_relative(entry.get("template_path", ""))
    if template_path:
        return template_path.parent

    p = TEMPLATES_DIR / family / sid
    if p.exists():
        return p

    raise FileNotFoundError(
        f"Template directory not found for strategy_id={sid}. "
        f"Checked: template_dir, template_path, and default {p}"
    )


def load_template_bundle(
    planner_output: Dict[str, Any],
    catalog_router_result: Dict[str, Any],
    run_dir: Path,
    iteration_idx: int,
) -> Dict[str, Any]:
    print_banner(f"STEP 4 / Template Loader / Iteration {iteration_idx}")
    stage_dir = ensure_dir(run_dir / f"iteration_{iteration_idx}" / "template_loader")

    plan_type = planner_output.get("plan_type")
    if plan_type == "primitive":
        candidate_ids = planner_output.get("selected_strategy_ids", [])
        primary_id = planner_output.get("primary_strategy_id")

        all_entries = {x["strategy_id"]: x for x in catalog_router_result["kept_candidates"]}
        selected_entries = []
        for sid in candidate_ids:
            if sid not in all_entries:
                raise KeyError(f"Primitive selected strategy ID not found in routed candidates: {sid}")
            selected_entries.append(all_entries[sid])

        payloads = [
            load_single_template_payload(entry, run_dir, iteration_idx, role=f"primitive_candidate_{idx}")
            for idx, entry in enumerate(selected_entries)
        ]
        primary_payload = None
        for p in payloads:
            if p["strategy_entry"]["strategy_id"] == primary_id:
                primary_payload = p
                break
        if primary_payload is None:
            primary_payload = payloads[0]

        bundle = {
            "bundle_type": "primitive",
            "plan_key": planner_output.get("plan_key"),
            "strategy_entry": primary_payload["strategy_entry"],
            "primary_entry": primary_payload["strategy_entry"],
            "candidate_entries": [p["strategy_entry"] for p in payloads],
            "candidate_templates": payloads,
            "primary_template": primary_payload,
            "test_policy": primary_payload.get("test_policy", {}),
        }
        write_json(stage_dir / "template_bundle.json", {
            "bundle_type": "primitive",
            "plan_key": bundle["plan_key"],
            "primary_strategy_id": primary_payload["strategy_entry"]["strategy_id"],
            "candidate_strategy_ids": [x["strategy_entry"]["strategy_id"] for x in payloads],
        })
        print(f"[INFO] primary_strategy_id : {primary_payload['strategy_entry']['strategy_id']}")
        print(f"[INFO] candidate_count      : {len(payloads)}")
        return bundle

    if plan_type == "composite":
        adder_id_set = planner_output.get("selected_adder_ids", [])
        multiplier_id_set = planner_output.get("selected_multiplier_ids", [])
        skeleton_id = planner_output.get("selected_skeleton_id")

        adder_all = {x["strategy_id"]: x for x in catalog_router_result["adder_candidates"]}
        multiplier_all = {x["strategy_id"]: x for x in catalog_router_result["multiplier_candidates"]}
        skeleton_all = {x["skeleton_id"]: x for x in catalog_router_result["skeleton_candidates"]}

        adder_payloads = [
            load_single_template_payload(adder_all[sid], run_dir, iteration_idx, role=f"adder_candidate_{idx}")
            for idx, sid in enumerate(adder_id_set)
        ]
        multiplier_payloads = [
            load_single_template_payload(multiplier_all[sid], run_dir, iteration_idx, role=f"multiplier_candidate_{idx}")
            for idx, sid in enumerate(multiplier_id_set)
        ]
        if skeleton_id not in skeleton_all:
            raise KeyError(f"Composite selected skeleton ID not found: {skeleton_id}")
        skeleton_payload = load_single_skeleton_payload(skeleton_all[skeleton_id], run_dir, iteration_idx)

        bundle = {
            "bundle_type": "composite",
            "plan_key": planner_output.get("plan_key"),
            "strategy_entry": {
                "strategy_id": planner_output.get("plan_key"),
                "family": planner_output.get("target_circuit"),
                "strategy_name": planner_output.get("plan_key"),
                "description": planner_output.get("selection_reason", ""),
            },
            "target_circuit": planner_output.get("target_circuit"),
            "component_mapping": planner_output.get("component_mapping", {}),
            "adder_templates": adder_payloads,
            "multiplier_templates": multiplier_payloads,
            "skeleton": skeleton_payload,
            "test_policy": skeleton_payload.get("test_policy", {}),
        }
        write_json(stage_dir / "template_bundle.json", {
            "bundle_type": "composite",
            "plan_key": bundle["plan_key"],
            "target_circuit": bundle["target_circuit"],
            "selected_adder_ids": [x["strategy_entry"]["strategy_id"] for x in adder_payloads],
            "selected_multiplier_ids": [x["strategy_entry"]["strategy_id"] for x in multiplier_payloads],
            "selected_skeleton_id": skeleton_payload["skeleton_entry"]["skeleton_id"],
        })
        print(f"[INFO] plan_key              : {bundle['plan_key']}")
        print(f"[INFO] adder_candidates     : {[x['strategy_entry']['strategy_id'] for x in adder_payloads]}")
        print(f"[INFO] multiplier_candidates: {[x['strategy_entry']['strategy_id'] for x in multiplier_payloads]}")
        print(f"[INFO] skeleton_id          : {skeleton_payload['skeleton_entry']['skeleton_id']}")
        return bundle

    raise ValueError(f"Unsupported plan_type in load_template_bundle: {plan_type}")


# =========================================================
# Agent stages
# =========================================================
def run_parser(llm: LLMClient, user_request: str, run_dir: Path, cfg: SystemConfig) -> Dict[str, Any]:
    print_banner("STEP 1 / Parser Agent")
    stage_dir = ensure_dir(run_dir / "parser")
    system_prompt = load_prompt("parser")
    user_prompt = f"User request:\n{user_request}\n"
    parsed = llm.chat_json_strict("parser", system_prompt, user_prompt, stage_dir)
    if "contract" not in parsed:
        raise ValueError("Parser output missing 'contract'")
    if "refined_spec" not in parsed:
        parsed["refined_spec"] = user_request
    parsed["contract"] = normalize_contract(parsed["contract"], cfg)
    write_json(stage_dir / "parser_output.json", parsed)
    print_sub("Parser structured output")
    print(pretty_json(parsed))
    return parsed

def _try_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def is_exact_like_entry(entry: Dict[str, Any]) -> bool:
    text = " ".join([
        str(entry.get("strategy_name", "")),
        str(entry.get("description", "")),
    ]).lower()

    exact_keywords = [
        "exact",
        "reference",
        "reference-correct",
        "dadda",
        "wallace",
        "ripple-carry",
        "carry-lookahead",
        "carry lookahead",
        "cla",
    ]
    if any(k in text for k in exact_keywords):
        return True

    ref = entry.get("reference_metrics", {}) or {}
    mae = _try_float_or_none(pick_metric_value(ref, ["MAE", "mae"]))
    wce = _try_float_or_none(pick_metric_value(ref, ["WCE", "wce"]))

    if mae is not None and wce is not None and mae == 0.0 and wce == 0.0:
        return True

    return False

def render_planner_candidate_table_compact(entries: List[Dict[str, Any]], max_rows: int = 20) -> str:
    rows = []
    header = [
        "strategy_id", "name", "bw", "sgn", "MAE", "WCE", "MSE", "AREA", "DELAY", "POWER"
    ]
    rows.append(" | ".join(header))
    rows.append(" | ".join(["---"] * len(header)))

    for raw in entries[:max_rows]:
        ref = raw.get("reference_metrics", {}) or {}
        rows.append(" | ".join([
            str(raw.get("strategy_id", "")),
            str(raw.get("strategy_name", "")),
            f"{raw.get('bitwidth_min', '')}-{raw.get('bitwidth_max', '')}",
            ",".join(raw.get("supported_signedness", [])),
            str(pick_metric_value(ref, ["MAE", "mae"]) or ""),
            str(pick_metric_value(ref, ["WCE", "wce"]) or ""),
            str(pick_metric_value(ref, ["MSE", "mse"]) or ""),
            str(pick_metric_value(ref, ["PDK45_AREA_um2", "area_um2", "PDK45_AREA"]) or ""),
            str(pick_metric_value(ref, ["PDK45_DELAY_ns", "delay_ns", "PDK45_DELAY"]) or ""),
            str(pick_metric_value(ref, ["PDK45_PWR_mW", "power_mw", "PDK45_PWR"]) or ""),
        ]))

    if len(entries) > max_rows:
        rows.append(f"... truncated: showing first {max_rows} / {len(entries)}")
    return "\n".join(rows)

def shortlist_candidates_for_planner(
    entries: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    def score(entry: Dict[str, Any]) -> Tuple[int, int, int, str]:
        ref = entry.get("reference_metrics", {}) or {}

        has_ref = 1 if ref else 0
        has_err = 1 if any(
            pick_metric_value(ref, [k]) is not None
            for k in ["MAE", "mae", "WCE", "wce", "MSE", "mse"]
        ) else 0

        text = f"{entry.get('strategy_name', '')} {entry.get('description', '')}".lower()
        exact_like = 1 if any(k in text for k in [
            "exact", "reference", "dadda", "wallace", "ripple", "cla", "loa", "trunc"
        ]) else 0

        return (-has_ref, -has_err, -exact_like, str(entry.get("strategy_id", "")))

    ranked = sorted(entries, key=score)
    return ranked[:limit]

def exact_like_subset(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    subset = [x for x in entries if is_exact_like_entry(x)]
    return subset if subset else entries

def run_planner(
    llm: LLMClient,
    parser_output: Dict[str, Any],
    catalog_router_result: Dict[str, Any],
    run_dir: Path,
    iteration_idx: int,
    cfg: SystemConfig,   # NEW
    banned_strategy_ids: Optional[List[str]] = None,
    critic_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print_banner(f"STEP 3 / Planner Agent / Iteration {iteration_idx}")
    stage_dir = ensure_dir(run_dir / f"iteration_{iteration_idx}" / "planner")
    planner_input_dir = ensure_dir(stage_dir / "planner_input")

    contract = parser_output["contract"]
    refined_spec = parser_output["refined_spec"]
    banned_set = set(banned_strategy_ids or [])
    router_mode = catalog_router_result["router_mode"]

    system_prompt = load_prompt("planner")

    if router_mode == "primitive":
        candidates_all = catalog_router_result["kept_candidates"]
        candidates = [x for x in candidates_all if x.get("strategy_id") not in banned_set]
        if not candidates:
            raise RuntimeError("No primitive candidate strategies left after applying banned_strategy_ids")

        planner_candidates = shortlist_candidates_for_planner(candidates, limit=20)

        payload = {
            "router_mode": "primitive",
            "contract": contract,
            "refined_spec": refined_spec,
            "family_desc": shorten_text_block(catalog_router_result.get("family_desc", ""), max_chars=1600, max_lines=40),
            "banned_strategy_ids": sorted(list(banned_set)),
            "candidate_count_total": len(candidates),
            "candidate_count_sent": len(planner_candidates),
            "critic_feedback": critic_feedback or {},
        }
        write_json(planner_input_dir / "planner_payload.json", payload)

        user_prompt = textwrap.dedent(
            f"""
            Planning mode:
            primitive

            Contract JSON:
            {json.dumps(contract, ensure_ascii=False, indent=2)}

            Refined specification:
            {shorten_text_block(refined_spec, max_chars=800, max_lines=20)}

            Family description:
            {payload['family_desc'] or '[EMPTY]'}

            Banned strategy IDs:
            {json.dumps(sorted(list(banned_set)), ensure_ascii=False)}

            Candidate count:
            total={len(candidates)}, sent={len(planner_candidates)}

            Candidate shortlist:
            {render_planner_candidate_table_compact(planner_candidates, max_rows=20)}

            Critic feedback from previous round (if any):
            {json.dumps(compact_critic_for_generator(critic_feedback), ensure_ascii=False, indent=2)}

            Select 1~{contract.get('top_k_strategies', 2)} compatible primitive templates.

            Required output format:
            {{
            "plan_type": "primitive",
            "selected_strategy_ids": ["...", "..."],
            "primary_strategy_id": "...",
            "fallback_strategy_ids": ["...", "..."],
            "selection_reason": "...",
            "expected_risks": ["..."],
            "initial_tuning_hint": {{ ... }}
            }}
            """
        ).strip()

        raw_plan = llm.chat_json_strict("planner", system_prompt, user_prompt, stage_dir)
        plan = normalize_planner_output(raw_plan, contract, catalog_router_result)

        write_json(stage_dir / "planner_output.json", plan)
        print_sub("Planner structured output")
        print(pretty_json(plan))
        return plan

    if router_mode == "composite":
        adder_all = catalog_router_result["adder_candidates"]
        multiplier_all = catalog_router_result["multiplier_candidates"]
        skeleton_all = catalog_router_result["skeleton_candidates"]

        force_exact_bootstrap = (
            cfg.composite_exact_bootstrap_first_iter
            and iteration_idx == 1
            and not critic_feedback
        )

        adder_seed_pool = exact_like_subset(adder_all) if force_exact_bootstrap else adder_all
        multiplier_seed_pool = exact_like_subset(multiplier_all) if force_exact_bootstrap else multiplier_all

        adder_candidates = [x for x in adder_seed_pool if x.get("strategy_id") not in banned_set]
        multiplier_candidates = [x for x in multiplier_seed_pool if x.get("strategy_id") not in banned_set]
        skeleton_candidates = [x for x in skeleton_all if x.get("skeleton_id") not in banned_set]

        if not adder_candidates:
            raise RuntimeError("No adder candidates left after applying banned_strategy_ids")
        if not multiplier_candidates:
            raise RuntimeError("No multiplier candidates left after applying banned_strategy_ids")
        if not skeleton_candidates:
            raise RuntimeError("No skeleton candidates left after applying banned_strategy_ids")

        payload = {
            "router_mode": "composite",
            "contract": contract,
            "refined_spec": refined_spec,
            "target_desc": catalog_router_result.get("target_desc", ""),
            "adder_family_desc": catalog_router_result.get("adder_family_desc", ""),
            "multiplier_family_desc": catalog_router_result.get("multiplier_family_desc", ""),
            "banned_strategy_ids": sorted(list(banned_set)),
            "adder_entries": [make_planner_candidate_view(x) for x in adder_candidates],
            "multiplier_entries": [make_planner_candidate_view(x) for x in multiplier_candidates],
            "skeleton_entries": [
                {
                    "skeleton_id": x.get("skeleton_id"),
                    "skeleton_name": x.get("skeleton_name"),
                    "description": x.get("description", ""),
                    "supported_signedness": x.get("supported_signedness", []),
                    "supported_architecture": x.get("supported_architecture", []),
                }
                for x in skeleton_candidates
            ],
            "force_exact_bootstrap": force_exact_bootstrap,
            "critic_feedback": critic_feedback or {},
        }
        write_json(planner_input_dir / "planner_payload.json", payload)

        user_prompt = textwrap.dedent(
            f"""
            Planning mode:
            composite

            Contract JSON:
            {json.dumps(contract, ensure_ascii=False, indent=2)}

            Refined specification:
            {refined_spec}

            Target circuit description:
            {catalog_router_result.get('target_desc', '') or '[EMPTY]'}

            Adder family description:
            {catalog_router_result.get('adder_family_desc', '') or '[EMPTY]'}

            Multiplier family description:
            {catalog_router_result.get('multiplier_family_desc', '') or '[EMPTY]'}

            Banned strategy IDs:
            {json.dumps(sorted(list(banned_set)), ensure_ascii=False)}

            Adder candidate table:
            {render_planner_candidate_table(adder_candidates)}

            Multiplier candidate table:
            {render_planner_candidate_table(multiplier_candidates)}

            Skeleton candidate list:
            {json.dumps(payload['skeleton_entries'], ensure_ascii=False, indent=2)}

            Critic feedback from previous round (if any):
            {json.dumps(critic_feedback or {}, ensure_ascii=False, indent=2)}

            Bootstrap exact mode:
            {json.dumps(force_exact_bootstrap)}

            If Bootstrap exact mode is true, you must prefer exact-like adder and multiplier candidates for the first composite baseline.
            Do not spend the first round on approximate-at-both-roles exploration.

            You are planning a composite approximate circuit. Select:
            - 1~{contract.get('top_k_adders', 2)} adders
            - 1~{contract.get('top_k_multipliers', 2)} multipliers
            - exactly 1 skeleton

            Required output format:
            {{
              "plan_type": "composite",
              "selected_adder_ids": ["...", "..."],
              "selected_multiplier_ids": ["...", "..."],
              "selected_skeleton_id": "...",
              "component_mapping": {{
                "product_multiplier": "...",
                "accumulator_adder": "..."
              }},
              "selection_reason": "...",
              "expected_risks": ["..."],
              "initial_tuning_hint": {{ ... }}
            }}
            """
        ).strip()

        raw_plan = llm.chat_json_strict("planner", system_prompt, user_prompt, stage_dir)
        plan = normalize_planner_output(raw_plan, contract, catalog_router_result)

        banned_composite_plan_keys = _extract_banned_composite_plan_keys(sorted(list(banned_set)))
        if plan["plan_key"] in banned_composite_plan_keys:
            plan = _fallback_non_banned_composite_plan(
                base_plan=plan,
                contract=contract,
                adder_candidates=adder_candidates,
                multiplier_candidates=multiplier_candidates,
                skeleton_candidates=skeleton_candidates,
                banned_plan_keys=banned_composite_plan_keys,
            )

        write_json(stage_dir / "planner_output.json", plan)
        print_sub("Planner structured output")
        print(pretty_json(plan))
        return plan

    raise ValueError(f"Unsupported router_mode in run_planner: {router_mode}")


def render_generator_prompt_name(mode: str) -> str:
    if mode == "instantiate":
        return "generator_instantiate"
    if mode == "repair":
        return "generator_repair"
    if mode == "micro_tune":
        return "generator_micro_tune"
    raise ValueError(f"Unsupported generator mode: {mode}")

def _declared_width_from_text(text: Optional[str]) -> int:
    if not text:
        return 1
    m = re.search(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", str(text))
    if not m:
        return 1
    msb = int(m.group(1))
    lsb = int(m.group(2))
    return abs(msb - lsb) + 1


def _make_synth_decl(direction: str, width: int, port_name: str) -> str:
    if width > 1:
        return f"{direction} [{width-1}:0] {port_name};"
    return f"{direction} {port_name};"


def _split_ansi_ports(header: str) -> List[str]:
    """
    Split ANSI-style module port list by top-level commas only.
    This avoids being confused by commas inside attributes/expressions/parentheses.
    """
    items: List[str] = []
    buf: List[str] = []

    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0

    for ch in header:
        if ch == "(":
            paren_depth += 1
            buf.append(ch)
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
            buf.append(ch)
        elif ch == "[":
            bracket_depth += 1
            buf.append(ch)
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
            buf.append(ch)
        elif ch == "{":
            brace_depth += 1
            buf.append(ch)
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
            buf.append(ch)
        elif ch == "," and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            item = "".join(buf).strip()
            if item:
                items.append(item)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        items.append(tail)

    return items


def _find_port_decl_text(rtl_code: str, direction: str, port_name: str) -> Optional[str]:
    """
    Try to infer a declaration-like text for a top-level port from either:
      1) classic non-ANSI Verilog declarations
      2) ANSI-style module header declarations

    Supports grouped declarations like:
        input [31:0] A, B;
        output reg signed [15:0] O;

    Returns a normalized declaration-like string such as:
        input [31:0] A;
        output [7:0] O;
    or None if not found.
    """
    direction = str(direction).strip().lower()
    port_name = str(port_name).strip()

    # ------------------------------------------------------------
    # 1) non-ANSI style body declarations
    #    Examples:
    #      input [31:0] A;
    #      input wire [31:0] A, B;
    #      output reg signed [15:0] O;
    # ------------------------------------------------------------
    decl_pat = re.compile(
        r"\b(input|output|inout)\b\s+([^;]+);",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for m in decl_pat.finditer(rtl_code):
        decl_dir = m.group(1).strip().lower()
        decl_body = m.group(2).strip()

        if decl_dir != direction:
            continue

        # 提取位宽
        width = _declared_width_from_text(decl_body)

        # 去掉常见修饰词，便于抽取端口名列表
        cleaned = decl_body
        cleaned = re.sub(r"\b(?:wire|reg|logic|signed|unsigned)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)   # 去掉位宽
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 现在 cleaned 应该类似 "A, B" / "O"
        names = [x.strip() for x in cleaned.split(",") if x.strip()]
        if port_name in names:
            return _make_synth_decl(direction, width, port_name)

    # ------------------------------------------------------------
    # 2) ANSI-style module header declarations
    #    Example:
    #      module m(
    #        input [31:0] A, B,
    #        output [31:0] O
    #      );
    # ------------------------------------------------------------
    header_m = re.search(
        r"\bmodule\b\s+[A-Za-z_]\w*\s*\((.*?)\)\s*;",
        rtl_code,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not header_m:
        return None

    header = header_m.group(1)
    items = _split_ansi_ports(header)

    current_direction: Optional[str] = None
    current_width: int = 1

    for raw_item in items:
        item = raw_item.strip()
        if not item:
            continue

        dir_m = re.match(r"^(input|output|inout)\b", item, flags=re.IGNORECASE)
        if dir_m:
            current_direction = dir_m.group(1).lower()
            current_width = _declared_width_from_text(item)

            cleaned = item
            cleaned = re.sub(r"^(input|output|inout)\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\b(?:wire|reg|logic|signed|unsigned)\b", " ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            names = [x.strip() for x in cleaned.split(",") if x.strip()]
            if current_direction == direction and port_name in names:
                return _make_synth_decl(direction, current_width, port_name)

        else:
            # 继承前一个 direction/width
            if current_direction != direction:
                continue

            cleaned = re.sub(r"\b(?:wire|reg|logic|signed|unsigned)\b", " ", item, flags=re.IGNORECASE)
            cleaned = re.sub(r"\[[^\]]+\]", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            names = [x.strip() for x in cleaned.split(",") if x.strip()]
            if port_name in names:
                return _make_synth_decl(direction, current_width, port_name)

    return None


def infer_top_interface_from_rtl(rtl_code: str) -> Dict[str, Any]:
    """
    Infer top module name and the declarations/widths of ports A, B, O.
    Raises a clear ValueError when inference fails, instead of crashing with
    NoneType.strip().
    """
    if not rtl_code or not str(rtl_code).strip():
        raise ValueError("RTL code is empty while inferring top interface")

    mm = re.search(r"\bmodule\b\s+([A-Za-z_]\w*)\s*\(", rtl_code)
    if not mm:
        raise ValueError("Cannot find top module declaration in RTL")

    module_name = mm.group(1)
    info: Dict[str, Any] = {"module_name": module_name}

    for direction, port in [("input", "A"), ("input", "B"), ("output", "O")]:
        decl = _find_port_decl_text(rtl_code, direction, port)
        if decl is None:
            raise ValueError(
                f"Cannot infer declaration for top-level port '{port}' "
                f"(direction={direction}) in module '{module_name}'. "
                f"Please ensure the RTL explicitly declares top-level ports A, B, O."
            )
        info[f"{port}_decl"] = decl.strip()
        info[f"{port}_width"] = _declared_width_from_text(decl)

    return info

def validate_generator_output(
    generated: Dict[str, Any],
    template_bundle: Dict[str, Any],
    mode: str,
    critic_feedback: Optional[Dict[str, Any]],
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    out = deepcopy(generated)

    forced_module_name = canonical_module_name_for_contract(contract)
    returned_module_name = safe_filename(out.get("module_name", "generated_module"))

    if forced_module_name:
        if returned_module_name != forced_module_name:
            raise ValueError(
                f"Generator returned module_name={returned_module_name}, "
                f"but required canonical module_name is {forced_module_name}"
            )
        out["module_name"] = forced_module_name
    else:
        out["module_name"] = returned_module_name

    out["generation_mode"] = mode

    rtl_code_raw = out.get("rtl_code")
    rtl_code = sanitize_rtl_code(rtl_code_raw)

    if not rtl_code.strip():
        raise ValueError("Generator output missing non-empty rtl_code after sanitization")
    if "module" not in rtl_code or "endmodule" not in rtl_code:
        raise ValueError("Sanitized rtl_code does not contain a valid module/endmodule block")

    out["rtl_code"] = rtl_code

    interface_info = infer_top_interface_from_rtl(rtl_code)

    if interface_info["module_name"] != out["module_name"]:
        raise ValueError(
            f"Top module name mismatch: RTL declares {interface_info['module_name']}, "
            f"but generator output says {out['module_name']}"
        )

    expected_w = coerce_int(contract.get("bit_width"), 8)
    if interface_info["A_width"] != expected_w:
        raise ValueError(
            f"Top-level input A width mismatch: expected {expected_w}, got {interface_info['A_width']}"
        )
    if interface_info["B_width"] != expected_w:
        raise ValueError(
            f"Top-level input B width mismatch: expected {expected_w}, got {interface_info['B_width']}"
        )

    requested_ow = coerce_int(contract.get("output_width"), 0)
    if requested_ow > 0 and interface_info["O_width"] != requested_ow:
        raise ValueError(
            f"Top-level output O width mismatch: expected {requested_ow}, got {interface_info['O_width']}"
        )

    out["interface_info"] = interface_info    

    bundle_type = template_bundle.get("bundle_type")
    out["plan_type"] = bundle_type

    if bundle_type == "primitive":
        valid_ids = {x["strategy_entry"]["strategy_id"] for x in template_bundle.get("candidate_templates", [])}
        if not out.get("strategy_id"):
            out["strategy_id"] = template_bundle["primary_template"]["strategy_entry"]["strategy_id"]
        sid = str(out.get("strategy_id")).strip()
        if sid not in valid_ids:
            raise ValueError(f"Generator returned strategy_id={sid}, which is not in selected primitive candidates")

    elif bundle_type == "composite":
        valid_adder_ids = {x["strategy_entry"]["strategy_id"] for x in template_bundle.get("adder_templates", [])}
        valid_multiplier_ids = {x["strategy_entry"]["strategy_id"] for x in template_bundle.get("multiplier_templates", [])}
        valid_skeleton_id = template_bundle["skeleton"]["skeleton_entry"]["skeleton_id"]

        used_component_ids = out.get("used_component_ids", {})
        if not isinstance(used_component_ids, dict):
            used_component_ids = {}

        if not used_component_ids.get("adder") and template_bundle.get("adder_templates"):
            used_component_ids["adder"] = template_bundle["adder_templates"][0]["strategy_entry"]["strategy_id"]
        if not used_component_ids.get("multiplier") and template_bundle.get("multiplier_templates"):
            used_component_ids["multiplier"] = template_bundle["multiplier_templates"][0]["strategy_entry"]["strategy_id"]
        if not used_component_ids.get("skeleton"):
            used_component_ids["skeleton"] = valid_skeleton_id

        if used_component_ids["adder"] not in valid_adder_ids:
            raise ValueError(f"Generator used invalid adder component: {used_component_ids['adder']}")
        if used_component_ids["multiplier"] not in valid_multiplier_ids:
            raise ValueError(f"Generator used invalid multiplier component: {used_component_ids['multiplier']}")
        if used_component_ids["skeleton"] != valid_skeleton_id:
            raise ValueError(f"Generator used invalid skeleton component: {used_component_ids['skeleton']}")

        out["used_component_ids"] = used_component_ids
        if not out.get("strategy_id"):
            out["strategy_id"] = (
                f"composite::{used_component_ids['adder']}__"
                f"{used_component_ids['multiplier']}__"
                f"{used_component_ids['skeleton']}"
            )

    else:
        raise ValueError(f"Unknown template_bundle bundle_type: {bundle_type}")

    if mode == "micro_tune":
        if not parse_bool((critic_feedback or {}).get("same_strategy_required"), True):
            raise ValueError("Micro-tune mode requires same_strategy_required=true")
        out["micro_tune_guard"] = {
            "same_strategy_enforced": True,
            "same_interface_required": parse_bool((critic_feedback or {}).get("same_interface_required"), True),
            "allowed_edit_types": parse_listish((critic_feedback or {}).get("allowed_edit_types")),
            "forbidden_actions": parse_listish((critic_feedback or {}).get("forbidden_actions")),
        }

    return out

def minijson(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def shorten_text_block(text: str, max_chars: int = 1200, max_lines: int = 40) -> str:
    if not text:
        return ""
    lines = str(text).strip().splitlines()[:max_lines]
    out = "\n".join(lines).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "\n...[truncated]"
    return out


def strip_verilog_comments(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines: List[str] = []
    for line in text.splitlines():
        line = re.sub(r"//.*$", "", line)
        lines.append(line.rstrip())
    return "\n".join(lines)


def compress_code_for_prompt(text: str) -> str:
    text = strip_verilog_comments(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_code_block(code: str, language: str = "verilog") -> str:
    code = compress_code_for_prompt(code)
    if not code:
        return ""
    return f"```{language}\n{code}\n```"


def compact_contract_for_prompt(contract: Dict[str, Any]) -> Dict[str, Any]:
    circuit_type = str(contract.get("circuit_type", "")).strip().lower()

    out: Dict[str, Any] = {
        "circuit_type": contract.get("circuit_type"),
        "bit_width": contract.get("bit_width"),
        "output_width": contract.get("output_width"),
        "signedness": contract.get("signedness"),
        "architecture_hint": contract.get("architecture_hint"),
        "error_constraints": contract.get("error_constraints"),
        "soft_objectives": contract.get("soft_objectives"),
    }

    if circuit_type in {"mac", "pe", "systolic_array"}:
        out["acc_width"] = contract.get("acc_width")
        out["array_size"] = contract.get("array_size")
        out["pipeline_depth"] = contract.get("pipeline_depth")
        out["latency"] = contract.get("latency")

    return {k: v for k, v in out.items() if v not in (None, "", [], {})}

def compact_planner_for_generator(planner_output: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "plan_type",
        "plan_key",
        "primary_strategy_id",
        "selected_strategy_ids",
        "selected_adder_ids",
        "selected_multiplier_ids",
        "selected_skeleton_id",
        "component_mapping",
        "selection_reason",
        "initial_tuning_hint",
    ]
    out = {
        k: planner_output.get(k)
        for k in keep_keys
        if planner_output.get(k) not in (None, "", [], {})
    }
    if "selection_reason" in out:
        out["selection_reason"] = shorten_text_block(str(out["selection_reason"]), max_chars=500, max_lines=12)
    return out


def compact_critic_for_generator(critic_feedback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(critic_feedback, dict) or not critic_feedback:
        return {}

    out: Dict[str, Any] = {
        "type": critic_feedback.get("type"),
        "subtype": critic_feedback.get("subtype"),
        "action_mode": critic_feedback.get("action_mode"),
        "reason": critic_feedback.get("reason"),
        "root_cause": critic_feedback.get("root_cause"),
        "same_strategy_required": critic_feedback.get("same_strategy_required"),
        "same_interface_required": critic_feedback.get("same_interface_required"),
        "allowed_edit_types": critic_feedback.get("allowed_edit_types", []),
        "forbidden_actions": critic_feedback.get("forbidden_actions", []),
        "micro_tune_targets": critic_feedback.get("micro_tune_targets", {}),
    }

    evidence = critic_feedback.get("evidence", {}) if isinstance(critic_feedback.get("evidence"), dict) else {}
    evidence_summary: Dict[str, Any] = {}

    error_details = evidence.get("error_details")
    if isinstance(error_details, list):
        failed_items = []
        for item in error_details:
            if item.get("pass") is False:
                failed_items.append({
                    "metric": item.get("metric"),
                    "target": item.get("target"),
                    "actual": item.get("actual"),
                })
        if failed_items:
            evidence_summary["failed_error_metrics"] = failed_items[:4]

    soft_eval = evidence.get("soft_eval")
    if isinstance(soft_eval, dict):
        evidence_summary["explicit_soft_target_pass"] = soft_eval.get("explicit_soft_target_pass")

    if evidence.get("selected_strategy_id"):
        evidence_summary["selected_strategy_id"] = evidence.get("selected_strategy_id")

    if evidence_summary:
        out["evidence_summary"] = evidence_summary

    if out.get("reason"):
        out["reason"] = shorten_text_block(str(out["reason"]), max_chars=300, max_lines=8)
    if out.get("root_cause"):
        out["root_cause"] = shorten_text_block(str(out["root_cause"]), max_chars=200, max_lines=6)

    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_ref_metrics(ref: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ref, dict):
        return {}
    keys = [
        "MAE", "mae",
        "WCE", "wce",
        "MSE", "mse",
        "EP_percent", "ep_percent",
        "PDK45_AREA_um2", "area_um2",
        "PDK45_DELAY_ns", "delay_ns",
        "PDK45_PWR_mW", "power_mw",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if ref.get(k) not in (None, "", "null"):
            out[k] = ref[k]
    return out


def render_template_summary_card(
    item: Dict[str, Any],
    code_key: str,
    include_code: bool,
    include_guide: bool = False,
    guide_max_chars: int = 700,
    code_max_chars: int = 12000,
) -> str:
    parts: List[str] = []

    strategy_id = str(item.get("strategy_id", "")).strip()
    strategy_name = str(item.get("strategy_name", "")).strip()
    description = str(item.get("description", "")).strip()
    knobs = item.get("knobs", {})
    interface_spec = str(item.get("interface_spec", "")).strip()
    guide_text = str(item.get("guide_text", "")).strip()
    ref_metrics = compact_ref_metrics(item.get("reference_metrics", {}))

    if strategy_id:
        parts.append(f"id: {strategy_id}")
    if strategy_name:
        parts.append(f"name: {strategy_name}")
    if description:
        parts.append("desc: " + shorten_text_block(description, max_chars=240, max_lines=6))
    if knobs:
        parts.append("knobs: " + minijson(knobs))
    if ref_metrics:
        parts.append("ref_metrics: " + minijson(ref_metrics))
    if interface_spec:
        parts.append("interface:\n" + shorten_text_block(interface_spec, max_chars=500, max_lines=18))
    if include_guide and guide_text:
        parts.append("guide:\n" + shorten_text_block(guide_text, max_chars=guide_max_chars, max_lines=24))

    if include_code:
        code_text = str(item.get(code_key, "") or "")
        if code_text:
            code_text = compress_code_for_prompt(code_text)
            if len(code_text) > code_max_chars:
                code_text = code_text[:code_max_chars].rstrip() + "\n// ...[truncated]"
            parts.append("code:\n" + make_code_block(code_text, "verilog"))

    return "\n".join(parts).strip()


def render_skeleton_summary_card(
    skeleton: Dict[str, Any],
    include_code: bool,
    include_guide: bool = False,
    guide_max_chars: int = 700,
    code_max_chars: int = 14000,
) -> str:
    parts: List[str] = []

    skeleton_id = str(skeleton.get("skeleton_id", "")).strip()
    description = str(skeleton.get("description", "")).strip()
    interface_spec = str(skeleton.get("interface_spec", "")).strip()
    guide_text = str(skeleton.get("guide_text", "")).strip()
    rtl_skeleton = str(skeleton.get("rtl_skeleton", "")).strip()

    if skeleton_id:
        parts.append(f"id: {skeleton_id}")
    if description:
        parts.append("desc: " + shorten_text_block(description, max_chars=240, max_lines=6))
    if interface_spec:
        parts.append("interface:\n" + shorten_text_block(interface_spec, max_chars=700, max_lines=24))
    if include_guide and guide_text:
        parts.append("guide:\n" + shorten_text_block(guide_text, max_chars=guide_max_chars, max_lines=24))
    if include_code and rtl_skeleton:
        code_text = compress_code_for_prompt(rtl_skeleton)
        if len(code_text) > code_max_chars:
            code_text = code_text[:code_max_chars].rstrip() + "\n// ...[truncated]"
        parts.append("code:\n" + make_code_block(code_text, "verilog"))

    return "\n".join(parts).strip()


def pick_template_by_id(items: List[Dict[str, Any]], strategy_id: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if str(item.get("strategy_id", "")).strip() == str(strategy_id).strip():
            return item
    return None


def compact_previous_design_notes(text: Optional[str]) -> str:
    if not text:
        return ""
    return shorten_text_block(text, max_chars=700, max_lines=24)


def build_generator_required_output_schema(
    mode: str,
    forced_module_name: Optional[str],
    bundle_type: str,
    skeleton_id: Optional[str] = None,
) -> str:
    module_name_value = forced_module_name if forced_module_name else "..."
    if bundle_type == "primitive":
        return (
            "{\n"
            f'  "module_name": "{module_name_value}",\n'
            '  "plan_type": "primitive",\n'
            '  "strategy_id": "...",\n'
            f'  "generation_mode": "{mode}",\n'
            '  "applied_knobs": { ... },\n'
            '  "changed_regions": ["...", "..."],\n'
            '  "design_notes": "...",\n'
            '  "rtl_code": "..."\n'
            "}"
        )
    return (
        "{\n"
        f'  "module_name": "{module_name_value}",\n'
        '  "plan_type": "composite",\n'
        '  "strategy_id": "...",\n'
        '  "used_component_ids": {\n'
        '    "adder": "...",\n'
        '    "multiplier": "...",\n'
        f'    "skeleton": "{skeleton_id or "..."}"\n'
        '  },\n'
        f'  "generation_mode": "{mode}",\n'
        '  "applied_knobs": { ... },\n'
        '  "changed_regions": ["...", "..."],\n'
        '  "design_notes": "...",\n'
        '  "rtl_code": "..."\n'
        "}"
    )

def run_generator(
    llm: LLMClient,
    parser_output: Dict[str, Any],
    planner_output: Dict[str, Any],
    template_bundle: Dict[str, Any],
    mode: str,
    iteration_idx: int,
    run_dir: Path,
    prior_rtl: Optional[str] = None,
    prior_design_notes: Optional[str] = None,
    critic_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    print_banner(f"STEP 5 / Generator Agent / Iteration {iteration_idx} / Mode={mode}")
    stage_dir = ensure_dir(run_dir / f"iteration_{iteration_idx}" / "generator")
    gen_input_dir = ensure_dir(stage_dir / "generator_input")

    contract = parser_output["contract"]
    refined_spec = parser_output["refined_spec"]
    bundle_type = template_bundle.get("bundle_type")

    family_overlay = build_generator_family_overlay(contract)
    forced_module_name = canonical_module_name_for_contract(contract)

    compact_contract = compact_contract_for_prompt(contract)
    compact_planner = compact_planner_for_generator(planner_output)
    compact_critic = compact_critic_for_generator(critic_feedback)
    refined_spec_short = shorten_text_block(refined_spec, max_chars=800, max_lines=20)
    prior_notes_short = compact_previous_design_notes(prior_design_notes)

    system_prompt = load_prompt(render_generator_prompt_name(mode))

    if bundle_type == "primitive":
        candidate_templates: List[Dict[str, Any]] = []
        for item in template_bundle.get("candidate_templates", []):
            candidate_templates.append({
                "strategy_id": item["strategy_entry"]["strategy_id"],
                "strategy_name": item["strategy_entry"].get("strategy_name", ""),
                "description": item["strategy_entry"].get("description", ""),
                "knobs": item.get("knobs", {}),
                "interface_spec": item.get("interface_spec", ""),
                "guide_text": item.get("guide_text", ""),
                "rtl_template": item.get("rtl_template", ""),
                "reference_metrics": item["strategy_entry"].get("reference_metrics", {}),
            })

        primary_id = str(
            planner_output.get("primary_strategy_id")
            or template_bundle["primary_template"]["strategy_entry"]["strategy_id"]
        ).strip()

        primary_template = next(
            (x for x in candidate_templates if x["strategy_id"] == primary_id),
            candidate_templates[0],
        )
        fallback_templates = [x for x in candidate_templates if x["strategy_id"] != primary_template["strategy_id"]]

        # instantiate 时保留最多 1 个 fallback 摘要；repair/micro_tune 时不再发 fallback
        fallback_templates = fallback_templates[:1] if mode == "instantiate" else []

        payload = {
            "plan_type": "primitive",
            "mode": mode,
            "contract": compact_contract,
            "refined_spec": refined_spec_short,
            "planner_output": compact_planner,
            "family_specific_overlay": family_overlay,
            "primary_template_id": primary_template["strategy_id"],
            "fallback_template_ids": [x["strategy_id"] for x in fallback_templates],
            "previous_rtl_present": bool(prior_rtl),
            "previous_design_notes_present": bool(prior_design_notes),
            "critic_feedback": compact_critic,
        }
        write_json(gen_input_dir / "generator_payload.json", payload)

        # 持久化到文件里，便于调试，但 prompt 不再直接塞全部 JSON
        write_text(gen_input_dir / f"primary__{primary_template['strategy_id']}.v", primary_template["rtl_template"])
        for item in fallback_templates:
            write_text(gen_input_dir / f"fallback__{item['strategy_id']}.v", item["rtl_template"])
        if prior_rtl:
            write_text(gen_input_dir / "prior_rtl.v", prior_rtl)
        if prior_notes_short:
            write_text(gen_input_dir / "prior_design_notes.txt", prior_notes_short)

        sections: List[str] = []
        if family_overlay:
            sections.append(family_overlay)

        sections.append(f"Generation mode: {mode}")
        sections.append("Plan type: primitive")
        sections.append(f"Contract: {minijson(compact_contract)}")
        sections.append(f"Refined specification: {refined_spec_short}")
        sections.append(f"Planner summary: {minijson(compact_planner)}")

        sections.append(
            "Primary template:\n" +
            render_template_summary_card(
                primary_template,
                code_key="rtl_template",
                include_code=True,
                include_guide=(mode == "instantiate"),
            )
        )

        if fallback_templates:
            fallback_cards = []
            for item in fallback_templates:
                fallback_cards.append(
                    render_template_summary_card(
                        item,
                        code_key="rtl_template",
                        include_code=False,
                        include_guide=False,
                    )
                )
            if fallback_cards:
                sections.append("Fallback candidates (summary only):\n" + "\n\n".join(fallback_cards))

        if mode in {"repair", "micro_tune"} and prior_rtl:
            sections.append("Current RTL to modify:\n" + make_code_block(prior_rtl, "verilog"))

        if mode in {"repair", "micro_tune"} and prior_notes_short:
            sections.append("Current design notes:\n" + prior_notes_short)

        if compact_critic:
            sections.append(f"Critic summary: {minijson(compact_critic)}")

        sections.append(
            "Instructions:\n"
            "- Stay grounded to the provided primary template.\n"
            "- Do not invent a totally unrelated primitive strategy.\n"
            "- Preserve the required top-level interface exactly.\n"
            "- In repair/micro_tune mode, modify the current RTL conservatively and keep the same strategy unless critic summary explicitly permits otherwise."
        )

        sections.append(
            "Required output format:\n" +
            build_generator_required_output_schema(
                mode=mode,
                forced_module_name=forced_module_name,
                bundle_type="primitive",
            )
        )

        user_prompt = "\n\n".join(s for s in sections if s.strip())

        raw_generated = llm.chat_json_strict("generator", system_prompt, user_prompt, stage_dir)
        generated = validate_generator_output(raw_generated, template_bundle, mode, critic_feedback, contract)

    elif bundle_type == "composite":
        adder_templates: List[Dict[str, Any]] = []
        multiplier_templates: List[Dict[str, Any]] = []

        for item in template_bundle.get("adder_templates", []):
            adder_templates.append({
                "strategy_id": item["strategy_entry"]["strategy_id"],
                "strategy_name": item["strategy_entry"].get("strategy_name", ""),
                "description": item["strategy_entry"].get("description", ""),
                "knobs": item.get("knobs", {}),
                "interface_spec": item.get("interface_spec", ""),
                "guide_text": item.get("guide_text", ""),
                "rtl_template": item.get("rtl_template", ""),
                "reference_metrics": item["strategy_entry"].get("reference_metrics", {}),
            })

        for item in template_bundle.get("multiplier_templates", []):
            multiplier_templates.append({
                "strategy_id": item["strategy_entry"]["strategy_id"],
                "strategy_name": item["strategy_entry"].get("strategy_name", ""),
                "description": item["strategy_entry"].get("description", ""),
                "knobs": item.get("knobs", {}),
                "interface_spec": item.get("interface_spec", ""),
                "guide_text": item.get("guide_text", ""),
                "rtl_template": item.get("rtl_template", ""),
                "reference_metrics": item["strategy_entry"].get("reference_metrics", {}),
            })

        skeleton = {
            "skeleton_id": template_bundle["skeleton"]["skeleton_entry"]["skeleton_id"],
            "description": template_bundle["skeleton"]["skeleton_entry"].get("description", ""),
            "interface_spec": template_bundle["skeleton"].get("interface_spec", ""),
            "guide_text": template_bundle["skeleton"].get("guide_text", ""),
            "rtl_skeleton": template_bundle["skeleton"].get("rtl_skeleton", ""),
        }

        component_mapping = template_bundle.get("component_mapping", {}) or {}
        chosen_adder_id = str(
            component_mapping.get("accumulator_adder")
            or (adder_templates[0]["strategy_id"] if adder_templates else "")
        ).strip()
        chosen_multiplier_id = str(
            component_mapping.get("product_multiplier")
            or (multiplier_templates[0]["strategy_id"] if multiplier_templates else "")
        ).strip()

        chosen_adder = pick_template_by_id(adder_templates, chosen_adder_id) or adder_templates[0]
        chosen_multiplier = pick_template_by_id(multiplier_templates, chosen_multiplier_id) or multiplier_templates[0]

        fallback_adders = [x for x in adder_templates if x["strategy_id"] != chosen_adder["strategy_id"]]
        fallback_multipliers = [x for x in multiplier_templates if x["strategy_id"] != chosen_multiplier["strategy_id"]]

        # instantiate 时保留最多 1 个 fallback 摘要；repair/micro_tune 时不再发 fallback
        fallback_adders = fallback_adders[:1] if mode == "instantiate" else []
        fallback_multipliers = fallback_multipliers[:1] if mode == "instantiate" else []

        payload = {
            "plan_type": "composite",
            "mode": mode,
            "target_circuit": template_bundle.get("target_circuit"),
            "contract": compact_contract,
            "refined_spec": refined_spec_short,
            "planner_output": compact_planner,
            "family_specific_overlay": family_overlay,
            "component_mapping": component_mapping,
            "chosen_adder_id": chosen_adder["strategy_id"],
            "chosen_multiplier_id": chosen_multiplier["strategy_id"],
            "chosen_skeleton_id": skeleton["skeleton_id"],
            "fallback_adder_ids": [x["strategy_id"] for x in fallback_adders],
            "fallback_multiplier_ids": [x["strategy_id"] for x in fallback_multipliers],
            "previous_rtl_present": bool(prior_rtl),
            "previous_design_notes_present": bool(prior_design_notes),
            "critic_feedback": compact_critic,
        }
        write_json(gen_input_dir / "generator_payload.json", payload)

        # 保存文件用于调试
        write_text(gen_input_dir / f"chosen_adder__{chosen_adder['strategy_id']}.v", chosen_adder["rtl_template"])
        write_text(gen_input_dir / f"chosen_multiplier__{chosen_multiplier['strategy_id']}.v", chosen_multiplier["rtl_template"])
        write_text(gen_input_dir / f"skeleton__{skeleton['skeleton_id']}.v", skeleton["rtl_skeleton"])
        for item in fallback_adders:
            write_text(gen_input_dir / f"fallback_adder__{item['strategy_id']}.v", item["rtl_template"])
        for item in fallback_multipliers:
            write_text(gen_input_dir / f"fallback_multiplier__{item['strategy_id']}.v", item["rtl_template"])
        if prior_rtl:
            write_text(gen_input_dir / "prior_rtl.v", prior_rtl)
        if prior_notes_short:
            write_text(gen_input_dir / "prior_design_notes.txt", prior_notes_short)

        sections = []
        if family_overlay:
            sections.append(family_overlay)

        sections.append(f"Generation mode: {mode}")
        sections.append("Plan type: composite")
        sections.append(f"Target circuit: {template_bundle.get('target_circuit')}")
        sections.append(f"Contract: {minijson(compact_contract)}")
        sections.append(f"Refined specification: {refined_spec_short}")
        sections.append(f"Planner summary: {minijson(compact_planner)}")
        sections.append(f"Component mapping: {minijson(component_mapping)}")

        sections.append(
            "Chosen accumulator adder:\n" +
            render_template_summary_card(
                chosen_adder,
                code_key="rtl_template",
                include_code=True,
                include_guide=(mode == "instantiate"),
            )
        )

        sections.append(
            "Chosen product multiplier:\n" +
            render_template_summary_card(
                chosen_multiplier,
                code_key="rtl_template",
                include_code=True,
                include_guide=(mode == "instantiate"),
            )
        )

        sections.append(
            "Chosen structural skeleton:\n" +
            render_skeleton_summary_card(
                skeleton,
                include_code=True,
                include_guide=(mode == "instantiate"),
            )
        )

        if fallback_adders:
            cards = []
            for item in fallback_adders:
                cards.append(
                    render_template_summary_card(
                        item,
                        code_key="rtl_template",
                        include_code=False,
                        include_guide=False,
                    )
                )
            if cards:
                sections.append("Fallback adders (summary only):\n" + "\n\n".join(cards))

        if fallback_multipliers:
            cards = []
            for item in fallback_multipliers:
                cards.append(
                    render_template_summary_card(
                        item,
                        code_key="rtl_template",
                        include_code=False,
                        include_guide=False,
                    )
                )
            if cards:
                sections.append("Fallback multipliers (summary only):\n" + "\n\n".join(cards))

        if mode in {"repair", "micro_tune"} and prior_rtl:
            sections.append("Current RTL to modify:\n" + make_code_block(prior_rtl, "verilog"))

        if mode in {"repair", "micro_tune"} and prior_notes_short:
            sections.append("Current design notes:\n" + prior_notes_short)

        if compact_critic:
            sections.append(f"Critic summary: {minijson(compact_critic)}")

        sections.append(
            "Instructions:\n"
            "- Stay grounded to the chosen adder, chosen multiplier, and chosen skeleton.\n"
            "- Do not replace the design with an unrelated architecture.\n"
            "- Preserve the skeleton-oriented top-level structure and forwarding behavior.\n"
            "- In repair/micro_tune mode, keep the same overall composite structure unless critic summary explicitly permits otherwise.\n"
            "- Keep top-level wiring and interface stable and testbench-friendly."
        )

        sections.append(
            "Required output format:\n" +
            build_generator_required_output_schema(
                mode=mode,
                forced_module_name=forced_module_name,
                bundle_type="composite",
                skeleton_id=skeleton["skeleton_id"],
            )
        )

        user_prompt = "\n\n".join(s for s in sections if s.strip())

        raw_generated = llm.chat_json_strict("generator", system_prompt, user_prompt, stage_dir)
        generated = validate_generator_output(raw_generated, template_bundle, mode, critic_feedback, contract)

    else:
        raise ValueError(f"Unsupported bundle_type in run_generator: {bundle_type}")

    module_name = generated["module_name"]

    write_json(stage_dir / "generator_output.json", generated)
    write_text(stage_dir / f"{module_name}.v", generated["rtl_code"])
    write_text(stage_dir / "design_notes.txt", str(generated.get("design_notes", "")))

    print_sub("Generator structured output")
    print(pretty_json({
        "module_name": generated.get("module_name"),
        "strategy_id": generated.get("strategy_id"),
        "plan_type": generated.get("plan_type"),
        "generation_mode": generated.get("generation_mode"),
        "used_component_ids": generated.get("used_component_ids", {}),
        "applied_knobs": generated.get("applied_knobs", {}),
        "changed_regions": generated.get("changed_regions", []),
    }))
    return generated


# =========================================================
# Testbench preparation
# =========================================================
def find_known_testbench_template(contract: Dict[str, Any]) -> Optional[Path]:
    signedness = contract.get("signedness", "unsigned")
    circuit_type = contract.get("circuit_type", "")
    p = TESTBENCH_DIR / f"tb_{signedness}_{circuit_type}.v"
    return p if p.exists() else None


def render_template(template_text: str, replacements: Dict[str, Any]) -> str:
    rendered = template_text
    for k, v in replacements.items():
        val = str(v)
        rendered = rendered.replace(f"{{{{{k}}}}}", val)
        rendered = rendered.replace(f"__{k}__", val)
        rendered = rendered.replace(f"${{{k}}}", val)
    return rendered


def decide_test_policy(contract: Dict[str, Any], template_bundle: Dict[str, Any], cfg: SystemConfig) -> Dict[str, Any]:
    test_req = contract.get("test_requirement", {})
    template_test_policy = template_bundle.get("test_policy", {}) or {}
    bit_width = coerce_int(contract.get("bit_width"), 8)
    circuit_type = str(contract.get("circuit_type", "")).strip().lower()

    exhaustive_upto = coerce_int(
        template_test_policy.get("prefer_exhaustive_upto_bits"),
        coerce_int(test_req.get("prefer_exhaustive_upto_bits"), cfg.exhaustive_threshold),
    )
    num_random = coerce_int(
        template_test_policy.get("num_random_tests"),
        coerce_int(test_req.get("num_random_tests"), cfg.random_tests),
    )
    enable_corner = parse_bool(
        template_test_policy.get("enable_corner_cases"),
        parse_bool(test_req.get("enable_corner_cases"), True),
    )
    enable_directed = parse_bool(
        template_test_policy.get("enable_directed_cases"),
        parse_bool(test_req.get("enable_directed_cases"), True),
    )

    # Composite circuits should not use primitive-style full exhaustive policy.
    # Even if W is small, the extra state dimensions (e.g. ACC_IN) make full enumeration impractical.
    if circuit_type in {"mac", "pe", "systolic_array"}:
        mode = "sampled"
        # For composite circuits, keep some random coverage even if caller forgot to set it.
        if num_random <= 0:
            num_random = max(512, cfg.random_tests)
    else:
        mode = "exhaustive" if bit_width <= exhaustive_upto else "sampled"

    return {
        "mode": mode,
        "prefer_exhaustive_upto_bits": exhaustive_upto,
        "num_random_tests": num_random,
        "enable_corner_cases": enable_corner,
        "enable_directed_cases": enable_directed,
    }


def build_testbench_context(
    contract: Dict[str, Any],
    module_name: str,
    test_policy: Dict[str, Any],
    interface_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    circuit_type = str(contract.get("circuit_type", "")).strip()
    signedness = str(contract.get("signedness", "unsigned")).strip() or "unsigned"
    bit_width = coerce_int(contract.get("bit_width"), 8)

    if interface_info:
        output_width = coerce_int(interface_info.get("O_width"), 0)
    else:
        output_width = 0

    if output_width <= 0:
        requested_ow = coerce_int(contract.get("output_width"), 0)
        if requested_ow > 0:
            output_width = requested_ow
        else:
            output_width = bit_width
    array_size = coerce_int(contract.get("array_size"), 4)
    pipeline_depth = coerce_int(contract.get("pipeline_depth"), 1)

    if circuit_type == "systolic_array":
        acc_width = coerce_int(contract.get("acc_width"), 2 * bit_width + 2 + ceil_log2(array_size))
        latency = coerce_int(contract.get("latency"), (3 * array_size - 2) + pipeline_depth)
    elif circuit_type == "mac":
        acc_width = coerce_int(contract.get("acc_width"), 2 * bit_width + 4)
        latency = coerce_int(contract.get("latency"), max(1, pipeline_depth))
    else:
        acc_width = coerce_int(contract.get("acc_width"), 2 * bit_width + 4)
        latency = coerce_int(contract.get("latency"), 1)

    replacements = {
        "MODULE_NAME": module_name,
        "BIT_WIDTH": bit_width,
        "OUTPUT_WIDTH": output_width,
        "SIGNEDNESS": signedness,
        "CIRCUIT_TYPE": circuit_type,
        "ACC_WIDTH": acc_width,
        "ARRAY_SIZE": array_size,
        "PIPELINE_DEPTH": pipeline_depth,
        "LATENCY": latency,
        "TEST_MODE": test_policy["mode"],
        "NUM_RANDOM_TESTS": test_policy["num_random_tests"],
        "ENABLE_CORNER_CASES": int(test_policy["enable_corner_cases"]),
        "ENABLE_DIRECTED_CASES": int(test_policy["enable_directed_cases"]),
    }

    compile_defines = {
        "TB_BIT_WIDTH": bit_width,
        "TB_OUTPUT_WIDTH": output_width,
        "TB_ACC_WIDTH": acc_width,
        "TB_ARRAY_SIZE": array_size,
        "TB_PIPELINE_DEPTH": pipeline_depth,
        "TB_LATENCY": latency,
        "TB_NUM_RANDOM_TESTS": test_policy["num_random_tests"],
        "TB_ENABLE_CORNER_CASES": int(test_policy["enable_corner_cases"]),
        "TB_ENABLE_DIRECTED_CASES": int(test_policy["enable_directed_cases"]),
    }
    compile_defines["TB_MODE_EXHAUSTIVE" if test_policy["mode"] == "exhaustive" else "TB_MODE_SAMPLED"] = 1

    return {
        "module_name": module_name,
        "circuit_type": circuit_type,
        "signedness": signedness,
        "bit_width": bit_width,
        "output_width": output_width,
        "acc_width": acc_width,
        "array_size": array_size,
        "pipeline_depth": pipeline_depth,
        "latency": latency,
        "test_policy": test_policy,
        "template_replacements": replacements,
        "compile_defines": compile_defines,
    }


def generate_testbench_with_agent(
    llm: LLMClient,
    contract: Dict[str, Any],
    module_name: str,
    rtl_code: str,
    iteration_dir: Path,
    cfg: SystemConfig,
    interface_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    if not llm.config.has_agent("tb_generator"):
        raise FileNotFoundError(
            f"No handwritten TB found for family='{contract.get('circuit_type')}' and tb_generator agent is not configured"
        )
    stage_dir = ensure_dir(iteration_dir / "tb_generator")
    system_prompt = load_prompt("tb_generator")
    test_policy = decide_test_policy(contract, {"test_policy": {}}, cfg)
    tb_context = build_testbench_context(contract, module_name, test_policy, interface_info)
    user_prompt = textwrap.dedent(
        f"""
        Contract JSON:
        {json.dumps(contract, ensure_ascii=False, indent=2)}

        Module name:
        {module_name}

        Testbench context:
        {json.dumps(tb_context, ensure_ascii=False, indent=2)}

        RTL code:
        {rtl_code}

        Required output format:
        {{
          "tb_filename": "...",
          "tb_code": "...",
          "tb_notes": "..."
        }}
        """
    ).strip()
    generated = llm.chat_json_strict("tb_generator", system_prompt, user_prompt, stage_dir)
    tb_filename = safe_filename(generated.get("tb_filename", f"tb_{contract.get('circuit_type', 'generated')}")) + ".v"
    if not generated.get("tb_code"):
        raise ValueError("tb_generator output missing tb_code")
    vdir = ensure_dir(iteration_dir / "verification")
    tb_path = vdir / tb_filename
    write_text(tb_path, generated["tb_code"])
    write_json(stage_dir / "tb_generator_output.json", generated)
    write_json(vdir / "testbench_context.json", tb_context)
    return tb_path, tb_context


def prepare_testbench(
    llm: LLMClient,
    contract: Dict[str, Any],
    module_name: str,
    rtl_code: str,
    template_bundle: Dict[str, Any],
    iteration_dir: Path,
    cfg: SystemConfig,
    interface_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    print_banner(f"STEP 6 / Prepare Testbench / {module_name}")
    vdir = ensure_dir(iteration_dir / "verification")
    known_tb = find_known_testbench_template(contract)
    test_policy = decide_test_policy(contract, template_bundle, cfg)
    tb_context = build_testbench_context(contract, module_name, test_policy, interface_info)

    if known_tb is not None:
        tb_text = known_tb.read_text(encoding="utf-8")
        rendered = render_template(tb_text, tb_context["template_replacements"])
        out_tb = vdir / known_tb.name
        write_text(out_tb, rendered)
        shutil.copy2(known_tb, vdir / f"template_source__{known_tb.name}")
        write_json(vdir / "testbench_context.json", tb_context)
        print(f"[INFO] Using handwritten TB: {known_tb}")
        return out_tb, tb_context

    if contract.get("circuit_type") in cfg.supported_families and cfg.known_family_tb_mandatory:
        raise FileNotFoundError(
            f"Missing handwritten testbench for supported family '{contract.get('circuit_type')}'"
        )

    print("[INFO] No handwritten TB found. Falling back to constrained LLM-based TB generation.")
    return generate_testbench_with_agent(llm, contract, module_name, rtl_code, iteration_dir, cfg)


# =========================================================
# Verification and PPA
# =========================================================
def run_iverilog_flow(
    rtl_file: Path,
    tb_file: Path,
    module_name: str,
    iteration_dir: Path,
    compile_defines: Optional[Dict[str, Any]],
    cfg: SystemConfig,
) -> Dict[str, Any]:
    print_banner(f"STEP 7 / Icarus Verilog Verification / {module_name}")
    vdir = ensure_dir(iteration_dir / "verification")
    sim_bin = vdir / f"sim_{module_name}"

    compile_cmd = ["iverilog", "-g2012"]
    if compile_defines:
        for k, v in compile_defines.items():
            compile_cmd.append(f"-D{k}={v}")
    compile_cmd.extend(["-o", str(sim_bin), str(tb_file), str(rtl_file)])

    print("[INFO] Compile command:")
    print(" ".join(shlex.quote(c) for c in compile_cmd))

    compile_meta = run_command(
        compile_cmd,
        ROOT,
        vdir / "iverilog_compile.stdout.txt",
        vdir / "iverilog_compile.stderr.txt",
        timeout=cfg.vvp_timeout,
    )

    print_text_preview(vdir / "iverilog_compile.stdout.txt", "iverilog compile stdout")
    print_text_preview(vdir / "iverilog_compile.stderr.txt", "iverilog compile stderr")

    sim_meta: Dict[str, Any] = {"skipped": True, "reason": "compile_failed_or_binary_missing"}

    if compile_meta.get("returncode") == 0 and sim_bin.exists():
        print("[INFO] Simulation command:")
        print(f"vvp {sim_bin}")

        sim_meta = run_command(
            ["vvp", str(sim_bin)],
            ROOT,
            vdir / "vvp.stdout.txt",
            vdir / "vvp.stderr.txt",
            timeout=cfg.vvp_timeout,
        )
        sim_meta["skipped"] = False

        print_text_preview(vdir / "vvp.stdout.txt", "vvp stdout")
        print_text_preview(vdir / "vvp.stderr.txt", "vvp stderr")
    else:
        print("[WARN] Skip simulation because compile failed or sim binary missing.")

    result = {
        "compile": compile_meta,
        "simulation": sim_meta,
        "sim_binary": str(sim_bin),
        "rtl_file": str(rtl_file),
        "tb_file": str(tb_file),
        "compile_defines": compile_defines or {},
    }
    write_json(vdir / "verification_result.json", result)
    return result

def try_parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            v = float(value)
            if math.isnan(v):
                return None
            return v
        except Exception:
            return None

    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"null", "none", "n/a", "na", "nan", "inf", "+inf", "-inf"}:
        return None

    try:
        v = float(text)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def resolve_dc_script(app_cfg: AppConfig) -> Optional[Path]:
    dc_cfg = app_cfg.raw.get("dc", {}) if isinstance(app_cfg.raw.get("dc"), dict) else {}
    p = dc_cfg.get("script_path") if isinstance(dc_cfg, dict) else None

    if p:
        pp = Path(p)
        if not pp.is_absolute():
            pp = ROOT / pp
        if pp.exists():
            return pp

    if DEFAULT_DC_SCRIPT_CANDIDATES.exists():
        return DEFAULT_DC_SCRIPT_CANDIDATES

    return None


def run_dc_flow(
    app_cfg: AppConfig,
    rtl_file: Path,
    module_name: str,
    iteration_dir: Path,
    contract: Dict[str, Any],
) -> Dict[str, Any]:
    print_banner(f"STEP 8 / DC Synthesis / {module_name}")
    dc_dir = ensure_dir(iteration_dir / "dc")
    dc_script = resolve_dc_script(app_cfg)

    if dc_script is None:
        result = {
            "status": "skipped",
            "reason": "no_dc_tcl_found",
            "rtl_file": str(rtl_file),
            "module_name": module_name,
        }
        write_json(dc_dir / "dc_result.json", result)
        return result

    shutil.copy2(dc_script, dc_dir / dc_script.name)
    env = {
        "AX_RTL_FILE": str(rtl_file.resolve()),
        "AX_TOP_MODULE": module_name,
        "AX_RUN_DIR": str(iteration_dir.resolve()),
        "AX_DC_REPORT_DIR": str(dc_dir.resolve()),
        "AX_BIT_WIDTH": str(contract.get("bit_width", 8)),
        "AX_SIGNEDNESS": str(contract.get("signedness", "unsigned")),
    }
    dc_meta = run_command(
        ["dc_shell", "-f", str(dc_script.resolve())],
        ROOT,
        dc_dir / "dc_shell.stdout.txt",
        dc_dir / "dc_shell.stderr.txt",
        env=env,
        timeout=app_cfg.system.dc_timeout,
    )
    result = {
        "status": "done" if dc_meta.get("returncode") == 0 else "failed",
        "dc_script": str(dc_script),
        "dc_meta": dc_meta,
        "rtl_file": str(rtl_file),
        "module_name": module_name,
    }
    write_json(dc_dir / "dc_result.json", result)
    return result


def parse_ax_markers_from_text(text: str) -> Dict[str, str]:
    markers: Dict[str, str] = {}
    if not text:
        return markers
    pattern = re.compile(r"^\s*(AX_[A-Z0-9_]+)\s*=\s*(.*?)\s*$", flags=re.MULTILINE)
    for m in pattern.finditer(text):
        markers[m.group(1).strip()] = m.group(2).strip()
    return markers


def parse_sim_metrics(iteration_dir: Path, verification_result: Dict[str, Any]) -> Dict[str, Any]:
    vdir = iteration_dir / "verification"
    texts = [
        gather_text_file(vdir / "iverilog_compile.stdout.txt"),
        gather_text_file(vdir / "iverilog_compile.stderr.txt"),
        gather_text_file(vdir / "vvp.stdout.txt"),
        gather_text_file(vdir / "vvp.stderr.txt"),
    ]
    markers: Dict[str, str] = {}
    for text in texts:
        markers.update(parse_ax_markers_from_text(text))

    error_metrics: Dict[str, Any] = {}
    flags: Dict[str, Any] = {}
    raw_flat: Dict[str, Any] = {}
    for k, v in markers.items():
        kk = k.replace("AX_", "")
        raw_flat[kk] = v
        if kk.startswith("ERROR_"):
            metric_name = normalize_name_token(kk[len("ERROR_"):])
            try:
                error_metrics[metric_name] = float(v)
            except Exception:
                error_metrics[metric_name] = v
        elif kk.startswith("CHECK_"):
            flags[lower_no_space(kk[len("CHECK_"):])] = v
        else:
            flags[lower_no_space(kk)] = v

    compile_pass = verification_result["compile"].get("returncode") == 0
    simulation_pass = verification_result["simulation"].get("returncode") == 0 and not verification_result["simulation"].get("skipped", True)
    sim_status_marker = str(markers.get("AX_SIM_STATUS", "")).upper().strip()
    if sim_status_marker in {"PASS", "PASSED"}:
        simulation_pass = True
    elif sim_status_marker in {"FAIL", "FAILED"}:
        simulation_pass = False

    xz_marker = markers.get("AX_XZ_DETECTED")
    xz_detected = parse_bool(xz_marker, False) if xz_marker is not None else False
    if xz_marker is None:
        xz_detected = bool(re.search(r"\b[xzXZ]{4,}\b", texts[2] or ""))

    structured = {
        "compile_pass": compile_pass,
        "simulation_pass": simulation_pass,
        "xz_detected": xz_detected,
        "markers": raw_flat,
        "error_metrics": error_metrics,
        "flags": flags,
    }
    write_json(vdir / "structured_sim_metrics.json", structured)
    return structured


def convert_power_to_mw(value: float, unit: str) -> float:
    unit = unit.strip().lower()
    if unit == "w":
        return value * 1000.0
    if unit == "mw":
        return value
    if unit in {"uw", "µw"}:
        return value / 1000.0
    if unit == "nw":
        return value / 1_000_000.0
    if unit == "pw":
        return value / 1_000_000_000.0
    return value


def search_first_float(patterns: List[re.Pattern], texts: List[str]) -> Optional[float]:
    for text in texts:
        if not text:
            continue
        for pat in patterns:
            m = pat.search(text)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
    return None


def parse_ppa_metrics(iteration_dir: Path, dc_result: Dict[str, Any]) -> Dict[str, Any]:
    dc_dir = iteration_dir / "dc"
    texts: List[str] = []

    for p in sorted(dc_dir.glob("*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".rpt", ".out", ".log"}:
            texts.append(gather_text_file(p))

    markers: Dict[str, str] = {}
    for text in texts:
        markers.update(parse_ax_markers_from_text(text))

    area_um2 = try_parse_optional_float(markers.get("AX_PPA_AREA_UM2"))
    delay_ns = try_parse_optional_float(markers.get("AX_PPA_DELAY_NS"))
    power_mw = try_parse_optional_float(markers.get("AX_PPA_POWER_MW"))

    if area_um2 is None:
        area_um2 = search_first_float(
            [
                re.compile(r"Total\s+cell\s+area\s*:\s*([0-9.+\-eE]+)", re.IGNORECASE),
                re.compile(r"Cell\s+Area\s*[:=]\s*([0-9.+\-eE]+)", re.IGNORECASE),
            ],
            texts,
        )

    if delay_ns is None:
        delay_ns = search_first_float(
            [
                re.compile(r"critical\s+path\s+(?:delay|length)\s*[:=]\s*([0-9.+\-eE]+)", re.IGNORECASE),
                re.compile(r"data\s+arrival\s+time\s*[:=]?\s*([0-9.+\-eE]+)", re.IGNORECASE),
            ],
            texts,
        )

    if power_mw is None:
        for text in texts:
            if not text:
                continue
            for pat in [
                re.compile(r"Total\s+Power\s*[:=]\s*([0-9.+\-eE]+)\s*([munpµ]?W)", re.IGNORECASE),
                re.compile(r"Total\s+Dynamic\s+Power\s*[:=]\s*([0-9.+\-eE]+)\s*([munpµ]?W)", re.IGNORECASE),
            ]:
                m = pat.search(text)
                if m:
                    try:
                        power_mw = convert_power_to_mw(float(m.group(1)), m.group(2))
                        break
                    except Exception:
                        pass
            if power_mw is not None:
                break

    structured = {
        "dc_status": dc_result.get("status", "unknown"),
        "area_um2": area_um2,
        "delay_ns": delay_ns,
        "power_mw": power_mw,
        "markers": {k.replace("AX_", ""): v for k, v in markers.items()},
    }

    if area_um2 is not None and delay_ns is not None:
        structured["adp_area_delay"] = area_um2 * delay_ns
    if power_mw is not None and delay_ns is not None:
        structured["pdp_mw_ns"] = power_mw * delay_ns

    write_json(dc_dir / "structured_ppa_metrics.json", structured)
    return structured


# =========================================================
# Compliance evaluation
# =========================================================
def compare_with_operator(actual: float, operator: str, target: float) -> bool:
    if operator == "<=":
        return actual <= target
    if operator == "<":
        return actual < target
    if operator == ">=":
        return actual >= target
    if operator == ">":
        return actual > target
    if operator in {"=", "=="}:
        return actual == target
    raise ValueError(f"Unsupported operator: {operator}")


def evaluate_error_constraints(contract: Dict[str, Any], sim_metrics: Dict[str, Any]) -> Dict[str, Any]:
    allowed_metrics = {
        "mae", "wce", "mse", "ep", "ep_percent",
        "mre", "mre_percent", "wce_percent",
        "mae_percent", "wcre_percent"
    }

    measured = sim_metrics.get("error_metrics", {}) or {}

    constraints = contract.get("error_constraints", []) or []

    if not constraints:
        constraints = normalize_error_constraints(contract.get("error_contract"))

    details = []
    all_pass = True
    valid_constraint_count = 0

    for item in constraints:
        metric_raw = item.get("metric")
        if metric_raw is None:
            continue

        metric = normalize_name_token(metric_raw)
        if metric not in allowed_metrics:
            continue

        operator = str(item.get("operator", "<="))
        target = coerce_float(item.get("value"), 0.0)

        detail = {
            "metric": metric,
            "operator": operator,
            "target": target,
            "actual": measured.get(metric),
            "pass": None,
            "missing_metric": False,
            "violation_ratio": None,
        }

        valid_constraint_count += 1
        actual = measured.get(metric)

        if actual is None:
            detail["missing_metric"] = True
            detail["pass"] = False
            all_pass = False
        else:
            try:
                actual_f = float(actual)
                detail["actual"] = actual_f
                detail["pass"] = compare_with_operator(actual_f, operator, target)

                if operator in {"<=", "<"}:
                    denom = max(abs(target), 1e-12)
                    detail["violation_ratio"] = actual_f / denom
                elif operator in {">=", ">"}:
                    denom = max(abs(actual_f), 1e-12)
                    detail["violation_ratio"] = target / denom

                if not detail["pass"]:
                    all_pass = False
            except Exception:
                detail["pass"] = False
                all_pass = False

        details.append(detail)

    raw_error_contract = contract.get("error_contract")
    declared_error_intent = False

    if valid_constraint_count > 0:
        declared_error_intent = True
    else:
        if isinstance(raw_error_contract, dict) and raw_error_contract:
            if (
                raw_error_contract.get("primary_metric") is not None
                or raw_error_contract.get("max_error_value") is not None
                or raw_error_contract.get("max_error_percent") is not None
                or str(raw_error_contract.get("notes", "")).strip()
            ):
                declared_error_intent = True

        if not declared_error_intent:
            parser_notes = contract.get("parser_notes", [])
            if parser_notes:
                joined = " ".join(str(x) for x in parser_notes)
                if re.search(r"(?i)\b(MAE|WCE|MSE|EP|MRE|WCRE)\b", joined):
                    declared_error_intent = True

    if valid_constraint_count == 0:
        if declared_error_intent:
            all_pass = False
            details.append({
                "metric": "__constraint_parse_failure__",
                "operator": "N/A",
                "target": None,
                "actual": None,
                "pass": False,
                "missing_metric": True,
                "violation_ratio": None,
                "notes": "User/Parser declared error constraints, but no valid normalized error_constraints were produced.",
            })
        else:
            all_pass = True

    return {
        "all_error_constraints_pass": all_pass,
        "details": details,
        "valid_constraint_count": valid_constraint_count,
    }


def evaluate_soft_objectives(contract: Dict[str, Any], ppa_metrics: Dict[str, Any]) -> Dict[str, Any]:
    soft = contract.get("soft_objectives", {}) or {}
    targets = soft.get("targets", {}) or {}

    measured_map = {
        "power_mw": ppa_metrics.get("power_mw"),
        "area_um2": ppa_metrics.get("area_um2"),
        "delay_ns": ppa_metrics.get("delay_ns"),
        "pdp_mw_ns": ppa_metrics.get("pdp_mw_ns"),
        "adp_area_delay": ppa_metrics.get("adp_area_delay"),
    }

    details = []
    explicit_soft_pass: Optional[bool] = True if targets else None
    saw_missing = False

    for metric, spec in targets.items():
        actual = measured_map.get(metric)
        operator = str(spec.get("operator", "<="))
        target = coerce_float(spec.get("value"), 0.0)

        detail = {
            "metric": metric,
            "operator": operator,
            "target": target,
            "actual": actual,
            "pass": None,
            "missing_metric": False,
        }

        if actual is None:
            detail["missing_metric"] = True
            detail["pass"] = None
            saw_missing = True
        else:
            try:
                passed = compare_with_operator(float(actual), operator, target)
                detail["pass"] = passed
                if not passed:
                    explicit_soft_pass = False
            except Exception:
                detail["pass"] = False
                explicit_soft_pass = False

        details.append(detail)

    if explicit_soft_pass is True and saw_missing:
        explicit_soft_pass = None

    return {
        "priority_order": soft.get("priority_order", []),
        "explicit_soft_target_pass": explicit_soft_pass,
        "details": details,
    }


def evaluate_contract_compliance(
    contract: Dict[str, Any],
    sim_metrics: Dict[str, Any],
    ppa_metrics: Dict[str, Any],
    iteration_dir: Path,
    cfg: SystemConfig,
) -> Dict[str, Any]:
    compile_pass = sim_metrics.get("compile_pass", False)
    simulation_pass = sim_metrics.get("simulation_pass", False)
    xz_detected = sim_metrics.get("xz_detected", False)

    error_eval = evaluate_error_constraints(contract, sim_metrics)
    soft_eval = evaluate_soft_objectives(contract, ppa_metrics)

    dc_status = ppa_metrics.get("dc_status")
    area_um2 = ppa_metrics.get("area_um2")
    delay_ns = ppa_metrics.get("delay_ns")
    power_mw = ppa_metrics.get("power_mw")

    ppa_ready = (
        dc_status == "done" and
        area_um2 is not None and
        delay_ns is not None and
        power_mw is not None
    )

    ppa_required_configured = parse_bool(contract.get("require_ppa"), cfg.require_ppa_for_accept)

    hard_pass = (
        compile_pass and
        simulation_pass and
        (not xz_detected) and
        error_eval["all_error_constraints_pass"]
    )

    # PPA only becomes blocking when:
    #   1) the contract/config says PPA is required, and
    #   2) PPA metrics are actually available
    effective_ppa_required = bool(ppa_required_configured and ppa_ready)

    soft_blocking_fail = bool(
        effective_ppa_required and
        soft_eval.get("explicit_soft_target_pass") is False
    )

    full_accept_ready = bool(hard_pass and not soft_blocking_fail)

    compliance = {
        "compile_pass": compile_pass,
        "simulation_pass": simulation_pass,
        "xz_detected": xz_detected,

        "error_evaluation": error_eval,
        "soft_evaluation": soft_eval,

        "ppa_required": ppa_required_configured,
        "ppa_required_configured": ppa_required_configured,
        "effective_ppa_required": effective_ppa_required,
        "ppa_ignored_due_to_unavailable": bool(ppa_required_configured and not ppa_ready),

        "ppa_ready": ppa_ready,
        "ppa_observation_only": not effective_ppa_required,
        "dc_status": dc_status,
        "ppa_snapshot": {
            "area_um2": area_um2,
            "delay_ns": delay_ns,
            "power_mw": power_mw,
        },

        "hard_pass": hard_pass,
        "full_accept_ready": full_accept_ready,
    }

    write_json(iteration_dir / "compliance_report.json", compliance)
    return compliance


# =========================================================
# Critic
# =========================================================
def build_critic_evidence_bundle(
    parser_output: Dict[str, Any],
    planner_output: Dict[str, Any],
    template_bundle: Dict[str, Any],
    generator_output: Dict[str, Any],
    verification_result: Dict[str, Any],
    sim_metrics: Dict[str, Any],
    dc_result: Dict[str, Any],
    ppa_metrics: Dict[str, Any],
    compliance: Dict[str, Any],
    iteration_dir: Path,
) -> Dict[str, Any]:
    bundle_type = template_bundle.get("bundle_type")
    template_summary: Dict[str, Any]

    if bundle_type == "primitive":
        template_summary = {
            "bundle_type": "primitive",
            "plan_key": template_bundle.get("plan_key"),
            "primary_strategy_id": template_bundle.get("primary_entry", {}).get("strategy_id"),
            "candidate_strategy_ids": [x["strategy_entry"]["strategy_id"] for x in template_bundle.get("candidate_templates", [])],
        }
    else:
        template_summary = {
            "bundle_type": "composite",
            "plan_key": template_bundle.get("plan_key"),
            "target_circuit": template_bundle.get("target_circuit"),
            "component_mapping": template_bundle.get("component_mapping", {}),
            "adder_strategy_ids": [x["strategy_entry"]["strategy_id"] for x in template_bundle.get("adder_templates", [])],
            "multiplier_strategy_ids": [x["strategy_entry"]["strategy_id"] for x in template_bundle.get("multiplier_templates", [])],
            "skeleton_id": template_bundle.get("skeleton", {}).get("skeleton_entry", {}).get("skeleton_id"),
        }

    return {
        "contract": parser_output["contract"],
        "refined_spec": parser_output["refined_spec"],
        "planner_output": planner_output,
        "template_bundle_summary": template_summary,
        "generator": {
            "module_name": generator_output.get("module_name", ""),
            "strategy_id": generator_output.get("strategy_id", ""),
            "plan_type": generator_output.get("plan_type", ""),
            "used_component_ids": generator_output.get("used_component_ids", {}),
            "generation_mode": generator_output.get("generation_mode", ""),
            "applied_knobs": generator_output.get("applied_knobs", {}),
            "changed_regions": generator_output.get("changed_regions", []),
            "design_notes": generator_output.get("design_notes", ""),
            "rtl_code": generator_output.get("rtl_code", ""),
        },
        "verification": verification_result,
        "sim_metrics": sim_metrics,
        "dc": dc_result,
        "ppa_metrics": ppa_metrics,
        "compliance": compliance,
        "artifact_paths": {
            "iteration_dir": str(iteration_dir),
            "generator_dir": str(iteration_dir / "generator"),
            "verification_dir": str(iteration_dir / "verification"),
            "dc_dir": str(iteration_dir / "dc"),
        },
    }

def make_rule_based_critic_output(
    evidence: Dict[str, Any],
    cfg: SystemConfig,
    strategy_micro_tune_count: int,
) -> Dict[str, Any]:
    compliance = evidence["compliance"]
    ppa_metrics = evidence["ppa_metrics"]
    strategy_id = evidence["generator"].get("strategy_id")

    compile_pass = compliance["compile_pass"]
    sim_pass = compliance["simulation_pass"]
    xz = compliance["xz_detected"]
    ppa_ready = compliance["ppa_ready"]
    hard_pass = compliance["hard_pass"]
    soft_eval = compliance["soft_evaluation"]
    err_details = compliance["error_evaluation"]["details"]

    if (not compile_pass) or (not sim_pass) or xz:
        return {
            "type": "A",
            "subtype": "A0",
            "type_name": "Implementation Failure",
            "route_to": "generator",
            "action_mode": "repair",
            "reason": "Compile/simulation failed or pervasive X/Z states detected.",
            "root_cause": "invalid_rtl_or_runtime_failure",
            "same_strategy_required": True,
            "same_interface_required": True,
            "banned_strategy_ids": [],
            "allowed_edit_types": ["syntax_fix", "interface_fix", "wire_fix", "register_fix"],
            "forbidden_actions": ["change_circuit_family", "change_strategy"],
            "micro_tune_targets": {},
            "evidence": {
                "compile_pass": compile_pass,
                "simulation_pass": sim_pass,
                "xz_detected": xz,
            },
        }

    missing_metrics = any(d.get("missing_metric") for d in err_details)
    if missing_metrics:
        return {
            "type": "B",
            "subtype": "B1",
            "type_name": "Implementation-Structured Mismatch",
            "route_to": "generator",
            "action_mode": "repair",
            "reason": "Simulation ran but required error metrics are missing; implementation or TB alignment is likely off.",
            "root_cause": "metric_observability_or_interface_mismatch",
            "same_strategy_required": True,
            "same_interface_required": True,
            "banned_strategy_ids": [],
            "allowed_edit_types": ["output_alignment_fix", "interface_fix", "module_behavior_fix"],
            "forbidden_actions": ["change_circuit_family", "change_strategy"],
            "micro_tune_targets": {},
            "evidence": {"error_details": err_details},
        }

    huge_violation = False
    slight_violation = False
    for d in err_details:
        if d.get("pass") is False:
            ratio = d.get("violation_ratio")
            if ratio is not None and float(ratio) >= cfg.huge_error_ratio:
                huge_violation = True
            else:
                slight_violation = True

    if huge_violation:
        return {
            "type": "B",
            "subtype": "B2",
            "type_name": "Strategy Mismatch",
            "route_to": "planner",
            "action_mode": "reselect_strategy",
            "reason": "Executable RTL exists, but error violation is too large for local repair; strategy likely mismatched.",
            "root_cause": "selected_strategy_too_aggressive_or_inappropriate",
            "same_strategy_required": False,
            "same_interface_required": True,
            "banned_strategy_ids": [strategy_id] if strategy_id else [],
            "allowed_edit_types": [],
            "forbidden_actions": [],
            "micro_tune_targets": {},
            "evidence": {"error_details": err_details, "selected_strategy_id": strategy_id},
        }

    soft_targets_failed = soft_eval.get("explicit_soft_target_pass") is False

    if slight_violation or soft_targets_failed:
        if strategy_micro_tune_count >= cfg.max_micro_tune_rounds_per_strategy:
            return {
                "type": "B",
                "subtype": "B2",
                "type_name": "Strategy Mismatch After Exhausted Local Tuning",
                "route_to": "planner",
                "action_mode": "reselect_strategy",
                "reason": "Local tuning budget on this strategy is exhausted; re-plan is preferred.",
                "root_cause": "micro_tune_budget_exhausted",
                "same_strategy_required": False,
                "same_interface_required": True,
                "banned_strategy_ids": [strategy_id] if strategy_id else [],
                "allowed_edit_types": [],
                "forbidden_actions": [],
                "micro_tune_targets": {},
                "evidence": {
                    "strategy_micro_tune_count": strategy_micro_tune_count,
                    "soft_eval": soft_eval,
                    "error_details": err_details,
                },
            }

        return {
            "type": "C",
            "subtype": "C1",
            "type_name": "Near-Feasible / Local-Tunable",
            "route_to": "generator",
            "action_mode": "micro_tune",
            "reason": "Design is structurally close; use bounded micro-tuning to recover error/PPA targets.",
            "root_cause": "over_aggressive_local_approximation_or_soft_ppa_gap",
            "same_strategy_required": True,
            "same_interface_required": True,
            "banned_strategy_ids": [],
            "allowed_edit_types": [
                "increase_exact_bits",
                "protect_boundary_carry",
                "replace_selected_submodule",
                "lightweight_compensation",
            ],
            "forbidden_actions": ["change_circuit_family", "change_strategy", "full_rewrite"],
            "micro_tune_targets": {
                "prefer_small_delta": True,
                "preserve_interface": True,
                "preserve_selected_strategy": True,
            },
            "evidence": {
                "error_details": err_details,
                "soft_eval": soft_eval,
                "ppa_ready": ppa_ready,
                "dc_status": ppa_metrics.get("dc_status"),
                "strategy_micro_tune_count": strategy_micro_tune_count,
            },
        }

    if hard_pass:
        return {
            "type": "D",
            "subtype": "D0",
            "type_name": "Accept",
            "route_to": "accept",
            "action_mode": "accept",
            "reason": "Hard constraints satisfied. PPA evidence is optional and non-blocking; current result is acceptable.",
            "root_cause": "contract_satisfied",
            "same_strategy_required": True,
            "same_interface_required": True,
            "banned_strategy_ids": [],
            "allowed_edit_types": [],
            "forbidden_actions": [],
            "micro_tune_targets": {},
            "evidence": {
                "hard_pass": hard_pass,
                "ppa_ready": ppa_ready,
                "dc_status": ppa_metrics.get("dc_status"),
                "soft_eval": soft_eval,
                "strategy_id": strategy_id,
            },
        }

    return {
        "type": "B",
        "subtype": "B1",
        "type_name": "Implementation-Structured Mismatch",
        "route_to": "generator",
        "action_mode": "repair",
        "reason": "Fallback rule: executable but not contract-ready; attempt same-strategy repair first.",
        "root_cause": "unclassified_repairable_issue",
        "same_strategy_required": True,
        "same_interface_required": True,
        "banned_strategy_ids": [],
        "allowed_edit_types": ["logic_fix", "interface_fix", "small_structure_fix"],
        "forbidden_actions": ["change_circuit_family", "change_strategy"],
        "micro_tune_targets": {},
        "evidence": {},
    }

def normalize_critic_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(raw)
    ctype = str(out.get("type") or out.get("chosen_type") or "").strip().upper()
    if ctype not in {"A", "B", "C", "D"}:
        ctype = "A"
    subtype = str(out.get("subtype") or "").strip().upper()
    if not subtype:
        subtype = {"A": "A0", "B": "B1", "C": "C1", "D": "D0"}[ctype]
    route_to = str(out.get("route_to") or "").strip().lower()
    action_mode = str(out.get("action_mode") or out.get("suggested_next_action") or "").strip().lower()
    if not route_to:
        route_to = {"A": "generator", "B": "generator", "C": "generator", "D": "accept"}[ctype]
    if not action_mode:
        action_mode = {"A": "repair", "B": "repair", "C": "micro_tune", "D": "accept"}[ctype]
    if ctype == "B" and subtype == "B2":
        route_to = "planner"
        action_mode = "reselect_strategy"
    out["type"] = ctype
    out["subtype"] = subtype
    out["route_to"] = route_to
    out["action_mode"] = action_mode
    out["same_strategy_required"] = parse_bool(out.get("same_strategy_required"), ctype in {"A", "C"} or subtype == "B1")
    out["same_interface_required"] = parse_bool(out.get("same_interface_required"), True)
    out["banned_strategy_ids"] = parse_listish(out.get("banned_strategy_ids"))
    out["allowed_edit_types"] = parse_listish(out.get("allowed_edit_types"))
    out["forbidden_actions"] = parse_listish(out.get("forbidden_actions"))
    if not isinstance(out.get("micro_tune_targets"), dict):
        out["micro_tune_targets"] = {}
    if not isinstance(out.get("evidence"), dict):
        out["evidence"] = {}
    return out


def run_critic(
    llm: LLMClient,
    parser_output: Dict[str, Any],
    planner_output: Dict[str, Any],
    template_bundle: Dict[str, Any],
    generator_output: Dict[str, Any],
    verification_result: Dict[str, Any],
    sim_metrics: Dict[str, Any],
    dc_result: Dict[str, Any],
    ppa_metrics: Dict[str, Any],
    compliance: Dict[str, Any],
    iteration_idx: int,
    run_dir: Path,
    cfg: SystemConfig,
    strategy_micro_tune_count: int,
) -> Dict[str, Any]:
    print_banner(f"STEP 10 / Critic / Iteration {iteration_idx}")
    stage_dir = ensure_dir(run_dir / f"iteration_{iteration_idx}" / "critic")
    policy_path = CRITIC_DIR / "critic_policy.txt"
    critic_policy = read_text(policy_path, default="") if policy_path else ""

    evidence = build_critic_evidence_bundle(
        parser_output,
        planner_output,
        template_bundle,
        generator_output,
        verification_result,
        sim_metrics,
        dc_result,
        ppa_metrics,
        compliance,
        run_dir / f"iteration_{iteration_idx}",
    )
    evidence["iteration_idx"] = iteration_idx
    write_json(stage_dir / "critic_evidence_bundle.json", evidence)
    if critic_policy:
        write_text(stage_dir / "critic_policy.txt", critic_policy)

    rule_output = make_rule_based_critic_output(evidence, cfg, strategy_micro_tune_count)
    write_json(stage_dir / "critic_rule_output.json", rule_output)


    if not (cfg.critic_use_llm and llm.config.has_agent("critic")):
        normalized = normalize_critic_output(rule_output)
        write_json(stage_dir / "critic_output.json", normalized)
        print_sub("Critic output (rule-based only)")
        print(pretty_json(normalized))
        return normalized

    system_prompt = load_prompt("critic")
    user_prompt = textwrap.dedent(
        f"""
        Contract:
        {json.dumps(parser_output['contract'], ensure_ascii=False, indent=2)}

        Critic policy:
        {critic_policy if critic_policy else '[EMPTY]'}

        Rule-based pre-classification:
        {json.dumps(rule_output, ensure_ascii=False, indent=2)}

        Evidence bundle:
        {json.dumps(evidence, ensure_ascii=False, indent=2)}

        You may keep the rule-based classification or refine it, but you must output strict JSON.

        Required output format:
        {{
          "type": "A|B|C|D",
          "subtype": "A0|B1|B2|C1|C2|D0",
          "type_name": "...",
          "route_to": "generator|planner|accept|abort",
          "action_mode": "repair|reselect_strategy|micro_tune|accept|abort",
          "reason": "...",
          "root_cause": "...",
          "same_strategy_required": true,
          "same_interface_required": true,
          "banned_strategy_ids": [],
          "allowed_edit_types": [],
          "forbidden_actions": [],
          "micro_tune_targets": {{ ... }},
          "evidence": {{ ... }}
        }}
        """
    ).strip()

    critic_output = llm.chat_json_strict("critic", system_prompt, user_prompt, stage_dir)
    normalized = normalize_critic_output(critic_output)
    write_json(stage_dir / "critic_output.json", normalized)
    print_sub("Critic output")
    print(pretty_json(normalized))
    return normalized


# =========================================================
# Orchestration helpers
# =========================================================
def validate_required_files(app_cfg: AppConfig) -> None:
    for name in [
        "parser", "planner", "generator_instantiate", "generator_repair",
        "generator_micro_tune", "critic"
    ]:
        load_prompt(name)
    if not CATALOGS_DIR.exists():
        raise FileNotFoundError(f"catalogs directory not found: {CATALOGS_DIR}")
    if not TESTBENCH_DIR.exists():
        ensure_dir(TESTBENCH_DIR)
    if not app_cfg.has_agent("parser") or not app_cfg.has_agent("planner") or not app_cfg.has_agent("generator"):
        raise RuntimeError("agent_config.json must define parser / planner / generator")


def collect_multiline_input() -> str:
    print("请输入近似电路设计需求。")
    print("支持多行输入；输入空行并回车结束。")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def find_strategy_by_id(entries: List[Dict[str, Any]], strategy_id: str) -> Dict[str, Any]:
    for entry in entries:
        if entry.get("strategy_id") == strategy_id:
            return entry
    raise KeyError(f"Strategy ID not found: {strategy_id}")


def summarize_iteration(
    iteration_idx: int,
    planner_output: Dict[str, Any],
    generator_output: Dict[str, Any],
    sim_metrics: Dict[str, Any],
    ppa_metrics: Dict[str, Any],
    compliance: Dict[str, Any],
    critic_output: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "iteration": iteration_idx,
        "plan_type": planner_output.get("plan_type"),
        "plan_key": planner_output.get("plan_key"),
        "primary_strategy_id": planner_output.get("primary_strategy_id"),
        "selected_strategy_ids": planner_output.get("selected_strategy_ids", []),
        "selected_adder_ids": planner_output.get("selected_adder_ids", []),
        "selected_multiplier_ids": planner_output.get("selected_multiplier_ids", []),
        "selected_skeleton_id": planner_output.get("selected_skeleton_id"),
        "module_name": generator_output.get("module_name"),
        "generator_strategy_id": generator_output.get("strategy_id"),
        "used_component_ids": generator_output.get("used_component_ids", {}),
        "generation_mode": generator_output.get("generation_mode"),

        "sim_metrics": extract_reportable_sim_metrics(sim_metrics),
        "ppa_metrics": extract_reportable_ppa_metrics(ppa_metrics),

        "hard_pass": compliance.get("hard_pass"),
        "ppa_ready": compliance.get("ppa_ready"),
        "full_accept_ready": compliance.get("full_accept_ready"),

        "critic_type": critic_output.get("type"),
        "critic_subtype": critic_output.get("subtype"),
        "critic_type_name": critic_output.get("type_name"),
        "route_to": critic_output.get("route_to"),
        "action_mode": critic_output.get("action_mode"),
        "reason": critic_output.get("reason"),
    }

def normalize_family_filter_values(values: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for part in str(value).split(","):
            token = part.strip().lower()
            if token and token not in out:
                out.append(token)
    return out


def load_prompt_suite(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt suite file not found: {path}")

    prompts: List[Dict[str, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            raise ValueError(f"Invalid JSONL at {path}:{lineno}: {e}")

        prompt_id = str(obj.get("id", "")).strip()
        family = str(obj.get("family", "")).strip().lower()
        text = str(obj.get("text", "")).strip()

        if not prompt_id:
            raise ValueError(f"Missing prompt id at {path}:{lineno}")
        if not family:
            raise ValueError(f"Missing family at {path}:{lineno}")
        if not text:
            raise ValueError(f"Missing text at {path}:{lineno}")

        prompts.append({
            "id": prompt_id,
            "family": family,
            "text": text,
        })

    if not prompts:
        raise RuntimeError(f"Prompt suite is empty: {path}")
    return prompts


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_family_result_path(family: str) -> Path:
    return EXPERIMENTS_DIR / f"results_{safe_filename(family)}.jsonl"

def extract_reportable_sim_metrics(sim_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    s = sim_metrics if isinstance(sim_metrics, dict) else {}
    return {
        "compile_pass": s.get("compile_pass"),
        "simulation_pass": s.get("simulation_pass"),
        "xz_detected": s.get("xz_detected"),
        "error_metrics": s.get("error_metrics", {}) or {},
        "flags": s.get("flags", {}) or {},
    }

def extract_reportable_ppa_metrics(ppa_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    p = ppa_metrics if isinstance(ppa_metrics, dict) else {}
    return {
        "dc_status": p.get("dc_status"),
        "area_um2": p.get("area_um2"),
        "delay_ns": p.get("delay_ns"),
        "power_mw": p.get("power_mw"),
        "adp_area_delay": p.get("adp_area_delay"),
        "pdp_mw_ns": p.get("pdp_mw_ns"),
    }


def critic_is_accept(final_critic: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(final_critic, dict):
        return False
    ctype = str(final_critic.get("type", "")).upper()
    route_to = str(final_critic.get("route_to", "")).lower()
    action_mode = str(final_critic.get("action_mode", "")).lower()
    return ctype == "D" or route_to == "accept" or action_mode == "accept"


def build_experiment_result_record(
    prompt_meta: Dict[str, Any],
    run_dir: Path,
    overall_status: Dict[str, Any],
    final_critic: Optional[Dict[str, Any]],
    final_sim_metrics: Optional[Dict[str, Any]],
    final_ppa_metrics: Optional[Dict[str, Any]],
    fatal_error: Optional[str] = None,
) -> Dict[str, Any]:
    if fatal_error:
        terminal_status = "fatal_error"
        final_critic_type = "FATAL"
    elif critic_is_accept(final_critic):
        terminal_status = "accepted"
        final_critic_type = final_critic.get("type") if final_critic else None
    else:
        terminal_status = "finished_without_accept"
        final_critic_type = final_critic.get("type") if final_critic else None

    return {
        "id": prompt_meta.get("id"),
        "family": str(prompt_meta.get("family", "")).strip().lower(),
        "text": prompt_meta.get("text"),
        "run_dir": str(run_dir),
        "terminal_status": terminal_status,
        "final_critic_type": final_critic_type,
        "final_critic_subtype": final_critic.get("subtype") if isinstance(final_critic, dict) else None,
        "final_route_to": final_critic.get("route_to") if isinstance(final_critic, dict) else None,
        "final_action_mode": final_critic.get("action_mode") if isinstance(final_critic, dict) else None,
        "final_reason": final_critic.get("reason") if isinstance(final_critic, dict) else None,
        "parsed_family": overall_status.get("parsed_family"),
        "sim_metrics": extract_reportable_sim_metrics(final_sim_metrics),
        "ppa_metrics": extract_reportable_ppa_metrics(final_ppa_metrics),
        "fatal_error": fatal_error,
        "started_at": overall_status.get("started_at"),
        "finished_at": overall_status.get("finished_at"),
    }


def parse_batch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AX-AGENT batch runner for experiments/prompt_suite.jsonl"
    )
    parser.add_argument(
        "--prompt-suite",
        default=str(PROMPT_SUITE_PATH),
        help="Path to the JSONL prompt suite file.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Families to run, e.g. --families adder multiplier",
    )
    return parser.parse_args()


def run_single_request(
    llm: LLMClient,
    app_cfg: AppConfig,
    user_request: str,
    prompt_meta: Dict[str, Any],
) -> Dict[str, Any]:
    prompt_id = str(prompt_meta.get("id", "unknown")).strip()
    prompt_family = str(prompt_meta.get("family", "unknown")).strip().lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    run_dir = ensure_dir(RUNS_DIR / f"run_{safe_filename(prompt_id)}_{timestamp}")
    write_text(run_dir / "user_request.txt", user_request)
    write_json(run_dir / "prompt_meta.json", prompt_meta)
    write_json(run_dir / "agent_config_snapshot.json", sanitize_agent_config(app_cfg.raw))

    overall_status: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "prompt_id": prompt_id,
        "prompt_family": prompt_family,
        "started_at": datetime.now().isoformat(),
        "iterations": [],
        "planner_history": [],
        "strategy_micro_tune_counter": {},
    }
    write_json(run_dir / "run_status.json", overall_status)

    print_banner(f"AX-AGENT / Prompt {prompt_id} / Family={prompt_family}")
    print_sub("User request")
    print(user_request)

    final_generator: Optional[Dict[str, Any]] = None
    final_critic: Optional[Dict[str, Any]] = None
    final_sim_metrics: Optional[Dict[str, Any]] = None
    final_ppa_metrics: Optional[Dict[str, Any]] = None

    try:
        parser_output = run_parser(llm, user_request, run_dir, app_cfg.system)
        contract = parser_output["contract"]
        overall_status["parsed_family"] = contract.get("circuit_type")

        write_json(run_dir / "persistent_contract.json", contract)
        write_text(run_dir / "refined_spec.txt", parser_output["refined_spec"])

        catalog_router_result = run_catalog_router(contract, run_dir)
        max_iterations = contract["iteration_budget"]["max_iterations"]

        print_sub("Iteration budget")
        print(f"Actual max_iterations used = {max_iterations}")

        banned_strategy_ids: List[str] = []
        planner_feedback: Optional[Dict[str, Any]] = None
        critic_feedback: Optional[Dict[str, Any]] = None
        current_planner_output: Optional[Dict[str, Any]] = None
        current_template_bundle: Optional[Dict[str, Any]] = None
        generator_mode = "instantiate"
        prior_rtl: Optional[str] = None
        prior_design_notes: Optional[str] = None

        current_planner_output = run_planner(
            llm=llm,
            parser_output=parser_output,
            catalog_router_result=catalog_router_result,
            run_dir=run_dir,
            iteration_idx=1,
            cfg=app_cfg.system,
            banned_strategy_ids=banned_strategy_ids,
            critic_feedback=planner_feedback,
        )
        overall_status["planner_history"].append({
            "iteration": 1,
            "planner_output": current_planner_output,
            "banned_strategy_ids": banned_strategy_ids[:],
        })
        current_template_bundle = load_template_bundle(
            current_planner_output,
            catalog_router_result,
            run_dir,
            1,
        )

        for iteration_idx in range(1, max_iterations + 1):
            print_banner(f"ITERATION {iteration_idx} / {max_iterations}")

            if iteration_idx > 1 and generator_mode == "instantiate" and current_template_bundle is None:
                current_planner_output = run_planner(
                    llm=llm,
                    parser_output=parser_output,
                    catalog_router_result=catalog_router_result,
                    run_dir=run_dir,
                    iteration_idx=iteration_idx,
                    cfg=app_cfg.system,   # FIXED
                    banned_strategy_ids=banned_strategy_ids,
                    critic_feedback=planner_feedback,
                )
                overall_status["planner_history"].append({
                    "iteration": iteration_idx,
                    "planner_output": current_planner_output,
                    "banned_strategy_ids": banned_strategy_ids[:],
                })
                current_template_bundle = load_template_bundle(
                    current_planner_output,
                    catalog_router_result,
                    run_dir,
                    iteration_idx,
                )

            if current_planner_output is None or current_template_bundle is None:
                raise RuntimeError("Planner output/template bundle missing before generator stage")

            current_strategy_id = str(
                current_planner_output.get("plan_key")
                or current_planner_output.get("primary_strategy_id")
                or "unknown_plan"
            )
            strategy_micro_tune_count = overall_status["strategy_micro_tune_counter"].get(current_strategy_id, 0)

            generator_output = run_generator(
                llm=llm,
                parser_output=parser_output,
                planner_output=current_planner_output,
                template_bundle=current_template_bundle,
                mode=generator_mode,
                iteration_idx=iteration_idx,
                run_dir=run_dir,
                prior_rtl=prior_rtl,
                prior_design_notes=prior_design_notes,
                critic_feedback=critic_feedback,
            )
            final_generator = generator_output

            iteration_dir = ensure_dir(run_dir / f"iteration_{iteration_idx}")
            rtl_file = iteration_dir / "generator" / f"{generator_output['module_name']}.v"
            if not rtl_file.exists():
                raise FileNotFoundError(f"Expected RTL file not found: {rtl_file}")

            tb_file, tb_context = prepare_testbench(
                llm=llm,
                contract=contract,
                module_name=generator_output["module_name"],
                rtl_code=generator_output["rtl_code"],
                template_bundle=current_template_bundle,
                iteration_dir=iteration_dir,
                cfg=app_cfg.system,
                interface_info=generator_output.get("interface_info", {}),
            )

            verification_result = run_iverilog_flow(
                rtl_file=rtl_file,
                tb_file=tb_file,
                module_name=generator_output["module_name"],
                iteration_dir=iteration_dir,
                compile_defines=tb_context.get("compile_defines", {}),
                cfg=app_cfg.system,
            )
            sim_metrics = parse_sim_metrics(iteration_dir, verification_result)
            final_sim_metrics = sim_metrics
            print_sub("Structured simulation metrics")
            print(pretty_json(sim_metrics))

            dc_result = run_dc_flow(
                app_cfg=app_cfg,
                rtl_file=rtl_file,
                module_name=generator_output["module_name"],
                iteration_dir=iteration_dir,
                contract=contract,
            )
            ppa_metrics = parse_ppa_metrics(iteration_dir, dc_result)
            final_ppa_metrics = ppa_metrics

            print_sub("Structured PPA metrics")
            print(pretty_json(ppa_metrics))

            compliance = evaluate_contract_compliance(
                contract=contract,
                sim_metrics=sim_metrics,
                ppa_metrics=ppa_metrics,
                iteration_dir=iteration_dir,
                cfg=app_cfg.system,
            )

            critic_output = run_critic(
                llm=llm,
                parser_output=parser_output,
                planner_output=current_planner_output,
                template_bundle=current_template_bundle,
                generator_output=generator_output,
                verification_result=verification_result,
                sim_metrics=sim_metrics,
                dc_result=dc_result,
                ppa_metrics=ppa_metrics,
                compliance=compliance,
                iteration_idx=iteration_idx,
                run_dir=run_dir,
                cfg=app_cfg.system,
                strategy_micro_tune_count=strategy_micro_tune_count,
            )
            final_critic = critic_output

            if critic_output.get("action_mode") == "micro_tune":
                overall_status["strategy_micro_tune_counter"][current_strategy_id] = strategy_micro_tune_count + 1

            iter_summary = summarize_iteration(
                iteration_idx=iteration_idx,
                planner_output=current_planner_output,
                generator_output=generator_output,
                sim_metrics=sim_metrics,
                ppa_metrics=ppa_metrics,
                compliance=compliance,
                critic_output=critic_output,
            )
            overall_status["iterations"].append(iter_summary)
            write_json(run_dir / "run_status.json", overall_status)

            print_sub(f"Iteration {iteration_idx} summary")
            print(pretty_json(iter_summary))

            prior_rtl = generator_output.get("rtl_code")
            prior_design_notes = generator_output.get("design_notes")
            critic_feedback = critic_output
            planner_feedback = critic_output

            ctype = str(critic_output.get("type", "")).upper()
            route_to = str(critic_output.get("route_to", "")).lower()
            action_mode = str(critic_output.get("action_mode", "")).lower()

            print_sub("Loop control decision")
            print(f"Critic type   : {ctype}")
            print(f"Critic subtype: {critic_output.get('subtype')}")
            print(f"Route to      : {route_to}")
            print(f"Action mode   : {action_mode}")

            if ctype == "D" or route_to == "accept" or action_mode == "accept":
                print("[INFO] Critic accepted the design. Stop iterations.")
                break
            if route_to == "abort" or action_mode == "abort":
                print("[INFO] Critic requested abort. Stop iterations.")
                break

            if route_to == "generator":
                generator_mode = "micro_tune" if action_mode == "micro_tune" else "repair"
                print(f"[INFO] Next round stays on strategy_id={current_strategy_id}, mode={generator_mode}")

            elif route_to == "planner":
                raw_ban_ids = parse_listish(critic_output.get("banned_strategy_ids"))
                ban_ids: List[str] = []

                # keep critic-suggested bans
                for sid in raw_ban_ids:
                    if sid and sid not in ban_ids:
                        ban_ids.append(sid)

                # FIX: primitive should ban actual strategy_id, not composite-like plan_key
                if current_planner_output.get("plan_type") == "primitive":
                    actual_sid = str(
                        generator_output.get("strategy_id")
                        or current_planner_output.get("primary_strategy_id")
                        or ""
                    ).strip()
                    if actual_sid and actual_sid not in ban_ids:
                        ban_ids.append(actual_sid)
                else:
                    plan_key = str(current_planner_output.get("plan_key", "")).strip()
                    if plan_key and plan_key not in ban_ids:
                        ban_ids.append(plan_key)

                for sid in ban_ids:
                    if sid and sid not in banned_strategy_ids:
                        banned_strategy_ids.append(sid)

                generator_mode = "instantiate"
                current_template_bundle = None
                critic_feedback = None
                prior_rtl = None
                prior_design_notes = None
                print(f"[INFO] Next round will re-plan. Banned strategies now: {banned_strategy_ids}")

            else:
                generator_mode = "repair"
                print("[WARN] Unknown critic route. Fallback to generator repair.")

        overall_status["finished_at"] = datetime.now().isoformat()
        overall_status["final_critic_type"] = final_critic.get("type") if final_critic else None
        overall_status["final_critic_subtype"] = final_critic.get("subtype") if final_critic else None
        overall_status["final_route_to"] = final_critic.get("route_to") if final_critic else None
        overall_status["final_action_mode"] = final_critic.get("action_mode") if final_critic else None

        write_json(run_dir / "run_status.json", overall_status)
        write_json(run_dir / "final_result.json", {
            "prompt_meta": prompt_meta,
            "user_request": user_request,
            "persistent_contract_path": str(run_dir / "persistent_contract.json"),
            "run_status": overall_status,
            "final_generator_output": final_generator,
            "final_critic_output": final_critic,
            "final_sim_metrics": extract_reportable_sim_metrics(final_sim_metrics),
            "final_ppa_metrics": extract_reportable_ppa_metrics(final_ppa_metrics),
        })

        result_record = build_experiment_result_record(
            prompt_meta=prompt_meta,
            run_dir=run_dir,
            overall_status=overall_status,
            final_critic=final_critic,
            final_sim_metrics=final_sim_metrics,
            final_ppa_metrics=final_ppa_metrics,
            fatal_error=None,
        )
        write_json(run_dir / "experiment_result.json", result_record)

        print_banner("AX-AGENT run completed")
        print(f"Run directory: {run_dir}")
        if final_critic:
            print(f"Final critic type : {final_critic.get('type')} - {final_critic.get('type_name')}")
            print(f"Final subtype     : {final_critic.get('subtype')}")
            print(f"Final route_to    : {final_critic.get('route_to')}")
            print(f"Final action_mode : {final_critic.get('action_mode')}")
            print(f"Reason            : {final_critic.get('reason')}")

        return result_record

    except Exception as e:
        overall_status["finished_at"] = datetime.now().isoformat()
        overall_status["fatal_error"] = str(e)

        write_text(run_dir / "fatal_error.txt", str(e))
        write_json(run_dir / "run_status.json", overall_status)

        print_banner("FATAL ERROR")
        print(str(e))
        print(f"Run directory: {run_dir}")

        result_record = build_experiment_result_record(
            prompt_meta=prompt_meta,
            run_dir=run_dir,
            overall_status=overall_status,
            final_critic=final_critic,
            final_sim_metrics=final_sim_metrics,
            final_ppa_metrics=final_ppa_metrics,
            fatal_error=str(e),
        )
        write_json(run_dir / "experiment_result.json", result_record)
        return result_record

# =========================================================
# Main orchestration
# =========================================================
def main() -> None:
    print_banner("AX-AGENT / Batch Experiment Runner")
    ensure_dir(RUNS_DIR)
    ensure_dir(EXPERIMENTS_DIR)

    args = parse_batch_args()

    try:
        app_cfg = AppConfig(AGENT_CONFIG_PATH)
        validate_required_files(app_cfg)
    except Exception as e:
        print(f"[FATAL] Startup validation failed: {e}")
        sys.exit(1)

    llm = LLMClient(app_cfg)

    prompt_suite_path = Path(args.prompt_suite)
    if not prompt_suite_path.is_absolute():
        prompt_suite_path = ROOT / prompt_suite_path

    try:
        prompt_suite = load_prompt_suite(prompt_suite_path)
    except Exception as e:
        print(f"[FATAL] Failed to load prompt suite: {e}")
        sys.exit(1)

    selected_families = normalize_family_filter_values(args.families)
    if selected_families:
        prompt_suite = [x for x in prompt_suite if x["family"] in set(selected_families)]

    if not prompt_suite:
        print("[FATAL] No prompts matched the requested families.")
        sys.exit(1)

    families_in_run: List[str] = []
    for item in prompt_suite:
        fam = item["family"]
        if fam not in families_in_run:
            families_in_run.append(fam)

    results_by_family: Dict[str, List[Dict[str, Any]]] = {fam: [] for fam in families_in_run}

    for fam in families_in_run:
        result_path = make_family_result_path(fam)
        if result_path.exists():
            result_path.unlink()

    batch_manifest = {
        "prompt_suite_path": str(prompt_suite_path),
        "selected_families": families_in_run,
        "prompt_count": len(prompt_suite),
        "started_at": datetime.now().isoformat(),
        "result_files": {fam: str(make_family_result_path(fam)) for fam in families_in_run},
    }
    write_json(EXPERIMENTS_DIR / "last_batch_manifest.json", batch_manifest)

    for idx, prompt_item in enumerate(prompt_suite, start=1):
        prompt_id = prompt_item["id"]
        prompt_family = prompt_item["family"]
        prompt_text = prompt_item["text"]

        print_banner(f"BATCH PROGRESS {idx} / {len(prompt_suite)} :: {prompt_id} :: {prompt_family}")

        result_record = run_single_request(
            llm=llm,
            app_cfg=app_cfg,
            user_request=prompt_text,
            prompt_meta=prompt_item,
        )

        results_by_family.setdefault(prompt_family, []).append(result_record)
        write_jsonl(make_family_result_path(prompt_family), results_by_family[prompt_family])

    batch_manifest["finished_at"] = datetime.now().isoformat()
    batch_manifest["counts_by_family"] = {
        fam: len(rows) for fam, rows in results_by_family.items()
    }
    batch_manifest["accepted_count_by_family"] = {
        fam: sum(1 for row in rows if row.get("terminal_status") == "accepted")
        for fam, rows in results_by_family.items()
    }
    write_json(EXPERIMENTS_DIR / "last_batch_manifest.json", batch_manifest)

    print_banner("BATCH RUN COMPLETED")
    print(pretty_json({
        "prompt_suite_path": str(prompt_suite_path),
        "families": families_in_run,
        "prompt_count": len(prompt_suite),
        "result_files": batch_manifest["result_files"],
        "accepted_count_by_family": batch_manifest["accepted_count_by_family"],
    }))


if __name__ == "__main__":
    main()
