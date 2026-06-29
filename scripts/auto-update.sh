#!/usr/bin/env bash
# turbofit auto-update system
# Checks for updates to managed components (git repos, pip/uv packages, npm/cargo bins)
# and applies them if anything changed. Designed to run from a systemd user timer.
#
# Config: ~/.config/turbofit/auto-update.yaml
# Log:    ~/.local/share/turbofit/logs/auto-update.log
#
# Usage:
#   auto-update.sh                 # run all enabled components
#   auto-update.sh --check         # only report what would change, do not apply
#   auto-update.sh --component X   # only run component X
#   auto-update.sh --summary       # emit Discord-ready summary as last line
#
# Robustness: one component failing never stops the next. All errors are logged
# and reflected in the summary. Exit code is 0 unless the script itself could
# not boot (missing config, etc.); per-component failures are surfaced in the
# summary instead, so a timer firing hourly does not go red on a single flaky
# repo.

set -uo pipefail

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
readonly HOME_DIR="/home/sovthpaw"
readonly CONFIG_FILE="${HOME_DIR}/.config/turbofit/auto-update.yaml"
readonly LOG_DIR="${HOME_DIR}/.local/share/turbofit/logs"
readonly LOG_FILE="${LOG_DIR}/auto-update.log"
readonly SUMMARY_FILE="${LOG_DIR}/auto-update-summary.txt"
readonly PY="python3.12"   # python3.11 has known issues on this host

# Ensure runtime dirs exist
mkdir -p "${LOG_DIR}"

# ----------------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------------
CHECK_ONLY=0
ONLY_COMPONENT=""
EMIT_SUMMARY=0
START_EPOCH=$(date +%s)
START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Per-run accumulators
declare -a UPDATED_NAMES=()
declare -a SKIPPED_NAMES=()
declare -a FAILED_NAMES=()
declare -a FAILED_ERRORS=()
declare -a RESTARTED_SERVICES=()

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
log() {
    # log <level> <message>
    local level="$1"; shift
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local line="[${ts}] [${level}] $*"
    echo "${line}" | tee -a "${LOG_FILE}" >&2
}

log_info()  { log "INFO"  "$@"; }
log_warn()  { log "WARN"  "$@"; }
log_error() { log "ERROR" "$@"; }

# ----------------------------------------------------------------------------
# Config parsing (YAML via python3.12)
# Reads auto-update.yaml and emits shell-evalable lines:
#   TURBOFIT_CHECK_INTERVAL=<n>
#   COMPONENTS="name1 name2 ..."
#   COMP_<name>_enabled=true|false
#   COMP_<name>_type=git|pip|npm|cargo
#   COMP_<name>_repo=<path>          (git)
#   COMP_<name>_rebuild=true|false   (git)
#   COMP_<name>_build_cmd=<cmd>      (git, optional)
#   COMP_<name>_package=<pkg>        (pip/npm/cargo)
#   COMP_<name>_restart_services=<csv list>
# ----------------------------------------------------------------------------
load_config() {
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        log_error "Config file not found: ${CONFIG_FILE}"
        exit 1
    fi

    # Export HOME so ~ expansion inside the python helper works regardless of
    # how systemd invoked us.
    export HOME="${HOME_DIR}"

    local cfg
    if ! cfg="$("${PY}" - "${CONFIG_FILE}" <<'PYEOF'
import os, sys, yaml, shlex

path = sys.argv[1]
with open(path) as f:
    data = yaml.safe_load(f) or {}

interval = int(data.get("check_interval", 3600))
print(f"TURBOFIT_CHECK_INTERVAL={interval}")

comps = data.get("components", {}) or {}
names = []
for name, spec in comps.items():
    if not isinstance(spec, dict):
        continue
    safe = name.replace("-", "_")
    names.append(name)
    # shlex.quote values so eval is safe
    def emit(key, val):
        if val is None:
            val = ""
        if isinstance(val, bool):
            val = "true" if val else "false"
        if isinstance(val, list):
            val = ",".join(str(v) for v in val)
        print(f"COMP_{safe}_{key}={shlex.quote(str(val))}")
    emit("enabled",           spec.get("enabled", True))
    emit("type",              spec.get("type", "git"))
    emit("repo",              spec.get("repo", ""))
    emit("rebuild",           spec.get("rebuild", False))
    emit("build_cmd",         spec.get("build_cmd", ""))
    emit("package",           spec.get("package", ""))
    emit("restart_services",  spec.get("restart_services", []))

# Emit as a single shlex-quoted string so `eval "${cfg}"` sets COMPONENTS to
# one space-separated value, not `VAR='a' 'b'` (which bash runs 'b' as a command).
print("COMPONENTS=" + shlex.quote(" ".join(names)))
PYEOF
    )"; then
        log_error "Failed to parse config (is python3.12 + pyyaml available?)"
        exit 1
    fi

    # eval is safe here: all values were shlex-quoted by the python helper.
    eval "${cfg}"

    if [[ -z "${COMPONENTS:-}" ]]; then
        log_warn "No components defined in config."
    fi
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# Expand ~ in a path. systemd user units set HOME, but be explicit.
expand_path() {
    local p="$1"
    p="${p/#\~/${HOME_DIR}}"
    echo "${p}"
}

# Restart a systemd user service if it is active. Returns 0 if (re)started,
# 1 if it was not running, 2 on error.
maybe_restart_service() {
    local svc="$1"
    if [[ -z "${svc}" ]]; then
        return 1
    fi
    # Normalize: strip any trailing .service if user included it
    svc="${svc%.service}"
    if ! systemctl --user is-active --quiet "${svc}.service" 2>/dev/null; then
        return 1
    fi
    log_info "Restarting ${svc}.service (was active)"
    if systemctl --user restart "${svc}.service" 2>>"${LOG_FILE}"; then
        return 0
    else
        log_error "Failed to restart ${svc}.service"
        return 2
    fi
}

# Record a per-component outcome.
record_updated() {
    UPDATED_NAMES+=("$1")
}
record_skipped() {
    SKIPPED_NAMES+=("$1")
}
record_failed() {
    FAILED_NAMES+=("$1")
    FAILED_ERRORS+=("$2")
}
record_restarted() {
    RESTARTED_SERVICES+=("$1")
}

# ----------------------------------------------------------------------------
# Component handlers
# ----------------------------------------------------------------------------

# update_git_component <name> <repo> <rebuild> <build_cmd> <restart_csv>
update_git_component() {
    local name="$1"
    local repo_raw="$2"
    local rebuild="$3"
    local build_cmd="$4"
    local restart_csv="$5"

    local repo
    repo="$(expand_path "${repo_raw}")"

    if [[ ! -d "${repo}/.git" ]]; then
        record_failed "${name}" "repo not found or not a git repo: ${repo}"
        log_error "[${name}] repo not found or not a git repo: ${repo}"
        return 1
    fi

    log_info "[${name}] fetching ${repo}"
    local before after
    before="$(git -C "${repo}" rev-parse HEAD 2>/dev/null || echo "")"

    if ! git -C "${repo}" fetch --quiet origin 2>>"${LOG_FILE}"; then
        record_failed "${name}" "git fetch failed"
        log_error "[${name}] git fetch failed"
        return 1
    fi

    # Determine upstream branch (origin/main preferred, fall back to current upstream)
    local upstream_branch
    upstream_branch="$(git -C "${repo}" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || echo "origin/main")"
    if [[ "${upstream_branch}" == "@{u}" || -z "${upstream_branch}" ]]; then
        upstream_branch="origin/main"
    fi

    local remote_head
    remote_head="$(git -C "${repo}" rev-parse "${upstream_branch}" 2>/dev/null || echo "")"
    if [[ -z "${remote_head}" ]]; then
        # Try origin/main explicitly as a fallback
        remote_head="$(git -C "${repo}" rev-parse origin/main 2>/dev/null || echo "")"
        upstream_branch="origin/main"
    fi

    if [[ -z "${remote_head}" ]]; then
        record_failed "${name}" "could not resolve remote HEAD"
        log_error "[${name}] could not resolve remote HEAD (${upstream_branch})"
        return 1
    fi

    if [[ "${before}" == "${remote_head}" ]]; then
        record_skipped "${name}"
        log_info "[${name}] already up to date (${before:0:8})"
        return 0
    fi

    if [[ "${CHECK_ONLY}" -eq 1 ]]; then
        record_updated "${name}"
        log_info "[${name}] would update ${before:0:8} -> ${remote_head:0:8} (check-only)"
        return 0
    fi

    log_info "[${name}] updating ${before:0:8} -> ${remote_head:0:8}"

    # Fast-forward / pull
    if ! git -C "${repo}" pull --ff-only --quiet 2>>"${LOG_FILE}"; then
        # Fall back to resetting to upstream if ff-only fails (e.g. local commits)
        if ! git -C "${repo}" reset --hard "${upstream_branch}" 2>>"${LOG_FILE}"; then
            record_failed "${name}" "git pull/reset failed"
            log_error "[${name}] git pull/reset failed"
            return 1
        fi
        log_warn "[${name}] used reset --hard (local commits discarded)"
    fi

    after="$(git -C "${repo}" rev-parse HEAD 2>/dev/null || echo "")"

    # Rebuild if requested
    if [[ "${rebuild}" == "true" && -n "${build_cmd}" ]]; then
        # Substitute a reasonable -j value if the build_cmd references $(nproc)
        local cmd="${build_cmd//\$\(nproc\)/$(nproc)}"
        log_info "[${name}] rebuilding: ${cmd}"
        # build_cmd is relative to the repo root; run in a subshell with cwd=repo
        if ! ( cd "${repo}" && bash -c "${cmd}" ) 2>>"${LOG_FILE}"; then
            record_failed "${name}" "rebuild failed"
            log_error "[${name}] rebuild failed"
            return 1
        fi
        log_info "[${name}] rebuild complete"
    fi

    record_updated "${name}"
    log_info "[${name}] updated ${before:0:8} -> ${after:0:8}"

    # Restart affected services
    if [[ -n "${restart_csv}" && "${restart_csv}" != "[]" ]]; then
        local IFS=','
        for svc in ${restart_csv}; do
            svc="${svc#[}"; svc="${svc%]}"
            svc="$(echo "${svc}" | xargs)"  # trim whitespace
            [[ -z "${svc}" ]] && continue
            maybe_restart_service "${svc}" && record_restarted "${svc}"
        done
    fi

    return 0
}

# update_pip_component <name> <package> <restart_csv>
# Handles uv tool, pipx, and plain pip/uv venv installs of Hermes Agent.
update_pip_component() {
    local name="$1"
    local package="$2"
    local restart_csv="$3"

    if [[ -z "${package}" ]]; then
        record_failed "${name}" "no package specified"
        return 1
    fi

    # Determine current version (best-effort)
    local before=""
    before="$("${PY}" -c "import importlib.metadata as m; print(m.version('${package}'))" 2>/dev/null || echo "")"

    if [[ "${CHECK_ONLY}" -eq 1 ]]; then
        record_updated "${name}"
        log_info "[${name}] would upgrade ${package} (check-only; current=${before:-unknown})"
        return 0
    fi

    log_info "[${name}] upgrading ${package} (current=${before:-unknown})"

    # Prefer the package's own update command for hermes-agent (it knows the
    # install method: uv tool / pipx / uv pip / system pip).
    if [[ "${package}" == "hermes-agent" ]]; then
        local hermes_bin
        hermes_bin="$(command -v hermes 2>/dev/null || echo "${HOME_DIR}/.local/bin/hermes")"
        if [[ -x "${hermes_bin}" ]]; then
            log_info "[${name}] running: hermes update"
            # hermes update may exit the shell on some code paths; capture.
            if "${hermes_bin}" update --gateway 2>>"${LOG_FILE}"; then
                local after=""
                after="$("${PY}" -c "import importlib.metadata as m; print(m.version('${package}'))" 2>/dev/null || echo "")"
                if [[ -n "${after}" && "${before}" != "${after}" ]]; then
                    record_updated "${name}"
                    log_info "[${name}] ${package} ${before:-?} -> ${after}"
                elif [[ -z "${before}" && -n "${after}" ]]; then
                    record_updated "${name}"
                    log_info "[${name}] ${package} installed -> ${after}"
                else
                    record_skipped "${name}"
                    log_info "[${name}] ${package} already at latest (${after:-unknown})"
                fi
            else
                record_failed "${name}" "hermes update exited non-zero"
                log_error "[${name}] hermes update failed"
                return 1
            fi
            # Hermes manages its own service restarts; honor restart_csv only if set.
            if [[ -n "${restart_csv}" && "${restart_csv}" != "[]" ]]; then
                local IFS=','
                for svc in ${restart_csv}; do
                    svc="$(echo "${svc}" | xargs)"
                    [[ -z "${svc}" ]] && continue
                    maybe_restart_service "${svc}" && record_restarted "${svc}"
                done
            fi
            return 0
        fi
    fi

    # Generic pip/uv path
    local uv_bin pipx_bin
    uv_bin="$(command -v uv 2>/dev/null || echo "")"
    pipx_bin="$(command -v pipx 2>/dev/null || echo "")"

    local cmd
    # pipx-managed?
    if [[ -n "${pipx_bin}" && -d "${HOME_DIR}/.local/share/pipx/venvs/${package}" ]]; then
        cmd=("${pipx_bin}" "upgrade" "${package}")
    elif [[ -n "${uv_bin}" ]]; then
        # uv tool install?
        if "${uv_bin}" tool list 2>/dev/null | grep -q "^${package}"; then
            cmd=("${uv_bin}" "tool" "upgrade" "${package}")
        else
            cmd=("${uv_bin}" "pip" "install" "--upgrade" "${package}")
        fi
    else
        cmd=("${PY}" "-m" "pip" "install" "--upgrade" "${package}")
    fi

    log_info "[${name}] running: ${cmd[*]}"
    if "${cmd[@]}" 2>>"${LOG_FILE}"; then
        local after=""
        after="$("${PY}" -c "import importlib.metadata as m; print(m.version('${package}'))" 2>/dev/null || echo "")"
        if [[ -n "${after}" && "${before}" != "${after}" ]]; then
            record_updated "${name}"
            log_info "[${name}] ${package} ${before:-?} -> ${after}"
        elif [[ -z "${before}" && -n "${after}" ]]; then
            record_updated "${name}"
            log_info "[${name}] ${package} installed -> ${after}"
        else
            record_skipped "${name}"
            log_info "[${name}] ${package} already at latest (${after:-unknown})"
        fi
    else
        record_failed "${name}" "pip/uv upgrade failed"
        log_error "[${name}] pip/uv upgrade failed"
        return 1
    fi

    if [[ -n "${restart_csv}" && "${restart_csv}" != "[]" ]]; then
        local IFS=','
        for svc in ${restart_csv}; do
            svc="$(echo "${svc}" | xargs)"
            [[ -z "${svc}" ]] && continue
            maybe_restart_service "${svc}" && record_restarted "${svc}"
        done
    fi
    return 0
}

# update_npm_component <name> <package> <restart_csv>
update_npm_component() {
    local name="$1"
    local package="$2"
    local restart_csv="$3"

    if [[ -z "${package}" ]]; then
        record_failed "${name}" "no package specified"
        return 1
    fi

    local npm_bin
    npm_bin="$(command -v npm 2>/dev/null || echo "")"
    if [[ -z "${npm_bin}" ]]; then
        record_failed "${name}" "npm not found"
        log_warn "[${name}] npm not found on PATH; skipping"
        return 1
    fi

    # Current version (global)
    local before=""
    before="$("${npm_bin}" ls -g --depth=0 2>/dev/null | grep -E "[^ ]+@[^ ]+$" | awk -F@ '{print $NF}' || echo "")"

    if [[ "${CHECK_ONLY}" -eq 1 ]]; then
        record_updated "${name}"
        log_info "[${name}] would npm update -g ${package} (check-only; current=${before:-unknown})"
        return 0
    fi

    log_info "[${name}] npm update -g ${package}"
    if "${npm_bin}" update -g "${package}" 2>>"${LOG_FILE}"; then
        local after=""
        after="$("${npm_bin}" ls -g --depth=0 2>/dev/null | grep "${package}@" | awk -F@ '{print $NF}' || echo "")"
        if [[ -n "${after}" && "${before}" != "${after}" ]]; then
            record_updated "${name}"
            log_info "[${name}] ${package} ${before:-?} -> ${after}"
        else
            record_skipped "${name}"
            log_info "[${name}] ${package} already at latest (${after:-unknown})"
        fi
    else
        record_failed "${name}" "npm update failed"
        log_error "[${name}] npm update failed"
        return 1
    fi

    if [[ -n "${restart_csv}" && "${restart_csv}" != "[]" ]]; then
        local IFS=','
        for svc in ${restart_csv}; do
            svc="$(echo "${svc}" | xargs)"
            [[ -z "${svc}" ]] && continue
            maybe_restart_service "${svc}" && record_restarted "${svc}"
        done
    fi
    return 0
}

# update_cargo_component <name> <package> <restart_csv>
update_cargo_component() {
    local name="$1"
    local package="$2"
    local restart_csv="$3"

    if [[ -z "${package}" ]]; then
        record_failed "${name}" "no package specified"
        return 1
    fi

    local cargo_bin
    cargo_bin="$(command -v cargo 2>/dev/null || echo "")"
    if [[ -z "${cargo_bin}" ]]; then
        record_failed "${name}" "cargo not found"
        log_warn "[${name}] cargo not found on PATH; skipping"
        return 1
    fi

    if [[ "${CHECK_ONLY}" -eq 1 ]]; then
        record_updated "${name}"
        log_info "[${name}] would cargo install --force ${package} (check-only)"
        return 0
    fi

    log_info "[${name}] cargo install --force ${package}"
    if "${cargo_bin}" install --force "${package}" 2>>"${LOG_FILE}"; then
        record_updated "${name}"
        log_info "[${name}] cargo install --force ${package} complete"
    else
        record_failed "${name}" "cargo install failed"
        log_error "[${name}] cargo install failed"
        return 1
    fi

    if [[ -n "${restart_csv}" && "${restart_csv}" != "[]" ]]; then
        local IFS=','
        for svc in ${restart_csv}; do
            svc="$(echo "${svc}" | xargs)"
            [[ -z "${svc}" ]] && continue
            maybe_restart_service "${svc}" && record_restarted "${svc}"
        done
    fi
    return 0
}

# ----------------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------------
run_component() {
    local name="$1"
    # Translate name to env-var-safe key
    local key="${name//-/_}"

    local enabled type repo rebuild build_cmd package restart_csv
    enabled="COMP_${key}_enabled"
    enabled="${!enabled:-true}"
    type="COMP_${key}_type"
    type="${!type:-git}"
    repo="COMP_${key}_repo"
    repo="${!repo:-}"
    rebuild="COMP_${key}_rebuild"
    rebuild="${!rebuild:-false}"
    build_cmd="COMP_${key}_build_cmd"
    build_cmd="${!build_cmd:-}"
    package="COMP_${key}_package"
    package="${!package:-}"
    restart_csv="COMP_${key}_restart_services"
    restart_csv="${!restart_csv:-}"

    if [[ "${enabled}" != "true" ]]; then
        record_skipped "${name}"
        log_info "[${name}] disabled in config"
        return 0
    fi

    case "${type}" in
        git)
            update_git_component "${name}" "${repo}" "${rebuild}" "${build_cmd}" "${restart_csv}"
            ;;
        pip)
            update_pip_component "${name}" "${package}" "${restart_csv}"
            ;;
        npm)
            update_npm_component "${name}" "${package}" "${restart_csv}"
            ;;
        cargo)
            update_cargo_component "${name}" "${package}" "${restart_csv}"
            ;;
        *)
            record_failed "${name}" "unknown type: ${type}"
            log_error "[${name}] unknown type: ${type}"
            return 1
            ;;
    esac
}

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
emit_summary() {
    local end_epoch dur
    end_epoch=$(date +%s)
    dur=$((end_epoch - START_EPOCH))

    local n_updated=${#UPDATED_NAMES[@]}
    local n_skipped=${#SKIPPED_NAMES[@]}
    local n_failed=${#FAILED_NAMES[@]}
    local n_restarted=${#RESTARTED_SERVICES[@]}

    local status_emoji=":white_check_mark:"
    if [[ "${n_failed}" -gt 0 ]]; then
        status_emoji=":warning:"
    fi
    if [[ "${n_updated}" -eq 0 && "${n_failed}" -eq 0 ]]; then
        status_emoji=":zzz:"
    fi

    {
        echo "=========================================="
        echo "turbofit auto-update ${status_emoji}"
        echo "started:  ${START_ISO}"
        echo "duration: ${dur}s"
        echo "updated:  ${n_updated}   skipped: ${n_skipped}   failed: ${n_failed}   restarted: ${n_restarted}"
        if [[ "${n_updated}" -gt 0 ]]; then
            echo ""
            echo "Updated:"
            for n in "${UPDATED_NAMES[@]}"; do echo "  + ${n}"; done
        fi
        if [[ "${n_skipped}" -gt 0 ]]; then
            echo ""
            echo "Skipped (up to date / disabled):"
            for n in "${SKIPPED_NAMES[@]}"; do echo "  = ${n}"; done
        fi
        if [[ "${n_failed}" -gt 0 ]]; then
            echo ""
            echo "Failed:"
            for i in "${!FAILED_NAMES[@]}"; do
                echo "  ! ${FAILED_NAMES[$i]}: ${FAILED_ERRORS[$i]}"
            done
        fi
        if [[ "${n_restarted}" -gt 0 ]]; then
            echo ""
            echo "Services restarted:"
            for s in "${RESTARTED_SERVICES[@]}"; do echo "  ~ ${s}"; done
        fi
        echo "=========================================="
    } | tee -a "${LOG_FILE}" >"${SUMMARY_FILE}"

    # Print just the summary block to stdout for Discord/notifications.
    cat "${SUMMARY_FILE}"
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check)        CHECK_ONLY=1; shift ;;
            --summary)      EMIT_SUMMARY=1; shift ;;
            --component)    ONLY_COMPONENT="$2"; shift 2 ;;
            --component=*)  ONLY_COMPONENT="${1#*=}"; shift ;;
            -h|--help)
                cat <<'USAGE'
turbofit auto-update.sh

Usage:
  auto-update.sh                 Run all enabled components
  auto-update.sh --check         Report what would change, apply nothing
  auto-update.sh --component X   Run only component X
  auto-update.sh --summary       Emit a Discord-ready summary block
  auto-update.sh --help          Show this help

Config: ~/.config/turbofit/auto-update.yaml
Log:    ~/.local/share/turbofit/logs/auto-update.log
USAGE
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                exit 2
                ;;
        esac
    done

    log_info "==== auto-update run started (${START_ISO}) ===="
    if [[ "${CHECK_ONLY}" -eq 1 ]]; then
        log_info "Running in CHECK-ONLY mode (no changes will be applied)"
    fi

    load_config

    # Read COMPONENTS as an array (it was emitted space-separated by the python helper)
    local comp_names=()
    if [[ -n "${COMPONENTS:-}" ]]; then
        # COMPONENTS was shlex-quoted by the helper; eval it into an array.
        eval "comp_names=(${COMPONENTS})"
    fi

    if [[ ${#comp_names[@]} -eq 0 ]]; then
        log_warn "No components to process."
        emit_summary
        exit 0
    fi

    for name in "${comp_names[@]}"; do
        if [[ -n "${ONLY_COMPONENT}" && "${name}" != "${ONLY_COMPONENT}" ]]; then
            continue
        fi
        # Each component runs in its own subshell-ish scope; never let one
        # failure abort the loop. run_component always returns 0 or 1 but
        # records its own outcome.
        run_component "${name}" || true
    done

    log_info "==== auto-update run finished ===="

    emit_summary
}

main "$@"
