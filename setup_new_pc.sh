#!/usr/bin/env bash
# One-shot environment setup for this project on a fresh machine.
#
# Follows doc/SIM_GUIDE.md section 1-2 exactly (system requirements, CoppeliaSim, the .venv and
# its pip list, and pointing CoppeliaSim at that venv) plus the extra packages this repo's
# scripts actually import that SIM_GUIDE's own list omits (mujoco, matplotlib) -- confirmed
# against `.venv/bin/pip freeze` on the machine this project was developed on.
#
# What this script does NOT and CANNOT provide, because they are not installable packages:
#   - collected data (`data/`) and trained checkpoints (`wm/runs/`) -- both gitignored, must be
#     copied from wherever they are backed up
#   - the vendored/reference subprojects (`airl-insect-walking/`, `amp/`, `Egocentric_VSM/`,
#     `Demo-JEPA/`) -- also gitignored, not part of the tracked wm/sim/scripts pipeline
#   - a working NVIDIA driver -- this only checks for one and tells you what to do if it's missing;
#     installing/upgrading a GPU driver can require a reboot and is not something to automate
#     blindly on someone else's machine
#
# Safe to re-run: every step checks whether its target already exists before doing anything.
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SIM="${SIM:-$HOME/CoppeliaSim}"
COPPELIASIM_VERSION="4.10.0"                # matches doc/SIM_GUIDE.md section 1 -- bump both if it changes
COPPELIASIM_REV="rev0"
# As of this release CoppeliaSim ships one tarball per supported Ubuntu version (no more generic
# "_Linux.tar.xz" -- confirmed 404 on that old naming scheme against downloads.coppeliarobotics.com
# on 2026-09-08). Only 22.04 and 24.04 builds are published; anything else falls back to 24.04.
UBUNTU_CODENAME="24_04"
if command -v lsb_release >/dev/null 2>&1; then
    case "$(lsb_release -rs 2>/dev/null)" in
        22.04) UBUNTU_CODENAME="22_04" ;;
        24.04) UBUNTU_CODENAME="24_04" ;;
    esac
fi
COPPELIASIM_TARBALL="CoppeliaSim_Edu_V4_10_0_${COPPELIASIM_REV}_Ubuntu${UBUNTU_CODENAME}.tar.xz"
COPPELIASIM_URL="https://downloads.coppeliarobotics.com/V4_10_0_${COPPELIASIM_REV}/${COPPELIASIM_TARBALL}"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$1" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$1" >&2; exit 1; }

[ -f "$REPO/README.md" ] && [ -d "$REPO/wm" ] || die "run this from inside the ioon-research checkout (expected wm/ and README.md next to this script)"

log "Repository: $REPO"
log "CoppeliaSim target directory: $SIM"

# --------------------------------------------------------------------------------------------
# 1. System packages
# --------------------------------------------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
    SUDO=""
    [ "$(id -u)" -ne 0 ] && SUDO="sudo"

    log "Installing system packages (Python toolchain + CoppeliaSim's runtime libraries)"
    # Python 3.10 per doc/SIM_GUIDE.md section 1. The library list below is the standard set
    # CoppeliaSim's Qt5/X11 stack needs at runtime on a minimal (non-desktop) Ubuntu install --
    # CoppeliaSim ships its own libQt5*.so and codec libs, but still dynamically links these
    # system X11/GL libraries.
    $SUDO apt-get update
    $SUDO apt-get install -y \
        python3.10 python3.10-venv python3-pip \
        build-essential git wget \
        libgl1 libglib2.0-0 \
        libxcb-glx0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
        libxcb-render-util0 libxcb-render0 libxcb-shape0 libxcb-shm0 libxcb-sync1 \
        libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 \
        libdbus-1-3 libnss3 libxcomposite1 libxcursor1 libxdamage1 libxi6 libxtst6 \
        libfontconfig1 libxrender1 libxrandr2
else
    warn "apt-get not found -- skipping system package install. Install Python 3.10 (+venv), " \
         "build-essential, git, wget, and CoppeliaSim's Qt5/X11 runtime libraries yourself " \
         "(see doc/SIM_GUIDE.md section 1) before continuing."
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver present: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
else
    warn "No working NVIDIA driver detected (nvidia-smi missing or failing). This project needs a " \
         "CUDA-capable GPU -- 11 GB was used in development, 8 GB is the practical minimum (see " \
         "doc/SIM_GUIDE.md section 1). Install the proprietary NVIDIA driver for your distro, " \
         "reboot, and re-run this script -- driver installation is not automated here since it can " \
         "require a reboot and touches the display server."
fi

# --------------------------------------------------------------------------------------------
# 2. Python environment (doc/SIM_GUIDE.md section 2.2)
# --------------------------------------------------------------------------------------------
cd "$REPO"
if [ -d .venv ]; then
    log ".venv already exists, skipping creation"
else
    log "Creating .venv (python3.10 -m venv --system-site-packages)"
    python3.10 -m venv .venv --system-site-packages
fi

log "Installing Python packages into .venv"
.venv/bin/pip install --upgrade pip
# The exact SIM_GUIDE.md list, plus mujoco (sim/collect/rollout_b1_mujoco.py) and matplotlib
# (every figures/ script) -- both are actually imported by this repo but missing from the
# guide's own install line. Reference versions known to work: torch 2.13.0+cu130,
# transformers 5.13.1, scikit-learn 1.7.2, numpy 2.2.6, mujoco 3.9.0 -- unpinned here since a
# pin from one machine's CUDA/driver combination is not guaranteed to resolve on another.
.venv/bin/pip install \
    torch transformers coppeliasim_zmqremoteapi_client pyzmq msgpack cbor2 \
    pandas numpy opencv-python-headless imageio imageio-ffmpeg scikit-learn umap-learn \
    tensorboard mujoco matplotlib

log "Verifying torch / CUDA / transformers"
.venv/bin/python3 -c "
import torch, transformers
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('transformers', transformers.__version__)
"

# --------------------------------------------------------------------------------------------
# 3. CoppeliaSim (doc/SIM_GUIDE.md section 2.1) -- best effort; this is a third-party binary
#    behind a versioned download URL this script cannot verify without network access at
#    write-time, so it falls back to manual instructions rather than failing the whole script.
# --------------------------------------------------------------------------------------------
if [ -x "$SIM/coppeliaSim.sh" ]; then
    log "CoppeliaSim already installed at $SIM, skipping download"
else
    log "Downloading CoppeliaSim $COPPELIASIM_VERSION Edu"
    TMP_TARBALL="$(mktemp -d)/$COPPELIASIM_TARBALL"
    if wget -q --show-progress -O "$TMP_TARBALL" "$COPPELIASIM_URL"; then
        mkdir -p "$SIM"
        tar -xf "$TMP_TARBALL" --strip-components=1 -C "$SIM"
        rm -f "$TMP_TARBALL"
        log "CoppeliaSim extracted to $SIM"
    else
        warn "Automatic download failed (URL guessed from the known naming scheme -- " \
             "$COPPELIASIM_URL -- point releases change this). Download CoppeliaSim " \
             "$COPPELIASIM_VERSION Edu for Linux manually from " \
             "https://www.coppeliarobotics.com/previousVersions or downloads.coppeliarobotics.com " \
             "and extract it to: $SIM"
    fi
fi

# --------------------------------------------------------------------------------------------
# 4. Point CoppeliaSim at the venv (doc/SIM_GUIDE.md section 2.3) -- required, otherwise scene
#    scripts run under the system interpreter, fail to import zmq/cbor2, and CoppeliaSim pauses
#    the simulation on script error with no visible failure during training.
# --------------------------------------------------------------------------------------------
USRSET_DIR="$HOME/.CoppeliaSim"
USRSET_FILE="$USRSET_DIR/usrset.txt"
DEFAULT_PYTHON_LINE="defaultPython = $REPO/.venv/bin/python3"

mkdir -p "$USRSET_DIR"
if [ -f "$USRSET_FILE" ] && grep -q '^defaultPython' "$USRSET_FILE"; then
    sed -i "s|^defaultPython.*|$DEFAULT_PYTHON_LINE|" "$USRSET_FILE"
    log "Updated existing defaultPython entry in $USRSET_FILE"
else
    printf '%s\n' "$DEFAULT_PYTHON_LINE" >> "$USRSET_FILE"
    log "Set defaultPython in $USRSET_FILE"
fi

# --------------------------------------------------------------------------------------------
# Done
# --------------------------------------------------------------------------------------------
log "Setup finished. Still needed before anything runs (not installable, must be copied):"
echo "  - data/       (collected clips -- gitignored)"
echo "  - wm/runs/    (trained checkpoints -- gitignored)"
echo "  - sim/env/expert_66k_aug3c_fcontact.csv (132 MB, exceeds GitHub's limit -- gitignored)"
echo
log "Verify the install (doc/SIM_GUIDE.md section 2.4):"
echo "  cd $SIM && ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23000   # separate terminal"
echo '  cd '"$REPO"' && .venv/bin/python3 -c "'
echo "  from coppeliasim_zmqremoteapi_client import RemoteAPIClient"
echo "  sim = RemoteAPIClient('localhost', port=23000).require('sim')"
echo '  print("connected, objects:", len(sim.getObjectsInTree(sim.handle_scene, sim.handle_all)))"'
