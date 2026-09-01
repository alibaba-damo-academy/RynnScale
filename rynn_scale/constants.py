import os
from enum import Enum

# Backend selection knobs, controlled via environment variables.

# MoE token dispatch backend: "all_to_all" | "deep_ep"
MOE_DISPATCH_BACKEND = os.environ.get("MOE_DISPATCH_BACKEND", "all_to_all")

# MoE grouped GEMM backend: "cutlass" | "triton" | "torch"
MOE_GEMM_BACKEND = os.environ.get("MOE_GEMM_BACKEND", "cutlass")

# MoE token permute/unpermute backend: "triton" | "torch"
MOE_PERMUTE_BACKEND = os.environ.get("MOE_PERMUTE_BACKEND", "torch")

# RoPE backend: "triton" | "torch"
ROPE_BACKEND = os.environ.get("ROPE_BACKEND", "torch")

# RMSNorm backend: "triton" | "torch"
NORM_BACKEND = os.environ.get("NORM_BACKEND", "torch")

# Cross-entropy loss backend: "torch" | "cce"
CROSS_ENTROPY_BACKEND = os.environ.get("CROSS_ENTROPY_BACKEND", "torch")

# Video decode backend: "ffmpeg" | "pyav" | "torchcodec"
# Defaults to "pyav": the dominant consumer is the VLA episode path, where the
# ffmpeg backend spawns one process per frame and rescans from frame 0.
# Note that SequentialVideoReader only implements "pyav" and "torchcodec".
VIDEO_DECODE_BACKEND = os.environ.get("VIDEO_DECODE_BACKEND", "pyav")

CACHE_DIR = os.environ.get(
    "RYNNSCALE_CACHE_DIR",
    os.path.join(
        os.environ.get(
            "XDG_CACHE_HOME",
            os.path.join(os.path.expanduser("~"), ".cache"),
        ),
        "rynn_scale",
    ),
)


class RobotType(Enum):
    FRANKA = "franka"
    DUAL_FRANKA = "dual_franka"
    UR_5 = "ur_5"
    UR_5E = "ur_5e"
    DUAL_UR_5 = "dual_ur_5"
    DUAL_UR_5E = "dual_ur_5e"
    DUAL_UR_5E_DEX = "dual_ur_5e_dex"
    AGILEX_COBOT_MAGIC_1 = "agilex_cobot_magic_1"
    AGILEX_COBOT_MAGIC_2 = "agilex_cobot_magic_2"
    AGILEX_PIPER = "agilex_piper"
    DUAL_AGILEX_PIPER = "dual_agilex_piper"
    ARX_X5 = "arx_x5"
    DUAL_ARX_X5 = "dual_arx_x5"
    ARX_LIFT2 = "arx_lift2"
    AGIBOT_G2 = "agibot_g2"
    AGIBOT_G1 = "agibot_g1"
    AGILEX_SPLIT_ALOHA = "agilex_split_aloha"
    GALAXEA_R1_LITE = "galaxea_r1_lite"
    TIANYI = "tianyi"
    TIENKUNG_1 = "tienkung_1"
    TIENKUNG_2 = "tienkung_2"
    MARVIN_WUJI = "marvin_wuji"
    ASTRIBOT = "astribot"
    WIDOWX_250S = "widowx_250s"
    FRANKA_OMRON = "franka_omron"
    GALAXEA_R1_PRO = "galaxea_r1_pro"
    GALBOT_G1 = "galbot_g1"
    REALMAN_RMC_AIDA = "realman_rmc_aida"
    UNITREE_G1 = "unitree_g1"
    DEXMAL_DOS_W1 = "dexmal_dos_w1"


class RotationRepresentation(Enum):
    EULER_XYZ = ("euler_xyz", 3)
    EULER_ZYX = ("euler_zyx", 3)
    QUAT_XYZW = ("quat_xyzw", 4)
    QUAT_WXYZ = ("quat_wxyz", 4)
    ROT_6D = ("rot_6d", 6)
    ROT_VEC = ("rot_vec", 3)

    def __new__(cls, label, dim):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.dim = dim
        return obj
