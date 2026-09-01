from typing import Union

from ..constants import RobotType
from ..registry import RENDERER_REGISTRY
from .agibot_g1 import AgibotG1Renderer
from .agibot_g2 import AgibotG2Renderer
from .agilex_cobot_magic_1 import AgilexCobotMagic1Renderer
from .agilex_cobot_magic_2 import AgilexCobotMagic2Renderer
from .agilex_piper import AgilexPiperRenderer
from .agilex_split_aloha import AgilexSplitAlohaRenderer
from .arx_lift2 import ArxLift2Renderer
from .arx_x5 import ArxX5Renderer
from .base import BaseRenderer
from .dexmal_dos_w1 import DexmalDosW1Renderer
from .franka import FrankaRenderer
from .galaxea_r1_lite import GalaxeaR1LiteRenderer
from .galaxea_r1_pro import GalaxeaR1ProRenderer
from .tianyi import TianyiRenderer
from .tienkung import TienkungRenderer
from .tienkung_2 import Tienkung2Renderer
from .ur_5 import Ur5Renderer
from .ur_5e import Ur5eRenderer
from .widowx_250s import Widowx250sRenderer


def build_renderer(
    robot_type: Union[RobotType, str],
    height: int = 480,
    width: int = 480,
    **kwargs,
) -> BaseRenderer:
    if isinstance(robot_type, RobotType):
        robot_type = robot_type.value
    return RENDERER_REGISTRY[robot_type](height=height, width=width, **kwargs)
