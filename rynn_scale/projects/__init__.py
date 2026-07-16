import os
import sys
import importlib.util

from ..utils.logging import get_logger


logger = get_logger(__name__)


def register_projects():
    if not os.path.isdir("projects"):
        return

    for entry in os.listdir("projects"):
        if os.path.isdir(os.path.join("projects", entry)):
            init_path = os.path.join("projects", entry, "__init__.py")
            if os.path.exists(init_path):
                try:
                    module_nmae = f"{__name__}.{entry}"
                    spec = importlib.util.spec_from_file_location(module_nmae, str(init_path))
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_nmae] = module
                    spec.loader.exec_module(module)
                    spec = importlib.util.spec_from_file_location(entry, str(init_path))
                except Exception as e:
                    import traceback; traceback.print_exc()
                    logger.warning(f"Failed to import projects/{entry}: {e}")
