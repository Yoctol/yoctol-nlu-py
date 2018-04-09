from os.path import abspath, dirname, join

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT_DIR = dirname(dirname(abspath(__file__)))
DEFAULT_FONT_PATH = join(ROOT_DIR, "data/simhei.ttf")


def _get_all_text_objects(obj):
    '''
    Get all text objects in a matplotlib Figure
    Helper for ``plt_set_font_style``
    '''
    queue = [obj]
    all_text = []
    while queue:
        currobj = queue.pop(0)
        if isinstance(currobj, matplotlib.text.Text):
            all_text.append(currobj)
        else:
            queue = queue + currobj.get_children()
    return all_text


def plt_set_font_style(font_style_path: str = None):
    """Setting font style of figure plotting by matplotlib

        Args:
            font_style_path (path of font style):
                If None, ``simhei.ttf`` will be used as default font style.

        Returns: None

    """

    if font_style_path is None:
        font_style_path = DEFAULT_FONT_PATH

    font_style = font_manager.FontProperties(fname=font_style_path)

    fig = plt.gcf()
    for textobj in _get_all_text_objects(fig):
        fontsize = textobj.get_fontsize()
        textobj.set_fontproperties(font_style)
        textobj.set_fontsize(fontsize)
