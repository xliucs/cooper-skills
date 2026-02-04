"""
Shared plotting style for publication-quality figures.
Blue-dominant palette, clean design, square subplots.

Usage:
    from plot_style import setup_style, BLUE_DARK, BLUE_MED, ACCENT_RED, fprint
    setup_style()
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Blue palette
BLUE_DARK = '#1565C0'
BLUE_MED = '#1E88E5'
BLUE_LIGHT = '#42A5F5'
BLUE_PALE = '#90CAF9'
BLUE_BG = '#E3F2FD'

# Accents (use sparingly)
ACCENT_RED = '#C62828'
ACCENT_ORANGE = '#E65100'
ACCENT_GREEN = '#2E7D32'
ACCENT_GRAY = '#616161'
ACCENT_LIGHT_GRAY = '#BDBDBD'

# Sequential blues for bar charts (dark to light)
BLUES_SEQ = ['#0D47A1', '#1565C0', '#1976D2', '#1E88E5', '#42A5F5', '#64B5F6', '#90CAF9']


def setup_style():
    """Apply consistent style to all figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': 300,
    })


def fprint(*a, **k):
    """Flush-print for visibility in long-running scripts."""
    print(*a, **k, flush=True)
