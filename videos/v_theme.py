"""
Shared theme for the thesis presentation videos.

COLOUR CONTRACT
---------------
Every colour below was checked with the dataviz palette validator
(OKLab dE, Machado-Oliveira-Fernandes CVD simulation at severity 1.0),
in dark mode against the surface #121316. Do not substitute by eye.

  categorical, all-pairs validated ..... BLUE / AMBER / PINK
      BLUE  <-> AMBER   dE 25.1 (CVD)  29.2 (normal vision)
      worst all-pairs   dE 11.6 (CVD)  21.6 (normal vision)
  sequential, single hue, monotone L ... RAMP[0..2]
  status, reserved, never a series ..... OK / BAD

Rules that keep it readable, and why:
  * Only TWO colours ever encode competing quantities: BLUE = learned,
    AMBER = classical. They are the most separated pair in the set.
  * Ordered data (a sweep) uses RAMP, not three categorical hues --
    ordering is carried by lightness, which survives any colour vision.
  * OK/BAD never label a data series. They mark a verdict, and always
    travel with a word or a glyph so colour is never the only cue.
  * Nothing relies on hue alone: every series is directly labelled.
"""

from manim import *

# ---------------------------------------------------------------- surface
BG      = "#121316"   # neutral very dark grey, deliberately no blue cast
PANEL   = "#1B1E24"   # raised panel
INK     = "#EEF1F5"   # primary text      16.4:1
MUTED   = "#9AA3B0"   # secondary text     7.3:1
GRID    = "#3A4048"   # axes and rules     1.8:1  (recessive by design)

# ------------------------------------------------------- categorical set
BLUE    = "#009CE0"   # the learned controller / DRL
AMBER   = "#CC7C00"   # classical static optimisation
PINK    = "#DE4E9E"   # third slot, used only where unavoidable

# ---------------------------------------------- sequential (ordered data)
RAMP    = ["#0084C4", "#00A8F8", "#84CCFC"]

# ----------------------------------------------------- status (reserved)
OK      = "#00B048"
BAD     = "#FC403C"

FONT    = "Segoe UI"

config.background_color = BG

# Frame is 14.222 x 8 units.  Keep everything inside this box so nothing
# is lost to projector overscan or a cropped slide.
SAFE_L, SAFE_R = -6.55, 6.55
SAFE_T, SAFE_B = 3.55, -3.55

TITLE_Y = 3.15
RULE_Y = 2.80
BODY_TOP = 2.45
STRIP_Y = -3.05


def txt(s, size=30, color=INK, weight=NORMAL, font=FONT):
    return Text(s, font=font, font_size=size, color=color, weight=weight)


class Slide(Scene):
    """Base slide: a title, a rule beneath it, and an optional takeaway strip.

    Every scene in this set uses the same three anchors so that a viewer's
    eye does not have to re-find the layout on each cut.
    """

    def header(self, title, sub=None):
        t = txt(title, 40, INK, BOLD).move_to([0, TITLE_Y, 0])
        rule = Line([SAFE_L, RULE_Y, 0], [SAFE_R, RULE_Y, 0], color=GRID, stroke_width=2)
        accent = Line([SAFE_L, RULE_Y, 0], [SAFE_L + 2.4, RULE_Y, 0], color=BLUE, stroke_width=4)
        g = VGroup(t, rule, accent)
        if sub:
            s = txt(sub, 24, MUTED).next_to(t, DOWN, buff=0.20)
            g.add(s)
            rule.shift(DOWN * 0.58)
            accent.shift(DOWN * 0.58)
        self.play(FadeIn(t, shift=DOWN * 0.2), Create(rule), run_time=0.7)
        self.play(Create(accent), run_time=0.4)
        if sub:
            self.play(FadeIn(g[3]), run_time=0.4)
        return g

    def takeaway(self, message, color=BLUE, size=27):
        """The one sentence the viewer should leave with."""
        body = txt(message, size, INK, BOLD)
        # never let the sentence outgrow its own box: shrink to fit rather
        # than bleed past the panel edge on a projector
        if body.width > 12.0:
            body.scale(12.0 / body.width)
        box = RoundedRectangle(
            width=min(body.width + 1.1, 13.1), height=body.height + 0.62,
            corner_radius=0.12, fill_color=PANEL, fill_opacity=1,
            stroke_color=GRID, stroke_width=1.5,
        )
        bar = RoundedRectangle(
            width=0.11, height=box.height - 0.22, corner_radius=0.05,
            fill_color=color, fill_opacity=1, stroke_width=0,
        ).align_to(box, LEFT).shift(RIGHT * 0.16)
        g = VGroup(box, bar, body).move_to([0, STRIP_Y, 0])
        body.move_to(box.get_center()).shift(RIGHT * 0.14)
        self.play(FadeIn(box, shift=UP * 0.15), Create(bar), run_time=0.5)
        self.play(Write(body), run_time=0.9)
        return g

    def bullet(self, s, color=BLUE, size=27):
        dot = RoundedRectangle(width=0.14, height=0.14, corner_radius=0.03,
                               fill_color=color, fill_opacity=1, stroke_width=0)
        t = txt(s, size, INK)
        return VGroup(dot, t).arrange(RIGHT, buff=0.28)

    def legend(self, entries, size=24):
        """entries = [(colour, label), ...]. Identity is never colour alone --
        the swatch always sits beside its written name."""
        items = []
        for c, label in entries:
            sw = RoundedRectangle(width=0.30, height=0.16, corner_radius=0.04,
                                  fill_color=c, fill_opacity=1, stroke_width=0)
            items.append(VGroup(sw, txt(label, size, MUTED)).arrange(RIGHT, buff=0.16))
        return VGroup(*items).arrange(RIGHT, buff=0.65)


def at(m, x, y, align=LEFT):
    """Place a mobject by an absolute anchor rather than by arrange().

    Columns in a table must line up across rows, and arrange() cannot do
    that -- it packs each row independently. Every table in this set is
    built by anchoring each cell to a fixed x.
    """
    m.move_to([x, y, 0], aligned_edge=align)
    return m


def bar(width, height, color, horizontal=False):
    """A data bar with rounded ends, anchored flat to its baseline.

    corner_radius stays small so the rounding reads as a finish, never as a
    change in the value the bar encodes.
    """
    w, h = (width, height) if not horizontal else (width, height)
    r = min(0.07, abs(w) / 2 - 0.001, abs(h) / 2 - 0.001) if min(abs(w), abs(h)) > 0.02 else 0.0
    return RoundedRectangle(width=max(abs(w), 0.02), height=max(abs(h), 0.02),
                            corner_radius=max(r, 0.0),
                            fill_color=color, fill_opacity=1, stroke_width=0)


def hbar_chart(rows, x0, y0, unit, row_h=0.52, gap=0.18, label_w=3.0,
               value_fmt="{:.2f}", size=23):
    """Horizontal bars, one per row.  rows = [(label, value, colour), ...]

    Returns (group, bars, labels, values).  Bars grow from a common left
    baseline at x0; a 2px-equivalent gap separates neighbours so adjacent
    fills never touch.
    """
    bars, labels, values = VGroup(), VGroup(), VGroup()
    for i, (name, val, col) in enumerate(rows):
        y = y0 - i * (row_h + gap)
        lab = txt(name, size, INK).move_to([x0 - 0.28, y, 0]).align_to([x0 - 0.28, 0, 0], RIGHT)
        w = max(val * unit, 0.02)
        b = bar(w, row_h, col)
        b.move_to([x0 + w / 2, y, 0])
        v = txt(value_fmt.format(val), size, MUTED).next_to(b, RIGHT, buff=0.22)
        bars.add(b); labels.add(lab); values.add(v)
    return VGroup(labels, bars, values), bars, labels, values


def axis_x(x0, x1, y, ticks=None, size=20):
    ax = Line([x0, y, 0], [x1, y, 0], color=GRID, stroke_width=2)
    g = VGroup(ax)
    for xv, lab in (ticks or []):
        t = Line([xv, y, 0], [xv, y - 0.12, 0], color=GRID, stroke_width=2)
        g.add(t, txt(lab, size, MUTED).next_to(t, DOWN, buff=0.10))
    return g
