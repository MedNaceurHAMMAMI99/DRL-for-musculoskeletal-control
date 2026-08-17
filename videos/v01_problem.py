"""Part 1 -- The problem.  Scenes 01 to 05."""

from manim import *
from v_theme import *


class S01_TheQuestion(Slide):
    """Opening card.  States the thesis question and nothing else."""

    def construct(self):
        title = txt("Learning Motor Control Strategies", 46, INK, BOLD)
        title2 = txt("in Musculoskeletal Systems", 46, INK, BOLD)
        sub = txt("using Deep Reinforcement Learning", 32, BLUE)
        VGroup(title, title2, sub).arrange(DOWN, buff=0.22).move_to([0, 1.75, 0])

        rule = Line([-3.2, 0.55, 0], [3.2, 0.55, 0], color=GRID, stroke_width=2)

        who = txt("Mohamed Naceur Hammami", 26, INK)
        where = txt("Tunisia Polytechnic School", 22, MUTED)
        VGroup(who, where).arrange(DOWN, buff=0.16).move_to([0, 0.05, 0])

        self.play(FadeIn(title, shift=UP * 0.3), run_time=0.8)
        self.play(FadeIn(title2, shift=UP * 0.3), run_time=0.6)
        self.play(Write(sub), run_time=0.8)
        self.play(Create(rule), run_time=0.5)
        self.play(FadeIn(who), FadeIn(where), run_time=0.7)
        self.wait(1.2)

        q1 = txt("For a reaching movement of a nine-muscle arm,", 29, INK)
        q2 = txt("does a controller trained by reinforcement learning", 29, INK)
        q3 = txt("produce the same per-muscle forces", 29, BLUE, BOLD)
        q4 = txt("as classical static optimisation?", 29, AMBER, BOLD)
        qs = VGroup(q1, q2, q3, q4).arrange(DOWN, buff=0.20)

        box = RoundedRectangle(width=qs.width + 1.4, height=qs.height + 1.0,
                               corner_radius=0.16, fill_color=PANEL, fill_opacity=1,
                               stroke_color=GRID, stroke_width=2)
        card = VGroup(box, qs).move_to([0, -1.45, 0])
        qs.move_to(box.get_center())

        self.play(FadeIn(box, shift=UP * 0.2), run_time=0.6)
        for m in qs:
            self.play(FadeIn(m, shift=RIGHT * 0.2), run_time=0.55)
        self.wait(2.5)


class S02_Redundancy(Slide):
    """Why the movement cannot tell you the forces."""

    def construct(self):
        self.header("The counting problem")

        left = txt("9 muscles", 40, BLUE, BOLD).move_to([-3.6, 1.75, 0])
        arrow = Arrow([-2.0, 1.75, 0], [1.1, 1.75, 0], color=MUTED,
                      stroke_width=5, max_tip_length_to_length_ratio=0.12)
        right = txt("4 ways to move", 40, AMBER, BOLD).move_to([3.3, 1.75, 0])
        self.play(FadeIn(left), run_time=0.5)
        self.play(GrowArrow(arrow), run_time=0.5)
        self.play(FadeIn(right), run_time=0.5)
        self.wait(0.8)

        note = txt("Nine controls.  Four things to control.", 28, MUTED).move_to([0, 0.95, 0])
        self.play(FadeIn(note), run_time=0.6)
        self.wait(1.4)

        # ---- the door and nine ropes -------------------------------------
        self.play(FadeOut(VGroup(left, arrow, right, note)), run_time=0.5)

        door = RoundedRectangle(width=1.05, height=3.0, corner_radius=0.06,
                                fill_color=PANEL, fill_opacity=1,
                                stroke_color=MUTED, stroke_width=3).move_to([3.5, -0.15, 0])
        hinge = Line(door.get_corner(UL), door.get_corner(DL), color=AMBER, stroke_width=6)
        dlab = txt("the door", 22, MUTED).next_to(door, DOWN, buff=0.25)
        self.play(Create(door), Create(hinge), FadeIn(dlab), run_time=0.8)

        ropes, pullers = VGroup(), VGroup()
        for i in range(9):
            y = 1.55 - i * 0.42
            r = Line([-3.9, y, 0], [2.95, -0.15, 0], color=BLUE, stroke_width=2.5)
            p = Dot([-3.9, y, 0], radius=0.10, color=BLUE)
            ropes.add(r); pullers.add(p)
        plab = txt("nine people, nine ropes", 22, MUTED).move_to([-3.55, -2.35, 0])

        self.play(LaggedStart(*[Create(r) for r in ropes], lag_ratio=0.07), run_time=1.3)
        self.play(LaggedStart(*[FadeIn(p) for p in pullers], lag_ratio=0.05), run_time=0.6)
        self.play(FadeIn(plab), run_time=0.4)
        self.wait(1.0)

        # three different pulling patterns, one identical door angle
        cap = txt("", 26, INK).move_to([-1.1, -2.9, 0])
        patterns = [
            ("all nine pull gently", [0.35] * 9),
            ("three pull hard, six rest", [1.0, 1.0, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
            ("four pull against five", [1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9]),
        ]
        for label, strengths in patterns:
            new_cap = txt(label, 26, INK).move_to([-1.1, -2.85, 0])
            anims = []
            for r, p, s in zip(ropes, pullers, strengths):
                anims.append(r.animate.set_stroke(width=1.2 + 5.0 * s,
                                                  opacity=0.35 + 0.65 * s))
                anims.append(p.animate.scale_to_fit_width(0.12 + 0.22 * s))
            self.play(Transform(cap, new_cap), *anims, run_time=0.9)
            self.wait(1.3)

        self.play(Indicate(door, color=AMBER, scale_factor=1.06), run_time=1.0)
        self.wait(0.6)
        self.play(FadeOut(cap), FadeOut(plab), FadeOut(dlab), run_time=0.4)

        self.takeaway("Watch only the door and all three look identical.", AMBER)
        self.wait(2.2)


class S03_Redundancy2(Slide):
    """The consequence: an extra rule is needed, and which rule is a choice."""

    def construct(self):
        self.header("Infinitely many answers", "so an extra rule must pick one")

        eq = MathTex(r"R^{\top} f = \tau", font_size=64, color=INK).move_to([0, 1.75, 0])
        self.play(Write(eq), run_time=1.0)

        under1 = txt("9 unknowns", 26, BLUE).next_to(eq, DOWN, buff=0.45).shift(LEFT * 1.5)
        under2 = txt("4 equations", 26, AMBER).next_to(eq, DOWN, buff=0.45).shift(RIGHT * 1.5)
        self.play(FadeIn(under1), FadeIn(under2), run_time=0.7)
        self.wait(1.0)

        line = txt("Underdetermined: a whole family of solutions, not one.",
                   28, MUTED).move_to([0, 0.35, 0])
        self.play(FadeIn(line), run_time=0.7)
        self.wait(1.5)

        rule = txt("Biomechanics picks one by minimum effort", 30, AMBER, BOLD)
        cite = txt("Crowninshield & Brand, 1981", 23, MUTED)
        VGroup(rule, cite).arrange(DOWN, buff=0.20).move_to([0, -0.85, 0])
        self.play(FadeIn(rule, shift=UP * 0.2), run_time=0.7)
        self.play(FadeIn(cite), run_time=0.5)
        self.wait(1.6)

        warn = txt("That rule is a theory of how the body shares load,\nnot a measurement of it.",
                   26, INK).move_to([0, -2.1, 0])
        self.play(FadeIn(warn), run_time=0.8)
        self.wait(2.4)


class S04_StaticOptimisation(Slide):
    """The classical method, assembled one piece at a time."""

    def construct(self):
        self.header("The classical answer", "static optimisation, read out term by term")

        obj = MathTex(r"\min_{f}\ \sum_{i=1}^{9}\left(\frac{f_i}{F_{\max,i}}\right)^{2}",
                      font_size=52, color=AMBER).move_to([0, 1.55, 0])
        self.play(Write(obj), run_time=1.4)
        g1 = txt("Make total effort as small as possible.", 27, INK).move_to([0, 0.65, 0])
        g2 = txt("Each muscle's effort is its force as a fraction of its own strength.",
                 24, MUTED).move_to([0, 0.25, 0])
        self.play(FadeIn(g1), run_time=0.6)
        self.play(FadeIn(g2), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(g1), FadeOut(g2), obj.animate.scale(0.8).move_to([0, 2.05, 0]),
                  run_time=0.8)

        c1 = MathTex(r"\text{subject to}\quad R^{\top} f = \tau",
                     font_size=44, color=INK).move_to([0, 1.05, 0])
        e1 = txt("the forces must actually produce the required joint torques",
                 24, MUTED).next_to(c1, DOWN, buff=0.22)
        self.play(Write(c1), run_time=0.9)
        self.play(FadeIn(e1), run_time=0.6)
        self.wait(1.8)

        c2 = MathTex(r"0 \le f_i \le F_{\max,i}", font_size=44, color=INK).move_to([0, -0.35, 0])
        e2 = txt("no muscle may pull harder than it can -- and none may push",
                 24, MUTED).next_to(c2, DOWN, buff=0.22)
        self.play(Write(c2), run_time=0.9)
        self.play(FadeIn(e2), run_time=0.6)
        self.wait(1.8)

        why = txt("Why squared?  Squaring punishes big values disproportionately,\n"
                  "so sharing work among several muscles beats overloading one.",
                  25, INK).move_to([0, -1.75, 0])
        self.play(FadeIn(why), run_time=0.8)
        self.wait(2.2)

        self.takeaway("It works -- but it re-solves this problem at every instant.", AMBER)
        self.wait(2.2)


class S05_MomentArms(Slide):
    """tau = R^T f -- the equation the whole thesis turns on."""

    def construct(self):
        self.header("How muscle force becomes joint torque")

        # schematic: a muscle pulling across a joint
        joint = Dot([-3.2, 0.9, 0], radius=0.13, color=INK)
        seg1 = Line([-3.2, 0.9, 0], [-1.4, 1.7, 0], color=MUTED, stroke_width=7)
        seg2 = Line([-3.2, 0.9, 0], [-3.9, -0.9, 0], color=MUTED, stroke_width=7)
        muscle = Line([-2.2, 1.35, 0], [-3.6, -0.25, 0], color=BLUE, stroke_width=5)
        mlab = txt("muscle", 21, BLUE).next_to(muscle, LEFT, buff=0.16)
        arm = VGroup(seg1, seg2, joint, muscle, mlab)
        self.play(Create(seg1), Create(seg2), FadeIn(joint), run_time=0.7)
        self.play(Create(muscle), FadeIn(mlab), run_time=0.6)

        lever = DashedLine([-3.2, 0.9, 0], [-2.72, 0.36, 0], color=AMBER, stroke_width=4)
        llab = txt("moment arm", 21, AMBER).next_to(lever, RIGHT, buff=0.14).shift(UP * 0.05)
        self.play(Create(lever), FadeIn(llab), run_time=0.7)
        self.wait(0.8)

        spanner = txt("leverage -- exactly like the length of a spanner",
                      24, MUTED).move_to([-3.3, -1.7, 0])
        self.play(FadeIn(spanner), run_time=0.6)
        self.wait(1.4)

        eq = MathTex(r"\tau = R^{\top} f", font_size=76, color=INK).move_to([3.1, 1.35, 0])
        self.play(Write(eq), run_time=1.0)

        k1 = txt("f  -- nine muscle forces", 26, BLUE).move_to([3.3, 0.30, 0])
        k2 = txt("R  -- the leverage table", 26, AMBER).move_to([3.3, -0.15, 0])
        k3 = txt("τ  -- four joint torques", 26, INK).move_to([3.3, -0.60, 0])
        for k in (k1, k2, k3):
            self.play(FadeIn(k, shift=RIGHT * 0.2), run_time=0.45)
        self.wait(1.4)

        note = txt("Nine numbers in, four numbers out.\nThat squeeze is the redundancy problem.",
                   25, INK).move_to([3.1, -1.85, 0])
        self.play(FadeIn(note), run_time=0.8)
        self.wait(1.8)

        self.play(FadeOut(arm), FadeOut(lever), FadeOut(llab), FadeOut(spanner),
                  FadeOut(k1), FadeOut(k2), FadeOut(k3), FadeOut(note),
                  eq.animate.move_to([0, 1.35, 0]), run_time=0.9)

        a = txt("Classical method:  solve it backwards.  Given τ, find f.", 27, AMBER)
        b = txt("Learned controller:  never solves it.  It outputs f directly,", 27, BLUE)
        c = txt("and the physics produces whatever τ follows.", 27, BLUE)
        VGroup(a, b, c).arrange(DOWN, buff=0.26).move_to([0, -0.35, 0])
        self.play(FadeIn(a, shift=UP * 0.15), run_time=0.7)
        self.wait(1.0)
        self.play(FadeIn(b, shift=UP * 0.15), run_time=0.7)
        self.play(FadeIn(c), run_time=0.5)
        self.wait(1.6)

        self.takeaway("This one equation is the pivot of the whole thesis.", INK)
        self.wait(2.2)
