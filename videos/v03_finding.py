"""Part 3 -- The key finding.  Scenes 12 to 16."""

from manim import *
from v_theme import *


class S12_DiscountHorizon(Slide):
    """What gamma actually does, and why the default was wrong here."""

    def construct(self):
        self.header("The discount factor", "one number decides how far ahead the agent looks")

        eq = MathTex(r"G_t = r_t + \gamma\, r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots",
                     font_size=44, color=INK).move_to([0, 1.60, 0])
        self.play(Write(eq), run_time=1.3)
        expl = txt("Rewards further in the future count for less.", 26, MUTED).move_to([0, 1.10, 0])
        self.play(FadeIn(expl), run_time=0.6)
        self.wait(1.2)

        rule = MathTex(r"\text{effective horizon} \;\approx\; \frac{1}{1-\gamma}\ \text{steps}",
                       font_size=44, color=BLUE).move_to([0, 0.40, 0])
        self.play(Write(rule), run_time=1.0)
        self.wait(1.4)

        # two horizons drawn against the episode, to scale
        x0, x1 = -5.6, 5.6
        span = x1 - x0
        ep = Line([x0, -0.80, 0], [x1, -0.80, 0], color=GRID, stroke_width=3)
        ep_l = at(txt("one episode = 100 steps = 2.0 s", 22, MUTED), x1, -0.48, RIGHT)
        self.play(Create(ep), FadeIn(ep_l), run_time=0.7)

        h_def = bar(span, 0.36, AMBER).move_to([x0 + span / 2, -1.22, 0])
        d_l = at(txt("γ = 0.99  (library default)  →  100 steps", 23, AMBER), x0, -1.58)

        h_tun = bar(span * 0.23, 0.36, BLUE).move_to([x0 + span * 0.23 / 2, -1.94, 0])
        t_l = at(txt("γ = 0.957  (selected)  →  23 steps = 0.46 s", 23, BLUE), x0, -2.30)

        for h, l in ((h_def, d_l), (h_tun, t_l)):
            w = h.width
            h.stretch_to_fit_width(0.02).align_to([x0, 0, 0], LEFT)
            self.play(h.animate.stretch_to_fit_width(w).align_to([x0, 0, 0], LEFT),
                      FadeIn(l), run_time=0.9)
        self.wait(1.2)

        # what the arm actually does, drawn against both horizons
        xm = x0 + span * 0.25
        mark = DashedLine([xm, -0.62, 0], [xm, -2.16, 0], color=INK, stroke_width=3)
        mlab = at(txt("the hand arrives here, at step 25", 22, INK, BOLD), x1, -2.30, RIGHT)
        self.play(Create(mark), run_time=0.7)
        self.play(FadeIn(mlab), run_time=0.6)
        self.wait(2.0)

        self.takeaway("The default horizon was four times longer than the movement.", BLUE)
        self.wait(2.4)


class S13_EntropyRefuted(Slide):
    """A confident, well-argued, wrong answer -- and the test that killed it."""

    def construct(self):
        self.header("A hypothesis that was wrong", "and it took an experiment to find out")

        h = txt("Belief:  SAC's target entropy keeps the policy too random to settle.",
                29, INK, BOLD).move_to([0, 1.85, 0])
        self.play(FadeIn(h), run_time=0.8)
        self.wait(1.0)

        ev = VGroup(
            self.bullet("the mechanism fits the symptom exactly", BLUE, 25),
            self.bullet("the default scales with muscle count, not with the task", BLUE, 25),
            self.bullet("all five best search trials chose about −19.6, not −9", BLUE, 25),
            self.bullet("it was written up as a contribution of the thesis", BLUE, 25),
        ).arrange(DOWN, buff=0.26, aligned_edge=LEFT).move_to([-0.7, 0.70, 0])
        for b in ev:
            self.play(FadeIn(b, shift=RIGHT * 0.15), run_time=0.5)
        self.wait(1.6)

        self.play(FadeOut(ev), FadeOut(h), run_time=0.6)

        test = txt("The test:  change that one parameter, in both directions.",
                   28, INK).move_to([0, 1.90, 0])
        self.play(FadeIn(test), run_time=0.7)

        rows = [
            ("all defaults", 0.2613, "0 %", MUTED),
            ("defaults + tuned entropy only", 0.2688, "−8 %", BAD),
            ("tuned − entropy reverted", 0.1552, "+106 %", OK),
            ("all tuned", 0.1610, "100 %", MUTED),
        ]
        unit = 13.5
        x0 = -2.4
        for i, (name, val, gap, col) in enumerate(rows):
            y = 1.10 - i * 0.72
            lab = at(txt(name, 24, INK), x0 - 0.28, y, RIGHT)
            b = bar(val * unit, 0.46, BLUE)
            b.move_to([x0 + val * unit / 2, y, 0])
            v = at(txt(f"{val:.4f} m", 23, INK), x0 + val * unit + 0.26, y)
            gl = at(txt(gap, 24, col, BOLD), 5.15, y)
            w = b.width
            b.stretch_to_fit_width(0.02).align_to([x0, 0, 0], LEFT)
            self.play(FadeIn(lab), run_time=0.25)
            self.play(b.animate.stretch_to_fit_width(w).align_to([x0, 0, 0], LEFT),
                      FadeIn(v), FadeIn(gl), run_time=0.6)
        hd = at(txt("gap closed", 22, MUTED), 5.15, 1.62)
        self.play(FadeIn(hd), run_time=0.3)
        self.wait(2.0)

        verdict = txt("Adding it recovers nothing.  Removing it costs nothing.\n"
                      "Both directions agree:  the target entropy is not the mechanism.",
                      27, INK, BOLD).move_to([0, -1.95, 0])
        self.play(FadeIn(verdict, shift=UP * 0.15), run_time=0.9)
        self.wait(2.2)

        self.takeaway("A hyperparameter search finds a configuration, never a cause.", BAD)
        self.wait(2.4)


class S14_Ablation(Slide):
    """Five parameters tested one at a time.  Four do nothing."""

    def construct(self):
        self.header("Which parameter was responsible", "library defaults, plus exactly one tuned value")

        rows = [
            ("nothing changed", 0.2613, "0 %", False),
            ("+ target entropy", 0.2688, "−8 %", False),
            ("+ learning rate", 0.2642, "−3 %", False),
            ("+ target-network rate τ", 0.2660, "−5 %", False),
            ("+ batch size", 0.2658, "−5 %", False),
            ("+ discount factor γ", 0.1553, "+106 %", True),
        ]
        unit = 13.0
        x0 = -1.9
        bars = []
        for i, (name, val, gap, hero) in enumerate(rows):
            y = 1.60 - i * 0.68
            lab = at(txt(name, 24, INK if hero else MUTED, BOLD if hero else NORMAL),
                     x0 - 0.28, y, RIGHT)
            b = bar(val * unit, 0.44, BLUE)
            b.move_to([x0 + val * unit / 2, y, 0])
            v = at(txt(f"{val:.4f}", 23, INK), x0 + val * unit + 0.26, y)
            gl = at(txt(gap, 24, OK if hero else MUTED, BOLD), 4.85, y)
            w = b.width
            b.stretch_to_fit_width(0.02).align_to([x0, 0, 0], LEFT)
            self.play(FadeIn(lab), run_time=0.22)
            self.play(b.animate.stretch_to_fit_width(w).align_to([x0, 0, 0], LEFT),
                      FadeIn(v), FadeIn(gl), run_time=0.55)
            bars.append((b, hero))
            if hero:
                ring = SurroundingRectangle(b, color=OK, buff=0.09, stroke_width=3,
                                            corner_radius=0.08)
                self.play(Create(ring), run_time=0.5)
        hd = at(txt("gap closed", 22, MUTED), 4.85, 2.10)
        xl = at(txt("final reaching error (m) -- shorter is better", 22, MUTED), x0, -2.55)
        self.play(FadeIn(hd), FadeIn(xl), run_time=0.5)
        self.wait(1.8)

        line = txt("Five parameters tested individually.\n"
                   "Four do nothing.  One explains everything.",
                   28, INK, BOLD).move_to([0, -3.02, 0])
        self.play(FadeIn(line, shift=UP * 0.15), run_time=0.9)
        self.wait(2.6)


class S15_AlgoBenchmark(Slide):
    """The benchmark that had to be thrown away and run again."""

    def construct(self):
        self.header("Comparing the four algorithms", "the first comparison was not a fair one")

        prob = txt("SAC ran with γ = 0.957.  The other three ran at the default 0.99.\n"
                   "One entrant correctly configured, three handicapped.",
                   27, INK).move_to([0, 1.85, 0])
        self.play(FadeIn(prob), run_time=0.9)
        self.wait(1.8)

        # paired bars: before / after, one colour per condition, both labelled
        algos = [("SAC", 0.2613, 0.1909), ("DDPG", 0.4161, 0.2510),
                 ("TD3", 0.2786, 0.2579), ("PPO", 0.4231, 0.4661)]
        unit = 8.2
        x0 = -2.0
        leg = self.legend([(AMBER, "before:  γ = 0.99"), (BLUE, "after:  γ = 0.957")])
        leg.move_to([0.9, 0.85, 0])
        self.play(FadeIn(leg), run_time=0.6)

        for i, (name, before, after) in enumerate(algos):
            yc = 0.20 - i * 0.82
            lab = at(txt(name, 26, INK, BOLD), x0 - 0.30, yc, RIGHT)
            b1 = bar(before * unit, 0.28, AMBER)
            b1.move_to([x0 + before * unit / 2, yc + 0.19, 0])
            b2 = bar(after * unit, 0.28, BLUE)
            b2.move_to([x0 + after * unit / 2, yc - 0.19, 0])
            v1 = at(txt(f"{before:.3f}", 21, MUTED), x0 + before * unit + 0.22, yc + 0.19)
            v2 = at(txt(f"{after:.3f}", 21, INK), x0 + after * unit + 0.22, yc - 0.19)
            chg = (after - before) / before * 100
            cl = at(txt(f"{chg:+.0f} %", 23, OK if chg < 0 else BAD, BOLD), 4.75, yc)
            self.play(FadeIn(lab), run_time=0.2)
            for b, v in ((b1, v1), (b2, v2)):
                w = b.width
                b.stretch_to_fit_width(0.02).align_to([x0, 0, 0], LEFT)
                self.play(b.animate.stretch_to_fit_width(w).align_to([x0, 0, 0], LEFT),
                          FadeIn(v), run_time=0.4)
            self.play(FadeIn(cl), run_time=0.3)
        xl = at(txt("final reaching error (m) -- shorter is better", 22, MUTED), x0, -3.02)
        self.play(FadeIn(xl), run_time=0.4)
        self.wait(2.0)

        note = txt("DDPG improves 40 % and overtakes TD3.  The ranking changes.\n"
                   "The textbook explanation given for the old ordering did not survive.",
                   26, INK).move_to([0, 1.60, 0])
        self.play(FadeOut(prob), run_time=0.4)
        self.play(FadeIn(note), run_time=0.9)
        self.wait(2.6)


class S16_GammaBeatsAlgorithm(Slide):
    """The comparison that actually matters."""

    def construct(self):
        self.header("What mattered more: the algorithm, or one setting?")

        a = txt("best vs second-best algorithm", 26, MUTED)
        a_v = txt("0.060 m", 46, MUTED, BOLD)
        ga = VGroup(a, a_v).arrange(DOWN, buff=0.28).move_to([-3.5, 1.10, 0])

        b = txt("correcting γ in DDPG alone", 26, INK)
        b_v = txt("0.165 m", 46, BLUE, BOLD)
        gb = VGroup(b, b_v).arrange(DOWN, buff=0.28).move_to([3.5, 1.10, 0])

        self.play(FadeIn(ga), run_time=0.8)
        self.wait(0.8)
        self.play(FadeIn(gb), run_time=0.8)
        self.wait(1.0)

        unit = 15.0
        b1 = bar(0.060 * unit, 0.55, MUTED)
        b1.move_to([-3.5 - 0.060 * unit / 2 + 0.060 * unit / 2, -0.35, 0])
        b1.move_to([-3.5, -0.35, 0])
        b2 = bar(0.165 * unit, 0.55, BLUE).move_to([3.5, -0.35, 0])
        self.play(GrowFromCenter(b1), run_time=0.5)
        self.play(GrowFromCenter(b2), run_time=0.7)
        self.wait(1.2)

        line = txt("Roughly three times the gap between algorithms.", 30, INK, BOLD
                   ).move_to([0, -1.35, 0])
        self.play(FadeIn(line, shift=UP * 0.15), run_time=0.8)
        self.wait(1.6)

        adv = txt("Effort spent choosing an algorithm is poorly allocated\n"
                  "against effort spent matching γ to the task timescale.",
                  27, INK).move_to([0, -2.30, 0])
        self.play(FadeIn(adv), run_time=0.9)
        self.wait(2.4)

        self.takeaway("Match the discount factor to the behaviour, not to a library default.", BLUE)
        self.wait(2.4)
