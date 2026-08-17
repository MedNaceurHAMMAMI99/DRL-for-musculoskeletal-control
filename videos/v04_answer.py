"""Part 4 -- The answer to the thesis question.  Scenes 17 to 24."""

from manim import *
from v_theme import *

MUSCLES = [
    ("deltoid anterior", 63.7, 59.2, 0.959),
    ("deltoid medial", 145.3, 112.3, 0.954),
    ("deltoid posterior", 35.5, 38.7, 0.859),
    ("triceps long", 180.4, 134.0, 0.910),
    ("biceps long", 52.1, 41.6, 0.893),
    ("brachialis", 113.9, 87.6, 0.871),
    ("biceps short", 90.7, 45.7, 0.764),
    ("triceps lateral", 79.1, 25.8, 0.416),
    ("triceps medial", 64.3, 14.2, 0.416),
]


class S17_ForceMethod(Slide):
    """How the two methods are made to face an identical problem."""

    def construct(self):
        self.header("How the comparison is made fair")

        pol = RoundedRectangle(width=3.9, height=1.15, corner_radius=0.12,
                               fill_color=PANEL, fill_opacity=1,
                               stroke_color=BLUE, stroke_width=3).move_to([-3.7, 1.75, 0])
        pol_l = txt("the learned policy", 25, BLUE, BOLD).move_to(pol.get_center())
        self.play(FadeIn(pol), FadeIn(pol_l), run_time=0.6)

        a1 = Arrow([-3.7, 1.10, 0], [-3.7, 0.30, 0], color=MUTED, stroke_width=5,
                   max_tip_length_to_length_ratio=0.25)
        tau = RoundedRectangle(width=4.6, height=1.05, corner_radius=0.12,
                               fill_color=PANEL, fill_opacity=1,
                               stroke_color=INK, stroke_width=3).move_to([-3.7, -0.28, 0])
        tau_l = MathTex(r"\tau\ \text{it actually produced}", font_size=32,
                        color=INK).move_to(tau.get_center())
        self.play(GrowArrow(a1), run_time=0.4)
        self.play(FadeIn(tau), FadeIn(tau_l), run_time=0.6)
        self.wait(0.8)

        a2 = Arrow([-1.35, -0.28, 0], [1.15, -0.28, 0], color=MUTED, stroke_width=5,
                   max_tip_length_to_length_ratio=0.12)
        a2_l = txt("hand the same τ to", 21, MUTED).next_to(a2, UP, buff=0.14)
        cls = RoundedRectangle(width=4.4, height=1.15, corner_radius=0.12,
                               fill_color=PANEL, fill_opacity=1,
                               stroke_color=AMBER, stroke_width=3).move_to([3.6, -0.28, 0])
        cls_l = txt("static optimisation", 25, AMBER, BOLD).move_to(cls.get_center())
        self.play(GrowArrow(a2), FadeIn(a2_l), run_time=0.6)
        self.play(FadeIn(cls), FadeIn(cls_l), run_time=0.6)
        self.wait(0.8)

        q = txt("What forces would IT have chosen, for that same torque?",
                27, INK, BOLD).move_to([0, -1.35, 0])
        self.play(FadeIn(q), run_time=0.8)
        self.wait(1.4)

        why = txt("Both methods now face an identical problem on an identical trajectory,\n"
                  "so any difference between them is purely a difference of strategy.",
                  25, MUTED).move_to([0, -2.15, 0])
        self.play(FadeIn(why), run_time=0.9)
        self.wait(1.6)

        chk = txt("Checked, not assumed:   ‖ Rᵀf − τ ‖  =  1.5 × 10⁻¹⁵ N·m", 25, OK, BOLD
                  ).move_to([0, -2.85, 0])
        self.play(FadeIn(chk), run_time=0.8)
        self.wait(2.4)


class S18_PatternAgrees(Slide):
    """Per-muscle forces, learned against classical."""

    def construct(self):
        self.header("Do the two methods use the same muscles?",
                    "mean force per muscle, over 2000 timesteps")

        leg = self.legend([(BLUE, "learned policy"), (AMBER, "static optimisation")])
        leg.move_to([2.6, 2.02, 0])
        self.play(FadeIn(leg), run_time=0.6)

        unit = 0.0235
        x0 = -1.55
        for i, (name, pol, cls, r) in enumerate(MUSCLES):
            y = 1.45 - i * 0.485
            lab = at(txt(name, 22, INK), x0 - 0.26, y, RIGHT)
            b1 = bar(pol * unit, 0.19, BLUE)
            b1.move_to([x0 + pol * unit / 2, y + 0.12, 0])
            b2 = bar(cls * unit, 0.19, AMBER)
            b2.move_to([x0 + cls * unit / 2, y - 0.12, 0])
            rl = at(txt(f"r = {r:.3f}", 21, BAD if r < 0.5 else MUTED,
                        BOLD if r < 0.5 else NORMAL), 5.05, y)
            self.play(FadeIn(lab), run_time=0.14)
            self.play(GrowFromEdge(b1, LEFT), GrowFromEdge(b2, LEFT), FadeIn(rl),
                      run_time=0.32)
        xl = at(txt("mean muscle force (N)", 21, MUTED), x0, -2.92)
        self.play(FadeIn(xl), run_time=0.4)
        self.wait(1.6)

        # call out the last two rows only -- the disagreement is local
        y_top = 1.45 - 7 * 0.485 + 0.30
        y_bot = 1.45 - 8 * 0.485 - 0.30
        box = RoundedRectangle(width=8.4, height=y_top - y_bot, corner_radius=0.10,
                               stroke_color=BAD, stroke_width=3, fill_opacity=0)
        box.move_to([1.55, (y_top + y_bot) / 2, 0])
        blab = txt("the two elbow extensors:  3 to 4.5 times the prescribed force",
                   24, BAD, BOLD).move_to([1.55, -3.22, 0])
        self.play(Create(box), run_time=0.7)
        self.play(FadeIn(blab), run_time=0.6)
        self.wait(2.6)


class S19_PatternVsEconomy(Slide):
    """Two questions with two different answers."""

    def construct(self):
        self.header("The answer splits in two")

        lbox = RoundedRectangle(width=6.2, height=2.55, corner_radius=0.16,
                                fill_color=PANEL, fill_opacity=1,
                                stroke_color=OK, stroke_width=3).move_to([-3.4, 1.05, 0])
        l1 = txt("THE PATTERN", 26, OK, BOLD).move_to([-3.4, 1.90, 0])
        l2 = txt("which muscles, in what proportion", 22, MUTED).move_to([-3.4, 1.50, 0])
        l3 = txt("r = 0.81 ± 0.04", 40, INK, BOLD).move_to([-3.4, 0.90, 0])
        l4 = txt("agrees", 28, OK, BOLD).move_to([-3.4, 0.30, 0])

        rbox = RoundedRectangle(width=6.2, height=2.55, corner_radius=0.16,
                                fill_color=PANEL, fill_opacity=1,
                                stroke_color=BAD, stroke_width=3).move_to([3.4, 1.05, 0])
        r1 = txt("THE ECONOMY", 26, BAD, BOLD).move_to([3.4, 1.90, 0])
        r2 = txt("how much force in total", 22, MUTED).move_to([3.4, 1.50, 0])
        r3 = txt("1.91 × the minimum", 40, INK, BOLD).move_to([3.4, 0.90, 0])
        r4 = txt("does not agree", 28, BAD, BOLD).move_to([3.4, 0.30, 0])

        self.play(FadeIn(lbox), run_time=0.5)
        for m in (l1, l2, l3, l4):
            self.play(FadeIn(m), run_time=0.4)
        self.wait(0.9)
        self.play(FadeIn(rbox), run_time=0.5)
        for m in (r1, r2, r3, r4):
            self.play(FadeIn(m), run_time=0.4)
        self.wait(1.6)

        # the null that the cosine number needed
        note = txt("A caution on the pattern number.", 27, INK, BOLD).move_to([0, -0.85, 0])
        n2 = txt("Muscle forces are never negative, so two completely unrelated force\n"
                 "vectors already look similar. Shuffling which muscle owns each force\n"
                 "still scores 0.373. That floor is arithmetic, not agreement.",
                 25, MUTED).move_to([0, -1.85, 0])
        self.play(FadeIn(note), run_time=0.6)
        self.play(FadeIn(n2), run_time=0.9)
        self.wait(2.4)

        self.takeaway("Quote the correlation, not the cosine: it is mean-centred and carries no floor.", INK)
        self.wait(2.4)


class S20_CoContraction(Slide):
    """Where the wasted effort goes."""

    def construct(self):
        self.header("Where the extra effort goes", "antagonists pulling against each other")

        el = np.array([-4.05, 0.95, 0])
        upper = Line(el + np.array([-0.30, 1.55, 0]), el, color=MUTED, stroke_width=11)
        fore = Line(el, el + np.array([1.45, -1.30, 0]), color=MUTED, stroke_width=11)
        jel = Dot(el, radius=0.14, color=INK)
        self.play(Create(upper), Create(fore), FadeIn(jel), run_time=0.7)

        flex = Line(el + np.array([-0.26, 0.95, 0]), el + np.array([0.42, -0.42, 0]),
                    color=BLUE, stroke_width=7)
        ext = Line(el + np.array([0.22, 1.00, 0]), el + np.array([0.62, -0.72, 0]),
                   color=PINK, stroke_width=7)
        self.play(Create(flex), run_time=0.5)
        self.play(Create(ext), run_time=0.5)
        fl = at(txt("biceps  -- bends the elbow", 22, BLUE), -6.30, -0.70)
        xl = at(txt("triceps -- straightens it", 22, PINK), -6.30, -1.12)
        self.play(FadeIn(fl), FadeIn(xl), run_time=0.6)
        self.wait(0.8)

        # the two torques, drawn head-to-head on their own line so the
        # cancellation is visible rather than implied
        ymid = -1.95
        a2 = Arrow([-6.30, ymid, 0], [-4.15, ymid, 0], color=BLUE,
                   stroke_width=7, max_tip_length_to_length_ratio=0.18, buff=0)
        a1 = Arrow([-1.85, ymid, 0], [-4.00, ymid, 0], color=PINK,
                   stroke_width=7, max_tip_length_to_length_ratio=0.18, buff=0)
        self.play(GrowArrow(a2), GrowArrow(a1), run_time=0.9)
        cancel = at(txt("net effect on the joint:  almost nothing", 23, INK, BOLD),
                    -6.30, -2.50)
        self.play(FadeIn(cancel), run_time=0.6)
        self.wait(1.2)

        t1 = txt("The elbow reaches roughly the right angle --", 27, INK)
        t2 = txt("and both groups have spent large forces", 27, INK)
        t3 = txt("to do what one alone would have done.", 27, INK)
        VGroup(t1, t2, t3).arrange(DOWN, buff=0.24).move_to([2.9, 1.30, 0])
        for t in (t1, t2, t3):
            self.play(FadeIn(t, shift=RIGHT * 0.15), run_time=0.55)
        self.wait(1.2)

        h1 = txt("Humans do this deliberately", 26, INK, BOLD)
        h2 = txt("when a joint must be stiff -- carrying a full cup,\n"
                 "steadying a hand. Throughout an unloaded reach\n"
                 "it is simply waste.", 24, MUTED)
        VGroup(h1, h2).arrange(DOWN, buff=0.22).move_to([2.9, -0.90, 0])
        self.play(FadeIn(h1), run_time=0.5)
        self.play(FadeIn(h2), run_time=0.8)
        self.wait(1.6)

        # status colour marks the verdict; the sentence itself stays in ink
        mark = RoundedRectangle(width=0.12, height=1.05, corner_radius=0.05,
                                fill_color=OK, fill_opacity=1, stroke_width=0)
        corr = txt("Corroborated:  two unrelated analyses --\n"
                   "the classical comparison, and a standard\n"
                   "EMG-derived index -- localise the same\n"
                   "defect to the same muscle group.", 22, INK)
        VGroup(mark, corr).arrange(RIGHT, buff=0.28).move_to([3.35, -2.45, 0])
        self.play(FadeIn(mark), FadeIn(corr), run_time=0.9)
        self.wait(2.4)


class S21_SeedVariance(Slide):
    """The single most informative result: same outcome, different solution."""

    def construct(self):
        self.header("Five identical runs", "differing only in the random seed")

        errs = [0.1142, 0.1086, 0.1034, 0.1066, 0.1041]
        energy = [327048, 483373, 273241, 361500, 398200]

        lt = txt("where the hand ended up", 26, INK).move_to([-3.5, 1.88, 0])
        rt = txt("how much muscle force it used", 26, INK).move_to([3.5, 1.88, 0])
        self.play(FadeIn(lt), FadeIn(rt), run_time=0.6)

        # left: final error, nearly identical
        base_l = -6.30
        for i, e in enumerate(errs):
            y = 1.30 - i * 0.52
            lab = at(txt(f"seed {i}", 21, MUTED), base_l, y)
            b = bar(e * 22.0, 0.32, BLUE)
            b.move_to([base_l + 0.95 + e * 22.0 / 2, y, 0])
            v = at(txt(f"{e:.4f} m", 21, INK), base_l + 0.95 + e * 22.0 + 0.20, y)
            self.play(FadeIn(lab), GrowFromEdge(b, LEFT), FadeIn(v), run_time=0.35)

        # right: energy, wildly different -- same scale treatment, different story
        base_r = 0.55
        for i, en in enumerate(energy):
            y = 1.30 - i * 0.52
            b = bar(en / 100000.0 * 1.05, 0.32, AMBER)
            b.move_to([base_r + en / 100000.0 * 1.05 / 2, y, 0])
            v = at(txt(f"{en:,}", 21, INK), base_r + en / 100000.0 * 1.05 + 0.20, y)
            self.play(GrowFromEdge(b, LEFT), FadeIn(v), run_time=0.35)
        self.wait(1.4)

        v1 = txt("varies by 4 %", 30, BLUE, BOLD).move_to([-3.5, -1.55, 0])
        v2 = txt("varies by 77 %", 30, AMBER, BOLD).move_to([3.5, -1.55, 0])
        self.play(FadeIn(v1, scale=1.1), run_time=0.7)
        self.play(FadeIn(v2, scale=1.1), run_time=0.7)
        self.wait(1.6)

        concl = txt("Five controllers arrive at the same place using materially different muscles.\n"
                    "That is the redundancy problem showing up directly in the results.",
                    27, INK).move_to([0, -2.18, 0])
        self.play(FadeIn(concl), run_time=0.9)
        self.wait(2.0)

        self.takeaway("Huge variance in the solution, none in the outcome.", AMBER)
        self.wait(2.4)


class S22_EffortSweep(Slide):
    """Weighting the effort term closes the gap -- as predicted."""

    def construct(self):
        self.header("Closing the gap", "raise the weight on effort, and ask again")

        pred = txt("Prediction, made before the experiment:  the effort term is 28 times\n"
                   "smaller than the precision bonus, so it cannot be influencing anything.",
                   26, INK).move_to([0, 1.95, 0])
        self.play(FadeIn(pred), run_time=0.9)
        self.wait(1.8)

        rows = [("w₂ = 1", 1.914, 0.150, 0.808, RAMP[0]),
                ("w₂ = 5", 1.406, 0.092, 0.919, RAMP[1]),
                ("w₂ = 20", 1.304, 0.025, 0.943, RAMP[2])]
        unit = 2.05
        x0 = -4.35
        floor_x = x0 + 1.263 * unit

        fl = DashedLine([floor_x, 1.02, 0], [floor_x, -1.28, 0], color=AMBER, stroke_width=3)
        fll = at(txt("1.263 -- what a properly tuned\nconventional controller achieves",
                     21, AMBER), floor_x + 0.22, 1.38)

        for i, (name, val, sd, r, col) in enumerate(rows):
            y = 0.55 - i * 0.62
            lab = at(txt(name, 24, INK, BOLD), x0 - 0.28, y, RIGHT)
            b = bar(val * unit, 0.42, col)
            b.move_to([x0 + val * unit / 2, y, 0])
            v = at(txt(f"{val:.3f} ± {sd:.3f}", 22, INK), x0 + val * unit + 0.26, y)
            self.play(FadeIn(lab), GrowFromEdge(b, LEFT), FadeIn(v), run_time=0.6)
        self.play(Create(fl), FadeIn(fll), run_time=0.7)
        xl = at(txt("effort used, relative to the classical minimum", 22, MUTED), x0, -1.55)
        self.play(FadeIn(xl), run_time=0.4)
        self.wait(1.8)

        res = VGroup(
            self.bullet("fidelity improves every step:  0.808 → 0.919 → 0.943", BLUE, 25),
            self.bullet("accuracy cost is modest:  8 % worse final error", BLUE, 25),
            self.bullet("seed spread collapses:  ±0.150 → ±0.092 → ±0.025", BLUE, 25),
        ).arrange(DOWN, buff=0.26, aligned_edge=LEFT).move_to([-0.4, -2.50, 0])
        for b in res:
            self.play(FadeIn(b, shift=RIGHT * 0.15), run_time=0.55)
        self.wait(1.8)

        self.play(FadeOut(res), run_time=0.5)
        big = txt("The spread collapsing is the real evidence:  weighting that dimension\n"
                  "pins the solution down, so the objective decides it, not the seed.",
                  26, INK, BOLD).move_to([0, -2.50, 0])
        self.play(FadeIn(big, shift=UP * 0.15), run_time=0.9)
        self.wait(2.8)


class S23_VsConventional(Slide):
    """The honest comparison, and a claim that was withdrawn."""

    def construct(self):
        self.header("Against a conventional controller", "both sides properly tuned")

        w1 = txt("An earlier draft claimed the learned policy was MORE economical\n"
                 "than a controller that optimises effort in closed form.",
                 26, INK).move_to([0, 1.95, 0])
        self.play(FadeIn(w1), run_time=0.9)
        self.wait(1.6)
        w2 = txt("It rested on one gain pair from a coarse 12-point grid.\n"
                 "A 30-point search overturned it.  The claim was withdrawn.",
                 26, BAD).move_to([0, 0.95, 0])
        self.play(FadeIn(w2), run_time=0.9)
        self.wait(2.0)
        self.play(FadeOut(w1), FadeOut(w2), run_time=0.6)

        hdrs = [("", -5.10), ("learned  (w₂ = 20)", -0.30), ("conventional", 3.75)]
        for label, x in hdrs:
            if label:
                self.play(FadeIn(at(txt(label, 25, BLUE if "learned" in label else AMBER, BOLD),
                                    x, 1.95)), run_time=0.35)

        rows = [
            ("final error", "0.107 m", "0.165 m", 0),
            ("effort ratio", "1.304", "1.263", 1),
            ("agreement with classical", "r = 0.943", "r = 0.931", 0),
        ]
        for i, (metric, a, b, winner) in enumerate(rows):
            y = 1.15 - i * 0.72
            self.play(FadeIn(at(txt(metric, 25, INK), -5.10, y)), run_time=0.3)
            ta = at(txt(a, 27, BLUE if winner == 0 else MUTED,
                        BOLD if winner == 0 else NORMAL), -0.30, y)
            tb = at(txt(b, 27, AMBER if winner == 1 else MUTED,
                        BOLD if winner == 1 else NORMAL), 3.75, y)
            self.play(FadeIn(ta), FadeIn(tb), run_time=0.45)
        self.wait(1.8)

        v = txt("Neither method dominates.", 34, INK, BOLD).move_to([0, -1.35, 0])
        self.play(FadeIn(v, scale=1.06), run_time=0.8)
        self.wait(1.0)

        d = txt("The learned policy reaches about 35 % more accurately.\n"
                "The conventional controller is about 3 % more economical --\n"
                "a margin comparable to the policy's own seed spread.",
                26, INK).move_to([0, -2.35, 0])
        self.play(FadeIn(d), run_time=0.9)
        self.wait(2.6)


class S24_Conclusion(Slide):
    """What the thesis answers, and how confident it is."""

    def construct(self):
        self.header("The answer, and how much to trust it")

        q = txt("Does DRL produce the same per-muscle forces as classical calculation?",
                28, INK, BOLD).move_to([0, 2.20, 0])
        self.play(FadeIn(q), run_time=0.9)
        self.wait(1.2)

        a1 = txt("Yes -- both the pattern and the magnitude --", 30, OK, BOLD)
        a2 = txt("provided the objective asks for both.", 30, INK)
        a3 = txt("The version first reported did not ask.", 26, MUTED)
        VGroup(a1, a2, a3).arrange(DOWN, buff=0.24).move_to([0, 1.05, 0])
        for m in (a1, a2, a3):
            self.play(FadeIn(m, shift=UP * 0.12), run_time=0.6)
        self.wait(1.8)

        rows = [
            ("γ causes the improvement", "high", OK),
            ("load-sharing pattern agrees", "high", OK),
            ("excess effort is elbow co-contraction", "high", OK),
            ("raising the effort weight closes the gap", "high", OK),
            ("learned vs conventional trade-off", "moderate", AMBER),
            ("algorithm ordering", "moderate", AMBER),
            ("synergy agreement", "low", BAD),
            ("anything about human motor control", "none", BAD),
        ]
        for i, (claim, conf, col) in enumerate(rows):
            y = -0.12 - i * 0.355
            self.play(FadeIn(at(txt(claim, 23, INK), -5.60, y)),
                      FadeIn(at(txt(conf, 23, col, BOLD), 3.05, y)), run_time=0.32)
        self.play(FadeIn(at(txt("confidence", 21, MUTED), 3.05, 0.30)), run_time=0.3)
        self.wait(2.0)

        bound = txt("No real arm, no EMG, and no human subject enters this work at any point.",
                    25, INK, BOLD).move_to([0, -3.15, 0])
        self.play(FadeIn(bound), run_time=0.9)
        self.wait(2.8)


class S25_SpeedWithdrawn(Slide):
    """The motivation that did not survive measurement."""

    def construct(self):
        self.header("A claim that did not survive", "the original motivation was speed")

        first = txt("First measurement:  classical 2165 µs   vs   network 295 µs", 28, INK)
        firstv = txt("a 7.3 × advantage", 34, BLUE, BOLD)
        grp = VGroup(first, firstv).arrange(DOWN, buff=0.28).move_to([0, 1.55, 0])
        self.play(FadeIn(first), run_time=0.7)
        self.play(FadeIn(firstv, scale=1.06), run_time=0.6)
        self.wait(1.8)
        self.play(FadeOut(grp), run_time=0.6)

        o1 = txt("It timed the wrong computation.", 27, BAD, BOLD)
        o2 = txt("2165 µs is the muscle-model solve inside forward simulation --\n"
                 "not the load-sharing problem this thesis actually asks about.",
                 24, MUTED)
        o3 = txt("Both sides were unoptimised Python.", 27, BAD, BOLD)
        VGroup(o1, o2, o3).arrange(DOWN, buff=0.26).move_to([0, 0.85, 0])
        for m in (o1, o2, o3):
            self.play(FadeIn(m, shift=RIGHT * 0.12), run_time=0.65)
        self.wait(1.6)

        rows = [("network forward pass", 288.8, BLUE),
                ("classical, general solver", 2370.5, AMBER),
                ("classical, direct solve", 23.9, AMBER)]
        unit = 0.0021
        x0 = -2.30
        for i, (name, val, col) in enumerate(rows):
            y = -0.75 - i * 0.58
            self.play(FadeIn(at(txt(name, 24, INK), x0 - 0.26, y, RIGHT)), run_time=0.25)
            b = bar(max(val * unit, 0.06), 0.36, col)
            b.move_to([x0 + max(val * unit, 0.06) / 2, y, 0])
            self.play(GrowFromEdge(b, LEFT),
                      FadeIn(at(txt(f"{val:,.1f} µs", 23, INK),
                                x0 + max(val * unit, 0.06) + 0.24, y)), run_time=0.5)
        self.wait(1.4)

        self.takeaway("Against a direct solve the network is about twelve times slower. Claim withdrawn.", BAD)
        self.wait(2.6)
