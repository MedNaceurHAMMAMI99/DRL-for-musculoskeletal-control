"""Part 2 -- The model and the method.  Scenes 06 to 11."""

from manim import *
from v_theme import *


class S06_TheArm(Slide):
    """What is actually being simulated."""

    def construct(self):
        self.header("The model", "three bodies, four degrees of freedom, nine muscles")

        sh = np.array([-4.85, 1.25, 0])
        el = np.array([-3.95, -0.30, 0])
        hd = np.array([-2.90, -1.65, 0])

        torso = Line(sh + np.array([-0.10, 0.85, 0]), sh + np.array([-0.45, -0.85, 0]),
                     color=GRID, stroke_width=14)
        upper = Line(sh, el, color=MUTED, stroke_width=11)
        fore = Line(el, hd, color=MUTED, stroke_width=11)
        jsh = Dot(sh, radius=0.15, color=INK)
        jel = Dot(el, radius=0.13, color=INK)
        hand = Dot(hd, radius=0.11, color=INK)

        self.play(Create(torso), run_time=0.5)
        self.play(Create(upper), Create(fore), run_time=0.7)
        self.play(FadeIn(jsh), FadeIn(jel), FadeIn(hand), run_time=0.4)

        lsh = txt("shoulder", 21, MUTED).next_to(jsh, LEFT, buff=0.30)
        lel = txt("elbow", 21, MUTED).next_to(jel, RIGHT, buff=0.30)
        lhd = txt("hand", 21, MUTED).next_to(hand, DOWN, buff=0.22)
        self.play(FadeIn(lsh), FadeIn(lel), FadeIn(lhd), run_time=0.5)
        self.wait(0.8)

        # nine muscles, drawn as lines spanning the joints they cross
        # nine muscle lines, fanned so none hides another
        specs = [
            (-0.34, -0.30, 0.16, 0.34), (-0.20, -0.46, 0.24, 0.24), (-0.06, -0.60, 0.30, 0.14),
            (0.22, -0.18, -0.26, 0.34), (0.30, -0.32, -0.18, 0.24), (0.38, -0.46, -0.10, 0.14),
        ]
        muscles = VGroup()
        for dx0, dy0, dx1, dy1 in specs:
            muscles.add(Line(sh + np.array([dx0, dy0, 0]), el + np.array([dx1, dy1, 0]),
                             color=BLUE, stroke_width=4.5))
        for dx0, dy0, dx1, dy1 in [(-0.46, 0.30, 0.30, -0.58), (-0.10, 0.46, 0.44, -0.42),
                                   (0.28, 0.38, 0.54, -0.24)]:
            muscles.add(Line(sh + np.array([dx0, dy0, 0]), sh + np.array([dx1, dy1, 0]),
                             color=BLUE, stroke_width=4.5))
        self.play(LaggedStart(*[Create(m) for m in muscles], lag_ratio=0.12), run_time=1.8)
        mlab = txt("9 muscles", 24, BLUE).move_to([-4.5, -2.55, 0])
        self.play(FadeIn(mlab), run_time=0.4)
        self.wait(1.0)

        rows = [
            ("shoulder flexion", "arm forward and back", True),
            ("shoulder abduction", "arm out to the side", True),
            ("shoulder rotation", "upper arm twisting", False),
            ("elbow flexion", "forearm bending", True),
        ]
        items = VGroup()
        for i, (name, what, actuated) in enumerate(rows):
            y = 1.55 - i * 0.52
            n = at(txt(name, 25, INK), -1.55, y)
            w = at(txt(what, 22, MUTED), 1.55, y)
            mark = at(txt("driven" if actuated else "restrained", 21,
                          OK if actuated else BAD, BOLD), 4.80, y)
            items.add(VGroup(n, w, mark))
        for row in items:
            self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.5)
        self.wait(1.4)

        note = txt("Shoulder rotation is restrained because the muscles\n"
                   "that turn the upper arm are not in this set.\n\n"
                   "The model had a joint no muscle could control.\n"
                   "A learner cannot fix that. It is a missing part.",
                   24, INK)
        at(note, -1.55, -1.55, LEFT)
        self.play(FadeIn(note), run_time=0.9)
        self.wait(2.6)


class S07_HillModel(Slide):
    """A muscle is not a motor."""

    def construct(self):
        self.header("How a muscle is modelled", "the Hill model: three elements")

        y = 1.35
        bone_l = Line([-5.6, y, 0], [-4.9, y, 0], color=GRID, stroke_width=12)
        bone_r = Line([-0.4, y, 0], [0.3, y, 0], color=GRID, stroke_width=12)

        ce = RoundedRectangle(width=2.0, height=0.60, corner_radius=0.10,
                              fill_color=BLUE, fill_opacity=1, stroke_width=0
                              ).move_to([-3.6, y + 0.42, 0])
        ce_l = txt("contractile", 20, BG, BOLD).move_to(ce.get_center())
        pee = RoundedRectangle(width=2.0, height=0.44, corner_radius=0.10,
                               fill_color=GRID, fill_opacity=1, stroke_width=0
                               ).move_to([-3.6, y - 0.42, 0])
        pee_l = txt("parallel spring", 19, MUTED).move_to(pee.get_center())
        see = RoundedRectangle(width=1.5, height=0.44, corner_radius=0.10,
                               fill_color=AMBER, fill_opacity=1, stroke_width=0
                               ).move_to([-1.5, y, 0])
        see_l = txt("tendon", 20, BG, BOLD).move_to(see.get_center())

        links = VGroup(
            Line([-4.9, y, 0], [-4.6, y, 0], color=MUTED, stroke_width=4),
            Line([-2.6, y, 0], [-2.25, y, 0], color=MUTED, stroke_width=4),
            Line([-0.75, y, 0], [-0.4, y, 0], color=MUTED, stroke_width=4),
        )
        self.play(Create(bone_l), Create(bone_r), run_time=0.5)
        self.play(FadeIn(ce), FadeIn(ce_l), run_time=0.6)
        self.play(FadeIn(pee), FadeIn(pee_l), run_time=0.5)
        self.play(FadeIn(see), FadeIn(see_l), Create(links), run_time=0.6)
        self.wait(1.2)

        eq = MathTex(r"F = \big[\, a\, f_L(\tilde{l})\, f_V(\tilde{v}) + f_{PE}(\tilde{l}) \,\big]"
                     r"\, F_{\max} \cos\alpha",
                     font_size=40, color=INK).move_to([2.9, 1.35, 0])
        self.play(Write(eq), run_time=1.4)
        self.wait(1.2)

        p1 = txt("Force depends on three things at once:", 27, INK).move_to([0, -0.05, 0])
        b1 = self.bullet("how hard it is being driven", BLUE, 25)
        b2 = self.bullet("how stretched it currently is", BLUE, 25)
        b3 = self.bullet("how fast it is changing length", BLUE, 25)
        bs = VGroup(b1, b2, b3).arrange(DOWN, buff=0.24, aligned_edge=LEFT).move_to([-2.6, -1.15, 0])
        self.play(FadeIn(p1), run_time=0.6)
        for b in bs:
            self.play(FadeIn(b, shift=RIGHT * 0.15), run_time=0.45)
        self.wait(1.0)

        fam = txt("Familiar from your own body:\n\n"
                  "a push-up is hardest at the bottom, where the\n"
                  "chest muscles are at an awkward length;\n\n"
                  "you can lower a weight you could never lift,\n"
                  "because muscles are stronger resisting than pulling.",
                  23, MUTED).move_to([3.3, -1.25, 0])
        self.play(FadeIn(fam), run_time=1.0)
        self.wait(2.6)

        self.takeaway("So 'how much force can this muscle make?' has no single answer.", BLUE)
        self.wait(2.2)


class S08_HowRLWorks(Slide):
    """The learning loop, and the one idea people get wrong about it."""

    def construct(self):
        self.header("How reinforcement learning works")

        agent = RoundedRectangle(width=3.5, height=1.5, corner_radius=0.14,
                                 fill_color=PANEL, fill_opacity=1,
                                 stroke_color=BLUE, stroke_width=3).move_to([-3.4, 1.02, 0])
        a_l = txt("AGENT", 28, BLUE, BOLD).move_to(agent.get_center() + UP * 0.26)
        a_s = txt("a neural network", 21, MUTED).move_to(agent.get_center() + DOWN * 0.28)

        env = RoundedRectangle(width=3.5, height=1.5, corner_radius=0.14,
                               fill_color=PANEL, fill_opacity=1,
                               stroke_color=AMBER, stroke_width=3).move_to([3.4, 1.02, 0])
        e_l = txt("ENVIRONMENT", 28, AMBER, BOLD).move_to(env.get_center() + UP * 0.26)
        e_s = txt("the simulated arm", 21, MUTED).move_to(env.get_center() + DOWN * 0.28)

        self.play(FadeIn(agent), FadeIn(a_l), FadeIn(a_s), run_time=0.6)
        self.play(FadeIn(env), FadeIn(e_l), FadeIn(e_s), run_time=0.6)

        top = CurvedArrow(agent.get_top() + UP * 0.05, env.get_top() + UP * 0.05,
                          angle=-0.55, color=BLUE, stroke_width=5, tip_length=0.22)
        t_l = txt("action:  9 muscle commands", 23, BLUE).next_to(top, UP, buff=0.10)
        bot = CurvedArrow(env.get_bottom() + DOWN * 0.05, agent.get_bottom() + DOWN * 0.05,
                          angle=-0.55, color=AMBER, stroke_width=5, tip_length=0.22)
        b_l = txt("observation:  38 numbers      +      reward:  one score",
                  23, AMBER).next_to(bot, DOWN, buff=0.10)

        self.play(Create(top), FadeIn(t_l), run_time=0.8)
        self.play(Create(bot), FadeIn(b_l), run_time=0.8)
        self.wait(0.6)
        for _ in range(2):
            self.play(ShowPassingFlash(top.copy().set_color(INK), time_width=0.5), run_time=0.8)
            self.play(ShowPassingFlash(bot.copy().set_color(INK), time_width=0.5), run_time=0.8)
        self.wait(0.5)

        rep = txt("Repeat a million times.  Make high-scoring behaviour more likely.",
                  26, INK).move_to([0, -1.55, 0])
        self.play(FadeIn(rep), run_time=0.7)
        self.wait(1.2)

        key = txt("The agent is never told HOW to move -- only how good the outcome was.",
                  27, INK, BOLD).move_to([0, -2.20, 0])
        self.play(FadeIn(key, shift=UP * 0.15), run_time=0.8)
        self.wait(1.6)

        self.takeaway("So any behaviour that scores well will be found -- including yours by mistake.", BLUE)
        self.wait(2.4)


class S09_TheReward(Slide):
    """The scoring rule, seven terms, each with a job."""

    def construct(self):
        self.header("The reward function", "seven terms -- every one of them earned its place")

        rows = [
            ("survival", "+ 4.5 every step", "makes staying alive worth more than quitting", OK),
            ("distance cost", "- 1.0 (e² + 0.5e)", "the linear half is what closes the last stretch", BLUE),
            ("effort", "- w₂ · mean (f/Fmax)²", "the classical criterion, in the reward", AMBER),
            ("smoothness", "- 0.5 · mean (Δa)²", "punishes twitching between steps", BLUE),
            ("precision bonus", "+ 3.0 · exp(-e / 0.15)", "a smooth pull toward the target", BLUE),
            ("success bonus", "+ 1.0 per step on target", "per step, so it must hold, not just arrive", OK),
            ("blow-up penalty", "- 10.0", "belt and braces against self-destruction", BAD),
        ]
        # fixed columns: swatch | name | expression | why it exists
        items = VGroup()
        for i, (name, expr, why, col) in enumerate(rows):
            y = 1.62 - i * 0.52
            sw = RoundedRectangle(width=0.12, height=0.32, corner_radius=0.04,
                                  fill_color=col, fill_opacity=1, stroke_width=0)
            at(sw, -6.35, y)
            n = at(txt(name, 24, INK), -6.10, y)
            e = at(txt(expr, 22, col), -3.60, y)
            w = at(txt(why, 21, MUTED), -0.30, y)
            items.add(VGroup(sw, n, e, w))

        for it in items:
            self.play(FadeIn(it, shift=RIGHT * 0.18), run_time=0.42)
        self.wait(1.8)

        note = txt("The survival term and the per-step success bonus are not decoration.\n"
                   "Each exists because its absence produced a specific failure.",
                   24, INK).move_to([0, -1.75, 0])
        self.play(FadeIn(note), run_time=0.8)
        self.wait(1.6)

        self.takeaway("A reward is a scoring rule, not an instruction.", BLUE)
        self.wait(2.2)


class S10_TheExploit(Slide):
    """The agent found that quitting scored better than playing."""

    def construct(self):
        self.header("What went wrong first", "the reward had an exploit in it")

        b1 = txt("Every per-step reward was zero or negative.", 28, INK)
        b2 = txt("A numerical blow-up ended the episode with no penalty.", 28, INK)
        VGroup(b1, b2).arrange(DOWN, buff=0.26).move_to([0, 1.85, 0])
        self.play(FadeIn(b1), run_time=0.7)
        self.play(FadeIn(b2), run_time=0.7)
        self.wait(1.4)

        # the accounting, side by side
        lbox = RoundedRectangle(width=5.6, height=1.85, corner_radius=0.14,
                                fill_color=PANEL, fill_opacity=1,
                                stroke_color=BAD, stroke_width=3).move_to([-3.3, 0.25, 0])
        l1 = txt("survive all 100 steps", 25, INK).move_to(lbox.get_center() + UP * 0.42)
        l2 = txt("-20", 46, BAD, BOLD).move_to(lbox.get_center() + DOWN * 0.32)

        rbox = RoundedRectangle(width=5.6, height=1.85, corner_radius=0.14,
                                fill_color=PANEL, fill_opacity=1,
                                stroke_color=OK, stroke_width=3).move_to([3.3, 0.25, 0])
        r1 = txt("quit at step 7", 25, INK).move_to(rbox.get_center() + UP * 0.42)
        r2 = txt("0", 46, OK, BOLD).move_to(rbox.get_center() + DOWN * 0.32)

        self.play(FadeIn(lbox), FadeIn(l1), run_time=0.5)
        self.play(FadeIn(l2, scale=1.2), run_time=0.6)
        self.wait(0.9)
        self.play(FadeIn(rbox), FadeIn(r1), run_time=0.5)
        self.play(FadeIn(r2, scale=1.2), run_time=0.6)
        self.wait(1.2)

        verdict = txt("Quitting was worth 20 more than participating.", 30, INK, BOLD
                      ).move_to([0, -1.20, 0])
        self.play(FadeIn(verdict, shift=UP * 0.15), run_time=0.8)
        self.wait(1.4)

        meas = txt("Measured over 50 evaluation episodes:", 25, MUTED).move_to([0, -1.90, 0])
        big = txt("100 %  ended in a deliberate blow-up at step 7 of 100", 30, BAD, BOLD
                  ).move_to([0, -2.42, 0])
        self.play(FadeIn(meas), run_time=0.5)
        self.play(FadeIn(big, scale=1.06), run_time=0.8)
        self.wait(2.0)

        self.play(FadeOut(VGroup(b1, b2, lbox, l1, l2, rbox, r1, r2, verdict, meas, big)),
                  run_time=0.7)
        f1 = txt("The algorithm was not broken.", 34, INK, BOLD)
        f2 = txt("It maximised the objective it was given, precisely.", 30, INK)
        f3 = txt("The objective was wrong.", 34, BAD, BOLD)
        VGroup(f1, f2, f3).arrange(DOWN, buff=0.40).move_to([0, 0.35, 0])
        self.play(FadeIn(f1), run_time=0.8); self.wait(0.9)
        self.play(FadeIn(f2), run_time=0.8); self.wait(0.9)
        self.play(FadeIn(f3, scale=1.08), run_time=0.9)
        self.wait(1.6)

        self.takeaway("Fixed by paying to survive every step -- so living beats quitting from every state.", OK)
        self.wait(2.4)


class S11_ReplayRatio(Slide):
    """A silent bug: every run was doing a quarter of its training."""

    def construct(self):
        self.header("A bug that produced no symptom", "the replay ratio")

        envs = VGroup()
        for i in range(4):
            b = RoundedRectangle(width=1.5, height=0.85, corner_radius=0.10,
                                 fill_color=PANEL, fill_opacity=1,
                                 stroke_color=AMBER, stroke_width=2.5)
            envs.add(VGroup(b, txt(f"env {i+1}", 21, AMBER).move_to(b.get_center())))
        envs.arrange(RIGHT, buff=0.35).move_to([-3.1, 1.65, 0])
        self.play(LaggedStart(*[FadeIn(e) for e in envs], lag_ratio=0.12), run_time=0.9)

        coll = txt("4 environments  →  4 transitions collected per iteration",
                   26, INK).move_to([-0.4, 0.75, 0])
        self.play(FadeIn(coll), run_time=0.7)
        self.wait(0.8)

        upd = RoundedRectangle(width=2.6, height=0.85, corner_radius=0.10,
                               fill_color=PANEL, fill_opacity=1,
                               stroke_color=BLUE, stroke_width=2.5).move_to([3.3, 1.65, 0])
        upd_l = txt("1 update", 24, BLUE, BOLD).move_to(upd.get_center())
        self.play(FadeIn(upd), FadeIn(upd_l), run_time=0.6)
        self.wait(1.0)

        # One measured quantity across three configurations, so one colour:
        # the comparison is carried by bar length. The broken row is called
        # out by a word and an outline, never by hue alone.
        rows = [
            ("as configured  (4 envs, 1 step)", 0.233, True),
            ("standard  (1 env, 1 step)", 0.933, False),
            ("fixed  (4 envs, gradient_steps = -1)", 0.933, False),
        ]
        unit = 4.6
        x0 = -0.7
        bars, labels, values, flags = VGroup(), VGroup(), VGroup(), VGroup()
        for i, (name, val, broken) in enumerate(rows):
            y = -0.30 - i * 0.78
            labels.add(at(txt(name, 23, INK), x0 - 0.28, y, RIGHT))
            b = bar(val * unit, 0.50, BLUE)
            b.move_to([x0 + val * unit / 2, y, 0])
            bars.add(b)
            values.add(at(txt(f"{val:.3f}", 24, INK, BOLD), x0 + val * unit + 0.24, y))
            if broken:
                ring = SurroundingRectangle(b, color=BAD, buff=0.09,
                                            stroke_width=3, corner_radius=0.08)
                tag = at(txt("the bug", 21, BAD, BOLD), x0 + val * unit + 1.55, y)
                flags.add(ring, tag)

        ax = axis_x(x0, x0 + unit, -2.60,
                    ticks=[(x0, "0"), (x0 + unit * 0.5, "0.5"), (x0 + unit, "1.0")])
        xlab = txt("gradient updates per environment step", 22, MUTED).move_to([x0 + unit / 2, -3.28, 0])

        self.play(FadeIn(labels), Create(ax), FadeIn(xlab), run_time=0.8)
        for i, (b, v) in enumerate(zip(bars, values)):
            tgt = b.width
            b.stretch_to_fit_width(0.02).align_to([x0, 0, 0], LEFT)
            self.play(b.animate.stretch_to_fit_width(tgt).align_to([x0, 0, 0], LEFT),
                      FadeIn(v), run_time=0.7)
            if i == 0:
                self.play(Create(flags[0]), FadeIn(flags[1]), run_time=0.5)
        self.wait(1.6)

        self.play(FadeOut(VGroup(envs, coll, upd, upd_l)), run_time=0.5)
        cons = txt("Every run had done about 233,000 gradient updates for a budget of 1,000,000 steps.\n"
                   "A quarter of the intended training -- and nothing on screen looked wrong.",
                   26, INK).move_to([0, 1.55, 0])
        self.play(FadeIn(cons), run_time=0.9)
        self.wait(2.8)
