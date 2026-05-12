from __future__ import annotations

import pygame
from typing import TYPE_CHECKING

from environment.task import Priority, Status
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    FONT_SIZE_LARGE, FONT_SIZE_MEDIUM, FONT_SIZE_SMALL,
    SIMULATION_DURATION,
)

if TYPE_CHECKING:
    from environment.scheduler_env import SchedulerEnvironment
    from metrics.tracker import MetricsTracker


#  Colour palette 
BG_DARK         = (15,  17,  26)
BG_PANEL        = (24,  28,  42)
BG_CARD         = (33,  38,  58)
BG_CARD_HOVER   = (44,  51,  77)
ACCENT          = (82, 130, 255)
ACCENT_DIM      = (50,  80, 160)
TEXT_PRIMARY    = (220, 225, 245)
TEXT_SECONDARY  = (140, 150, 180)
TEXT_DIM        = ( 80,  90, 120)
BORDER          = ( 40,  46,  70)
SEPARATOR       = ( 35,  42,  62)

# Priority colours (matching spec: LOW=green, MEDIUM=yellow, HIGH=orange, CRITICAL=red)
PRIORITY_COLOR = {
    Priority.LOW     : ( 80, 200, 120),
    Priority.MEDIUM  : (240, 200,  60),
    Priority.HIGH    : (240, 130,  40),
    Priority.CRITICAL: (220,  50,  50),
}

STATUS_COLOR = {
    Status.WAITING   : ACCENT_DIM,
    Status.RUNNING   : ( 50, 180, 230),
    Status.COMPLETED : ( 60, 200, 100),
    Status.FAILED    : (200,  50,  50),
}

WORKER_FREE_COLOR   = ( 40, 160, 100)
WORKER_BUSY_COLOR   = ( 80, 130, 220)
WORKER_FAILED_COLOR = (180,  50,  50)


class Renderer:
    """
    Manages all Pygame drawing.  Call render() once per tick.
    """

    #  layout constants 
    HEADER_H   = 60
    METRICS_H  = 130
    PANEL_Y    = HEADER_H + 8
    PANEL_H    = WINDOW_HEIGHT - HEADER_H - METRICS_H - 20
    QUEUE_W    = 380
    WORKERS_X  = QUEUE_W + 16
    WORKERS_W  = WINDOW_WIDTH - QUEUE_W - 24

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        pygame.font.init()
        self.font_lg  = pygame.font.SysFont("Segoe UI",  FONT_SIZE_LARGE,  bold=True)
        self.font_md  = pygame.font.SysFont("Segoe UI",  FONT_SIZE_MEDIUM, bold=False)
        self.font_sm  = pygame.font.SysFont("Segoe UI",  FONT_SIZE_SMALL,  bold=False)
        self.font_sm_b= pygame.font.SysFont("Segoe UI",  FONT_SIZE_SMALL,  bold=True)

    #  main entry point 

    def render(
        self,
        env              : "SchedulerEnvironment",
        tracker          : "MetricsTracker",
        scheduler_name   : str,
        paused           : bool = False,
    ) -> None:
        self.screen.fill(BG_DARK)

        self._draw_header(env, scheduler_name, paused)
        self._draw_queue_panel(env)
        self._draw_workers_panel(env)
        self._draw_metrics_panel(env, tracker)

        pygame.display.flip()

    #  header 

    def _draw_header(
        self,
        env           : "SchedulerEnvironment",
        scheduler_name: str,
        paused        : bool,
    ) -> None:
        rect = pygame.Rect(0, 0, WINDOW_WIDTH, self.HEADER_H)
        pygame.draw.rect(self.screen, BG_PANEL, rect)
        pygame.draw.line(
            self.screen, BORDER,
            (0, self.HEADER_H - 1), (WINDOW_WIDTH, self.HEADER_H - 1), 1
        )

        # Title
        title = self.font_lg.render(
            "Adaptive Task Scheduling  ·  Monte Carlo Tree Search", True, TEXT_PRIMARY
        )
        self.screen.blit(title, (16, 12))

        # Scheduler badge
        badge_txt = self.font_md.render(
            f"  Scheduler: {scheduler_name}  ", True, BG_DARK
        )
        bw = badge_txt.get_width() + 4
        bx = WINDOW_WIDTH - bw - 160
        pygame.draw.rect(self.screen, ACCENT, (bx, 15, bw, 28), border_radius=6)
        self.screen.blit(badge_txt, (bx + 2, 19))

        # Tick / progress
        pct = min(env.tick / max(1, SIMULATION_DURATION), 1.0)
        bar_x, bar_y, bar_w, bar_h = WINDOW_WIDTH - 145, 18, 130, 8
        pygame.draw.rect(self.screen, BG_CARD, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(
            self.screen, ACCENT,
            (bar_x, bar_y, int(bar_w * pct), bar_h), border_radius=4
        )
        tick_txt = self.font_sm.render(
            f"Tick {env.tick} / {SIMULATION_DURATION}", True, TEXT_SECONDARY
        )
        self.screen.blit(tick_txt, (WINDOW_WIDTH - 145, 30))

        if paused:
            p_txt = self.font_md.render("PAUSED", True, (240, 200, 60))
            self.screen.blit(p_txt, (WINDOW_WIDTH // 2 - 30, 18))

    #  queue panel 

    def _draw_queue_panel(self, env: "SchedulerEnvironment") -> None:
        px, py = 8, self.PANEL_Y
        pw, ph = self.QUEUE_W, self.PANEL_H

        # Panel background
        pygame.draw.rect(self.screen, BG_PANEL, (px, py, pw, ph), border_radius=8)
        pygame.draw.rect(self.screen, BORDER,   (px, py, pw, ph), width=1, border_radius=8)

        # Panel title
        heading = self.font_md.render(
            f"⏳  Waiting Queue  ({len(env.waiting_tasks)} tasks)", True, TEXT_PRIMARY
        )
        self.screen.blit(heading, (px + 12, py + 10))
        pygame.draw.line(
            self.screen, SEPARATOR,
            (px + 8, py + 34), (px + pw - 8, py + 34), 1
        )

        # Task cards
        cy       = py + 42
        card_h   = 50
        card_gap = 4
        visible  = (ph - 44) // (card_h + card_gap)
        tasks    = env.waiting_tasks[:visible]

        for task in tasks:
            self._draw_task_card(task, env.tick, px + 8, cy, pw - 16, card_h)
            cy += card_h + card_gap

        # "more tasks" hint
        overflow = len(env.waiting_tasks) - visible
        if overflow > 0:
            more = self.font_sm.render(
                f"  … and {overflow} more task(s)", True, TEXT_DIM
            )
            self.screen.blit(more, (px + 12, cy + 2))

    def _draw_task_card(
        self,
        task, current_tick: int,
        x: int, y: int, w: int, h: int,
    ) -> None:
        pri_color = PRIORITY_COLOR[task.priority]

        # Card background
        pygame.draw.rect(self.screen, BG_CARD, (x, y, w, h), border_radius=6)
        # Priority stripe on the left
        pygame.draw.rect(self.screen, pri_color, (x, y, 4, h), border_radius=3)

        # Task ID + priority
        id_txt = self.font_sm_b.render(f"T{task.task_id}", True, TEXT_PRIMARY)
        self.screen.blit(id_txt, (x + 10, y + 6))

        pri_txt = self.font_sm.render(task.priority.name, True, pri_color)
        self.screen.blit(pri_txt, (x + 10, y + 26))

        # Duration / remaining
        dur_txt = self.font_sm.render(
            f"dur {task.duration}t", True, TEXT_SECONDARY
        )
        self.screen.blit(dur_txt, (x + 80, y + 6))

        # Deadline slack
        slack = task.ticks_until_deadline(current_tick)
        slack_color = (
            (220, 50, 50) if slack < 5 else
            (240, 180, 40) if slack < 15 else
            TEXT_SECONDARY
        )
        dl_txt = self.font_sm.render(f"⏰ {slack}t left", True, slack_color)
        self.screen.blit(dl_txt, (x + 80, y + 26))

        # Urgency bar (right side)
        urgency = min(task.urgency_score(current_tick) / 40.0, 1.0)
        bx = x + w - 70
        pygame.draw.rect(self.screen, BORDER, (bx, y + 16, 60, 8), border_radius=4)
        pygame.draw.rect(
            self.screen, pri_color,
            (bx, y + 16, int(60 * urgency), 8), border_radius=4
        )
        urg_label = self.font_sm.render("urgency", True, TEXT_DIM)
        self.screen.blit(urg_label, (bx, y + 28))

    #  workers panel 

    def _draw_workers_panel(self, env: "SchedulerEnvironment") -> None:
        px = self.WORKERS_X
        py = self.PANEL_Y
        pw = self.WORKERS_W
        ph = self.PANEL_H

        pygame.draw.rect(self.screen, BG_PANEL, (px, py, pw, ph), border_radius=8)
        pygame.draw.rect(self.screen, BORDER,   (px, py, pw, ph), width=1, border_radius=8)

        heading = self.font_md.render("🖥️  Workers", True, TEXT_PRIMARY)
        self.screen.blit(heading, (px + 12, py + 10))
        pygame.draw.line(
            self.screen, SEPARATOR,
            (px + 8, py + 34), (px + pw - 8, py + 34), 1
        )

        card_h   = 110
        card_gap = 10
        cy       = py + 44

        for worker in env.workers:
            self._draw_worker_card(worker, env.tick, px + 8, cy, pw - 16, card_h)
            cy += card_h + card_gap

        # Completed count badge
        done_txt = self.font_md.render(
            f"✓  Completed: {len(env.completed)}", True, (80, 220, 130)
        )
        self.screen.blit(done_txt, (px + 12, cy + 6))
        fail_txt = self.font_sm.render(
            f"✗  Failed: {len(env.failed)}", True, (220, 80, 80)
        )
        self.screen.blit(fail_txt, (px + 12, cy + 30))

    def _draw_worker_card(
        self,
        worker, current_tick: int,
        x: int, y: int, w: int, h: int,
    ) -> None:
        if worker.failed:
            border_color = WORKER_FAILED_COLOR
            state_txt    = "OFFLINE"
            state_color  = WORKER_FAILED_COLOR
        elif worker.is_busy:
            border_color = WORKER_BUSY_COLOR
            state_txt    = "BUSY"
            state_color  = WORKER_BUSY_COLOR
        else:
            border_color = WORKER_FREE_COLOR
            state_txt    = "FREE"
            state_color  = WORKER_FREE_COLOR

        pygame.draw.rect(self.screen, BG_CARD, (x, y, w, h), border_radius=8)
        pygame.draw.rect(
            self.screen, border_color,
            (x, y, w, h), width=2, border_radius=8
        )

        # Worker ID
        wid_txt = self.font_md.render(f"Worker {worker.worker_id}", True, TEXT_PRIMARY)
        self.screen.blit(wid_txt, (x + 12, y + 10))

        # State badge
        s_surf = self.font_sm_b.render(f" {state_txt} ", True, BG_DARK)
        sw = s_surf.get_width() + 4
        pygame.draw.rect(
            self.screen, state_color,
            (x + w - sw - 10, y + 8, sw, 22), border_radius=5
        )
        self.screen.blit(s_surf, (x + w - sw - 8, y + 11))

        # Current task details
        if worker.current_task is not None:
            task = worker.current_task
            pri_color = PRIORITY_COLOR[task.priority]

            t_txt = self.font_sm_b.render(
                f"Task T{task.task_id}  [{task.priority.name}]", True, pri_color
            )
            self.screen.blit(t_txt, (x + 12, y + 38))

            # Progress bar
            done_frac = max(0.0, 1.0 - task.remaining / max(1, task.duration))
            bar_w_full = w - 24
            pygame.draw.rect(
                self.screen, BORDER,
                (x + 12, y + 60, bar_w_full, 10), border_radius=5
            )
            pygame.draw.rect(
                self.screen, state_color,
                (x + 12, y + 60, int(bar_w_full * done_frac), 10), border_radius=5
            )
            prog_txt = self.font_sm.render(
                f"{task.remaining}t left  /  deadline slack {task.ticks_until_deadline(current_tick)}t",
                True, TEXT_SECONDARY,
            )
            self.screen.blit(prog_txt, (x + 12, y + 76))

        elif worker.failed:
            rec_txt = self.font_sm.render(
                f"Recovery in {worker.recovery_ticks} ticks", True, WORKER_FAILED_COLOR
            )
            self.screen.blit(rec_txt, (x + 12, y + 44))
        else:
            idle_txt = self.font_sm.render("Waiting for task…", True, TEXT_DIM)
            self.screen.blit(idle_txt, (x + 12, y + 44))

        # Utilisation stat
        busy_frac = (
            worker.total_ticks_busy / max(1, current_tick)
        )
        util_txt = self.font_sm.render(
            f"utilisation {busy_frac*100:.0f}%   done {worker.tasks_completed}",
            True, TEXT_DIM,
        )
        self.screen.blit(util_txt, (x + 12, y + h - 18))

    #  metrics panel 

    def _draw_metrics_panel(
        self,
        env    : "SchedulerEnvironment",
        tracker: "MetricsTracker",
    ) -> None:
        my = WINDOW_HEIGHT - self.METRICS_H - 4
        mw = WINDOW_WIDTH - 16

        pygame.draw.rect(self.screen, BG_PANEL, (8, my, mw, self.METRICS_H), border_radius=8)
        pygame.draw.rect(self.screen, BORDER,   (8, my, mw, self.METRICS_H), width=1, border_radius=8)

        heading = self.font_md.render("📊  Real-Time Metrics", True, TEXT_PRIMARY)
        self.screen.blit(heading, (20, my + 10))
        pygame.draw.line(
            self.screen, SEPARATOR,
            (12, my + 32), (mw + 4, my + 32), 1
        )

        summary = tracker.summary()
        cols = [
            ("Completed",       str(summary["total_completed"]),          (80, 220, 130)),
            ("Failed",          str(summary["total_failed"]),              (220, 80, 80)),
            ("Miss Rate",       f"{summary['deadline_miss_rate']*100:.1f}%", (240, 180, 60)),
            ("Avg Wait",        f"{summary['avg_waiting_time']:.1f}t",    TEXT_SECONDARY),
            ("Utilisation",     f"{summary['worker_utilisation']*100:.1f}%", ACCENT),
            ("Avg Queue",       f"{summary['avg_queue_length']:.1f}",     TEXT_SECONDARY),
            ("Cum. Reward",     f"{summary['cumulative_reward']:.1f}",
             (80, 220, 130) if summary["cumulative_reward"] >= 0 else (220, 80, 80)),
        ]

        col_w = mw // len(cols)
        for i, (label, value, color) in enumerate(cols):
            cx = 16 + i * col_w
            lbl = self.font_sm.render(label, True, TEXT_DIM)
            val = self.font_lg.render(value, True, color)
            self.screen.blit(lbl, (cx, my + 38))
            self.screen.blit(val, (cx, my + 56))

        # Keyboard hint
        hint = self.font_sm.render(
            "Space: pause/resume  |  1: FIFO  |  2: Priority  |  3: MCTS  |  R: reset  |  ESC: quit",
            True, TEXT_DIM,
        )
        self.screen.blit(hint, (16, my + self.METRICS_H - 20))
