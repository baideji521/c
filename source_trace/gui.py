"""图形界面（tkinter，Python 自带，无需额外依赖）。

GUI 只是 ``source_trace.pipeline`` 的前端：核心算法不依赖界面，
关闭 GUI 后 CLI / Python API 依然可独立运行。

启动：
    python -m source_trace.gui
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .config import SourceTraceConfig
from .pipeline import TraceResult, run_trace
from .utils.log import get_logger, setup_logging


class _QueueHandler(logging.Handler):
    """把日志投递到 Tk 主线程的队列。"""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(("log", self.format(record)))
        except Exception:
            pass


class SourceTraceGUI:
    STATUS_COLOR = {
        "HIGH": "#1a7f37",
        "MEDIUM": "#9a6700",
        "LOW": "#bc4c00",
        "UNKNOWN": "#82071e",
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("成片原素材反溯工具  Source Trace")
        self.root.geometry("1080x760")
        self.msg_q: queue.Queue = queue.Queue()
        self.result: TraceResult | None = None
        self.worker: threading.Thread | None = None
        self.sources_dir: Path | None = None
        self.query_path: Path | None = None

        self._build_ui()
        setup_logging()
        handler = _QueueHandler(self.msg_q)
        handler.setFormatter(logging.Formatter("%(message)s"))
        get_logger().addHandler(handler)
        self.root.after(120, self._poll)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}

        top = ttk.LabelFrame(self.root, text="输入")
        top.pack(fill="x", **pad)

        ttk.Label(top, text="成片：").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.query_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.query_var, width=90).grid(row=0, column=1, sticky="we", pady=6)
        ttk.Button(top, text="选择文件", command=self._pick_query).grid(row=0, column=2, padx=6)

        ttk.Label(top, text="原素材目录：").grid(row=1, column=0, sticky="w", padx=6)
        self.sources_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.sources_var, width=90).grid(row=1, column=1, sticky="we")
        ttk.Button(top, text="选择文件夹", command=self._pick_sources).grid(row=1, column=2, padx=6)

        ttk.Label(top, text="输出目录：").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        self.output_var = tk.StringVar(value=str(Path.cwd() / "output" / "gui"))
        ttk.Entry(top, textvariable=self.output_var, width=90).grid(row=2, column=1, sticky="we", pady=6)
        ttk.Button(top, text="选择文件夹", command=self._pick_output).grid(row=2, column=2, padx=6)
        top.columnconfigure(1, weight=1)

        opt = ttk.LabelFrame(self.root, text="选项")
        opt.pack(fill="x", **pad)
        self.model_var = tk.StringVar(value="isc21")
        ttk.Label(opt, text="描述子：").pack(side="left", padx=6)
        ttk.Combobox(opt, textvariable=self.model_var, values=["isc21", "colorhist"], width=10,
                     state="readonly").pack(side="left")
        self.method_var = tk.StringVar(value="tn")
        ttk.Label(opt, text="时序对齐：").pack(side="left", padx=6)
        ttk.Combobox(opt, textvariable=self.method_var, values=["tn", "dp", "dtw", "hv", "none"], width=8,
                     state="readonly").pack(side="left")
        self.device_var = tk.StringVar(value="auto")
        ttk.Label(opt, text="设备：").pack(side="left", padx=6)
        ttk.Combobox(opt, textvariable=self.device_var, values=["auto", "cuda", "cpu"], width=8,
                     state="readonly").pack(side="left")
        self.clip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="自动裁剪原始片段", variable=self.clip_var).pack(side="left", padx=10)
        self.verify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="生成验证图", variable=self.verify_var).pack(side="left")
        self.tta_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="困难样本 TTA", variable=self.tta_var).pack(side="left", padx=10)

        act = ttk.Frame(self.root)
        act.pack(fill="x", **pad)
        self.run_btn = ttk.Button(act, text="开始反溯", command=self._start)
        self.run_btn.pack(side="left", padx=6)
        ttk.Button(act, text="播放成片", command=self._play_query).pack(side="left", padx=4)
        ttk.Button(act, text="播放原素材", command=self._play_source).pack(side="left", padx=4)
        ttk.Button(act, text="打开输出目录", command=self._open_output).pack(side="left", padx=4)
        ttk.Button(act, text="导出全部原始片段", command=self._export_clips).pack(side="left", padx=4)

        prog = ttk.Frame(self.root)
        prog.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True)
        self.stage_var = tk.StringVar(value="就绪")
        ttk.Label(prog, textvariable=self.stage_var, width=28).pack(side="left", padx=8)

        mid = ttk.LabelFrame(self.root, text="反溯结果")
        mid.pack(fill="both", expand=True, **pad)
        cols = ("id", "query", "source", "source_time", "visual", "temporal", "conf", "status")
        heads = ("#", "成片时间", "来源素材", "原始时间", "相似度", "时序得分", "Confidence", "状态")
        widths = (40, 150, 170, 170, 80, 80, 100, 90)
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=12)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        for status, color in self.STATUS_COLOR.items():
            self.tree.tag_configure(status, foreground=color)

        bot = ttk.LabelFrame(self.root, text="运行日志")
        bot.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(bot, height=10, wrap="none")
        self.log_text.pack(side="left", fill="both", expand=True)
        sb2 = ttk.Scrollbar(bot, orient="vertical", command=self.log_text.yview)
        sb2.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=sb2.set, state="disabled")

    # ------------------------------------------------------------- 事件
    def _pick_query(self) -> None:
        p = filedialog.askopenfilename(
            title="选择成片",
            filetypes=[("视频", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v"), ("全部文件", "*.*")],
        )
        if p:
            self.query_var.set(p)

    def _pick_sources(self) -> None:
        p = filedialog.askdirectory(title="选择原始素材目录")
        if p:
            self.sources_var.set(p)
            self._list_sources(Path(p))

    def _pick_output(self) -> None:
        p = filedialog.askdirectory(title="选择输出目录")
        if p:
            self.output_var.set(p)

    def _list_sources(self, d: Path) -> None:
        from .config import VIDEO_EXTS

        files = [p.name for p in sorted(d.iterdir()) if p.suffix.lower() in VIDEO_EXTS]
        self._append_log(f"素材目录 {d} 共 {len(files)} 个视频：" + "、".join(files[:10]))

    def _start(self) -> None:
        q, s = self.query_var.get().strip(), self.sources_var.get().strip()
        if not q or not Path(q).is_file():
            messagebox.showwarning("提示", "请选择有效的成片文件")
            return
        if not s or not Path(s).is_dir():
            messagebox.showwarning("提示", "请选择有效的原始素材目录")
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("提示", "任务正在运行中")
            return

        self.query_path = Path(q)
        self.sources_dir = Path(s)
        self.tree.delete(*self.tree.get_children())
        self.progress.configure(value=0)
        self.run_btn.configure(state="disabled")
        self.stage_var.set("启动中...")

        cfg = SourceTraceConfig()
        cfg.feature.model = self.model_var.get()
        cfg.feature.device = self.device_var.get()
        cfg.alignment.method = self.method_var.get()
        cfg.refinement.tta = bool(self.tta_var.get())
        cfg.export.contact_sheet = bool(self.verify_var.get())
        cfg.export.compare_video = bool(self.verify_var.get())

        out = Path(self.output_var.get())
        do_clips = bool(self.clip_var.get())
        do_verify = bool(self.verify_var.get())

        def _job() -> None:
            try:
                res = run_trace(
                    query=self.query_path,
                    sources=self.sources_dir,
                    output=out,
                    cfg=cfg,
                    export_clips=do_clips,
                    verify=do_verify,
                    progress=lambda stage, done, total: self.msg_q.put(("progress", (stage, done, total))),
                )
                self.msg_q.put(("done", res))
            except Exception as exc:  # noqa: BLE001 - GUI 需要展示任何异常
                self.msg_q.put(("error", f"{type(exc).__name__}: {exc}"))

        self.worker = threading.Thread(target=_job, daemon=True)
        self.worker.start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "progress":
                    stage, done, total = payload
                    pct = 100.0 * done / max(total, 1)
                    self.progress.configure(value=pct)
                    self.stage_var.set(f"{stage}  {done}/{total}")
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self.run_btn.configure(state="normal")
                    self.stage_var.set("失败")
                    messagebox.showerror("运行失败", str(payload))
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _on_done(self, res: TraceResult) -> None:
        self.result = res
        self.run_btn.configure(state="normal")
        self.progress.configure(value=100)
        self.stage_var.set(f"完成（{res.timings.get('total', 0):.1f}s）")
        for s in res.segments:
            src_time = (
                "-" if s.source_start is None else f"{s.source_start:.2f} – {s.source_end:.2f}"
            )
            self.tree.insert(
                "",
                "end",
                values=(
                    s.id,
                    f"{s.query_start:.2f} – {s.query_end:.2f}",
                    s.source or "UNKNOWN",
                    src_time,
                    f"{s.scores.get('visual_score', 0.0):.3f}",
                    f"{s.scores.get('temporal_score', 0.0):.3f}",
                    f"{s.confidence:.3f}",
                    s.status,
                ),
                tags=(s.status,),
            )

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ------------------------------------------------------------- 播放/打开
    def _selected_segment(self):
        sel = self.tree.selection()
        if not sel or self.result is None:
            return None
        sid = int(self.tree.item(sel[0], "values")[0])
        for s in self.result.segments:
            if s.id == sid:
                return s
        return None

    def _open_path(self, path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(path))  # noqa: S606 - 由用户主动触发
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("打开失败", f"{path}\n{exc}")

    def _play_query(self) -> None:
        if self.query_path and self.query_path.exists():
            self._open_path(self.query_path)
        else:
            messagebox.showinfo("提示", "请先选择成片")

    def _play_source(self) -> None:
        seg = self._selected_segment()
        if seg is None or not seg.source or self.sources_dir is None:
            messagebox.showinfo("提示", "请先在结果表中选中一个有来源的片段")
            return
        self._open_path(self.sources_dir / seg.source)

    def _open_output(self) -> None:
        out = Path(self.output_var.get())
        if out.exists():
            self._open_path(out)
        else:
            messagebox.showinfo("提示", "输出目录尚未生成")

    def _export_clips(self) -> None:
        if self.result is None:
            messagebox.showinfo("提示", "请先完成一次反溯")
            return
        from .config import ExportConfig
        from .export.ffmpeg import clip_name, export_clip

        cfg = ExportConfig()
        out_dir = Path(self.output_var.get()) / "segments"
        n = 0
        for s in self.result.segments:
            if s.status == "UNKNOWN" or not s.source or s.source_start is None or self.sources_dir is None:
                continue
            name = clip_name(s.id, s.source, s.source_start, s.source_end or s.source_start)
            r = export_clip(self.sources_dir / s.source, s.source_start, s.source_end or 0.0, out_dir / name, cfg)
            n += int(r.ok)
        self._append_log(f"已导出 {n} 个原始片段 -> {out_dir}")
        messagebox.showinfo("完成", f"已导出 {n} 个原始片段：\n{out_dir}")


def main() -> int:
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.2)
    except Exception:
        pass
    SourceTraceGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
