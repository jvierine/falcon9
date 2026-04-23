import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

import plot_fragments as pf


def load_fragment_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    os.chdir(script_dir)
    try:
        _, _, fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = pf.get_fragments()
    finally:
        os.chdir(cwd)
    return fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times


class FragmentMergeSelector:
    def __init__(self, fragment_ids, fragment_geo_pos, fragment_times, save_file, preselect=None):
        self.fragment_ids = list(fragment_ids)
        self.fragment_geo_pos = [np.asarray(geo) for geo in fragment_geo_pos]
        self.fragment_times = [np.asarray(t, dtype=float) for t in fragment_times]
        self.save_file = save_file
        self.preselect = [] if preselect is None else list(preselect)

        self.selected_ids = []
        self.artists = {}
        self.selected_color_map = plt.get_cmap("tab10")
        self.default_line_color = "0.75"
        self.default_text_color = "0.35"

        self.t0 = np.min([np.min(t) for t in self.fragment_times if len(t) > 0])

        self.fig, (self.ax_lon_alt, self.ax_time_alt) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        plt.subplots_adjust(bottom=0.14, right=0.82)

        self.selection_text = self.fig.text(0.83, 0.90, "", va="top", ha="left", fontsize=10, family="monospace")
        self.help_text = self.fig.text(
            0.83,
            0.55,
            "Left click a trajectory or label to toggle.\n"
            "Buttons or keys:\n"
            "  c  clear selection\n"
            "  p  print selection\n"
            "  s  save selection\n"
            "  q  close",
            va="top",
            ha="left",
            fontsize=9,
        )

        self._plot_fragments()
        self._add_buttons()
        self._apply_preselect()
        self._update_status()

        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _plot_fragments(self):
        for i, fid in enumerate(self.fragment_ids):
            geo = self.fragment_geo_pos[i]
            times = self.fragment_times[i]
            if len(geo) == 0:
                continue

            lon = geo[:, 1]
            alt_km = geo[:, 2] / 1e3
            time_rel = times - self.t0

            lon_alt_line, = self.ax_lon_alt.plot(
                lon,
                alt_km,
                ".-",
                color=self.default_line_color,
                linewidth=0.9,
                markersize=3,
                alpha=0.7,
                picker=5,
                zorder=1,
            )
            time_alt_line, = self.ax_time_alt.plot(
                time_rel,
                alt_km,
                ".-",
                color=self.default_line_color,
                linewidth=0.9,
                markersize=3,
                alpha=0.7,
                picker=5,
                zorder=1,
            )

            lon_alt_text = self.ax_lon_alt.text(
                lon[0],
                alt_km[0],
                fid,
                color=self.default_text_color,
                fontsize=8,
                picker=True,
                zorder=2,
            )
            time_alt_text = self.ax_time_alt.text(
                time_rel[0],
                alt_km[0],
                fid,
                color=self.default_text_color,
                fontsize=8,
                picker=True,
                zorder=2,
            )

            for artist in [lon_alt_line, time_alt_line, lon_alt_text, time_alt_text]:
                artist.set_gid(fid)

            self.artists[fid] = {
                "lon_alt_line": lon_alt_line,
                "time_alt_line": time_alt_line,
                "lon_alt_text": lon_alt_text,
                "time_alt_text": time_alt_text,
            }

        self.ax_lon_alt.set_xlabel("Longitude (deg)")
        self.ax_lon_alt.set_ylabel("Altitude (km)")
        self.ax_lon_alt.set_title("Fragments: longitude vs altitude")
        self.ax_lon_alt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        self.ax_time_alt.set_xlabel("Time since first detection (s)")
        self.ax_time_alt.set_title("Fragments: time vs altitude")
        self.ax_time_alt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    def _add_buttons(self):
        save_ax = self.fig.add_axes([0.12, 0.04, 0.12, 0.06])
        print_ax = self.fig.add_axes([0.27, 0.04, 0.12, 0.06])
        clear_ax = self.fig.add_axes([0.42, 0.04, 0.12, 0.06])
        close_ax = self.fig.add_axes([0.57, 0.04, 0.12, 0.06])

        self.save_button = Button(save_ax, "Save")
        self.print_button = Button(print_ax, "Print")
        self.clear_button = Button(clear_ax, "Clear")
        self.close_button = Button(close_ax, "Close")

        self.save_button.on_clicked(self.save_selection)
        self.print_button.on_clicked(self.print_selection)
        self.clear_button.on_clicked(self.clear_selection)
        self.close_button.on_clicked(self.close)

    def _apply_preselect(self):
        for fid in self.preselect:
            if fid in self.artists and fid not in self.selected_ids:
                self.selected_ids.append(fid)
        self._refresh_artists()

    def _color_for_selected(self, fid):
        idx = self.selected_ids.index(fid) % 10
        return self.selected_color_map(idx)

    def _refresh_artists(self):
        for fid in self.fragment_ids:
            if fid not in self.artists:
                continue

            item = self.artists[fid]
            if fid in self.selected_ids:
                color = self._color_for_selected(fid)
                linewidth = 2.4
                markersize = 5
                alpha = 0.95
                zorder = 10
                fontweight = "bold"
                bbox = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, linewidth=0.8, alpha=0.9)
            else:
                color = self.default_line_color
                linewidth = 0.9
                markersize = 3
                alpha = 0.7
                zorder = 1
                fontweight = "normal"
                bbox = None

            for key in ["lon_alt_line", "time_alt_line"]:
                item[key].set_color(color)
                item[key].set_linewidth(linewidth)
                item[key].set_markersize(markersize)
                item[key].set_alpha(alpha)
                item[key].set_zorder(zorder)

            for key in ["lon_alt_text", "time_alt_text"]:
                item[key].set_color(color if fid in self.selected_ids else self.default_text_color)
                item[key].set_fontweight(fontweight)
                item[key].set_bbox(bbox)
                item[key].set_zorder(zorder + 1)

        self.fig.canvas.draw_idle()

    def _selection_list_text(self):
        if len(self.selected_ids) == 0:
            return "Selected IDs:\n  []"

        lines = ["Selected IDs:"]
        for i, fid in enumerate(self.selected_ids, start=1):
            lines.append("  %d. %s" % (i, fid))
        lines.append("")
        lines.append("Python list:")
        lines.append("  %s" % repr(self.selected_ids))
        return "\n".join(lines)

    def _update_status(self):
        self.selection_text.set_text(self._selection_list_text())
        self.fig.canvas.draw_idle()

    def toggle_fragment(self, fid):
        if fid in self.selected_ids:
            self.selected_ids.remove(fid)
        else:
            self.selected_ids.append(fid)
        self._refresh_artists()
        self._update_status()

    def on_pick(self, event):
        artist = event.artist
        fid = artist.get_gid()
        if fid is None:
            return
        self.toggle_fragment(fid)

    def on_key(self, event):
        if event.key == "c":
            self.clear_selection()
        elif event.key == "p":
            self.print_selection()
        elif event.key == "s":
            self.save_selection()
        elif event.key in ["q", "enter"]:
            self.close()

    def print_selection(self, event=None):
        print("merge_ids=%s" % (repr(self.selected_ids)))

    def save_selection(self, event=None):
        out = {
            "merge_ids": self.selected_ids,
        }
        with open(self.save_file, "w") as f:
            json.dump(out, f, indent=2)
        print("saved %s" % (self.save_file))
        self._update_status()

    def clear_selection(self, event=None):
        self.selected_ids = []
        self._refresh_artists()
        self._update_status()

    def close(self, event=None):
        plt.close(self.fig)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Interactive GUI for selecting fragment IDs to merge."
    )
    parser.add_argument(
        "--save-file",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "selected_merge_ids.json"),
        help="Path to JSON file written by the Save button.",
    )
    parser.add_argument(
        "--preselect",
        nargs="*",
        default=[],
        help="Fragment IDs to highlight when the GUI starts.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the GUI without opening an interactive window.",
    )
    parser.add_argument(
        "--figure-file",
        default=None,
        help="Optional path for saving the initial figure layout.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    fragment_ids, fragment_pos, fragment_pos_err, fragment_geo_pos, fragment_times = load_fragment_data()
    _ = fragment_pos
    _ = fragment_pos_err

    gui = FragmentMergeSelector(
        fragment_ids=fragment_ids,
        fragment_geo_pos=fragment_geo_pos,
        fragment_times=fragment_times,
        save_file=args.save_file,
        preselect=args.preselect,
    )

    if args.figure_file is not None:
        gui.fig.savefig(args.figure_file, bbox_inches="tight")

    if args.no_show:
        return

    plt.show()


if __name__ == "__main__":
    main()
