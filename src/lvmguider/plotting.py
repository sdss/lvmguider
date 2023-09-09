#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-03
# @Filename: plotting.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

from typing import TYPE_CHECKING, Callable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.text
import matplotlib.transforms as mtransforms
import numpy
import seaborn
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

from lvmguider import config


if TYPE_CHECKING:
    import pandas
    from matplotlib.figure import Figure

    from lvmguider.dataclasses import CoAdd_CameraSolution, GlobalSolution


matplotlib.use("agg")


AnyPath = str | os.PathLike


def plot_qa(
    solution: GlobalSolution,
    root: AnyPath | None = None,
    save_subplots: bool = True,
):
    """Produces guider QA plots from a global solution object."""

    root = pathlib.Path(root or (solution.path.parent / "qa"))
    root.mkdir(parents=True, exist_ok=True)

    stem = solution.path.stem

    # Set up seaborn and matplotlib.
    seaborn.set_theme(
        style="darkgrid",
        palette="deep",
        font="serif",
        font_scale=1.2,  # type: ignore
    )

    with plt.ioff():
        # Plot PA
        outpath_pa = root / (stem + "_pa.pdf")
        plot_position_angle(outpath_pa, solution, save_subplots=save_subplots)

        # Plot ZP
        outpath_zp = root / (stem + "_zp.pdf")
        plot_zero_point_or_fwhm(
            outpath_zp,
            solution,
            save_subplots=save_subplots,
            column="zero_point",
        )

        # Plot FWHM
        outpath_fwhm = root / (stem + "_fwhm.pdf")
        plot_zero_point_or_fwhm(
            outpath_fwhm,
            solution,
            save_subplots=save_subplots,
            column="fwhm",
        )

        # Plot guider offsets
        outpath_fwhm = root / (stem + "_guide_offsets.pdf")
        plot_guider_offsets(
            outpath_fwhm,
            solution,
            save_subplots=save_subplots,
        )

    seaborn.reset_orig()


def get_subplot_path(
    orig_path: AnyPath,
    suffix: str = "",
    extension="png",
    subdir: str = "subplots",
):
    """Creates a path for a subplot."""

    orig_path = pathlib.Path(orig_path)

    orig_parent = orig_path.parent
    stem = orig_path.stem

    path = orig_parent

    if subdir:
        path /= subdir

    return path / (stem + suffix + f".{extension}")


def create_subplot(func: Callable, path: AnyPath, *args, dpi: int = 100, **kwargs):
    """Creates a subplot by calling a plotting function."""

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    func(ax, *args, **kwargs)
    fig.savefig(str(path), dpi=dpi)


def save_subplot(
    fig: Figure,
    axes: Axes | list[Axes],
    path: AnyPath,
    pad: float | Sequence[float] = (0.01, 0.0),
    bbox_range: Sequence[float] | None = None,
):
    """Saves a subplot, including axes labels, tick labels, and titles."""

    # https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib

    if isinstance(axes, Axes):
        axes = [axes]

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    if bbox_range is not None:
        fig.savefig(
            str(path),
            bbox_inches=mtransforms.Bbox(
                # This is in "figure fraction" for the bottom half
                # input in [[xmin, ymin], [xmax, ymax]]
                numpy.array(
                    [
                        [bbox_range[0], bbox_range[1]],
                        [bbox_range[2], bbox_range[3]],
                    ]
                )
            ).transformed(
                (fig.transFigure - fig.dpi_scale_trans)  # type: ignore
            ),
        )
        return

    items = []
    # For text objects, we need to draw the figure first, otherwise
    # the extents are undefined.
    for ax in axes:
        ax.figure.canvas.draw()

        items += ax.get_xticklabels()
        items += ax.get_yticklabels()
        items += [ax.get_xaxis().get_label(), ax.get_yaxis().get_label()]
        items += [ax, ax.title]  # type: ignore

    bbox = Bbox.union([item.get_window_extent() for item in items])

    if isinstance(pad, Sequence):
        bbox = bbox.expanded(1.0 + pad[0], 1.0 + pad[1])
    else:
        bbox = bbox.expanded(1.0 + pad, 1.0 + pad)

    extent = bbox.transformed(fig.dpi_scale_trans.inverted())  # type: ignore

    fig.savefig(str(path), bbox_inches=extent)


def get_camera_figure(
    left: float = 0.1,
    right: float = 0.9,
    bottom: float = 0.075,
    top: float = 0.91,
    wspace: float = 0.1,
    hspace: float = 0.2,
):
    """Returns a customised figure and axes."""

    fig, axd = plt.subplot_mosaic(
        [["east", "west"], ["global", "global"]],
        figsize=(11, 8),
    )
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=wspace,
        hspace=hspace,
    )

    for name, ax in axd.items():
        ax.axis("off")

        ax.set_xlabel("Frame number")

        if name == "global":
            ax.set_title("Full frame")
        else:
            ax.set_title(f"Camera {str(name).capitalize()}")

        ax.ticklabel_format(useOffset=False)
        ax.xaxis.get_major_locator().set_params(integer=True)

    return fig, axd


def plot_position_angle(
    outpath: pathlib.Path,
    solution: GlobalSolution,
    save_subplots: bool = False,
):
    """Plots the position angle."""

    fig, axd = get_camera_figure(wspace=0.4)

    # Plot top panels with the East and West camera PA and PA error.
    for camera_solution in solution.coadd_solutions:
        if camera_solution.frames is None or len(camera_solution.frames) == 0:
            continue

        camera = camera_solution.camera

        frame_data = camera_solution.frame_data()
        reprocessed = camera_solution.frames[0].reprocessed

        if not reprocessed:
            frame_data = frame_data.loc[frame_data.wcs_mode == "gaia"]

        if len(frame_data.pa.dropna()) < 2:
            continue

        # Unhide plot.
        ax = axd[camera]  # type: ignore
        ax.axis("on")

        # Plot PA.
        axes = _plot_pa_axes(
            ax,
            frame_data,
            pa_error_mode="mean",
            legend=False,
        )

        # Remove the inside labels since they are identical left and right.
        if camera == "east":
            axes[1].set_ylabel("")
        else:
            axes[0].set_ylabel("")

        ax.set_xlabel("")

        if save_subplots:
            create_subplot(
                _plot_pa_axes,
                get_subplot_path(outpath, f"_{camera}"),
                frame_data,
                title=f"Camera {camera.capitalize()}",
                pa_error_mode="mean",
                legend=True,
            )

    # Now use the PAs from the guider data (i.e., full frame).
    guider_data = solution.guider_data()
    guider_data = guider_data.loc[guider_data.guide_mode == "guide"]

    if len(guider_data.pa.dropna()) >= 2:
        ax = axd["global"]  # type: ignore
        ax.axis("on")

        _plot_pa_axes(ax, guider_data, pa_error_mode="mean", legend=True)

        if save_subplots:
            create_subplot(
                _plot_pa_axes,
                get_subplot_path(outpath),
                guider_data,
                title="Full frame",
                pa_error_mode="mean",
                legend=True,
            )

    fig.suptitle(f"Position angle for {solution.path.name}")
    fig.savefig(str(outpath))

    plt.close("all")


def _plot_pa_axes(
    ax: Axes,
    data: pandas.DataFrame,
    title: str = "",
    pa_error_mode: str = "first",
    legend: bool = True,
):
    """Plot PA axes."""

    ax.ticklabel_format(useOffset=True)

    pa_threshold = config["coadds"]["warnings"]["pa_error"]

    # Create right axis for the PA error.
    right_ax = ax.twinx()
    right_ax.ticklabel_format(useOffset=True)

    # Plot absolute position angle.
    (pa_plot,) = ax.plot(
        data.frameno,
        data.pa,
        "b-",
        zorder=20,
        label="Position angle",
    )

    # For the cameras we calculate the error wrt the mean PA.
    # For the full frame PAs we use the initial PA and calculate the "drift".
    if pa_error_mode == "first":
        ref_pa = data.pa.dropna().iloc[0]
    else:
        ref_pa = data.pa.dropna().mean()

    # Plot PA error.
    (pa_error_plot,) = right_ax.plot(
        data.frameno,
        ref_pa - data.pa,
        "g-",
        zorder=20,
        label="PA error",
    )

    # If the PA error goes over the limit, show a band with the threshold.
    abs_error = (ref_pa - data.pa).abs().max()
    if abs_error > pa_threshold:
        right_ax.axhspan(
            ymin=-pa_threshold,
            ymax=pa_threshold,
            facecolor="red",
            edgecolor="None",
            linestyle="None",
            alpha=0.2,
            zorder=5,
        )

    if legend:
        ax.legend(
            handles=[pa_plot, pa_error_plot],
            loc="upper left",
        ).set_zorder(100)
        ax.margins(0.05, 0.2)

    # The left and right y-axis are different z-stacks so it's not possible to mix
    # and match. This moves the right y-axis to the background.
    right_ax.set_zorder(-1)
    ax.patch.set_visible(False)  # type: ignore
    ax.grid(False)
    right_ax.patch.set_visible(True)  # type: ignore

    ax.set_xlabel("Frame number")
    ax.set_ylabel("Position angle [deg]", labelpad=10)
    right_ax.set_ylabel("Position angle error [deg]", labelpad=10)

    if title:
        ax.set_title(title)

    return [ax, right_ax]


def plot_zero_point_or_fwhm(
    outpath: pathlib.Path,
    solution: GlobalSolution,
    save_subplots: bool = False,
    column="zero_point",
):
    """Plots the zero point or FWHM of an exposure."""

    fig, axd = get_camera_figure()

    # Get guider data first. We'll use this to select camera frames to use.
    guider_data = solution.guider_data()
    guider_data = guider_data.loc[guider_data.guide_mode == "guide"]

    # Plot top panels with the East and West camera zero point or FWHM.
    for camera_solution in solution.coadd_solutions:
        camera = camera_solution.camera

        # Get all the frames and the ZP/FWHM value of the co-added camera image.
        frame_data = camera_solution.frame_data()

        # We want to reject the frames in which we were acquiring since those may
        # have starts blurred or moving. If there's at least one "gaia" in the
        # wcs_mode, use that. Otherwise this must be a reprocessed set of frames
        # which all have "astrometry.net". Then use the guider_data framenos.
        if "gaia" in frame_data.wcs_mode.values:
            frame_data = frame_data.loc[frame_data.wcs_mode == "gaia", :]
        else:
            valid = numpy.in1d(frame_data.frameno, guider_data.frameno)
            frame_data = frame_data.loc[valid]

        coadd_value = getattr(camera_solution, column)

        if len(frame_data) == 0 and numpy.isnan(coadd_value):
            continue

        # Un-hide the axes.
        ax = axd[camera]  # type: ignore
        ax.axis("on")

        # Plot data.
        ax = _plot_zero_point_or_fwhm_axes(
            ax,
            camera_solution,
            frame_data,
            coadd_value,
            column=column,
            legend=False,
        )

        # Move the West camera y-axis to the right to unclutter the central section.
        if camera == "west":
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        ax.set_xlabel("")

        if save_subplots:
            create_subplot(
                _plot_zero_point_or_fwhm_axes,
                get_subplot_path(outpath, f"_{camera}"),
                camera_solution,
                frame_data,
                coadd_value,
                column=column,
                title=f"Camera {camera.capitalize()}",
                legend=True,
            )

    # Now do the full frame. Almost identical to above.
    global_value = getattr(solution, column)

    if len(guider_data.zero_point.dropna()) >= 2 or not numpy.isnan(global_value):
        ax = axd["global"]  # type: ignore
        ax.axis("on")

        _plot_zero_point_or_fwhm_axes(
            ax,
            solution,
            guider_data,
            global_value,
            column=column,
            legend=True,
        )

        if save_subplots:
            create_subplot(
                _plot_zero_point_or_fwhm_axes,
                get_subplot_path(outpath),
                solution,
                guider_data,
                global_value,
                column=column,
                title="Full frame",
                legend=True,
            )

    if column == "zero_point":
        fig.suptitle(f"Zero point for {solution.path.name}")
    else:
        fig.suptitle(f"FWHM for {solution.path.name}")

    fig.savefig(str(outpath))

    plt.close("all")


def _plot_zero_point_or_fwhm_axes(
    ax: Axes,
    solution: GlobalSolution | CoAdd_CameraSolution,
    data: pandas.DataFrame,
    coadd: float,
    column="zero_point",
    title: str | None = None,
    legend: bool = True,
):
    """Plot zero point or FWHM axes."""

    handles = []

    # Plot the data for each frame.
    (plot,) = ax.plot(
        data.frameno,
        data[column],
        "b-",
        label="Frames",
    )
    handles.append(plot)

    # If the co-added value exists, plot it as a horizontal dashed line.
    if not numpy.isnan(coadd):
        median_line = ax.axhline(
            y=coadd,
            color="k",
            linestyle="dashed",
            label="Co-add",
            alpha=0.5,
        )
        handles.append(median_line)

    # Add label with warnings.
    if len(data) > 0 and data["telescope"].iloc[0] != "spec":
        if column == "zero_point":
            warn = solution.transp_warning()
        else:
            warn = solution.fwhm_warning()
    else:
        warn = False

    if warn:
        ax.text(
            0.98,
            0.94,
            "WARNING",
            transform=ax.transAxes,
            fontsize=14,
            ha="right",
            va="center",
            color="r",
        )

    if legend:
        loc = "lower right" if column == "zero_point" else "upper left"
        ax.legend(handles=handles, loc=loc).set_zorder(100)
        ax.margins(0.05, 0.25)

    ax.set_xlabel("Frame number")

    if column == "zero_point":
        ax.set_ylabel("Zero Point [mag]", labelpad=10)
    else:
        ax.set_ylabel("FWHM [arcsec]", labelpad=10)

    if title:
        ax.set_title(title)

    return ax


def plot_guider_offsets(
    outpath: pathlib.Path,
    solution: GlobalSolution,
    save_subplots: bool = False,
):
    """Plots guider data (RA/Dec/PA offsets and applied corrections)."""

    fig, axd = plt.subplot_mosaic(
        [["sep", "sep"], ["meas", "applied"]],
        figsize=(11, 8),
    )

    fig.subplots_adjust(
        left=0.1,
        right=0.9,
        bottom=0.075,
        top=0.93,
        wspace=0.1,
        hspace=0.3,
    )

    gdata = solution.guider_data()
    gdata = gdata.loc[gdata.guide_mode == "guide"]

    # Top panel. On the left y axis plot the separation to the field
    # centre. In the right y axis plot the PA error (field - measured).
    gdata["separation"] = numpy.hypot(gdata.ra_off, gdata.dec_off)
    sep_data = gdata.loc[:, ["frameno", "separation"]]

    sep_ax = axd["sep"]  # type: ignore
    (sep_plot,) = sep_ax.plot(sep_data.frameno, sep_data.separation, "b-")
    sep_ax.set_xlabel("Frame number")
    sep_ax.set_ylabel("Separation [arcsec]", labelpad=10)

    sep_ax_r = sep_ax.twinx()
    pa_off = gdata.pa_field - (gdata.pa - 180)
    (pa_off_plot,) = sep_ax_r.plot(gdata.frameno, pa_off, "g-")
    sep_ax_r.set_ylabel("Position angle error [deg]", labelpad=10)
    sep_ax_r.legend(
        handles=[sep_plot, pa_off_plot],
        labels=["Separation", "PA error"],
        loc="upper right",
    )
    sep_ax_r.grid(False)

    # Bottom left panel. Plot the measured offsets in RA, Dec..
    meas_ax = axd["meas"]  # type: ignore
    meas_ax.plot(gdata.frameno, gdata.ra_off, "b-", label="RA", linewidth=0.5)
    meas_ax.plot(gdata.frameno, gdata.dec_off, "r-", label="Dec", linewidth=0.5)
    meas_ax.legend(loc="lower left")
    meas_ax.set_xlabel("Frame number")
    meas_ax.set_ylabel("Measured error [arcsec]", labelpad=10)

    # Bottom right panel. Plot the applied corrections in RA, Dec.
    # No point in plotting PA corrections since they are not applied
    # during guiding and we have excluded acquisition.
    appl_ax = axd["applied"]  # type: ignore
    appl_ax.plot(gdata.frameno, gdata.ax0_applied, "b-", linewidth=0.5)
    appl_ax.plot(gdata.frameno, gdata.ax1_applied, "r-", linewidth=0.5)
    appl_ax.set_xlabel("Frame number")
    appl_ax.set_ylabel("Applied offset [arcsec]")
    appl_ax.yaxis.tick_right()
    appl_ax.yaxis.set_label_position("right")

    if save_subplots:
        save_subplot(
            fig,
            [sep_ax, sep_ax_r],
            get_subplot_path(outpath, ""),
            bbox_range=[0.0, 0.47, 1.0, 0.98],
        )

        save_subplot(
            fig,
            [meas_ax],
            get_subplot_path(outpath, "_measured"),
            bbox_range=[0.0, 0.0, 0.505, 0.48],
        )

        save_subplot(
            fig,
            [appl_ax],
            get_subplot_path(outpath, "_applied"),
            bbox_range=[0.505, 0.0, 1.0, 0.48],
        )

    fig.suptitle(f"Guider data for {solution.path.name}")
    fig.savefig(str(outpath))
