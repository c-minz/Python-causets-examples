#!/usr/bin/env python
'''
Created on 26 Dec 2020

This is a script to create animations of lightcones in causal sets.

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import List, Tuple, Callable, Dict, Any, Union
from causets import EmbeddedCauset, CausetEvent, spacetimes, shapes
from causets import causetplotting as cplt, colorschemes as cs
import matplotlib as mpl
from matplotlib import pyplot as plt, axes as plta, animation
import numpy as np
import lightcone_animations_coordinates as LAC
# Path to movie encoder:
mpl.rcParams['animation.ffmpeg_path'] = \
    r'C:\\Program Files\\FFmpeg\\ffmpeg-v4.3.2\\bin\\ffmpeg.exe'

# ==============================
# Causet and spacetime:
dt: float = 0.002  # time step: 0.002 for video length of about 20s
causets_data: List[Tuple[str, str, List[int], List[List[float]]]] = [
    #('de Sitter', *LAC.get_1simplex(1.2, spacetime='de Sitter')),
    #('de Sitter', *LAC.get_2simplex(1.2, spacetime='de Sitter')),
    #('Minkowski', *LAC.get_1simplex(1.2, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_2simplex(1.2, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_3simplex(1.2, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_latticeD2(0.6, spacetime='Minkowski')),
    ('Minkowski', *LAC.get_latticeD3_oct(0.6, spacetime='Minkowski')),
    #('Minkowski',
    # *LAC.get_latticeD3_oct_cut(0.6, spacetime='Minkowski')),  # overwrites _oct!
    ('Minkowski', *LAC.get_latticeD4_oct(0.6, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_latticeD3_hcp(0.45, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_latticeD3_fcc(0.45, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_latticeD3_rho(0.45, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_latticeD3_slab(0.27, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_latticeD3_slabpinf(0.27, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_latticeD3_slabfinf(0.27, spacetime='Minkowski')),
    #('Minkowski', *LAC.get_latticeD3_slabpert(0.27, spacetime='Minkowski')),
    #('Schwarzschild', *LAC.get_1simplex(1.2, spacetime='Schwarzschild')),
]

# ==============================
# General parameters:
createThumbnail: bool = True
createVideo: bool = True
output_fps: float = 60.0  # frames per second
output_dpi: float = 100.0  # dots per inch
output_size: List[float] = [19.2, 10.8]  # inch
cs.setGlobalColorScheme('UniYork')
backgroundcolor: str = '#22262a'
foregroundcolor = 'white'
thumbnail_format: str = 'png'
thumbnail_fontsize: int = 20
thumbnail_eventcolor: str = foregroundcolor
thumbnail_linkcolor: str = 'cs:orange'
eventlabels: Any = False  # {'color': foregroundcolor}
draw_shape: bool = True
t_increase: float = 0.25
shapecolor: str = foregroundcolor
titlefontweight: str = 'bold'
titlefontsize: int = 15
label_t: str = 'time coordinate'
label_s: str = 'space coordinate'
section_t: str = 'Spacetime diagram:'
section_s1: str = 'Space slice with the future fading in:'
section_s2: str = 'Space slice with the past fading out:'
section_fontstyle: str = 'italic'
labelfontweight: str = 'bold'
labelfontsize: int = 13
licencetext: str = ''  # 'Copyright by C. Minz\n' + \
# 'BSD 3-Clause License\n' + \
# 'https://github.com/c-minz'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linewidth'] = 0.25
conelinewidth: float = 2.0
conealpha: float = 0.75
conealpha_low: float = 0.15
conefade: str = 'linear'
eventcolor: str = 'cs:orange'
eventsize: float = 9.0
linkwidth: float = 4.0
spacetimes.default_samplingsize = 64
shapes.default_samplingsize = 64

M_name: str
name: str
perm: List[int]
coords: List[List[float]]
for M_name, name, perm, coords in causets_data:
    # ==============================
    # Create objects:
    M_shape: Union[shapes.CoordinateShape, None] = None  # default
    if M_name in {'black hole', 'Schwarzschild'}:
        M_shape = shapes.CoordinateShape(2, 'cuboid', edges=[2., 2.],
                                         center=[0., 1.])
    elif M_name == 'Minkowski':
        M_shape = shapes.CoordinateShape(len(coords[0]), 'cube', edge=2.)

    C: EmbeddedCauset = EmbeddedCauset(coordinates=coords, spacetime=M_name,
                                       shape=M_shape)
    M: spacetimes.Spacetime = C.Spacetime
    M_shape = C.Shape
    output_file: str = f'{M_name} {name}'
    title: str = 'Lightcones of a causal ' + name + ' embedded in ' + \
        str(M.Dim) + '-dimensional ' + M_name + ' spacetime'
    C_diagram: List[CausetEvent] = [C.find(i) for i in perm] if perm else []

    if createThumbnail and C_diagram:
        fig = plt.figure(figsize=output_size, dpi=output_dpi,
                         facecolor=backgroundcolor)
        cplt.plotDiagram([C.find(i) for i in perm], perm,
                         plotAxes=plt.gca(), labels=eventlabels,
                         events={'markersize': eventsize,
                                 'markerfacecolor': thumbnail_eventcolor},
                         links={'linewidth': linkwidth,
                                'color': thumbnail_linkcolor},
                         axislim={'xlim': [-1.1, 1.1], 'ylim': [-1.1, 1.1]})
        plt.suptitle(title, x=0.5, y=0.1, va='bottom',
                     fontsize=thumbnail_fontsize, color=foregroundcolor,
                     fontweight=titlefontweight)
        plt.savefig(output_file + '.' + thumbnail_format,
                    format=thumbnail_format, dpi=output_dpi,
                    facecolor=backgroundcolor)

    if createVideo:
        # ==============================
        # Animation parameters:
        t_start: float
        t_end: float
        t_start, t_end = M_shape.Limits(0)
        t_start = t_start - t_increase
        t_end = t_end + t_increase
        t_range: np.ndarray = np.arange(t_start, t_end + dt, dt)
        t_len: float = t_end - t_start
        t_depth: float = 1.00 if M_name == 'Minkowski' else 1.75
        infotext: str = 'Above: Hasse diagram of \n' if perm else '\n'
        infotext = infotext + ('a subset of a causal \n' + name
                               if name.find('lattice') >= 0
                               else 'a causal ' + name + '\n')
        infotext = infotext + '\n\n\n\n' + \
            'On the right: animated \n' + \
            'slice through the past and \n' + \
            'future lightcones of the \n' + \
            'causal set embedded in \n' + \
            f'{M.Dim}-dimensional {M_name} \n' + \
            'spacetime.\n\n' + \
            'The animated slice is at \n' + \
            'time coordinate:\n'
        conealpha_STfactor: float
        if C.Dim > 3:  # for high dimensions
            conealpha_STfactor = 0.075
        elif M_name == 'Minkowski':
            conealpha_STfactor = 0.15
        else:
            conealpha_STfactor = 0.35

        # ==============================
        # Create animation:
        Pmain_dims: List[int] = [1, 2, 3]
        Pmain_conealpha: float = conealpha_STfactor * conealpha
        Pmain_proj: Union[str, None] = '3d'
        Psub_dims: List[int] = [1, 2]
        if C.Dim == 2:
            Pmain_dims = [1, 0]
            Pmain_proj = None
            Pmain_conealpha = conealpha
        elif C.Dim == 3:
            Pmain_dims = [1, 2, 0]

        mpl.rcParams['savefig.facecolor'] = backgroundcolor
        mpl.rcParams['axes.edgecolor'] = foregroundcolor
        fig = plt.figure(figsize=output_size, dpi=output_dpi,
                         facecolor=backgroundcolor)
        fig.suptitle(title, color=foregroundcolor,
                     fontsize=titlefontsize, fontweight=titlefontweight)
        gs = mpl.gridspec.GridSpec(2, 32, figure=fig,
                                   left=0.02, right=0.98, top=0.92, bottom=0.08)
        ax_info = fig.add_subplot(gs[1, 0:5], facecolor=backgroundcolor)
        info = mpl.patches.Rectangle((0., 0.), 2., 2., fill=False,
                                     transform=ax_info.transAxes, clip_on=False)
        ax_main = fig.add_subplot(gs[:, 5:23], facecolor=backgroundcolor,
                                  projection=Pmain_proj)
        ax_main.tick_params(color=foregroundcolor, labelcolor=foregroundcolor)
        ax_top = fig.add_subplot(gs[0, -8:], facecolor=backgroundcolor)
        ax_top.tick_params(color=foregroundcolor, labelcolor=foregroundcolor)
        ax_bot = fig.add_subplot(gs[1, -8:], facecolor=backgroundcolor)
        ax_bot.tick_params(color=foregroundcolor, labelcolor=foregroundcolor)
        ax_top.set_axis_off()
        ax_bot.set_axis_off()

        if C_diagram:
            cplt.plotDiagram(C_diagram, perm,
                             plotAxes=fig.add_subplot(gs[0, 0:5], alpha=0.0),
                             labels=eventlabels, events={
                                 'markerfacecolor': eventcolor},
                             axislim={'xlim': [-1.1, 1.1], 'ylim': [-1.1, 1.1]})

        Pmain: Callable[[float], Dict[str, Any]]
        if C.Dim > 3:
            Pmain = cplt.Plotter(
                C, plotAxes=ax_main, dims=Pmain_dims,
                pastcones={'linewidth': conelinewidth,
                           'alpha': conealpha_STfactor * conealpha_low},
                futurecones={'linewidth': conelinewidth,
                             'alpha': Pmain_conealpha},
                conetimedepth=t_depth, conetimefade=conefade, timedepth=t_depth,
                labels=eventlabels, links={'linewidth': linkwidth},
                events={'markersize': eventsize,
                        'markerfacecolor': eventcolor},
                axislim='shape', shape=C.Shape)
        else:
            Pmain = cplt.Plotter(
                C, plotAxes=ax_main, dims=Pmain_dims,
                pastcones={'linewidth': conelinewidth,
                           'alpha': Pmain_conealpha},
                futurecones={'linewidth': conelinewidth,
                             'alpha': Pmain_conealpha},
                conetimedepth=t_depth, conetimefade=conefade,
                labels=eventlabels, links={'linewidth': linkwidth},
                events={'markersize': eventsize,
                        'markerfacecolor': eventcolor},
                axislim='shape', shape=C.Shape)

        Ptop: Callable[[float], Dict[str, Any]]
        Pbot: Callable[[float], Dict[str, Any]]
        if C.Dim > 2:
            Ptop = cplt.Plotter(
                C, plotAxes=ax_top, dims=Psub_dims,
                pastcones={'linewidth': conelinewidth, 'alpha': conealpha_low},
                futurecones={'linewidth': conelinewidth, 'alpha': conealpha},
                conetimedepth=t_depth, conetimefade=conefade, timedepth=t_depth,
                labels=False, links={'linewidth': linkwidth},
                events={'markersize': eventsize,
                        'markerfacecolor': eventcolor},
                axislim='shape', shape=C.Shape)
            Pbot = cplt.Plotter(
                C, plotAxes=ax_bot, dims=Psub_dims,
                pastcones={'linewidth': conelinewidth, 'alpha': conealpha},
                futurecones={'linewidth': conelinewidth,
                             'alpha': conealpha_low},
                conetimedepth=-t_depth, conetimefade=conefade, timedepth=-t_depth,
                labels=False, links={'linewidth': linkwidth},
                events={'markersize': eventsize,
                        'markerfacecolor': eventcolor},
                axislim='shape', shape=C.Shape)

        def set_axeslabels(ax: plta.Axes, dims: List[int]) -> None:
            ax.set_xlabel(label_t if dims[0] == 0 else label_s,
                          color=foregroundcolor, fontweight=labelfontweight)
            ax.set_ylabel(label_t if dims[1] == 0 else label_s,
                          color=foregroundcolor, fontweight=labelfontweight)
            if len(dims) > 2:
                ax.set_zlabel(label_t if dims[2] == 0 else label_s,
                              color=foregroundcolor, fontweight=labelfontweight)

        # ==============================
        # Run and show progress in console:
        progressbar_len: int = 20
        i: int = 1  # counter for console progress bar

        def animator(t: float) -> None:
            global i
            ax_info.clear()
            ax_info.set_xlim([0.0, 5.0])
            ax_info.set_ylim([-2.5, 2.5])
            ax_info.text(0.0, 3.5, infotext + ('t = %1.2f' % t),
                         ha='left', va='top', wrap=True,
                         color=foregroundcolor, fontsize=labelfontsize,
                         linespacing=1.5)
            ax_info.text(0.0, -2.0, licencetext,
                         ha='left', va='bottom', wrap=True,
                         color='gray', fontsize=10,
                         linespacing=1.5)
            ax_info.set_axis_off()
            ax_main.clear()
            if draw_shape:
                if C.Dim > 2:
                    C.Shape.plot(Pmain_dims, ax_main,
                                 edgecolor=None, color=shapecolor)
                else:
                    C.Shape.plot(Pmain_dims, ax_main,
                                 edgecolor=None, facecolor=shapecolor)
            Pmain(t)
            set_axeslabels(ax_main, Pmain_dims)
            if C.Dim > 3:
                ax_main.set_title(section_s1, color=foregroundcolor,
                                  fontstyle=section_fontstyle, fontsize=labelfontsize)
            else:
                ax_main.set_title(section_t, color=foregroundcolor,
                                  fontstyle=section_fontstyle, fontsize=labelfontsize)
            if len(Pmain_dims) > 2:
                ax_main_panecolor: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
                ax_main.view_init(elev=0.0, azim=t * 360 / t_len)
                ax_main.xaxis.set_pane_color(ax_main_panecolor)
                ax_main.yaxis.set_pane_color(ax_main_panecolor)
                ax_main.zaxis.set_pane_color(ax_main_panecolor)
            if C.Dim > 2:
                ax_top.clear()
                if draw_shape:
                    C.Shape.plot(Psub_dims, ax_top,
                                 edgecolor=None, facecolor=shapecolor)
                Ptop(t)
                set_axeslabels(ax_top, Psub_dims)
                ax_top.set_title(section_s1, color=foregroundcolor,
                                 fontstyle=section_fontstyle, fontsize=labelfontsize)
                ax_bot.clear()
                if draw_shape:
                    C.Shape.plot(Psub_dims, ax_bot,
                                 edgecolor=None, facecolor=shapecolor)
                Pbot(t)
                set_axeslabels(ax_bot, Psub_dims)
                ax_bot.set_title(section_s2, color=foregroundcolor,
                                 fontstyle=section_fontstyle, fontsize=labelfontsize)
            fig.set_facecolor(backgroundcolor)
            if t - t_start - i * t_len / progressbar_len >= 0:
                print('#', end='')
                i += 1

        print(f'Generating animation ({name}, {M_name}):')
        print('0% |' + '_' * progressbar_len + '| 100%\n   |', end='')
        anim = animation.FuncAnimation(fig, animator, frames=t_range)
        animwriter = animation.FFMpegWriter(fps=output_fps)
        anim.save(output_file + '.mp4', writer=animwriter)
        for j in range(i, progressbar_len + 1):
            print('#', end='')
        print('|\n\n')
if createVideo:
    print('Finished all animations in the list.\n\n')
