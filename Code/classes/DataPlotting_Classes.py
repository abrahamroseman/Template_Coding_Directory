#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # How to Import to Code Document
########################################
# import os, sys
# mainCodeDirectory = os.path.abspath("../..")
# path = os.path.join(mainCodeDirectory, "classes")
# sys.path.append(path)

# # Importing
# from DataPlotting_Classes import DataPlotting_Classes


# In[45]:


# DataPlotting_Classes
# ============================================================

# Libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Plotting Specific Libraries
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import TwoSlopeNorm

#Class
class DataPlotting_Classes:

    # UltimatePlotting_Class
    # ============================================================
    class UltimatePlotting_Class:
        """
        UltimatePlotting_Class.MakeAlignedGridFigure to make the figure object.
        UltimatePlotting_Class.LinePlot and ContourPlot to add on to a specific axis in a figure.
        """
    
        ##################################################################################################################################
        #LinePlot and ContourPlot Helper Functions
        ##################################################################################################################################
        @staticmethod
        def GetFontStyles(scale=1.0):
            """
            Central place defining font family and sizes for every plot
            element. Returns a dictionary keyed by element type. Change
            values here (or pass scale) and every plot styled through
            ApplyFontStyles updates consistently.
    
            scale : multiply all font sizes by this factor (e.g. 1.5 for a
                presentation figure, 0.8 for a dense multi-panel figure).
            """
            fontStyles = {
                "title":      {"fontsize": 14 * scale, "fontfamily": "sans-serif", "fontweight": "bold"},
                "label":      {"fontsize": 12 * scale, "fontfamily": "sans-serif", "fontweight": "normal"},
                "tick":       {"fontsize": 10 * scale, "fontfamily": "sans-serif"},
                "legend":     {"fontsize": 10 * scale},
                "cbarLabel":  {"fontsize": 12 * scale, "fontfamily": "sans-serif"},
                "cbarTick":   {"fontsize": 10 * scale, "fontfamily": "sans-serif"},
                "suptitle":   {"fontsize": 16 * scale, "fontfamily": "sans-serif", "fontweight": "bold"},
            }
            return [fontStyles]
    
        @staticmethod
        def ApplyFontStyles(ax, fontStyles):
            """
            Apply the font dictionary from GetFontStyles to one axis: ticks,
            axis labels, title, and legend (if present).
            """
            ax.tick_params(axis="both", labelsize=fontStyles["tick"]["fontsize"])
            ax.xaxis.label.set_fontsize(fontStyles["label"]["fontsize"])
            ax.yaxis.label.set_fontsize(fontStyles["label"]["fontsize"])
            ax.xaxis.label.set_fontfamily(fontStyles["label"]["fontfamily"])
            ax.yaxis.label.set_fontfamily(fontStyles["label"]["fontfamily"])
            ax.title.set_fontsize(fontStyles["title"]["fontsize"])
            ax.title.set_fontweight(fontStyles["title"]["fontweight"])
    
            legend = ax.get_legend()
            if legend is not None:
                plt.setp(legend.get_texts(), fontsize=fontStyles["legend"]["fontsize"])

        @staticmethod
        def ApplyColorbarFontStyles(cbar, fontStyles):
            """
            Apply the font dictionary from GetFontStyles to a colorbar: its
            tick labels and its label text (if set).
            """
            cbar.ax.tick_params(labelsize=fontStyles["cbarTick"]["fontsize"])
            if cbar.ax.yaxis.label.get_text():
                cbar.ax.yaxis.label.set_fontsize(fontStyles["cbarLabel"]["fontsize"])
                cbar.ax.yaxis.label.set_fontfamily(fontStyles["cbarLabel"]["fontfamily"])
            if cbar.ax.xaxis.label.get_text():
                cbar.ax.xaxis.label.set_fontsize(fontStyles["cbarLabel"]["fontsize"])
                cbar.ax.xaxis.label.set_fontfamily(fontStyles["cbarLabel"]["fontfamily"])
        
        @staticmethod
        def GenerateTicksAndLim(data, nTicks=6, nLevels=15):
            if isinstance(data, (list, tuple)) and len(data) > 0 and not np.isscalar(data[0]):
                data = np.concatenate([np.asarray(d).ravel() for d in data])
            else:
                data = np.asarray(data)
        
            dMin, dMax = np.nanmin(data), np.nanmax(data)
            locator = MaxNLocator(nbins=nTicks - 1)
            ticks = locator.tick_values(dMin, dMax)
            lim = (ticks[0], ticks[-1])

            levels = np.linspace(lim[0], lim[1], nLevels)
            return [ticks,lim,levels]

        @staticmethod
        def StretchColorbarAcrossAxes(cax, axesList, cbarSide="right"):
            """
            Stretch a single colorbar axis (cax) so it visually spans the
            combined extent of multiple main axes, instead of sitting next
            to just one of them. Use after all axesList have been plotted
            (positions must be finalized), and after the one PlotContour
            call that actually drew a colorbar into cax.

            cax : the single colorbar axis to reposition (e.g. caxes[i, j]
                from the one cell where showCbar=True).
            axesList : list of axes (2 or more) the colorbar should span,
                e.g. [axes[0, 0], axes[1, 0]] for two stacked rows, or
                [axes[0, 0], axes[0, 1]] for two side-by-side columns.
            cbarSide : "right" (vertical colorbar, stretch spans the
                combined VERTICAL extent of axesList) or "bottom"
                (horizontal colorbar, stretch spans the combined
                HORIZONTAL extent of axesList).
            """
            positions = [ax.get_position() for ax in axesList]
            caxPos = cax.get_position()

            if cbarSide == "right":
                yMin = min(pos.y0 for pos in positions)
                yMax = max(pos.y1 for pos in positions)
                cax.set_position([caxPos.x0, yMin, caxPos.width, yMax - yMin])
            else:  # "bottom"
                xMin = min(pos.x0 for pos in positions)
                xMax = max(pos.x1 for pos in positions)
                cax.set_position([xMin, caxPos.y0, xMax - xMin, caxPos.height])
    
        ##################################################################################################################################
        #LinePlot and ContourPlot Functions
        ##################################################################################################################################
    
        @staticmethod
        def PlotLine(ax=None,figSize=None,
                     xData=[1,2,3], yData=[1,2,3],label=None,
                     title=None, xLabel=None, yLabel=None,
                     xTicks=None,yTicks=None, xLim=None,yLim=None,
                     fontScale=1.0, showLegend=None,
                     **plotKwargs):
            """
            ...
            xTicks / yTicks : explicit tick locations, e.g. np.arange(0, 11, 2).
                Pass on whichever call should set them (usually the first);
                later overlay calls can omit them to leave ticks as-is.
            ...
            """
            if ax is None:
                fig, ax = plt.subplots(figsize=figSize)
            else:
                fig = ax.figure
    
            ax.plot(xData, yData, label=label, **plotKwargs)
    
            if title is not None:
                ax.set_title(title)
            if xLabel is not None:
                ax.set_xlabel(xLabel)
            if yLabel is not None:
                ax.set_ylabel(yLabel)
            if xTicks is not None:
                ax.set_xticks(xTicks)
            if yTicks is not None:
                ax.set_yticks(yTicks)
            if xLim is not None:
                ax.set_xlim(xLim)
            if yLim is not None:
                ax.set_ylim(yLim)
    
            anyLabels = any(line.get_label() and not line.get_label().startswith("_") for line in ax.get_lines())
            if showLegend or (showLegend is None and anyLabels):
                ax.legend()
    
            [fontStyles] = DataPlotting_Classes.UltimatePlotting_Class.GetFontStyles(fontScale)
            DataPlotting_Classes.UltimatePlotting_Class.ApplyFontStyles(ax, fontStyles)
    
            return [fig, ax]

        @staticmethod
        def PlotContour(ax=None,cax=None,figSize=None, #figure and axis
                        xData=None,yData=None,zData=None, #data
                        plotType='contourf', #plotting function
                        symmetric=False, centerZero=False, #colorbar symmetry
                        showCbar=True,cbarOrientation="vertical",cbarLabel=None, #colorbar setup
                        xTicks=None,yTicks=None,xLim=None,yLim=None, #data ticks and limits
                        nLevels=13,nLevelsNeg=None,nLevelsPos=None, #colorbar ticks and limits
                        cbarTicks=None,colorLimits=None,useLocalColorRange=False, #colorbar ticks and limits
                        title=None,xLabel=None,yLabel=None, #labels
                        fontScale=1.0, **contourKwargs): #other arguments
            
            if plotType not in ('contourf', 'contour', 'pcolormesh'):
                raise ValueError(f"plotType must be 'contourf', 'contour', or 'pcolormesh', got {plotType!r}")
            if ax is None:
                fig, ax = plt.subplots(figsize=figSize)
            else:
                fig = ax.figure
                
            #code for using local color range
            if colorLimits is None and useLocalColorRange and xLim is not None and yLim is not None:
                xMask = (xData >= xLim[0]) & (xData <= xLim[1])
                yMask = (yData >= yLim[0]) & (yData <= yLim[1])
                visibleData = zData[np.ix_(yMask, xMask)]
                colorLimits = (np.nanmin(visibleData), np.nanmax(visibleData))
            if colorLimits is None:
                colorLimits = (np.nanmin(zData), np.nanmax(zData))
            if symmetric:
                vmin, vmax = colorLimits
                vabs = max(abs(vmin), abs(vmax))
                colorLimits = (-vabs, vabs)
            if not ({'levels', 'vmin', 'vmax', 'norm'} & contourKwargs.keys()):
                vmin, vmax = colorLimits
                if plotType == 'pcolormesh':
                    contourKwargs['vmin'] = vmin
                    contourKwargs['vmax'] = vmax
                else:
                    if (symmetric or centerZero) and vmin < 0 < vmax:
                        # separate level counts per side; 0 is the shared boundary
                        nNeg = nLevelsNeg if nLevelsNeg is not None else nLevels // 2 + 1
                        nPos = nLevelsPos if nLevelsPos is not None else nLevels // 2 + 1
                        negLevels = np.linspace(vmin, 0, nNeg)
                        posLevels = np.linspace(0, vmax, nPos)
                        contourKwargs['levels'] = np.unique(np.concatenate([negLevels, posLevels]))
                    else:
                        contourKwargs['levels'] = np.linspace(vmin, vmax, nLevels)
                if (symmetric or centerZero) and vmin < 0 < vmax:
                    contourKwargs['norm'] = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            elif plotType == 'pcolormesh' and 'levels' in contourKwargs:
                lv = contourKwargs.pop('levels')
                contourKwargs.setdefault('vmin', lv[0])
                contourKwargs.setdefault('vmax', lv[-1])
                
            #plotting
            if plotType == 'contourf':
                cf = ax.contourf(xData, yData, zData, **contourKwargs)
            elif plotType == 'contour':
                cf = ax.contour(xData, yData, zData, **contourKwargs)
            else:  # pcolormesh
                cf = ax.pcolormesh(xData, yData, zData, **contourKwargs)
                
            #labels, ticks, and limits
            if title is not None:
                ax.set_title(title)
            if xLabel is not None:
                ax.set_xlabel(xLabel)
            if yLabel is not None:
                ax.set_ylabel(yLabel)
            if xTicks is not None:
                ax.set_xticks(xTicks)
            if yTicks is not None:
                ax.set_yticks(yTicks)
            if xLim is not None:
                ax.set_xlim(xLim)
            if yLim is not None:
                ax.set_ylim(yLim)
                
            #colorbar setup
            cbar = None
            if showCbar:
                cbar = fig.colorbar(cf, cax=cax, ax=ax if cax is None else None, orientation=cbarOrientation)
                if cbarLabel is not None:
                    cbar.set_label(cbarLabel)
                if cbarTicks is not None:
                    cbar.set_ticks(cbarTicks)
                elif 'levels' in contourKwargs:
                    # default ticks to the exact level boundaries, so the first/last
                    # ticks are flush with the colorbar ends (the auto locator can
                    # skip the endpoints when a TwoSlopeNorm is in use)
                    cbar.set_ticks(contourKwargs['levels'])
            elif cax is not None:
                cax.set_visible(False)
            #fonts
            [fontStyles] = DataPlotting_Classes.UltimatePlotting_Class.GetFontStyles(fontScale)
            DataPlotting_Classes.UltimatePlotting_Class.ApplyFontStyles(ax, fontStyles)
            if cbar is not None:
                DataPlotting_Classes.UltimatePlotting_Class.ApplyColorbarFontStyles(cbar, fontStyles)
            return [fig, ax, cbar]
    
        ##################################################################################################################################
        #MakeAlignedGridFigure Functions
        ##################################################################################################################################
        @staticmethod    
        def BuildAxisLayout(n, needsCbar, ratios, cbarThicknessRatio, cbarPaddingRatio, gapRatio):
            """
            Shared layout logic for one dimension (used for both columns and
            rows). Works out the underlying GridSpec sizes along that dimension,
            plus which gridspec slot each logical row/column's main axis and
            colorbar axis lives in.
        
            All spacing (between logical entries, and between a main axis and its
            own colorbar) is inserted here as explicit blank spacer slots, sized
            as a fraction of the plot size they sit next to. GridSpec's own
            wspace/hspace are left at 0 so they never add a second, uncontrolled
            gap on top of these.
            """
            gsSizes = []
            mainIdx = []
            cbarIdx = []
            slot = 0
        
            for k in range(n):
                size = ratios[k]
        
                if k > 0 and gapRatio > 0:
                    gsSizes.append(gapRatio * size)  # gap before this row/column
                    slot += 1
        
                gsSizes.append(size)
                mainIdx.append(slot)
                slot += 1
        
                if needsCbar[k]:
                    if cbarPaddingRatio > 0:
                        gsSizes.append(cbarPaddingRatio * size)  # blank spacer, no axis here
                        slot += 1
                    gsSizes.append(cbarThicknessRatio * size)
                    cbarIdx.append(slot)
                    slot += 1
                else:
                    cbarIdx.append(None)
        
            return [gsSizes, mainIdx, cbarIdx]
        
        @staticmethod
        def AddAxisPair(fig, gs, i, j, mainRowIdx, mainColIdx, cbarRowIdx, cbarColIdx, hasCbar, cbarSide, axes, shareX, shareY):
            """
            Add one main axis (and its colorbar axis, if flagged) at logical cell
            (i, j), wiring up sharex/sharey against the first row/column, and
            placing the colorbar beside ('right') or below ('bottom') the axis.
            """
            sharexWith = axes[0, j] if (shareX and i > 0) else None
            shareyWith = axes[i, 0] if (shareY and j > 0) else None
        
            ax = fig.add_subplot(gs[mainRowIdx[i], mainColIdx[j]], sharex=sharexWith, sharey=shareyWith)
        
            cax = None
            if hasCbar[i, j]:
                if cbarSide[i, j] == "right":
                    cax = fig.add_subplot(gs[mainRowIdx[i], cbarColIdx[j]])
                else:  # "bottom"
                    cax = fig.add_subplot(gs[cbarRowIdx[i], mainColIdx[j]])
        
            return [ax, cax]
    
        ##################################################################################################################################
        #MakeAlignedGridFigure Function
        ##################################################################################################################################
        @staticmethod
        def MakeAlignedGridFigure(
            nRows, nCols,    
            figSize=None,
            shareX=False, shareY=False,
        
            hasCbar=None,
            cbarSide="right",
            colGapRatio=0.15, rowGapRatio=0.15,
            heightRatios=None, widthRatios=None,
            cbarThicknessRatio=0.06, 
            cbarPaddingRatioRight=0.05,cbarPaddingRatioBottom=0.15,
        ):
            """
            Build an nRows x nCols grid of axes where every axis in a column has
            the same width (and every axis in a row has the same height), even
            when only some cells carry a colorbar.
        
            nRows / nCols: (n, n) int tuple. How many rows and columns for axis
            figSize: (length, width) float tuple. Size of figure.
            shareX / shareY : share x within columns / y within rows.
        
            hasCbar : (nRows, nCols) bool array. True = this cell gets a colorbar.
                Default: all False (no colorbars anywhere).
            cbarSide : "right" or "bottom", or an (nRows, nCols) array of those,
                to mix placements cell by cell. "right" reserves a colorbar
                column for that whole logical column; "bottom" reserves a
                colorbar row for that whole logical row.
            cbarThicknessRatio : colorbar thickness, as a fraction of the plot size
                it sits next to (width if "right", height if "bottom").
            cbarPaddingRatio : gap between a main axis and its colorbar, as a
                fraction of the plot size it sits next to. The only thing
                controlling that gap.
            colGapRatio / rowGapRatio : gap between separate logical columns /
                rows, as a fraction of the plot size next to the gap. Replaces
                wSpace/hSpace so spacing never leaks into the colorbar gap.
            
        
            Returns [fig, axes, caxes]. axes and caxes are (nRows, nCols) object
            arrays; caxes entries are None where hasCbar is False.
            """
            hasCbar = np.zeros((nRows, nCols), dtype=bool) if hasCbar is None else np.array(hasCbar, dtype=bool)
            cbarSide = np.full((nRows, nCols), cbarSide, dtype=object) if isinstance(cbarSide, str) else np.array(cbarSide, dtype=object)
            widthRatios = [1.0] * nCols if widthRatios is None else widthRatios
            heightRatios = [1.0] * nRows if heightRatios is None else heightRatios
        
            needsRightCbar = [np.any(hasCbar[:, j] & (cbarSide[:, j] == "right")) for j in range(nCols)]
            needsBottomCbar = [np.any(hasCbar[i, :] & (cbarSide[i, :] == "bottom")) for i in range(nRows)]
        
            [gsWidths, mainColIdx, cbarColIdx] = DataPlotting_Classes.UltimatePlotting_Class.BuildAxisLayout(
                nCols, needsRightCbar, widthRatios, cbarThicknessRatio, cbarPaddingRatioRight, colGapRatio
            )
            [gsHeights, mainRowIdx, cbarRowIdx] = DataPlotting_Classes.UltimatePlotting_Class.BuildAxisLayout(
                nRows, needsBottomCbar, heightRatios, cbarThicknessRatio, cbarPaddingRatioBottom, rowGapRatio
            )
        
            fig = plt.figure(figsize=figSize)
            gs = GridSpec(
                nrows=len(gsHeights), ncols=len(gsWidths),
                width_ratios=gsWidths, height_ratios=gsHeights,
                wspace=0, hspace=0, figure=fig,
            )
        
            axes = np.empty((nRows, nCols), dtype=object)
            caxes = np.empty((nRows, nCols), dtype=object)
        
            for j in range(nCols):
                for i in range(nRows):
                    [axes[i, j], caxes[i, j]] = DataPlotting_Classes.UltimatePlotting_Class.AddAxisPair(
                        fig, gs, i, j, mainRowIdx, mainColIdx, cbarRowIdx, cbarColIdx,
                        hasCbar, cbarSide, axes, shareX, shareY
                    )
        
            if shareX:
                for j in range(nCols):
                    for i in range(nRows - 1):
                        plt.setp(axes[i, j].get_xticklabels(), visible=False)
        
            if shareY:
                for i in range(nRows):
                    for j in range(1, nCols):
                        plt.setp(axes[i, j].get_yticklabels(), visible=False)
        
            return [fig, axes, caxes]
    
        ##################################################################################################################################
        #Testing
        ##################################################################################################################################
        @staticmethod
        def Test():
            x = np.linspace(0, 10, 200)
            X, Y = np.meshgrid(x, np.linspace(-3, 3, 150))
            Z = np.sin(X) * np.exp(-(Y ** 2) / 2)
        
            hasCbar = [
                [False, False],
                [True, True],
            ]
            cbarSide = [
                ["right", "bottom"],
                ["right", "bottom"],
            ]
        
            [fig, axes, caxes] = DataPlotting_Classes.UltimatePlotting_Class.MakeAlignedGridFigure(
                nRows=2, nCols=2,
                shareX=True,shareY=True,
                
                hasCbar=hasCbar,
                cbarSide=cbarSide,
                figSize=(16, 8),
                heightRatios=(1, 1),
                colGapRatio=0.15, rowGapRatio=0.15,
                cbarPaddingRatioRight=0.05,
                cbarPaddingRatioBottom=0.20,
            )
    
            # Line Plot
            ############################################################################################################
            xData = x
            yData1 = np.sin(x)
            yData2 = yData1 * 0.5
            yData3 = np.cos(x)
            
            [xTicks,xLim,_] = DataPlotting_Classes.UltimatePlotting_Class.GenerateTicksAndLim(xData)
            [yTicks,yLim,_] = DataPlotting_Classes.UltimatePlotting_Class.GenerateTicksAndLim([yData1, yData2, yData3])
            
            axis = axes[0, 0]
            DataPlotting_Classes.UltimatePlotting_Class.PlotLine(ax=axis,
                                            xData=xData, yData=yData1, color="blue",
                                            label="sin", yLabel="signal",
                                            xTicks=xTicks, xLim=xLim,
                                            yTicks=yTicks, yLim=yLim)
            DataPlotting_Classes.UltimatePlotting_Class.PlotLine(ax=axis,
                                            xData=xData, yData=yData2, color="green",
                                            ls="--", label="sin/2")
            
            axis = axes[0, 1]
            DataPlotting_Classes.UltimatePlotting_Class.PlotLine(ax=axis,
                                            xData=xData, yData=yData3, color="black",
                                            label="cos")
            # Contour Plot
            ############################################################################################################
            [zTicks,zLim,zLevels] = DataPlotting_Classes.UltimatePlotting_Class.GenerateTicksAndLim(Z,nLevels=31)
    
            DataPlotting_Classes.UltimatePlotting_Class.PlotContour(
                ax=axes[1, 0], cax=caxes[1, 0],
                xData=X, yData=Y, zData=Z,
                levels=zLevels, cmap="viridis",
                vmin=zLim[0], vmax=zLim[1],
                cbarOrientation="vertical", cbarLabel="amplitude", cbarTicks=zTicks,
                xLabel="x", yLabel="y",
            )
    
            DataPlotting_Classes.UltimatePlotting_Class.PlotContour(
                ax=axes[1, 1], cax=caxes[1, 1],
                xData=X, yData=Y, zData=Z,
                levels=zLevels, cmap="plasma",
                cbarOrientation="horizontal", cbarLabel="amplitude", cbarTicks=zTicks,
                xLabel="x",
            )
            
            # fig.suptitle("Title", y=0.93)

            return fig

        @staticmethod
        def Test2():
            """
            Test showing how to make a shared colorbar plot
            (1) use GenerateTicksAndLim on list of both arrays flattened
            (2) set showCbar=False on one of the axes and also set hasCbar=False for that axis
            """
            x = np.linspace(0, 10, 200)
            X, Y = np.meshgrid(x, np.linspace(-3, 3, 150))
            Z1 = np.sin(X) * np.exp(-(Y ** 2) / 2)
            Z2 = np.cos(X) * np.exp(-(Y ** 2) / 3) * 1.4
            Z3 = np.sin(X) * np.exp(-(Y ** 2) / 4) * -1.4

            # 3 rows, 1 column. Only row 0 reserves a colorbar column.
            hasCbar = [[True], [False], [False]]
            cbarSide = [["right"], ["right"], ["right"]]

            [fig, axes, caxes] = DataPlotting_Classes.UltimatePlotting_Class.MakeAlignedGridFigure(
                nRows=3, nCols=1,
                shareX=True, shareY=True,
                hasCbar=hasCbar,
                cbarSide=cbarSide,
                figSize=(7, 9),
                heightRatios=(1, 1, 1),
                rowGapRatio=0.1,
                cbarPaddingRatioRight=0.05,
            )

            # Shared color scale computed from ALL THREE datasets together
            combined = np.concatenate([Z1.ravel(), Z2.ravel(), Z3.ravel()])
            [zTicks, zLim, zLevels] = DataPlotting_Classes.UltimatePlotting_Class.GenerateTicksAndLim(combined, nLevels=31)

            cmap='RdBu_r'
            
            DataPlotting_Classes.UltimatePlotting_Class.PlotContour(
                ax=axes[0, 0], cax=caxes[0, 0],
                xData=X, yData=Y, zData=Z1,
                levels=zLevels, cmap=cmap,
                cbarOrientation="vertical", cbarLabel="amplitude", cbarTicks=zTicks,
                yLabel="y",
            )

            DataPlotting_Classes.UltimatePlotting_Class.PlotContour(
                ax=axes[1, 0], cax=None, showCbar=False,
                xData=X, yData=Y, zData=Z2,
                levels=zLevels, cmap=cmap,
                yLabel="y",
            )

            DataPlotting_Classes.UltimatePlotting_Class.PlotContour(
                ax=axes[2, 0], cax=None, showCbar=False,
                xData=X, yData=Y, zData=Z3,
                levels=zLevels, cmap=cmap,
                xLabel="x", yLabel="y",
            )

            # Stretch the single colorbar to visually span all three rows
            DataPlotting_Classes.UltimatePlotting_Class.StretchColorbarAcrossAxes(
                cax=caxes[0, 0],
                axesList=[axes[0, 0], axes[1, 0], axes[2, 0]],
                cbarSide="right",
            )

            return fig


    # ============================================================
    # Functions
    # ============================================================
    class Functions:
        def ApplyScientificNotation_axes(axes, dimension='x', use_math_text=True, power_limits=(-1, 1), decimals=2, scientific=True):
            """
            Apply scientific notation with mantissas rounded to a fixed number of decimals.
            """
            for axis in axes:
                formatter = RoundedScalarFormatter(
                    decimals=decimals,
                    useMathText=use_math_text,
                    powerlimits=power_limits,
                    scientific=scientific
                )
                # Map dimension flags to actual axis objects
                if 'x' in dimension:
                    axis.xaxis.set_major_formatter(formatter)
                if 'y' in dimension:
                    axis.yaxis.set_major_formatter(formatter)
        
        class RoundedScalarFormatter(ScalarFormatter):
            def __init__(self, decimals=2, useMathText=True, powerlimits=(-1, 1), scientific=True):
                super().__init__(useMathText=useMathText)
                self.decimals = decimals
                self.set_scientific(scientific)
                self.set_powerlimits(powerlimits)
                self.set_useOffset(False)
        
            def _set_format(self):
                self.format = f"%.{self.decimals}f"
        
        def ApplyScientificNotation_colorbars(cbars, decimals=2):
            for cbar in cbars:
                fmt = FixedDecimalScalarFormatter(decimals=decimals, useMathText=True)
                fmt.set_powerlimits((-3, 3))   # enforce sci-notation range
                
                cbar.formatter = fmt
                cbar.update_ticks()
        
        class FixedDecimalScalarFormatter(ScalarFormatter):
            def __init__(self, decimals=2, useMathText=True, **kwargs):
                self.decimals = decimals
                super().__init__(useMathText=useMathText, **kwargs)
        
            def _set_format(self, vmin=None, vmax=None):
                """
                Matplotlib requires _set_format(self, vmin, vmax)
                We override it only to set the desired decimal format string.
                """
                # mantissa format: e.g., ".2f"
                fmt = f"%.{self.decimals}f"
        
                # scientific notation handled by ScalarFormatter internals
                # we only update the mantissa part
                self.format = fmt
                self._useOffset = True   # keep exponent in offsetText

        def MatchLimitsBetweenAxes(axes, dim='x'):
            """
            Match axis limits and ticks across multiple axes.
            This version computes the global min/max tick
            directly instead of requiring an axis to already contain them.
            """
        
            # gather ticks from all axes
            all_ticks = []
            for ax in axes:
                ticks = ax.get_xticks() if dim == 'x' else ax.get_yticks()
                if len(ticks) > 1:
                    all_ticks.append(ticks)
        
            if not all_ticks:
                return None
        
            # global min and max tick endpoints
            lo = min(t[0] for t in all_ticks)
            hi = max(t[-1] for t in all_ticks)
        
            # choose a reference tick array with the largest number of ticks
            ref_ticks = max(all_ticks, key=len)
        
            # but shift its first and last value to the global min/max
            ref_ticks = ref_ticks.copy()
            ref_ticks[0] = lo
            ref_ticks[-1] = hi
        
            # determine the limits corresponding to the tick endpoints
            ref_lim = (lo, hi)
        
            # apply to all axes
            for ax in axes:
                if dim == 'x':
                    ax.set_xlim(ref_lim)
                    ax.set_xticks(ref_ticks)
                else:
                    ax.set_ylim(ref_lim)
                    ax.set_yticks(ref_ticks)
        
            return ref_lim

    # FigureSavingFunctions
    # ============================================================
    class FigureSavingFunctions:

        @staticmethod
        def GetFigurePath(scriptName, fileName, figureSubDirectory="",
                          outputDirectory="../../Output",
                          extension="jpg",
                          makeDirs=True):
    
            directory = os.path.join(
                outputDirectory,
                "Figures",
                scriptName,
                figureSubDirectory
            )
    
            if makeDirs:
                os.makedirs(directory, exist_ok=True)
    
            if not fileName.lower().endswith(f".{extension.lower()}"):
                fileName = f"{fileName}.{extension}"
    
            return os.path.join(directory, fileName)
        
        class PDFSaver_Class:
            def __init__(self, fileName,
                         verbose=False):
                if not fileName.lower().endswith(".pdf"):
                    fileName = f"{fileName}.pdf"
                self.pdf = PdfPages(f'{fileName}.pdf')
                if verbose:
                    print(f'Saved PDF: {fileName}.pdf')
        
            def Save(self, fig, rasterize_zorder=2):
                if rasterize_zorder is not None:
                    for ax in fig.axes:
                        ax.set_rasterization_zorder(rasterize_zorder)
                self.pdf.savefig(fig)
                plt.close(fig)
        
            def Close(self):
                self.pdf.close()
        
            # setup as context manager
            def __enter__(self):
                return self
        
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.Close()
        
            @staticmethod
            def Test():

                PDFSaver_Class = DataPlotting_Classes.FigureSavingFunctions.PDFSaver_Class
                
                # --- manual open/close style ---
                PlotSaver = PDFSaver_Class("test_manual.pdf",verbose=True)
                fig1 = plt.figure()
                plt.plot([1, 2, 3])
                PlotSaver.Save(fig1)
                
                fig2 = plt.figure()
                plt.plot([3, 2, 1])
                PlotSaver.Save(fig2)
                
                PlotSaver.Close()
        
                # --- context manager style ---
                with PDFSaver_Class("test_with.pdf",verbose=True) as PlotSaver:
                    fig = plt.figure()
                    plt.plot([1, 2, 3])
                    PlotSaver.Save(fig)
                    
                    fig = plt.figure()
                    plt.plot([3, 2, 1])
                    PlotSaver.Save(fig)
    
        class ImageSaver_Class:
            """Saves a single figure to a single image file (e.g. .png, .jpg)."""
    
            @staticmethod
            def Save(fig, filename, extension="jpg", dpi=150,
                     verbose=False):
                # only append extension if filename doesn't already end with one
                if not filename.lower().endswith(f".{extension.lower()}"):
                    filename = f"{filename}.{extension}"
                fig.savefig(filename, dpi=dpi)
                plt.close(fig)
                
                if verbose:
                    print(f"Saved image: {filename}")
    
            @staticmethod
            def Test():
                fig1 = plt.figure()
                plt.plot([1, 2, 3])

                ImageSaver = DataPlotting_Classes.FigureSavingFunctions.ImageSaver_Class
                ImageSaver.Save(fig1, "test_image", extension="png",verbose=True)

