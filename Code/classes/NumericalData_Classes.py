#!/usr/bin/env python
# coding: utf-8

# In[78]:


# # How to Import to Code Document
########################################
# import os, sys
# mainCodeDirectory = os.path.abspath("../..")
# path = os.path.join(mainCodeDirectory, "classes")
# sys.path.append(path)

# # Importing
# from NumericalData_Classes import NumericalData_Classes


# In[81]:


# NumericalData_Classes
# ============================================================

# Libraries
import os, sys
import numpy as np
import matplotlib.pyplot as plt

#Class
class NumericalData_Classes:
    
    # Differentiation_Class
    # ============================================================
    class Differentiation_Class:
    
        @staticmethod
        def Derivative(array, axis=0, coord=1):
            """
            array : data of any dimensionality, living on whatever grid it lives on
                    (cell centers for q/u/v/theta, or faces for w -- doesn't matter).
            axis  : the axis of `array` to differentiate along.
            coord : float -> uniform grid spacing (simple case).
                    1D array -> the ACTUAL coordinate of each point in `array` along
                                `axis` (e.g. zh for center-valued fields, zf for
                                face-valued fields). Must be the SAME length as
                                array.shape[axis].
            """
            array = np.asarray(array)
            result = np.zeros_like(array, dtype=float)
            slice_at_axis = lambda s: tuple(
                s if dim == axis else slice(None) for dim in range(array.ndim)
            )
            coord_array = np.asarray(coord)
            is_nonuniform = coord_array.ndim > 0 and coord_array.size > 1
    
            if is_nonuniform:
                if coord_array.shape[0] != array.shape[axis]:
                    raise ValueError(
                        f"coord must have length {array.shape[axis]} (matching array "
                        f"along axis {axis} -- one coordinate per data point), "
                        f"got length {coord_array.shape[0]}"
                    )
                c = coord_array.reshape([-1 if d == axis else 1 for d in range(array.ndim)])
                result[slice_at_axis(slice(1, -1))] = (
                    array[slice_at_axis(slice(2, None))] - array[slice_at_axis(slice(0, -2))]
                ) / (c[slice_at_axis(slice(2, None))] - c[slice_at_axis(slice(0, -2))])
                result[slice_at_axis(0)] = (
                    (array[slice_at_axis(1)] - array[slice_at_axis(0)])
                    / (c[slice_at_axis(1)] - c[slice_at_axis(0)])
                )
                result[slice_at_axis(-1)] = (
                    (array[slice_at_axis(-1)] - array[slice_at_axis(-2)])
                    / (c[slice_at_axis(-1)] - c[slice_at_axis(-2)])
                )
            else:
                d = float(coord)
                result[slice_at_axis(slice(1, -1))] = (
                    array[slice_at_axis(slice(2, None))] - array[slice_at_axis(slice(0, -2))]
                ) / (2 * d)
                result[slice_at_axis(0)] = (array[slice_at_axis(1)] - array[slice_at_axis(0)]) / d
                result[slice_at_axis(-1)] = (array[slice_at_axis(-1)] - array[slice_at_axis(-2)]) / d
    
            return result
    
        @staticmethod
        def SecondDerivative(array, axis=0, coord=1):
            """
            Second derivative of `array` along `axis`. Interior points only --
            edges are left at zero (no one-sided 2nd-derivative stencil, matching
            how Laplacians are normally handled without extra boundary points).
    
            coord : float -> uniform grid spacing.
                    1D array -> the actual coordinate of each point in `array`
                                along `axis` (same length as array.shape[axis]);
                                uses the proper non-uniform-grid 2nd-derivative
                                formula (correct even when spacing is stretched).
            """
            array = np.asarray(array)
            result = np.zeros_like(array, dtype=float)
            slice_at_axis = lambda s: tuple(
                s if dim == axis else slice(None) for dim in range(array.ndim)
            )
            coord_array = np.asarray(coord)
            is_nonuniform = coord_array.ndim > 0 and coord_array.size > 1
    
            if is_nonuniform:
                if coord_array.shape[0] != array.shape[axis]:
                    raise ValueError(
                        f"coord must have length {array.shape[axis]} (matching array "
                        f"along axis {axis}), got length {coord_array.shape[0]}"
                    )
                c = coord_array.reshape([-1 if d == axis else 1 for d in range(array.ndim)])
                h1 = c[slice_at_axis(slice(1, -1))] - c[slice_at_axis(slice(0, -2))]
                h2 = c[slice_at_axis(slice(2, None))] - c[slice_at_axis(slice(1, -1))]
                f_im1 = array[slice_at_axis(slice(0, -2))]
                f_i = array[slice_at_axis(slice(1, -1))]
                f_ip1 = array[slice_at_axis(slice(2, None))]
                result[slice_at_axis(slice(1, -1))] = 2 * (
                    f_im1 / (h1 * (h1 + h2))
                    - f_i / (h1 * h2)
                    + f_ip1 / (h2 * (h1 + h2))
                )
            else:
                d = float(coord)
                result[slice_at_axis(slice(1, -1))] = (
                    array[slice_at_axis(slice(0, -2))]
                    - 2 * array[slice_at_axis(slice(1, -1))]
                    + array[slice_at_axis(slice(2, None))]
                ) / d ** 2
    
            return result
    
        @staticmethod
        def Divergence(f_u, f_v, f_w=None, axis_u=-1, axis_v=-2, axis_w=-3,
                        coord_u=1, coord_v=1, coord_w=1):
            """
            Horizontal or 3D divergence, built from Derivative().
            """
            div = (
                NumericalData_Classes.Differentiation_Class.Derivative(f_u, axis=axis_u, coord=coord_u)
                + NumericalData_Classes.Differentiation_Class.Derivative(f_v, axis=axis_v, coord=coord_v)
            )
            if f_w is not None:
                div = div + NumericalData_Classes.Differentiation_Class.Derivative(f_w, axis=axis_w, coord=coord_w)
            return div
    
        @staticmethod
        def Laplacian(f, axis_x=-1, axis_y=-2, axis_z=None,
                      coord_x=1, coord_y=1, coord_z=1):
            """
            Horizontal or 3D Laplacian, built from SecondDerivative().
        
            f : the scalar field to differentiate.
            axis_z / coord_z : leave axis_z=None for horizontal-only Laplacian
                               (d2f/dx2 + d2f/dy2). Provide axis_z for the full
                               3D Laplacian (+ d2f/dz2).
            """
            lap = (
                NumericalData_Classes.Differentiation_Class.SecondDerivative(f, axis=axis_x, coord=coord_x)
                + NumericalData_Classes.Differentiation_Class.SecondDerivative(f, axis=axis_y, coord=coord_y)
            )
            if axis_z is not None:
                lap = lap + NumericalData_Classes.Differentiation_Class.SecondDerivative(f, axis=axis_z, coord=coord_z)
            return lap
    
        @staticmethod
        def Test():
            print("Test One")
            """
            Testing Simple 1D Derivative
            """
            array = np.array([0, 1, 2, 3])
            result = NumericalData_Classes.Differentiation_Class.Derivative(array, axis=0, coord=[0, 0.25, 0.5, 1])
            print("array  :", array)
            print("d(array)/d(axis=0)  :", result)
    
            print("Test Two")
            """
            Taking Derivative, Divergence, and Laplacian of 2d Fields
            """
            ny, nx = 30, 30
            y = np.linspace(0, 2 * np.pi, ny)
            x = np.linspace(0, 2 * np.pi, nx)
            X, Y = np.meshgrid(x, y)                    # axis 0 = y, axis 1 = x
    
            # Known analytic test fields so we can check against exact answers:
            f_u = np.sin(X)
            f_v = np.cos(Y)
    
            dfv_dy_numeric = NumericalData_Classes.Differentiation_Class.Derivative(f_v, axis=0, coord=y)
            dfv_dy_exact = -np.sin(Y)
    
            div_numeric = NumericalData_Classes.Differentiation_Class.Divergence(f_u, f_v, axis_u=1, axis_v=0,
                                                           coord_u=x, coord_v=y)
            div_exact = np.cos(X) - np.sin(Y)
    
            # --- Figure 1: the two test fields, f_u and f_v ---
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            cs0 = axes[0].contourf(X, Y, f_u, levels=20)
            fig.colorbar(cs0, ax=axes[0], label="f_u")
            axes[0].set_title("f_u = sin(x)")
            axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    
            cs1 = axes[1].contourf(X, Y, f_v, levels=20)
            fig.colorbar(cs1, ax=axes[1], label="f_v")
            axes[1].set_title("f_v = cos(y)")
            axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
            fig.suptitle("Original fields")
    
            # --- Figure 2: numerical vs exact df_v/dy ---
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            cs0 = axes[0].contourf(X, Y, dfv_dy_numeric, levels=20)
            fig.colorbar(cs0, ax=axes[0], label="df_v/dy")
            axes[0].set_title("Numerical df_v/dy")
            axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    
            cs1 = axes[1].contourf(X, Y, dfv_dy_exact, levels=20)
            fig.colorbar(cs1, ax=axes[1], label="df_v/dy")
            axes[1].set_title("Exact df_v/dy = -sin(y)")
            axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
            fig.suptitle("Derivative check")
    
            # --- Figure 3: numerical vs exact divergence ---
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            cs0 = axes[0].contourf(X, Y, div_numeric, levels=20)
            fig.colorbar(cs0, ax=axes[0], label="divergence")
            axes[0].set_title("Numerical divergence")
            axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    
            cs1 = axes[1].contourf(X, Y, div_exact, levels=20)
            fig.colorbar(cs1, ax=axes[1], label="divergence")
            axes[1].set_title("Exact divergence = cos(x) - sin(y)")
            axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
            fig.suptitle("Divergence check")
    
            print("Test Three")
            """
            Taking Laplacian of a 2d Field
            """
            f_lap = np.sin(X) + np.cos(Y)
            lap_numeric = NumericalData_Classes.Differentiation_Class.Laplacian(f_lap, axis_x=1, axis_y=0,
                                                          coord_x=x, coord_y=y)
            lap_exact = -np.sin(X) - np.cos(Y)
    
            # --- Figure 4: numerical vs exact Laplacian ---
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            cs0 = axes[0].contourf(X, Y, lap_numeric, levels=20)
            fig.colorbar(cs0, ax=axes[0], label="laplacian")
            axes[0].set_title("Numerical Laplacian")
            axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    
            cs1 = axes[1].contourf(X, Y, lap_exact, levels=20)
            fig.colorbar(cs1, ax=axes[1], label="laplacian")
            axes[1].set_title("Exact Laplacian = -sin(x) - cos(y)")
            axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
            fig.suptitle("Laplacian check")

    # Integration_Class
    # ============================================================
    class Integration_Class:
    
        @staticmethod
        def Integrate(array, axis=0, coord=1):
            """
            Cumulative integral of `array` along `axis`, using the trapezoidal
            rule. Mirrors Derivative()'s interface/conventions -- same array
            shape in, same array shape out.
    
            array : data of any dimensionality.
            axis  : the axis of `array` to integrate along.
            coord : float -> uniform grid spacing (simple case).
                    1D array -> the ACTUAL coordinate of each point in `array`
                                along `axis` (same length as array.shape[axis]).
    
            Returns an array the same shape as `array`: the running (cumulative)
            integral from the first point up to each point along `axis`. The
            first index along `axis` is always 0 (nothing integrated yet).
            """
            array = np.asarray(array)
            slice_at_axis = lambda s: tuple(
                s if dim == axis else slice(None) for dim in range(array.ndim)
            )
    
            coord_array = np.asarray(coord)
            is_nonuniform = coord_array.ndim > 0 and coord_array.size > 1
    
            if is_nonuniform:
                if coord_array.shape[0] != array.shape[axis]:
                    raise ValueError(
                        f"coord must have length {array.shape[axis]} (matching array "
                        f"along axis {axis}), got length {coord_array.shape[0]}"
                    )
                c = coord_array.reshape([-1 if d == axis else 1 for d in range(array.ndim)])
                dx = c[slice_at_axis(slice(1, None))] - c[slice_at_axis(slice(0, -1))]
            else:
                dx = float(coord)
    
            # trapezoidal rule: each segment's contribution is the average of
            # its two endpoint values times the segment width
            segment_avg = (
                array[slice_at_axis(slice(1, None))] + array[slice_at_axis(slice(0, -1))]
            ) / 2
            segment_integral = segment_avg * dx
    
            result = np.zeros_like(array, dtype=float)
            result[slice_at_axis(slice(1, None))] = np.cumsum(segment_integral, axis=axis)
            return result
    
        @staticmethod
        def Test():
            # known analytic case: integral of cos(x) from 0 is sin(x)
            x = np.linspace(0, 2 * np.pi, 200)
            f = np.cos(x)
    
            numeric = NumericalData_Classes.Integration_Class.Integrate(f, axis=0, coord=x)
            exact = np.sin(x) - np.sin(x[0])
    
            print("max error:", np.max(np.abs(numeric - exact)))
    
            plt.figure()
            plt.plot(x, exact, label="exact: sin(x) - sin(0)")
            plt.plot(x, numeric, '--', label="numeric cumulative integral")
            plt.legend()
            plt.xlabel("x")
            plt.title("Integrate(cos(x)) vs exact")

    # AreaStatistics_Class
    # ============================================================
    class AreaStatistics_Class:
    
        @staticmethod
        def Ultimate_AreaStatistic(data, dims=('t', 'z', 'y', 'x'), avg_over=('t', 'y', 'x'),
                                 func=np.nanmean, **func_kwargs):
            """
            Reduce a NumPy array over selected dimensions by name, using any
            NumPy reduction function that accepts an `axis` argument.
    
            func : np.nanmean, np.nanmedian, np.nansum, np.nanstd, np.nanvar,
                   np.nanmin, np.nanmax, np.nanpercentile (pass q= via func_kwargs), etc.
            """
            axes = tuple(dims.index(d) for d in avg_over)
            out_dims = tuple(d for d in dims if d not in avg_over)
            out = func(data, axis=axes, **func_kwargs)
            return out, out_dims
    
        @staticmethod
        def NanDivide(numerator, denominator):
            """
            Elementwise division that returns NaN wherever the denominator is 0,
            instead of raising a ZeroDivisionError / RuntimeWarning or returning
            inf.
            """
            numerator = np.asarray(numerator, dtype=float)
            denominator = np.asarray(denominator, dtype=float)
    
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(denominator != 0, numerator / denominator, np.nan)
    
            return result
            
        @staticmethod
        def Test():
            arr4d = np.random.rand(3, 4, 5, 6)  # (t, z, y, x)
            out, dims = NumericalData_Classes.AreaStatistics_Class.Ultimate_AreaStatistic(
                arr4d, dims=('t', 'z', 'y', 'x'), avg_over=('z', 'y', 'x'), func=np.nanmean,
            )
            print("input shape :", arr4d.shape)
            print("output shape:", out.shape)
            print("output dims :", dims)

    # Utilities_Class
    # ============================================================
    class Utilities_Class:
        """
        General-purpose numerical helper functions that don't fit the
        differentiation or statistics classes -- elementwise safety wrappers,
        unit conversions, small math utilities, etc.
        """
    
        @staticmethod
        def nandivide(numerator, denominator):
            """
            Elementwise division that returns NaN wherever the denominator is 0.
            """
            numerator = np.asarray(numerator, dtype=float)
            denominator = np.asarray(denominator, dtype=float)
    
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(denominator != 0, numerator / denominator, np.nan)
    
            return result
    
        @staticmethod
        def Test():
            numerator = [0,1,2,3]
            denominator = [1,0,0,1]
    
            division = NumericalData_Classes.Utilities_Class.nandivide(numerator,denominator)
            
            print("numerator :", numerator)
            print("denominator :", denominator)
            print("division :", division)

