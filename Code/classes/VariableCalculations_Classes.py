#!/usr/bin/env python
# coding: utf-8

# In[78]:


# # How to Import to Code Document
########################################
# import os, sys
# sys.path.append(os.path.join(os.path.abspath("../.."), "classes"))

# # Importing
# from VariableCalculations_Classes import VariableCalculations_Classes


# In[81]:


# VariableCalculations_Classes
# ============================================================

# Libraries
import os, sys
import numpy as np

#Class
class VariableCalculations_Classes:

    """
    Class for calculating various variables in atmospheric science given 
    model output data.
    """
    class Thermodynamics:
        """
        Thermodynamic variables.
        """

        # ============================================================
        # ========== Constants (CM1 / Emanuel 1994) ==========
        # ============================================================
        def Constants():
            p0 = 1e5
            Rd, Rv = 287.04, 461.5
            eps = Rd / Rv                       # 0.62197
            Cpd, Cpv = 1005.7, 1870.0 #+-2.5, +-25
            Cpl, Cpi = 4190.0, 2106.0           #+-30,+-6 #Cl=Cpl=Cvl, Ci=Cpi=Cvi
            T0, es0 = 273.15, 611.0
            # latent heat of fusion at 0°C
            #+-0.01 (inside coefficient) (between -100 and 0 deg C)
            Lv0, Lf0, Ls0 = 2.501e6, 0.3337e6, 2.834e6 
            return [p0,Rd,Rv,eps,Cpd,Cpv,Cpl,Cpi,
                    T0,es0,Lv0,Lf0,Ls0]
        [p0,Rd,Rv,eps,Cpd,Cpv,Cpl,Cpi,
                    T0,es0,Lv0,Lf0,Ls0] = Constants()

        # ============================================================
        # ========== Helpers ==========
        # ============================================================

        @classmethod
        def LatentHeat(self, T, phase='vaporization'):
            """Kirchhoff's formula: L(T) = L0 + dCp * (T - T0)."""
            L0, dCp = {'vaporization': (self.Lv0, self.Cpv - self.Cpl),
                       'fusion':       (self.Lf0, self.Cpl - self.Cpi),
                       'sublimation':  (self.Ls0, self.Cpv - self.Cpi)}[phase]
            return L0 + dCp * (T - self.T0)

        @classmethod
        def SaturationVaporPressure(self, T, phase='liquid'):
            """Clausius-Clapeyron with L(T) from Kirchhoff's formula."""
            latentPhase = {'liquid': 'vaporization', 'ice': 'sublimation'}[phase]
            inner = (self.LatentHeat(T, latentPhase) / self.Rv) * (1 / self.T0 - 1 / T)
            return self.es0 * np.exp(inner)

        @classmethod
        def VaporPressure(self, data):
            #rv=eps*(e/(P-e)) ==> e = rv*P/(eps+rv); e==p_v
            e = data.qv * data.prs / (data.qv + self.eps)
            return e

        # ============================================================
        # ========== Calculations ==========
        # ============================================================

        @classmethod
        def AddThetaRho(self, data, condensate=('qc', 'qr', 'qi', 'qs', 'qg'),
                        includeCondensate=True):
            """
            Density Potential Temperature.
            As calculated in Emanuel (1994).
            """
            if includeCondensate:
                ql = sum(data[q] for q in condensate if q in data)
                name, longName = 'th_rho', 'density potential temperature'
            else:
                ql = 0
                name, longName = 'th_v', 'virtual potential temperature'
            thVar = data.th * (1 + data.qv / self.eps) / (1 + data.qv + ql)
            thVar.attrs = {'long_name': longName, 'units': 'K'}
            return data.assign(**{name: thVar})

        @classmethod
        def AddTemperature(self, data):
            T = data.th * (data.prs / self.p0) ** (self.Rd / self.Cpd)
            T.attrs = {'long_name': 'temperature', 'units': 'K'}
            return data.assign(T=T)

        @classmethod
        def AddRelativeHumidity(self, data):
            e = self.VaporPressure(data)
            RH_vapor = e / self.SaturationVaporPressure(data['T'], 'liquid')
            RH_ice = e / self.SaturationVaporPressure(data['T'], 'ice')
            RH_vapor.attrs = {'long_name': 'relative humidity over liquid', 'units': ''}
            RH_ice.attrs = {'long_name': 'relative humidity over ice', 'units': ''}
            return data.assign(RH_vapor=RH_vapor, RH_ice=RH_ice)

        @classmethod
        def AddMoistStaticEnergy(self, data, liquid=('qc', 'qr'),
                                 ice=('qi', 'qs', 'qg'), includeIce=True, g=9.81):
            """
            Moist Static Energy (J/kg).
            As calculated in Emanuel (1994):
                h = (Cpd + rT*Cpl)*T + Lv(T)*rv - Lf(T)*ri + (1 + rT)*g*z
            with includeIce=False dropping the ice terms (ri = 0).
            """
            rv = data.qv
            rl = sum(data[q] for q in liquid if q in data)
            ri = sum(data[q] for q in ice if q in data) if includeIce else 0
            rT = rv + rl + ri

            T = self.AddTemperature(data)['T']
            gZ = g * data['zh'] * 1000                     # zh in km (CM1)

            MSE = ((self.Cpd + rT * self.Cpl) * T
                   + self.LatentHeat(T, 'vaporization') * rv
                   - self.LatentHeat(T, 'fusion') * ri
                   + (1 + rT) * gZ)
            MSE.attrs = {'long_name': 'moist static energy', 'units': 'J kg$^{-1}$'}
            return data.assign(MSE=MSE)
            
        @classmethod
        def AddThetaE(self, data, condensate=('qc', 'qr')):
            """
            Equivalent Potential Temperature.
            As calculated in Emanuel (1994).
            """
            rv,P = data.qv, data.prs
            T = self.AddTemperature(data)['T']
            rt = rv + sum(data[q] for q in condensate if q in data)
            e = self.VaporPressure(data)
            Pd=P-e #<== P=Pd+e
            RH = e / self.SaturationVaporPressure(T, 'liquid')
            divisor = self.Cpd + (self.Cpl * rt)
            th_e = (T * (self.p0 / Pd) ** (self.Rd / divisor)
                       * RH ** (-rv * self.Rv / divisor)
                       * np.exp(self.LatentHeat(T) * rv / (divisor * T)))
            th_e.attrs = {'long_name': 'equivalent potential temperature', 'units': 'K'}
            return data.assign(th_e=th_e)

    class Perturbations:
        """Horizontal-mean perturbations and buoyancy."""

        @staticmethod
        def CalculatePerturbation(data, varName,
                                  horizDims=('xh', 'xf', 'yh', 'yf'), returnMean=False):
            dims = [d for d in data[varName].dims if d in horizDims]
            ref = data[varName].mean(dim=dims)                  # vertical profile
            pert = (data[varName] - ref).rename(varName + '_prime')
            pert.attrs = dict(data[varName].attrs)
            pert.attrs['long_name'] = varName + "'"
            return (pert, ref) if returnMean else pert

        @classmethod
        def AddBuoyancy(self, data, varName='th_v', g=9.81):
            pert, ref = self.CalculatePerturbation(data, varName,
                                                  returnMean=True)
            B = g * pert / ref
            B.attrs = {'long_name': 'buoyancy', 'units': 'm s$^{-2}$'}
            return data.assign(B=B)

        @classmethod
        def AddPerturbations(self, data, varNames=('qv', 'th_v')):
            for v in varNames:
                p = self.CalculatePerturbation(data, v)
                data = data.assign(**{p.name: p})
            return data

    class Dynamics:
        """Dynamical variables."""

        # ============================================================
        # ========== Helpers ==========
        # ============================================================

        @staticmethod
        def HorizontalDivergence(uField, vField, data, xDim='xh', yDim='yh'):
            """
            du/dx + dv/dy for two xr.DataArrays on scalar points, returned as an
            xr.DataArray. Centered differences in the interior, one-sided at edges.
            """
            Diff = NumericalData_Classes.Differentiation_Class
            div = Diff.Divergence(uField.values, vField.values,
                                  uAxis=uField.get_axis_num(xDim),
                                  vAxis=vField.get_axis_num(yDim),
                                  uCoord=data[xDim].values * 1000,   # km -> m (CM1)
                                  vCoord=data[yDim].values * 1000)
            return xr.DataArray(div, dims=uField.dims, coords=uField.coords)

        # ============================================================
        # ========== Calculations ==========
        # ============================================================

        @classmethod
        def AddTKE(self, data, windVars=('uinterp', 'vinterp', 'winterp')):
            """
            Resolved turbulent kinetic energy (m^2/s^2):
                TKE = 0.5 * (u'^2 + v'^2 + w'^2)
            with primes as deviations from the horizontal-mean vertical profile.
            """
            Perts = VariableCalculations_Classes.Perturbations
            TKE = 0.5 * sum(Perts.CalculatePerturbation(data, v) ** 2
                            for v in windVars)
            TKE.attrs = {'long_name': 'resolved turbulent kinetic energy',
                         'units': 'm$^{2}$ s$^{-2}$'}
            return data.assign(TKE=TKE)

        @classmethod
        def AddConvergence(self, data, uVar='uinterp', vVar='vinterp'):
            """
            Horizontal convergence (1/s):
                conv = -(du/dx + dv/dy)
            Positive = convergence, negative = divergence.
            """
            conv = -self.HorizontalDivergence(data[uVar], data[vVar], data)
            conv.attrs = {'long_name': 'horizontal convergence', 'units': 's$^{-1}$'}
            return data.assign(conv=conv)

        @classmethod
        def AddMoistureConvergence(self, data, uVar='uinterp', vVar='vinterp',
                                   qVar='qv', rhoVar='rho'):
            """
            Horizontal moisture flux convergence (kg m^-3 s^-1):
                HMC = -(d(rho*qv*u)/dx + d(rho*qv*v)/dy)
            Positive = moisture convergence.
            """
            rhoq = data[rhoVar] * data[qVar]            # water vapor density
            HMC = -self.HorizontalDivergence(rhoq * data[uVar], rhoq * data[vVar], data)
            HMC.attrs = {'long_name': 'horizontal moisture flux convergence',
                         'units': 'kg m$^{-3}$ s$^{-1}$'}
            return data.assign(HMC=HMC)

        @classmethod
        def AddMoistureGradient(self, data, qVar='qv', rhoVar='rho',
                                xDim='xh', yDim='yh'):
            """
            Horizontal moisture gradient terms (kg m^-4):
                d(rho*qv)/dx, d(rho*qv)/dy
            Consistent with HMC, whose advection part is -(u*d(rho*qv)/dx + v*d(rho*qv)/dy).
            """
            Diff = NumericalData_Classes.Differentiation_Class
            rhoq = data[rhoVar] * data[qVar]            # water vapor density

            dqdx = Diff.Derivative(rhoq.values, axis=rhoq.get_axis_num(xDim),
                                   coord=data[xDim].values * 1000)   # km -> m (CM1)
            dqdy = Diff.Derivative(rhoq.values, axis=rhoq.get_axis_num(yDim),
                                   coord=data[yDim].values * 1000)

            dqdx = xr.DataArray(dqdx, dims=rhoq.dims, coords=rhoq.coords)
            dqdy = xr.DataArray(dqdy, dims=rhoq.dims, coords=rhoq.coords)
            dqdx.attrs = {'long_name': 'zonal moisture gradient', 'units': 'kg m$^{-4}$'}
            dqdy.attrs = {'long_name': 'meridional moisture gradient', 'units': 'kg m$^{-4}$'}
            return data.assign(dqdx=dqdx, dqdy=dqdy)

# #--------------------------------------------------
# #Example Running
# #--------------------------------------------------
# #Load Classes
# mainCodeDirectory = os.path.abspath("/mnt/lustre/koa/koastore/torri_group/air_directory/Projects/3.Moist-Preconditioning-Project/Coding_Directory/Code")
# classDirectory = os.path.join(mainCodeDirectory, "classes")
# sys.path.append(classDirectory)

# from OutputData_Classes import OutputData_Classes
# from InputData_Classes import InputData_Classes
# from NumericalData_Classes import NumericalData_Classes
# from DataPlotting_Classes import DataPlotting_Classes

