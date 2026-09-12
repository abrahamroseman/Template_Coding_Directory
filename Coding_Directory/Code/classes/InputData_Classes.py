#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # How to Import to Code Document
########################################
# import os, sys
# mainCodeDirectory = os.path.abspath("../..")
# path = os.path.join(mainCodeDirectory, "classes")
# sys.path.append(path)

# # Importing
# from InputData_Classes import InputData_Classes


# In[ ]:


# InputData_Classes
# ============================================================

#Libraries
import os
from datetime import timedelta
from glob import glob
from tqdm import tqdm

import numpy as np

import xarray as xr
from netCDF4 import Dataset

#Class
class InputData_Classes:
    
    # ModelData_Class
    # ============================================================
    class ModelData_Class_OldExample:
        def __init__(self, mainDirectory, scratchDirectory, simulationNumber):
            self.mainDirectory = mainDirectory
            self.scratchDirectory = scratchDirectory
            self.simulationNumber = simulationNumber
            
            # Initialize directories and metadata
            (self.dataDirectory, 
             self.parcelDirectory, 
             self.res, 
             self.t_res, 
             self.Np_str, 
             self.Nz_str) = self.GetDataDirectories()
    
            # Load coordinate data only (lightweight)
            self.GetCoordinateData()
            [self.dt, self.dz, self.dy, self.dx] = self.GetGridSpacing()
            self.Np = self.GetCoordinateParcel()
            self.timeStrings = self.GetTimeStrings(self.time)
    
            self.time_hrs = np.arange(self.Ntime)*self.dt/3600+6
            [self.kms, self.mins] = self.GetIndexMeasurements()
    
            # Load Variable Names
            self.varList = self.GetVariableNames()
    
            # Print summary
            self.Summary()
    
        # ============================================================
        # ========== Data Loading Functions ==========
        # ============================================================
    
        def GetDataDirectories(self):
            """Return directory paths and metadata based on simulation number."""
            if self.simulationNumber == 1:
                Directory = os.path.join(self.mainDirectory, 'Model/cm1r20.3/run/MODEL_OUTPUT/Simulation_One/')
                res, t_res, Np_str, Nz_str = '1km', '5min', '1e6', '34'
            elif self.simulationNumber == 2:
                Directory = os.path.join(self.mainDirectory, 'Model/cm1r20.3/run/MODEL_OUTPUT/Simulation_Two/')
                res, t_res, Np_str, Nz_str = '1km', '1min', '50e6', '95'
            elif self.simulationNumber == 3:
                Directory = os.path.join(self.mainDirectory, 'Model/cm1r20.3/run/MODEL_OUTPUT/Simulation_Three/')
                res, t_res, Np_str, Nz_str = '1km', '3min', '20e6', '95'
            elif self.simulationNumber == 4:
                Directory = os.path.join(self.mainDirectory, 'Model/cm1r20.3/run/MODEL_OUTPUT/Simulation_Four/')
                res, t_res, Np_str, Nz_str = '1km', '3min', '20e6', '95'
            elif self.simulationNumber == 5:
                Directory = os.path.join(self.mainDirectory, 'Model/cm1r20.3/run/MODEL_OUTPUT/Simulation_Five/')
                res, t_res, Np_str, Nz_str = '1km', '1min', '50e6', '95'
            # elif self.simulationNumber == 6:
            #     Directory = os.path.join(self.mainDirectory, 'Model/cm1r20.3/run/MODEL_OUTPUT/Simulation_Six/')
            #     res, t_res, Np_str, Nz_str = '0.25km', '1min', '50e6', '95'
            else:
                raise ValueError("Invalid simulationNumber (must be 1, 2, or 3).")
    
            dataDirectory = os.path.join(Directory, f"cm1out_{res}_{t_res}_{Nz_str}nz.nc")
            #########
            if not os.path.exists(dataDirectory):
                self.is_data_timestepbytimestep = True
                dataDirectory = os.path.join(Directory, f"cm1out_000001.nc")
            else:
                self.is_data_timestepbytimestep = False
            #########
            parcelDirectory = os.path.join(Directory, f"cm1out_pdata_{res}_{t_res}_{Np_str}np.nc")
            return dataDirectory, parcelDirectory, res, t_res, Np_str, Nz_str
    
        def GetCoordinateData(self):
            """
            Extract coordinate arrays (time, zf, zh, yf, yh, xf, xh) 
            from the CM1 dataset and immediately close the file.
            """
            with xr.open_dataset(self.dataDirectory, decode_timedelta=True) as ds:
                coords = ['time', 'zf', 'zh', 'yf', 'yh', 'xf', 'xh']
                extracted = {k: ds[k].values for k in coords}
    
            if self.is_data_timestepbytimestep: #*
                self.all_files = self.GetAllTimestepFilePaths(self.dataDirectory)
                extracted['time'] = np.array(self.GetTimeCoordinate(self.all_files))*1e9
    
            # Assign coordinate arrays and their lengths
            for k, v in extracted.items():
                setattr(self, k, v)
                setattr(self, f"N{k}", len(v))
                
            return extracted
    
        def GetCoordinateParcel(self):
            """
            Extract coordinate arrays (time, zf, zh, yf, yh, xf, xh) 
            from the CM1 lagrangian parcel dataset and immediately close the file.
            """
            with xr.open_dataset(self.parcelDirectory, decode_timedelta=True) as ds:
                p = ds['xh'].values
                Np = len(p)
            return Np
    
        def GetGridSpacing(self):
            dt=(self.time[1]-self.time[0]).item()/1e9 #secs
            dz = np.diff(self.zf) * 1000
            dy = (self.yh[1].item() - self.yh[0].item()) * 1000
            dx = (self.xh[1].item() - self.xh[0].item()) * 1000
            return dt, dz, dy, dx
    
        def GetIndexMeasurements(self):
            kms=np.argmax(self.xh-self.xh[0] >= 1)
            mins=1/(self.dt/60)
            return kms, mins
    
        def GetTimeStrings(self, times):
            """Convert CM1 time array (nanoseconds) to formatted strings."""
            return [str(timedelta(seconds=float(s))).replace(":", "-") for s in times / 1e9]
    
        def GetVariableNames(self):
            """Get list of variable names available in the CM1 dataset."""
            with xr.open_dataset(self.dataDirectory, decode_timedelta=True) as ds:
                varList = list(ds.data_vars)
            return varList
    
        # ============================================================
        # ========== Additionally code when dealing with timestep by timestep data ==========
        # ============================================================
    
        def GetAllTimestepFilePaths(self,dataDirectory):
            # 1. Split the path into directory and filename
            directory = os.path.dirname(dataDirectory)
            filename = os.path.basename(dataDirectory)
            
            # 2. Identify prefix and extension
            start_idx = filename.find('_')
            end_idx = filename.rfind('.')
            
            if start_idx == -1 or end_idx == -1:
                return [dataDirectory] if os.path.exists(dataDirectory) else []
            
            prefix = filename[:start_idx + 1]
            extension = filename[end_idx:]
            
            if not os.path.isdir(directory):
                return []
            
            # 3. List and Filter: Remove files containing 'pdata' or 'stats'
            all_files = [
                os.path.join(directory, f)               
                for f in os.listdir(directory)           
                if f.startswith(prefix) 
                and f.endswith(extension)
                and 'pdata' not in f   # Excludes pdata files
                and 'stats' not in f   # Excludes stats files
            ]
            
            # 4. Sort alphabetically
            all_files.sort()
            
            return all_files
        
        def GetTimeCoordinate(self,all_files):
            t0 = int(Dataset(all_files[0], 'r').variables['time'][:])
            t1 = int(Dataset(all_files[1], 'r').variables['time'][:])
            dt = t1 - t0
            times = [t0 + (i * dt) for i in range(len(all_files))]
            return times
            
        # ============================================================
        # ========== On-demand Variable Access ==========
        # ============================================================
    
        def OpenData(self, decode_timedelta=True, verbose=False):
            """
            Opens full data netcdf file
            Depreciated for simulations that is output timestep by timestep
            Instead use DataManager.GetTimestepData from the DataManager_Class class
            """
            #EXAMPLE: ds = ModelData.OpenData()
            #         ...
            #         del ds
    
            if self.is_data_timestepbytimestep:
                pbar = tqdm(total=len(self.all_files), desc="Opening files")
    
                def update_pbar(ds):
                    pbar.update(1)
                    return ds
                
                data = xr.open_mfdataset(self.all_files,
                                         preprocess=update_pbar,
                                         decode_timedelta=True, 
                                         concat_dim="time", combine="nested",coords="minimal", compat="override",parallel=True)
                pbar.close()
            else:
                data = xr.open_dataset(self.dataDirectory, decode_timedelta=decode_timedelta)
                if verbose: print(f"Opened dataset: {self.dataDirectory}")
            return data
    
        def OpenData_SingleTime(self, t, decode_timedelta=True, verbose=False):
            """
            Opens full data netcdf file
            Depreciated for simulations that is output timestep by timestep
            Instead use DataManager.GetTimestepData from the DataManager_Class class
            """
            all_files = self.GetAllTimestepFilePaths(self.dataDirectory)
            data = xr.open_dataset(all_files[t], decode_timedelta=decode_timedelta)
            if verbose: print(f"Opened dataset: {all_files[t]}")
            return data
    
        def OpenParcel(self, decode_timedelta=True):
            """
            Opens full lagrangian parcel data netcdf file
            """
            #EXAMPLE: ds = ModelData.OpenParcel()
            #         ...
            #         del ds
            parcel = xr.open_dataset(self.parcelDirectory, decode_timedelta=decode_timedelta)
            print(f"Opened dataset: {self.parcelDirectory}")
            return parcel
    
        def SubsetDataVars(self, data):
            varList = ["thflux", "qvflux", "tsk", "cape", 
                       "cin", "lcl", "lfc", "th",
                       "prs", "rho", "qv", "qc",
                       "qr", "qi", "qs","qg", 
                       "buoyancy", "uinterp", "vinterp", "winterp",]
            
            varList += ["ptb_hadv", "ptb_vadv", "ptb_hidiff", "ptb_vidiff",
                        "ptb_hturb", "ptb_vturb", "ptb_mp", "ptb_rdamp", 
                        "ptb_rad", "ptb_div", "ptb_diss",]
            
            varList += ["qvb_hadv", "qvb_vadv", "qvb_hidiff", "qvb_vidiff", 
                        "qvb_hturb", "qvb_vturb", "qvb_mp",]
            
            varList += ["wb_hadv", "wb_vadv", "wb_hidiff", "wb_vidiff",
                        "wb_hturb", "wb_vturb", "wb_pgrad", "wb_rdamp", "wb_buoy",]
        
            # Filter only available variables in the dataset
            available_vars = [v for v in varList if v in data.variables]
        
            if not available_vars:
                raise ValueError("None of the requested variables were found in the dataset.")
        
            return data[available_vars]
    
        def GetVariable(self, varName, isel=None):
            #EXAMPLE: w = ModelData.GetVariable('winterp', isel={'time': slice(0,2), 'zh': 0, 'yh': 0, 'xh': 0}) #example getting a variable
            """
            Open the full NetCDF file, extract a variable (optionally subset via .isel), 
            then close immediately. Returns the variable data as a NumPy array.
    
            Parameters
            ----------
            varName : str
                Name of the variable to extract.
            isel : dict, optional
                Dictionary of indices to select (e.g., {'time': 0, 'zh': slice(0,10)}).
            decode_timedelta : bool, optional
                Whether to decode CF-style timedelta coordinates (default: True).
            """
            with xr.open_dataset(self.dataDirectory, decode_timedelta=True) as ds:
                if varName not in ds.variables:
                    raise KeyError(f"Variable '{varName}' not found in dataset.")
                da = ds[varName]
                if isel is not None:
                    da = da.isel(**isel)
                varData = da.data  # load into memory before closing
            return varData
    
        # ============================================================
        # === Information ========================================
        # ============================================================
    
        def Summary(self):
            """Print a summary of the simulation configuration."""
            print("=== CM1 Data Summary ===")
            print(f" Simulation #:   {self.simulationNumber}")
            print(f" Resolution:     {self.res}")
            print(f" Time step:      {self.t_res}")
            print(f" Vertical levels:{self.Nz_str}")
            print(f" Parcels:        {self.Np_str}")
            print(f" Data file:      {self.dataDirectory}")
            print(f" Parcel file:    {self.parcelDirectory}")
            print(f" Time steps:     {len(self.time)}")
            print("=========================","\n")

    # GeneralModelData_Class
    # ============================================================
    class InputData_Class_developing:
        """
        General version of ModelData_Class.
    
        dataDirectory : directory holding the data file(s)
        filePattern   : glob pattern that matches the data file(s) within
                      dataDirectory, e.g. "cm1out_*.nc" or "cm1out_1km_5min_34nz.nc".
                      Matched files are sorted alphabetically to define time order,
                      so zero-padded numbering (modeldata_000001.nc, ...) works naturally.
        metaData      : optional dict of any descriptive info you want carried
                      around with the data (e.g. {"resolution": "1km", "region": "forest"). 
        coordNames    : which coordinate variables to extract from the first file
                      (defaults to the CM1 set).
        timeCoord     : name of the time coordinate (default "time").
        """
    
        def __init__(self, dataDirectory, filePattern,
                     metaData=None,
                     coordNames=("time", "zf", "zh", "yf", "yh", "xf", "xh"),
                     timeCoord="time",
                     verbose=True):
            self.dataDirectory = dataDirectory
            self.filePattern = filePattern
            self.metaData = metaData or {}
            self.coordNames = list(coordNames)
            self.timeCoord = timeCoord
            self.verbose = verbose
    
            # Attach metadata entries as attributes (self.res, self.t_res, ...)
            for key, value in self.metaData.items():
                setattr(self, key, value)
    
            # --- Discover files (sorted -> defines time order)
            self.fileList = self.GetFileList()
            self.is_data_timestepbytimestep = len(self.fileList) > 1
    
            # --- Load coordinate data (lightweight; from first file)
            self.GetCoordinateData()
    
            # --- Time bookkeeping
            if self.is_data_timestepbytimestep:
                # each file holds one (or few) timesteps; build the full time
                # coordinate by reading one value per file
                self.time = self.GetTimeCoordinate(self.fileList)
                self.Ntime = len(self.time)
            self.timeStrings = self.GetTimeStrings(self.time)
    
            # --- Grid spacing (only for coords that exist)
            self.dt, self.dz, self.dy, self.dx = self.GetGridSpacing()
    
            # --- Variable names (from first file)
            self.varList = self.GetVariableNames()
    
            if self.verbose:
                self.Summary()
    
        # ============================================================
        # ========== Data Discovery ==========
        # ============================================================
    
        def GetFileList(self):
            """Find all files matching filePattern in dataDirectory, sorted."""
            fileList = sorted(glob(os.path.join(self.dataDirectory, self.filePattern)))
            if not fileList:
                raise FileNotFoundError(
                    f"No files found in {self.dataDirectory} matching {self.filePattern}")
            return fileList
    
        # ============================================================
        # ========== Data Loading Functions ==========
        # ============================================================
    
        def GetCoordinateData(self):
            """
            Extract the requested coordinate arrays from the FIRST file and
            immediately close it. Each coordinate becomes an attribute
            (self.time, self.zh, ...) along with its length (self.Ntime, self.Nzh, ...).
            Coordinates listed in coordNames but missing from the file are skipped
            (with a note if verbose).
            """
            extracted = {}
            with xr.open_dataset(self.fileList[0], decode_timedelta=True) as ds:
                for k in self.coordNames:
                    if k in ds:
                        extracted[k] = ds[k].values
                    elif self.verbose:
                        print(f"Note: coordinate '{k}' not found in {self.fileList[0]}; skipping.")
    
            for k, v in extracted.items():
                setattr(self, k, v)
                setattr(self, f"N{k}", len(np.atleast_1d(v)))
    
            return extracted
    
        def GetTimeCoordinate(self, fileList):
            """
            Build the full time coordinate for timestep-by-timestep data by
            reading the time value from each file (first entry per file).
            Returns a numpy array with the same dtype convention as the files.
            """
            times = []
            for f in fileList:
                with xr.open_dataset(f, decode_timedelta=True) as ds:
                    times.append(np.atleast_1d(ds[self.timeCoord].values)[0])
            return np.array(times)
    
        def GetTimeStrings(self, times):
            """
            Convert the time coordinate into filesystem-safe strings
            (e.g. "0-05-00" for 5 minutes), matching the original class's
            str(timedelta).replace(":", "-") convention. Handles both
            numpy timedelta64 arrays and raw nanosecond numbers.
            """
            times = np.asarray(times)
            if np.issubdtype(times.dtype, np.timedelta64):
                seconds = times / np.timedelta64(1, "s")
            else:
                seconds = times / 1e9  # assume nanoseconds, as in the original class
            return [str(timedelta(seconds=float(s))).replace(":", "-") for s in seconds]
    
        def GetGridSpacing(self):
            """
            Compute dt (s), dz (m array), dy (m), dx (m) from whichever coordinates
            are available; any that can't be computed come back as None.
            Assumes the original class's unit conventions (z/y/x coords in km).
            """
            dt = dz = dy = dx = None
    
            if hasattr(self, "time") and len(np.atleast_1d(self.time)) > 1:
                t = np.asarray(self.time)
                if np.issubdtype(t.dtype, np.timedelta64):
                    dt = (t[1] - t[0]) / np.timedelta64(1, "s")
                else:
                    dt = (t[1] - t[0]).item() / 1e9
    
            if hasattr(self, "zf") and len(np.atleast_1d(self.zf)) > 1:
                dz = np.diff(self.zf) * 1000
            if hasattr(self, "yh") and len(np.atleast_1d(self.yh)) > 1:
                dy = (self.yh[1].item() - self.yh[0].item()) * 1000
            if hasattr(self, "xh") and len(np.atleast_1d(self.xh)) > 1:
                dx = (self.xh[1].item() - self.xh[0].item()) * 1000
    
            return dt, dz, dy, dx
    
        def GetVariableNames(self):
            """Get list of data-variable names from the first file."""
            with xr.open_dataset(self.fileList[0], decode_timedelta=True) as ds:
                varList = list(ds.data_vars)
            return varList
    
        # ============================================================
        # === Information ========================================
        # ============================================================
    
        def Summary(self):
            """Print a summary of the data configuration."""
            print("=== General Model Data Summary ===")
            print(f" dataDirectory:  {self.dataDirectory}")
            print(f" filePattern:    {self.filePattern}")
            print(f" # files:        {len(self.fileList)}")
            print(f" timestep-by-timestep: {self.is_data_timestepbytimestep}")
            for key, value in self.metaData.items():
                print(f" {key}: {value}")
            print(f" Time steps:     {len(np.atleast_1d(self.time))}")
            print(f" dt, dy, dx:     {self.dt}, {self.dy}, {self.dx}")
            print(f" Variables:      {self.varList}")
            print("=========================", "\n")

