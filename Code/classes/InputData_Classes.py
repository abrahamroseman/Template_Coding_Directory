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

# Libraries
import os
from datetime import timedelta
from glob import glob
from types import SimpleNamespace

import numpy as np
import xarray as xr; import h5py; import json
import netCDF4

class InputData_Classes:

    class Model_InputData_Class:
        """
        Class for loading and reading model data for Cloud Model One (CM1),
        or any similarly-structured model, across multiple data types
        (e.g. "eulerianData", "lagrangianData") in a single call.

        dataDirectory, filePattern, coordNames : manual mode. Provide these
                        directly to load a single dataset (stored as self.data).
        pathResolverClass, resolverName, pathResolverKwargs : resolver mode.
                        pathResolverClass exposes named resolver functions
                        (see PathResolvers). The chosen resolver returns a dict:
                            { dataTypeName: {"dataDirectory": ..., "filePattern": ...,
                                              "coordNames": (...), ...}, ... }
                        Each dataTypeName becomes an attribute on this object
                        (self.eulerianData, self.lagrangianData, ...) holding its own data.
        dataTypes     : optional list to filter which dataTypeNames from the
                        resolver's dict actually get loaded (default: all).
        initialTimeLT : optional override for the initial local time (decimal
                        hours) used to build timeHoursLT. Not every model's
                        files carry this info, so it can be supplied here
                        instead — either a single number (applied to every
                        dataType) or a dict keyed by dataTypeName. Takes
                        priority over anything the resolver's config provides.
        """

        def __init__(self, dataDirectory=None, filePattern=None, coordNames=None,
                     pathResolverClass=None,
                     resolverName="CloudModelOne", pathResolverKwargs=None,
                     dataTypes=None,
                     timeCoord="time",
                     initialTimeLT=None,
                     metaData=None,
                     verbose=False):

            self.metaData = metaData or {}
            self.timeCoord = timeCoord
            self.verbose = verbose

            for key, value in self.metaData.items():
                setattr(self, key, value)

            if pathResolverClass is None:
                pathResolverClass = InputData_Classes.PathResolvers

            [dataTypeConfig] = self.ResolvePaths(
                dataDirectory, filePattern, coordNames,
                pathResolverClass, resolverName, pathResolverKwargs)

            if dataTypes is not None:
                dataTypeConfig = {k: v for k, v in dataTypeConfig.items() if k in dataTypes}

            self.dataTypes = list(dataTypeConfig.keys())

            for dataTypeName, config in dataTypeConfig.items():
                userInitialTimeLT = (initialTimeLT.get(dataTypeName) if isinstance(initialTimeLT, dict)
                                      else initialTimeLT)
                [dataset] = self.LoadDataset(config, self.timeCoord, userInitialTimeLT, self.verbose)
                setattr(self, dataTypeName, dataset)

            if self.verbose:
                self.Summary()

        # ============================================================
        # ========== Path Resolution ==========
        # ============================================================

        def ResolvePaths(self, dataDirectory, filePattern, coordNames,
                          pathResolverClass, resolverName, pathResolverKwargs):
            """
            Manual mode: if dataDirectory/filePattern given directly, wrap them
            as a single dataType named "data".
            Resolver mode: otherwise, look up resolverName in pathResolverClass
            and call it; it must return a dict of dataTypeName -> config dict.
            """
            if dataDirectory is not None and filePattern is not None:
                resolvedCoordNames = coordNames or ("time", "zf", "zh", "yf", "yh", "xf", "xh")
                dataTypeConfig = {
                    "data": {
                        "dataDirectory": dataDirectory,
                        "filePattern": filePattern,
                        "coordNames": resolvedCoordNames,
                    }
                }
                return [dataTypeConfig]

            if pathResolverClass is None or resolverName is None:
                raise ValueError(
                    "Must provide either (dataDirectory and filePattern) "
                    "or (pathResolverClass and resolverName).")

            [resolverFunction] = pathResolverClass.GetResolver(resolverName)
            pathResolverKwargs = pathResolverKwargs or {}
            [dataTypeConfig] = resolverFunction(**pathResolverKwargs)

            if not isinstance(dataTypeConfig, dict):
                raise ValueError(
                    "Resolver must return a dict mapping dataTypeName -> config dict.")

            return [dataTypeConfig]

        # ============================================================
        # ========== Per-Dataset Loading ==========
        # ============================================================

        def LoadDataset(self, config, timeCoord, userInitialTimeLT, verbose):
            """
            Load one dataset (e.g. "eulerianData" or "lagrangianData") from its config
            dict and return it as a SimpleNamespace bundling fileList,
            coordinates, time info, grid spacing, local time, and variable names.
            """
            dataDirectory = config["dataDirectory"]
            filePattern = config["filePattern"]
            excludePatterns = config.get("excludePatterns")
            coordNames = list(config.get("coordNames", ()))

            [fileList] = self.GetFileList(dataDirectory, filePattern, excludePatterns)
            isDataTimestepByTimestep = len(fileList) > 1

            dataset = SimpleNamespace(
                dataDirectory=dataDirectory,
                filePattern=filePattern,
                fileList=fileList,
                isDataTimestepByTimestep=isDataTimestepByTimestep,
                coordNames=coordNames,
            )

            [coordinateData] = self.GetCoordinateData(fileList, coordNames, verbose)
            for key, value in coordinateData.items():
                setattr(dataset, key, value)
                setattr(dataset, f"N{key}", len(np.atleast_1d(value)))

            if isDataTimestepByTimestep:
                [time] = self.GetTimeCoordinate(fileList, timeCoord)
                dataset.time = time
                dataset.Ntime = len(time)

            [timeStrings] = self.GetTimeStrings(dataset.time)
            dataset.timeStrings = timeStrings

            [gridSpacing] = self.GetGridSpacing(dataset, timeCoord)
            dataset.gridSpacing = gridSpacing

            [gridsPerKm, gridsPerMin] = self.GetGridsPerUnit(gridSpacing)
            dataset.gridsPerKm = gridsPerKm
            dataset.gridsPerMin = gridsPerMin

            [initialTimeLT, timeHoursLT] = self.GetLocalTime(
                fileList, config, userInitialTimeLT, dataset.Ntime, gridSpacing)
            dataset.initialTimeLT = initialTimeLT
            dataset.timeHoursLT = timeHoursLT

            [varList] = self.GetVariableNames(fileList)
            dataset.varList = varList

            dataset.domainSubsetBounds = config.get("domainSubsetBounds")
            
            dataset.OpenData = lambda file=None, **kwargs: (
                InputData_Classes.DataOpeners.OpenData(fileList, concatDim=timeCoord, file=file, **kwargs)[0])
            dataset.SaveH5 = lambda data, file, label, **kwargs: (
                InputData_Classes.DataExporters.SaveH5(dataset, data, file, label, **kwargs))
            dataset.LoadH5 = lambda label, file=None: (
                InputData_Classes.DataExporters.LoadH5(dataset, label, file)[0])
            return [dataset]

        # ============================================================
        # ========== Data Discovery ==========
        # ============================================================

        def GetFileList(self, dataDirectory, filePattern, excludePatterns=None):
            """Find all files matching filePattern in dataDirectory, sorted,
            excluding any that also match excludePatterns (e.g. other CM1 output
            types like pdata/stats that happen to share the same glob prefix)."""
            fileList = sorted(glob(os.path.join(dataDirectory, filePattern)))

            if excludePatterns:
                [fileList] = self.FilterExcluded(fileList, dataDirectory, excludePatterns)

            if not fileList:
                raise FileNotFoundError(
                    f"No files found in {dataDirectory} matching {filePattern}"
                    + (f" (after excluding {excludePatterns})" if excludePatterns else ""))
            return [fileList]

        def FilterExcluded(self, fileList, dataDirectory, excludePatterns):
            """Helper: drop any files matching excludePatterns from fileList."""
            excludedFiles = set()
            for pattern in excludePatterns:
                excludedFiles.update(glob(os.path.join(dataDirectory, pattern)))
            filtered = [f for f in fileList if f not in excludedFiles]
            return [filtered]

        # ============================================================
        # ========== Data Loading Functions ==========
        # ============================================================

        def GetCoordinateData(self, fileList, coordNames, verbose):
            """
            Extract the requested coordinate arrays from the FIRST file.
            Coordinates listed in coordNames but missing from the file are
            skipped (with a note if verbose).
            """
            extracted = {}
            with xr.open_dataset(fileList[0], decode_timedelta=True) as ds:
                for k in coordNames:
                    if k in ds:
                        extracted[k] = ds[k].values
                    elif verbose:
                        print(f"Note: coordinate '{k}' not found in {fileList[0]}; skipping.")
            return [extracted]

        def GetTimeCoordinate(self, fileList, timeCoord, method="constructed", dtSeconds=None):
            """
            Time coordinate (timedelta64[ns]), one entry per file.
            method : "xarray" | "netCDF4" (read every file) |
                     "constructed" (t0 + arange(N)*dt; checked against the last file)
            """
            def Read(f):
                with netCDF4.Dataset(f, "r") as nc:
                    return float(np.atleast_1d(nc.variables[timeCoord][:])[0])

            if method == "xarray":
                seconds = [xr.open_dataset(f, decode_timedelta=True)[timeCoord].values.ravel()[0]
                           / np.timedelta64(1, "s") for f in fileList]
            elif method == "netCDF4":
                seconds = [Read(f) for f in fileList]
            elif method == "constructed":
                t0 = Read(fileList[0])
                dt = Read(fileList[1]) - t0; print(dt)#*
                seconds = t0 + np.arange(len(fileList)) * dt
                if not np.isclose(seconds[-1], Read(fileList[-1])):
                    raise ValueError("Files missing or unevenly spaced; use method='netCDF4'.")
            else:
                raise ValueError(f"Unknown method '{method}'")

            return [(np.asarray(seconds) * 1e9).astype("timedelta64[ns]")]

        def GetTimeStrings(self, times):
            """
            Convert the time coordinate into filesystem-safe strings
            (e.g. "0-05-00" for 5 minutes).
            """
            times = np.asarray(times)
            if np.issubdtype(times.dtype, np.timedelta64):
                seconds = times / np.timedelta64(1, "s")
            else:
                seconds = times / 1e9  # assume nanoseconds
            timeStrings = [str(timedelta(seconds=float(s))).replace(":", "-") for s in seconds]
            return [timeStrings]

        def GetGridSpacing(self, dataset, timeCoord):
            """
            Compute spacing for every coordinate on dataset. Returns a
            SimpleNamespace with attributes named "d" + coordName (e.g.
            "dtime", "dzf", "dxh").
            """
            gridSpacing = SimpleNamespace()

            for coordName in dataset.coordNames:
                if not hasattr(dataset, coordName):
                    continue
                values = getattr(dataset, coordName)
                if len(np.atleast_1d(values)) <= 1:
                    continue

                if coordName == timeCoord:
                    spacing = self.GetTimeSpacing(values)
                else:
                    spacing = self.GetSpatialSpacing(values)

                setattr(gridSpacing, f"d{coordName}", spacing)

            return [gridSpacing]
        
        def GetGridsPerUnit(self, gridSpacing):
            """
            gridsPerKms: index of the first vertical level where dz >= 1000 m
            (useful for slicing to km-scale resolution). gridsPerMins: number of
            timesteps per minute, from dt.

            Useful if for example wanting to determine how many gridboxes 
            are a radius of a (certain number of kms) * gridsPerKms.
            """
            if not (hasattr(gridSpacing, "dxh") and hasattr(gridSpacing, "dtime")):
                return [None, None]
        
            gridsPerKm = 1 / (gridSpacing.dxh / 1000)
            gridsPerMin = 1 / (gridSpacing.dtime / 60)
            return [gridsPerKm, gridsPerMin]

        def GetTimeSpacing(self, values):
            """Helper: spacing (s) for a time coordinate, scalar or diff array."""
            t = np.asarray(values)
            if np.issubdtype(t.dtype, np.timedelta64):
                diffs = np.diff(t) / np.timedelta64(1, "s")
            else:
                diffs = np.diff(t) / 1e9  # assume nanoseconds
            return self.CollapseIfUniform(diffs)

        def GetSpatialSpacing(self, values):
            """Helper: spacing (m) for a spatial coordinate, scalar or diff array.
            Assumes km input, per CM1 convention."""
            diffs = np.diff(np.asarray(values)) * 1000
            return self.CollapseIfUniform(diffs)

        def CollapseIfUniform(self, diffs):
            """Helper: return a single scalar if all diffs are equal (evenly-spaced
            coordinate, e.g. xh/yh), otherwise the full diff array (unevenly-spaced,
            e.g. staggered zf levels)."""
            if np.allclose(diffs, diffs[0]):
                return diffs[0].item()
            return diffs

        def GetLocalTime(self, fileList, config, userInitialTimeLT, Ntime, gridSpacing):
            """
            Determine initialTimeLT (decimal hours) and build timeHoursLT from
            it, or return (None, None) if it can't be determined. Priority:
              1. userInitialTimeLT, passed explicitly by the caller — for
                 models/files that don't carry this info at all.
              2. config["initialTimeLT"], a literal hardcoded by the resolver.
              3. config["initialTimeAttrs"] = {"hour": attrName, "minute": attrName},
                 read from the first file's global attributes (e.g. CM1).
            Needs gridSpacing.dtime to build timeHoursLT; if that's missing
            (e.g. only one timestep), returns (initialTimeLT, None).
            """
            initialTimeAttrs = config.get("initialTimeAttrs")
            if userInitialTimeLT is not None:
                initialTimeLT = userInitialTimeLT
            elif "initialTimeLT" in config:
                initialTimeLT = config["initialTimeLT"]
            elif initialTimeAttrs:
                with xr.open_dataset(fileList[0], decode_timedelta=True) as ds:
                    hour = ds.attrs.get(initialTimeAttrs.get("hour"), 0)
                    minute = ds.attrs.get(initialTimeAttrs.get("minute"), 0)
                initialTimeLT = hour + minute / 60
            else:
                return [None, None]

            if not hasattr(gridSpacing, "dtime"):
                return [initialTimeLT, None]

            timeHoursLT = np.arange(Ntime) * (gridSpacing.dtime / 3600) + initialTimeLT
            return [initialTimeLT, timeHoursLT]

        def GetVariableNames(self, fileList):
            """Get list of data-variable names from the first file."""
            with xr.open_dataset(fileList[0], decode_timedelta=True) as ds:
                varList = list(ds.data_vars)
            return [varList]

        # ============================================================
        # === Information ========================================
        # ============================================================

        def Summary(self):
            """Print a summary of every loaded dataset."""
            print("=== Model Input Data Summary ===")
            for key, value in self.metaData.items():
                print(f" {key}: {value}")
            for i, dataTypeName in enumerate(self.dataTypes):
                dataset = getattr(self, dataTypeName)
                self.PrintDatasetSummary(dataTypeName, dataset)
                if i < len(self.dataTypes) - 1:
                    print()
            print("=" * 32, "\n")

        def PrintDatasetSummary(self, dataTypeName, dataset):
            """Helper: print one dataset's block within the overall Summary."""
            print(f" --- {dataTypeName} ---")
            print(f"  dataDirectory: {dataset.dataDirectory}")
            print(f"  filePattern:   {dataset.filePattern}")
            print(f"  # files:       {len(dataset.fileList)}")
            print(f"  timestep-by-timestep: {dataset.isDataTimestepByTimestep}")
            print(f"  Time steps:    {len(np.atleast_1d(dataset.time))}")
            print(f"  Initial LT:    {dataset.initialTimeLT}")
            print(f"  Variables:     {dataset.varList}")

    class PathResolvers:
        """Namespace of named path-resolver functions, selectable by string key.
        Each resolver returns a dict: dataTypeName -> config dict, with keys
        "dataDirectory", "filePattern", "coordNames", and optionally
        "excludePatterns", "initialTimeLT", "initialTimeAttrs".
        """

        @staticmethod
        def CloudModelOne(inputDirectory, simulationLabel):
            simulationFolders = {"1": "Simulation_1", "2": "Simulation_2"}
            folderName = simulationFolders[str(simulationLabel)]
            dataDirectory = os.path.join(inputDirectory, "Model", folderName)

            dataTypeConfig = {
                "eulerianData": {
                    "dataDirectory": dataDirectory,
                    "filePattern": "cm1out_*.nc",
                    "excludePatterns": ("cm1out_pdata*.nc", "cm1out_stats*.nc"),
                    "coordNames": ("time", "zf", "zh", "yf", "yh", "xf", "xh"),
                    "initialTimeAttrs": {"hour": "hour", "minute": "minute"},
                    "domainSubsetBounds": {"xh": slice(20, 280), "zf": slice(0, 16)}, #... etc #in same units of the coords

                },
                "lagrangianData": {
                    "dataDirectory": dataDirectory,
                    "filePattern": "cm1out_pdata.nc",
                    "coordNames": ("time",),
                    "initialTimeAttrs": {"hour": "hour", "minute": "minute"},
                },
            }
            return [dataTypeConfig]

        # Add more resolvers here as needed, e.g.:
        # @staticmethod
        # def MPAS(inputDirectory, runLabel):
        #     ...

        @classmethod
        def GetRegistry(self):
            """Maps resolver name (string) -> resolver function."""
            registry = {
                "CloudModelOne": self.CloudModelOne,
                # "WRF": self.WRF,
            }
            return [registry]

        @classmethod
        def GetResolver(self, resolverName):
            """Look up a resolver function by name, with a helpful error if missing."""
            [registry] = self.GetRegistry()
            if resolverName not in registry:
                raise KeyError(
                    f"Unknown resolverName '{resolverName}'. Options: {list(registry)}")
            return [registry[resolverName]]

    class DataOpeners:
        """Namespace of stateless functions for opening netCDF file(s) into
        an xarray Dataset, given a file list and optional file-index selection."""
    
        @staticmethod
        def OpenData(fileList, concatDim="time", file=None, **kwargs):
            """
            Open one or more netCDF files as a single xarray Dataset.
            ***Caller is responsible for closing [.close()] and deleting (del) the returned Dataset for IO efficiency
            (i.e. opening many files in a loop).***
            """
            fileList = [fileList] if isinstance(fileList, str) else list(fileList)
    
            [selectedFiles] = InputData_Classes.DataOpeners.SelectFiles(fileList, file)
    
            if len(selectedFiles) == 1:
                [ds] = InputData_Classes.DataOpeners.OpenSingleFile(selectedFiles[0], **kwargs)
            else:
                [ds] = InputData_Classes.DataOpeners.OpenMultipleFiles(selectedFiles, concatDim, **kwargs)
    
            return [ds]
    
        @staticmethod
        def SelectFiles(fileList, file):
            """Helper: resolve the file selector into the actual subset of
            fileList to open."""
            if file is None:
                return [fileList]
            if isinstance(file, int):
                return [[fileList[file]]]
            indices = list(file)
            return [[fileList[i] for i in indices]]
    
        @staticmethod
        def OpenSingleFile(filePath, **kwargs):
            """Helper: open a single netCDF file as an xarray Dataset."""
            ds = xr.open_dataset(filePath, decode_timedelta=True, **kwargs)
            return [ds]
    
        @staticmethod
        def OpenMultipleFiles(fileList, concatDim, **kwargs):
            """Helper: open and combine multiple netCDF files into a single
            xarray Dataset, concatenated along concatDim."""
            ds = xr.open_mfdataset(
                fileList, concat_dim=concatDim, combine="nested",
                decode_timedelta=True, **kwargs)
            return [ds]
        
    class DataExporters:
        """Namespace of stateless functions for exporting an already-prepared
        xarray Dataset to disk (e.g. as HDF5), tracked via a single shared JSON
        manifest (Derived/manifest.json) covering every label, so files can be
        found again later by label alone."""
    
        @staticmethod
        def SaveH5(dataset, data, file, label, **kwargs):
            """
            Save data (an already-prepared xr.Dataset) as HDF5 into
            <dataset.dataDirectory>/Postprocessed/<label>/, and record it in the
            shared <dataset.dataDirectory>/Postprocessed/manifest.json (structured
            as {label: {file: relativePath}}), so all saved labels are tracked
            together in one manifest.
            """
            postprocessedRoot = os.path.join(dataset.dataDirectory, "Postprocessed")
            postprocessedDir = os.path.join(postprocessedRoot, label)
            os.makedirs(postprocessedDir, exist_ok=True)
            manifestPath = os.path.join(postprocessedRoot, "manifest.json")
        
            sourceFile = dataset.fileList[file]
            number = os.path.splitext(os.path.basename(sourceFile))[0].split("_")[-1]
            outputFile = f"cm1_{number}_{label}.h5"
            outputPath = os.path.join(postprocessedDir, outputFile)
        
            with h5py.File(outputPath, "w") as f:
                for name, da in data.data_vars.items():
                    [values] = InputData_Classes.DataExporters.ToH5Compatible(da.values)
                    f.create_dataset(name, data=values, **kwargs)
                for name, coord in data.coords.items():
                    [values] = InputData_Classes.DataExporters.ToH5Compatible(coord.values)
                    f.create_dataset(name, data=values, **kwargs)
        
            [manifest] = InputData_Classes.DataExporters.LoadManifest(manifestPath)
            manifest.setdefault(label, {})[str(file)] = os.path.join(label, outputFile)
            InputData_Classes.DataExporters.SaveManifest(manifestPath, manifest)
        
            return [outputPath]
        
        @staticmethod
        def ToH5Compatible(values):
            """
            Helper: h5py can't store numpy timedelta64/datetime64 dtypes directly.
            Convert timedelta64 arrays to float64 seconds, and datetime64 arrays
            to int64 nanoseconds-since-epoch, so they can be written as an HDF5
            dataset. Any other dtype is passed through unchanged.
            """
            values = np.asarray(values)
            if np.issubdtype(values.dtype, np.timedelta64):
                values = values / np.timedelta64(1, "s")
            elif np.issubdtype(values.dtype, np.datetime64):
                values = values.astype("datetime64[ns]").astype(np.int64)
            return [values]
        
        @staticmethod
        def LoadH5(dataset, label, file=None):
            """
            Read back HDF5 file(s) previously saved via SaveH5 under this label,
            using the shared Postprocessed/manifest.json to find them (no
            path/filename needed). file=None loads every saved file for this label
            (dict keyed by file index); an int loads just that one (dict of
            {varName: array}).
            """
            postprocessedRoot = os.path.join(dataset.dataDirectory, "Postprocessed")
            manifestPath = os.path.join(postprocessedRoot, "manifest.json")
            [manifest] = InputData_Classes.DataExporters.LoadManifest(manifestPath)
            labelEntries = manifest.get(label, {})
        
            if file is not None:
                outputPath = os.path.join(postprocessedRoot, labelEntries[str(file)])
                [data] = InputData_Classes.DataExporters.ReadH5File(outputPath)
                return [data]
        
            result = {}
            for key, relPath in sorted(labelEntries.items(), key=lambda kv: int(kv[0])):
                [result[int(key)]] = InputData_Classes.DataExporters.ReadH5File(
                    os.path.join(postprocessedRoot, relPath))
            return [result]
        
        @staticmethod
        def LoadManifest(manifestPath):
            """Helper: load manifest.json, or an empty dict if it doesn't exist yet."""
            if not os.path.exists(manifestPath):
                return [{}]
            with open(manifestPath, "r") as f:
                return [json.load(f)]
        
        @staticmethod
        def SaveManifest(manifestPath, manifest):
            """Helper: write manifest.json."""
            with open(manifestPath, "w") as f:
                json.dump(manifest, f, indent=2)
        
        @staticmethod
        def ReadH5File(path):
            """Helper: read one HDF5 file into a dict of {name: array}."""
            data = {}
            with h5py.File(path, "r") as f:
                for name in f:
                    data[name] = f[name][()]
            return [data]

# #--------------------------------------------------
# #Example Loading
# #--------------------------------------------------
# #Load Classes
# mainCodeDirectory = os.path.abspath("/mnt/lustre/koa/koastore/torri_group/air_directory/Projects/3.Moist-Preconditioning-Project/Coding_Directory/Code")
# classDirectory = os.path.join(mainCodeDirectory, "classes")
# sys.path.append(classDirectory)

# from OutputData_Classes import OutputData_Classes
# from InputData_Classes import InputData_Classes

# #Loading DatabaseManager
# DatabaseManager\
# =OutputData_Classes.DatabaseManager_Class(classDirectory=classDirectory,
#                                           inputDirectory="../../Input/", outputDirectory="../../Output/",
#                                           verbose=True)

# InputData = InputData_Classes.Model_InputData_Class(pathResolverKwargs={"inputDirectory": DatabaseManager.inputDirectory,
#                                                                         "simulationLabel": "1"})