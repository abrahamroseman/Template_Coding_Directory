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
# from OutputData_Classes import OutputData_Classes


# In[ ]:


# OutputData_Classes
# ============================================================

#Libraries
import os
import json
import h5py; import pickle
import numpy as np

#Class
class OutputData_Classes:
    
    # DatabaseManager_Class
    # ============================================================
    class DatabaseManager_Class:
        def __init__(self,
                     classDirectory=("/mnt/lustre/koa/koastore/torri_group/air_directory/"
                                     "Projects/Template_Coding_Directory/Code/classes"),
                     inputDirectory="../../Input/", outputDirectory="../../Output/",
                     verbose=True):
    
            self.classDirectory = classDirectory
            self.inputDirectory = self.resolve_path(inputDirectory)
            self.outputDirectory = self.resolve_path(outputDirectory)
            self.verbose = verbose
    
            # Log file that records scriptName/fileName -> outputSubDirectory
            self.outputLogFile = os.path.join(self.outputDirectory, "output_log.json")
    
            if self.verbose:
                self.Summary()
    
        # ============================================================
        # ========== Functions ==========
        # ============================================================
                
        def SaveOutput(self,
                       outputDictionary,
                       scriptName, dataName, outputSubDirectory, fileName,
                       subDataName=None,
                       dtype=None, makeSubDirectory=True,
                       fileType="h5"):
            """
            Save outputDictionary to:
                self.outputDirectory / outputSubDirectory / <actualFileName>.<ext>
            If subDataName is given, the actual saved fileName becomes
            f"{fileName}_{subDataName}". The log stores outputSubDirectory, the
            BASE fileName, and fileType once under dataName, and appends
            subDataName to a "subFileNames" list (since these are shared across
            all sub-entries).
            """
            saveFunction, _, extension = OutputData_Classes.Functions.GetSaveLoadFunctions(fileType)
        
            actualFileName = fileName if subDataName is None else f"{fileName}_{subDataName}"
        
            out_dir = os.path.join(self.outputDirectory, outputSubDirectory)
            out_file = os.path.join(out_dir, f"{actualFileName}{extension}")
        
            saveFunction(
                outputDictionary=outputDictionary,
                filepath=out_file,
                dtype=dtype,
                makeDirs=makeSubDirectory,
            )
        
            # Pass the BASE fileName (not actualFileName) and fileType to the log
            self.update_log(scriptName, dataName, outputSubDirectory, fileName, subDataName,
                             fileType=fileType)
        
        def LoadOutput(self, scriptName, dataName, subDataName=None, verbose=True):
            """
            Load data by looking up (scriptName, dataName) in the output log to get
            outputSubDirectory, the base fileName, and fileType. If subDataName is
            given, it's appended to the base fileName to form the actual file on disk.
            """
            log = self.load_log()
        
            if scriptName not in log:
                raise KeyError(f"No entries logged for scriptName '{scriptName}'. "
                                f"Available: {list(log.keys())}")
            if dataName not in log[scriptName]:
                raise KeyError(f"No entry logged for dataName '{dataName}' under scriptName '{scriptName}'. "
                                f"Available: {list(log[scriptName].keys())}")
        
            entry = log[scriptName][dataName]
            outputSubDirectory = entry["outputSubDirectory"]
            baseFileName = entry["fileName"]
            fileType = entry["fileType"]
        
            if subDataName is not None:
                availableSubNames = entry.get("subFileNames", [])
                if subDataName not in availableSubNames:
                    raise KeyError(f"No subDataName '{subDataName}' logged under "
                                    f"'{scriptName}/{dataName}'. Available: {availableSubNames}")
                fileName = f"{baseFileName}_{subDataName}"
            else:
                fileName = baseFileName
        
            _, loadFunction, extension = OutputData_Classes.Functions.GetSaveLoadFunctions(fileType)
        
            in_dir = os.path.join(self.outputDirectory, outputSubDirectory)
            in_file = os.path.join(in_dir, f"{fileName}{extension}")
        
            outputDictionary = loadFunction(filepath=in_file, verbose=verbose)
        
            return outputDictionary
                    
        def Load_Or_RunAndSave(self,
                       function,
                       scriptName, dataName, outputSubDirectory, fileName,
                       subDataName=None,
                       dtype=None, makeSubDirectory=True,
                       args=None, kwargs=None,
                       forceRecalculate=False,
                       fileType="h5",
                       verbose=True):
            """
            Delegates to Functions.Load_Or_RunAndSave, but injects self.LoadOutput
            and self.SaveOutput as the loadFunction/saveFunction. This keeps the
            fast file-existence check in Functions.Load_Or_RunAndSave, while the
            actual load/save still goes through the log-aware DatabaseManager
            methods -- so update_log happens automatically inside SaveOutput,
            with no separate log-update step needed here.
        
            fileType : "h5" (default) or "pickle" -- forwarded to both the
                       file-existence check and the actual save/load calls.
            """
            actualFileName = fileName if subDataName is None else f"{fileName}_{subDataName}"
            fileDirectory = os.path.join(self.outputDirectory, outputSubDirectory)
        
            def loadFunction(filepath, verbose=True):
                # filepath is ignored -- self.LoadOutput resolves the path itself
                # via the log, using the closed-over scriptName/dataName/subDataName.
                return self.LoadOutput(
                    scriptName=scriptName, dataName=dataName,
                    subDataName=subDataName, verbose=verbose
                )
        
            def saveFunction(data, filepath, dtype=None):
                # filepath is ignored -- self.SaveOutput builds the path itself and
                # ALSO updates the log as a side effect.
                self.SaveOutput(
                    outputDictionary=data,
                    scriptName=scriptName, dataName=dataName,
                    outputSubDirectory=outputSubDirectory, fileName=fileName,
                    subDataName=subDataName,
                    dtype=dtype, makeSubDirectory=makeSubDirectory,
                    fileType=fileType,
                )
        
            return OutputData_Classes.Functions.Load_Or_RunAndSave(
                calculateFunction=function,
                fileDirectory=fileDirectory,
                fileName=actualFileName,
                fileType=fileType,
                args=args, kwargs=kwargs,
                loadFunction=loadFunction, saveFunction=saveFunction,
                dtype=dtype,
                forceRecalculate=forceRecalculate,
                verbose=verbose,
            )
    
        # ============================================================
        # ========== Helper Functions ==========
        # ============================================================
    
        def resolve_path(self, path):
            """Join a relative path onto classDirectory; leave absolute paths as-is."""
            if os.path.isabs(path):
                return path
            return os.path.normpath(os.path.join(self.classDirectory, path))
    
        def load_log(self):
            """Read the output log from disk (empty dict if it doesn't exist yet)."""
            if os.path.exists(self.outputLogFile):
                with open(self.outputLogFile, 'r') as f:
                    return json.load(f)
            return {}
        
        def update_log(self, scriptName, dataName, outputSubDirectory, fileName,
                       subDataName=None, fileType="h5"):
            """
            Record outputSubDirectory + base fileName + fileType once under dataName,
            and append subDataName (if given) to a shared "subFileNames" list.
            """
            log = self.load_log()
            if scriptName not in log:
                log[scriptName] = {}
        
            if dataName not in log[scriptName]:
                log[scriptName][dataName] = {
                    "outputSubDirectory": outputSubDirectory,
                    "fileName": fileName,
                    "fileType": fileType,
                    "subFileNames": []
                }
            else:
                # keep outputSubDirectory/fileName/fileType current in case they've changed
                log[scriptName][dataName]["outputSubDirectory"] = outputSubDirectory
                log[scriptName][dataName]["fileName"] = fileName
                log[scriptName][dataName]["fileType"] = fileType
                log[scriptName][dataName].setdefault("subFileNames", [])
        
            if subDataName is not None and subDataName not in log[scriptName][dataName]["subFileNames"]:
                log[scriptName][dataName]["subFileNames"].append(subDataName)
        
            os.makedirs(os.path.dirname(self.outputLogFile), exist_ok=True)
            with open(self.outputLogFile, 'w') as f:
                json.dump(log, f, indent=2)
                
        
        def ShowLog(self):
            """Print everything recorded in the output log."""
            log = self.load_log()
            print("=== Output Log ===")
            for scriptName, dataEntries in log.items():
                print(f" {scriptName}:")
                for dataName, entry in dataEntries.items():
                    subNames = entry.get("subFileNames", [])
                    if subNames:
                        print(f"   {dataName} -> {entry['outputSubDirectory']}/{entry['fileName']}_[subDataName].h5")
                        print(f"      subFileNames: {subNames}")
                    else:
                        print(f"   {dataName} -> {entry['outputSubDirectory']}/{entry['fileName']}.h5")
            print("==================", "\n")
    
        # ============================================================
        # ========== Test Functions ==========
        # ============================================================
    
        def Summary(self):
            """Print a summary of the directory configuration."""
            print("=== DataManager Summary ===")
            print(f" inputDirectory  #: {self.inputDirectory}")
            print(f" outputDirectory #: {self.outputDirectory}")
            print(f" outputLogFile         #: {self.outputLogFile}")
            print("=========================", "\n")
        
        
        def Test(self,t=1):
            """
            Self-test: save a small outputDictionary (a=[0,1,2,3], b=[1,2,3,4])
            with SaveOutput under subDataName="time1" (which logs it nested under
            dataName), then read it back with LoadOutput and check the round-trip.
            """
            print("=== Running DataManager Test ===\n")
        
            outputDictionary = {
                "a": [0, 1, 2, 3],
                "b": [1, 2, 3, 4],
            }
        
            # Save (this also writes the log entry, nested under dataName/subDataName)
            self.SaveOutput(
                outputDictionary=outputDictionary,
                scriptName="Algorithm_1", dataName="testData", subDataName=f"time_{t}",
                outputSubDirectory="data/test_directory",
                fileName="test_data_name",
                dtype="int16",
            )
        
            # Check input/output data
            print("Variables to save:")
            for var_name, arr in outputDictionary.items():
                print(f"  {var_name}: {list(arr)}")
        
            # Load back using scriptName + dataName + subDataName (via the log)
            loadedDictionary = self.LoadOutput(
                scriptName="Algorithm_1", dataName="testData", subDataName=f"time_{t}"
            )
        
            # Check input/output data
            print("Loaded variables:")
            for var_name, arr in loadedDictionary.items():
                print(f"  {var_name}: {list(arr)}")
    
        def Test2(self):
            """
            Self-test: use Load_Or_RunAndSave (with a no-parameter) function that
            returns a small outputDictionary (a=[0,1,2,3], b=[1,2,3,4]), saved under
            subDataName="time1" (logged nested under dataName), then read back.
            """
            print("=== Running DataManager Test2 ===\n")
        
            def function():
                return {
                    "a": [0, 1, 2, 3],
                    "b": [1, 2, 3, 4],
                }
        
            # Runs function() if not cached, saves + logs it, and returns the result
            outputDictionary = self.Load_Or_RunAndSave(
                function=function, args=None, kwargs=None,
                scriptName="Algorithm_1", dataName="testData", subDataName="time1",
                outputSubDirectory="data/test_directory",
                fileName="test_data_name",
                dtype="int16",
            )
        
            # Check input/output data
            print("Loaded or Saved variables:")
            for var_name, arr in outputDictionary.items():
                print(f"  {var_name}: {list(arr)}")

    # Functions
    # ============================================================
    class Functions:
        
        @staticmethod
        def SaveOutput_H5(outputDictionary, filepath, dtype=None, makeDirs=True):
            """
            Generic HDF5 saving function. Saves outputDictionary (a dict of
            {var_name: array}) to filepath. If outputDictionary is not a dict,
            it is wrapped as {"data": outputDictionary} first.
        
            dtype : None -> let h5py infer dtype for each variable
                    single dtype -> broadcast to all variables
                    list of dtypes -> must match len(outputDictionary), one per variable
            """
            if not isinstance(outputDictionary, dict):
                outputDictionary = {"data": outputDictionary}
        
            if makeDirs:
                dirName = os.path.dirname(filepath)
                if dirName:
                    os.makedirs(dirName, exist_ok=True)
        
            if dtype is None:
                dtype_list = [None] * len(outputDictionary)
            elif isinstance(dtype, list):
                if len(dtype) != len(outputDictionary):
                    raise ValueError("Length of dtype list must match number of variables in outputDictionary.")
                dtype_list = dtype
            else:
                dtype_list = [dtype] * len(outputDictionary)
        
            with h5py.File(filepath, 'w') as f:
                for (var_name, arr), dt in zip(outputDictionary.items(), dtype_list):
                    f.create_dataset(var_name, data=arr, dtype=dt, compression="gzip")
        
            print(f"Saved output file: {filepath}\n")
        
        @staticmethod
        def LoadOutput_H5(filepath, dtype=None, verbose=True):
            """
            Generic HDF5 loading function. Returns a dict of {var_name: array}
            read from filepath.
            """
            outputDictionary = {}
            with h5py.File(filepath, 'r') as f:
                for var_name in f.keys():
                    outputDictionary[var_name] = f[var_name][:]
        
            if verbose:
                print(f"Loaded output file: {filepath}\n")
        
            return outputDictionary
        
        @staticmethod
        def SaveOutput_Pickle(outputDictionary, filepath, dtype=None, makeDirs=True):
            """
            Generic pickle saving function. Saves outputDictionary (a dict of
            {var_name: value}) to filepath. If outputDictionary is not a dict,
            it is wrapped as {"data": outputDictionary} first.
            """
            if not isinstance(outputDictionary, dict):
                outputDictionary = {"data": outputDictionary}
        
            if makeDirs:
                dirName = os.path.dirname(filepath)
                if dirName:
                    os.makedirs(dirName, exist_ok=True)
        
            with open(filepath, 'wb') as file:
                pickle.dump(outputDictionary, file)
        
            print(f"Saved output file: {filepath}\n")
        
        @staticmethod
        def LoadOutput_Pickle(filepath, verbose=True):
            """
            Generic pickle loading function. Returns the dict stored at filepath.
            """
            with open(filepath, 'rb') as file:
                outputDictionary = pickle.load(file)
        
            if verbose:
                print(f"Loaded output file: {filepath}\n")
        
            return outputDictionary

        @staticmethod
        def GetSaveLoadFunctions(fileType):
            """
            Given a fileType ("h5" or "pickle"), return the matching
            (saveFunction, loadFunction, extension) trio.
            """
            if fileType == "h5":
                return (OutputData_Classes.Functions.SaveOutput_H5,
                        OutputData_Classes.Functions.LoadOutput_H5,
                        ".h5")
            elif fileType == "pickle":
                return (OutputData_Classes.Functions.SaveOutput_Pickle,
                        OutputData_Classes.Functions.LoadOutput_Pickle,
                        ".pkl")
            else:
                raise ValueError(f"Unknown fileType '{fileType}'.")
        
        @staticmethod
        def Load_Or_RunAndSave(calculateFunction, fileDirectory=".", fileName="filename", fileType="h5",
                               args=None, kwargs=None, 
                               loadFunction=None, saveFunction=None,
                               dtype=None, 
                               calculatedData=None, forceRecalculate=False, 
                               verbose=True):
            """
            Loads data from fileDirectory/fileName.<ext> if it exists.
            Otherwise, runs the provided function and saves the output.
            """
            defaultSaveFunction, defaultLoadFunction, extension = \
                OutputData_Classes.Functions.GetSaveLoadFunctions(fileType)
        
            if loadFunction is None:
                loadFunction = defaultLoadFunction
            if saveFunction is None:
                saveFunction = defaultSaveFunction
        
            filepath = os.path.join(fileDirectory, f"{fileName}{extension}")
        
            # Check if the file already exists
            if os.path.exists(filepath) and not forceRecalculate:
                data = loadFunction(filepath, verbose=verbose)
                return data
            else:
                if verbose:
                    print(f"Data from {filepath} not found. Running calculation...")
                # Run the target function
                data = calculateFunction(*(args or ()), **(kwargs or {})) if calculatedData is None else calculatedData
        
                # Save the result for future use
                saveFunction(data, filepath, dtype=dtype)
        
                return data

    # OtherFunctions
    # ============================================================
    class Other_Functions:

        @staticmethod
        def OpenMultipleFiles(fileList, concatDim="time",
                              chunkSpec=None,
                              renameDims=None,
                              sliceSpec=None,
                              engine=None):
            """
            Open and concatenate a list of .h5/.nc files into a single xarray.Dataset,
            stacked along a new dimension `concatDim` (e.g. "time"), with each file
            contributing one slice along that dimension.
        
            fileList  : list of files getting concatenated
            concatDim : which dimension is getting concatenated over multiple files
            chunkSpec : dask chunk sizes per dimension, e.g. {"time": 1, "phony_dim_0": -1}
            renameDims: any dimensions to rename (e.g. renameDims={"phony_dim_0": "p"})
            sliceSpec : optional dict of {dimName: sliceObject} pairs, applied to EACH
                        file individually before concatenation, e.g.
                            sliceSpec = {"phony_dim_0": slice(0, 10), "phony_dim_1": slice(5, 15)}
            engine    : which engine xr.open_mfdataset uses (h5netcdf for .h5, None for .nc)
            """
            if not fileList:
                raise FileNotFoundError("No files provided to OpenMultipleFiles.")
        
            if engine is None:
                ext = os.path.splitext(fileList[0])[1].lower()
                engine = "h5netcdf" if ext in (".h5", ".hdf5") else None
        
            needsPreprocess = sliceSpec is not None
        
            def preprocess(ds):
                if sliceSpec is not None:
                    applicableSlices = {dim: sl for dim, sl in sliceSpec.items() if dim in ds.dims}
                    if applicableSlices:
                        ds = ds.isel(applicableSlices)
                return ds
        
            open_kwargs = dict(
                combine="nested", concat_dim=concatDim,
                compat="override", coords="minimal",
                join="override", parallel=True,
            )
            if needsPreprocess:
                open_kwargs["preprocess"] = preprocess
            if chunkSpec is not None:
                open_kwargs["chunks"] = chunkSpec
            if engine == "h5netcdf":
                open_kwargs["engine"] = "h5netcdf"
                open_kwargs["phony_dims"] = "sort"
            elif engine is not None:
                open_kwargs["engine"] = engine
        
            ds = xr.open_mfdataset(fileList, **open_kwargs)
        
            if renameDims:
                applicableRenames = {k: v for k, v in renameDims.items() if k in ds.dims}
                if applicableRenames:
                    ds = ds.rename(applicableRenames)
        
            return ds
                            

    # SlurmJobArray_Class
    # ============================================================    
    class SlurmJobArray_Class:
        """
        For splitting any data dimension 
        into a set number of jobs to run with "job array" in HPC systems (i.e. slurm, PBS, etc)
        """
        def __init__(self, total_elements, 
                     num_jobs, UsingJobArray,
                     custom_job_id=None,
                     verbose=True):
            self.total_elements = total_elements
            self.num_jobs = num_jobs
            self.UsingJobArray = UsingJobArray
            
            # Get job ID (default = 1 if not running under Slurm)
            if custom_job_id is None:
                self.job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
                if self.job_id == 0:
                    self.job_id = 1
            elif custom_job_id is not None:
                self.job_id = custom_job_id
            
            # Precompute range info
            self.job_range = total_elements // num_jobs
            self.remaining = total_elements % num_jobs
            
            # Compute job range for this job
            self.start_job, self.end_job = self._get_job_range(self.job_id)
    
            # Print summary
            if verbose:
                self.Summary()
    
        # ------------------------------------------------------------
        def _get_job_range(self, job_id):
            if self.UsingJobArray == True:
                """Compute start and end indices for this job."""
                job_id -= 1
                start_job = job_id * self.job_range + min(job_id, self.remaining)
                end_job = start_job + self.job_range + (1 if job_id < self.remaining else 0)
                if job_id == self.num_jobs - 1:
                    end_job = self.total_elements
            elif self.UsingJobArray == False:
                start_job, end_job = 0, self.total_elements
            return start_job, end_job
    
        # ------------------------------------------------------------
        def Test(self):
            """Print start/end for all jobs to verify chunking logic."""
            start, end = [], []
            for job_id in range(1, self.num_jobs + 1):
                s, e = self._get_job_range(job_id)
                print(f"Job {job_id}: {s} → {e}")
                start.append(s)
                end.append(e)
            print("Unique starts:", len(np.unique(start)) == len(start))
            print("Unique ends:", len(np.unique(end)) == len(end))
            print("No zero-length ranges:", np.all(np.array(start) != np.array(end)))
    
        def Summary(self):
            print(f"Running timesteps from {self.start_job}:{self.end_job-1}","\n")

