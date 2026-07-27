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
            [self.inputDirectory] = self.resolve_path(inputDirectory)
            [self.outputDirectory] = self.resolve_path(outputDirectory)
            self.verbose = verbose
    
            # Log file that records scriptName/fileName -> subDataName
            self.outputLogFile = os.path.join(self.outputDirectory, "output_log.json")
    
            if self.verbose:
                self.Summary()
    
        # ============================================================
        # ========== Functions ==========
        # ============================================================
                
        def SaveOutput(self,
                       outputDictionary,
                       folderName, scriptName, dataName,
                       *,
                       subFolderName=None, subDataName=None,
                       fileName=None, subFileName=None,
                       dtype=None, makeDirectory=True,
                       fileType="h5"):
            """
            Save outputDictionary to:
                self.outputDirectory / folderName [/ subFolderName] / scriptName / dataName [/ subDataName] / <fileName>[_subFileName].<ext>
            """
            if fileName is None:
                fileName = dataName
        
            [saveFunction, _, extension] = OutputData_Classes.Functions.GetSaveLoadFunctions(fileType)
        
            actualFileName = fileName if subFileName is None else f"{fileName}_{subFileName}"
        
            pathParts = [self.outputDirectory, folderName]
            if subFolderName is not None:
                pathParts.append(subFolderName)
            pathParts += [scriptName, dataName]
            if subDataName is not None:
                pathParts.append(subDataName)
            out_dir = os.path.join(*pathParts)
            out_file = os.path.join(out_dir, f"{actualFileName}{extension}")
        
            saveFunction(
                outputDictionary=outputDictionary,
                filepath=out_file,
                dtype=dtype,
                makeDirs=makeDirectory,
                attrs={
                    "folderName": folderName,
                    "subFolderName": subFolderName,
                    "scriptName": scriptName,
                    "dataName": dataName,
                    "subDataName": subDataName,
                    "fileName": fileName,
                    "subFileName": subFileName,
                },
            )
        
            self.update_log(folderName, scriptName, dataName, subFolderName, subDataName,
                             fileName, subFileName, fileType=fileType)
                
        def LoadOutput(self, folderName, scriptName, dataName,
                       subFileName=None, verbose=True):
            log = self.load_log()
        
            if folderName not in log:
                raise KeyError(f"No entries logged for folderName '{folderName}'. "
                                f"Available: {list(log.keys())}")
            if scriptName not in log[folderName]:
                raise KeyError(f"No entries logged for scriptName '{scriptName}' under folderName '{folderName}'. "
                                f"Available: {list(log[folderName].keys())}")
            if dataName not in log[folderName][scriptName]:
                raise KeyError(f"No entry logged for dataName '{dataName}' under "
                                f"'{folderName}/{scriptName}'. Available: {list(log[folderName][scriptName].keys())}")
        
            entry = log[folderName][scriptName][dataName]
            subFolderName = entry.get("subFolderName")
            subDataName = entry.get("subDataName")
            baseFileName = entry["fileName"]
            fileType = entry["fileType"]
        
            if subFileName is not None:
                availableSubNames = entry.get("subFileNames", [])
                if subFileName not in availableSubNames:
                    raise KeyError(f"No subFileName '{subFileName}' logged under "
                                    f"'{folderName}/{scriptName}/{dataName}'. Available: {availableSubNames}")
                fileName = f"{baseFileName}_{subFileName}"
            else:
                fileName = baseFileName
        
            [_, loadFunction, extension] = OutputData_Classes.Functions.GetSaveLoadFunctions(fileType)
        
            pathParts = [self.outputDirectory, folderName]
            if subFolderName is not None:
                pathParts.append(subFolderName)
            pathParts += [scriptName, dataName]
            if subDataName is not None:
                pathParts.append(subDataName)
            in_dir = os.path.join(*pathParts)
            in_file = os.path.join(in_dir, f"{fileName}{extension}")
        
            [outputDictionary] = loadFunction(filepath=in_file, verbose=verbose)
        
            return [outputDictionary]
                            
        def Load_Or_RunAndSave(self,
                               function,
                               folderName, scriptName, dataName,
                               *,
                               subFolderName=None, subDataName=None,
                               fileName=None, subFileName=None,
                               dtype=None, makeDirectory=True,
                               args=None, kwargs=None,
                               forceRecalculate=False,
                               fileType="h5",
                               verbose=True):
            if fileName is None:
                fileName = dataName
        
            actualFileName = fileName if subFileName is None else f"{fileName}_{subFileName}"
        
            pathParts = [self.outputDirectory, folderName]
            if subFolderName is not None:
                pathParts.append(subFolderName)
            pathParts += [scriptName, dataName]
            if subDataName is not None:
                pathParts.append(subDataName)
            fileDirectory = os.path.join(*pathParts)
        
            def loadFunction(filepath, verbose=True):
                return self.LoadOutput(
                    folderName=folderName, scriptName=scriptName, dataName=dataName,
                    subFileName=subFileName, verbose=verbose)
        
            def saveFunction(data, filepath, dtype=None):
                self.SaveOutput(
                    outputDictionary=data,
                    folderName=folderName, scriptName=scriptName, dataName=dataName,
                    subFolderName=subFolderName, subDataName=subDataName,
                    fileName=fileName, subFileName=subFileName,
                    dtype=dtype, makeDirectory=makeDirectory,
                    fileType=fileType,
                )
        
            [result] = OutputData_Classes.Functions.Load_Or_RunAndSave(
                calculateFunction=function,
                fileDirectory=fileDirectory,
                fileName=actualFileName,
                fileType=fileType,
                args=args, kwargs=kwargs,
                loadFunction=loadFunction, saveFunction=saveFunction,
                dtype=dtype,
                forceRecalculate=forceRecalculate,
                verbose=verbose)
        
            return result
            
        # ============================================================
        # ========== Helper Functions ==========
        # ============================================================
    
        def resolve_path(self, path):
            """Join a relative path onto classDirectory; leave absolute paths as-is."""
            if os.path.isabs(path):
                return [path]
            return [os.path.normpath(os.path.join(self.classDirectory, path))]
    
        def load_log(self):
            """Read the output log from disk (empty dict if it doesn't exist yet)."""
            if os.path.exists(self.outputLogFile):
                with open(self.outputLogFile, 'r') as f:
                    return json.load(f)
            return {}
                
        def update_log(self, folderName, scriptName, dataName, subFolderName, subDataName,
                       fileName, subFileName=None, fileType="h5"):
            log = self.load_log()
            log.setdefault(folderName, {})
            log[folderName].setdefault(scriptName, {})
        
            if dataName not in log[folderName][scriptName]:
                log[folderName][scriptName][dataName] = {
                    "subFolderName": subFolderName,
                    "subDataName": subDataName,
                    "fileName": fileName,
                    "fileType": fileType,
                    "subFileNames": []
                }
            else:
                log[folderName][scriptName][dataName]["subFolderName"] = subFolderName
                log[folderName][scriptName][dataName]["subDataName"] = subDataName
                log[folderName][scriptName][dataName]["fileName"] = fileName
                log[folderName][scriptName][dataName]["fileType"] = fileType
                log[folderName][scriptName][dataName].setdefault("subFileNames", [])
        
            if subFileName is not None and subFileName not in log[folderName][scriptName][dataName]["subFileNames"]:
                log[folderName][scriptName][dataName]["subFileNames"].append(subFileName)
        
            os.makedirs(os.path.dirname(self.outputLogFile), exist_ok=True)
            with open(self.outputLogFile, 'w') as f:
                json.dump(log, f, indent=2)
                        
        def ShowLog(self):
            log = self.load_log()
            print("=== Output Log ===")
            for folderName, scriptEntries in log.items():
                print(f"{folderName}:")
                for scriptName, dataEntries in scriptEntries.items():
                    print(f" {scriptName}:")
                    for dataName, entry in dataEntries.items():
                        subFolderPart = f"{entry.get('subFolderName')}/" if entry.get("subFolderName") else ""
                        subNames = entry.get("subFileNames", [])
                        if subNames:
                            print(f"   {dataName} -> {subFolderPart}{entry['subDataName']}/{entry['fileName']}_[subFileName].h5")
                            print(f"      subFileNames: {subNames}")
                        else:
                            print(f"   {dataName} -> {subFolderPart}{entry['subDataName']}/{entry['fileName']}.h5")
            print("==================", "\n")

        def RebuildLog(self, verbose=True):
            rebuiltLog = {}
            extensionToAttrReader = {
                ".h5": OutputData_Classes.Functions.GetAttrs_H5,
                ".pkl": OutputData_Classes.Functions.GetAttrs_Pickle,
            }
            extensionToType = {".h5": "h5", ".pkl": "pickle"}
        
            if not os.path.isdir(self.outputDirectory):
                if verbose:
                    print(f"No outputDirectory found at {self.outputDirectory}, nothing to rebuild.\n")
                return [rebuiltLog]
        
            for root, _, files in os.walk(self.outputDirectory):
                for f in files:
                    ext = os.path.splitext(f)[1]
                    if ext not in extensionToAttrReader:
                        continue
        
                    filepath = os.path.join(root, f)
                    try:
                        [attrs] = extensionToAttrReader[ext](filepath)
                    except Exception as e:
                        if verbose:
                            print(f"Warning: couldn't read attrs from {filepath} ({e}) -- skipping.\n")
                        continue
        
                    requiredKeys = ("folderName", "scriptName", "dataName", "fileName")
                    if not all(k in attrs and attrs[k] for k in requiredKeys):
                        if verbose:
                            print(f"Warning: {filepath} missing folderName/scriptName/dataName/fileName attrs "
                                  f"(saved before folderName was added?) -- skipping.\n")
                        continue
                            
                    folderName = attrs["folderName"]
                    subFolderName = attrs.get("subFolderName")
                    scriptName = attrs["scriptName"]
                    dataName = attrs["dataName"]
                    subDataName = attrs.get("subDataName")
                    fileName = attrs["fileName"]
                    subFileName = attrs.get("subFileName")
                            
                    rebuiltLog.setdefault(folderName, {})
                    rebuiltLog[folderName].setdefault(scriptName, {})
                    if dataName not in rebuiltLog[folderName][scriptName]:
                        rebuiltLog[folderName][scriptName][dataName] = {
                            "subFolderName": subFolderName,
                            "subDataName": subDataName,
                            "fileName": fileName,
                            "fileType": extensionToType[ext],
                            "subFileNames": [],
                        }
        
                    entry = rebuiltLog[folderName][scriptName][dataName]
                    if subFileName is not None and subFileName not in entry["subFileNames"]:
                        entry["subFileNames"].append(subFileName)
        
            os.makedirs(os.path.dirname(self.outputLogFile), exist_ok=True)
            with open(self.outputLogFile, 'w') as f:
                json.dump(rebuiltLog, f, indent=2)
        
            if verbose:
                print(f"Rebuilt log written to: {self.outputLogFile}\n")
        
            return [rebuiltLog]
    
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
            with SaveOutput under subFileName="time1" (which logs it nested under
            dataName), then read it back with LoadOutput and check the round-trip.
            """
            print("=== Running DataManager Test ===\n")
        
            outputDictionary = {
                "a": [0, 1, 2, 3],
                "b": [1, 2, 3, 4],
            }
        
            # Save (this also writes the log entry, nested under dataName/subFileName)
            self.SaveOutput(
                outputDictionary=outputDictionary,
                folderName="1_Folder_One", subFolderName= "Demo", scriptName="DemoScript", 
                dataName="Testing_Data_Output_And_Loading",subDataName="testSubDataName",
                fileName="testFileName",subFileName=f"time_{t}",
                dtype="int16",
            )
                        
            # Check input/output data
            print("Variables to save:")
            for var_name, arr in outputDictionary.items():
                print(f"  {var_name}: {list(arr)}")
        
            # Load back using scriptName + dataName + subFileName (via the log)
            [loadedDictionary] = self.LoadOutput(
                folderName="1_Folder_One", scriptName="DemoScript", dataName="Testing_Data_Output_And_Loading",
                subFileName=f"time_{t}")
        
            # Check input/output data
            print("Loaded variables:")
            for var_name, arr in loadedDictionary.items():
                print(f"  {var_name}: {list(arr)}")

    # Functions
    # ============================================================
    class Functions:
            
        @staticmethod
        def SaveOutput_H5(outputDictionary, filepath, dtype=None, makeDirs=True, attrs=None):
            """
            Generic HDF5 saving function. Saves outputDictionary (a dict of
            {var_name: array}) to filepath. If outputDictionary is not a dict,
            it is wrapped as {"data": outputDictionary} first.
        
            attrs : optional dict of metadata written as top-level HDF5 attributes
                    (e.g. {"scriptName": ..., "dataName": ..., "fileName": ..., "subFileName": ...}).
                    Lets RebuildLog recover exact values later without parsing filenames.
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
        
                if attrs:
                    for key, value in attrs.items():
                        # h5py attrs can't store None -- use empty string as the "absent" marker
                        f.attrs[key] = value if value is not None else ""
        
            print(f"Saved output file: {filepath}\n")
            
        @staticmethod
        def LoadOutput_H5(filepath, dtype=None, verbose=True,
                          loadToMemory=True):
            """
            Generic HDF5 loading function. 
            If loadToMemory==True:
            Returns a dict of {varName: array} read from filepath.
            Else:
                Returns the dataFile object itself (*user must remember to close*)
            """
            if not loadToMemory:
                dataFile = h5py.File(filepath, 'r')
                
                if verbose:
                    print(f"Loaded data as file object: {filepath}\n")
                return [dataFile]
                
            else:
                outputDictionary = {}
                with h5py.File(filepath, 'r') as dataFile:
                    for varName in dataFile.keys():
                        outputDictionary[varName] = dataFile[varName][:]
                        
                if verbose:
                    print(f"Loaded data into dictionary: {filepath}\n")
                return [outputDictionary]

        @staticmethod
        def GetAttrs_H5(filepath):
            """
            Reads only the top-level attributes from an HDF5 file, without loading
            any datasets. Cheap even for large files -- used by RebuildLog to recover
            scriptName/dataName/fileName/subFileName without guessing from filenames.
            """
            with h5py.File(filepath, 'r') as f:
                attrDictionary = dict(f.attrs)
        
            # Convert the "" placeholder back to None for subFileName/subDataName
            for key in ("subFileName", "subDataName"):
                if key in attrDictionary and attrDictionary[key] == "":
                    attrDictionary[key] = None
        
            return [attrDictionary]
                
        @staticmethod
        def SaveOutput_Pickle(outputDictionary, filepath, dtype=None, makeDirs=True, attrs=None):
            """
            attrs, if given, is stored under a reserved "_attrs" key alongside the data.
            """
            if not isinstance(outputDictionary, dict):
                outputDictionary = {"data": outputDictionary}
        
            if makeDirs:
                dirName = os.path.dirname(filepath)
                if dirName:
                    os.makedirs(dirName, exist_ok=True)
        
            dataToSave = dict(outputDictionary)
            if attrs:
                dataToSave["_attrs"] = attrs
        
            with open(filepath, 'wb') as file:
                pickle.dump(dataToSave, file)
        
            print(f"Saved output file: {filepath}\n")
                
        @staticmethod
        def LoadOutput_Pickle(filepath, verbose=True):
            with open(filepath, 'rb') as file:
                outputDictionary = pickle.load(file)
        
            outputDictionary.pop("_attrs", None)   # keep normal loads clean
        
            if verbose:
                print(f"Loaded output file: {filepath}\n")
        
            return [outputDictionary]

        @staticmethod
        def GetAttrs_Pickle(filepath):
            with open(filepath, 'rb') as file:
                outputDictionary = pickle.load(file)
            return [outputDictionary.get("_attrs", {})]

        @staticmethod
        def GetSaveLoadFunctions(fileType):
            """
            Given a fileType ("h5" or "pickle"), return the matching
            (saveFunction, loadFunction, extension) trio.
            """
            if fileType == "h5":
                return [OutputData_Classes.Functions.SaveOutput_H5,
                        OutputData_Classes.Functions.LoadOutput_H5,
                        ".h5"]
            elif fileType == "pickle":
                return [OutputData_Classes.Functions.SaveOutput_Pickle,
                        OutputData_Classes.Functions.LoadOutput_Pickle,
                        ".pkl"]
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
            [defaultSaveFunction, defaultLoadFunction, extension] = \
                OutputData_Classes.Functions.GetSaveLoadFunctions(fileType)
        
            if loadFunction is None:
                loadFunction = defaultLoadFunction
            if saveFunction is None:
                saveFunction = defaultSaveFunction
        
            filepath = os.path.join(fileDirectory, f"{fileName}{extension}")
        
            if os.path.exists(filepath) and not forceRecalculate:
                [data] = loadFunction(filepath, verbose=verbose)   # <- unpack here
            else:
                if verbose:
                    print(f"Data from {filepath} not found. Running calculation...")
                data = calculateFunction(*(args or ()), **(kwargs or {})) if calculatedData is None else calculatedData
                saveFunction(data, filepath, dtype=dtype)
        
            return [data]

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
            return [start_job, end_job]
    
        # ------------------------------------------------------------
        def Test(self):
            """Print start/end for all jobs to verify chunking logic."""
            start, end = [], []
            for job_id in range(1, self.num_jobs + 1):
                [s, e] = self._get_job_range(job_id)
                print(f"Job {job_id}: {s} → {e}")
                start.append(s)
                end.append(e)
            print("Unique starts:", len(np.unique(start)) == len(start))
            print("Unique ends:", len(np.unique(end)) == len(end))
            print("No zero-length ranges:", np.all(np.array(start) != np.array(end)))
    
        def Summary(self):
            print(f"Running timesteps from {self.start_job}:{self.end_job-1}","\n")

