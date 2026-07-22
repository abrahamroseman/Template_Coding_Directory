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
# from VectorizedOperations_Classes import VectorizedOperations_Classes


# In[163]:


# NumericalData_Classes
# ============================================================

# Libraries
import os, sys
import numpy as np

#Class
class VectorizedOperations_Classes:
    class MaskSearch:
        @staticmethod
        def FindTrue(mask, axis=0):
            """
            Find first and last True along one axis within a boolean mask.
            """
            firstTrue = np.argmax(mask,axis=axis)
            lastTrue = (mask.shape[axis] - 1) - np.argmax(np.flip(mask, axis=axis), axis=axis)
            hasTrue = mask.any(axis=axis)    
            return [firstTrue,lastTrue,hasTrue]
        @staticmethod
        def ApplyFindTrue(array,whichTrue,hasTrue,fill=np.nan):
            """
            Applies the results from FindTrue() to an array of the same shape as mask
            """
            idx = np.expand_dims(whichTrue, axis=axis)          
            gathered = np.take_along_axis(array, idx, axis=axis)
            gathered = np.squeeze(gathered, axis=axis)          
            findTrueValues = np.where(hasTrue, gathered, fill)
            return findTrueValues
        @staticmethod
        def SubsetIndices(array, innerIndex, outerIndex, axis=0):
            """
            Mask `array` along `axis`, keeping only positions where
            innerIndex <= idx <= outerIndex (broadcast over the other dims).
            """
            n = array.shape[axis]
            
            # build an index array of shape (1,...,n,...,1) with n along `axis`
            idx_shape = [1] * array.ndim
            idx_shape[axis] = n
            idx = np.arange(n).reshape(idx_shape)
            
            mask = (idx >= innerIndex) & (idx <= outerIndex)
            
            output = np.full_like(array, fill_value=np.nan, dtype=float)
            output[mask] = array[mask]
            return output

        @staticmethod
        def NthTrue(mask, n, axis=0):
            """
            Index of the n-th True along axis (0-indexed).
            """
            cum = np.cumsum(mask, axis=axis)
            target = (cum == (n + 1)) & mask
            hasNth = target.any(axis=axis)
            idx = np.argmax(target, axis=axis)
            return idx, hasNth

        @staticmethod
        def Test():
            mask=np.array([[False,True,True,False],
                           [True,True,False,False],
                           [True,False,True,False]])
            array=np.arange(mask.shape[0]*mask.shape[1]).reshape(mask.shape)
            print('mask: \n', mask)
            print('array: ', array,'\n')
            
            [firstTrue,lastTrue,hasTrue] = VectorizedOperations_Classes.MaskSearch.FindTrue(mask, axis=0)
            print('firstTrue: ', firstTrue)
            print('lastTrue: ', lastTrue)
            print('hasTrue: ', hasTrue, '\n')
            
            print("firstTrue_values: ", VectorizedOperations_Classes.MaskSearch.ApplyFindTrue(array,firstTrue,hasTrue))
            print("lastTrue_values: ", VectorizedOperations_Classes.MaskSearch.ApplyFindTrue(array,lastTrue,hasTrue))
        @staticmethod
        def Test2():
            array = np.arange(12).reshape(3,4)
            print('array: \n', array,'\n')
            
            innerIndex = np.array([1,0,0,1])[np.newaxis,:]
            outerIndex = np.array([1,2,2,1])[np.newaxis,:]
            print('innerIndex: ', innerIndex)
            print('outerIndex: ', outerIndex,'\n')
            
            result = SubsetIndices(array, innerIndex, outerIndex, axis=0)
            print('result: \n', result)
    class IndexSearch:
        @staticmethod
        def NearestIndex(array, target, axis=0):
            """Index of the closest value to `target` along axis (target can broadcast)."""
            diff = np.abs(array - target)
            return np.argmin(diff, axis=axis)

