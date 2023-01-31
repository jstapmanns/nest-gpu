import numpy
from cython.operator cimport dereference as deref
from cython cimport view
from libc.string cimport strlen, memcpy
from libc.stdlib cimport malloc, free
cimport llapi_helpers as llapi_h
from llapi_helpers import SynGroup, NestedLoopAlgo, NodeSeq, RemoteNodeSeq, ConnectionList, list_to_numpy_array

'''
helping functions
'''
cdef int GetNBoolParam():
    "Get number of kernel boolean parameters"

    cdef int ret = NESTGPU_GetNBoolParam()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNIntParam():
    "Get number of kernel int parameters"

    cdef int ret = NESTGPU_GetNIntParam()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef object GetIntParamNames():
    "Get list of kernel int parameter names"

    cdef int n_param = GetNIntParam()
    cdef char** param_name_pp = NESTGPU_GetIntParamNames()
    param_name_list = []
    for i in range(n_param):
        param_name_list.append(llapi_h.cstring_to_pystring(param_name_pp[i]))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return param_name_list

cdef int GetNFloatParam():
    "Get number of kernel float parameters"

    cdef int ret = NESTGPU_GetNFloatParam()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetRecordDataRows(int i_record):
    cdef int ret = NESTGPU_GetRecordDataRows(i_record)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetRecordDataColumns(int i_record):
    cdef int ret = NESTGPU_GetRecordDataColumns(i_record)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetSynGroupParam(object syn_group, object param_name, float val):
    "Set synapse group parameter value"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in SetSynGroupParam")
    cdef int i_syn_group = syn_group.i_syn_group
    cdef int ret = NESTGPU_SetSynGroupParam(i_syn_group, param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef object RandomNormal(size_t n, float mean, float stddev):
    "Generate n random floats with normal distribution in CUDA memory"
    cdef float* first = NESTGPU_RandomNormal(n, mean, stddev)
    ret = numpy.asarray(<float[:n]>first)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef object RandomNormalClipped(size_t n, float mean, float stddev,
        float vmin, float vmax, float vstep=0):
    "Generate n random floats with normal clipped distribution in CUDA memory"
    cdef float* first = NESTGPU_RandomNormalClipped(n, mean, stddev, vmin, vmax, vstep)
    ret = numpy.asarray(<float[:n]>first)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNeuronParamSize(int i_node, object param_name):
    "Get neuron parameter array size"
    cdef int ret = NESTGPU_GetNeuronParamSize(i_node, param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNeuronVarSize(int i_node, object var_name):
    "Get neuron variable array size"
    cdef int ret = NESTGPU_GetNeuronVarSize(i_node, var_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetSynGroupNParam(syn_group):
    "Get number of synapse parameters for a given synapse group"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupNParam")
    cdef int i_syn_group = syn_group.i_syn_group
    cdef int ret = NESTGPU_GetSynGroupNParam(i_syn_group)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef object GetSynGroupParamNames(object syn_group):
    "Get list of synapse group parameter names"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupParamNames")
    cdef int i_syn_group = syn_group.i_syn_group
    cdef int n_param = GetSynGroupNParam(syn_group)
    cdef char** param_name_pp = NESTGPU_GetSynGroupParamNames(i_syn_group)
    cdef char* param_name_p
    cdef list param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name_list.append(llapi_h.cstring_to_pystring(param_name_p))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return param_name_list

cdef float GetSynGroupParam(object syn_group, object param_name):
    "Get synapse group parameter value"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupParam")
    cdef int i_syn_group = syn_group.i_syn_group
    cdef float ret = NESTGPU_GetSynGroupParam(i_syn_group, param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNIntVar(int i_node):
    "Get number of integer variables for a given node"
    cdef int ret = NESTGPU_GetNIntVar(i_node)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNScalVar(int i_node):
    "Get number of scalar variables for a given node"
    cdef int ret = NESTGPU_GetNScalVar(i_node)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNScalParam(int i_node):
    "Get number of scalar parameters for a given node"
    cdef int ret = NESTGPU_GetNScalParam(i_node)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNPortVar(int i_node):
    "Get number of scalar variables for a given node"
    cdef int ret = NESTGPU_GetNPortVar(i_node)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNPortParam(int i_node):
    "Get number of scalar parameters for a given node"
    cdef int ret = NESTGPU_GetNPortParam(i_node)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNArrayVar(int i_node):
    "Get number of scalar variables for a given node"
    cdef int ret = NESTGPU_GetNArrayVar(i_node)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNArrayParam(int i_node):
    "Get number of scalar parameters for a given node"
    cdef int ret = NESTGPU_GetNArrayParam(i_node)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int GetNGroupParam(int i_node):
    "Get number of scalar parameters for a given node"
    cdef int ret = NESTGPU_GetNGroupParam(i_node)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int IsNeuronGroupParam(int i_node, object param_name):
    "Check name of neuron scalar parameter"
    cdef int ret = (NESTGPU_IsNeuronGroupParam(i_node, param_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronGroupParam(object nodes, object param_name, float val):
    cdef int ret = NESTGPU_SetNeuronGroupParam(nodes.i0, nodes.n,
            param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int IsNeuronScalParam(int i_node, object param_name):
    "Check name of neuron scalar parameter"
    cdef int ret = (NESTGPU_IsNeuronScalParam(i_node, param_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronScalParam(int i_node, int n_node, object param_name, float val):
    "Set neuron scalar parameter value"
    cdef int ret = NESTGPU_SetNeuronScalParam(i_node, n_node, param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int IsNeuronPortParam(int i_node, object param_name):
    "Check name of neuron scalar parameter"
    cdef int ret = (NESTGPU_IsNeuronPortParam(i_node, param_name.encode('utf-8'))!= 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int IsNeuronArrayParam(int i_node, object param_name):
    "Check name of neuron scalar parameter"
    cdef int ret = (NESTGPU_IsNeuronArrayParam(i_node, param_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronArrayParam(int i_node, int n_node, object param_name, object param_list):
    "Set neuron array parameter value"
    cdef int array_size = len(param_list)
    #TODO: is the following declaration correct?
    cdef numpy.ndarray array = list_to_numpy_array(param_list)
    cdef int ret = NESTGPU_SetNeuronArrayParam(i_node, n_node, param_name.encode('utf-8'),
            llapi_h.np_float_array_to_pointer(array),
            #&llapi_h.pylist_to_float_vec(param_list)[0],
                                       array_size)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int IsNeuronIntVar(int i_node, object var_name):
    "Check name of neuron integer variable"
    cdef int ret = (NESTGPU_IsNeuronIntVar(i_node, var_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronIntVar(int i_node, int n_node, object var_name, int val):
    "Set neuron integer variable value"
    cdef int ret = NESTGPU_SetNeuronIntVar(i_node, n_node, var_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int IsNeuronScalVar(int i_node, object var_name):
    "Check name of neuron scalar variable"
    cdef int ret = (NESTGPU_IsNeuronScalVar(i_node, var_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronScalVar(int i_node, int n_node, object var_name, float val):
    "Set neuron scalar variable value"
    cdef int ret = NESTGPU_SetNeuronScalVar(i_node, n_node, var_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int IsNeuronPortVar(int i_node, object var_name):
    "Check name of neuron scalar variable"
    cdef int ret = (NESTGPU_IsNeuronPortVar(i_node, var_name.encode('utf-8'))!= 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int IsNeuronArrayVar(int i_node, object var_name):
    "Check name of neuron array variable"
    cdef int ret = (NESTGPU_IsNeuronArrayVar(i_node, var_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronArrayVar(int i_node, int n_node, object var_name, object var_list):
    "Set neuron array variable value"
    cdef int array_size = len(var_list)
    cdef numpy.ndarray array = list_to_numpy_array(var_list)
    cdef int ret = NESTGPU_SetNeuronArrayVar(i_node, n_node, var_name.encode('utf-8'),
            llapi_h.np_float_array_to_pointer(array),
            #&llapi_h.pylist_to_float_vec(var_list)[0],
                                       array_size)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronPtArrayParam(object nodes, object param_name, object param_list):
    "Set neuron list array parameter value"
    cdef int n_node = len(nodes)
    cdef numpy.ndarray node_array = list_to_numpy_array(nodes)

    cdef int array_size = len(param_list)
    cdef numpy.ndarray param_array = list_to_numpy_array(param_list)
    cdef int ret = NESTGPU_SetNeuronPtArrayParam(llapi_h.np_int_array_to_pointer(node_array),
                                          n_node, param_name.encode('utf-8'),
                                          llapi_h.np_float_array_to_pointer(param_array),
                                          #&llapi_h.pylist_to_float_vec(param_list)[0],
                                          array_size)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronPtIntVar(object nodes, object var_name, int val):
    "Set neuron list integer variable value"
    cdef int n_node = len(nodes)
    cdef numpy.ndarray array = list_to_numpy_array(nodes)
    cdef int ret = NESTGPU_SetNeuronPtIntVar(llapi_h.np_int_array_to_pointer(array),
                                       n_node, var_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronPtScalVar(object nodes, object var_name, float val):
    "Set neuron list scalar variable value"
    cdef int n_node = len(nodes)
    cdef numpy.ndarray array = list_to_numpy_array(nodes)
    cdef int ret = NESTGPU_SetNeuronPtScalVar(llapi_h.np_int_array_to_pointer(array),
                                       n_node, var_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronPtArrayVar(object nodes, object var_name, object var_list):
    "Set neuron list array variable value"
    cdef int n_node = len(nodes)
    cdef numpy.ndarray node_array = list_to_numpy_array(nodes)

    cdef int array_size = len(var_list)
    cdef numpy.ndarray var_array = list_to_numpy_array(var_list)
    cdef int ret = NESTGPU_SetNeuronPtArrayVar(llapi_h.np_int_array_to_pointer(node_array),
                                        n_node, var_name.encode('utf-8'),
                                        llapi_h.np_float_array_to_pointer(var_array),
                                        #&llapi_h.pylist_to_float_vec(var_list)[0],
                                        array_size)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronScalParamDistr(int i_node, int n_node, object param_name):
    "Set neuron scalar parameter value using distribution or array"
    cdef int ret = NESTGPU_SetNeuronScalParamDistr(i_node,
                                          n_node, param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronScalVarDistr(int i_node, int n_node, object var_name):
    "Set neuron scalar variable using distribution or array"
    cdef int ret = NESTGPU_SetNeuronScalVarDistr(i_node,
                                          n_node, var_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret


cdef int SetNeuronPortParamDistr(int i_node,int  n_node, object param_name):
    "Set neuron port parameter value using distribution or array"
    cdef int ret = NESTGPU_SetNeuronPortParamDistr(i_node,
                                          n_node, param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronPortVarDistr(int i_node,int  n_node, object var_name):
    "Set neuron port variable using distribution or array"
    cdef int ret = NESTGPU_SetNeuronPortVarDistr(i_node,
                                        n_node, var_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronPtScalParamDistr(object nodes, object param_name):
    "Set neuron list scalar parameter using distribution or array"
    cdef int n_node = len(nodes)
    cdef numpy.ndarray node_array = list_to_numpy_array(nodes)
    cdef int ret = NESTGPU_SetNeuronPtScalParamDistr(llapi_h.np_int_array_to_pointer(node_array),
                                            n_node, param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronPtScalVarDistr(object nodes, object var_name):
    "Set neuron list scalar variable using distribution or array"
    cdef int n_node = len(nodes)
    cdef numpy.ndarray node_array = list_to_numpy_array(nodes)
    cdef int ret = NESTGPU_SetNeuronPtScalVarDistr(llapi_h.np_int_array_to_pointer(node_array),
                                          n_node, var_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronPtPortParamDistr(object nodes, object param_name):
    "Set neuron list port parameter using distribution or array"
    cdef int n_node = len(nodes)
    cdef numpy.ndarray node_array = list_to_numpy_array(nodes)
    cdef int ret = NESTGPU_SetNeuronPtPortParamDistr(llapi_h.np_int_array_to_pointer(node_array),
                                            n_node, param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetNeuronPtPortVarDistr(object nodes, object var_name):
    "Set neuron list port variable using distribution or array"
    cdef int n_node = len(nodes)
    cdef numpy.ndarray node_array = list_to_numpy_array(nodes)
    cdef int ret = NESTGPU_SetNeuronPtPortVarDistr(llapi_h.np_int_array_to_pointer(node_array),
                                          n_node, var_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetDistributionIntParam(object param_name, int val):
    "Set distribution integer parameter"
    cdef int ret = NESTGPU_SetDistributionIntParam(param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetDistributionScalParam(object param_name, int val):
    "Set distribution scalar parameter"
    cdef int ret = NESTGPU_SetDistributionScalParam(param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int SetDistributionVectParam(object param_name, int val, int i):
    "Set distribution vector parameter"
    cdef int ret = NESTGPU_SetDistributionVectParam(param_name.encode('utf-8'), val, i)
    if GetErrorCode() != 0:
        raise ValueError(GetErrorMessage())
    return ret

cdef int SetDistributionFloatPtParam(object param_name, object arr):
    "Set distribution pointer to float parameter"
    cdef numpy.ndarray array = list_to_numpy_array(arr)
    cdef int ret = NESTGPU_SetDistributionFloatPtParam(param_name.encode('utf-8'),
            llapi_h.np_int_array_to_pointer(array))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

cdef int IsDistributionFloatParam(object param_name):
    "Check name of distribution float parameter"
    cdef int ret = (NESTGPU_IsDistributionFloatParam(param_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret


'''
low level api
'''
def llapi_setOnException(int on_exception):
    "Define whether handle exceptions (1) or exit (0) in case of errors"
    return NESTGPU_SetOnException(on_exception)

def llapi_mpiId():
    "Get MPI Id"
    ret = NESTGPU_MpiId()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_connSpecInit():
    "Initialize connection rules specification"
    ret = NESTGPU_ConnSpecInit()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setConnSpecParam(object param_name, int val):
    "Set connection parameter"
    ret = NESTGPU_SetConnSpecParam(param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_connSpecIsParam(object param_name):
    "Check name of connection parameter"
    ret = (NESTGPU_ConnSpecIsParam(param_name.encode('utf-8')) != 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_synSpecInit():
    "Initializa synapse specification"
    ret = NESTGPU_SynSpecInit()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setSynSpecIntParam(object param_name, int val):
    "Set synapse int parameter"
    ret = NESTGPU_SetSynSpecIntParam(param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setSynSpecFloatParam(object param_name, float val):
    "Set synapse float parameter"
    ret = NESTGPU_SetSynSpecFloatParam(param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setSynSpecFloatPtParam(object param_name, object arr):
    "Set synapse pointer to float parameter"
    array = list_to_numpy_array(arr)
    print('using llapi_setSynSpecFloatPtParam() to set {}'.format(param_name))
    # TODO: I think both of the cases below are covered by converting the
    #       intput arr to a numpy array.
    #if (type(arr) is list)  | (type(arr) is tuple):
    #    arr = (ctypes.c_float * len(arr))(*arr)
    ret = NESTGPU_SetSynSpecFloatPtParam(param_name.encode('utf-8'),
            llapi_h.np_float_array_to_pointer(array))
            #&llapi_h.pylist_to_float_vec(arr)[0])
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_ruleArraySize(conn_dict, source, target):
    if conn_dict["rule"]=="one_to_one":
        array_size = len(source)
    elif conn_dict["rule"]=="all_to_all":
        array_size = len(source)*len(target)
    elif conn_dict["rule"]=="fixed_total_number":
        array_size = conn_dict["total_num"]
    elif conn_dict["rule"]=="fixed_indegree":
        array_size = len(target)*conn_dict["indegree"]
    elif conn_dict["rule"]=="fixed_outdegree":
        array_size = len(source)*conn_dict["outdegree"]
    else:
        raise ValueError("Unknown number of connections for this rule")
    return array_size

def llapi_setSynParamFromArray(param_name, par_dict, array_size):
    print('using llapi_setSynParamFromArray to set {}'.format(param_name))
    arr_param_name = param_name + "_array"
    if (not llapi_synSpecIsFloatPtParam(arr_param_name)):
        raise ValueError("Synapse parameter cannot be set by arrays or distributions")
    arr = llapi_dictToArray(par_dict, array_size)
    llapi_setSynSpecFloatPtParam(arr_param_name, arr)

def llapi_setSynGroupStatus(syn_group, params, val=None):
    "Set synapse group parameters using dictionaries"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in SetSynGroupStatus")
    if ((type(params)==dict) & (val==None)):
        for param_name in params:
            llapi_setSynGroupStatus(syn_group, param_name, params[param_name])
    elif (type(params)==str):
            return SetSynGroupParam(syn_group, params, val)
    else:
        raise ValueError("Wrong argument in SetSynGroupStatus")
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())

def llapi_synSpecIsIntParam(object param_name):
    "Check name of synapse int parameter"
    ret = (NESTGPU_SynSpecIsIntParam(param_name.encode('utf-8')) != 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_synSpecIsFloatParam(object param_name):
    "Check name of synapse float parameter"
    ret = (NESTGPU_SynSpecIsFloatParam(param_name.encode('utf-8')) != 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_synSpecIsFloatPtParam(object param_name):
    "Check name of synapse pointer to float parameter"
    ret = (NESTGPU_SynSpecIsFloatPtParam(param_name.encode('utf-8')) != 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_dictToArray(param_dict, array_size):
    dist_name = None
    arr = None
    low = -1.0e35
    high = 1.0e35
    mu = None
    sigma = None
    vstep = 0

    for param_name in param_dict:
        pval = param_dict[param_name]
        if param_name=="array":
            dist_name = "array"
            arr = pval
        elif param_name=="distribution":
            dist_name = pval
        elif param_name=="low":
            low = pval
        elif param_name=="high":
            high = pval
        elif param_name=="mu":
            mu = pval
        elif param_name=="sigma":
            sigma = pval
        elif param_name=="step":
            vstep = pval
        else:
            raise ValueError("Unknown parameter name in dictionary")

    if dist_name=="array":
        if (type(arr) is list) | (type(arr) is tuple):
            if len(arr) != array_size:
                raise ValueError("Wrong array size.")

            arr = list_to_numpy_array(arr)
        return arr
    elif dist_name=="normal":
        return RandomNormal(array_size, mu, sigma)
    elif dist_name=="normal_clipped":
        return RandomNormalClipped(array_size, mu, sigma, low, high, vstep)
    else:
        raise ValueError("Unknown distribution")

def llapi_getBoolParamNames():
    "Get list of kernel boolean parameter names"

    cdef int n_param = GetNBoolParam()
    cdef char** param_name_pp = NESTGPU_GetBoolParamNames()
    param_name_list = []
    for i in range(n_param):
        param_name_list.append(llapi_h.cstring_to_pystring(param_name_pp[i]))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return param_name_list

def llapi_isBoolParam(object param_name):
    "Check name of kernel boolean parameter"

    ret = (NESTGPU_IsBoolParam(param_name.encode('utf-8')) != 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getBoolParam(object param_name):
    "Get kernel boolean parameter value"

    cdef cbool ret = NESTGPU_GetBoolParam(param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setBoolParam(param_name, val):
    "Set kernel boolean parameter value"

    ret = NESTGPU_SetBoolParam(param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_isIntParam(object param_name):
    "Check name of kernel int parameter"

    ret = (NESTGPU_IsIntParam(param_name.encode('utf-8')) != 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getIntParam(object param_name):
    "Get kernel int parameter value"

    ret = NESTGPU_GetIntParam(param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setIntParam(object param_name, int val):
    "Set kernel int parameter value"

    ret = NESTGPU_SetIntParam(param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getFloatParamNames():
    "Get list of kernel float parameter names"

    cdef int n_param = GetNFloatParam()
    cdef char** param_name_pp = NESTGPU_GetFloatParamNames()
    param_name_list = []
    for i in range(n_param):
        param_name_list.append(llapi_h.cstring_to_pystring(param_name_pp[i]))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return param_name_list

def llapi_isFloatParam(object param_name):
    "Check name of kernel float parameter"

    ret = (NESTGPU_IsFloatParam(param_name.encode('utf-8')) != 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getFloatParam(object param_name):
    "Get kernel float parameter value"

    ret = NESTGPU_GetFloatParam(param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setFloatParam(object param_name, float val):
    "Set kernel float parameter value"

    # TODO: dangerous type conversion from python double to float?
    ret = NESTGPU_SetFloatParam(param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getErrorCode():
    "Get error code from NESTGPU exception"
    return NESTGPU_GetErrorCode()

def llapi_getErrorMessage():
    cdef char* err_message = NESTGPU_GetErrorMessage()
    #print(strlen(err_message))
    #cdef char[:] mview = <char[:strlen(err_message)]>(err_message)
    #np_byte_array = numpy.asarray(mview)
    #ret = ''.join(np.char.decode(np_byte_array, encoding='utf-8'))
    return llapi_h.cstring_to_pystring(err_message)

def llapi_connect(int i_source_node, int i_target_node,
			unsigned char port, unsigned char syn_group,
            float weight, float delay):
    print('using cython llapi_connect()')
    return NESTGPU_Connect(i_source_node, i_target_node,port, syn_group,
            weight, delay)

def llapi_connectSeqSeq(int i_source, int n_source, int i_target, int n_target):
    print('using cython llapi_connectSeqSeq()')
    return NESTGPU_ConnectSeqSeq(i_source, n_source, i_target, n_target)

def llapi_connectSeqGroup(int i_source, int n_source, object i_target, int n_target):
    print('using cython llapi_connectSeqGroup()')
    # TODO: The following line leads to unexpected behaviour if the indices in i_target
    #       are > 2147483647, which is the maximum int32.
    array = list_to_numpy_array(i_target)
    return NESTGPU_ConnectSeqGroup(i_source, n_source,
            llapi_h.np_int_array_to_pointer(array), n_target)

def llapi_connectGroupSeq(object i_source, int n_source, int i_target, int n_target):
    print('using cython llapi_connectGroupSeq()')
    array = list_to_numpy_array(i_source)
    return NESTGPU_ConnectGroupSeq(llapi_h.np_int_array_to_pointer(array),
            n_source, i_target, n_target)

def llapi_connectGroupGroup(object i_source, int n_source, object i_target, int n_target):
    print('using cython llapi_connectGroupGroup()')
    source_array = list_to_numpy_array(i_source)
    target_array = list_to_numpy_array(i_target)
    return NESTGPU_ConnectGroupGroup(llapi_h.np_int_array_to_pointer(source_array),
            n_source, llapi_h.np_int_array_to_pointer(target_array), n_target)

def llapi_remoteConnectSeqSeq(int i_source_host, int i_source, int n_source,
        int i_target_host, int i_target, int n_target):
    print('using cython llapi_remoteConnectSeqSeq()')
    return NESTGPU_RemoteConnectSeqSeq(i_source_host, i_source, n_source,
            i_target_host, i_target, n_target)

def llapi_remoteConnectSeqGroup(int i_source_host, int i_source, int n_source,
        int i_target_host, object i_target, int n_target):
    print('using cython llapi_remoteConnectSeqGroup()')
    array = list_to_numpy_array(i_target)
    return NESTGPU_RemoteConnectSeqGroup(i_source_host, i_source, n_source,
            i_target_host, llapi_h.np_int_array_to_pointer(array), n_target)

def llapi_remoteConnectGroupSeq(int i_source_host, object i_source, int n_source,
        int i_target_host, int i_target, int n_target):
    print('using cython llapi_remoteConnectGroupSeq()')
    array = list_to_numpy_array(i_source)
    return NESTGPU_RemoteConnectGroupSeq(i_source_host, llapi_h.np_int_array_to_pointer(array),
            n_source, i_target_host, i_target, n_target)

def llapi_remoteConnectGroupGroup(int i_source_host, object i_source, int n_source,
        int i_target_host, object i_target, int n_target):
    print('using cython llapi_remoteConnectGroupGroup()')
    source_array = list_to_numpy_array(i_source)
    target_array = list_to_numpy_array(i_target)
    return NESTGPU_RemoteConnectGroupGroup(i_source_host,
            llapi_h.np_int_array_to_pointer(source_array), n_source,
            i_target_host, llapi_h.np_int_array_to_pointer(target_array), n_target)

def llapi_create(model, int n, int n_port):
    print('using cython llapi_create() to create {} {}(s)'.format(n, model))
    return NESTGPU_Create(model.encode('utf-8'), n, n_port)

def llapi_setNestedLoopAlgo(int nested_loop_algo):
    return NESTGPU_SetNestedLoopAlgo(nested_loop_algo)

# TODO: should we change all the char* arguments to string as it is done in NEST?
#       answer: no, we do not want to change the C api, so we use the char* array.
# TODO: Does it make sense to replace python lists by numpy arrays?

def llapi_getSeqSeqConnections(int i_source, int n_source, int i_target,
        int n_target, int syn_group):
    #print('using cython llapi_getSeqSeqConnections()')
    cdef int n_conn
    cdef int* c_ret
    c_ret = NESTGPU_GetSeqSeqConnections(i_source, n_source, i_target, n_target, syn_group, &n_conn)
    ret = llapi_h.int_array_to_conn_list(c_ret, n_conn)
    return ret

def llapi_getSeqGroupConnections(int i_source, int n_source, object i_target,
        int n_target, int syn_group):
    #print('using cython llapi_getSeqGroupConnections()')
    target_array = list_to_numpy_array(i_target)
    cdef int n_conn
    cdef int* c_ret
    c_ret = NESTGPU_GetSeqGroupConnections(i_source, n_source,
            llapi_h.np_int_array_to_pointer(target_array), n_target, syn_group, &n_conn)
    ret = llapi_h.int_array_to_conn_list(c_ret, n_conn)
    return ret

def llapi_getGroupSeqConnections(object i_source, int n_source, int i_target,
        int n_target, int syn_group):
    #print('using cython llapi_getGroupSeqConnections()')
    source_array = list_to_numpy_array(i_source)
    cdef int n_conn
    cdef int* c_ret
    c_ret = NESTGPU_GetGroupSeqConnections(llapi_h.np_int_array_to_pointer(source_array),
            n_source, i_target, n_target, syn_group, &n_conn)
    #if GetErrorCode() != 0:
    #    raise ValueError(GetErrorMessage())
    ret = llapi_h.int_array_to_conn_list(c_ret, n_conn)
    return ret

def llapi_getGroupGroupConnections(object i_source, int n_source, object i_target,
        int n_target, int syn_group):
    #print('using cython llapi_getGroupGroupConnections()')
    source_array = list_to_numpy_array(i_source)
    target_array = list_to_numpy_array(i_target)
    cdef int n_conn
    cdef int* c_ret
    c_ret = NESTGPU_GetGroupGroupConnections(llapi_h.np_int_array_to_pointer(source_array),
            n_source, llapi_h.np_int_array_to_pointer(target_array), n_target, syn_group, &n_conn)
    ret = llapi_h.int_array_to_conn_list(c_ret, n_conn)
    return ret

def llapi_createRecord(object file_name, object var_name_arr,
        object i_node_arr, object port_arr, int n_node):
    print('using cython llapi_createRecord()')
    node_array = list_to_numpy_array(i_node_arr)
    port_array = list_to_numpy_array(port_arr)
    ret = NESTGPU_CreateRecord(file_name.encode('utf-8'),
            llapi_h.pystring_list_to_cstring_array(var_name_arr),
            llapi_h.np_int_array_to_pointer(node_array),
            llapi_h.np_int_array_to_pointer(port_array), n_node)
    return ret

def llapi_getRecordData(int i_record):
    print('using cython llapi_getRecordData()')
    n_row = GetRecordDataRows(i_record)
    n_col = GetRecordDataColumns(i_record)
    cdef float** data_array_p = NESTGPU_GetRecordData(i_record)
    np_data_array = llapi_h.float_array2d_to_numpy2d(data_array_p, n_row, n_col)
    ret = np_data_array.tolist()
    return ret

def llapi_setSimTime(float sim_time):
    print('using cython llapi_setSimTime()')
    ret = NESTGPU_SetSimTime(sim_time)
    return ret

def llapi_simulate():
    print('using cython llapi_simulate()')
    ret = NESTGPU_Simulate()
    return ret

def llapi_setNeuronPtScalParam(object nodes, object param_name, float val):
    "Set neuron list scalar parameter value"
    n_node = len(nodes)
    array = list_to_numpy_array(nodes)
    ret = NESTGPU_SetNeuronPtScalParam(llapi_h.np_int_array_to_pointer(array),
                                         n_node, param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_calibrate():
    "Calibrate simulation"
    ret = NESTGPU_Calibrate()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setRandomSeed(seed):
    "Set seed for random number generation"
    ret = NESTGPU_SetRandomSeed(seed)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setTimeResolution(time_res):
    "Set time resolution in ms"
    ret = NESTGPU_SetTimeResolution(time_res)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getTimeResolution():
    "Get time resolution in ms"
    ret = NESTGPU_GetTimeResolution()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setMaxSpikeBufferSize(max_size):
    "Set maximum size of spike buffer per node"
    ret = NESTGPU_SetMaxSpikeBufferSize(max_size)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getMaxSpikeBufferSize():
    "Get maximum size of spike buffer per node"
    ret = NESTGPU_GetMaxSpikeBufferSize()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setVerbosityLevel(verbosity_level):
    "Set verbosity level"
    ret = NESTGPU_SetVerbosityLevel(verbosity_level)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getNeuronParam(int i_node, int n_node, object param_name):
    "Get neuron parameter value"
    cdef float* first = NESTGPU_GetNeuronParam(i_node,
                                       n_node, param_name.encode('utf-8'))

    cdef int array_size = GetNeuronParamSize(i_node, param_name)
    ret = numpy.asarray(<float[:n_node*array_size]>first)
    if (array_size>1):
        ret = ret.reshape((n_node, array_size))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getNeuronPtParam(object nodes, object param_name):
    "Get neuron list scalar parameter value"
    n_node = len(nodes)
    cdef float* first = NESTGPU_GetNeuronPtParam(&llapi_h.pylist_to_int_vec(nodes)[0],
                                         n_node, param_name.encode('utf-8'))
    cdef int array_size = GetNeuronParamSize(nodes[0], param_name)
    ret = numpy.asarray(<float[:n_node*array_size]>first)
    if (array_size>1):
        ret = ret.reshape((n_node, array_size))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getArrayParam(int i_node, int n_node, object param_name):
    "Get neuron array parameter"
    data_list = []
    cdef float* first
    for j_node in range(n_node):
        i_node1 = i_node + j_node
        first = NESTGPU_GetArrayParam(i_node1, param_name.encode('utf-8'))
        array_size = GetNeuronParamSize(i_node1, param_name)
        row_arr = numpy.asarray(<float[:array_size]>first)
        data_list.append(row_arr)

    ret = data_list
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getNeuronGroupParam(int i_node, object param_name):
    "Check name of neuron group parameter"
    ret = NESTGPU_GetNeuronGroupParam(i_node, param_name.encode('utf-8'))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getNeuronIntVar(int i_node, int n_node, object var_name):
    "Get neuron integer variable value"
    cdef int* first = NESTGPU_GetNeuronIntVar(i_node,
                                        n_node, var_name.encode('utf-8'))
    data_array = numpy.asarray(<int[:n_node]>first)
    ret = data_array
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getNeuronVar(int i_node, int n_node, object var_name):
    "Get neuron variable value"
    cdef float* first = NESTGPU_GetNeuronVar(i_node,
                                       n_node, var_name.encode('utf-8'))

    cdef int array_size = GetNeuronVarSize(i_node, var_name)
    ret = numpy.asarray(<float[:n_node*array_size]>first)
    if (array_size>1):
        ret = ret.reshape((n_node, array_size))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getNeuronPtIntVar(object nodes, object var_name):
    "Get neuron list integer variable value"
    n_node = len(nodes)
    cdef int* first = NESTGPU_GetNeuronPtIntVar(&llapi_h.pylist_to_int_vec(nodes)[0],
                                          n_node, var_name.encode('utf-8'))
    data_array = numpy.asarray(<int[:n_node]>first)
    # TODO: the reshaping below is required for compatibility but seems unnecessary
    # because it simply adds an extra dimension which is removed in GetStatus()
    ret = data_array.reshape((n_node,1))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getNeuronPtVar(object nodes, object var_name):
    "Get neuron list scalar variable value"
    n_node = len(nodes)
    cdef float* first = NESTGPU_GetNeuronPtVar(&llapi_h.pylist_to_int_vec(nodes)[0],
                                       n_node, var_name.encode('utf-8'))
    cdef int array_size = GetNeuronVarSize(nodes[0], var_name)
    ret = numpy.asarray(<float[:n_node*array_size]>first)
    if (array_size>1):
        ret = ret.reshape((n_node, array_size))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getArrayVar(int i_node, int n_node, object var_name):
    "Get neuron array variable"
    data_list = []
    cdef float* first
    for j_node in range(n_node):
        i_node1 = i_node + j_node
        first = NESTGPU_GetArrayVar(i_node1, var_name.encode('utf-8'))
        array_size = GetNeuronVarSize(i_node1, var_name)
        row_arr = numpy.asarray(<float[:array_size]>first)
        data_list.append(row_arr)

    ret = data_list
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getNeuronListArrayVar(object node_list, object var_name):
    "Get neuron array variable"
    data_list = []
    cdef float* first
    for i_node in node_list:
        first = NESTGPU_GetArrayVar(i_node, var_name.encode('utf-8'))
        array_size = GetNeuronVarSize(i_node, var_name)
        row_arr = numpy.asarray(<float[:array_size]>first)
        data_list.append(row_arr)

    ret = data_list
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getNeuronListArrayParam(node_list, param_name):
    "Get neuron array parameter"
    data_list = []
    cdef float* first
    for i_node in node_list:
        first = NESTGPU_GetArrayParam(i_node, param_name.encode('utf-8'))
        array_size = GetNeuronParamSize(i_node, param_name)
        row_arr = numpy.asarray(<float[:array_size]>first)
        data_list.append(row_arr)

    ret = data_list
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret


def llapi_connectMpiInit(int argc, object var_name_list):
    "Initialize MPI connections"
    from mpi4py import MPI
    ret = NESTGPU_ConnectMpiInit(argc, llapi_h.pystring_list_to_cstring_array(var_name_list))
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_mpiNp():
    "Get MPI Np"
    ret = NESTGPU_MpiNp()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getConnectionStatus(int i_source, int i_group, int i_conn):
    cdef int i_target = 0
    cdef unsigned char i_port = 0#''.encode('utf-8')
    cdef unsigned char i_syn = 0#''.encode('utf-8')
    cdef float delay = 0.0
    cdef float weight = 0.0
    ret = NESTGPU_GetConnectionStatus(i_source, i_group, i_conn,
                    &i_target, &i_port, &i_syn, &delay, &weight)
    ret_dict = {'target':i_target, 'port':ord(i_port), 'syn':ord(i_syn),
            'delay':delay, 'weight':weight}
    return ret_dict

def llapi_getRecSpikeTimes(int i_node, int n_node):
    "Get recorded spike times for node group"
    print('using cython llapi_getRecSpikeTimes()')
    cdef int* n_spike_times_pt
    cdef float** spike_times_pt
    spike_time_list = []
    ret1 = NESTGPU_GetRecSpikeTimes(i_node, n_node,
                                    &n_spike_times_pt, &spike_times_pt)
    for i_n in range(n_node):
        spike_time_list.append([])
        n_spike = n_spike_times_pt[i_n]
        for i_spike in range(n_spike):
            spike_time_list[i_n].append(spike_times_pt[i_n][i_spike])

    ret = spike_time_list
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_activateSpikeCount(int i_node, int n_node):
    "Activate spike count for node group"
    ret = NESTGPU_ActivateSpikeCount(i_node, n_node)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_activateRecSpikeTimes(int i_node, int n_node, int max_n_rec_spike_times):
    "Activate spike time recording for node group"
    ret = NESTGPU_ActivateRecSpikeTimes(i_node, n_node, max_n_rec_spike_times)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setRecSpikeTimesStep(int i_node, int n_node, int rec_spike_times_step):
    "Setp number of time steps for buffering spike time recording"
    ret = NESTGPU_SetRecSpikeTimesStep(i_node, n_node, rec_spike_times_step)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_getSynGroupStatus(object syn_group, object var_key=None):
    "Get synapse group status"
    if type(syn_group)!=SynGroup:
        raise ValueError("Wrong argument type in GetSynGroupStatus")
    if (type(var_key)==list) | (type(var_key)==tuple):
        status_list = []
        for var_elem in var_key:
            var_value = llapi_getSynGroupStatus(syn_group, var_elem)
            status_list.append(var_value)
        return status_list
    elif (var_key==None):
        status_dict = {}
        name_list = GetSynGroupParamNames(syn_group)
        for param_name in name_list:
            val = llapi_getSynGroupStatus(syn_group, param_name)
            status_dict[param_name] = val
        return status_dict
    elif (type(var_key)==str) | (type(var_key)==bytes):
            return GetSynGroupParam(syn_group, var_key)
    else:
        raise ValueError("Unknown key type in GetSynGroupStatus", type(var_key))

def llapi_getIntVarNames(int i_node):
    "Get list of scalar variable names"
    cdef int n_var = GetNIntVar(i_node)
    cdef char** var_name_pp = NESTGPU_GetIntVarNames(i_node)
    cdef char* var_name_p
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name_list.append(llapi_h.cstring_to_pystring(var_name_p))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return var_name_list

def llapi_getScalVarNames(int i_node):
    "Get list of scalar variable names"
    cdef int n_var = GetNScalVar(i_node)
    cdef char** var_name_pp = NESTGPU_GetScalVarNames(i_node)
    cdef char* var_name_p
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name_list.append(llapi_h.cstring_to_pystring(var_name_p))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return var_name_list

def llapi_getScalParamNames(int i_node):
    "Get list of scalar parameter names"
    cdef int n_param = GetNScalParam(i_node)
    cdef char** param_name_pp = NESTGPU_GetScalParamNames(i_node)
    cdef char* param_name_p
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name_list.append(llapi_h.cstring_to_pystring(param_name_p))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return param_name_list

def llapi_getPortVarNames(int i_node):
    "Get list of scalar variable names"
    cdef int n_var = GetNPortVar(i_node)
    cdef char** var_name_pp = NESTGPU_GetPortVarNames(i_node)
    cdef char* var_name_p
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name_list.append(llapi_h.cstring_to_pystring(var_name_p))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return var_name_list

def llapi_getPortParamNames(int i_node):
    "Get list of scalar parameter names"
    cdef int n_param = GetNPortParam(i_node)
    cdef char** param_name_pp = NESTGPU_GetPortParamNames(i_node)
    cdef char* param_name_p
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name_list.append(llapi_h.cstring_to_pystring(param_name_p))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return param_name_list

def llapi_getArrayVarNames(int i_node):
    "Get list of scalar variable names"
    cdef int n_var = GetNArrayVar(i_node)
    cdef char** var_name_pp = NESTGPU_GetArrayVarNames(i_node)
    cdef char* var_name_p
    var_name_list = []
    for i in range(n_var):
        var_name_p = var_name_pp[i]
        var_name_list.append(llapi_h.cstring_to_pystring(var_name_p))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return var_name_list

def llapi_getArrayParamNames(int i_node):
    "Get list of scalar parameter names"
    cdef int n_param = GetNArrayParam(i_node)
    cdef char** param_name_pp = NESTGPU_GetArrayParamNames(i_node)
    cdef char* param_name_p
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name_list.append(llapi_h.cstring_to_pystring(param_name_p))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return param_name_list

def llapi_getGroupParamNames(int i_node):
    "Get list of scalar parameter names"
    cdef int n_param = GetNGroupParam(i_node)
    cdef char** param_name_pp = NESTGPU_GetGroupParamNames(i_node)
    cdef char* param_name_p
    param_name_list = []
    for i in range(n_param):
        param_name_p = param_name_pp[i]
        param_name_list.append(llapi_h.cstring_to_pystring(param_name_p))

    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return param_name_list

def llapi_remoteCreate(int i_host, object model, int n, int n_port):
    print('using cython llapi_remoteCreate() to create {} {}(s)'.format(n, model))
    return NESTGPU_RemoteCreate(i_host, model.encode('utf-8'), n, n_port)

def llapi_createSynGroup(object model_name):
    print('using llapi_createSynGroup() to create {}'.format(model_name))
    return NESTGPU_CreateSynGroup(model_name.encode('utf-8'))

def llapi_setNeuronStatus(object nodes, object var_name, object val):
    "Set neuron group scalar or array variable or parameter"
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    if (type(val)==dict):
        if ((type(nodes)==NodeSeq
             and (IsNeuronScalParam(nodes.i0, var_name)
                  or IsNeuronScalVar(nodes.i0, var_name)
                  or IsNeuronPortParam(nodes.i0, var_name)
                  or IsNeuronPortVar(nodes.i0, var_name)))
            or IsNeuronScalParam(nodes[0], var_name)
            or IsNeuronScalVar(nodes[0], var_name)
            or IsNeuronPortParam(nodes[0], var_name)
            or IsNeuronPortVar(nodes[0], var_name)):
            for dict_param_name in val:
                pval = val[dict_param_name]
                if dict_param_name=="array":
                    SetDistributionFloatPtParam("array_pt", pval)
                    distr_idx = distribution_dict["array"]
                    SetDistributionIntParam("distr_idx", distr_idx)
                elif dict_param_name=="distribution":
                    distr_idx = distribution_dict[pval]
                    SetDistributionIntParam("distr_idx", distr_idx)
                else:
                    if IsDistributionFloatParam(dict_param_name):
                        if ((type(nodes)==NodeSeq
                            and (IsNeuronScalParam(nodes.i0, var_name)
                                 or IsNeuronScalVar(nodes.i0, var_name)))
                            or IsNeuronScalParam(nodes[0], var_name)
                            or IsNeuronScalVar(nodes[0], var_name)):
                            SetDistributionIntParam("vect_size", 1)
                            SetDistributionScalParam(dict_param_name, pval)
                        elif ((type(nodes)==NodeSeq
                            and (IsNeuronPortParam(nodes.i0, var_name)
                                 or IsNeuronPortVar(nodes.i0, var_name)))
                            or IsNeuronPortParam(nodes[0], var_name)
                            or IsNeuronPortVar(nodes[0], var_name)):
                            SetDistributionIntParam("vect_size", len(pval))
                            for i, value in enumerate(pval):
                                SetDistributionVectParam(dict_param_name,
                                                          value, i)
                    else:
                        print("Parameter name: ", dict_param_name)
                        raise ValueError("Unknown distribution parameter")
            # set values from array or from distribution
            if type(nodes)==NodeSeq:
                if IsNeuronScalParam(nodes.i0, var_name):
                    SetNeuronScalParamDistr(nodes.i0, nodes.n, var_name)
                elif IsNeuronScalVar(nodes.i0, var_name):
                    SetNeuronScalVarDistr(nodes.i0, nodes.n, var_name)
                elif IsNeuronPortParam(nodes.i0, var_name):
                    SetNeuronPortParamDistr(nodes.i0, nodes.n, var_name)
                elif IsNeuronPortVar(nodes.i0, var_name):
                    SetNeuronPortVarDistr(nodes.i0, nodes.n, var_name)
                else:
                    raise ValueError("Unknown neuron variable or parameter")

            else:
                if IsNeuronScalParam(nodes[0], var_name):
                    SetNeuronPtScalParamDistr(nodes, var_name)
                elif IsNeuronScalVar(nodes[0], var_name):
                    SetNeuronPtScalVarDistr(nodes, var_name)
                elif IsNeuronPortParam(nodes[0], var_name):
                    SetNeuronPtPortParamDistr(nodes, var_name)
                elif IsNeuronPortVar(nodes[0], var_name):
                    SetNeuronPtPortVarDistr(nodes, var_name)
                else:
                    raise ValueError("Unknown neuron variable or parameter")

        else:
            print("Parameter or variable ", var_name)
            raise ValueError("cannot be initialized by arrays or distributions")

    elif type(nodes)==NodeSeq:
        if IsNeuronGroupParam(nodes.i0, var_name):
            SetNeuronGroupParam(nodes, var_name, val)
        elif IsNeuronScalParam(nodes.i0, var_name):
            SetNeuronScalParam(nodes.i0, nodes.n, var_name, val)
        elif (IsNeuronPortParam(nodes.i0, var_name) |
              IsNeuronArrayParam(nodes.i0, var_name)):
            SetNeuronArrayParam(nodes.i0, nodes.n, var_name, val)
        elif IsNeuronIntVar(nodes.i0, var_name):
            SetNeuronIntVar(nodes.i0, nodes.n, var_name, val)
        elif IsNeuronScalVar(nodes.i0, var_name):
            SetNeuronScalVar(nodes.i0, nodes.n, var_name, val)
        elif (IsNeuronPortVar(nodes.i0, var_name) |
              IsNeuronArrayVar(nodes.i0, var_name)):
            SetNeuronArrayVar(nodes.i0, nodes.n, var_name, val)
        else:
            raise ValueError("Unknown neuron variable or parameter")
    else:
        if IsNeuronScalParam(nodes[0], var_name):
            SetNeuronPtScalParam(nodes, var_name, val)
        elif (IsNeuronPortParam(nodes[0], var_name) |
              IsNeuronArrayParam(nodes[0], var_name)):
            SetNeuronPtArrayParam(nodes, var_name, val)
        elif IsNeuronIntVar(nodes[0], var_name):
            SetNeuronPtIntVar(nodes, var_name, val)
        elif IsNeuronScalVar(nodes[0], var_name):
            SetNeuronPtScalVar(nodes, var_name, val)
        elif (IsNeuronPortVar(nodes[0], var_name) |
              IsNeuronArrayVar(nodes[0], var_name)):
            SetNeuronPtArrayVar(nodes, var_name, val)
        else:
            raise ValueError("Unknown neuron variable or parameter")


