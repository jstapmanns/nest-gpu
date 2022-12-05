import numpy
from cython.operator cimport dereference as deref
from cython cimport view
from libc.string cimport strlen, memcpy
from libc.stdlib cimport malloc, free

class ConnectionId(object):
    def __init__(self, i_source, i_group, i_conn):
        self.i_source = i_source
        self.i_group = i_group
        self.i_conn = i_conn

class SynGroup(object):
    def __init__(self, i_syn_group):
        self.i_syn_group = i_syn_group

'''
helping functions
'''
cdef int* np_int_array_to_pointer(object array):
    '''
    function to get a pointer to the first element of a numpy array of dtype np.int32
    returns int* to the first element of array
    '''
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a 1-dimensional NumPy array of ints, got {}'.format(
            type(array)))
    if not array.ndim == 1:
        raise TypeError('array must be a 1-dimensional NumPy array, got {}-dimensional NumPy array'.format(
            array.ndim))
    if not numpy.issubdtype(array.dtype, numpy.integer):
        raise TypeError('array must be a NumPy array of ints, got {}'.format(array.dtype))

    # Pointer to the first element in the Numpy array
    cdef int* array_int_ptr
    cdef int* c_array = <int *> malloc(len(array) * sizeof(int))
    for i in range(len(array)):
        c_array[i] = array[i]

    array_int_ptr = &c_array[0]

    return array_int_ptr

cdef float* np_float_array_to_pointer(object array):
    '''
    function to get a pointer to the first element of a numpy array of dtype np.int32
    returns int* to the first element of array
    '''
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a 1-dimensional NumPy array of ints, got {}'.format(
            type(array)))
    if not array.ndim == 1:
        raise TypeError('array must be a 1-dimensional NumPy array, got {}-dimensional NumPy array'.format(
            array.ndim))
    if not numpy.issubdtype(array.dtype, numpy.float32):
        raise TypeError('array must be a NumPy array of floats, got {}'.format(array.dtype))

    cdef float* array_float_ptr
    # TODO: the commented code using memory view does not seem to be a safe way to obtain a pointer
    #       to the first element of the array because tests revealed that the life time of the array
    #       is too short and the memory is freed while the data is still required by NESTGPU.
    # Get pointers to the first element in the Numpy array
    #cdef float[::1] array_float_mv

    #if numpy.issubdtype(array.dtype, numpy.float32):
    #    array_float_mv = numpy.ascontiguousarray(array, dtype=numpy.float32)
    #    array_float_ptr = < float* > &array_float_mv[0]
    #else:
    #    raise TypeError('array must be a NumPy array of floats, got {}'.format(array.dtype))

    # TODO: instead, the solution below seems to work properly.
    #       Still need to check: is the for loop slow?
    cdef float *c_array = <float *> malloc(len(array) * sizeof(float))
    for i in range(len(array)):
        c_array[i] = array[i]

    array_float_ptr = &c_array[0]
    return array_float_ptr

cdef long* np_long_array_to_pointer(object array):
    '''
    function to get a pointer to the first element of a numpy array of dtype np.int64
    returns long* to the first element of array
    '''
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a 1-dimensional NumPy array of ints, got {}'.format(
            type(array)))
    if not array.ndim == 1:
        raise TypeError('array must be a 1-dimensional NumPy array, got {}-dimensional NumPy array'.format(
            array.ndim))
    if not numpy.issubdtype(array.dtype, numpy.integer):
        raise TypeError('array must be a NumPy array of ints, got {}'.format(array.dtype))

    # Pointer to the first element in the Numpy array
    cdef long* array_long_ptr
    cdef long* c_array = <long *> malloc(len(array) * sizeof(long))
    for i in range(len(array)):
        c_array[i] = array[i]

    array_long_ptr = &c_array[0]

    return array_long_ptr

cdef object int_array_to_pylist(int* first, int n_elements):
    '''
    this function converts a C array of ints into a python list.
    arguments:
    first: pointer to first element of the C array
    n_elements: number of elements (size of the array)
    returns: python list with the same content as the C array
    '''
    pylist = [None]*n_elements
    for i in range(n_elements):
        pylist[i] = deref(first)
        first+=1

    return pylist

cdef object int_array_to_conn_list(int* first, int n_conn):
    '''
    this function is used in the llapi_getConnections routines.
    It converts a C array of ints into a pyhton list that
    contains the connection IDs.
    '''
    #print('llapi: number of connections: {}'.format(n_conn))
    conn_arr = numpy.asarray(<int[:3*n_conn]>first)
    conn_list = []
    for i_conn in range(n_conn):
        conn_id = ConnectionId(conn_arr[i_conn*3], conn_arr[i_conn*3 + 1],
                   conn_arr[i_conn*3 + 2])
        conn_list.append(conn_id)

    conn_list
    #print('llpai: length of conn_list: {}'.format(len(conn_list)))
    return conn_list

cdef object float_array2d_to_numpy2d(float** first_p, int n_row, int n_col):
    '''
    this function converts a C array of float arrays into a numpy 2d array.
    '''
    cdef float* first
    cdef float[:,::1] memview_arr = numpy.empty((n_row, n_col), dtype=numpy.float32)
    cdef float[::1] memview_row
    for i in range(n_row):
        first = first_p[i]
        memview_row = <float[:n_col]>first
        memview_arr[i] = memview_row

    ret = numpy.asarray(memview_arr)
    return ret


cdef object cstring_to_pystring(char* cstr):
    np_byte_array = numpy.asarray(<char[:strlen(cstr)]>(cstr))
    ret = ''.join(numpy.char.decode(np_byte_array, encoding='utf-8'))
    return ret

cdef char** pystring_list_to_cstring_array(object pystring_list):
    cdef char** cstring_array = <char**>malloc(sizeof(char*) * len(pystring_list))
    # TODO: Is this dangerous? Do we have to store the encoded bytes in a separate list before
    # we fill the c array?
    for i, s in enumerate(pystring_list):
        py_byte_string = s.encode('utf-8')
        cstring_array[i] = py_byte_string

    return cstring_array

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
        param_name_list.append(cstring_to_pystring(param_name_pp[i]))

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
    array = numpy.array(arr, dtype=numpy.float32, copy=True, order='C')
    print('using llapi_setSynSpecFloatPtParam()')
    # TODO: I think both of the cases below are covered by converting the
    #       intput arr to a numpy array.
    #if (type(arr) is list)  | (type(arr) is tuple):
    #    arr = (ctypes.c_float * len(arr))(*arr)
    ret = NESTGPU_SetSynSpecFloatPtParam(param_name.encode('utf-8'),
            np_float_array_to_pointer(array))
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

            arr = numpy.array(arr, dtype=numpy.float32, copy=True, order='C')
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
        param_name_list.append(cstring_to_pystring(param_name_pp[i]))

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
        param_name_list.append(cstring_to_pystring(param_name_pp[i]))

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
    return cstring_to_pystring(err_message)

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
    array = numpy.array(i_target, dtype=numpy.int32, copy=True, order='C')
    return NESTGPU_ConnectSeqGroup(i_source, n_source, np_int_array_to_pointer(array), n_target)

def llapi_connectGroupSeq(object i_source, int n_source, int i_target, int n_target):
    print('using cython llapi_connectGroupSeq()')
    array = numpy.array(i_source, dtype=numpy.int32, copy=True, order='C')
    return NESTGPU_ConnectGroupSeq(np_int_array_to_pointer(array), n_source, i_target, n_target)

def llapi_connectGroupGroup(object i_source, int n_source, object i_target, int n_target):
    print('using cython llapi_connectGroupGroup()')
    source_array = numpy.array(i_source, dtype=numpy.int32, copy=True, order='C')
    target_array = numpy.array(i_target, dtype=numpy.int32, copy=True, order='C')
    return NESTGPU_ConnectGroupGroup(np_int_array_to_pointer(source_array), n_source,
            np_int_array_to_pointer(target_array), n_target)

def llapi_create(model, int n, int n_port):
    print('using cython llapi_create() to create {} {}(s)'.format(n, model))
    return NESTGPU_Create(model.encode('utf-8'), n, n_port)

# TODO: should we change all the char* arguments to string as it is done in NEST?
#       answer: no, we do not want to change the C api, so we use the char* array.
# TODO: can we make use of boost? (not necessary but the new NEST api does, I think.
# TODO: Does it make sense to replace python lists by numpy arrays?

def llapi_getSeqSeqConnections(int i_source, int n_source, int i_target,
        int n_target, int syn_group):
    #print('using cython llapi_getSeqSeqConnections()')
    cdef int n_conn
    cdef int* c_ret
    c_ret = NESTGPU_GetSeqSeqConnections(i_source, n_source, i_target, n_target, syn_group, &n_conn)
    ret = int_array_to_conn_list(c_ret, n_conn)
    return ret

def llapi_getSeqGroupConnections(int i_source, int n_source, object i_target,
        int n_target, int syn_group):
    #print('using cython llapi_getSeqGroupConnections()')
    target_array = numpy.array(i_target, dtype=numpy.int32, copy=True, order='C')
    cdef int n_conn
    cdef int* c_ret
    c_ret = NESTGPU_GetSeqGroupConnections(i_source, n_source, np_int_array_to_pointer(target_array),
            n_target, syn_group, &n_conn)
    ret = int_array_to_conn_list(c_ret, n_conn)
    return ret

def llapi_getGroupSeqConnections(object i_source, int n_source, int i_target,
        int n_target, int syn_group):
    #print('using cython llapi_getGroupSeqConnections()')
    source_array = numpy.array(i_source, dtype=numpy.int32, copy=True, order='C')
    cdef int n_conn
    cdef int* c_ret
    c_ret = NESTGPU_GetGroupSeqConnections(np_int_array_to_pointer(source_array), n_source, i_target,
            n_target, syn_group, &n_conn)
    #if GetErrorCode() != 0:
    #    raise ValueError(GetErrorMessage())
    ret = int_array_to_conn_list(c_ret, n_conn)
    return ret

def llapi_getGroupGroupConnections(object i_source, int n_source, object i_target,
        int n_target, int syn_group):
    #print('using cython llapi_getGroupGroupConnections()')
    source_array = numpy.array(i_source, dtype=numpy.int32, copy=True, order='C')
    target_array = numpy.array(i_target, dtype=numpy.int32, copy=True, order='C')
    cdef int n_conn
    cdef int* c_ret
    c_ret = NESTGPU_GetGroupGroupConnections(np_int_array_to_pointer(source_array), n_source,
            np_int_array_to_pointer(target_array), n_target, syn_group, &n_conn)
    ret = int_array_to_conn_list(c_ret, n_conn)
    return ret

def llapi_createRecord(object file_name, object var_name_arr,
        object i_node_arr, object port_arr, int n_node):
    print('using cython llapi_createRecord()')
    node_array = numpy.array(i_node_arr, dtype=numpy.int32, copy=True, order='C')
    port_array = numpy.array(port_arr, dtype=numpy.int32, copy=True, order='C')
    ret = NESTGPU_CreateRecord(file_name.encode('utf-8'),
            pystring_list_to_cstring_array(var_name_arr), np_int_array_to_pointer(node_array),
            np_int_array_to_pointer(port_array), n_node)
    return ret

def llapi_getRecordData(int i_record):
    print('using cython llapi_getRecordData()')
    n_row = GetRecordDataRows(i_record)
    n_col = GetRecordDataColumns(i_record)
    cdef float** data_array_p = NESTGPU_GetRecordData(i_record)
    np_data_array = float_array2d_to_numpy2d(data_array_p, n_row, n_col)
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

def llapi_setNeuronScalParam(int i_node, int n_node, object param_name, float val):
    "Set neuron scalar parameter value"
    ret = NESTGPU_SetNeuronScalParam(i_node, n_node, param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronArrayParam(int i_node, int n_node, object param_name, object param_list):
    "Set neuron array parameter value"
    cdef int array_size = len(param_list)
    array = numpy.array(param_list, dtype=numpy.float32, copy=True, order='C')
    ret = NESTGPU_SetNeuronArrayParam(i_node, n_node, param_name.encode('utf-8'),
                                       np_float_array_to_pointer(array), array_size)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronPtScalParam(object nodes, object param_name, float val):
    "Set neuron list scalar parameter value"
    n_node = len(nodes)
    array = numpy.array(nodes, dtype=numpy.int32, copy=True, order='C')
    ret = NESTGPU_SetNeuronPtScalParam(np_int_array_to_pointer(array),
                                         n_node, param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronPtArrayParam(object nodes, object param_name, object param_list):
    "Set neuron list array parameter value"
    n_node = len(nodes)
    node_array = numpy.array(nodes, dtype=numpy.int32, copy=True, order='C')

    array_size = len(param_list)
    param_array = numpy.array(param_list, dtype=numpy.float32, copy=True, order='C')
    ret = NESTGPU_SetNeuronPtArrayParam(np_int_array_to_pointer(node_array),
                                          n_node, param_name.encode('utf-8'),
                                          np_float_array_to_pointer(param_array),
                                          array_size)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_isNeuronScalParam(int i_node, object param_name):
    "Check name of neuron scalar parameter"
    ret = (NESTGPU_IsNeuronScalParam(i_node, param_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_isNeuronPortParam(int i_node, object param_name):
    "Check name of neuron scalar parameter"
    ret = (NESTGPU_IsNeuronPortParam(i_node, param_name.encode('utf-8'))!= 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_isNeuronArrayParam(int i_node, object param_name):
    "Check name of neuron scalar parameter"
    ret = (NESTGPU_IsNeuronArrayParam(i_node, param_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_isNeuronGroupParam(int i_node, object param_name):
    "Check name of neuron scalar parameter"
    ret = (NESTGPU_IsNeuronGroupParam(i_node, param_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronIntVar(int i_node, int n_node, object var_name, int val):
    "Set neuron integer variable value"
    ret = NESTGPU_SetNeuronIntVar(i_node, n_node, var_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronScalVar(int i_node, int n_node, object var_name, float val):
    "Set neuron scalar variable value"
    ret = NESTGPU_SetNeuronScalVar(i_node, n_node, var_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronArrayVar(int i_node, int n_node, object var_name, object var_list):
    "Set neuron array variable value"
    array_size = len(var_list)
    array = numpy.array(var_list, dtype=numpy.float32, copy=True, order='C')
    ret = NESTGPU_SetNeuronArrayVar(i_node, n_node, var_name.encode('utf-8'),
                                       np_float_array_to_pointer(array),
                                       array_size)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronPtIntVar(object nodes, object var_name, int val):
    "Set neuron list integer variable value"
    n_node = len(nodes)
    array = numpy.array(nodes, dtype=numpy.int32, copy=True, order='C')
    ret = NESTGPU_SetNeuronPtIntVar(np_int_array_to_pointer(array),
                                       n_node, var_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronPtScalVar(object nodes, object var_name, float val):
    "Set neuron list scalar variable value"
    n_node = len(nodes)
    array = numpy.array(nodes, dtype=numpy.int32, copy=True, order='C')
    ret = NESTGPU_SetNeuronPtScalVar(np_int_array_to_pointer(array),
                                       n_node, var_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronPtArrayVar(object nodes, object var_name, object var_list):
    "Set neuron list array variable value"
    n_node = len(nodes)
    node_array = numpy.array(nodes, dtype=numpy.int32, copy=True, order='C')

    array_size = len(var_list)
    var_array = numpy.array(var_list, dtype=numpy.float32, copy=True, order='C')
    ret = NESTGPU_SetNeuronPtArrayVar(np_int_array_to_pointer(node_array),
                                        n_node, var_name.encode('utf-8'),
                                        np_float_array_to_pointer(var_array),
                                        array_size)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_isNeuronIntVar(int i_node, object var_name):
    "Check name of neuron integer variable"
    ret = (NESTGPU_IsNeuronIntVar(i_node, var_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_isNeuronScalVar(int i_node, object var_name):
    "Check name of neuron scalar variable"
    ret = (NESTGPU_IsNeuronScalVar(i_node, var_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_isNeuronPortVar(int i_node, object var_name):
    "Check name of neuron scalar variable"
    ret = (NESTGPU_IsNeuronPortVar(i_node, var_name.encode('utf-8'))!= 0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_isNeuronArrayVar(int i_node, object var_name):
    "Check name of neuron array variable"
    ret = (NESTGPU_IsNeuronArrayVar(i_node, var_name.encode('utf-8'))!=0)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_setNeuronGroupParam(int i_node, int n_node, object param_name, float val):
    ret = NESTGPU_SetNeuronGroupParam(i_node, n_node, param_name.encode('utf-8'), val)
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

def llapi_calibrate():
    "Calibrate simulation"
    ret = NESTGPU_Calibrate()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

