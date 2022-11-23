import numpy
from cython.operator cimport dereference as deref
from cython cimport view
from libc.string cimport strlen

class ConnectionId(object):
    def __init__(self, i_source, i_group, i_conn):
        self.i_source = i_source
        self.i_group = i_group
        self.i_conn = i_conn

'''
helping functions
'''
cdef int* np_int_array_to_pointer(object array):
    '''
    function to get a pointer to the first element of a numpy array of dtype np.int32
    returns int* to the first element of array
    '''
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a 1-dimensional NumPy array of ints, got {}'.format(type(array)))
    if not array.ndim == 1:
        raise TypeError('array must be a 1-dimensional NumPy array, got {}-dimensional NumPy array'.format(array.ndim))

    # Get pointers to the first element in the Numpy array
    cdef int[:] array_int_mv
    cdef int* array_int_ptr

    if numpy.issubdtype(array.dtype, numpy.integer):
        array_int_mv = numpy.ascontiguousarray(array, dtype=numpy.int32)
        # TODO: the following line could be dangerous if sizeof(int) != 8.
        array_int_ptr = < int* > &array_int_mv[0]
    else:
        raise TypeError('array must be a NumPy array of ints, got {}'.format(array.dtype))

    return array_int_ptr

cdef long* np_long_array_to_pointer(object array):
    '''
    function to get a pointer to the first element of a numpy array of dtype np.int64
    returns long* to the first element of array
    '''
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a 1-dimensional NumPy array of ints, got {}'.format(type(array)))
    if not array.ndim == 1:
        raise TypeError('array must be a 1-dimensional NumPy array, got {}-dimensional NumPy array'.format(array.ndim))

    # Get pointers to the first element in the Numpy array
    cdef long[:] array_long_mv
    cdef long* array_long_ptr

    if numpy.issubdtype(array.dtype, numpy.integer):
        array_long_mv = numpy.ascontiguousarray(array, dtype=numpy.int64)
        array_long_ptr = &array_long_mv[0]
    else:
        raise TypeError('array must be a NumPy array of ints, got {}'.format(array.dtype))

    return array_long_ptr

cdef int_array_to_pylist(int* first, int n_elements):
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

cdef object cstring_to_pystring(char* cstr):
    np_byte_array = numpy.asarray(<char[:strlen(cstr)]>(cstr))
    ret = ''.join(numpy.char.decode(np_byte_array, encoding='utf-8'))
    return ret

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

'''
low level api
'''
def llapi_mpiId():
    "Get MPI Id"
    ret = NESTGPU_MpiId()
    if llapi_getErrorCode() != 0:
        raise ValueError(llapi_getErrorMessage())
    return ret

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
# TODO: can we make use of boost?

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

