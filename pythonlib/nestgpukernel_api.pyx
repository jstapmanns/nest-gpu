import numpy
from cython.operator cimport dereference as deref
from cython cimport view

class ConnectionId(object):
    def __init__(self, i_source, i_group, i_conn):
        self.i_source = i_source
        self.i_group = i_group
        self.i_conn = i_conn

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
    print('using cython llapi_getSeqSeqConnections()')
    cdef int* n_conn
    cdef int* ret
    ret = NESTGPU_GetSeqSeqConnections(i_source, n_source, i_target, n_target, syn_group, n_conn)
    return 0

def llapi_getSeqGroupConnections(int i_source, int n_source, object i_target,
        int n_target, int syn_group):
    print('using cython llapi_getSeqGroupConnections()')
    target_array = numpy.array(i_target, dtype=numpy.int32, copy=True, order='C')
    cdef int* n_conn
    cdef int* ret
    ret = NESTGPU_GetSeqGroupConnections(i_source, n_source, np_int_array_to_pointer(target_array),
            n_target, syn_group, n_conn)
    return 0

def llapi_getGroupSeqConnections(object i_source, int n_source, int i_target,
        int n_target, int syn_group):
    print('using cython llapi_getGroupSeqConnections()')
    source_array = numpy.array(i_source, dtype=numpy.int32, copy=True, order='C')
    cdef int n_conn
    cdef int* c_ret
    c_ret = NESTGPU_GetGroupSeqConnections(np_int_array_to_pointer(source_array), n_source, i_target,
            n_target, syn_group, &n_conn)
    print('llapi: number of connections: {}'.format(n_conn))
    conn_arr = numpy.asarray(<int[:3*n_conn]>c_ret)
    #print('llapi: content of array: {}'.format(int_array_to_pylist(ret, 3*n_conn)))
    #print('llapi: content of array: {}'.format(conn_arr))
    conn_list = []
    for i_conn in range(n_conn):
        conn_id = ConnectionId(conn_arr[i_conn*3], conn_arr[i_conn*3 + 1],
                   conn_arr[i_conn*3 + 2])
        conn_list.append(conn_id)

    ret = conn_list
    print('llpai: length of conn_list: {}'.format(len(conn_list)))

    #if GetErrorCode() != 0:
    #    raise ValueError(GetErrorMessage())
    return ret

def llapi_getGroupGroupConnections(object i_source, int n_source, object i_target,
        int n_target, int syn_group):
    print('using cython llapi_getGroupGroupConnections()')
    source_array = numpy.array(i_source, dtype=numpy.int32, copy=True, order='C')
    target_array = numpy.array(i_target, dtype=numpy.int32, copy=True, order='C')
    cdef int* n_conn
    cdef int* ret
    ret = NESTGPU_GetGroupGroupConnections(np_int_array_to_pointer(source_array), n_source,
            np_int_array_to_pointer(target_array), n_target, syn_group, n_conn)
    return 0

