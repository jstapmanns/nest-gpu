import numpy
from cython.operator cimport dereference as deref
from cython cimport view
from libc.string cimport strlen, memcpy
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

'''
class definitions
'''
class ConnectionId(object):
    def __init__(self, i_source, i_group, i_conn):
        self.i_source = i_source
        self.i_group = i_group
        self.i_conn = i_conn

class SynGroup(object):
    def __init__(self, i_syn_group):
        self.i_syn_group = i_syn_group

class NodeSeq(object):
    def __init__(self, i0, n=1):
        if i0 == None:
            i0 = 0
            n = -1
        self.i0 = i0
        self.n = n

    def Subseq(self, first, last):
        if last<0 and last>=-self.n:
            last = last%self.n
        if first<0 | last<first:
            raise ValueError("Sequence subset range error")
        if last>=self.n:
            raise ValueError("Sequence subset out of range")
        return NodeSeq(self.i0 + first, last - first + 1)
    def __getitem__(self, i):
        if type(i)==slice:
            if i.step != None:
                raise ValueError("Subsequence cannot have a step")
            return self.Subseq(i.start, i.stop-1)

        if i<-self.n:
            raise ValueError("Sequence index error")
        if i>=self.n:
            raise ValueError("Sequence index out of range")
        if i<0:
            i = i%self.n
        return self.i0 + i
    def ToList(self):
        return list(range(self.i0, self.i0 + self.n))
    def __len__(self):
        return self.n

class RemoteNodeSeq(object):
    def __init__(self, i_host=0, node_seq=NodeSeq(None)):
        self.i_host = i_host
        self.node_seq = node_seq

'''
helping functions
'''
cdef vector[int] pylist_to_int_vec(object pylist):
    array = numpy.array(pylist, dtype=numpy.int32, copy=True, order='C')
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a 1-dimensional NumPy array of ints, got {}'.format(
            type(array)))
    if not array.ndim == 1:
        raise TypeError('array must be a 1-dimensional NumPy array, got {}-dimensional NumPy array'.format(
            array.ndim))
    if not numpy.issubdtype(array.dtype, numpy.int32):
        raise TypeError('array must be a NumPy array of ints, got {}'.format(array.dtype))

    cdef vector[int] vec
    vec = array
    return vec

cdef int* np_int_array_to_pointer(object array):
    # TODO: implement check to ensure that array entries are not larger than int32
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

cdef vector[float] pylist_to_float_vec(object pylist):
    array = numpy.array(pylist, dtype=numpy.float32, copy=True, order='C')
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a 1-dimensional NumPy array of ints, got {}'.format(
            type(array)))
    if not array.ndim == 1:
        raise TypeError('array must be a 1-dimensional NumPy array, got {}-dimensional NumPy array'.format(
            array.ndim))
    if not numpy.issubdtype(array.dtype, numpy.float32):
        raise TypeError('array must be a NumPy array of floats, got {}'.format(array.dtype))

    cdef vector[float] vec
    vec = array
    return vec

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

    #cdef vector[float] c_array
    #for val in array:
    #    c_array.push_back(val)
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


