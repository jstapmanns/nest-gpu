import numpy

cdef int* np_int_array_to_pointer(object array):
    if not isinstance(array, numpy.ndarray):
        raise TypeError('array must be a 1-dimensional NumPy array of ints, got {}'.format(type(array)))
    if not array.ndim == 1:
        raise TypeError('array must be a 1-dimensional NumPy array, got {}-dimensional NumPy array'.format(array.ndim))

    # Get pointers to the first element in the Numpy array
    cdef int[:] array_mv
    cdef int* array_ptr

    if numpy.issubdtype(array.dtype, numpy.integer):
        array_mv = numpy.ascontiguousarray(array, dtype=numpy.int)
        array_ptr = &array_mv[0]
    else:
        raise TypeError('array must be a NumPy array of ints, got {}'.format(array.dtype))

    return array_ptr

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
    array = numpy.array(i_target, dtype=int, copy=True, order='C')
    return NESTGPU_ConnectSeqGroup(i_source, n_source, np_int_array_to_pointer(array), n_target)

def llapi_connectGroupSeq(object i_source, int n_source, int i_target, int n_target):
    print('using cython llapi_connectGroupSeq()')
    array = numpy.array(i_source, dtype=int, copy=True, order='C')
    return NESTGPU_ConnectGroupSeq(np_int_array_to_pointer(array), n_source, i_target, n_target)

def llapi_connectGroupGroup(object i_source, int n_source, object i_target, int n_target):
    print('using cython llapi_connectGroupGroup()')
    source_array = numpy.array(i_source, dtype=int, copy=True, order='C')
    target_array = numpy.array(i_target, dtype=int, copy=True, order='C')
    return NESTGPU_ConnectGroupGroup(np_int_array_to_pointer(source_array), n_source,
            np_int_array_to_pointer(target_array), n_target)

def llapi_create(model, long n, int n_port):
    print('using cython llapi_create() to create {} {}(s)'.format(n, model))
    return NESTGPU_Create(model.encode('utf-8'), n, n_port)

# TODO: should we change all the char* arguments to string as it is done in NEST?
