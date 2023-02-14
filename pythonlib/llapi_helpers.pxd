from libcpp.vector cimport vector
from libc.stdint cimport int64_t
cdef int* np_int_array_to_pointer(object array)
cdef int64_t* np_int64_array_to_pointer(object array)
cdef float* np_float_array_to_pointer(object array)
cdef vector[int] pylist_to_int_vec(object pylist)
cdef vector[float] pylist_to_float_vec(object pylist)
cdef long* np_long_array_to_pointer(object array)
cdef object int_array_to_pylist(int* first, int n_elements)
cdef object int_array_to_conn_list(int64_t* first, int64_t n_conn)
cdef object float_array2d_to_numpy2d(float** first_p, int n_row, int n_col)
cdef object cstring_to_pystring(char* cstr)
cdef char** pystring_list_to_cstring_array(object pystring_list)
