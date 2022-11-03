from libcpp.string cimport string
from libcpp.vector cimport vector

#from nestgpukernel_api cimport NESTGPU_Connect
#from nestgpukernel_api cimport NESTGPU_Create

def llapi_connect(int i_source_node, int i_target_node,
			unsigned char port, unsigned char syn_group,
            float weight, float delay):
    print('using cython llapi_connect()')
    NESTGPU_Connect(i_source_node, i_target_node,port, syn_group,
            weight, delay)

def llapi_create(model, long n, int n_port):
    print('using cython llapi_create() to create {} {}(s)'.format(n, model))
    NESTGPU_Create(model.encode('utf-8'), n, n_port)

# TODO: should we change all the char* arguments to string as it is done in NEST?
