""" Python interface for NESTGPU"""
import sys, platform
import os
import unicodedata
import gc
import nestgpukernel_api as ng_kernel
from llapi_helpers import SynGroup, NodeSeq, RemoteNodeSeq, ConnectionList


print('\n              -- NEST GPU --\n')
print('  Copyright (C) 2004 The NEST Initiative\n')
print(' This program is provided AS IS and comes with')
print(' NO WARRANTY. See the file LICENSE for details.\n')
print(' Homepage: https://github.com/nest/nest-gpu')
print()


ng_kernel.llapi_setOnException(1)

def waitenter(val):
    if (sys.version_info >= (3, 0)):
        return input(val)
    else:
        return raw_input(val)

def SetRandomSeed(seed):
    "Set seed for random number generation"
    ret = ng_kernel.llapi_setRandomSeed(seed)
    return ret

def SetTimeResolution(time_res):
    "Set time resolution in ms"
    ret = ng_kernel.llapi_setTimeResolution(time_res)
    return ret

def GetTimeResolution():
    "Get time resolution in ms"
    ret = ng_kernel.llapi_getTimeResolution()
    return ret

def SetMaxSpikeBufferSize(max_size):
    "Set maximum size of spike buffer per node"
    ret = ng_kernel.llapi_setMaxSpikeBufferSize(max_size)
    return ret

def GetMaxSpikeBufferSize():
    "Get maximum size of spike buffer per node"
    ret = ng_kernel.llapi_getMaxSpikeBufferSize()
    return ret

def SetSimTime(sim_time):
    "Set neural activity simulated time in ms"
    ret = ng_kernel.llapi_setSimTime(sim_time)
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return ret

def SetVerbosityLevel(verbosity_level):
    "Set verbosity level"
    ret = ng_kernel.llapi_setVerbosityLevel(verbosity_level)
    return ret

def Create(model_name, n_node=1, n_ports=1, status_dict=None):
    "Create a neuron group"
    if (type(status_dict)==dict):
        node_group = Create(model_name, n_node, n_ports)
        SetStatus(node_group, status_dict)
        return node_group

    elif status_dict!=None:
        raise ValueError("Wrong argument in Create")

    i_node = ng_kernel.llapi_create(model_name, n_node, n_ports)
    ret = NodeSeq(i_node, n_node)
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return ret

def SetNestedLoopAlgo(nested_loop_algo):
    "Set CUDA nested loop algorithm"
    ret = ng_kernel.llapi_setNestedLoopAlgo(nested_loop_algo)
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return ret

def CreateRecord(file_name, var_name_list, i_node_list, i_port_list):
    "Create a record of neuron variables"
    n_node = len(i_node_list)
    ret = ng_kernel.llapi_createRecord(file_name, var_name_list, i_node_list, i_port_list, n_node)
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return ret

def GetRecordData(i_record):
    "Get record data"
    ret = ng_kernel.llapi_getRecordData(i_record)
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return ret

def Calibrate():
    "Calibrate simulation"
    ret = ng_kernel.llapi_calibrate()
    return ret


def Simulate(sim_time=1000.0):
    "Simulate neural activity"
    SetSimTime(sim_time)
    ret = ng_kernel.llapi_simulate()
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return ret


def Connect(source, target, conn_dict, syn_dict):
    "Connect two node groups"
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")

    gc.disable()
    ng_kernel.llapi_connSpecInit()
    ng_kernel.llapi_synSpecInit()
    for param_name in conn_dict:
        if param_name=="rule":
            for i_rule in range(len(ng_kernel.conn_rule_name)):
                if conn_dict[param_name]==ng_kernel.conn_rule_name[i_rule]:
                    break
            if i_rule < len(ng_kernel.conn_rule_name):
                ng_kernel.llapi_setConnSpecParam(param_name, i_rule)
            else:
                raise ValueError("Unknown connection rule")
        elif ng_kernel.llapi_connSpecIsParam(param_name):
            ng_kernel.llapi_setConnSpecParam(param_name, conn_dict[param_name])
        else:
            raise ValueError("Unknown connection parameter")

    array_size = ng_kernel.llapi_ruleArraySize(conn_dict, source, target)

    for param_name in syn_dict:
        if ng_kernel.llapi_synSpecIsIntParam(param_name):
            val = syn_dict[param_name]
            if ((param_name=="synapse_group") & (type(val)==SynGroup)):
                val = val.i_syn_group
            ng_kernel.llapi_setSynSpecIntParam(param_name, val)
        elif ng_kernel.llapi_synSpecIsFloatParam(param_name):
            fpar = syn_dict[param_name]
            if (type(fpar)==dict):
                for dict_param_name in fpar:
                    pval = fpar[dict_param_name]
                    if dict_param_name=="array":
                        arr = pval
                        arr_param_name = param_name + "_array"
                        if (not ng_kernel.llapi_synSpecIsFloatPtParam(arr_param_name)):
                            raise ValueError("Synapse parameter cannot be set"
                                             " by arrays")

                        ng_kernel.llapi_setSynSpecFloatPtParam(arr_param_name, arr)
                    elif dict_param_name=="distribution":
                        distr_idx = ng_kernel.distribution_dict[pval]
                        distr_param_name = param_name + "_distribution"
                        if (not ng_kernel.llapi_synSpecIsIntParam(distr_param_name)):
                            raise ValueError("Synapse parameter cannot be set"
                                             " by distributions")

                        ng_kernel.llapi_setSynSpecIntParam(distr_param_name, distr_idx)
                    else:
                        param_name2 = param_name + "_" + dict_param_name
                        if ng_kernel.llapi_synSpecIsFloatParam(param_name2):
                            ng_kernel.llapi_setSynSpecFloatParam(param_name2, pval)
                        else:
                            print(param_name2)
                            raise ValueError("Unknown distribution parameter")

            else:
                ng_kernel.llapi_setSynSpecFloatParam(param_name, fpar)

        elif ng_kernel.llapi_synSpecIsFloatPtParam(param_name):
            ng_kernel.llapi_setSynSpecFloatPtParam(param_name, syn_dict[param_name])
        else:
            raise ValueError("Unknown synapse parameter")
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        ret = ng_kernel.llapi_connectSeqSeq(source.i0, source.n, target.i0, target.n)
    else:
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            ret = ng_kernel.llapi_connectSeqGroup(source.i0, source.n, target,
                                            len(target))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            ret = ng_kernel.llapi_connectGroupSeq(source, len(source),
                                            target.i0, target.n)
        else:
            ret = ng_kernel.llapi_connectGroupGroup(source, len(source),
                                              target, len(target))
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    gc.enable()
    return ret


def SetStatus(gen_object, params, val=None):
    "Set neuron or synapse group parameters or variables using dictionaries"

    if type(gen_object)==RemoteNodeSeq:
        if gen_object.i_host==ng_kernel.llapi_mpiId():
            SetStatus(gen_object.node_seq, params, val)
        return

    gc.disable()
    if type(gen_object)==SynGroup:
        ret = ng_kernel.llapi_setSynGroupStatus(gen_object, params, val)
        gc.enable()
        return ret
    nodes = gen_object
    if val != None:
         ng_kernel.llapi_setNeuronStatus(nodes, params, val)
    elif type(params)==dict:
        for param_name in params:
            ng_kernel.llapi_setNeuronStatus(nodes, param_name, params[param_name])
    elif (type(params)==list)  | (type(params) is tuple):
        if len(params) != len(nodes):
            raise ValueError("List should have the same size as nodes")
        for param_dict in params:
            if type(param_dict)!=dict:
                raise ValueError("Type of list elements should be dict")
            for param_name in param_dict:
                ng_kernel.llapi_setNeuronStatus(nodes, param_name, param_dict[param_name])
    else:
        raise ValueError("Wrong argument in SetStatus")
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    gc.enable()

# TODO: this function is usually called only somewhere within SetStatus
#       only one of the tests is using it directly. I think, SetStatus()
#       should be used instead.
def SetSynGroupParam(syn_group, param_name, val):
    return ng_kernel.llapi_setSynGroupParam(syn_group, param_name, val)

def GetConnections(source=None, target=None, syn_group=-1):
    "Get connections between two node groups"
    if source==None:
        source = NodeSeq(None)
    if target==None:
        target = NodeSeq(None)
    if (type(source)==int):
        source = [source]
    if (type(target)==int):
        target = [target]
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")

    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        ret = ng_kernel.llapi_getSeqSeqConnections(source.i0, source.n,
                                                  target.i0, target.n,
                                                  syn_group)
    else:
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            ret = ng_kernel.llapi_getSeqGroupConnections(source.i0, source.n,
                                                        target,
                                                        len(target),
                                                        syn_group)
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            ret = ng_kernel.llapi_getGroupSeqConnections(source, len(source),
                                                        target.i0, target.n,
                                                        syn_group)
        else:
            ret = ng_kernel.llapi_getGroupGroupConnections(source, len(source),
                                                        target, len(target),
                                                        syn_group)

    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return ret


def SetNeuronGroupParam(nodes, param_name, val):
    "Set neuron group parameter value"
    if type(nodes)!=NodeSeq:
        raise ValueError("Wrong argument type in SetNeuronGroupParam")

    ret = ng_kernel.llapi_setNeuronGroupParam(nodes.i0, nodes.n, param_name.encode('utf-8'), val)
    return ret


def GetKernelStatus(var_key=None):
    "Get kernel status"
    if (type(var_key)==list) | (type(var_key)==tuple):
        status_list = []
        for var_elem in var_key:
            var_value = GetKernelStatus(var_elem)
            status_list.append(var_value)
        return status_list
    elif (var_key==None):
        status_dict = {}
        name_list = ng_kernel.llapi_getFloatParamNames() + ng_kernel.llapi_getIntParamNames() + ng_kernel.llapi_getBoolParamNames()
        for param_name in name_list:
            val = GetKernelStatus(param_name)
            status_dict[param_name] = val
        return status_dict
    elif (type(var_key)==str) | (type(var_key)==bytes):
        if ng_kernel.llapi_isFloatParam(var_key):
            return ng_kernel.llapi_getFloatParam(var_key)
        elif ng_kernel.llapi_isIntParam(var_key):
            return ng_kernel.llapi_getIntParam(var_key)
        elif ng_kernel.llapi_isBoolParam(var_key):
            return ng_kernel.llapi_getBoolParam(var_key)
        else:
            raise ValueError("Unknown parameter in GetKernelStatus", var_key)
    else:
        raise ValueError("Unknown key type in GetKernelStatus", type(var_key))

def SetKernelStatus(params, val=None):
    "Set kernel parameters using dictionaries"
    if ((type(params)==dict) & (val==None)):
        for param_name in params:
            SetKernelStatus(param_name, params[param_name])
    elif (type(params)==str):
        if ng_kernel.llapi_isFloatParam(params):
            return ng_kernel.llapi_setFloatParam(params, val)
        elif ng_kernel.llapi_isIntParam(params):
            return ng_kernel.llapi_setIntParam(params, val)
        elif ng_kernel.llapi_isBoolParam(params):
            return ng_kernel.llapi_setBoolParam(params, val)
        else:
            raise ValueError("Unknown parameter in SetKernelStatus", params)
    else:
        raise ValueError("Wrong argument in SetKernelStatus")       
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())

def ConnectMpiInit():
    "Initialize MPI connections"
    from mpi4py import MPI
    argc=len(sys.argv)
    ret = ng_kernel.llapi_connectMpiInit(argc, sys.argv)
    return ret

def MpiNp():
    "Get MPI Np"
    ret = ng_kernel.llapi_mpiNp()
    return ret

def MpiId():
    "Get MPI Id"
    ret = ng_kernel.llapi_mpiId()
    return ret

def Rank():
    "Get MPI rank"
    return MpiId()

def RemoteConnect(i_source_host, source, i_target_host, target,
                  conn_dict, syn_dict):
    "Connect two node groups of differen mpi hosts"
    if (type(i_source_host)!=int) | (type(i_target_host)!=int):
        raise ValueError("Error in host index")
    if (type(source)!=list) & (type(source)!=tuple) & (type(source)!=NodeSeq):
        raise ValueError("Unknown source type")
    if (type(target)!=list) & (type(target)!=tuple) & (type(target)!=NodeSeq):
        raise ValueError("Unknown target type")

    ng_kernel.llapi_connSpecInit()
    ng_kernel.llapi_synSpecInit()
    for param_name in conn_dict:
        if param_name=="rule":
            for i_rule in range(len(ng_kernel.conn_rule_name)):
                if conn_dict[param_name]==ng_kernel.conn_rule_name[i_rule]:
                    break
            if i_rule < len(ng_kernel.conn_rule_name):
                ng_kernel.llapi_setConnSpecParam(param_name, i_rule)
            else:
                raise ValueError("Unknown connection rule")

        elif ng_kernel.llapi_connSpecIsParam(param_name):
            ng_kernel.llapi_setConnSpecParam(param_name, conn_dict[param_name])
        else:
            raise ValueError("Unknown connection parameter")

    array_size = ng_kernel.llapi_ruleArraySize(conn_dict, source, target)

    for param_name in syn_dict:
        if ng_kernel.llapi_synSpecIsIntParam(param_name):
            ng_kernel.llapi_setSynSpecIntParam(param_name, syn_dict[param_name])
        elif ng_kernel.llapi_synSpecIsFloatParam(param_name):
            fpar = syn_dict[param_name]
            if (type(fpar)==dict):
                ng_kernel.llapi_setSynParamFromArray(param_name, fpar, array_size)
            else:
                ng_kernel.llapi_setSynSpecFloatParam(param_name, fpar)

        elif ng_kernel.llapi_synSpecIsFloatPtParam(param_name):
            ng_kernel.llapi_setSynSpecFloatPtParam(param_name, syn_dict[param_name])
        else:
            raise ValueError("Unknown synapse parameter")
    if (type(source)==NodeSeq) & (type(target)==NodeSeq) :
        ret = ng_kernel.llapi_remoteConnectSeqSeq(i_source_host, source.i0, source.n,
                                            i_target_host, target.i0, target.n)

    else:
        if (type(source)==NodeSeq) & (type(target)!=NodeSeq):
            ret = ng_kernel.llapi_remoteConnectSeqGroup(i_source_host, source.i0,
                                                  source.n, i_target_host,
                                                  target, len(target))
        elif (type(source)!=NodeSeq) & (type(target)==NodeSeq):
            ret = ng_kernel.llapi_remoteConnectGroupSeq(i_source_host, source,
                                                  len(source),
                                                  i_target_host, target.i0,
                                                  target.n)
        else:
            ret = ng_kernel.llapi_remoteConnectGroupGroup(i_source_host,
                                                    source,
                                                    len(source),
                                                    i_target_host,
                                                    target,
                                                    len(target))
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return ret

def ActivateSpikeCount(nodes):
    "Activate spike count for node group"
    if type(nodes)!=NodeSeq:
        raise ValueError("Argument type of ActivateSpikeCount must be NodeSeq")

    ret = ng_kernel.llapi_activateSpikeCount(nodes.i0, nodes.n)
    return ret

def ActivateRecSpikeTimes(nodes, max_n_rec_spike_times):
    "Activate spike time recording for node group"
    if type(nodes)!=NodeSeq:
        raise ValueError("Argument type of ActivateRecSpikeTimes must be NodeSeq")

    ret = ng_kernel.llapi_activateRecSpikeTimes(nodes.i0, nodes.n,
                                          max_n_rec_spike_times)
    return ret

def SetRecSpikeTimesStep(nodes, rec_spike_times_step):
    "Setp number of time steps for buffering spike time recording"
    if type(nodes)!=NodeSeq:
        raise ValueError("Argument type of SetRecSpikeTimesStep must be NodeSeq")

    ret = ng_kernel.llapi_setRecSpikeTimesStep(nodes.i0, nodes.n,
                                       rec_spike_times_step)
    return ret

def GetStatus(gen_object, var_key=None):
    "Get neuron group, connection or synapse group status"
    if type(gen_object)==SynGroup:
        return ng_kernel.llapi_getSynGroupStatus(gen_object, var_key)
    elif type(gen_object)==NodeSeq:
        gen_object = gen_object.ToList()
    if (type(gen_object)==list) | (type(gen_object)==tuple):
        status_list = []
        for gen_elem in gen_object:
            elem_dict = GetStatus(gen_elem, var_key)
            status_list.append(elem_dict)
        return status_list
    if (type(var_key)==list) | (type(var_key)==tuple):
        status_list = []
        for var_elem in var_key:
            var_value = GetStatus(gen_object, var_elem)
            status_list.append(var_value)
        return status_list
    elif (var_key==None):
        if (type(gen_object)==ConnectionList):
            status_dict = GetConnectionStatus(gen_object)
        elif (type(gen_object)==int):
            i_node = gen_object
            status_dict = {}
            name_list = ng_kernel.llapi_getIntVarNames(i_node) \
                        + ng_kernel.llapi_getScalVarNames(i_node) \
                        + ng_kernel.llapi_getScalParamNames(i_node) \
                        + ng_kernel.llapi_getPortVarNames(i_node) \
                        + ng_kernel.llapi_getPortParamNames(i_node) \
                        + ng_kernel.llapi_getArrayVarNames(i_node) \
                        + ng_kernel.llapi_getArrayParamNames(i_node) \
                        + ng_kernel.llapi_getGroupParamNames(i_node)
            for var_name in name_list:
                val = GetStatus(i_node, var_name)
                status_dict[var_name] = val
        else:
            raise ValueError("Unknown object type in GetStatus")
        return status_dict
    elif (type(var_key)==str) | (type(var_key)==bytes):
        if (type(gen_object)==ConnectionList):
            if ng_kernel.llapi_isConnectionFloatParam(var_key):
                return ng_kernel.llapi_getConnectionFloatParam(gen_object, var_key)
            elif ng_kernel.llapi_isConnectionIntParam(var_key):
                return ng_kernel.llapi_getConnectionIntParam(gen_object, var_key)
            else:
                raise ValueError("Unknown connection parameter in GetStatus")
        elif (type(gen_object)==int):
            i_node = gen_object
            ret = ng_kernel.llapi_getNeuronStatus([i_node], var_key)[0]
            return ret
        else:
            raise ValueError("Unknown object type in GetStatus")

    else:
        raise ValueError("Unknown key type in GetStatus", type(var_key))

'''
def GetConnectionStatus(conn_id):
    i_source = conn_id.i_source
    i_group = conn_id.i_group
    i_conn = conn_id.i_conn

    conn_status_dict = ng_kernel.llapi_getConnectionStatus(i_source, i_group, i_conn)

    conn_status_dict['source'] = i_source
    return conn_status_dict
'''

# In general, this function is included in GetStatus()
# but some tests use it from outside, so we need a wrapper
# TODO: Shouldn't we replace all GetNeuronStatus() by GetStatus()?
def GetNeuronStatus(nodes, var_name):
    return ng_kernel.llapi_getNeuronStatus(nodes, var_name)

def GetConnectionStatus(conn):
    "Get all parameters of connection list conn"
    if (type(conn)==ConnectionList):
        conn = conn.conn_list
    elif (type(conn)==int):
        conn = [conn]
    if ((type(conn)!=list) and (type(conn)!=tuple)):
        raise ValueError("GetConnectionStatus argument type must be "
                         "ConnectionList, int, list or tuple")
    status_list = ng_kernel.llapi_getConnectionStatus(conn)

    return status_list

def GetRecSpikeTimes(nodes):
    if type(nodes)!=NodeSeq:
        raise ValueError("First argument type of GetRecSpikeTimes must be NodeSeq")

    ret = ng_kernel.llapi_getRecSpikeTimes(nodes.i0, nodes.n)
    return ret

def RemoteCreate(i_host, model_name, n_node=1, n_ports=1, status_dict=None):
    "Create a remote neuron group"
    if (type(status_dict)==dict):
        remote_node_group = RemoteCreate(i_host, model_name, n_node, n_ports)
        SetStatus(remote_node_group, status_dict)
        return remote_node_group

    elif status_dict!=None:
        raise ValueError("Wrong argument in RemoteCreate")

    i_node = ng_kernel.llapi_remoteCreate(i_host, model_name, n_node, n_ports)
    node_seq = NodeSeq(i_node, n_node)
    ret = RemoteNodeSeq(i_host, node_seq)
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return ret

def CreateSynGroup(model_name, status_dict=None):
    "Create a synapse group"
    if (type(status_dict)==dict):
        syn_group = CreateSynGroup(model_name)
        SetStatus(syn_group, status_dict)
        return syn_group
    elif status_dict!=None:
        raise ValueError("Wrong argument in CreateSynGroup")

    i_syn_group = ng_kernel.llapi_createSynGroup(model_name)
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    return SynGroup(i_syn_group)

