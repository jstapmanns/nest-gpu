""" Python interface for NESTGPU"""
import sys, platform
import os
import unicodedata
import gc
import nestgpukernel_api as ng_kernel
from llapi_helpers import SynGroup, NodeSeq, RemoteNodeSeq, ConnectionId


print('\n              -- NEST GPU --\n')
print('  Copyright (C) 2004 The NEST Initiative\n')
print(' This program is provided AS IS and comes with')
print(' NO WARRANTY. See the file LICENSE for details.\n')
print(' Homepage: https://github.com/nest/nest-gpu')
print()


conn_rule_name = ("one_to_one", "all_to_all", "fixed_total_number",
                  "fixed_indegree", "fixed_outdegree")

ng_kernel.llapi_setOnException(1)

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


def GetNeuronStatus(nodes, var_name):
    "Get neuron group scalar or array variable or parameter"
    print('nestgpu.py GetNeuronStatus()')
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    if type(nodes)==NodeSeq:
        print('GetNeuronStatus() nodes is NodeSeq')
        if (ng_kernel.llapi_isNeuronScalParam(nodes.i0, var_name) |
            ng_kernel.llapi_isNeuronPortParam(nodes.i0, var_name)):
            ret = ng_kernel.llapi_getNeuronParam(nodes.i0, nodes.n, var_name)
        elif ng_kernel.llapi_isNeuronArrayParam(nodes.i0, var_name):
            ret = ng_kernel.llapi_getArrayParam(nodes.i0, nodes.n, var_name)
        elif (ng_kernel.llapi_isNeuronIntVar(nodes.i0, var_name)):
            ret = ng_kernel.llapi_getNeuronIntVar(nodes.i0, nodes.n, var_name)
        elif (ng_kernel.llapi_isNeuronScalVar(nodes.i0, var_name) |
              ng_kernel.llapi_isNeuronPortVar(nodes.i0, var_name)):
            ret = ng_kernel.llapi_getNeuronVar(nodes.i0, nodes.n, var_name)
        elif ng_kernel.llapi_isNeuronArrayVar(nodes.i0, var_name):
            ret = ng_kernel.llapi_getArrayVar(nodes.i0, nodes.n, var_name)
        elif ng_kernel.llapi_isNeuronGroupParam(nodes.i0, var_name):
            ret = ng_kernel.llapi_getNeuronStatus(nodes.ToList(), var_name)
        else:
            raise ValueError("Unknown neuron variable or parameter")
    else:
        print('GetNeuronStatus() nodes is not a NodeSeq')
        if (ng_kernel.llapi_isNeuronScalParam(nodes[0], var_name) |
            ng_kernel.llapi_isNeuronPortParam(nodes[0], var_name)):
            print('case 1')
            ret = ng_kernel.llapi_getNeuronPtParam(nodes, var_name)
        elif ng_kernel.llapi_isNeuronArrayParam(nodes[0], var_name):
            print('case 2')
            ret = ng_kernel.llapi_getNeuronListArrayParam(nodes, var_name)
        elif (ng_kernel.llapi_isNeuronIntVar(nodes[0], var_name)):
            print('case 3')
            print('nodes: {}'.format(nodes))
            ret = ng_kernel.llapi_getNeuronPtIntVar(nodes, var_name)
        elif (ng_kernel.llapi_isNeuronScalVar(nodes[0], var_name) |
              ng_kernel.llapi_isNeuronPortVar(nodes[0], var_name)):
            print('case 4')
            ret = ng_kernel.llapi_getNeuronPtVar(nodes, var_name)
        elif ng_kernel.llapi_isNeuronArrayVar(nodes[0], var_name):
            print('case 5')
            ret = ng_kernel.llapi_getNeuronListArrayVar(nodes, var_name)
        elif ng_kernel.llapi_isNeuronGroupParam(nodes[0], var_name):
            print('case 6')
            ret = []
            for i_node in nodes:
                ret.append(ng_kernel.llapi_getNeuronGroupParam(i_node, var_name))
        else:
            raise ValueError("Unknown neuron variable or parameter")
    return ret

def SetNeuronStatus(nodes, var_name, val):
    "Set neuron group scalar or array variable or parameter"
    if (type(nodes)!=list) & (type(nodes)!=tuple) & (type(nodes)!=NodeSeq):
        raise ValueError("Unknown node type")
    if (type(val)==dict):
        array_size = len(nodes)
        arr = ng_kernel.llapi_dictToArray(val, array_size)
        for i in range(array_size):
            SetNeuronStatus([nodes[i]], var_name, arr[i])
        return

    if type(nodes)==NodeSeq:
        if ng_kernel.llapi_isNeuronGroupParam(nodes.i0, var_name):
            ng_kernel.llapi_setNeuronGroupParam(nodes, var_name, val)
        elif ng_kernel.llapi_isNeuronScalParam(nodes.i0, var_name):
            ng_kernel.llapi_setNeuronScalParam(nodes.i0, nodes.n, var_name, val)
        elif (ng_kernel.llapi_isNeuronPortParam(nodes.i0, var_name) |
              ng_kernel.llapi_isNeuronArrayParam(nodes.i0, var_name)):
            ng_kernel.llapi_setNeuronArrayParam(nodes.i0, nodes.n, var_name, val)
        elif ng_kernel.llapi_isNeuronIntVar(nodes.i0, var_name):
            ng_kernel.llapi_setNeuronIntVar(nodes.i0, nodes.n, var_name, val)
        elif ng_kernel.llapi_isNeuronScalVar(nodes.i0, var_name):
            ng_kernel.llapi_setNeuronScalVar(nodes.i0, nodes.n, var_name, val)
        elif (ng_kernel.llapi_isNeuronPortVar(nodes.i0, var_name) |
              ng_kernel.llapi_isNeuronArrayVar(nodes.i0, var_name)):
            ng_kernel.llapi_setNeuronArrayVar(nodes.i0, nodes.n, var_name, val)
        else:
            raise ValueError("Unknown neuron variable or parameter")
    else:
        if ng_kernel.llapi_isNeuronScalParam(nodes[0], var_name):
            ng_kernel.llapi_setNeuronPtScalParam(nodes, var_name, val)
        elif (ng_kernel.llapi_isNeuronPortParam(nodes[0], var_name) |
              ng_kernel.llapi_isNeuronArrayParam(nodes[0], var_name)):
            ng_kernel.llapi_setNeuronPtArrayParam(nodes, var_name, val)
        elif ng_kernel.llapi_isNeuronIntVar(nodes[0], var_name):
            ng_kernel.llapi_setNeuronPtIntVar(nodes, var_name, val)
        elif ng_kernel.llapi_isNeuronScalVar(nodes[0], var_name):
            ng_kernel.llapi_setNeuronPtScalVar(nodes, var_name, val)
        elif (ng_kernel.llapi_isNeuronPortVar(nodes[0], var_name) |
              ng_kernel.llapi_isNeuronArrayVar(nodes[0], var_name)):
            ng_kernel.llapi_setNeuronPtArrayVar(nodes, var_name, val)
        else:
            raise ValueError("Unknown neuron variable or parameter")


def Calibrate():
    "Calibrate simulation"
    ret = ng_kernel.llapi_nESTGPU_Calibrate()
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
            for i_rule in range(len(conn_rule_name)):
                if conn_dict[param_name]==conn_rule_name[i_rule]:
                    break
            if i_rule < len(conn_rule_name):
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
                ng_kernel.llapi_setSynParamFromArray(param_name, fpar, array_size)
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
        ret = ng_kernel.setSynGroupStatus(gen_object, params, val)
        gc.enable()
        return ret
    nodes = gen_object
    if val != None:
         SetNeuronStatus(nodes, params, val)
    elif type(params)==dict:
        for param_name in params:
            SetNeuronStatus(nodes, param_name, params[param_name])
    elif (type(params)==list)  | (type(params) is tuple):
        if len(params) != len(nodes):
            raise ValueError("List should have the same size as nodes")
        for param_dict in params:
            if type(param_dict)!=dict:
                raise ValueError("Type of list elements should be dict")
            for param_name in param_dict:
                SetNeuronStatus(nodes, param_name, param_dict[param_name])
    else:
        raise ValueError("Wrong argument in SetStatus")
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
    gc.enable()

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
            for i_rule in range(len(conn_rule_name)):
                if conn_dict[param_name]==conn_rule_name[i_rule]:
                    break
            if i_rule < len(conn_rule_name):
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
    print('nestgpu.py, GetStatus(), gen_object type: {}, gen_object: {}'.format(
        type(gen_object), gen_object))
    if type(gen_object)==SynGroup:
        return GetSynGroupStatus(gen_object, var_key)

    if type(gen_object)==NodeSeq:
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
        if (type(gen_object)==ConnectionId):
            status_dict = GetConnectionStatus(gen_object)
        elif (type(gen_object)==int):
            i_node = gen_object
            status_dict = {}
            name_list = GetIntVarNames(i_node) \
                        + GetScalVarNames(i_node) + GetScalParamNames(i_node) \
                        + GetPortVarNames(i_node) + GetPortParamNames(i_node) \
                        + GetArrayVarNames(i_node) \
                        + GetArrayParamNames(i_node) \
                        + GetGroupParamNames(i_node)
            for var_name in name_list:
                val = GetStatus(i_node, var_name)
                status_dict[var_name] = val
        else:
            raise ValueError("Unknown object type in GetStatus")
        return status_dict
    elif (type(var_key)==str) | (type(var_key)==bytes):
        if (type(gen_object)==ConnectionId):
            print('type(gen_object)==ConnectionId')
            status_dict = GetConnectionStatus(gen_object)
            return status_dict[var_key]
        elif (type(gen_object)==int):
            print('type(gen_object)==int')
            i_node = gen_object
            return GetNeuronStatus([i_node], var_key)[0]
        else:
            raise ValueError("Unknown object type in GetStatus")

    else:
        raise ValueError("Unknown key type in GetStatus", type(var_key))

def GetRecSpikeTimes(nodes):
    if type(nodes)!=NodeSeq:
        raise ValueError("First argument type of GetRecSpikeTimes must be NodeSeq")

    ret = ng_kernel.llapi_getRecSpikeTimes(nodes.i0, nodes.n)
    return ret
