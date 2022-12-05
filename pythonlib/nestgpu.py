""" Python interface for NESTGPU"""
import sys, platform
import os
import unicodedata
import gc
import nestgpukernel_api as ng_kernel


print('\n              -- NEST GPU --\n')
print('  Copyright (C) 2004 The NEST Initiative\n')
print(' This program is provided AS IS and comes with')
print(' NO WARRANTY. See the file LICENSE for details.\n')
print(' Homepage: https://github.com/nest/nest-gpu')
print()


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

class ConnectionId(object):
    def __init__(self, i_source, i_group, i_conn):
        self.i_source = i_source
        self.i_group = i_group
        self.i_conn = i_conn

class SynGroup(object):
    def __init__(self, i_syn_group):
        self.i_syn_group = i_syn_group

def to_byte_str(s):
    if type(s)==str:
        return s.encode('ascii')
    elif type(s)==bytes:
        return s
    else:
        raise ValueError("Variable cannot be converted to string")

conn_rule_name = ("one_to_one", "all_to_all", "fixed_total_number",
                  "fixed_indegree", "fixed_outdegree")

ng_kernel.llapi_setOnException(1)

def SetSimTime(sim_time):
    "Set neural activity simulated time in ms"
    ret = ng_kernel.llapi_setSimTime(sim_time)
    if ng_kernel.llapi_getErrorCode() != 0:
        raise ValueError(ng_kernel.llapi_getErrorMessage())
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


# TODO: translate subroutines
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

    ret = NESTGPU_SetNeuronGroupParam(nodes.i0, nodes.n, param_name.encode('utf-8'), val)
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
        raise ValueError("Unknown key type in GetSynGroupStatus", type(var_key))

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
