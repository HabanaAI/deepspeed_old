import os
import pytest

def enable_hpu(flag):
    with open('unit/enable_hpu.py', 'w') as fd:
        fd.write("import pytest\n")
        fd.write(f"pytest.use_hpu='{flag}'\n")

def get_hpu_dev_version():
    try:
        command_output = os.popen('hl-smi -L')
        command_output_list = command_output.read()
        device_id = [s for s in command_output_list.split('\n') if 'Device Id' in s][0].split()[-1]
        if ('0x1da31000' in device_id) or ('0x1da31001' in device_id):
            return "Gaudi"
        elif '0x1da31020' in device_id:
            return "Gaudi2"
    except:
        pass
    return None


def is_hpu_supported(config):
    # FP16 is not supported by HPU.
    if config.get('fp16'):
        if config.get('fp16', None).get('enabled', None) == True:
            if get_hpu_dev_version() == 'Gaudi':
                return False, "FP16 datatype is not supported by HPU"
            if get_hpu_dev_version() == 'Gaudi2':
                return False, "FP16 datatype support is not added for Gaudi2. SW-111219"
    # Fused ADAM is not supported
    if config.get('optimizer'):
        if config.get('optimizer', None).get('params', None):
            if config.get('optimizer', None).get('params', None).get('torch_adam', None) == False:
                return False, "Fused ADAM optimizer is not supported by HPU"
            if config.get('optimizer', None).get('type', None) == "Lamb":
                return False, "LAMB optimizer is not supported by HPU"
            if config.get('optimizer', None).get('type', None) == "OneBitAdam":
                return False, "OneBitAdam optimizer is not supported by HPU"
    # Zero-3 is not supported
    if config.get('zero_optimization'):
        if config.get('zero_optimization', None).get('stage', None) == 3:
            return False, "DeepSpeed Stage3 is not supported by HPU"
    # sparse gradients is not supported by HPU.
    if 'sparse_gradients' in config:
        if config['sparse_gradients'] == True:
            return False, "sparse_gradients is not supported by HPU"

    return True, ''
