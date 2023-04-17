#!/bin/bash
#
# Copyright (C) 2022 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Author: Nadav Elyahu <nelyahu@habana.ai>
#

python_ver="python3"
# --- helper functions ---

function deepspeed_fork_test_usage()
{
    if [ $1 == "run_deepspeed_fork_test" ]; then
        echo -e "usage: $1 [options]\n"
        echo -e "******\n***** need to complete the help...******\n******"
        echo -e "  -h,  --help                 Prints this help"

    fi
}

function run_deepspeed_fork_test()
{
    local __deepspeed_fork_test_exe="$__python_cmd -m pytest"

    local __scriptname=$(__get_func_name)
    local __print_tests=""
    local __filter=""
    local __xml=""
    local __failures=""
    local __seed=""
    local __marker=""
    local __color=""
    local __log_level="4"
    local __dir="$DEEPSPEED_FORK_ROOT/hpu_tests/"

    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -l  | --list-tests )
            __print_tests="yes"
            ;;
        -s  | --specific-test )
            shift
            __filter="-k \"$1\""
            ;;
        -a | --marker )
            shift
            #__marker="-m \"$1\""
            echo "script argument --marker is not supported at the moment"
            return 1
            ;;
        -m  | --maxfail )
            shift
            __failures="--maxfail=$1"
            ;;
        -x  | --xml )
            shift
            __xml="--junit-xml=$1"
            ;;
        -nr  | --rand_disable )
            shift
            __rand=""
            ;;
        --no-color )
            __color="--color=no"
            ;;
        --log-level )
            __log_level="$1"
            ;;
        -r | --release )
            # does nothing
            ;;
        -h  | --help )
            deepspeed_fork_test_usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            deepspeed_fork_test_usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    if [ -n "$__print_tests" ]; then
        # ${__deepspeed_fork_test_exe} __dir --collect-only
        # return $?
        return 0
    fi

    (set -x; eval LOG_LEVEL_ALL=${__log_level} ${__deepspeed_fork_test_exe} $__dir -v $__failures $__filter $__xml $__seed $__color ${__marker})

    # return error code of the test
    return $?
    #return 0
}


get_deepspeed_fork_dir_path()
{
    if [[ -z "${DEEPSPEED_FORK_ROOT}" ]]; then
        dir_path="/root/deepspeed-fork"
    else
        dir_path="${DEEPSPEED_FORK_ROOT}"
    fi
    if [ ! -d "$dir_path" ]; then
        echo "$1 is not found."
        return 1
    fi
    echo $dir_path
}


install_deepspeed_unit_test_requirements(){
    deepspeed_fork_path=$(get_deepspeed_fork_dir_path)
    local res=$?
    if [ $res -eq "0" ]; then
        for FILE in ${deepspeed_fork_path}/requirements/*; do pip install -r $FILE; done
        pip install pytest
        pip install pytest-html
        pip install lxml
    else
        return $res
    fi
    return 0
}


#copy_artifact_data $__output_dir $__artifact_path
copy_artifact_data()
{
    cp -f -r $1/*.csv  ${2}/
}


deepspeed_fork_unit_tests_usage()
{
    deepspeed_fork_path=$(get_deepspeed_fork_dir_path)
    local res=$?
    if [ $res -eq "0" ]; then
        $python_ver $deepspeed_fork_path/.ci/scripts/$1 -h
    else
        return $res
    fi
    return 0
}


run_deepspeed_fork_unit_tests()
{
    local __scriptname="run_deepspeed_unit_test.py"
    local __output_dir=""
    local __artifact_path=""
    local __cmd=""
    deepspeed_fork_path=$(get_deepspeed_fork_dir_path)
    local res=$?
    if [ $res -eq "0" ]; then
        __cmd="${python_ver} $deepspeed_fork_path/.ci/scripts/${__scriptname} "
    else
        return $res
    fi
    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -tm  | --test_mode )
            shift
            __cmd+=" --test_mode  ${1}"
            ;;
        -use_hpu  | --use_hpu )
            __cmd+=" --use_hpu "
            ;;
        -o| --output_dir )
            shift
            __output_dir="${1}"
            __cmd+=" --output_dir ${1} "
            ;;
        -ts| --test_script )
            shift
            __cmd+=" --test_script ${1} "
            ;;
        -tc| --test_case )
            shift
            __cmd+=" --test_case ${1} "
            ;;
        -m| --marker )
            shift
            __cmd+=" --marker \"${1}\" "
            ;;
        -h  | --help )
            deepspeed_fork_unit_tests_usage $__scriptname
            return 0
            ;;
        -x| --x )
            shift
            __artifact_path=${1}
            __cmd+=" --artifact_dir ${1} "
            ;;
        *)
            echo "The parameter $1 is not allowed"
            deepspeed_fork_unit_tests_usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done
    deepspeed_fork_path=$(get_deepspeed_fork_dir_path)
    local res=$?
    if [ $res -eq "0" ]; then
        __cmd+=" --deepspeed_test_path ${deepspeed_fork_path}/tests "
    else
        return $res
    fi
    $__cmd
    local ret=$?
    if [  $__artifact_path ]; then
        copy_artifact_data $__output_dir $__artifact_path
    fi
    return $ret
}
