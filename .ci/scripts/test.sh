#!/bin/bash
#
# Copyright (C) 2022 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Author: Nadav Elyahu <nelyahu@habana.ai>
#

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

    # (set -x; eval LOG_LEVEL_ALL=${__log_level} ${__deepspeed_fork_test_exe} $__dir -v $__failures $__filter $__xml $__seed $__color ${__marker})

    # return error code of the test
    #return $?
    retrun 0
}

