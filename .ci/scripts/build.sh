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
function deepspeed_fork_build_help()
{
    echo -e "\n- The following is a list of available functions in deepspeed-fork/build.sh"
    echo -e "build_deepspeed_fork                 - Installs deepspeed python package from deepspeed-fork"
    echo
}

function deepspeed_fork_build_usage()
{
    if [ $1 == "build_deepspeed_fork" ]; then
        echo -e "\nusage: $1 [options]\n"
        echo -e "options:\n"

        echo -e "  -d,  --development          Installs deepspeed to run directly from the repository path, otherwise will run from python's site-packages library"
        echo -e "  -f,  --force-reinstall      Forces to re-install the library"
        echo -e "  -r,  --release              Does nothing"
        echo -e "  -c,  --configure            Does nothing"
        echo -e "  -h,  --help                 Prints this help"
        echo

    fi
}

build_deepspeed_fork ()
{
    SECONDS=0

    _verify_exists_dir "$DEEPSPEED_FORK_ROOT" $DEEPSPEED_FORK_ROOT

    local __scriptname=$(__get_func_name)

    local __developement=""
    local __configure=""
    local __force_reinstall=""


    # parameter while-loop
    while [ -n "$1" ];
    do
        case $1 in
        -d  | --developement )
            __developement="-e"
            ;;
        -f  | --force-reinstall )
            __force_reinstall="--force-reinstall"
            ;;
        -r  | --release | -c | --configure )
            # does nothing
            ;;
        -h  | --help )
            deepspeed_fork_build_usage $__scriptname
            return 0
            ;;
        *)
            echo "The parameter $1 is not allowed"
            deepspeed_fork_build_usage $__scriptname
            return 1 # error
            ;;
        esac
        shift
    done

    pip install $__developement $__force_reinstall $DEEPSPEED_FORK_ROOT/.

    printf "\nElapsed time: %02u:%02u:%02u \n\n" $(($SECONDS / 3600)) $((($SECONDS / 60) % 60)) $(($SECONDS % 60))
    return 0
}
