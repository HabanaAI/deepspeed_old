# *****************************************************************************
# Copyright (C) 2023 HabanaLabs, Ltd.
# All Rights Reserved.

# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# *****************************************************************************

import os
import sys
import re
import time
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import shutil
from bs4 import BeautifulSoup
from datetime import datetime

parser = argparse.ArgumentParser(
    description='Execute and post process the Deepspeed unit test.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--output_dir', metavar='<output_dir>',
                    help='Output directory.')

parser.add_argument('--test_script', metavar='<test_script>',
                    help='Execute Specific script.')

parser.add_argument('--test_case', metavar='<test_case>',
                    help='Execute Specific test case.')

parser.add_argument('--deepspeed_test_path', metavar='<deepspeed_test_path>',
                    help='DeepSpeed test path.')

parser.add_argument('--use_hpu',
                    help='Enable habana backend, default False.',
                    action='store_true', default=False)

parser.add_argument('--marker', metavar='<marker>', nargs='*',
                    help='Pytest marker.')

parser.add_argument('--test_mode', metavar='<test_mode>',
                    help='Type of test.',
                    choices=['training', 'inference'], default='training')

parser.add_argument('--artifact_dir', metavar='<artifact_dir>',
                    help='Artifact directory.')


def get_test_error_info(logs):
    error_log = []
    if re.search('error', logs, re.IGNORECASE) or re.search('fail', logs, re.IGNORECASE):
        for line in logs.splitlines():
            if re.search('error', line, re.IGNORECASE) or re.search('fail', line, re.IGNORECASE):
                error_log.append(line)
    else:
        error_log.append("No error/failure logs observed.")
    return ', '.join(error_log)


def get_test_skip_info(args, file, test_name):
    file_name = os.path.join(
        args.output_dir, 'xml_files', os.path.basename(file).replace('.html', '.xml'))
    test_name = '::'.join(test_name.split('::')[1:])
    flag = False
    with open(file_name, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    test_cases = bs_data.find_all('testcase')
    for test_case in test_cases:
        if check_test_name(test_case, test_name):
            try:
                if test_case('skipped')[0].get('type') == 'pytest.skip':
                    return test_case('skipped')[0].get('message')
            except:
                return ""
    return ''


def check_test_name(attrib, test_name):
    if attrib['name'] == 'test':
        class_name = test_name.split('::')[0]
        if class_name in attrib['classname']:
            return True
    elif attrib['name'] in test_name:
        return True
    else:
        return False


def get_xfail_info(args, file, test_name):
    file_name = os.path.join(
        args.output_dir, 'xml_files', os.path.basename(file).replace('.html', '.xml'))
    test_name = '::'.join(test_name.split('::')[1:])
    flag = False
    with open(file_name, 'r') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    test_cases = bs_data.find_all('testcase')
    for test_case in test_cases:
        if check_test_name(test_case, test_name):
            try:
                if test_case('skipped')[0].get('type') == 'pytest.xfail':
                    return test_case('skipped')[0].get('message')
            except:
                return ""
    return ''


def parse_html(args, file):
    # empty list
    data = []

    # the HTML file
    soup = BeautifulSoup(open(file), 'html.parser')

    list_header = ["Test Status", "Test Name",
                   "Test Duration", "Test Links", "Test Logs"]

    # for getting the data
    HTML_data = soup.find_all("table")[1].find_all("tbody")

    for element in HTML_data:
        sub_data = []
        for sub_element in element:
            if '\n' == sub_element:
                continue
            for key in sub_element:
                if '\n' == key:
                    continue
                try:
                    sub_data.append(key.get_text())
                except:
                    continue
        data.append(sub_data)

    df = pd.DataFrame(data=data, columns=list_header)
    del (df["Test Links"])
    df['Fail/XFail Reason'] = ''
    df['XFail INFO'] = ''
    df['Module Name'] = ''
    df['Skip Reason'] = ''
    for ind in df.index:
        df['Module Name'] = df['Test Name'][ind].split('.py')[0] + '.py'
        if df['Test Status'][ind] == 'Failed' or df['Test Status'][ind] == 'XFailed':
            # Find the XFAIL error from xml.
            if df['Test Status'][ind] == 'XFailed':
                df['XFail INFO'][ind] = get_xfail_info(
                    args, file, df['Test Name'][ind])
            df['Fail/XFail Reason'][ind] = get_test_error_info(
                df['Test Logs'][ind])
        if df['Test Status'][ind] == 'Skipped':
            df['Skip Reason'][ind] = get_test_skip_info(args, file, df['Test Name'][ind])

    del (df["Test Logs"])
    df = df.reindex(columns=['Module Name', 'Test Name', 'Test Duration',
                    'Test Status', 'Fail/XFail Reason', 'XFail INFO', 'Skip Reason'])
    return df


def execute_tests(args):
    begin = time.time()
    # Run Specific test case
    if args.test_case or args.test_script or args.marker:
        test_name = ''
        test_path = os.path.join(args.deepspeed_test_path, 'unit')

        if args.test_script:
            test_name = os.path.splitext(os.path.basename(args.test_script))[0]
            test_path = os.path.join(test_path, args.test_script)
        cmd = 'python3 -m pytest {} -o junit_family=legacy -vv '.format(
            test_path)
        if args.marker:
            test_name = "ci_promote_tests"
            cmd += ' -m {} '.format(' '.join(args.marker))
        if args.test_case:
            test_name = args.test_case
            cmd += ' -k {} '.format(args.test_case)
        if args.use_hpu:
            cmd = 'REPLACE_FP16=1 {} --use_hpu True '.format(cmd)
        if args.test_mode == 'inference' and not args.marker:
            cmd = ' {} -m \'nightly or inference or seq_inference or sequential \' '.format(
                cmd)
        seed = os.getenv('PYTEST_SEED', default=None)
        if seed:
            cmd = '{} --randomly-seed={}'.format(cmd, seed)
        cmd = '{} --html={} --junitxml {}  2>&1 | tee -a {}'.format(cmd, os.path.join(
            args.output_dir, 'html_files', test_name + '.html'), os.path.join(
            args.output_dir, 'xml_files', test_name + '.xml'), os.path.join(args.output_dir, 'logs.txt'))
        print(cmd)
        os.system('echo cmd >> {}'.format(
            os.path.join(args.output_dir, 'logs.txt')))
        os.system(cmd)
        if args.artifact_dir:
            try:
                shutil.copy2(os.path.join(args.output_dir, 'xml_files', test_name + '.xml'), args.artifact_dir)
            except:
                print("Unexpected error: {}".format(sys.exc_info()[0]))

    # Run All test case
    else:
        for subdir, dirs, files in os.walk(os.path.join(args.deepspeed_test_path, 'unit')):
            for test in files:
                if 'test' in test and (not '.pyc' in test):
                    cmd = 'python3 -m pytest {} -o junit_family=legacy -vv  '.format(
                        os.path.join(subdir, test), os.path.join(args.output_dir, test + '.html'))
                    if args.use_hpu:
                        cmd = 'REPLACE_FP16=1 {} --use_hpu True '.format(cmd)
                    if args.test_mode == 'inference':
                        cmd = ' {} -m \'nightly or inference or seq_inference or sequential \' '.format(
                            cmd)
                    seed = os.getenv('PYTEST_SEED', default=None)
                    if seed:
                        cmd = '{} --randomly-seed={}'.format(cmd, seed)
                    cmd = '{} --html={} --junitxml {}  2>&1 | tee -a {}'.format(cmd, os.path.join(
                        args.output_dir, 'html_files', test.replace('.py', '.html')), os.path.join(
                            args.output_dir, 'xml_files', test.replace('.py', '.xml')), os.path.join(args.output_dir, 'logs.txt'))
                    print(cmd)
                    os.system('echo cmd >> {}'.format(
                        os.path.join(args.output_dir, 'logs.txt')))
                    os.system(cmd)
                    if args.artifact_dir:
                        try:
                            shutil.copy2(os.path.join(args.output_dir, 'xml_files', test.replace('.py', '.xml')), args.artifact_dir)
                        except:
                            print("Unexpected error: {}".format(sys.exc_info()[0]))
    end = time.time()
    print('Total Execution time: ', end - begin)


def generate_report(args):
    html_files = os.path.join(
        args.output_dir, 'html_files', '*.html')
    filenames = glob.glob(html_files)
    filenames = sorted(filenames)
    df = pd.DataFrame()
    for file in filenames:
        df = pd.concat([df, parse_html(args, file)])
    file_name = 'DeepSpeed_test_report_' + \
        datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    if args.use_hpu:
        file_name = file_name + '_hpu_{}.csv'.format(args.test_mode)
    else:
        file_name = file_name + '_gpu_{}.csv'.format(args.test_mode)
    df.to_csv(os.path.join(args.output_dir,
              file_name), index=False, header=True)


def remove_old_data(args):
    # Remove previous run output directory, if any
    os.system("rm -rf {}".format(args.output_dir))


if __name__ == '__main__':
    args = parser.parse_args()
    remove_old_data(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'xml_files'))
        os.makedirs(os.path.join(args.output_dir, 'html_files'))
    os.chdir(args.deepspeed_test_path)
    execute_tests(args)
    generate_report(args)
