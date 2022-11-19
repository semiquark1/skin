#!/usr/bin/env python3
#
# Author:	Ellak Somfai, 2015-2016,2018,2020.

"""Support for scripts"""

######## 2.7 compatibility
from __future__ import division, print_function
import sys
if sys.version.startswith('2'):
    str = basestring
######## end of compatibility

__version__ = '2021-03-26'

# standard library
import sys
import os
import argparse
import errno
import datetime
import time
from socket import getfqdn

# other common libraries
try:
    import yaml
except ImportError:
    pass    # will cause error only in SubCommand.append_yaml()


######## exception 

class Error(Exception): pass


######## class AttributeMap

class AttributeMap(object):
    def __init__(self, *args, **kwargs):
        for arg in args:
            # AttributeMap or argparse.Namespace
            for key in vars(arg):
                setattr(self, key, getattr(arg, key))
        self.set(**kwargs)

    def copy(self):
        """return a copy"""
        return AttributeMap(self)

    def override(self, **kwargs):
        """return a copy with some attributes set or overriden"""
        ret = self.copy()
        ret.set(**kwargs)
        return ret

    def set(self, *args, **kwargs):
        """either set(key, value) or set(key1=value1, ...)"""
        if len(args) == 2:
            if kwargs:
                raise Error('set(): illegal arguments')
            kwargs[args[0]] = args[1]
        elif len(args) != 0:
            raise Error('set(): illegal arguments')
        # add or replace attributes 
        for key, value in kwargs.items():
            # need to bypass __getattribute__
            setattr(self, key, value)

    def setdefault(self, *args, **kwargs):
        """either setdefault(key, value) or setdefault(key1=value1, ...)"""
        if len(args) == 2:
            if kwargs:
                raise Error('setdefault(): illegal arguments')
            kwargs[args[0]] = args[1]
        elif len(args) != 0:
            raise Error('setdefault(): illegal arguments')
        # add attributes if not yet present
        for key, value in kwargs.items():
            # need to bypass __getattribute__
            if key not in vars(self).keys():
                setattr(self, key, value)

    def assert_contains(self, *args):
        keys = vars(self).keys()
        for arg in args:
            arg = arg.replace(',', ' ')
            for key in arg.split():
                if key not in keys:
                    raise AssertionError('key "{}" missing'.format(key))

    def __repr__(self):
        return 'AttributeMap({})'.format(
                ', '.join(['{}={!r}'.format(key, value) for key,value in
                    sorted(vars(self).items())]))

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        # bypass __getattribute__
        if key not in self.__dict__.keys():
            return default
        return getattr(self, key)

    def keys(self):
        return self.__dict__.keys()


######## class SubCommand

def _format_elapsed(seconds: float):
    seconds_orig = seconds
    days, seconds = divmod(seconds, 24 * 60 * 60)
    hours, seconds = divmod(seconds, 60 * 60)
    minutes, seconds = divmod(seconds, 60)
    days = int(days); hours = int(hours); minutes = int(minutes)
    assert abs(days * 24*60*60 + hours * 60*60 + minutes * 60 + seconds
            - seconds_orig) < 1e-6
    if days:
        return f'{days}+{hours:02d}:{minutes:02d}'
    else:
        return f'{hours:02d}:{minutes:02d}'

# https://stackoverflow.com/a/43357954
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class SubCommand(object):
    name = None     # override iff multi-subcommand module
    description = ''

    @classmethod
    def parameters(cls, arg=[]):
        if isinstance(arg, AttributeMap):
            return arg
        if isinstance(arg, str):
            import shlex
            arg = shlex.split(arg)   # treat '' and ""
        # else: list of str
        parser =  cls.build_parser()
        args_argparse = parser.parse_args(arg)
        args_argparse.cmdline = ' '.join(arg)
        base = cls.__module__
        if base == '__main__':
            base = os.path.basename(os.path.realpath(sys.argv[0]))
            if base.endswith('.py'):
                base = base[:-3]
        if cls.name is None:
            args_argparse.filebase = '_'.join([base] + arg)
        else:
            args_argparse.filebase = '_'.join([base, cls.name] + arg)
        args_argparse.filebase = args_argparse.filebase.replace('/', '~')
        args_argparse.simple_filebase  = args_argparse.filebase.replace(
                '--', '').replace('=', '')
        cls.process_arguments(args_argparse, parser)
        return AttributeMap(args_argparse)

    @classmethod
    def build_parser(cls):
        prog = os.path.basename(sys.argv[0])
        if cls.name is not None:
            prog += '  {} '.format(cls.name)
        parser = argparse.ArgumentParser(
                prog=prog,
                formatter_class=argparse.RawTextHelpFormatter,
                description=cls.description.format(prog=prog))
        cls.add_arguments(parser)
        return parser

    @classmethod
    def print_usage(cls, file=None):
        cls.build_parser().print_usage(file)

    @classmethod
    def print_help(cls, file=None):
        cls.build_parser().print_help(file)

    def __init__(self, p=''):
        self.p = self.parameters(p)
        self._date_start = datetime.datetime.now().astimezone()
        self._time_start = time.time()

    def append_yaml(self, path, skip_stats=False, default_flow_style=False,
            **kwargs):
        try:
            with open(path) as f:
                info = yaml.safe_load(f)
        except FileNotFoundError:
            info = dict()
        if not skip_stats:
            date_now = datetime.datetime.now().astimezone()
            time_now = time.time()
            info_stats = dict(
                    cmdline = self.p.cmdline,
                    date_beg = self._date_start.isoformat('_', 'seconds'),
                    date_end = date_now.isoformat('_', 'seconds'),
                    elapsed = _format_elapsed(time_now - self._time_start),
                    elapsed_s = float(f'{time_now - self._time_start:.1f}'),
                    host = getfqdn(),
                    **kwargs,
                    )
            info.update(info_stats)
        info.update(kwargs)
        with open(path, 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=default_flow_style)
        return info

    def elapsed(self, display_s=False):
        time_now = time.time()
        ret = _format_elapsed(time_now - self._time_start)
        if display_s:
            ret += f' = {time_now - self._time_start:.1f} s'
        return ret

    # override in subclass if needed
    @classmethod
    def add_arguments(cls, parser):
        pass

    # override in subclass if needed
    @classmethod
    def process_arguments(cls, args, parser):
        pass

    # override in subclass
    def run(self):
        raise NotImplementedError


def skip(subcommand):
    'class decorator: ignore this subcommand'
    subcommand._skip = True
    return subcommand


######## completer(), main_single() and main_multi()

def completer(arg):
    """Generate bash complete suggestions

    arg is either a SubCommand subclass, or a list of SubCommand subclasses.
    """
    if isinstance(arg, list):
        subcommands = arg
        is_multi = True
    else:
        subcommand = arg
        is_multi = False
    comp_line = os.environ['COMP_LINE'][:int(os.environ['COMP_POINT'])]
    comp_cmd = sys.argv[2]
    comp_word = sys.argv[3]
    comp_prevword = sys.argv[4]
    def suggest(txt):
        if txt.startswith(comp_word):
            print(txt)
    if (is_multi and len(comp_line.split()) <= 2 and comp_cmd == comp_prevword):
        # subcommand names
        for subcmd in subcommands:
            suggest(subcmd.name)
    else:
        # subcommand arguments
        if is_multi:
            subcmd_name = comp_line.split()[1]
            subcommand = [i for i in subcommands if i.name == subcmd_name][0]
        for act in subcommand.build_parser()._actions:
            for sugg in act.option_strings:
                suggest(sugg)
            if len(act.option_strings) == 0:
                if act.metavar:
                    suggest('`'+act.metavar+'`')
                else:
                    suggest('`'+act.dest+'`')
    sys.exit(0)

def main_single(module_name, subcommand, import_psim_fn=None):
    'Closure for main(), single SubCommand subclass'
    def main_fn(*args):
        """main() function for scripts with single SubCommands

        main():                         print usage
        main("-h"):                     print help for subcommand
        main("..args in a string.."):   call subcommand
        main(MySubCmd.parameters(..)):  call subcommand
        """
        try:
            assert subcommand.name is None, ('name is not None '
                    'in single-subcommand module')
            if len(args) == 0 and module_name == '__main__':
                args = sys.argv[1:]
            if len(args) >= 1 and args[0] == '--completer':
                completer(subcommand)
                # does not return, calls sys.exit()
            if len(args) == 1 and isinstance(args[0], str):
                args = args[0]
            p = subcommand.parameters(args)
            if import_psim_fn is not None:
                import_psim_fn(p.dim)
            app = subcommand(p)
            app.run()
        except Error as e:
            print(e, file=sys.stderr)
            sys.exit(1)
    return main_fn

def main_multi(module_name):
    """Closure for main(), multiple SubCommand subclasses"""
    def main_fn(*args):
        """main() function for scripts with multiple SubCommands

        main():                                     print global help
        main(subcmd_name, "-h"):                    print help for subcommand
        main(subcmd_name, "..args in a string.."):  call subcommand
        main(subcmd_name, MySubCmd.parameters(..)): call subcommand
        """
        try:
            # obtain subcommands: subclasses of SubCommand
            # first try reading source file: order preserved for global help
            module = __import__(module_name)
            subcommands = []
            try:
                with open(module.__file__) as f:
                    for line in f:
                        line = line.strip()
                        # this test is a cludge, can be easily fooled
                        if (line.startswith('class ')
                                and 'SubCommand' in line):
                            name = line.split()[1].split('(')[0].strip()
                            obj = getattr(module, name)
                            if (isinstance(obj, type)
                                    and issubclass(obj, SubCommand)
                                    and obj != SubCommand
                                    and not getattr(obj, '_skip', False)):
                                subcommands.append(obj)
            except:
                pass
            # if did not find any: try dir(module)
            if len(subcommands) == 0:
                for name in dir(module):
                    obj = getattr(module, name)
                    try:
                        if (issubclass(obj, SubCommand)
                                and obj != SubCommand
                                and not getattr(obj, '_skip', False)):
                            subcommands.append(obj)
                    except:
                        pass
            # test that all subcommands have name
            for subcmd in subcommands:
                assert subcmd.name is not None, ('name is None '
                        'in multi-subcommand module')
            # handle *args
            if len(args) == 0 and module_name == '__main__':
                args = sys.argv[1:]
            if len(args) == 0 or args[0].strip().startswith(('-h', '--h')):
                # print usage
                for subcmd in subcommands:
                    subcmd.print_usage()
                return
            if len(args) >= 1 and args[0] == '--completer':
                completer(subcommands)
                # does not return, calls sys.exit()
            # execute subcommand
            for subcmd in subcommands:
                if subcmd.name == args[0]:
                    if len(args) == 2 and isinstance(args[1], str):
                        app = subcmd(args[1])
                    else:
                        app = subcmd(list(args[1:]))
                    app.run()
                    return
            raise Error('unknown subcommand: ' + args[0])
        except Error as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    return main_fn


######## makedirs()

def makedirs(dir=None, filepath=None, mode=0o777):
    """Create directory as os.makedirs, do nothing (no error) if dir exists"""
    if dir is None:
        dir = os.path.dirname(filepath)
    if dir == '':
        return
    try:
        os.makedirs(dir, mode)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

# vim: set sw=4 sts=4 expandtab :
