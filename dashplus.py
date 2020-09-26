'''
DashPlus
========
This is an extended version of the DashUI that comes with Maya.
Props to Ian Waters, Autodesk, the MASH team and anyone else who may have
had a hand in the original feature.
'''

from __future__ import print_function

__version__ = '0.1.2'
__author__ = 'Dan Bradham'
__email__ = 'danielbradham@gmail.com'
__license__ = 'MIT'

import os
import operator
import json
from contextlib import contextmanager
from collections import namedtuple
from types import ModuleType
from itertools import groupby
from collections import deque
import sys

# This is a maya only tool - we can rely on Qt for Python (PySide2)
from PySide2 import QtWidgets, QtGui, QtCore

from maya import cmds
import DashLegacy
import Dash
import DashCommand


# Python 3 Compatability
if sys.version_info > (2, 7):
    basestring = (str,)
    operator.div = operator.truediv


if '_dash' not in sys.modules:
    # Store the real Dash objects
    _dash = ModuleType('_dash')
    _dash.DashUILegacy = DashLegacy.DashUI
    _dash.DashUI = Dash.DashUI
    _dash.DashCommand = DashCommand.DashCommand
    _dash.dashHelp = DashCommand.dashHelp
    sys.modules['_dash'] = _dash
else:
    # We've already stored the original Dash objects we're going to patch
    _dash = sys.modules['_dash']


_registry = []
_functions = {}


def install():
    '''Install dashplus.

    Patches DashCommand.DashCommand to make it easy to add additional commands.
    '''

    if cmds.about(batch=True):
        # Disabled in batch mode
        return

    if not cmds.pluginInfo('MASH', query=True, loaded=True):
        cmds.loadPlugin('MASH')

    DashLegacy.DashUI = DashPlusUI
    Dash.DashUI = DashPlusUI
    for cmd in read_commands():
        if cmd_exists(cmd['DashCommand'], cmd['ShortDashCommand'], _registry):
            continue
        cmd['Hint'] = '()'
        cmd['Icon'] = None
        _registry.append(cmd)

    log('Installed.', in_view=False)
    help()


def uninstall():
    '''Uninstall dashplus.

    Removes the DashCommand.DashCommand patch.
    '''

    if cmds.about(batch=True):
        # Disabled in batch mode
        return

    DashLegacy.DashUI = _dash.DashUILegacy
    Dash.DashUI = _dash.DashUI

    log('Uninstalled.', in_view=False)


def help():
    '''Print the list of available dash commands and some instructions.'''

    print('\n')
    print(sys.modules[__name__].__doc__)

    header = '{DashCommand}: {Description}'
    invoke = '    {DashCommand}{Hint}\n    {ShortDashCommand}{Hint}\n'
    print('Available dash commands')
    print('-----------------------\n')
    for cmd in _registry:
        print(header.format(**cmd))
        print(invoke.format(**cmd))


def log(message, *args, **kwargs):
    '''Log a message to the console.'''

    if isinstance(message, basestring):
        message = message % args

    if kwargs.get('in_view', True):
        cmds.inViewMessage(
            amg='dashplus:' + message,
            pos='midCenter',
            fade=True,
        )

    print('DASHPLUS | %s' % message)


def error(message, *args, **kwargs):
    '''Raise a RuntimeError and show the message in the viewport.'''

    if isinstance(message, basestring):
        message = message % args

    if kwargs.get('in_view', True):
        cmds.inViewMessage(
            amg='dashplus:' + message,
            pos='midCenter',
            fade=True,
        )

    cmds.error('DASHPLUS | %s' % message)


class DashPlus(object):
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.div,
        '\\': operator.div,
        '%': operator.mod,
        '^': operator.xor,
    }
    cbox_ops = [op + '=' for op in ops.keys()]

    def __init__(self, text):
        self.text = text

    @property
    def is_expr(self):
        return self.text.startswith('=')

    @property
    def is_py_expr(self):
        return self.text.startswith(':')

    @property
    def is_cbox_expr(self):
        return (
            len(self.text) >= 2
            and self.text[:2] in self.cbox_ops
        )

    @property
    def is_cbox_time_expr(self):
        return len(self.text) >= 3 and self.text[2] == ':'

    def tokenize_expr(self):
        text = self.text.lstrip('=').strip()
        token = namedtuple('Token', 'text expr')
        tokens = []

        start_char = '$'
        end_chars = [' ', ')', ',']
        chunk = ''
        depth = 0
        for i, char in enumerate(text):
            if char == start_char:
                chunk += char
                continue
            if not chunk:
                continue
            if char == '(':
                chunk += char
                depth += 1
            elif char == ')':
                depth -= 1
                if depth <= 0:
                    tokens.append(token(chunk, chunk[1:]))
                    chunk = ''
                else:
                    chunk += ')'
            elif char in end_chars:
                tokens.append(token(chunk, chunk[1:]))
                chunk = ''
            else:
                chunk += char
        if chunk:
            tokens.append(token(chunk, chunk[1:]))
        return tokens

    def tokenize_dash(self):
        text = self.text.strip()
        token = namedtuple('Token', 'type text')
        tokens = []

        chunk = ''
        depth = 0
        for i, char in enumerate(text):
            chunk += char
            if char == '(':
                if depth == 0:
                    # We found a function call
                    chunks = chunk[:-1].split()
                    if len(chunks) > 1:
                        # only the actual function name is useful to us
                        tokens.append(token('unknown', ' '.join(chunks[:-1])))
                    tokens.append(token('func', chunks[-1]))
                    chunk = char
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    # We reached the root level again, everything up to this
                    # part is the functions arguments
                    tokens.append(token('args', chunk))
                    chunk = ''
            elif chunk == ' ' and depth == 0:
                chunk = ''

        if chunk:
            # There was some stuff on the end of the text
            # we can ignore this or raise an error
            tokens.append(token('unknown', chunk))

        if depth != 0:
            error('Mismatched parenthesis')

        num_funcs = len([t for t in tokens if t.type == 'func'])
        num_args = len([t for t in tokens if t.type == 'args'])
        if num_funcs != num_args:
            error('Every function call needs arguments')

        return tokens

    def execute(self):
        # TODO: Wrap in undo chunk
        with undo('DashPlus: %s' % self.text):
            executor = self.get_executor()
            executor()

    def get_executor(self):
        if self.is_expr:
            return self.execute_expr
        elif self.is_cbox_expr:
            return self.execute_cbox_expr
        elif self.is_py_expr:
            return self.execute_python
        else:
            return self.execute_dash

    def execute_expr(self):
        expr = self.text.lstrip('=').strip()
        tokens = self.tokenize_expr()
        namespace = self.get_execution_namespace()
        for i, t, channel in selected_channels():
            namespace.update(
                i=i,
                t=t,
                c=channel,
                v=channel.get(),
                j=0
            )
            chan_expr = expr
            for token in tokens:
                # Special case $this token expands to the current object
                if '$this' in token.text:
                    chan_expr = chan_expr.replace('$this', channel.node_name)
                    continue

                # Evaluate the text after $ variables as python then replace
                value = eval(token.expr, namespace, namespace)
                if value is None:
                    error('Failed to evaluate: %s', token.text)
                chan_expr = chan_expr.replace(token.text, repr(value))
            channel.set_expression(chan_expr, expr)

    def execute_cbox_expr(self):
        sep = ('=', '=:')[self.is_cbox_time_expr]
        parts = self.text.split(sep)
        if len(parts) != 2:
            error('Malformed channelBox expression.')
        op_str, eval_str = parts
        op = self.ops[op_str]
        namespace = self.get_execution_namespace()
        for i, t, channel in selected_channels():
            namespace.update(
                i=i,
                t=t,
                c=channel,
                time=cmds.currentTime(q=True),
                v=channel.get(),
                j=0
            )
            if self.is_cbox_time_expr and channel.has_keys:
                # Move, Scale, and modulate times
                if op_str in ['+', '-', '\\', '/', '*']:
                    right_value = eval(eval_str, namespace, namespace)
                    if right_value is None:
                        error('ChannelBox expression returned no result.')
                    if op_str in ['+', '-']:
                        channel.move_times(op(0, right_value))
                    if op_str in ['\\', '/', '*']:
                        channel.scale_times(op(1.0, right_value), 0.0)
                else:
                    for j, time, left_value in channel.keys():
                        namespace.update(time=time, v=left_value, j=j)
                        right_value = eval(eval_str, namespace, namespace)
                        if right_value is None:
                            error('ChannelBox expression returned no result.')
                        if op_str == '^':
                            left_value = int(left_value)
                            right_value = int(right_value)
                        channel.set_time(j, op(left_value, right_value))
            else:
                # Operate only on the current frames value
                left_value = channel.get()
                right_value = eval(eval_str, namespace, namespace)
                if right_value is None:
                    error('ChannelBox expression returned no result.')
                if op_str == '^':
                    left_value = int(left_value)
                    right_value = int(right_value)
                channel.set(op(left_value, right_value))

    def execute_dash(self):
        '''This is a superset of Dash standard. Also supporting command
        chaining.'''

        log('Executing dash command.')
        tokens = self.tokenize_dash()
        for func, args in zip(tokens[0::2], tokens[1::2]):
            if (func.type, args.type) != ('func', 'args'):
                error('Malformed function call')
            cmd = find_command(func.text)
            if not cmd:
                error('Could not find command: %s', func.text)
            namespace = self.get_execution_namespace()
            exec_str = cmd['MayaExec']
            eval_str = cmd['MayaEval'] + args.text
            exec(exec_str, namespace, namespace)
            eval(eval_str, namespace, namespace)

    def execute_python(self):
        log('Executing dash python command.')
        eval_str = self.text.lstrip(':').strip()
        namespace = self.get_execution_namespace()
        for i, t, channel in selected_channels():
            namespace.update(
                i=i,
                t=t,
                c=channel,
                time=cmds.currentTime(q=True),
                v=channel.get(),
                j=0
            )
            if channel.has_keys:
                for j, time, value in channel.keys():
                    namespace.update(time=time, v=value, j=j)
                    result = eval(eval_str, namespace, namespace)
                    if result is not None:
                        channel.set_value(j, result)
            else:
                result = eval(eval_str, namespace, namespace)
                if result is not None:
                    channel.set(result)

    def get_execution_namespace(self):
        import math as pymath
        import random as pyrandom
        from maya import cmds
        from maya import mel

        # Fast and loose
        dashplus_math = {
            k: v.__func__ for k, v in vars(math).items()
            if not k.startswith('_')
        }

        namespace = {}
        namespace.update(globals())
        namespace.update(vars(pymath))
        namespace.update(vars(pyrandom))
        namespace.update(cmds=cmds, mel=mel)
        namespace.update(dashplus_math)
        return namespace


class UIState(object):

    history = deque([None])
    pinned = False

    @classmethod
    def toggle_pin(cls):
        cls.pinned = not cls.pinned

    @classmethod
    def prev_history(cls):
        if len(cls.history) == 1:
            return ''
        if cls.history[1] is None:
            return cls.history[0]
        cls.history.rotate(-1)
        return cls.history[0]

    @classmethod
    def next_history(cls):
        if cls.history[0] is None:
            return ''
        cls.history.rotate(1)
        return cls.history[0]

    @classmethod
    def add_history(cls, command_str):
        while cls.history[-1] is not None:
            cls.history.rotate(-1)
        if command_str != cls.history[0]:
            cls.history.appendleft(command_str)
        cls.history.rotate(1)


class DashPlusUI(QtWidgets.QDialog):
    '''Complete replacement for the Dash UI. This thing matches the geometry
    of the currently hovered channelBox row making it fit really snug into
    the Maya UI. It also has a pin button and a menu filled with the available
    dash commands.
    '''

    def __init__(self, parent=None):
        super(DashPlusUI, self).__init__(parent)
        rect = channelBox_row_rect()
        height = rect.height()

        # Use a monospace font - More editorish
        self.font = QtGui.QFont("Monospace")
        self.font.setStyleHint(QtGui.QFont.TypeWriter)

        # Line Edit
        self.lineEdit = QtWidgets.QLineEdit()
        self.lineEdit.installEventFilter(self)
        self.lineEdit.returnPressed.connect(self.execute_command)
        self.lineEdit.setFont(self.font)
        self.set_default_text()

        # Pin Button
        self.pin_icon = QtGui.QIcon(':/pinned.png')
        self.pin_button = QtWidgets.QToolButton()
        self.pin_button.setCheckable(True)
        self.pin_button.setChecked(UIState.pinned)
        self.pin_button.setFixedSize(height, height)
        self.pin_button.setIcon(self.pin_icon)
        self.pin_button.setIconSize(QtCore.QSize(height, height))
        self.pin_button.toggled.connect(UIState.toggle_pin)

        # Menu button
        self.menu_icon = QtGui.QIcon(":/menu_key.png")
        self.menu_button = QtWidgets.QToolButton()
        self.menu_button.setFixedSize(height, height)
        self.menu_button.setIcon(self.menu_icon)
        self.menu_button.setIconSize(QtCore.QSize(height, height))
        self.menu_button.clicked.connect(self.show_command_menu)

        # Layout
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setDirection(QtWidgets.QBoxLayout.LeftToRight)
        self.layout.setStretch(2, 1)
        self.layout.addWidget(self.pin_button)
        self.layout.addWidget(self.menu_button)
        self.layout.addWidget(self.lineEdit)
        self.setLayout(self.layout)

        # Window attributes
        self.setWindowFlags(
            QtCore.Qt.Popup
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.NoDropShadowWindowHint
        )
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setGeometry(rect)
        self.lineEdit.setFocus()

    def set_default_text(self):
        # If there is a 1 line expression, let's show it
        if len(get_selected_channels()) == 1:
            _, _, first = next(selected_channels())
            expr = first.get_gen_expression() or first.get_expression()
            if expr and '\n' not in expr:
                self.lineEdit.setText('=' + expr.split('=')[-1])

    def show_command_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.setToolTipsVisible(True)
        menu.setFixedWidth(self.rect().width())
        menu.setFont(self.font)
        item = '{ShortDashCommand:<3} {DashCommand}{Hint}'
        for cmd in sorted(_registry, key=lambda x: x['ShortDashCommand']):
            action = menu.addAction(item.format(**cmd))
            if cmd['Icon']:
                action.setIcon(QtGui.QIcon(cmd['Icon']))
            action.triggered.connect(self.command_menu_action(cmd))
            action.setToolTip(cmd['Description'])
        menu.popup(self.mapToGlobal(self.rect().bottomLeft()))

    def command_menu_action(self, cmd):
        def do_menu_action():
            line_text = self.lineEdit.text().rstrip()
            text = '{ShortDashCommand}()'.format(**cmd)
            self.lineEdit.setText(' '.join([line_text, text]))
            self.lineEdit.cursorBackward(False, 1)
            self.lineEdit.setFocus()
        return do_menu_action

    def eventFilter(self, widget, event):
        '''*PATCH* event filter was not accepting events on up and down
        causing pickwalking to occur after history changed.
        '''
        we_give_a_shit_about_this_event = (
            event.type() == QtCore.QEvent.KeyPress
            and widget is self.lineEdit
        )

        if we_give_a_shit_about_this_event:
            if event.key() == QtCore.Qt.Key_Up:
                self.lineEdit.setText(UIState.prev_history())
                event.accept()
                return True
            if event.key() == QtCore.Qt.Key_Down:
                self.lineEdit.setText(UIState.next_history())
                event.accept()
                return True

        return QtWidgets.QWidget.eventFilter(self, widget, event)

    def execute_command(self):
        text = self.lineEdit.text()
        cmd = DashPlus(text)
        try:
            cmd.execute()
            UIState.add_history(text)
            if not UIState.pinned:
                self.close()
            log('Success!')
        except Exception:
            log('Execution Failed!')
            raise


def channelBox_row_rect():
    '''Get the geometry of the currently hovered row in the channelBox.

    We know the Dash line is only requested when when right clicking over the
    channelBox, so this should be stable enough in practice.
    '''

    # Get the channel box TableView
    app = QtWidgets.QApplication.instance()
    pos = QtGui.QCursor().pos()
    widget = app.widgetAt(pos)
    table = widget.parent()

    # parse out some important info
    local_pos = table.mapFromGlobal(pos)
    row = table.rowAt(local_pos.y())
    cols = table.model().columnCount()

    # Return the rect for the current row
    top = 0
    for i in range(row):
        top += table.rowHeight(row)
    width = 0
    for i in range(cols):
        width += table.columnWidth(i)
    height = table.rowHeight(row)
    topLeft = table.mapToGlobal(QtCore.QPoint(0, top))

    return QtCore.QRect(topLeft.x(), topLeft.y(), width, height)


@contextmanager
def undo(name='unnamed', undo_on_fail=True):
    '''undo context manager'''

    exc_raised = False
    try:
        cmds.undoInfo(openChunk=True, chunkName=name)
        yield
    except Exception:
        exc_raised = True
        raise
    finally:
        cmds.undoInfo(closeChunk=True)
        if exc_raised:
            cmds.undo()


def get_dash_file():
    '''Get the path to Dash.json'''

    mash_path = os.environ.get('MASH_LOCATION', '')
    json_path = mash_path + 'scripts/Dash.json'
    return json_path


def read_commands():
    '''Read Dash.json from the MASH scripts folder'''

    json_path = get_dash_file()

    if not os.path.isfile(json_path):
        return []

    with open(json_path) as f:
        data = json.load(f)

    return data


def cmd_exists(cmd, short_cmd, commands):
    '''Check if a command exists in a list of DashCommands'''

    for c in commands:
        names = [c['DashCommand'], c['ShortDashCommand']]
        if cmd in names or short_cmd in names:
            return True
    return False


def add_command(
    MayaExec,
    MayaEval,
    DashCommand,
    ShortDashCommand,
    Description,
    Hint='()',
    Icon=None,
):
    '''Add a new Dash command.

    Arguments:
        MayaExec (str): Command to exec like "import random"
        MayaEval (str): Command to eval like "random.choice"
        DashCommand (str): Name of command
        ShortDashCommand (str): Short name of command
        Description (str): Description of command
    '''

    if cmd_exists(DashCommand, ShortDashCommand, _registry):
        log('Failed to add %s. Command already exists.', DashCommand)
        return

    data = {
        'MayaExec': MayaExec,
        'MayaEval': MayaEval,
        'DashCommand': DashCommand,
        'ShortDashCommand': ShortDashCommand,
        'Description': Description,
        'Hint': Hint,
        'Icon': Icon,
    }
    _registry.append(data)


def remove_command(name):
    '''Remove a registered command by name.'''

    for i, cmd in enumerate(_registry):
        if name in (cmd['DashCommand'], cmd['ShortDashCommand']):
            _registry.pop(i)
            _functions.pop(cmd['DashCommand'], None)
            _functions.pop(cmd['ShortDashCommand'], None)
            return


def get_function(name):
    '''Get a registered dash command by name.'''

    return _functions[name]


def find_command(name):
    for cmd in _registry:
        if name in (cmd['DashCommand'], cmd['ShortDashCommand']):
            return cmd


def command(name, short_name, hint='()', icon=None):
    '''Dash command decorator. Apply to functions to make them available
    from the Dash line.
    '''

    from functools import wraps

    def wrapper(fn):

        @wraps(fn)
        def caller(*args):
            return fn(*args)

        add_command(
            MayaExec='import dashplus',
            MayaEval='dashplus.get_function("%s")' % name,
            DashCommand=name,
            ShortDashCommand=short_name,
            Description=fn.__doc__,
            Hint=hint,
            Icon=icon,
        )
        _functions[name] = caller

        return caller
    return wrapper


def get_selected_channels():
    '''Returns a list of the selected channel box or spreadsheet channels.'''

    return DashCommand.getAllSelectedChannels()


def selected_channels():
    '''Parameterizes the current maya selection. Useful when creating stepped
    commands.

    Examples:
        Given the following selection...
        ['null1', 'null2', 'null3']

        Return...
        [(0, 0.0, 'null1'), (0, 0.5, 'null2'), (0, 1.0, 'null3')]
    '''

    objects = cmds.ls(selection=True, long=True)
    channels = get_selected_channels()
    try:
        step = 1.0 / (len(objects) - 1.0)
    except ZeroDivisionError:
        step = 1.0
    for i, node_name in enumerate(objects):
        t = i * step
        for ci, channel in enumerate(channels):
            yield i, t, Channel(node_name, channel, ci)


def grouped_channels():
    '''Parameterizes the current maya selection. Useful when creating stepped
    commands.

    Examples:
        Given the following selection...
        ['null1', 'null2', 'null3']

        Return...
        [(0, 0.0, 'null1'), (0, 0.5, 'null2'), (0, 1.0, 'null3')]
    '''

    objects = cmds.ls(selection=True, long=True)
    channel_names = get_selected_channels()
    channels = []
    for node_name in objects:
        for channel in channel_names:
            channels.append(Channel(node_name, channel))
    return [
        (n, list(g))
        for n, g in groupby(channels, key=lambda c: c.channel)
    ]


class Channel(object):
    '''A convenient wrapper around an attribute making it easier to manipulate
    animation curves and values.

        chan = Channel('pSphere1.tx')

    Get and Set the current value:
        tx = chan.get()
        chan.set(tx + 2)

    Raise all keyframe values to the power of 2:
        from math import pow
        for i, time, value in chan.keys():
            chan.set_value(i, pow(value, 2))

    Scale all keyframe times and values:
        chan.scale_times(2, pivot=1)
        chan.scale_values(2, pivot=3)

    Use indexing to get times and values:
        first_frame, first_value = chan.time(0), chan.value(0)
        last_frame, last_value = chan.time(-1), chan.value(-1)
    '''

    def __init__(self, node_name, channel, i=0):
        self.node_name = node_name
        self.channel = channel
        self.path = node_name + '.' + channel
        self.i = i

    def get(self):
        '''Get the current value'''
        return cmds.getAttr(self.path)

    def set(self, value):
        '''Set the current value'''
        cmds.setAttr(self.path, value)

    @property
    def num_keys(self):
        return cmds.keyframe(self.path, q=True, keyframeCount=True)

    @property
    def has_keys(self):
        return bool(self.num_keys)

    def to_index(self, index):
        if isinstance(index, tuple):
            return index
        if isinstance(index, int):
            if index < 0:
                i = self.num_keys + index
                return (i, i)
            else:
                return (index, index)
        return (0, self.num_keys)

    def time(self, index=None):
        index = self.to_index(index)
        return cmds.keyframe(
            self.path,
            q=True,
            index=index,
            timeChange=True
        )[0]

    def value(self, index=None):
        index = self.to_index(index)
        return cmds.keyframe(
            self.path,
            q=True,
            index=index,
            valueChange=True
        )[0]

    def times(self, index=None):
        index = self.to_index(index)
        return cmds.keyframe(
            self.path,
            q=True,
            index=index,
            timeChange=True
        )

    def values(self, index=None):
        index = self.to_index(index)
        return cmds.keyframe(
            self.path,
            q=True,
            index=index,
            valueChange=True
        )

    def keys(self):
        index = (0, self.num_keys)
        iterator = enumerate(zip(self.times(index), self.values(index)))
        for i, (time, value) in iterator:
            yield i, time, value

    def set_key(self, time=None, value=None):
        time = time or cmds.currentTime(q=True)
        value = value or self.get()
        cmds.setKeyframe(self.path, time=time, value=value)

    def set_value(self, index, value, time=None):
        index = self.to_index(index)
        time = time or cmds.currentTime(q=True)
        if index[0] < self.num_keys:
            cmds.keyframe(self.path, index=index, valueChange=value)
        else:
            cmds.setKeyframe(self.path, insert=True, value=value, time=time)

    def set_time(self, index, time):
        index = self.to_index(index)
        if index[0] < self.num_keys:
            cmds.keyframe(self.path, index=index, timeChange=time)
        else:
            cmds.setKeyframe(self.path, insert=True, time=time)

    def scale_values(self, scale, pivot):
        cmds.scaleKey(self.path, valueScale=scale, valuePivot=pivot)

    def scale_times(self, scale, pivot):
        cmds.scaleKey(self.path, timeScale=scale, timePivot=pivot)

    def move_values(self, offset, index=None):
        index = self.to_index(index)
        cmds.keyframe(
            self.path,
            relative=True,
            index=index,
            valueChange=offset,
        )

    def move_times(self, offset, index=None):
        index = self.to_index(index)
        cmds.keyframe(
            self.path,
            relative=True,
            index=index,
            timeChange=offset,
        )

    def format_expression(self, expr):
        if not expr.startswith(self.channel):
            expr = self.channel + ' = ' + expr
        return expr

    def inputs(self):
        return cmds.listConnections(self.path, d=False, s=True, plugs=True)

    def outputs(self):
        return cmds.listConnections(self.path, d=True, s=False, plugs=True)

    def get_expression_node(self):
        node = cmds.listConnections(
            self.path,
            d=False,
            s=True,
            scn=True,
            type='expression'
        )
        if node:
            return node[0]

    def get_gen_expression(self, expr_node=None):
        expr_node = expr_node or self.get_expression_node()
        if expr_node:
            expr_attr = expr_node + '.gen_expr'
            if cmds.objExists(expr_attr):
                return cmds.getAttr(expr_attr)

    def set_gen_expression(self, gen_expr, expr_node=None):
        expr_node = expr_node or self.get_expression_node()
        if expr_node:
            expr_attr = expr_node + '.gen_expr'
            if not cmds.objExists(expr_attr):
                cmds.addAttr(expr_node, ln='gen_expr', dt='string')
            cmds.setAttr(expr_attr, gen_expr, type='string')

    def get_expression(self):
        expr_node = self.get_expression_node()
        if expr_node:
            return cmds.expression(expr_node, query=True, string=True)

    def set_expression(self, expr, gen_expr=''):
        args = []
        kwargs = dict(
            object=self.node_name,
            string=self.format_expression(expr),
            name=self.path.replace('.', '_').split('|')[-1]
        )
        expr_node = self.get_expression_node()
        if expr_node:
            args.append(expr_node)
            kwargs['edit'] = True
        else:
            inputs = self.inputs()
            if inputs:
                cmds.disconnectAttr(inputs[0], self.path)

        expr_node = cmds.expression(*args, **kwargs)
        self.set_gen_expression(gen_expr, expr_node)

    def copy(self, index=None):
        index = self.to_index(index)
        cmds.copyKey(self.path, index=index)
        return index

    def paste(self, channels, index=None, option='insert'):
        index = self.copy(self)
        cmds.pasteKey(
            [c.path for c in channels],
            index=index,
            option=option
        )


class math:

    @staticmethod
    def fit01(value, new_min, new_max):
        '''Fit a value between 0 and 1 to a new range'''

        return new_min + (value * (new_max - new_min))

    @staticmethod
    def fit(value, new_min, new_max, old_min, old_max):
        '''Fit a value from old_min to old_max into new_min to new_max'''

        try:
            new_range = new_max - new_min
            old_range = old_max - old_min
            return new_min + ((value - old_min) / old_range * new_range)
        except ZeroDivisionError:
            return value


@command(
    'step_scale',
    'ss',
    hint='(min, max, [pivot])',
    icon=':/stepTangent.png'
)
def step_scale(*args):
    '''Stagger the values of multiple objects from a value pivot.'''

    if len(args) not in (2, 3):
        error('Missing required arguments: ss(min, max, [pivot])')

    if len(args) == 2:
        min_scale, max_scale = args
        pivot = 0.0
    else:
        min_scale, max_scale, pivot = args

    for i, t, channel in selected_channels():
        scale = math.fit01(t, min_scale, max_scale)
        if channel.has_keys:
            channel.scale_values(scale, pivot)
        else:
            channel.set((channel.get() - pivot) * scale)


@command('step_time', 'sst', hint='(min, max)', icon=':/setKeyOnTranslate.png')
def step_time(*args):
    '''Stagger the times of multiple objects by an overall amount.'''

    if len(args) != 2:
        error('Missing required arguments: sst(min, max)')

    min_scale, max_scale = args

    for i, t, channel in selected_channels():
        offset = math.fit01(t, min_scale, max_scale)
        if channel.has_keys:
            channel.move_times(offset)


@command('scale', 's', hint='(amount, [pivot])', icon=':/scale_M.png')
def scale(*args):
    '''Scale values from a value pivot.'''

    nargs = len(args)
    if nargs not in (1, 2):
        error('Missing arguments: ss(amount, [pivot])')

    if nargs == 1:
        amount, pivot = args[0], 0
    else:
        amount, pivot = args

    for i, t, channel in selected_channels():
        if channel.has_keys:
            channel.scale_values(amount, pivot)
        else:
            channel.set(channel.get() * amount)


@command(
    'scale_time',
    'st',
    hint='(amount, [pivot])',
    icon=':/setKeyOnScale.png'
)
def scale_time(*args):
    '''Scale times from a value pivot.'''

    nargs = len(args)
    if nargs not in (1, 2):
        error('Missing arguments: st(amount, [pivot])')

    if nargs == 1:
        amount, pivot = args[0], 0
    else:
        amount, pivot = args

    for i, t, channel in selected_channels():
        if channel.has_keys:
            channel.scale_times(amount, pivot)


@command('key_value', 'kv', hint='(index, value)', icon=':/setKeyframe.png')
def key_value(*args):
    '''Set the value of the keys at an index.

    Example:
        sv(1, 0) - Set the first key to 0
        sv(-1, 0) - Set the last key to 0
    '''

    if len(args) != 2:
        error('Missing required arguments: kv(index, value)')

    index, value = args

    for i, t, channel in selected_channels():
        channel.set_value(index, value)


@command('key_time', 'kt', hint='(index, time)', icon=':/setKeyframe.png')
def key_time(*args):
    '''Set the time of the keys at an index.

    Example:
        sv(1, 1) - Move the first key to frame 1
        sv(-1, 10) - Move the last key to frame 10
    '''

    if len(args) != 2:
        error('Missing required arguments: kv(index, value)')

    index, time = args

    for i, t, channel in selected_channels():
        channel.set_time(index, time)


@command('gen_keys', 'gk', hint='(n, step, [start])', icon=':/keyIntoClip.png')
def gen_keys(*args):
    '''Generate n keys at a regular interval.

    Example:
        gk(4, 12) - 4 keys with a spacing of 12 frames
        gk(4, 12, 10) - 4 keys with a spacing of 12 frames starting at frame 10
    '''

    nargs = len(args)
    if nargs not in (2, 3):
        error('Incorrect arguments: gk(n, step, [start])')

    if nargs == 2:
        n, step, start = args[0], args[1], cmds.currentTime(q=True)
    else:
        n, step, start = args

    for x in range(n):
        frame = start + x * step
        for i, t, channel in selected_channels():
            channel.set_key(frame)


@command(
    'wave',
    'w',
    hint='(n, step, amplitude, [start])',
    icon=':/sineCurveProfile.png'
)
def wave(*args):
    '''Generate a sine wave of keys.

    Example:
        w(4, 12, 2) - 4 keys with a spacing of 12 frames and an amplitude of 2
    '''

    nargs = len(args)
    if nargs not in (3, 4):
        error('Incorrect arguments: gk(n, step, [start])')

    if nargs == 3:
        n, step, amp = args
        start = cmds.currentTime(q=True)
    else:
        n, step, amp, start = args

    for i, x in enumerate(range(n)):
        frame = start + x * step
        value = math.fit01(i % 2, -amp, amp)
        for i, t, channel in selected_channels():
            channel.set_key(frame, value)


@command('fit01', 'f01', hint='(min, max)', icon=':/setRange.svg')
def fit01(*args):
    '''Remap a value from 0 to 1 to a new range.'''

    if len(args) != 2:
        error('Missing arguments: fit01(min, max)')

    mn, mx = args

    for i, t, channel in selected_channels():
        if channel.has_keys:
            for j, time, value in channel.keys():
                new_value = math.fit01(value, mn, mx)
                channel.set_value(j, new_value)
        else:
            new_value = math.fit01(channel.get(), mn, mx)
            channel.set(new_value)


@command(
    'fit',
    'f',
    hint='(min, max, old_min, old_max)',
    icon=':/setRange.svg'
)
def fit(*args):
    '''Remap values from an old range to a new range.'''

    if len(args) != 4:
        error('Missing arguments: fit(min, max, old_min, old_max)')

    nmn, nmx, omn, omx = args

    for i, t, channel in selected_channels():
        if channel.has_keys:
            for j, time, value in channel.keys():
                new_value = math.fit(value, nmn, nmx, omn, omx)
                channel.set_value(j, new_value)
        else:
            new_value = math.fit(channel.get(), nmn, nmx, omn, omx)
            channel.set(new_value)


@command('normalize', 'n', icon=':/normCurves.png')
def normalize(*args):
    '''Normalize all values.

    Scales curves to fit between the min and max values of all selected curves.
    '''

    if len(args):
        error('Normalize accepts no arguments.')

    mx = -float('inf')
    mn = float('inf')
    channels = []

    # Find extents
    for i, t, channel in selected_channels():
        channels.append((i, t, channel))
        if channel.has_keys:
            channel.mx = -float('inf')
            channel.mn = float('inf')
            for j, time, value in channel.keys():
                mn = min(value, mn)
                mx = max(value, mx)
                channel.mn = min(value, channel.mn)
                channel.mx = max(value, channel.mx)

    # Apply normalization
    for i, t, channel in channels:
        if channel.has_keys:
            for j, time, value in channel.keys():
                new_value = math.fit(value, mn, mx, channel.mn, channel.mx)
                channel.set_value(j, new_value)


@command('normalize_time', 'nt', icon=':/normCurves.png')
def normalize_time(*args):
    '''Normalize keyframe times.'''

    if len(args):
        error('No arguments expected.')

    mx = -float('inf')
    mn = float('inf')
    channels = []

    # Find extents
    for i, t, channel in selected_channels():
        channels.append((i, t, channel))
        if channel.has_keys:
            channel.mx = -float('inf')
            channel.mn = float('inf')
            for j, time, value in channel.keys():
                mn = min(time, mn)
                mx = max(time, mx)
                channel.mn = min(time, channel.mn)
                channel.mx = max(time, channel.mx)

    # Apply normalization
    for i, t, channel in channels:
        if channel.has_keys:
            for j, time, value in channel.keys():
                new_time = math.fit(time, mn, mx, channel.mn, channel.mx)
                channel.set_time(j, new_time)


@command('align_start', 'as', icon=':/arrowLeft.png')
def align_start(*args):
    '''Align first frame of all animCurves.'''

    if len(args):
        error('No arguments expected.')

    mn = float('inf')
    channels = []

    # Find extents
    for i, t, channel in selected_channels():
        channels.append((i, t, channel))
        if channel.has_keys:
            channel.mn = channel.time(0)
            mn = min(channel.mn, mn)

    # Move keyframes
    for i, t, channel in channels:
        if channel.has_keys:
            channel.move_times(mn - channel.mn)


@command('align_end', 'ae', icon=':/arrowRight.png')
def align_end(*args):
    '''Align last frame of all animCurves'''

    if len(args):
        error('No arguments expected.')

    mx = -float('inf')
    channels = []

    # Find extents
    for i, t, channel in selected_channels():
        channels.append((i, t, channel))
        if channel.has_keys:
            channel.mx = channel.time(-1)
            mx = max(channel.mx, mx)

    # Move keyframes
    for i, t, channel in channels:
        if channel.has_keys:
            channel.move_times(mx - channel.mx)


@command('align_center', 'ac', icon=':/arrowDown.png')
def align_center(*args):
    '''Align last frame of all animCurves'''

    if len(args):
        error('No arguments expected.')

    mx = -float('inf')
    mn = float('inf')
    channels = []

    # Find extents
    for i, t, channel in selected_channels():
        channels.append((i, t, channel))
        if channel.has_keys:
            channel.mx = channel.time(-1)
            channel.mn = channel.time(0)
            mn = min(channel.mn, mn)
            mx = max(channel.mx, mx)

    mid = mn + ((mx - mn) * 0.5)
    # Apply normalization
    for i, t, channel in channels:
        if channel.has_keys:
            chan_mid = channel.mn + ((channel.mx - channel.mn) * 0.5)
            channel.move_times(mid - chan_mid)


@command('match', 'm', icon=':/alignCurve.svg')
def match(*args):
    '''Copy and pasted keys from the first object in your selection.'''

    if len(args):
        error('No arguments expected.')

    source_obj = cmds.ls(selection=True, long=True)[0]
    channel_names = get_selected_channels()
    sources = {chan: Channel(source_obj, chan) for chan in channel_names}

    destinations = grouped_channels()
    for channel_name, group in destinations:
        source = sources[channel_name]
        if source.has_keys:
            source.paste(group, option='replaceCompletely')
        else:
            value = source.get()
            for dest in group:
                if dest.has_keys:
                    log('Skipped %s: it has keys.' % dest.path)
                else:
                    dest.set(value)


@command('match_times', 'mt', icon=':/alignCurve.svg')
def match_times(*args):
    '''Snap key times by index to the first object in your selection.'''

    if len(args):
        error('No arguments expected.')

    source_obj = cmds.ls(selection=True, long=True)[0]
    channel_names = get_selected_channels()
    sources = {chan: Channel(source_obj, chan) for chan in channel_names}

    destinations = grouped_channels()
    for channel_name, group in destinations:
        source = sources[channel_name]
        if source.has_keys:
            for i, time, value in source.keys():
                for dest in group:
                    if dest.has_keys:
                        dest.set_time(i, time)


@command('linear', 'l', hint='(max)|(min, max)', icon=':/distanceDim.png')
def linear_dist(*args):
    '''Distribute values from min value to max value.

    Examples:
        l(10): distribute values from 0 to 10
        l(10, 20): distribute values from 10 to 20
    '''

    if len(args) not in (1, 2):
        error('Incorrect number of arguments: 1 or 2 allowed')

    if len(args) == 1:
        min_value, max_value = 0, args[0]
    else:
        min_value, max_value = args

    for i, t, channel in selected_channels():
        channel.set(math.fit01(t, min_value, max_value))


def test_dash_plus():
    '''Make sure we are choosing the right execution mode'''

    cmd = DashPlus('= $v * 2')
    assert cmd.get_executor() == cmd.execute_expr

    for text in ['+=1', '-=1', '*=1', '\\=1', '/=1']:
        cmd = DashPlus(text)
        assert cmd.get_executor() == cmd.execute_cbox_expr

    for text in ['+=:1', '-=:1', '*=:1', '\\=:1', '/=:1']:
        cmd = DashPlus(text)
        assert cmd.get_executor() == cmd.execute_cbox_expr
        assert cmd.is_cbox_time_expr

    cmd = DashPlus(': c.set(100)')
    assert cmd.get_executor() == cmd.execute_python

    cmd = DashPlus('m() x(3, 5, funk("who")) asdf(2)')
    assert cmd.get_executor() == cmd.execute_dash
